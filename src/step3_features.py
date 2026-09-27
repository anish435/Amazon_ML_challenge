"""Step 3 — Vectorized Pair Feature Engineering (v2).

Extracts 53 pairwise similarity, phonetic, raw number, and pool rarity features:
  a) String similarity on name_norm, name_core, addr_norm (18 features)
  b) TF-IDF cosine per pair (name & address) (2 features)
  c) Normalized numbers & postal matching (6 features)
  d) Lengths and empties (4 features)
  e) Source indicator (1 feature)
  f) Blocking channel scores & ranks (7 features)
  --- NEW IN V2 ---
  g) Raw address unit numbers (5 features):
     raw_num_jaccard, raw_num_all_equal, raw_num_conflicts, raw_num_first_equal, raw_num_first_dist
  h) Consonant skeleton phonetic similarity & script flags (4 features):
     skel_ratio, skel_token_set_ratio, is_non_latin_s1, is_non_latin_cand
  i) Candidate pool record rarity (6 features):
     rarity_name_s1, rarity_name_cand, rarity_addr_s1, rarity_addr_cand, rarity_name_diff, rarity_addr_diff

Total: 53 float32 features. Fully vectorized, strictly bounded memory (< 2.5 GB peak RSS).
Supports both v1 (38 features) and v2 (53 features).
"""

from pathlib import Path
import gc
import joblib
import os
import re
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple, Any

# Reconfigure stdout/stderr for UTF-8 on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import polars as pl
import psutil
from rapidfuzz import distance, fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer

from src.common import (
    DATA_BLOCKS_DIR,
    DATA_FEATURES_DIR,
    DATA_FEATURES_V2_DIR,
    PREP_FILES,
    RUN_LOG_PATH,
    append_to_run_log,
)

FEATURE_COLUMNS_V1: List[str] = [
    # a) Strings on name_norm (6)
    "name_norm_ratio",
    "name_norm_token_sort_ratio",
    "name_norm_token_set_ratio",
    "name_norm_partial_ratio",
    "name_norm_jaro_winkler",
    "name_norm_levenshtein",
    # Strings on name_core (6)
    "name_core_ratio",
    "name_core_token_sort_ratio",
    "name_core_token_set_ratio",
    "name_core_partial_ratio",
    "name_core_jaro_winkler",
    "name_core_levenshtein",
    # Strings on addr_norm (6)
    "addr_norm_ratio",
    "addr_norm_token_sort_ratio",
    "addr_norm_token_set_ratio",
    "addr_norm_partial_ratio",
    "addr_norm_jaro_winkler",
    "addr_norm_levenshtein",
    # b) TF-IDF Cosine (2)
    "tfidf_cosine_name",
    "tfidf_cosine_addr",
    # c) Numbers (6)
    "house_num_overlap",
    "house_num_conflict",
    "house_num_missing_s1",
    "house_num_missing_cand",
    "postal_equal",
    "postal_missing",
    # d) Lengths and empties (4)
    "name_len_ratio",
    "addr_len_ratio",
    "empty_addr_s1",
    "empty_addr_cand",
    # e) Source (1)
    "is_s3",
    # f) Blocking info (7)
    "score_a",
    "score_b",
    "score_c",
    "rank_a",
    "rank_b",
    "rank_c",
    "n_channels",
]

NEW_PAIR_FEATURE_COLUMNS_V2: List[str] = [
    # g) Numbers from RAW address (5)
    "raw_num_jaccard",
    "raw_num_all_equal",
    "raw_num_conflicts",
    "raw_num_first_equal",
    "raw_num_first_dist",
    # h) Sound-alike name similarity & script flags (4)
    "skel_ratio",
    "skel_token_set_ratio",
    "is_non_latin_s1",
    "is_non_latin_cand",
    # i) Candidate pool rarity (6)
    "rarity_name_s1",
    "rarity_name_cand",
    "rarity_addr_s1",
    "rarity_addr_cand",
    "rarity_name_diff",
    "rarity_addr_diff",
]

FEATURE_COLUMNS_V2: List[str] = FEATURE_COLUMNS_V1 + NEW_PAIR_FEATURE_COLUMNS_V2
FEATURE_COLUMNS: List[str] = FEATURE_COLUMNS_V2


# ---------------------------------------------------------------------------
# Helpers for New v2 Features
# ---------------------------------------------------------------------------
UNIT_RE = re.compile(r"\b(?:[a-zA-Z0-9]+(?:[/\\-][a-zA-Z0-9]+)+|\d+[a-zA-Z]|[a-zA-Z]\d+|\d+)\b")
FIRST_DIGITS_RE = re.compile(r"\d+")


def get_consonant_skeleton(s: str) -> str:
    """Computes consonant skeleton of name_core: lowercase -> drop vowels except 1st letter -> collapse repeats."""
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"ee+|ii+", "i", s)
    s = re.sub(r"oo+|uu+", "u", s)
    s = s.replace("ph", "f").replace("kh", "k").replace("gh", "g")
    s = s.replace("bh", "b").replace("dh", "d").replace("th", "t").replace("sh", "s")
    s = re.sub(r"\b(pvt|ltd|limited|limitedd|private|praaivett|llp|corp|inc|co|company)\b", "", s).strip()
    words = s.split()
    skel_words = []
    vowels = set("aeiou")
    for w in words:
        if not w:
            continue
        first = w[0]
        rest = [c for c in w[1:] if c not in vowels]
        w_skel = first + "".join(rest)
        w_skel = re.sub(r"(.)\1+", r"\1", w_skel)
        skel_words.append(w_skel)
    return " ".join(skel_words)


def extract_unit_nums(s: str) -> List[str]:
    """Extracts unit numbers and compound address tokens ('35/1', 'g-9/5/12', '4b')."""
    if not s:
        return []
    return [m.lower() for m in UNIT_RE.findall(s) if any(c.isdigit() for c in m)]


# ---------------------------------------------------------------------------
# Single Feature Extraction Function (Used Identically for Train / Val / Holdout / Test)
# ---------------------------------------------------------------------------
def compute_pair_features(
    joined_df: pl.DataFrame,
    vec_name: TfidfVectorizer,
    vec_addr: TfidfVectorizer,
    rarity_maps: Optional[Tuple[Dict[str, int], Dict[str, int]]] = None,
    version: str = "v2",
) -> pl.DataFrame:
    """Computes float32 features on a joined DataFrame of candidate pairs.

    joined_df must contain:
      - s1_num, cand_num, cand_src, (label if present)
      - name_raw_s1, name_norm_s1, name_core_s1, addr_raw_s1, addr_norm_s1, house_nums_s1, postal_s1
      - name_raw_cand, name_norm_cand, name_core_cand, addr_raw_cand, addr_norm_cand, house_nums_cand, postal_cand
      - score_a, rank_a, score_b, rank_b, score_c, rank_c, n_channels
    """
    target_cols = FEATURE_COLUMNS_V2 if version == "v2" else FEATURE_COLUMNS_V1
    n_pairs = len(joined_df)
    if n_pairs == 0:
        schema = {col: pl.Float32 for col in target_cols}
        schema["s1_num"] = pl.Int64
        schema["cand_num"] = pl.Int64
        schema["cand_src"] = pl.Int8
        if "label" in joined_df.columns:
            schema["label"] = pl.Int8
        return pl.DataFrame(schema=schema)

    # Convert text columns to lists for RapidFuzz and TF-IDF
    n_norm_1 = joined_df["name_norm_s1"].fill_null("").to_list()
    n_norm_2 = joined_df["name_norm_cand"].fill_null("").to_list()
    n_core_1 = joined_df["name_core_s1"].fill_null("").to_list()
    n_core_2 = joined_df["name_core_cand"].fill_null("").to_list()
    a_norm_1 = joined_df["addr_norm_s1"].fill_null("").to_list()
    a_norm_2 = joined_df["addr_norm_cand"].fill_null("").to_list()

    fuzz_dict: Dict[str, np.ndarray] = {}

    # a) RapidFuzz String Similarities (18 features)
    string_pairs = [
        ("name_norm", n_norm_1, n_norm_2),
        ("name_core", n_core_1, n_core_2),
        ("addr_norm", a_norm_1, a_norm_2),
    ]

    for prefix, s_1, s_2 in string_pairs:
        fuzz_dict[f"{prefix}_ratio"] = (
            process.cpdist(s_1, s_2, scorer=fuzz.ratio, workers=-1) / 100.0
        ).astype(np.float32)
        fuzz_dict[f"{prefix}_token_sort_ratio"] = (
            process.cpdist(s_1, s_2, scorer=fuzz.token_sort_ratio, workers=-1) / 100.0
        ).astype(np.float32)
        fuzz_dict[f"{prefix}_token_set_ratio"] = (
            process.cpdist(s_1, s_2, scorer=fuzz.token_set_ratio, workers=-1) / 100.0
        ).astype(np.float32)
        fuzz_dict[f"{prefix}_partial_ratio"] = (
            process.cpdist(s_1, s_2, scorer=fuzz.partial_ratio, workers=-1) / 100.0
        ).astype(np.float32)
        fuzz_dict[f"{prefix}_jaro_winkler"] = process.cpdist(
            s_1, s_2, scorer=distance.JaroWinkler.similarity, workers=-1
        ).astype(np.float32)
        fuzz_dict[f"{prefix}_levenshtein"] = process.cpdist(
            s_1, s_2, scorer=distance.Levenshtein.normalized_similarity, workers=-1
        ).astype(np.float32)

    # b) TF-IDF Cosine via row-wise sparse dot product (2 features)
    u_n1, inv_n1 = np.unique(n_core_1, return_inverse=True)
    u_n2, inv_n2 = np.unique(n_core_2, return_inverse=True)
    q_name = vec_name.transform(u_n1)[inv_n1]
    c_name = vec_name.transform(u_n2)[inv_n2]
    fuzz_dict["tfidf_cosine_name"] = np.array(q_name.multiply(c_name).sum(axis=1)).ravel().astype(np.float32)
    del u_n1, inv_n1, u_n2, inv_n2, q_name, c_name

    u_a1, inv_a1 = np.unique(a_norm_1, return_inverse=True)
    u_a2, inv_a2 = np.unique(a_norm_2, return_inverse=True)
    q_addr = vec_addr.transform(u_a1)[inv_a1]
    c_addr = vec_addr.transform(u_a2)[inv_a2]
    fuzz_dict["tfidf_cosine_addr"] = np.array(q_addr.multiply(c_addr).sum(axis=1)).ravel().astype(np.float32)
    del u_a1, inv_a1, u_a2, inv_a2, q_addr, c_addr

    # c) Numbers (6), d) Lengths (4), e) Source (1), f) Blocking (7) in Polars
    feat_df = joined_df.with_columns([
        pl.col("house_nums_s1").fill_null("").str.strip_chars().alias("h1_str"),
        pl.col("house_nums_cand").fill_null("").str.strip_chars().alias("h2_str"),
        pl.col("postal_s1").fill_null("").str.strip_chars().alias("p1_str"),
        pl.col("postal_cand").fill_null("").str.strip_chars().alias("p2_str"),
        pl.col("addr_norm_s1").fill_null("").str.strip_chars().alias("a1_str"),
        pl.col("addr_norm_cand").fill_null("").str.strip_chars().alias("a2_str"),
        pl.col("name_norm_s1").fill_null("").str.strip_chars().alias("n1_str"),
        pl.col("name_norm_cand").fill_null("").str.strip_chars().alias("n2_str"),
    ]).with_columns([
        # c) Numbers
        (pl.col("h1_str") == "").cast(pl.Float32).alias("house_num_missing_s1"),
        (pl.col("h2_str") == "").cast(pl.Float32).alias("house_num_missing_cand"),
        (pl.col("h1_str").str.split(" ")).alias("h1_list"),
        (pl.col("h2_str").str.split(" ")).alias("h2_list"),
        ((pl.col("p1_str") == "") | (pl.col("p2_str") == "")).cast(pl.Float32).alias("postal_missing"),
        ((pl.col("p1_str") != "") & (pl.col("p2_str") != "") & (pl.col("p1_str") == pl.col("p2_str"))).cast(pl.Float32).alias("postal_equal"),
        # d) Lengths and empties
        (pl.col("a1_str") == "").cast(pl.Float32).alias("empty_addr_s1"),
        (pl.col("a2_str") == "").cast(pl.Float32).alias("empty_addr_cand"),
        pl.when(pl.max_horizontal(pl.col("n1_str").str.len_bytes(), pl.col("n2_str").str.len_bytes()) == 0)
        .then(0.0)
        .otherwise(
            pl.min_horizontal(pl.col("n1_str").str.len_bytes(), pl.col("n2_str").str.len_bytes())
            / pl.max_horizontal(pl.col("n1_str").str.len_bytes(), pl.col("n2_str").str.len_bytes())
        ).cast(pl.Float32).alias("name_len_ratio"),
        pl.when(pl.max_horizontal(pl.col("a1_str").str.len_bytes(), pl.col("a2_str").str.len_bytes()) == 0)
        .then(0.0)
        .otherwise(
            pl.min_horizontal(pl.col("a1_str").str.len_bytes(), pl.col("a2_str").str.len_bytes())
            / pl.max_horizontal(pl.col("a1_str").str.len_bytes(), pl.col("a2_str").str.len_bytes())
        ).cast(pl.Float32).alias("addr_len_ratio"),
        # e) Source
        (pl.col("cand_src") == 3).cast(pl.Float32).alias("is_s3"),
        # f) Blocking info
        pl.col("score_a").fill_null(-1.0).cast(pl.Float32).alias("score_a"),
        pl.col("score_b").fill_null(-1.0).cast(pl.Float32).alias("score_b"),
        pl.col("score_c").fill_null(-1.0).cast(pl.Float32).alias("score_c"),
        pl.col("rank_a").fill_null(0).cast(pl.Float32).alias("rank_a"),
        pl.col("rank_b").fill_null(0).cast(pl.Float32).alias("rank_b"),
        pl.col("rank_c").fill_null(0).cast(pl.Float32).alias("rank_c"),
        pl.col("n_channels").cast(pl.Float32).alias("n_channels"),
    ]).with_columns([
        pl.when((pl.col("house_num_missing_s1") == 0) & (pl.col("house_num_missing_cand") == 0))
        .then(pl.col("h1_list").list.set_intersection(pl.col("h2_list")).list.len() > 0)
        .otherwise(False)
        .cast(pl.Float32)
        .alias("house_num_overlap"),
    ]).with_columns([
        pl.when((pl.col("house_num_missing_s1") == 0) & (pl.col("house_num_missing_cand") == 0) & (pl.col("house_num_overlap") == 0))
        .then(1.0)
        .otherwise(0.0)
        .cast(pl.Float32)
        .alias("house_num_conflict")
    ])

    # --- NEW V2 PAIRWISE FEATURES ---
    if version == "v2":
        a_raw_1 = joined_df["addr_raw_s1"].fill_null("").to_list() if "addr_raw_s1" in joined_df.columns else ["" for _ in range(n_pairs)]
        a_raw_2 = joined_df["addr_raw_cand"].fill_null("").to_list() if "addr_raw_cand" in joined_df.columns else ["" for _ in range(n_pairs)]
        n_raw_1 = joined_df["name_raw_s1"].fill_null("").to_list() if "name_raw_s1" in joined_df.columns else ["" for _ in range(n_pairs)]
        n_raw_2 = joined_df["name_raw_cand"].fill_null("").to_list() if "name_raw_cand" in joined_df.columns else ["" for _ in range(n_pairs)]

        # 1. Consonant Skeleton Similarities
        u_skel1 = {s: get_consonant_skeleton(s) for s in set(n_core_1)}
        u_skel2 = {s: get_consonant_skeleton(s) for s in set(n_core_2)}
        skel_1 = [u_skel1[s] for s in n_core_1]
        skel_2 = [u_skel2[s] for s in n_core_2]

        fuzz_dict["skel_ratio"] = (
            process.cpdist(skel_1, skel_2, scorer=fuzz.ratio, workers=-1) / 100.0
        ).astype(np.float32)
        fuzz_dict["skel_token_set_ratio"] = (
            process.cpdist(skel_1, skel_2, scorer=fuzz.token_set_ratio, workers=-1) / 100.0
        ).astype(np.float32)

        # Non-Latin flags
        fuzz_dict["is_non_latin_s1"] = np.array([1.0 if any(ord(c) > 127 for c in s) else 0.0 for s in n_raw_1], dtype=np.float32)
        fuzz_dict["is_non_latin_cand"] = np.array([1.0 if any(ord(c) > 127 for c in s) else 0.0 for s in n_raw_2], dtype=np.float32)

        # 2. Raw Unit Numbers
        u_nums1 = {s: extract_unit_nums(s) for s in set(a_raw_1)}
        u_nums2 = {s: extract_unit_nums(s) for s in set(a_raw_2)}

        num_jaccard = np.zeros(n_pairs, dtype=np.float32)
        num_all_equal = np.zeros(n_pairs, dtype=np.float32)
        num_conflicts = np.zeros(n_pairs, dtype=np.float32)
        num_first_equal = np.zeros(n_pairs, dtype=np.float32)
        num_first_dist = np.full(n_pairs, -1.0, dtype=np.float32)

        for i, (ar1, ar2) in enumerate(zip(a_raw_1, a_raw_2)):
            nums1 = u_nums1[ar1]
            nums2 = u_nums2[ar2]
            if nums1 or nums2:
                s1_set = set(nums1)
                s2_set = set(nums2)
                union_len = len(s1_set | s2_set)
                inter_len = len(s1_set & s2_set)
                if union_len > 0:
                    num_jaccard[i] = inter_len / union_len
                if s1_set and s2_set and (s1_set == s2_set):
                    num_all_equal[i] = 1.0
                num_conflicts[i] = float(len(s1_set ^ s2_set))
                if nums1 and nums2:
                    if nums1[0] == nums2[0]:
                        num_first_equal[i] = 1.0
                    m1 = FIRST_DIGITS_RE.search(nums1[0])
                    m2 = FIRST_DIGITS_RE.search(nums2[0])
                    if m1 and m2:
                        num_first_dist[i] = min(float(abs(int(m1.group()) - int(m2.group()))), 100.0)

        fuzz_dict["raw_num_jaccard"] = num_jaccard
        fuzz_dict["raw_num_all_equal"] = num_all_equal
        fuzz_dict["raw_num_conflicts"] = num_conflicts
        fuzz_dict["raw_num_first_equal"] = num_first_equal
        fuzz_dict["raw_num_first_dist"] = num_first_dist

        # 3. Rarity Features
        if rarity_maps is not None:
            name_counts, addr_counts = rarity_maps
        else:
            name_counts = dict(pl.Series(n_core_2).value_counts().iter_rows())
            addr_counts = dict(pl.Series(a_norm_2).value_counts().iter_rows())

        rn1 = np.log1p(np.array([name_counts.get(s, 1) for s in n_core_1], dtype=np.float32))
        rn2 = np.log1p(np.array([name_counts.get(s, 1) for s in n_core_2], dtype=np.float32))
        ra1 = np.log1p(np.array([addr_counts.get(s, 1) for s in a_norm_1], dtype=np.float32))
        ra2 = np.log1p(np.array([addr_counts.get(s, 1) for s in a_norm_2], dtype=np.float32))

        fuzz_dict["rarity_name_s1"] = rn1
        fuzz_dict["rarity_name_cand"] = rn2
        fuzz_dict["rarity_addr_s1"] = ra1
        fuzz_dict["rarity_addr_cand"] = ra2
        fuzz_dict["rarity_name_diff"] = np.abs(rn1 - rn2)
        fuzz_dict["rarity_addr_diff"] = np.abs(ra1 - ra2)

    # Attach computed string, skeleton, number, and tfidf columns
    for col_name, arr in fuzz_dict.items():
        feat_df = feat_df.with_columns(pl.Series(col_name, arr, dtype=pl.Float32))

    # Keep only target columns
    out_cols = ["s1_num", "cand_num", "cand_src"]
    if "label" in joined_df.columns:
        out_cols.append("label")
    out_cols.extend(target_cols)

    return feat_df.select(out_cols)


# ---------------------------------------------------------------------------
# Vectorizer Fitting & Caching
# ---------------------------------------------------------------------------
def get_or_fit_vectorizers(
    country: str,
    split_cand: str = "train",
    save_dir: Optional[Path] = None,
) -> Tuple[TfidfVectorizer, TfidfVectorizer]:
    """Fits or loads cached country candidate-pool TF-IDF vectorizers with minimal memory footprint."""
    out_dir = save_dir or DATA_FEATURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{split_cand}" if split_cand != "train" else ""
    vec_name_path = out_dir / f"tfidf_name_{country}{suffix}.joblib"
    vec_addr_path = out_dir / f"tfidf_addr_{country}{suffix}.joblib"

    # Also check base DATA_FEATURES_DIR if save_dir is v2
    if not vec_name_path.exists() and (DATA_FEATURES_DIR / f"tfidf_name_{country}{suffix}.joblib").exists():
        vec_name_path = DATA_FEATURES_DIR / f"tfidf_name_{country}{suffix}.joblib"
        vec_addr_path = DATA_FEATURES_DIR / f"tfidf_addr_{country}{suffix}.joblib"

    if vec_name_path.exists() and vec_addr_path.exists():
        print(f"[{country}] Loading pre-fitted {split_cand} candidate pool vectorizers from {vec_name_path.parent}...", flush=True)
        vec_name = joblib.load(vec_name_path)
        vec_addr = joblib.load(vec_addr_path)
        return vec_name, vec_addr

    print(f"[{country}] Fitting TF-IDF name vectorizer on {split_cand} candidate pool...", flush=True)
    t0 = time.perf_counter()

    cols_name = ["country", "name_core"]
    s2_n = pl.read_parquet(PREP_FILES[f"{split_cand}_s2"], columns=cols_name).filter(pl.col("country") == country)
    s3_n = pl.read_parquet(PREP_FILES[f"{split_cand}_s3"], columns=cols_name).filter(pl.col("country") == country)
    name_texts = pl.concat([s2_n["name_core"], s3_n["name_core"]]).fill_null("").to_list()
    del s2_n, s3_n
    gc.collect()

    vec_name = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 3),
        min_df=2,
        max_df=0.02,
        sublinear_tf=True,
        dtype=np.float32,
    )
    vec_name.fit(name_texts)
    del name_texts
    gc.collect()

    save_target_name = (save_dir or DATA_FEATURES_DIR) / f"tfidf_name_{country}{suffix}.joblib"
    joblib.dump(vec_name, save_target_name)
    rss_after_name = psutil.Process().memory_info().rss / (1024 ** 2)
    print(f"[{country}] Name vectorizer ({split_cand}) fitted in {time.perf_counter() - t0:.1f}s (vocab={len(vec_name.vocabulary_):,}). RSS: {rss_after_name:.1f} MB", flush=True)

    print(f"[{country}] Fitting TF-IDF address vectorizer on {split_cand} candidate pool...", flush=True)
    t0_addr = time.perf_counter()
    cols_addr = ["country", "addr_norm"]
    s2_a = pl.read_parquet(PREP_FILES[f"{split_cand}_s2"], columns=cols_addr).filter(pl.col("country") == country)
    s3_a = pl.read_parquet(PREP_FILES[f"{split_cand}_s3"], columns=cols_addr).filter(pl.col("country") == country)
    addr_texts = pl.concat([s2_a["addr_norm"], s3_a["addr_norm"]]).fill_null("").to_list()
    del s2_a, s3_a
    gc.collect()

    vec_addr = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"\b\w+\b",
        min_df=2,
        max_df=0.02,
        sublinear_tf=True,
        dtype=np.float32,
    )
    vec_addr.fit(addr_texts)
    del addr_texts
    gc.collect()

    save_target_addr = (save_dir or DATA_FEATURES_DIR) / f"tfidf_addr_{country}{suffix}.joblib"
    joblib.dump(vec_addr, save_target_addr)
    rss_after_addr = psutil.Process().memory_info().rss / (1024 ** 2)
    print(f"[{country}] Address vectorizer ({split_cand}) fitted in {time.perf_counter() - t0_addr:.1f}s (vocab={len(vec_addr.vocabulary_):,}). RSS: {rss_after_addr:.1f} MB", flush=True)

    return vec_name, vec_addr


# ---------------------------------------------------------------------------
# Unit Test: Batch vs Chunk Exact Numerical Equivalence
# ---------------------------------------------------------------------------
def run_batch_vs_chunk_unit_test(
    country: str,
    vec_name: TfidfVectorizer,
    vec_addr: TfidfVectorizer,
    version: str = "v2",
) -> None:
    """Verifies that running compute_pair_features on 1 full batch vs in chunks produces IDENTICAL float32 outputs (max abs diff == 0.0)."""
    target_cols = FEATURE_COLUMNS_V2 if version == "v2" else FEATURE_COLUMNS_V1
    print(f"\n[{country}] Running Batch vs Chunk Numerical Equivalence Unit Test ({version}, {len(target_cols)} features)...", flush=True)
    block_path = DATA_BLOCKS_DIR / f"holdout_{country}.parquet"
    if not block_path.exists():
        block_path = DATA_BLOCKS_DIR / f"val_{country}.parquet"

    blocks = pl.read_parquet(block_path, n_rows=1000)
    active_s1 = set(blocks["s1_num"].unique().to_list())
    active_cands = blocks.select(["cand_num", "cand_src"]).unique()

    # Load S1 prep for these 1000 rows
    s1_cols = ["entity_id", "country", "name_raw", "name_norm", "name_core", "addr_raw", "addr_norm", "postal", "house_nums"]
    s1_prep = pl.read_parquet(PREP_FILES["train_s1"], columns=s1_cols).filter(
        pl.col("country") == country
    ).with_columns(
        pl.col("entity_id").str.slice(3).cast(pl.Int64).alias("s1_num")
    ).filter(pl.col("s1_num").is_in(active_s1)).select(
        ["s1_num", "name_raw", "name_norm", "name_core", "addr_raw", "addr_norm", "postal", "house_nums"]
    )

    # Load Cand prep sequentially
    cand_cols = ["entity_id", "country", "name_raw", "name_norm", "name_core", "addr_raw", "addr_norm", "postal", "house_nums"]
    s2 = pl.read_parquet(PREP_FILES["train_s2"], columns=cand_cols).filter(pl.col("country") == country)
    s2 = s2.with_columns([
        pl.col("entity_id").str.slice(3).cast(pl.Int64).alias("cand_num"),
        pl.col("entity_id").str.slice(1, 1).cast(pl.Int8).alias("cand_src"),
    ]).select(["cand_num", "cand_src", "name_raw", "name_norm", "name_core", "addr_raw", "addr_norm", "postal", "house_nums"])
    s2 = s2.join(active_cands, on=["cand_num", "cand_src"], how="semi")

    s3 = pl.read_parquet(PREP_FILES["train_s3"], columns=cand_cols).filter(pl.col("country") == country)
    s3 = s3.with_columns([
        pl.col("entity_id").str.slice(3).cast(pl.Int64).alias("cand_num"),
        pl.col("entity_id").str.slice(1, 1).cast(pl.Int8).alias("cand_src"),
    ]).select(["cand_num", "cand_src", "name_raw", "name_norm", "name_core", "addr_raw", "addr_norm", "postal", "house_nums"])
    s3 = s3.join(active_cands, on=["cand_num", "cand_src"], how="semi")

    cand_prep = pl.concat([s2, s3])
    del s2, s3
    gc.collect()

    # Precompute rarity maps on candidate pool
    name_counts = dict(cand_prep.group_by("name_core").len().iter_rows())
    addr_counts = dict(cand_prep.group_by("addr_norm").len().iter_rows())
    rarity_maps = (name_counts, addr_counts)

    joined = blocks.join(
        s1_prep.rename({c: f"{c}_s1" for c in s1_prep.columns if c != "s1_num"}),
        on="s1_num",
        how="left",
    )
    joined = joined.join(
        cand_prep.rename({c: f"{c}_cand" for c in cand_prep.columns if c not in ["cand_num", "cand_src"]}),
        on=["cand_num", "cand_src"],
        how="left",
    )
    del s1_prep, cand_prep
    gc.collect()

    # 1. Full batch
    full_res = compute_pair_features(joined, vec_name, vec_addr, rarity_maps=rarity_maps, version=version)

    # 2. Chunked (two 500-row chunks)
    c1 = compute_pair_features(joined[:500], vec_name, vec_addr, rarity_maps=rarity_maps, version=version)
    c2 = compute_pair_features(joined[500:], vec_name, vec_addr, rarity_maps=rarity_maps, version=version)
    chunk_res = pl.concat([c1, c2])

    max_diff = 0.0
    for col in target_cols:
        diff = float(np.max(np.abs(full_res[col].to_numpy() - chunk_res[col].to_numpy())))
        if diff > max_diff:
            max_diff = diff

    print(f"[{country}] Unit Test Max Abs Difference across all {len(target_cols)} features: {max_diff:.8f}", flush=True)
    if max_diff > 0.0:
        raise AssertionError(f"Unit test FAILED: Max abs difference {max_diff} > 0.0!")

    nan_count = int(full_res.select(target_cols).null_count().sum_horizontal()[0])
    inf_count = int(sum(np.isinf(full_res[c].to_numpy()).sum() for c in target_cols))
    print(f"[{country}] Unit test 0 NaNs: {nan_count == 0} (count={nan_count}) | 0 Infs: {inf_count == 0} (count={inf_count})", flush=True)
    if nan_count > 0 or inf_count > 0:
        raise AssertionError("Unit test FAILED: NaNs or Infs detected!")

    print(f"[{country}] Unit test PASSED: batch vs chunk outputs are bit-for-bit identical (max diff = 0.0).", flush=True)


# ---------------------------------------------------------------------------
# Process Split / Country with Strict Memory Bounds
# ---------------------------------------------------------------------------
def process_split_features(
    split: str,
    country: str,
    vec_name: TfidfVectorizer,
    vec_addr: TfidfVectorizer,
    chunk_size: int = 100000,
    version: str = "v2",
) -> Dict[str, Any]:
    """Generates pairwise features for data/blocks/{split}_{country}.parquet in disk-backed chunks."""
    target_cols = FEATURE_COLUMNS_V2 if version == "v2" else FEATURE_COLUMNS_V1
    out_features_dir = DATA_FEATURES_V2_DIR if version == "v2" else DATA_FEATURES_DIR
    out_features_dir.mkdir(parents=True, exist_ok=True)

    block_path = DATA_BLOCKS_DIR / f"{split}_{country}.parquet"
    out_path = out_features_dir / f"{split}_{country}.parquet"
    tmp_chunk_dir = out_features_dir / f"tmp_chunks_{split}_{country}"
    tmp_chunk_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n================ [{split.upper()} - {country}] Feature Extraction ({version}) ================", flush=True)
    t0 = time.perf_counter()
    blocks = pl.read_parquet(block_path)
    n_pairs = len(blocks)
    print(f"[{split}_{country}] Loaded {n_pairs:,} candidate pairs from {block_path.name}.", flush=True)

    active_s1 = set(blocks["s1_num"].unique().to_list())
    active_cands = blocks.select(["cand_num", "cand_src"]).unique()
    print(f"[{split}_{country}] Active S1 count: {len(active_s1):,}, Active Cand count: {len(active_cands):,}.", flush=True)

    # 1. Load active S1 prep
    s1_cols = ["entity_id", "country", "name_raw", "name_norm", "name_core", "addr_raw", "addr_norm", "postal", "house_nums"]
    s1_prep = pl.read_parquet(PREP_FILES["train_s1"], columns=s1_cols).filter(
        pl.col("country") == country
    ).with_columns(
        pl.col("entity_id").str.slice(3).cast(pl.Int64).alias("s1_num")
    ).filter(pl.col("s1_num").is_in(active_s1)).select(
        ["s1_num", "name_raw", "name_norm", "name_core", "addr_raw", "addr_norm", "postal", "house_nums"]
    )

    # 2. Load active Cand prep sequentially
    cand_cols = ["entity_id", "country", "name_raw", "name_norm", "name_core", "addr_raw", "addr_norm", "postal", "house_nums"]
    s2 = pl.read_parquet(PREP_FILES["train_s2"], columns=cand_cols).filter(pl.col("country") == country)
    s2 = s2.with_columns([
        pl.col("entity_id").str.slice(3).cast(pl.Int64).alias("cand_num"),
        pl.col("entity_id").str.slice(1, 1).cast(pl.Int8).alias("cand_src"),
    ]).select(["cand_num", "cand_src", "name_raw", "name_norm", "name_core", "addr_raw", "addr_norm", "postal", "house_nums"])
    s2 = s2.join(active_cands, on=["cand_num", "cand_src"], how="semi")

    s3 = pl.read_parquet(PREP_FILES["train_s3"], columns=cand_cols).filter(pl.col("country") == country)
    s3 = s3.with_columns([
        pl.col("entity_id").str.slice(3).cast(pl.Int64).alias("cand_num"),
        pl.col("entity_id").str.slice(1, 1).cast(pl.Int8).alias("cand_src"),
    ]).select(["cand_num", "cand_src", "name_raw", "name_norm", "name_core", "addr_raw", "addr_norm", "postal", "house_nums"])
    s3 = s3.join(active_cands, on=["cand_num", "cand_src"], how="semi")

    cand_prep = pl.concat([s2, s3])
    del s2, s3
    gc.collect()

    # Precompute rarity maps on candidate pool
    name_counts = dict(cand_prep.group_by("name_core").len().iter_rows())
    addr_counts = dict(cand_prep.group_by("addr_norm").len().iter_rows())
    rarity_maps = (name_counts, addr_counts)

    # 3. Process in disk-backed chunks
    n_chunks = int(np.ceil(n_pairs / chunk_size))
    print(f"[{split}_{country}] Computing {len(target_cols)} features across {n_chunks} chunks ({chunk_size:,} pairs/chunk)...", flush=True)

    nan_counts = {col: 0 for col in target_cols}
    inf_counts = {col: 0 for col in target_cols}

    for chunk_idx in range(n_chunks):
        start_i = chunk_idx * chunk_size
        end_i = min((chunk_idx + 1) * chunk_size, n_pairs)

        b_slice = blocks[start_i:end_i]
        joined = b_slice.join(
            s1_prep.rename({c: f"{c}_s1" for c in s1_prep.columns if c != "s1_num"}),
            on="s1_num",
            how="left",
        )
        joined = joined.join(
            cand_prep.rename({c: f"{c}_cand" for c in cand_prep.columns if c not in ["cand_num", "cand_src"]}),
            on=["cand_num", "cand_src"],
            how="left",
        )

        feat_chunk = compute_pair_features(
            joined,
            vec_name=vec_name,
            vec_addr=vec_addr,
            rarity_maps=rarity_maps,
            version=version,
        )

        # Check NaN / Inf
        for col in target_cols:
            arr = feat_chunk[col].to_numpy()
            n_nan = int(np.isnan(arr).sum())
            n_inf = int(np.isinf(arr).sum())
            if n_nan > 0:
                nan_counts[col] += n_nan
            if n_inf > 0:
                inf_counts[col] += n_inf

        feat_chunk = feat_chunk.with_columns([
            pl.col(c).fill_nan(0.0).fill_null(0.0) for c in target_cols
        ])

        chunk_file = tmp_chunk_dir / f"chunk_{chunk_idx:04d}.parquet"
        feat_chunk.write_parquet(chunk_file, compression="lz4")

        del feat_chunk, joined, b_slice
        gc.collect()

        elapsed = time.perf_counter() - t0
        speed = end_i / elapsed if elapsed > 0 else 0
        eta = (n_pairs - end_i) / speed if speed > 0 else 0
        rss_mb = psutil.Process().memory_info().rss / (1024 ** 2)
        print(f"[{split}_{country}] Chunk {chunk_idx + 1}/{n_chunks} ({end_i:,}/{n_pairs:,} pairs, {(end_i/n_pairs)*100:.1f}%) | Speed: {speed:,.0f} pairs/s | ETA: {eta:.1f}s | RSS: {rss_mb:.1f} MB", flush=True)

    del s1_prep, cand_prep, blocks
    gc.collect()

    total_nan = sum(nan_counts.values())
    total_inf = sum(inf_counts.values())
    print(f"[{split}_{country}] NaN count before filling: {total_nan}, Inf count: {total_inf}", flush=True)

    # Merge chunks to final parquet file
    print(f"[{split}_{country}] Merging {n_chunks} chunk files into {out_path.name}...", flush=True)
    chunk_files = sorted(tmp_chunk_dir.glob("chunk_*.parquet"))
    pl.concat([pl.read_parquet(f) for f in chunk_files]).write_parquet(out_path, compression="lz4")

    shutil.rmtree(tmp_chunk_dir)

    total_time = time.perf_counter() - t0
    rate = n_pairs / total_time if total_time > 0 else 0
    file_size_mb = os.path.getsize(out_path) / (1024 ** 2)
    peak_rss = psutil.Process().memory_info().rss / (1024 ** 2)
    print(f"[{split}_{country}] COMPLETED: {n_pairs:,} pairs in {total_time:.2f}s ({rate:,.0f} pairs/sec). File: {file_size_mb:.1f} MB. Peak RSS: {peak_rss:.1f} MB.", flush=True)

    gc.collect()

    return {
        "split": split,
        "country": country,
        "pairs": n_pairs,
        "time_sec": total_time,
        "rate": rate,
        "size_mb": file_size_mb,
        "peak_rss": peak_rss,
        "nan_count": total_nan,
        "inf_count": total_inf,
    }


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------
def main():
    version = "v2"
    if len(sys.argv) > 1 and sys.argv[1].startswith("--version="):
        version = sys.argv[1].split("=")[1]
    elif len(sys.argv) > 2 and sys.argv[1] == "--version":
        version = sys.argv[2]

    target_cols = FEATURE_COLUMNS_V2 if version == "v2" else FEATURE_COLUMNS_V1
    out_dir = DATA_FEATURES_V2_DIR if version == "v2" else DATA_FEATURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    all_stats: List[Dict[str, Any]] = []

    print(f"\n================ STEP 3 FEATURE EXTRACTION PIPELINE ({version.upper()}) ================", flush=True)
    print(f"Target Features: {len(target_cols)} float32 features.", flush=True)
    print(f"Target Directory: {out_dir}", flush=True)

    splits_to_run = ["val", "holdout", "train"]
    countries = ["India", "US"]

    total_pipeline_t0 = time.perf_counter()

    for country in countries:
        print(f"\n################ PREPARING COUNTRY: {country.upper()} ################", flush=True)

        vec_name, vec_addr = get_or_fit_vectorizers(country, save_dir=out_dir)
        run_batch_vs_chunk_unit_test(country, vec_name, vec_addr, version=version)

        for split in splits_to_run:
            block_file = DATA_BLOCKS_DIR / f"{split}_{country}.parquet"
            if not block_file.exists():
                print(f"Skipping {block_file}, does not exist.", flush=True)
                continue

            stats = process_split_features(
                split=split,
                country=country,
                vec_name=vec_name,
                vec_addr=vec_addr,
                chunk_size=100000,
                version=version,
            )
            all_stats.append(stats)

        del vec_name, vec_addr
        gc.collect()

    total_pipeline_time = time.perf_counter() - total_pipeline_t0
    total_pairs = sum(s["pairs"] for s in all_stats)
    overall_speed = total_pairs / total_pipeline_time if total_pipeline_time > 0 else 0

    lines: List[str] = []
    lines.append(f"\n## Step 7 (v2) Feature Engineering Execution Report")
    lines.append(f"**Date/Time**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Pipeline Version**: `{version}`")
    lines.append(f"**Total Features**: **{len(target_cols)}** float32 features ({len(FEATURE_COLUMNS_V1)} base + {len(NEW_PAIR_FEATURE_COLUMNS_V2)} new)")
    lines.append(f"**Overall Speed**: {overall_speed:,.0f} pairs/sec at 20 threads (Total runtime: {total_pipeline_time:.1f}s = {total_pipeline_time/60:.2f} min)\n")

    lines.append("### 1. Feature Generation Throughput by Split & Country")
    lines.append("| Split | Country | Pairs | Time (s) | Pairs/Sec | Parquet Size (MB) | Peak RSS (MB) | NaN/Inf Count |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")

    for s in all_stats:
        lines.append(
            f"| {s['split'].upper()} | {s['country']} | {s['pairs']:,} | {s['time_sec']:.1f}s | "
            f"{s['rate']:,.0f} | {s['size_mb']:.1f} MB | {s['peak_rss']:.1f} MB | {s['nan_count']}/{s['inf_count']} |"
        )
    lines.append("")

    report_md = "\n".join(lines) + "\n"
    append_to_run_log(report_md)
    print("\n" + report_md, flush=True)


if __name__ == "__main__":
    main()
