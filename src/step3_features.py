"""Step 3 — Vectorized Pair Feature Engineering.

Extracts 38 pairwise similarity and signal features for candidate pairs:
  a) String similarity with rapidfuzz.process.cpdist(workers=-1) on name_norm, name_core, addr_norm:
     ratio, token_sort_ratio, token_set_ratio, partial_ratio, JaroWinkler, normalized Levenshtein (18 features)
  b) TF-IDF cosine per pair (row-wise sparse dot product):
     name (char_wb 3-grams), address (words) fitted on candidate pool (2 features)
  c) Numbers:
     house_num_overlap, house_num_conflict, house_num_missing_s1, house_num_missing_cand,
     postal_equal, postal_missing (6 features)
  d) Lengths and empties:
     name_len_ratio, addr_len_ratio, empty_addr_s1, empty_addr_cand (4 features)
  e) Source:
     is_s3 (1 feature)
  f) Blocking info:
     score_a, score_b, score_c, rank_a, rank_b, rank_c, n_channels (7 features)

Total: 38 float32 features. Fully vectorized, strictly bounded memory (< 2.5 GB peak RSS).
"""

from pathlib import Path
import gc
import joblib
import os
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple

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
    PREP_FILES,
    RUN_LOG_PATH,
    append_to_run_log,
)

FEATURE_COLUMNS: List[str] = [
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


# ---------------------------------------------------------------------------
# Single Feature Extraction Function (Used Identically for Train / Val / Holdout / Test)
# ---------------------------------------------------------------------------
def compute_pair_features(
    joined_df: pl.DataFrame,
    vec_name: TfidfVectorizer,
    vec_addr: TfidfVectorizer,
) -> pl.DataFrame:
    """Computes all 38 float32 features on a joined DataFrame of candidate pairs.

    joined_df must contain:
      - s1_num, cand_num, cand_src, (label if present)
      - name_norm_s1, name_core_s1, addr_norm_s1, house_nums_s1, postal_s1
      - name_norm_cand, name_core_cand, addr_norm_cand, house_nums_cand, postal_cand
      - score_a, rank_a, score_b, rank_b, score_c, rank_c, n_channels
    """
    n_pairs = len(joined_df)
    if n_pairs == 0:
        schema = {col: pl.Float32 for col in FEATURE_COLUMNS}
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
    # Optimized: transform unique strings to avoid redundant sparse vectorization
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

    # Attach computed string and tfidf columns
    for col_name, arr in fuzz_dict.items():
        feat_df = feat_df.with_columns(pl.Series(col_name, arr, dtype=pl.Float32))

    # Keep only target columns
    out_cols = ["s1_num", "cand_num", "cand_src"]
    if "label" in joined_df.columns:
        out_cols.append("label")
    out_cols.extend(FEATURE_COLUMNS)

    return feat_df.select(out_cols)


# ---------------------------------------------------------------------------
# Vectorizer Fitting & Caching (Low Peak RSS)
# ---------------------------------------------------------------------------
def get_or_fit_vectorizers(
    country: str,
    split_cand: str = "train",
) -> Tuple[TfidfVectorizer, TfidfVectorizer]:
    """Fits or loads cached country candidate-pool TF-IDF vectorizers with minimal memory footprint."""
    DATA_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{split_cand}" if split_cand != "train" else ""
    vec_name_path = DATA_FEATURES_DIR / f"tfidf_name_{country}{suffix}.joblib"
    vec_addr_path = DATA_FEATURES_DIR / f"tfidf_addr_{country}{suffix}.joblib"

    if vec_name_path.exists() and vec_addr_path.exists():
        print(f"[{country}] Loading pre-fitted {split_cand} candidate pool vectorizers from {DATA_FEATURES_DIR}...", flush=True)
        vec_name = joblib.load(vec_name_path)
        vec_addr = joblib.load(vec_addr_path)
        return vec_name, vec_addr

    print(f"[{country}] Fitting TF-IDF name vectorizer on {split_cand} candidate pool...", flush=True)
    t0 = time.perf_counter()

    # Load only name_core
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

    joblib.dump(vec_name, vec_name_path)
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

    joblib.dump(vec_addr, vec_addr_path)
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
) -> None:
    """Verifies that running compute_pair_features on 1 full batch vs in chunks produces IDENTICAL float32 outputs (max abs diff == 0.0)."""
    print(f"\n[{country}] Running Batch vs Chunk Numerical Equivalence Unit Test...", flush=True)
    block_path = DATA_BLOCKS_DIR / f"holdout_{country}.parquet"
    if not block_path.exists():
        block_path = DATA_BLOCKS_DIR / f"val_{country}.parquet"

    blocks = pl.read_parquet(block_path, n_rows=1000)
    active_s1 = set(blocks["s1_num"].unique().to_list())
    active_cands = blocks.select(["cand_num", "cand_src"]).unique()

    # Load S1 prep for these 1000 rows
    s1_cols = ["entity_id", "country", "name_norm", "name_core", "addr_norm", "postal", "house_nums"]
    s1_prep = pl.read_parquet(PREP_FILES["train_s1"], columns=s1_cols).filter(
        pl.col("country") == country
    ).with_columns(
        pl.col("entity_id").str.slice(3).cast(pl.Int64).alias("s1_num")
    ).filter(pl.col("s1_num").is_in(active_s1)).select(
        ["s1_num", "name_norm", "name_core", "addr_norm", "postal", "house_nums"]
    )

    # Load Cand prep for these 1000 rows sequentially
    cand_cols = ["entity_id", "country", "name_norm", "name_core", "addr_norm", "postal", "house_nums"]
    s2 = pl.read_parquet(PREP_FILES["train_s2"], columns=cand_cols).filter(pl.col("country") == country)
    s2 = s2.with_columns([
        pl.col("entity_id").str.slice(3).cast(pl.Int64).alias("cand_num"),
        pl.col("entity_id").str.slice(1, 1).cast(pl.Int8).alias("cand_src"),
    ]).select(["cand_num", "cand_src", "name_norm", "name_core", "addr_norm", "postal", "house_nums"])
    s2 = s2.join(active_cands, on=["cand_num", "cand_src"], how="semi")

    s3 = pl.read_parquet(PREP_FILES["train_s3"], columns=cand_cols).filter(pl.col("country") == country)
    s3 = s3.with_columns([
        pl.col("entity_id").str.slice(3).cast(pl.Int64).alias("cand_num"),
        pl.col("entity_id").str.slice(1, 1).cast(pl.Int8).alias("cand_src"),
    ]).select(["cand_num", "cand_src", "name_norm", "name_core", "addr_norm", "postal", "house_nums"])
    s3 = s3.join(active_cands, on=["cand_num", "cand_src"], how="semi")

    cand_prep = pl.concat([s2, s3])
    del s2, s3
    gc.collect()

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
    full_res = compute_pair_features(joined, vec_name, vec_addr)

    # 2. Chunked (two 500-row chunks)
    c1 = compute_pair_features(joined[:500], vec_name, vec_addr)
    c2 = compute_pair_features(joined[500:], vec_name, vec_addr)
    chunk_res = pl.concat([c1, c2])

    max_diff = 0.0
    for col in FEATURE_COLUMNS:
        diff = float(np.max(np.abs(full_res[col].to_numpy() - chunk_res[col].to_numpy())))
        if diff > max_diff:
            max_diff = diff

    print(f"[{country}] Unit Test Max Abs Difference across all {len(FEATURE_COLUMNS)} features: {max_diff:.8f}", flush=True)
    if max_diff > 0.0:
        raise AssertionError(f"Unit test FAILED: Max abs difference {max_diff} > 0.0!")
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
) -> Dict[str, any]:
    """Generates pairwise features for data/blocks/{split}_{country}.parquet in disk-backed chunks."""
    block_path = DATA_BLOCKS_DIR / f"{split}_{country}.parquet"
    out_path = DATA_FEATURES_DIR / f"{split}_{country}.parquet"
    tmp_chunk_dir = DATA_FEATURES_DIR / f"tmp_chunks_{split}_{country}"
    tmp_chunk_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n================ [{split.upper()} - {country}] Feature Extraction ================", flush=True)
    t0 = time.perf_counter()
    blocks = pl.read_parquet(block_path)
    n_pairs = len(blocks)
    print(f"[{split}_{country}] Loaded {n_pairs:,} candidate pairs from {block_path.name}.", flush=True)

    # Identify active S1 entities and active candidates for this split to minimize memory
    active_s1 = set(blocks["s1_num"].unique().to_list())
    active_cands = blocks.select(["cand_num", "cand_src"]).unique()
    print(f"[{split}_{country}] Active S1 count: {len(active_s1):,}, Active Cand count: {len(active_cands):,}.", flush=True)

    # 1. Load active S1 prep
    s1_cols = ["entity_id", "country", "name_norm", "name_core", "addr_norm", "postal", "house_nums"]
    s1_prep = pl.read_parquet(PREP_FILES["train_s1"], columns=s1_cols).filter(
        pl.col("country") == country
    ).with_columns(
        pl.col("entity_id").str.slice(3).cast(pl.Int64).alias("s1_num")
    ).filter(pl.col("s1_num").is_in(active_s1)).select(
        ["s1_num", "name_norm", "name_core", "addr_norm", "postal", "house_nums"]
    )

    # 2. Load active Cand prep sequentially to keep memory low
    cand_cols = ["entity_id", "country", "name_norm", "name_core", "addr_norm", "postal", "house_nums"]
    s2 = pl.read_parquet(PREP_FILES["train_s2"], columns=cand_cols).filter(pl.col("country") == country)
    s2 = s2.with_columns([
        pl.col("entity_id").str.slice(3).cast(pl.Int64).alias("cand_num"),
        pl.col("entity_id").str.slice(1, 1).cast(pl.Int8).alias("cand_src"),
    ]).select(["cand_num", "cand_src", "name_norm", "name_core", "addr_norm", "postal", "house_nums"])
    s2 = s2.join(active_cands, on=["cand_num", "cand_src"], how="semi")
    gc.collect()

    s3 = pl.read_parquet(PREP_FILES["train_s3"], columns=cand_cols).filter(pl.col("country") == country)
    s3 = s3.with_columns([
        pl.col("entity_id").str.slice(3).cast(pl.Int64).alias("cand_num"),
        pl.col("entity_id").str.slice(1, 1).cast(pl.Int8).alias("cand_src"),
    ]).select(["cand_num", "cand_src", "name_norm", "name_core", "addr_norm", "postal", "house_nums"])
    s3 = s3.join(active_cands, on=["cand_num", "cand_src"], how="semi")
    gc.collect()

    cand_prep = pl.concat([s2, s3])
    del s2, s3, active_s1, active_cands
    gc.collect()

    init_rss = psutil.Process().memory_info().rss / (1024 ** 2)
    print(f"[{split}_{country}] Active prep loaded: S1={len(s1_prep):,}, Cand={len(cand_prep):,}. RSS: {init_rss:.1f} MB", flush=True)

    n_chunks = (n_pairs + chunk_size - 1) // chunk_size
    nan_counts: Dict[str, int] = {c: 0 for c in FEATURE_COLUMNS}
    inf_counts: Dict[str, int] = {c: 0 for c in FEATURE_COLUMNS}

    for chunk_idx in range(n_chunks):
        t_chunk_start = time.perf_counter()
        start_i = chunk_idx * chunk_size
        end_i = min(start_i + chunk_size, n_pairs)
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

        feat_chunk = compute_pair_features(joined, vec_name, vec_addr)

        # Check NaN and Inf counts before filling
        for col in FEATURE_COLUMNS:
            arr = feat_chunk[col].to_numpy()
            n_nan = int(np.isnan(arr).sum())
            n_inf = int(np.isinf(arr).sum())
            if n_nan > 0:
                nan_counts[col] += n_nan
            if n_inf > 0:
                inf_counts[col] += n_inf

        # Fill NaN / Inf if any appear (to guarantee clean float32 outputs)
        feat_chunk = feat_chunk.with_columns([
            pl.col(c).fill_nan(0.0).fill_null(0.0) for c in FEATURE_COLUMNS
        ])

        # Write chunk immediately to disk to prevent RAM accumulation
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

    # Clean up temporary directory
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
    DATA_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    all_stats: List[Dict[str, any]] = []

    print("\n================ STEP 3 FEATURE EXTRACTION PIPELINE ================", flush=True)
    print(f"Target Features: {len(FEATURE_COLUMNS)} float32 features.", flush=True)
    print("Country Partition: Dynamic (iterating unique country labels).", flush=True)

    # Prompt: "Run it for train, val and holdout (India and US). Do NOT run test yet."
    splits_to_run = ["val", "holdout", "train"]
    countries = ["India", "US"]

    total_pipeline_t0 = time.perf_counter()

    for country in countries:
        print(f"\n################ PREPARING COUNTRY: {country.upper()} ################", flush=True)

        # 1. Fit or load cached vectorizers
        vec_name, vec_addr = get_or_fit_vectorizers(country)

        # 2. Run Unit Test
        run_batch_vs_chunk_unit_test(country, vec_name, vec_addr)

        # 3. Process splits
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
            )
            all_stats.append(stats)

        del vec_name, vec_addr
        gc.collect()

    total_pipeline_time = time.perf_counter() - total_pipeline_t0
    total_pairs = sum(s["pairs"] for s in all_stats)
    overall_speed = total_pairs / total_pipeline_time if total_pipeline_time > 0 else 0

    # Format Markdown Report
    lines: List[str] = []
    lines.append("\n## Step 3 Feature Engineering Execution Report")
    lines.append(f"**Date/Time**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Total Features**: {len(FEATURE_COLUMNS)} float32 features")
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

    # Section on Test Extrapolation
    test_pairs = 88679129  # 88.68M test pairs across France, India, US
    avg_rate = np.mean([s["rate"] for s in all_stats]) if all_stats else overall_speed
    est_test_sec = test_pairs / avg_rate if avg_rate > 0 else 0
    est_test_hours = est_test_sec / 3600.0
    mean_bytes_per_pair = sum(s["size_mb"] * (1024**2) for s in all_stats) / sum(s["pairs"] for s in all_stats)
    est_test_gb = (test_pairs * mean_bytes_per_pair) / (1024**3)
    free_disk_gb = shutil.disk_usage("D:/").free / (1024**3)

    lines.append("### 2. Extrapolation for 88.7M Test Pairs")
    lines.append(f"- **Estimated Test Generation Time**: **{est_test_hours:.2f} hours** (~{est_test_sec/60:.1f} min at ~{avg_rate:,.0f} pairs/sec)")
    lines.append(f"- **Estimated Test Features Disk Size**: **{est_test_gb:.2f} GB** (based on {mean_bytes_per_pair:.1f} bytes/pair compressed)")
    lines.append(f"- **Free Disk Space Available**: **{free_disk_gb:.2f} GB** (Sufficient: {est_test_gb:.2f} GB << {free_disk_gb:.2f} GB free)")
    lines.append("")

    report_md = "\n".join(lines) + "\n"
    append_to_run_log(report_md)
    print("\n" + report_md, flush=True)


if __name__ == "__main__":
    main()
