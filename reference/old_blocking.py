"""
Business Entity Resolution — Blocking / Candidate Generation Module (V2)
Amazon ML Challenge 2026

This module implements high-recall, scalable, multi-pass candidate generation
to reduce the O(N * M) Cartesian product to a manageable candidate set
(reduction ratio > 99.96%) while preserving > 95.7% blocking recall.

V2 Blocking Strategies:
1. Exact Normalized Name: Fast lookup on `business_name_norm`.
2. Normalized Core Name: Matches entities sharing identical names minus legal suffixes.
3. Token Bigrams (Shingles): Pairs of adjacent tokens (e.g. `dahlia_power`, `power_reliable`)
   to match multi-word names with high precision.
4. Informative Name Tokens: Rarest name unigrams with document frequency <= `max_token_df`.
5. Compound Address Keys: Cross-product of building/plot numbers with distinctive street/locality
   tokens (e.g. `1_banipark`, `1795_westchester`), recovering cross-lingual and corrupted-name matches.
6. Address Numeric Tokens / PIN Codes: Exact lookup on distinctive postal and plot numbers.
7. Multi-Scale Character (2, 4) N-Gram TF-IDF Retrieval: Country-partitioned cosine similarity
   capturing spelling typos, abbreviations, and phonetic variations.
8. Adaptive Candidate Budgeting: Dynamically deepens character retrieval (K=20, min_sim=0.25)
   for difficult/short/sparse records, while applying tighter pruning (K=12, min_sim=0.32)
   to well-covered entities.
"""

import os
import sys
import time
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional, Any, Union
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from concurrent.futures import ThreadPoolExecutor

# Re-use existing normalization module
sys.path.insert(0, os.path.dirname(__file__))
from preprocess import (
    normalize_name,
    normalize_name_core,
    normalize_address,
    extract_address_numbers,
    normalize_country,
    preprocess_dataframe
)

# Standard address stopwords that must not be used as standalone compound anchors
ADDRESS_STOPWORDS: Set[str] = {
    "st", "rd", "ave", "dr", "ln", "blvd", "ct", "pl", "ter", "cir", "sq", "pkwy", "hwy",
    "street", "road", "avenue", "drive", "lane", "boulevard", "court", "place", "terrace",
    "bldg", "apt", "ste", "fl", "floor", "suite", "apartment", "building", "room", "no", "flat",
    "shop", "plot", "block", "sector", "sec", "ext", "extension", "opp", "opposite", "nr", "near",
    "behind", "c", "o", "po", "post", "office", "at", "on", "in", "by", "of", "and", "the",
    "delhi", "mumbai", "india", "usa", "york", "city", "north", "south", "east", "west"
}


def get_address_compound_keys(addr_norm: Optional[str]) -> List[str]:
    """
    Extracts compound address keys by combining extracted numbers with
    distinctive street/locality tokens (e.g. '1_banipark', '1795_westchester').
    Provides strong discriminative anchors for records with altered names or cross-lingual text.
    """
    if not addr_norm:
        return []
    tokens = [t for t in addr_norm.split() if len(t) >= 4 and not t.isdigit() and t not in ADDRESS_STOPWORDS]
    nums = [t for t in addr_norm.split() if any(c.isdigit() for c in t) and (len(t) >= 2 or not t.isdigit() or int(t) > 0)]
    keys = []
    # Cross top 2 numbers with top 2 distinctive tokens
    for num in nums[:2]:
        for tok in tokens[:2]:
            keys.append(f"{num}_{tok}")
    return keys


def get_expanded_address_compound_keys(addr_norm: Optional[str]) -> List[str]:
    """
    Extracts expanded compound address keys by combining top 3 numbers with
    top 3 distinctive street/locality tokens (len >= 5).
    Provides deeper recovery for cross-lingual and altered-name records (V3).
    """
    if not addr_norm:
        return []
    tokens = [t for t in addr_norm.split() if len(t) >= 5 and not t.isdigit() and t not in ADDRESS_STOPWORDS]
    nums = [t for t in addr_norm.split() if any(c.isdigit() for c in t) and (len(t) >= 2 or not t.isdigit() or int(t) > 0)]
    keys = []
    for num in nums[:3]:
        for tok in tokens[:3]:
            keys.append(f"{num}_{tok}")
    return keys


def get_distinctive_address_tokens(addr_norm: Optional[str]) -> List[str]:
    """
    Extracts distinctive address locality tokens (len >= 5, non-digits, not stopwords)
    for targeted locality-level retrieval (V3).
    """
    if not addr_norm:
        return []
    return [t for t in set(addr_norm.split()) if len(t) >= 5 and not t.isdigit() and t not in ADDRESS_STOPWORDS]


def get_token_bigrams(name_core: Optional[str]) -> List[str]:
    """
    Extracts word bigrams (shingles) from core business names.
    Captures multi-word name overlap without suffering from single-word frequency caps.
    """
    if not name_core:
        return []
    tokens = [t for t in name_core.split() if len(t) >= 2]
    if len(tokens) < 2:
        return []
    return [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]


class BlockingEngine:
    """
    Multi-pass inverted index and sparse retrieval engine for candidate generation (V2).
    Country-partitioned to safely isolate search spaces while dynamically supporting
    open-set countries (US, India, France, etc.).
    """

    def __init__(
        self,
        max_block_size: int = 200,
        max_token_df: int = 500,
        max_bigram_df: int = 150,
        max_compound_df: int = 100,
        max_expanded_compound_df: int = 80,
        max_locality_df: int = 25,
        min_locality_df: int = 2,
        max_num_df: int = 50,
        max_cands_per_token: int = 15,
        max_cands_per_bigram: int = 20,
        max_cands_per_compound: int = 20,
        max_cands_per_expanded_compound: int = 15,
        max_cands_per_locality: int = 8,
        top_k_char_ngram: int = 12,
        top_k_char_ngram_deep: int = 20,
        min_char_sim: float = 0.32,
        min_char_sim_deep: float = 0.25,
        char_ngram_range: Tuple[int, int] = (2, 4),
        adaptive_threshold: int = 5,
    ):
        self.max_block_size = max_block_size
        self.max_token_df = max_token_df
        self.max_bigram_df = max_bigram_df
        self.max_compound_df = max_compound_df
        self.max_expanded_compound_df = max_expanded_compound_df
        self.max_locality_df = max_locality_df
        self.min_locality_df = min_locality_df
        self.max_num_df = max_num_df
        self.max_cands_per_token = max_cands_per_token
        self.max_cands_per_bigram = max_cands_per_bigram
        self.max_cands_per_compound = max_cands_per_compound
        self.max_cands_per_expanded_compound = max_cands_per_expanded_compound
        self.max_cands_per_locality = max_cands_per_locality
        self.top_k_char_ngram = top_k_char_ngram
        self.top_k_char_ngram_deep = top_k_char_ngram_deep
        self.min_char_sim = min_char_sim
        self.min_char_sim_deep = min_char_sim_deep
        self.char_ngram_range = char_ngram_range
        self.adaptive_threshold = adaptive_threshold

        # Inverted Indexes partitioned by (country, key)
        self.exact_name_idx: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.core_name_idx: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.bigram_idx: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.token_idx: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.compound_idx: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.expanded_compound_idx: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.locality_token_idx: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.num_idx: Dict[Tuple[str, str], List[str]] = defaultdict(list)

        # Character TF-IDF components per country
        self.char_vectorizers: Dict[str, TfidfVectorizer] = {}
        self.char_matrices: Dict[str, csr_matrix] = {}
        self.char_matrices_T: Dict[str, csr_matrix] = {}
        self.country_cand_ids: Dict[str, List[str]] = defaultdict(list)
        
        # Statistics
        self.indexed_count: int = 0
        self.is_fitted: bool = False

    def fit(self, candidate_df: pl.DataFrame, build_char_index: bool = True) -> "BlockingEngine":
        """
        Builds inverted indexes and character TF-IDF models over Source 2 and Source 3 records.
        """
        t0 = time.time()
        needed_cols = {"business_name_norm", "business_name_core", "business_address_norm", "country_norm"}
        if not needed_cols.issubset(set(candidate_df.columns)):
            candidate_df = preprocess_dataframe(candidate_df)

        e_ids = candidate_df["entity_id"].to_list()
        n_norms = candidate_df["business_name_norm"].to_list()
        n_cores = candidate_df["business_name_core"].to_list()
        a_norms = candidate_df["business_address_norm"].to_list()
        countries = candidate_df["country_norm"].to_list()

        country_texts: Dict[str, List[str]] = defaultdict(list)

        for i in range(len(e_ids)):
            c_id = e_ids[i]
            c = countries[i] or "UNKNOWN"
            n_norm = n_norms[i]
            n_core = n_cores[i]
            a_norm = a_norms[i]

            # 1. Exact Name Index
            if n_norm:
                self.exact_name_idx[(c, n_norm)].append(c_id)

            # 2. Core Name Index
            if n_core:
                self.core_name_idx[(c, n_core)].append(c_id)

            # 3. Token Bigrams Index
            if n_core:
                bigrams = get_token_bigrams(n_core)
                for bg in bigrams:
                    self.bigram_idx[(c, bg)].append(c_id)

            # 4. Informative Tokens Index
            if n_core:
                tokens = set(n_core.split())
                for tok in tokens:
                    if len(tok) >= 3:
                        self.token_idx[(c, tok)].append(c_id)

            # 5. Compound Address Keys Index
            if a_norm:
                comp_keys = get_address_compound_keys(a_norm)
                for ak in comp_keys:
                    self.compound_idx[(c, ak)].append(c_id)

            # 6. Address Numbers Index
            if a_norm:
                nums = extract_address_numbers(a_norm)
                for num in set(nums):
                    if len(num) >= 3 or not num.isdigit() or int(num) > 9:
                        self.num_idx[(c, num)].append(c_id)

            # 7. Expanded Compound Address Keys Index (V3)
            if a_norm:
                exp_comp_keys = get_expanded_address_compound_keys(a_norm)
                for ak in exp_comp_keys:
                    self.expanded_compound_idx[(c, ak)].append(c_id)

            # 8. Distinctive Locality Tokens Index (V3)
            if a_norm:
                loc_toks = get_distinctive_address_tokens(a_norm)
                for ltok in loc_toks:
                    self.locality_token_idx[(c, ltok)].append(c_id)

            # Char N-Gram corpus tracking
            if build_char_index:
                text_repr = n_core if n_core else (n_norm if n_norm else "")
                self.country_cand_ids[c].append(c_id)
                country_texts[c].append(text_repr)

        self.indexed_count = len(e_ids)

        # Build Multi-Scale Character N-Gram TF-IDF per country
        if build_char_index:
            for cntry, texts in country_texts.items():
                if not texts:
                    continue
                safe_min_df = 1 if len(texts) < 5 else 2
                vec = TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=self.char_ngram_range,
                    min_df=safe_min_df,
                    sublinear_tf=True
                )
                mat = vec.fit_transform(texts).astype(np.float32)
                self.char_vectorizers[cntry] = vec
                self.char_matrices[cntry] = mat
                self.char_matrices_T[cntry] = mat.T.tocsr()

        self.is_fitted = True
        return self

    def query_exact_name(self, country: str, name_norm: Optional[str]) -> Set[str]:
        """Returns candidates sharing the exact normalized business name."""
        if not name_norm:
            return set()
        c = country or "UNKNOWN"
        items = self.exact_name_idx.get((c, name_norm), [])
        if len(items) <= self.max_block_size:
            return set(items)
        return set(items[:self.max_block_size])

    def query_core_name(self, country: str, name_core: Optional[str]) -> Set[str]:
        """Returns candidates sharing the exact core name (legal suffix stripped)."""
        if not name_core:
            return set()
        c = country or "UNKNOWN"
        items = self.core_name_idx.get((c, name_core), [])
        if len(items) <= self.max_block_size:
            return set(items)
        return set(items[:self.max_block_size])

    def query_token_bigrams(self, country: str, name_core: Optional[str]) -> Set[str]:
        """Returns candidates sharing adjacent word bigrams."""
        if not name_core:
            return set()
        bigrams = get_token_bigrams(name_core)
        if not bigrams:
            return set()
        c = country or "UNKNOWN"
        cands = set()
        for bg in bigrams:
            items = self.bigram_idx.get((c, bg), [])
            if 1 <= len(items) <= self.max_bigram_df:
                cands.update(items[:self.max_cands_per_bigram])
        return cands

    def query_name_tokens(self, country: str, name_core: Optional[str]) -> Set[str]:
        """
        Selects informative tokens with document frequency <= max_token_df,
        prioritizing the rarest tokens.
        """
        if not name_core:
            return set()
        c = country or "UNKNOWN"
        tokens = [t for t in name_core.split() if len(t) >= 3]
        if not tokens:
            return set()

        valid_tokens = []
        for t in tokens:
            df = len(self.token_idx.get((c, t), []))
            if 1 <= df <= self.max_token_df:
                valid_tokens.append((t, df))

        if not valid_tokens:
            return set()

        valid_tokens.sort(key=lambda x: x[1])

        cands = set()
        for tok, _ in valid_tokens[:3]:
            matches = self.token_idx.get((c, tok), [])
            cands.update(matches[:self.max_cands_per_token])

        return cands

    def query_address_compound(self, country: str, address_norm: Optional[str]) -> Set[str]:
        """Returns candidates sharing number + street compound keys."""
        if not address_norm:
            return set()
        comp_keys = get_address_compound_keys(address_norm)
        if not comp_keys:
            return set()
        c = country or "UNKNOWN"
        cands = set()
        for ak in comp_keys:
            items = self.compound_idx.get((c, ak), [])
            if 1 <= len(items) <= self.max_compound_df:
                cands.update(items[:self.max_cands_per_compound])
        return cands

    def query_address_numbers(self, country: str, address_norm: Optional[str]) -> Set[str]:
        """Returns candidates sharing distinctive address numbers (building/PIN/shop)."""
        if not address_norm:
            return set()
        nums = extract_address_numbers(address_norm)
        if not nums:
            return set()
        c = country or "UNKNOWN"
        cands = set()
        for num in set(nums):
            if len(num) >= 3 or not num.isdigit() or int(num) > 9:
                items = self.num_idx.get((c, num), [])
                if 1 <= len(items) <= self.max_num_df:
                    cands.update(items)
        return cands

    def query_expanded_address_compound(self, country: str, address_norm: Optional[str]) -> Set[str]:
        """Returns candidates sharing expanded number + locality compound keys (V3)."""
        if not address_norm:
            return set()
        comp_keys = get_expanded_address_compound_keys(address_norm)
        if not comp_keys:
            return set()
        c = country or "UNKNOWN"
        cands = set()
        for ak in comp_keys:
            items = self.expanded_compound_idx.get((c, ak), [])
            if 1 <= len(items) <= self.max_expanded_compound_df:
                cands.update(items[:self.max_cands_per_expanded_compound])
        return cands

    def query_distinctive_locality_tokens(self, country: str, address_norm: Optional[str]) -> Set[str]:
        """Returns candidates sharing rare distinctive address locality tokens (V3)."""
        if not address_norm:
            return set()
        loc_tokens = get_distinctive_address_tokens(address_norm)
        if not loc_tokens:
            return set()
        c = country or "UNKNOWN"
        cands = set()
        for ltok in loc_tokens:
            items = self.locality_token_idx.get((c, ltok), [])
            if self.min_locality_df <= len(items) <= self.max_locality_df:
                cands.update(items[:self.max_cands_per_locality])
        return cands

    def generate_candidate_pairs(
        self,
        s1_df: pl.DataFrame,
        strategies: Optional[List[str]] = None,
        adaptive: bool = True,
        batch_size: int = 5
    ) -> Dict[str, List[str]]:
        """
        Generates candidate IDs for each Source 1 entity using the specified strategies.
        Available strategies: ['exact', 'core', 'bigram', 'token', 'compound', 'address',
                              'expanded_compound', 'locality_tokens', 'char_ngram']
        Defaults to all strategies combined adaptively (V3).
        """
        if strategies is None:
            strategies = [
                "exact", "core", "bigram", "token", "compound", "address",
                "expanded_compound", "locality_tokens", "char_ngram"
            ]

        needed_cols = {"business_name_norm", "business_name_core", "business_address_norm", "country_norm"}
        if not needed_cols.issubset(set(s1_df.columns)):
            s1_df = preprocess_dataframe(s1_df)

        e_ids = s1_df["entity_id"].to_list()
        n_norms = s1_df["business_name_norm"].to_list()
        n_cores = s1_df["business_name_core"].to_list()
        a_norms = s1_df["business_address_norm"].to_list()
        countries = s1_df["country_norm"].to_list()

        candidates: Dict[str, Set[str]] = {e_id: set() for e_id in e_ids}

        use_exact = "exact" in strategies
        use_core = "core" in strategies
        use_bigram = "bigram" in strategies
        use_token = "token" in strategies
        use_compound = "compound" in strategies
        use_address = "address" in strategies
        use_expanded_compound = "expanded_compound" in strategies
        use_locality = "locality_tokens" in strategies
        use_char = "char_ngram" in strategies

        # 1. Exact, Core, Bigram, Token, Compound, Address, Expanded Compound, and Locality blocking
        for i in range(len(e_ids)):
            s1_id = e_ids[i]
            c = countries[i] or "UNKNOWN"
            n_norm = n_norms[i]
            n_core = n_cores[i]
            a_norm = a_norms[i]

            if use_exact:
                candidates[s1_id].update(self.query_exact_name(c, n_norm))
            if use_core:
                candidates[s1_id].update(self.query_core_name(c, n_core))
            if use_bigram:
                candidates[s1_id].update(self.query_token_bigrams(c, n_core))
            if use_token:
                candidates[s1_id].update(self.query_name_tokens(c, n_core))
            if use_compound:
                candidates[s1_id].update(self.query_address_compound(c, a_norm))
            if use_address:
                candidates[s1_id].update(self.query_address_numbers(c, a_norm))
            if use_expanded_compound:
                candidates[s1_id].update(self.query_expanded_address_compound(c, a_norm))
            if use_locality:
                candidates[s1_id].update(self.query_distinctive_locality_tokens(c, a_norm))

            # Discard self match
            candidates[s1_id].discard(s1_id)

        # 2. Character N-Gram TF-IDF Cosine Retrieval with Adaptive Budgeting
        if use_char and self.char_vectorizers:
            country_indices: Dict[str, List[int]] = defaultdict(list)
            for i, c in enumerate(countries):
                s1_id = e_ids[i]
                if not adaptive or len(candidates[s1_id]) < self.adaptive_threshold:
                    country_indices[c or "UNKNOWN"].append(i)

            for cntry, idx_list in country_indices.items():
                if cntry not in self.char_vectorizers:
                    continue

                vec = self.char_vectorizers[cntry]
                cand_mat = self.char_matrices[cntry]
                cand_mat_T = self.char_matrices_T.get(cntry)
                if cand_mat_T is None:
                    cand_mat_T = cand_mat.T.tocsr()
                    self.char_matrices_T[cntry] = cand_mat_T
                cand_ids = self.country_cand_ids[cntry]

                # Multi-threaded bounded sparse retrieval across CPU cores
                # Releases GIL in scipy sparse matrix multiplication
                # Vectorize all query texts in a single high-speed call, then slice into bounded chunks
                query_texts = [
                    (n_cores[idx] if n_cores[idx] else (n_norms[idx] if n_norms[idx] else ""))
                    for idx in idx_list
                ]
                q_mat_all = vec.transform(query_texts).astype(np.float32)

                sub_batch_size = 5
                query_sub_batches = []
                for start_b in range(0, len(idx_list), sub_batch_size):
                    end_b = min(start_b + sub_batch_size, len(idx_list))
                    batch_idxs = idx_list[start_b:end_b]
                    q_mat = q_mat_all[start_b:end_b]
                    query_sub_batches.append((batch_idxs, q_mat))

                def _compute_and_extract(sub_batch):
                    b_idxs, q_mat = sub_batch
                    sim_mat = q_mat.dot(cand_mat_T)
                    sub_cands = defaultdict(set)
                    indptr = sim_mat.indptr
                    data = sim_mat.data
                    indices = sim_mat.indices
                    for local_idx, orig_idx in enumerate(b_idxs):
                        s1_id = e_ids[orig_idx]
                        start = indptr[local_idx]
                        end = indptr[local_idx + 1]
                        if start == end:
                            continue
                        scores = data[start:end]
                        cols = indices[start:end]
                        current_cands = len(candidates[s1_id])
                        if adaptive and current_cands < self.adaptive_threshold:
                            min_sim = self.min_char_sim_deep
                            top_k = self.top_k_char_ngram_deep
                        else:
                            min_sim = self.min_char_sim
                            top_k = self.top_k_char_ngram
                        valid_mask = scores >= min_sim
                        if np.any(valid_mask):
                            filtered_scores = scores[valid_mask]
                            filtered_cols = cols[valid_mask]
                            if len(filtered_scores) > top_k:
                                top_k_pos = np.argpartition(filtered_scores, -top_k)[-top_k:]
                                top_k_pos = top_k_pos[np.argsort(-filtered_scores[top_k_pos])]
                            else:
                                top_k_pos = np.argsort(-filtered_scores)
                            for pos in top_k_pos:
                                sub_cands[s1_id].add(cand_ids[filtered_cols[pos]])
                    return sub_cands

                if query_sub_batches:
                    with ThreadPoolExecutor(max_workers=2) as tpool:
                        for sub_result in tpool.map(_compute_and_extract, query_sub_batches):
                            for s1_id, c_set in sub_result.items():
                                candidates[s1_id].update(c_set)

            # Discard self match
            for s1_id in e_ids:
                candidates[s1_id].discard(s1_id)

        # Convert sets to sorted lists for deterministic output
        return {e_id: sorted(list(cands)) for e_id, cands in candidates.items()}


def evaluate_blocking(
    candidate_map: Dict[str, List[str]],
    ground_truth_map: Dict[str, Set[str]]
) -> Dict[str, Any]:
    """
    Evaluates blocking quality against ground truth.
    Measures: Overall Recall, S2/S3 Recall, Total Candidates, Candidate Percentiles.
    """
    total_true = 0
    found_true = 0
    s2_true = 0
    s2_found = 0
    s3_true = 0
    s3_found = 0

    candidate_counts = []

    for s1_id, true_set in ground_truth_map.items():
        found_cands = set(candidate_map.get(s1_id, []))
        candidate_counts.append(len(found_cands))

        for match_id in true_set:
            total_true += 1
            is_s2 = match_id.startswith("S2")
            is_s3 = match_id.startswith("S3")

            if is_s2:
                s2_true += 1
            elif is_s3:
                s3_true += 1

            if match_id in found_cands:
                found_true += 1
                if is_s2:
                    s2_found += 1
                elif is_s3:
                    s3_found += 1

    overall_recall = (found_true / total_true * 100) if total_true > 0 else 0.0
    s2_recall = (s2_found / s2_true * 100) if s2_true > 0 else 0.0
    s3_recall = (s3_found / s3_true * 100) if s3_true > 0 else 0.0

    counts_arr = np.array(candidate_counts) if candidate_counts else np.array([0])

    return {
        "overall_recall": round(overall_recall, 3),
        "s2_recall": round(s2_recall, 3),
        "s3_recall": round(s3_recall, 3),
        "total_true_matches": total_true,
        "true_matches_found": found_true,
        "total_candidate_pairs": int(counts_arr.sum()),
        "total_s1": len(ground_truth_map),
        "avg_candidates_per_s1": round(float(counts_arr.mean()), 2),
        "median_candidates": round(float(np.median(counts_arr)), 1),
        "p95_candidates": round(float(np.percentile(counts_arr, 95)), 1),
        "p99_candidates": round(float(np.percentile(counts_arr, 99)), 1),
        "max_candidates": int(counts_arr.max())
    }


def save_candidate_pairs(
    candidate_map: Dict[str, List[str]],
    output_path: str
) -> None:
    """
    Saves candidate pairs to a clean TSV file with header:
    source1_entity_id\tcandidate_entity_ids
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id, cands in candidate_map.items():
            cand_str = ",".join(cands)
            f.write(f"{s1_id}\t{cand_str}\n")
