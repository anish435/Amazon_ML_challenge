"""Step 4 — Model Benchmark, 2-Stage Training, Context Features, and 1:1 Greedy Threshold Assignment.

Pipeline Structure:
  3.1 Quick model benchmark (1M-row sample of train features):
      - LightGBM CPU (num_threads=20) vs XGBoost GPU (device="cuda", tree_method="hist")
      - Reports training time and validation AUC, picks faster if AUC is equal/better (within 0.001)
  3.2 Stage 1 — Pair Model:
      - Trained on all train_* features (India + US together, no country feature)
      - 2-fold OUT-OF-FOLD (OOF) predictions on train (split by S1 entity modulo 2)
      - Full model trained on all train_* to score val_* and holdout_*
  3.3 Stage 2 — Within-S1 Context Model:
      - Computes 6 within-S1 context features from p1:
        rank of p1 in S1, rank in source, diff vs best, diff vs second-best, s1 best p1, count p1 > 0.5
      - Early stopping on validation. Kept if it beats Stage 1 on val macro F0.5
  3.4 Greedy 1:1 Bipartite Assignment + Threshold Tuning:
      - Each candidate cid goes only to highest-scoring S1 entity (ties kept, count reported)
      - Sweeps tau from 0.30 to 0.99 (step 0.01) on val; tests shared vs per-country tau
  3.5 Holdout Evaluation:
      - Confirms ONCE on holdout using tools/eval_macro_f05.py (zero candidate entities included)
      - Reports macro F0.5, precision, recall, % empty, avg matches/entity, and tau curve
  3.6 Artifact Persistence & Error Analysis:
      - Saves models and predictions to data/models/
      - Samples 10 False Positives + 10 False Negatives with raw text and failure diagnostics
      - Appends comprehensive report to reports/run_log.md
"""

from pathlib import Path
import gc
import json
import os
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

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
import psutil
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import xgboost as xgb

from rapidfuzz import fuzz

from src.common import (
    DATA_BLOCKS_DIR,
    DATA_FEATURES_DIR,
    DATA_FEATURES_V2_DIR,
    DATA_MODELS_DIR,
    DATA_MODELS_V2_DIR,
    PREP_FILES,
    RAW_FILES,
    RUN_LOG_PATH,
    append_to_run_log,
)
from tools.eval_macro_f05 import macro_f05_dataframe

BASE_FEATURE_COLUMNS_V1: List[str] = [
    # a) Strings on name_norm (6)
    "name_norm_ratio", "name_norm_token_sort_ratio", "name_norm_token_set_ratio",
    "name_norm_partial_ratio", "name_norm_jaro_winkler", "name_norm_levenshtein",
    # Strings on name_core (6)
    "name_core_ratio", "name_core_token_sort_ratio", "name_core_token_set_ratio",
    "name_core_partial_ratio", "name_core_jaro_winkler", "name_core_levenshtein",
    # Strings on addr_norm (6)
    "addr_norm_ratio", "addr_norm_token_sort_ratio", "addr_norm_token_set_ratio",
    "addr_norm_partial_ratio", "addr_norm_jaro_winkler", "addr_norm_levenshtein",
    # b) TF-IDF Cosine (2)
    "tfidf_cosine_name", "tfidf_cosine_addr",
    # c) Numbers (6)
    "house_num_overlap", "house_num_conflict", "house_num_missing_s1",
    "house_num_missing_cand", "postal_equal", "postal_missing",
    # d) Lengths and empties (4)
    "name_len_ratio", "addr_len_ratio", "empty_addr_s1", "empty_addr_cand",
    # e) Source (1)
    "is_s3",
    # f) Blocking info (7)
    "score_a", "score_b", "score_c", "rank_a", "rank_b", "rank_c", "n_channels",
]

NEW_BASE_FEATURE_COLUMNS_V2: List[str] = [
    # g) Numbers from RAW address (5)
    "raw_num_jaccard", "raw_num_all_equal", "raw_num_conflicts",
    "raw_num_first_equal", "raw_num_first_dist",
    # h) Sound-alike name similarity & script flags (4)
    "skel_ratio", "skel_token_set_ratio", "is_non_latin_s1", "is_non_latin_cand",
    # i) Candidate pool rarity (6)
    "rarity_name_s1", "rarity_name_cand", "rarity_addr_s1", "rarity_addr_cand",
    "rarity_name_diff", "rarity_addr_diff",
]

BASE_FEATURE_COLUMNS_V2: List[str] = BASE_FEATURE_COLUMNS_V1 + NEW_BASE_FEATURE_COLUMNS_V2
BASE_FEATURE_COLUMNS: List[str] = BASE_FEATURE_COLUMNS_V2

CONTEXT_FEATURE_COLUMNS_V1: List[str] = [
    "p1_rank",
    "p1_rank_src",
    "s1_best_p1",
    "s1_n_cand_above_half",
    "p1_diff_best",
    "p1_diff_second",
]

GROUP_FEATURE_COLUMNS_V2: List[str] = [
    "cand_group_max_name_sim",
    "cand_group_max_addr_sim",
    "cand_group_other_src",
]

CONTEXT_FEATURE_COLUMNS_V2: List[str] = CONTEXT_FEATURE_COLUMNS_V1 + GROUP_FEATURE_COLUMNS_V2
CONTEXT_FEATURE_COLUMNS: List[str] = CONTEXT_FEATURE_COLUMNS_V2


# ---------------------------------------------------------------------------
# Data Loading Helper
# ---------------------------------------------------------------------------
def load_feature_split(split: str, countries: List[str], features_dir: Optional[Path] = None) -> pl.DataFrame:
    """Loads feature parquet files for the given split and countries, adding string entity IDs."""
    if features_dir is None:
        features_dir = DATA_FEATURES_V2_DIR if DATA_FEATURES_V2_DIR.exists() else DATA_FEATURES_DIR
    dfs: List[pl.DataFrame] = []
    for country in countries:
        fpath = features_dir / f"{split}_{country}.parquet"
        if not fpath.exists():
            raise FileNotFoundError(f"Feature file not found: {fpath}")
        df = pl.read_parquet(fpath)
        # Add country column for stratified analysis (NOT used as a model feature)
        df = df.with_columns(pl.lit(country).alias("country"))
        dfs.append(df)

    combined = pl.concat(dfs)
    # Add string ID representations for evaluation
    combined = combined.with_columns([
        ("S1-" + pl.col("s1_num").cast(pl.String)).alias("s1"),
        ("S" + pl.col("cand_src").cast(pl.String) + "-" + pl.col("cand_num").cast(pl.String)).alias("cid"),
    ])
    return combined


def load_split_s1_entities(split: str, countries: List[str]) -> pl.DataFrame:
    """Loads all S1 entities belonging to a split (both with candidates and zero candidates)."""
    dfs: List[pl.DataFrame] = []
    for c in countries:
        blk_path = DATA_BLOCKS_DIR / f"{split}_{c}.parquet"
        zero_path = DATA_BLOCKS_DIR / f"{split}_{c}_zero_candidates.parquet"
        s1_active = pl.read_parquet(blk_path, columns=["s1_num"]).unique()
        s1_zero = pl.read_parquet(zero_path, columns=["s1_num"]).unique() if zero_path.exists() else pl.DataFrame({"s1_num": []})
        s1_all = pl.concat([s1_active, s1_zero]).unique()
        s1_all = s1_all.with_columns([
            ("S1-" + pl.col("s1_num").cast(pl.String)).alias("s1"),
            pl.lit(c).alias("country"),
        ]).select(["s1", "country"])
        dfs.append(s1_all)
    return pl.concat(dfs)


# ---------------------------------------------------------------------------
# Section 3.1 Model Benchmark: LightGBM CPU vs XGBoost GPU
# ---------------------------------------------------------------------------
def run_model_benchmark(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    feature_cols: List[str],
) -> Tuple[str, Dict[str, Any]]:
    """Benchmarks LightGBM CPU vs XGBoost GPU on 1M sample of training data."""
    print("\n================ 3.1 MODEL BENCHMARK (1M SAMPLE) ================", flush=True)

    n_sample = min(1_000_000, len(train_df))
    print(f"Sampling {n_sample:,} pairs from train (seed 42)...", flush=True)
    sample_df = train_df.sample(n_sample, seed=42)

    X_train = sample_df.select(feature_cols).to_numpy()
    y_train = sample_df["label"].to_numpy().astype(np.int8)

    X_val = val_df.select(feature_cols).to_numpy()
    y_val = val_df["label"].to_numpy().astype(np.int8)

    results: Dict[str, Any] = {}

    # a) LightGBM CPU (num_threads=20)
    print("\n[Benchmark A] LightGBM CPU (num_threads=20)...", flush=True)
    t0 = time.perf_counter()
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        num_threads=20,
        random_state=42,
        n_jobs=20,
    )
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    time_lgb = time.perf_counter() - t0
    preds_lgb = lgb_model.predict_proba(X_val)[:, 1]
    auc_lgb = float(roc_auc_score(y_val, preds_lgb))
    print(f"LightGBM CPU finished in {time_lgb:.2f}s | Val AUC: {auc_lgb:.5f} | Best Iteration: {lgb_model.best_iteration_}", flush=True)
    results["lightgbm_cpu"] = {
        "time": time_lgb,
        "auc": auc_lgb,
        "best_iteration": lgb_model.best_iteration_,
        "gpu_used": False,
    }

    # b) XGBoost GPU: device="cuda", tree_method="hist"
    print("\n[Benchmark B] XGBoost GPU (device='cuda', tree_method='hist')...", flush=True)
    time_xgb = 9999.0
    auc_xgb = 0.0
    xgb_success = False
    try:
        t0 = time.perf_counter()
        xgb_model = xgb.XGBClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            device="cuda",
            random_state=42,
            early_stopping_rounds=50,
        )
        xgb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        time_xgb = time.perf_counter() - t0
        preds_xgb = xgb_model.predict_proba(X_val)[:, 1]
        auc_xgb = float(roc_auc_score(y_val, preds_xgb))
        xgb_success = True
        print(f"XGBoost GPU finished in {time_xgb:.2f}s | Val AUC: {auc_xgb:.5f} | Best Iteration: {xgb_model.best_iteration}", flush=True)
        results["xgboost_gpu"] = {
            "time": time_xgb,
            "auc": auc_xgb,
            "best_iteration": xgb_model.best_iteration,
            "gpu_used": True,
        }
    except Exception as e:
        print(f"XGBoost GPU benchmark failed or CUDA unavailable: {e}. Falling back to LightGBM CPU.", flush=True)
        results["xgboost_gpu"] = {
            "time": 9999.0,
            "auc": 0.0,
            "error": str(e),
            "gpu_used": False,
        }

    # Decision rule: Choose FASTER if val AUC is equal or better (within 0.001); otherwise LightGBM CPU
    chosen_lib = "lightgbm_cpu"
    if xgb_success:
        if (time_xgb < time_lgb) and (auc_xgb >= auc_lgb - 0.001):
            chosen_lib = "xgboost_gpu"
            print(f"\nDECISION: XGBoost GPU chosen (Time: {time_xgb:.2f}s vs {time_lgb:.2f}s, AUC: {auc_xgb:.5f} vs {auc_lgb:.5f}).", flush=True)
        else:
            print(f"\nDECISION: LightGBM CPU chosen (Time: {time_lgb:.2f}s vs {time_xgb:.2f}s, AUC: {auc_lgb:.5f} vs {auc_xgb:.5f}).", flush=True)
    else:
        print(f"\nDECISION: LightGBM CPU chosen (XGBoost GPU unavailable).", flush=True)

    del X_train, y_train, X_val, y_val, sample_df
    gc.collect()
    return chosen_lib, results


# ---------------------------------------------------------------------------
# Section 3.3 Within-S1 Context Feature Computation
# ---------------------------------------------------------------------------
def compute_within_s1_group_features(
    df: pl.DataFrame,
    cand_text_df: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """Computes candidate group features against S1's top-2 OTHER candidates by p1.

    Assumes df is already sorted by ['s1_num', p1] descending.
    Features:
      - cand_group_max_name_sim: max rapidfuzz token_set_ratio on name_core against top-2 other candidates
      - cand_group_max_addr_sim: max rapidfuzz token_set_ratio on addr_norm against top-2 other candidates
      - cand_group_other_src: 1.0 if ANY top-2 other candidate has cand_src != current cand_src, else 0.0
    """
    needs_drop = False
    if "cand_name_core" not in df.columns or "cand_addr_norm" not in df.columns:
        if cand_text_df is None:
            s2 = pl.read_parquet(PREP_FILES["train_s2"], columns=["entity_id", "name_core", "addr_norm"]).with_columns([
                pl.col("entity_id").str.slice(3).cast(pl.Int64).alias("cand_num"),
                pl.lit(2, dtype=pl.Int8).alias("cand_src"),
            ])
            s3 = pl.read_parquet(PREP_FILES["train_s3"], columns=["entity_id", "name_core", "addr_norm"]).with_columns([
                pl.col("entity_id").str.slice(3).cast(pl.Int64).alias("cand_num"),
                pl.lit(3, dtype=pl.Int8).alias("cand_src"),
            ])
            cand_text_df = pl.concat([s2, s3]).select([
                "cand_num", "cand_src",
                pl.col("name_core").alias("cand_name_core"),
                pl.col("addr_norm").alias("cand_addr_norm"),
            ])
        df = df.join(cand_text_df, on=["cand_num", "cand_src"], how="left")
        needs_drop = True

    s1_nums = df["s1_num"].to_numpy()
    cand_srcs = df["cand_src"].to_numpy()
    names = df["cand_name_core"].fill_null("").to_list()
    addrs = df["cand_addr_norm"].fill_null("").to_list()

    n = len(df)
    max_name_sim = np.zeros(n, dtype=np.float32)
    max_addr_sim = np.zeros(n, dtype=np.float32)
    other_src = np.zeros(n, dtype=np.float32)

    change_idx = np.flatnonzero(np.diff(s1_nums)) + 1
    starts = np.concatenate(([0], change_idx))
    ends = np.concatenate((change_idx, [n]))

    for s, e in zip(starts, ends):
        m = e - s
        if m <= 1:
            continue
        elif m == 2:
            sim_name = fuzz.token_set_ratio(names[s], names[s + 1]) / 100.0
            sim_addr = fuzz.token_set_ratio(addrs[s], addrs[s + 1]) / 100.0
            diff_src = 1.0 if cand_srcs[s] != cand_srcs[s + 1] else 0.0
            max_name_sim[s] = sim_name
            max_name_sim[s + 1] = sim_name
            max_addr_sim[s] = sim_addr
            max_addr_sim[s + 1] = sim_addr
            other_src[s] = diff_src
            other_src[s + 1] = diff_src
        else:
            n0, a0, src0 = names[s], addrs[s], cand_srcs[s]
            n1, a1, src1 = names[s + 1], addrs[s + 1], cand_srcs[s + 1]
            n2, a2, src2 = names[s + 2], addrs[s + 2], cand_srcs[s + 2]

            sim_n01 = fuzz.token_set_ratio(n0, n1) / 100.0
            sim_a01 = fuzz.token_set_ratio(a0, a1) / 100.0
            sim_n02 = fuzz.token_set_ratio(n0, n2) / 100.0
            sim_a02 = fuzz.token_set_ratio(a0, a2) / 100.0
            sim_n12 = fuzz.token_set_ratio(n1, n2) / 100.0
            sim_a12 = fuzz.token_set_ratio(a1, a2) / 100.0

            max_name_sim[s] = max(sim_n01, sim_n02)
            max_addr_sim[s] = max(sim_a01, sim_a02)
            other_src[s] = 1.0 if (src1 != src0 or src2 != src0) else 0.0

            max_name_sim[s + 1] = max(sim_n01, sim_n12)
            max_addr_sim[s + 1] = max(sim_a01, sim_a12)
            other_src[s + 1] = 1.0 if (src0 != src1 or src2 != src1) else 0.0

            for i in range(s + 2, e):
                ni, ai, srci = names[i], addrs[i], cand_srcs[i]
                sim_ni0 = fuzz.token_set_ratio(ni, n0) / 100.0
                sim_ni1 = fuzz.token_set_ratio(ni, n1) / 100.0
                sim_ai0 = fuzz.token_set_ratio(ai, a0) / 100.0
                sim_ai1 = fuzz.token_set_ratio(ai, a1) / 100.0
                max_name_sim[i] = max(sim_ni0, sim_ni1)
                max_addr_sim[i] = max(sim_ai0, sim_ai1)
                other_src[i] = 1.0 if (src0 != srci or src1 != srci) else 0.0

    df = df.with_columns([
        pl.Series("cand_group_max_name_sim", max_name_sim, dtype=pl.Float32),
        pl.Series("cand_group_max_addr_sim", max_addr_sim, dtype=pl.Float32),
        pl.Series("cand_group_other_src", other_src, dtype=pl.Float32),
    ])

    if needs_drop:
        df = df.drop(["cand_name_core", "cand_addr_norm"])
    return df


def add_within_s1_context_features(
    df: pl.DataFrame,
    p_col: str = "p1",
    version: str = "v1",
    cand_text_df: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """Computes within-S1 context features from pair model score p1.

    Zero cross-S1 leakage (strictly within each S1 entity's candidate list).
    v1: exactly 6 context features.
    v2: 6 context features + 3 candidate group features against top-2 other candidates.
    """
    # Sort descending by p_col within s1_num
    sorted_df = df.sort(["s1_num", p_col], descending=[False, True])

    ctx_df = sorted_df.with_columns([
        pl.int_range(1, pl.len() + 1).over("s1_num").cast(pl.Float32).alias("p1_rank"),
        pl.int_range(1, pl.len() + 1).over(["s1_num", "cand_src"]).cast(pl.Float32).alias("p1_rank_src"),
        pl.col(p_col).first().over("s1_num").cast(pl.Float32).alias("s1_best_p1"),
        (pl.col(p_col) > 0.5).cast(pl.Int32).sum().over("s1_num").cast(pl.Float32).alias("s1_n_cand_above_half"),
    ]).with_columns([
        (pl.col(p_col) - pl.col("s1_best_p1")).cast(pl.Float32).alias("p1_diff_best"),
        pl.when(pl.len().over("s1_num") >= 2)
        .then(pl.col(p_col) - pl.col(p_col).get(1).over("s1_num"))
        .otherwise(pl.col(p_col))
        .cast(pl.Float32)
        .alias("p1_diff_second"),
    ])

    if version == "v2":
        ctx_df = compute_within_s1_group_features(ctx_df, cand_text_df=cand_text_df)

    return ctx_df


# ---------------------------------------------------------------------------
# Model Trainer Wrapper (LightGBM CPU or XGBoost GPU)
# ---------------------------------------------------------------------------
class ModelTrainer:
    """Unified wrapper around the selected gradient boosting backend."""

    def __init__(self, lib_choice: str, seed: int = 42):
        self.lib_choice = lib_choice
        self.seed = seed

    def train_classifier(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        num_trees: int = 1500,
        lr: float = 0.05,
    ) -> Tuple[Any, float]:
        t0 = time.perf_counter()
        if self.lib_choice == "xgboost_gpu":
            clf = xgb.XGBClassifier(
                n_estimators=num_trees,
                learning_rate=lr,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                device="cuda",
                random_state=self.seed,
                early_stopping_rounds=50,
            )
            clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            clf = lgb.LGBMClassifier(
                n_estimators=num_trees,
                learning_rate=lr,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                num_threads=20,
                random_state=self.seed,
                n_jobs=20,
            )
            clf.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
            )
        elapsed = time.perf_counter() - t0
        return clf, elapsed

    def predict_proba(self, clf: Any, X: np.ndarray) -> np.ndarray:
        return clf.predict_proba(X)[:, 1].astype(np.float32)


# ---------------------------------------------------------------------------
# Section 3.4 Greedy 1-to-1 Bipartite Assignment
# ---------------------------------------------------------------------------
def apply_greedy_one_to_one(
    pred_df: pl.DataFrame,
    score_col: str,
    tau: float,
) -> Tuple[pl.DataFrame, int]:
    """Applies greedy 1-to-1 matching: each candidate cid goes to its highest scoring S1 (score >= tau).

    Ties: keep the higher score; if exactly equal, keep both. Returns (assigned_df, tie_count).
    """
    df_f = pred_df.select(["s1", "cid", "cand_num", "cand_src", score_col]).filter(pl.col(score_col) >= tau)
    if len(df_f) == 0:
        return pred_df.select(["s1", "cid"]).clear(), 0

    max_scores = df_f.group_by(["cand_num", "cand_src"]).agg(max_score=pl.col(score_col).max())
    joined = df_f.join(max_scores, on=["cand_num", "cand_src"])
    assigned = joined.filter(pl.col(score_col) == pl.col("max_score"))
    ties = len(assigned) - len(max_scores)

    return assigned.select(["s1", "cid"]), ties


# ---------------------------------------------------------------------------
# Section 3.4 Threshold Tau Sweep & Selection
# ---------------------------------------------------------------------------
def sweep_threshold_tau(
    val_preds: pl.DataFrame,
    score_col: str,
    s1_all_df: pl.DataFrame,
    gt_df: pl.DataFrame,
) -> Tuple[float, float, Dict[float, float], Dict[str, float]]:
    """Sweeps tau from 0.30 to 0.99 (step 0.01) on validation to find the optimal macro F0.5."""
    best_tau = 0.50
    best_macro = -1.0
    tau_curve: Dict[float, float] = {}

    taus = np.round(np.arange(0.30, 1.00, 0.01), 2)
    val_slim = val_preds.select(["s1", "cid", "cand_num", "cand_src", score_col])

    countries = s1_all_df["country"].unique().to_list()
    country_curves: Dict[str, Dict[float, float]] = {c: {} for c in countries}

    for tau in taus:
        assigned_df, _ = apply_greedy_one_to_one(val_slim, score_col=score_col, tau=tau)
        macro_score, country_scores = macro_f05_dataframe(assigned_df, gt_df, s1_all_df)
        tau_curve[float(tau)] = macro_score
        for c, score_c in country_scores.items():
            if c in country_curves:
                country_curves[c][float(tau)] = score_c
        if macro_score > best_macro:
            best_macro = macro_score
            best_tau = float(tau)

    # Country-specific best tau
    best_per_country: Dict[str, float] = {}
    for c, curve in country_curves.items():
        if curve:
            best_c_tau = max(curve.keys(), key=lambda t: curve[t])
            best_per_country[c] = float(best_c_tau)

    return best_tau, best_macro, tau_curve, best_per_country


# ---------------------------------------------------------------------------
# Full Evaluation Metrics Calculator
# ---------------------------------------------------------------------------
def compute_split_evaluation(
    pred_df: pl.DataFrame,
    s1_all_df: pl.DataFrame,
    gt_df: pl.DataFrame,
) -> Dict[str, Any]:
    """Computes macro F0.5, Precision, Recall, % Empty, and Avg Matches/Entity."""
    macro_f05, country_scores = macro_f05_dataframe(pred_df, gt_df, s1_all_df)

    p_uniq = pred_df.select(["s1", "cid"]).unique()
    # Filter ground truth pairs strictly to the entities in this split
    t_uniq = gt_df.join(s1_all_df.select(["s1"]), on="s1").select(["s1", "cid"]).unique()
    tp_pairs = len(p_uniq.join(t_uniq, on=["s1", "cid"]))
    prec = (tp_pairs / len(p_uniq)) if len(p_uniq) > 0 else 0.0
    rec = (tp_pairs / len(t_uniq)) if len(t_uniq) > 0 else 0.0

    # Entity counts
    total_s1 = len(s1_all_df)
    s1_pred_counts = p_uniq.group_by("s1").agg(pred_matches=pl.len())
    s1_eval = s1_all_df.join(s1_pred_counts, on="s1", how="left").with_columns(
        pl.col("pred_matches").fill_null(0)
    )

    empty_pred_pct = (len(s1_eval.filter(pl.col("pred_matches") == 0)) / total_s1) * 100
    avg_pred_matches = float(s1_eval["pred_matches"].mean())

    # Ground truth singletons and avg matches
    gt_counts = t_uniq.group_by("s1").agg(true_matches=pl.len())
    s1_gt = s1_all_df.join(gt_counts, on="s1", how="left").with_columns(
        pl.col("true_matches").fill_null(0)
    )
    true_empty_pct = (len(s1_gt.filter(pl.col("true_matches") == 0)) / total_s1) * 100
    avg_true_matches = float(s1_gt["true_matches"].mean())

    return {
        "macro_f05": macro_f05,
        "country_f05": country_scores,
        "precision": prec,
        "recall": rec,
        "empty_pred_pct": empty_pred_pct,
        "true_empty_pct": true_empty_pct,
        "avg_pred_matches": avg_pred_matches,
        "avg_true_matches": avg_true_matches,
        "pred_pairs": len(p_uniq),
        "true_pairs": len(t_uniq),
        "tp_pairs": tp_pairs,
    }


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Step 4 Training & Model Evaluation")
    parser.add_argument("--version", choices=["v1", "v2"], default="v2", help="Pipeline version (v1 or v2)")
    args = parser.parse_args()
    version = args.version

    features_dir = DATA_FEATURES_V2_DIR if version == "v2" else DATA_FEATURES_DIR
    models_dir = DATA_MODELS_V2_DIR if version == "v2" else DATA_MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    base_feature_cols = BASE_FEATURE_COLUMNS_V2 if version == "v2" else BASE_FEATURE_COLUMNS_V1
    context_feature_cols = CONTEXT_FEATURE_COLUMNS_V2 if version == "v2" else CONTEXT_FEATURE_COLUMNS_V1
    all_stage2_features = base_feature_cols + context_feature_cols

    countries = ["India", "US"]
    timings: Dict[str, float] = {}

    print(f"\n================ STEP 4 TRAINING & THRESHOLD OPTIMIZATION ({version.upper()}) ================", flush=True)
    print(f"Features Directory: {features_dir}")
    print(f"Models Directory:   {models_dir}")
    print(f"Base Features:      {len(base_feature_cols)} | Stage 2 Features: {len(all_stage2_features)}")

    # 1. Load Data
    t0_data = time.perf_counter()
    print("Loading feature datasets (train, val, holdout)...", flush=True)
    train_df = load_feature_split("train", countries, features_dir=features_dir)
    val_df = load_feature_split("val", countries, features_dir=features_dir)
    holdout_df = load_feature_split("holdout", countries, features_dir=features_dir)
    timings["data_load"] = time.perf_counter() - t0_data

    print(f"Loaded train pairs: {len(train_df):,} | val pairs: {len(val_df):,} | holdout pairs: {len(holdout_df):,}", flush=True)

    # Load Ground Truth
    gt_raw = pl.read_csv(RAW_FILES["train_gt"], separator="\t", quote_char=None)
    gt_rows: List[Dict[str, str]] = []
    for r in gt_raw.iter_rows(named=True):
        s1_id = r["source1_entity_id"]
        c_ids = [c for c in (r["matched_entity_ids"] or "").split(",") if c]
        for c in c_ids:
            gt_rows.append({"s1": s1_id, "cid": c})
    all_gt_pairs = pl.DataFrame(gt_rows)
    del gt_raw, gt_rows
    gc.collect()

    # Load ALL S1 entities for validation and holdout splits (including zero-candidate S1)
    val_s1_all = load_split_s1_entities("val", countries)
    holdout_s1_all = load_split_s1_entities("holdout", countries)
    print(f"Validation S1 entities: {len(val_s1_all):,} | Holdout S1 entities: {len(holdout_s1_all):,}", flush=True)

    # 2. Section 3.1: Model Benchmark (1M sample)
    t0_bench = time.perf_counter()
    chosen_lib, bench_results = run_model_benchmark(train_df, val_df, base_feature_cols)
    timings["benchmark"] = time.perf_counter() - t0_bench

    trainer = ModelTrainer(chosen_lib, seed=42)

    # 3. Section 3.2: Stage 1 Pair Model
    print(f"\n================ 3.2 STAGE 1 PAIR MODEL TRAINING ({len(base_feature_cols)} FEATURES) ================", flush=True)
    t0_s1 = time.perf_counter()

    # 2-Fold Out-of-Fold (OOF) on train (split by S1 entity)
    fold_0_mask = (train_df["s1_num"] % 2 == 0).to_numpy()
    fold_1_mask = ~fold_0_mask

    train_X = train_df.select(base_feature_cols).to_numpy()
    train_y = train_df["label"].to_numpy().astype(np.int8)

    val_X = val_df.select(base_feature_cols).to_numpy()
    val_y = val_df["label"].to_numpy().astype(np.int8)

    holdout_X = holdout_df.select(base_feature_cols).to_numpy()

    p1_oof = np.zeros(len(train_df), dtype=np.float32)

    print("Training Stage 1 Fold 0 (predicting Fold 1)...", flush=True)
    clf_f0, t_f0 = trainer.train_classifier(
        train_X[fold_0_mask], train_y[fold_0_mask],
        train_X[fold_1_mask], train_y[fold_1_mask],
    )
    p1_oof[fold_1_mask] = trainer.predict_proba(clf_f0, train_X[fold_1_mask])

    print("Training Stage 1 Fold 1 (predicting Fold 0)...", flush=True)
    clf_f1, t_f1 = trainer.train_classifier(
        train_X[fold_1_mask], train_y[fold_1_mask],
        train_X[fold_0_mask], train_y[fold_0_mask],
    )
    p1_oof[fold_0_mask] = trainer.predict_proba(clf_f1, train_X[fold_0_mask])

    del clf_f0, clf_f1
    gc.collect()

    oof_auc = float(roc_auc_score(train_y, p1_oof))
    print(f"Stage 1 Train 2-Fold OOF AUC: {oof_auc:.5f}", flush=True)

    # Full Stage 1 Model on ALL train
    print("Training Full Stage 1 Model on all train data...", flush=True)
    stage1_full_clf, t_s1_full = trainer.train_classifier(
        train_X, train_y, val_X, val_y
    )
    val_p1 = trainer.predict_proba(stage1_full_clf, val_X)
    holdout_p1 = trainer.predict_proba(stage1_full_clf, holdout_X)

    val_s1_auc = float(roc_auc_score(val_y, val_p1))
    print(f"Full Stage 1 Model Val AUC: {val_s1_auc:.5f}", flush=True)
    timings["stage1_training"] = time.perf_counter() - t0_s1

    # Attach p1 to DataFrames
    train_df = train_df.with_columns(pl.Series("p1", p1_oof, dtype=pl.Float32))
    val_df = val_df.with_columns(pl.Series("p1", val_p1, dtype=pl.Float32))
    holdout_df = holdout_df.with_columns(pl.Series("p1", holdout_p1, dtype=pl.Float32))

    # Free base arrays to keep memory well under 5 GB
    del train_X, train_y, val_X, holdout_X
    gc.collect()

    # Evaluate Stage 1 on Val
    best_tau_s1, best_macro_s1, tau_curve_s1, _ = sweep_threshold_tau(
        val_df, score_col="p1", s1_all_df=val_s1_all, gt_df=all_gt_pairs
    )
    print(f"Stage 1 Best Val Macro F0.5: {best_macro_s1:.5f} at tau={best_tau_s1:.2f}", flush=True)

    # 4. Section 3.3: Stage 2 Context Model
    print(f"\n================ 3.3 STAGE 2 CONTEXT MODEL TRAINING ({len(all_stage2_features)} FEATURES) ================", flush=True)
    t0_s2 = time.perf_counter()

    cand_text_df = None
    if version == "v2":
        t0_cands = time.perf_counter()
        print("Loading candidate text for Stage 2 group feature extraction...", flush=True)
        s2_c = pl.read_parquet(PREP_FILES["train_s2"], columns=["entity_id", "name_core", "addr_norm"]).with_columns([
            pl.col("entity_id").str.slice(3).cast(pl.Int64).alias("cand_num"),
            pl.lit(2, dtype=pl.Int8).alias("cand_src"),
        ])
        s3_c = pl.read_parquet(PREP_FILES["train_s3"], columns=["entity_id", "name_core", "addr_norm"]).with_columns([
            pl.col("entity_id").str.slice(3).cast(pl.Int64).alias("cand_num"),
            pl.lit(3, dtype=pl.Int8).alias("cand_src"),
        ])
        cand_text_df = pl.concat([s2_c, s3_c]).select([
            "cand_num", "cand_src",
            pl.col("name_core").alias("cand_name_core"),
            pl.col("addr_norm").alias("cand_addr_norm"),
        ])
        del s2_c, s3_c
        gc.collect()
        print(f"Loaded {len(cand_text_df):,} candidate strings in {time.perf_counter() - t0_cands:.2f}s", flush=True)

    print("Computing within-S1 context and group features on train...", flush=True)
    train_ctx_df = add_within_s1_context_features(train_df, p_col="p1", version=version, cand_text_df=cand_text_df)
    del train_df
    gc.collect()

    print("Computing within-S1 context and group features on val & holdout...", flush=True)
    val_ctx_df = add_within_s1_context_features(val_df, p_col="p1", version=version, cand_text_df=cand_text_df)
    holdout_ctx_df = add_within_s1_context_features(holdout_df, p_col="p1", version=version, cand_text_df=cand_text_df)

    del cand_text_df
    gc.collect()

    print(f"Total Stage 2 Features: {len(all_stage2_features)} ({len(base_feature_cols)} base + {len(context_feature_cols)} context/group)", flush=True)

    train_s2_X = train_ctx_df.select(all_stage2_features).to_numpy()
    train_s2_y = train_ctx_df["label"].to_numpy().astype(np.int8)
    del train_ctx_df
    gc.collect()

    val_s2_X = val_ctx_df.select(all_stage2_features).to_numpy()
    val_s2_y = val_ctx_df["label"].to_numpy().astype(np.int8)

    holdout_s2_X = holdout_ctx_df.select(all_stage2_features).to_numpy()

    print("Training Stage 2 Model on train with early stopping on val...", flush=True)
    stage2_clf, t_s2_train = trainer.train_classifier(
        train_s2_X, train_s2_y, val_s2_X, val_s2_y
    )
    del train_s2_X, train_s2_y
    gc.collect()

    val_p2 = trainer.predict_proba(stage2_clf, val_s2_X)
    holdout_p2 = trainer.predict_proba(stage2_clf, holdout_s2_X)
    val_s2_auc = float(roc_auc_score(val_s2_y, val_p2))
    del val_s2_X, val_s2_y, holdout_s2_X
    gc.collect()

    print(f"Stage 2 Model Val AUC: {val_s2_auc:.5f}", flush=True)
    timings["stage2_training"] = time.perf_counter() - t0_s2

    val_ctx_df = val_ctx_df.with_columns(pl.Series("p2", val_p2, dtype=pl.Float32))
    holdout_ctx_df = holdout_ctx_df.with_columns(pl.Series("p2", holdout_p2, dtype=pl.Float32))

    # Evaluate Stage 2 on Val
    best_tau_s2, best_macro_s2, tau_curve_s2, per_country_tau_s2 = sweep_threshold_tau(
        val_ctx_df, score_col="p2", s1_all_df=val_s1_all, gt_df=all_gt_pairs
    )
    print(f"Stage 2 Best Val Macro F0.5: {best_macro_s2:.5f} at tau={best_tau_s2:.2f}", flush=True)

    # Model Selection Rule: Keep Stage 2 only if it beats Stage 1 on val macro F0.5
    if best_macro_s2 > best_macro_s1:
        chosen_stage = "Stage 2"
        chosen_clf = stage2_clf
        chosen_features = all_stage2_features
        chosen_score_col = "p2"
        chosen_tau = best_tau_s2
        chosen_macro_val = best_macro_s2
        chosen_val_df = val_ctx_df
        chosen_holdout_df = holdout_ctx_df
        chosen_tau_curve = tau_curve_s2
        chosen_country_tau = per_country_tau_s2
        print(f"\nDECISION: Stage 2 Context Model ACCEPTED (Val Macro F0.5: {best_macro_s2:.5f} > Stage 1: {best_macro_s1:.5f}, Gain: +{best_macro_s2 - best_macro_s1:.5f})", flush=True)
    else:
        chosen_stage = "Stage 1"
        chosen_clf = stage1_full_clf
        chosen_features = base_feature_cols
        chosen_score_col = "p1"
        chosen_tau = best_tau_s1
        chosen_macro_val = best_macro_s1
        chosen_val_df = val_df
        chosen_holdout_df = holdout_df
        chosen_tau_curve = tau_curve_s1
        chosen_country_tau = per_country_tau_s2  # fallback
        print(f"\nDECISION: Stage 1 Pair Model KEPT (Stage 2 did not improve val macro F0.5).", flush=True)

    # 5. Section 3.5 Validation and Holdout Final Evaluation
    print("\n================ 3.5 EVALUATION & HOLDOUT CONFIRMATION ================", flush=True)
    t0_eval = time.perf_counter()

    val_assigned, val_ties = apply_greedy_one_to_one(chosen_val_df, score_col=chosen_score_col, tau=chosen_tau)
    val_metrics = compute_split_evaluation(val_assigned, val_s1_all, all_gt_pairs)
    print(f"Validation Overall Macro F0.5: {val_metrics['macro_f05']:.5f} (Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, Exact Ties: {val_ties})", flush=True)
    for c, sc in val_metrics["country_f05"].items():
        print(f"  Validation {c} Macro F0.5: {sc:.5f}", flush=True)

    # Confirm ONCE on Holdout (never tuned on holdout)
    holdout_assigned, holdout_ties = apply_greedy_one_to_one(chosen_holdout_df, score_col=chosen_score_col, tau=chosen_tau)
    holdout_metrics = compute_split_evaluation(holdout_assigned, holdout_s1_all, all_gt_pairs)
    print(f"\nHOLDOUT Overall Macro F0.5: {holdout_metrics['macro_f05']:.5f} (Precision: {holdout_metrics['precision']:.4f}, Recall: {holdout_metrics['recall']:.4f}, Exact Ties: {holdout_ties})", flush=True)
    for c, sc in holdout_metrics["country_f05"].items():
        print(f"  Holdout {c} Macro F0.5: {sc:.5f}", flush=True)

    timings["evaluation"] = time.perf_counter() - t0_eval

    # 6. Save Artifacts & Predictions
    t0_save = time.perf_counter()
    print(f"\nSaving models and predictions to {models_dir}...", flush=True)

    joblib.dump(stage1_full_clf, models_dir / "stage1_model.joblib")
    joblib.dump(stage2_clf, models_dir / "stage2_model.joblib")

    model_metadata = {
        "version": version,
        "chosen_stage": chosen_stage,
        "chosen_lib": chosen_lib,
        "features": chosen_features,
        "tau": chosen_tau,
        "per_country_tau": chosen_country_tau,
        "val_macro_f05": val_metrics["macro_f05"],
        "holdout_macro_f05": holdout_metrics["macro_f05"],
        "val_metrics": val_metrics,
        "holdout_metrics": holdout_metrics,
        "bench_results": bench_results,
    }
    with open(models_dir / "model_summary.json", "w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=2)

    with open(models_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(chosen_features, f, indent=2)

    # Save validation & holdout predictions for error analysis
    val_ctx_df.select(["s1_num", "cand_num", "cand_src", "p1", "p2", "label"]).write_parquet(
        models_dir / "val_predictions.parquet", compression="lz4"
    )
    holdout_ctx_df.select(["s1_num", "cand_num", "cand_src", "p1", "p2", "label"]).write_parquet(
        models_dir / "holdout_predictions.parquet", compression="lz4"
    )
    timings["save_artifacts"] = time.perf_counter() - t0_save

    # 7. Feature Importances (Top 25)
    if hasattr(chosen_clf, "feature_importances_"):
        raw_imp = chosen_clf.feature_importances_
    else:
        raw_imp = np.zeros(len(chosen_features))
    imp_order = np.argsort(raw_imp)[::-1]
    top_features = [(chosen_features[i], float(raw_imp[i])) for i in imp_order[:25]]

    # Importance of new v2 features specifically
    new_features_eval = NEW_BASE_FEATURE_COLUMNS_V2 + GROUP_FEATURE_COLUMNS_V2
    new_feat_importances = []
    for nf in new_features_eval:
        if nf in chosen_features:
            idx = chosen_features.index(nf)
            new_feat_importances.append((nf, float(raw_imp[idx])))
    new_feat_importances.sort(key=lambda x: x[1], reverse=True)

    # 8. GATE CHECK (Part 5)
    v1_holdout_macro = 0.94528
    v1_val_macro = 0.94359
    gate_threshold = v1_holdout_macro + 0.003  # 0.94828
    gate_passed = (holdout_metrics["macro_f05"] >= gate_threshold) and (val_metrics["macro_f05"] > v1_val_macro)

    print("\n================ PART 5: GATE CHECK ================", flush=True)
    print(f"v1 Baseline:       Val Macro F0.5 = {v1_val_macro:.5f} | Holdout Macro F0.5 = {v1_holdout_macro:.5f}")
    print(f"v2 Result:         Val Macro F0.5 = {val_metrics['macro_f05']:.5f} | Holdout Macro F0.5 = {holdout_metrics['macro_f05']:.5f}")
    print(f"Gate Requirements: Holdout >= {gate_threshold:.5f} (+0.003) AND Val > {v1_val_macro:.5f}")
    diff_val = val_metrics["macro_f05"] - v1_val_macro
    diff_holdout = holdout_metrics["macro_f05"] - v1_holdout_macro
    print(f"Val Delta:         {diff_val:+.5f}")
    print(f"Holdout Delta:     {diff_holdout:+.5f}")

    if gate_passed:
        gate_status_str = f"PASSED (Holdout {holdout_metrics['macro_f05']:.5f} >= {gate_threshold:.5f} and Val {val_metrics['macro_f05']:.5f} > {v1_val_macro:.5f})"
        print(f"\n>>> [GATE PASSED] v2 satisfies improvement threshold! Proceeding to Step 6 test scoring. <<<\n", flush=True)
    else:
        gate_status_str = f"FAILED (Holdout {holdout_metrics['macro_f05']:.5f} < {gate_threshold:.5f} or Val {val_metrics['macro_f05']:.5f} <= {v1_val_macro:.5f})"
        print(f"\n>>> [GATE FAILED] v2 does not meet required gain (+0.003). STOPPING. Fallback to v1. <<<\n", flush=True)

    # 9. Format Report Markdown
    lines: List[str] = []
    lines.append(f"\n## Step 7 (v2) Retraining & Evaluation Report")
    lines.append(f"**Date/Time**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Pipeline Version**: {version.upper()}")
    lines.append(f"**Selected Model Architecture**: {chosen_stage} ({chosen_lib})")
    lines.append(f"**Optimal Threshold (tau)**: **{chosen_tau:.2f}**")
    lines.append(f"**Gate Decision**: **{gate_status_str}**\n")

    lines.append("### 1. v1 vs v2 Performance Comparison")
    lines.append("| Metric | v1 (Baseline) | v2 (New Features) | Delta |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| **Val Macro F0.5** | {v1_val_macro:.5f} | **{val_metrics['macro_f05']:.5f}** | {diff_val:+.5f} |")
    lines.append(f"| Val India Macro F0.5 | 0.93883 | {val_metrics['country_f05'].get('India', 0.0):.5f} | {val_metrics['country_f05'].get('India', 0.0) - 0.93883:+.5f} |")
    lines.append(f"| Val US Macro F0.5 | 0.94835 | {val_metrics['country_f05'].get('US', 0.0):.5f} | {val_metrics['country_f05'].get('US', 0.0) - 0.94835:+.5f} |")
    lines.append(f"| Val Precision | 0.9774 | {val_metrics['precision']:.4f} | {val_metrics['precision'] - 0.9774:+.4f} |")
    lines.append(f"| Val Recall | 0.8976 | {val_metrics['recall']:.4f} | {val_metrics['recall'] - 0.8976:+.4f} |")
    lines.append(f"| **Holdout Macro F0.5** | {v1_holdout_macro:.5f} | **{holdout_metrics['macro_f05']:.5f}** | {diff_holdout:+.5f} |")
    lines.append(f"| Holdout India Macro F0.5 | 0.94056 | {holdout_metrics['country_f05'].get('India', 0.0):.5f} | {holdout_metrics['country_f05'].get('India', 0.0) - 0.94056:+.5f} |")
    lines.append(f"| Holdout US Macro F0.5 | 0.95000 | {holdout_metrics['country_f05'].get('US', 0.0):.5f} | {holdout_metrics['country_f05'].get('US', 0.0) - 0.95000:+.5f} |")
    lines.append(f"| Holdout Precision | 0.9782 | {holdout_metrics['precision']:.4f} | {holdout_metrics['precision'] - 0.9782:+.4f} |")
    lines.append(f"| Holdout Recall | 0.9004 | {holdout_metrics['recall']:.4f} | {holdout_metrics['recall'] - 0.9004:+.4f} |")
    lines.append("")

    lines.append("### 2. Feature Importances of New v2 Features")
    lines.append("| Feature Name | Feature Group | Importance |")
    lines.append("|---|---|---:|")
    for nf, imp in new_feat_importances:
        grp = "Raw Unit Numbers" if "raw_num" in nf else ("Sound-alike / Script" if "skel" in nf or "non_latin" in nf else ("Pool Rarity" if "rarity" in nf else "Within-S1 Group"))
        lines.append(f"| `{nf}` | {grp} | {imp:,.1f} |")
    lines.append("")

    lines.append("### 3. Top 15 Overall Feature Importances")
    lines.append("| Rank | Feature | Importance |")
    lines.append("|---|---|---:|")
    for idx, (f_name, imp) in enumerate(top_features[:15], 1):
        lines.append(f"| {idx} | `{f_name}` | {imp:,.1f} |")
    lines.append("")

    lines.append("### 4. Part Timings & Utilization")
    for part, dur in timings.items():
        lines.append(f"- **{part}**: {dur:.2f}s ({dur/60:.2f} min)")
    peak_rss = psutil.Process().memory_info().rss / (1024 ** 2)
    lines.append(f"- **Peak RSS**: **{peak_rss:.1f} MB**")
    lines.append("")

    report_md = "\n".join(lines) + "\n"
    append_to_run_log(report_md)
    print("\n" + report_md, flush=True)

    if not gate_passed:
        sys.exit(2)  # Non-zero exit code indicating gate stopped


if __name__ == "__main__":
    main()
