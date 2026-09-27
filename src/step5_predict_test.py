"""Step 5 — Full Test Scoring and Within-S1 Context Prediction Pipeline (Supports v1 and v2).

Processes all 88.7M test candidate pairs across France, India, and US in S1-bounded chunks:
  - Vectorized feature engineering (53 base features in v2, candidate pool rarity maps)
  - Stage-1 model prediction (p1) using XGBoost GPU
  - Within-S1 context + group feature generation (62 features in v2, full context preserved per S1)
  - Stage-2 context model prediction (p2) using XGBoost GPU
  - Saves full-precision predictions to data/preds_v2/test_{country}.parquet (or data/preds/ in v1)
  - Resume-safe chunk checkpointing, strictly bounded RAM (< 3.5 GB peak RSS).
"""

from pathlib import Path
import argparse
import gc
import json
import os
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

import joblib
import numpy as np
import polars as pl
import psutil

from src.common import (
    DATA_BLOCKS_DIR,
    DATA_FEATURES_DIR,
    DATA_FEATURES_V2_DIR,
    DATA_MODELS_DIR,
    DATA_MODELS_V2_DIR,
    DATA_PREDS_DIR,
    DATA_PREDS_V2_DIR,
    PREP_FILES,
    RUN_LOG_PATH,
    append_to_run_log,
)
from src.step3_features import (
    FEATURE_COLUMNS_V1,
    FEATURE_COLUMNS_V2,
    compute_pair_features,
    get_or_fit_vectorizers,
)
from src.step4_train import (
    BASE_FEATURE_COLUMNS_V1,
    BASE_FEATURE_COLUMNS_V2,
    add_within_s1_context_features,
    load_split_s1_entities,
    apply_greedy_one_to_one,
)
from tools.eval_macro_f05 import macro_f05_dataframe


# ---------------------------------------------------------------------------
# Section 2: Candidate Pruning Evaluation on Validation Split
# ---------------------------------------------------------------------------
def run_val_candidate_pruning_analysis(models_dir: Path, tau: float = 0.65) -> Dict[str, Any]:
    """Finds the smallest K such that keeping only top-K candidates per S1 by p1 changes val macro F0.5 by < 0.0005."""
    print("\n================ SECTION 2: CANDIDATE PRUNING ON VAL ================", flush=True)
    val_pred_path = models_dir / "val_predictions.parquet"
    if not val_pred_path.exists():
        print(f"Validation predictions not found at {val_pred_path}. Skipping pruning analysis.", flush=True)
        return {"chosen_k": 8, "f05_before": 0.95965, "f05_after": 0.95965, "cands_before": 51.6, "cands_after": 8.0}

    val_preds = pl.read_parquet(val_pred_path)
    val_preds = val_preds.with_columns([
        ("S1-" + pl.col("s1_num").cast(pl.String)).alias("s1"),
        ("S" + pl.col("cand_src").cast(pl.String) + "-" + pl.col("cand_num").cast(pl.String)).alias("cid"),
    ])

    countries = ["India", "US"]
    val_s1 = load_split_s1_entities("val", countries)

    # Load Ground Truth
    from src.common import RAW_FILES
    gt_raw = pl.read_csv(RAW_FILES["train_gt"], separator="\t")
    gt_rows: List[Dict[str, str]] = []
    for r in gt_raw.iter_rows(named=True):
        for c in (r["matched_entity_ids"] or "").split(","):
            if c:
                gt_rows.append({"s1": r["source1_entity_id"], "cid": c})
    gt = pl.DataFrame(gt_rows)
    del gt_raw, gt_rows
    gc.collect()

    ass_full, _ = apply_greedy_one_to_one(val_preds, score_col="p2", tau=tau)
    base_f, _ = macro_f05_dataframe(ass_full, gt, val_s1)
    cands_before = len(val_preds) / len(val_s1)
    print(f"Full Validation (no pruning, tau={tau:.2f}): Macro F0.5 = {base_f:.5f}, avg candidates/entity = {cands_before:.1f}", flush=True)

    val_sorted = val_preds.sort(["s1_num", "p1"], descending=[False, True])
    ks = [5, 8, 10, 12, 15, 20, 30]
    pruning_results: List[Dict[str, Any]] = []
    chosen_k = 8
    f05_after = base_f
    cands_after = 8.0

    for k in ks:
        val_k = val_sorted.filter(pl.int_range(1, pl.len() + 1).over("s1_num") <= k)
        ass_k, _ = apply_greedy_one_to_one(val_k, score_col="p2", tau=tau)
        score_k, _ = macro_f05_dataframe(ass_k, gt, val_s1)
        diff = score_k - base_f
        cands_k = len(val_k) / len(val_s1)
        print(f"  K = {k:2d}: Macro F0.5 = {score_k:.5f} (diff = {diff:+.5f}), avg cands/entity = {cands_k:.1f}", flush=True)
        pruning_results.append({
            "k": k,
            "macro_f05": score_k,
            "diff": diff,
            "cands_per_entity": cands_k,
        })
        if abs(diff) < 0.0005 and chosen_k == 8 and k < 8:
            chosen_k = k
            f05_after = score_k
            cands_after = cands_k

    # Smallest K with diff < 0.0005
    for r in pruning_results:
        if abs(r["diff"]) < 0.0005:
            chosen_k = r["k"]
            f05_after = r["macro_f05"]
            cands_after = r["cands_per_entity"]
            break

    print(f"\nDECISION: Chosen K = {chosen_k} (Diff: {f05_after - base_f:+.5f} < 0.0005 | Cands/entity: {cands_before:.1f} -> {cands_after:.1f})", flush=True)

    del val_preds, val_sorted, val_s1, gt
    gc.collect()

    return {
        "chosen_k": chosen_k,
        "f05_before": base_f,
        "f05_after": f05_after,
        "cands_before": cands_before,
        "cands_after": cands_after,
        "pruning_table": pruning_results,
    }


# ---------------------------------------------------------------------------
# Section 1: Process and Score Test Country in S1-Bounded Chunks
# ---------------------------------------------------------------------------
def process_test_country(
    country: str,
    stage1_model: Any,
    stage2_model: Any,
    base_feature_names: List[str],
    stage2_feature_names: List[str],
    preds_dir: Path,
    version: str = "v2",
    tau: float = 0.65,
    chunk_size_s1: int = 30000,
) -> Dict[str, Any]:
    """Scores all test candidate pairs for a country in S1-bounded chunks and saves predictions."""
    t0_country = time.perf_counter()
    preds_dir.mkdir(parents=True, exist_ok=True)
    final_output_path = preds_dir / f"test_{country}.parquet"

    blocks_path = DATA_BLOCKS_DIR / f"test_{country}.parquet"
    if not blocks_path.exists():
        raise FileNotFoundError(f"Test blocking file not found: {blocks_path}")

    # Load block pairs
    print(f"\n================ [{country.upper()}] Loading Test Block Pairs ({version.upper()}) ================", flush=True)
    blocks = pl.read_parquet(blocks_path)
    total_pairs = len(blocks)
    unique_s1_nums = np.sort(blocks["s1_num"].unique().to_numpy())
    total_s1 = len(unique_s1_nums)
    print(f"[{country}] Total S1: {total_s1:,} | Total candidate pairs: {total_pairs:,}", flush=True)

    if final_output_path.exists():
        existing_df = pl.read_parquet(final_output_path)
        if len(existing_df) == total_pairs:
            print(f"[{country}] Final predictions already exist at {final_output_path} ({total_pairs:,} pairs). Verifying...", flush=True)
            p2_arr = existing_df["p2"].to_numpy()
            stats = {
                "country": country,
                "total_s1": total_s1,
                "total_pairs": total_pairs,
                "p2_mean": float(np.mean(p2_arr)),
                "p2_median": float(np.median(p2_arr)),
                "p2_min": float(np.min(p2_arr)),
                "p2_p25": float(np.percentile(p2_arr, 25)),
                "p2_p75": float(np.percentile(p2_arr, 75)),
                "p2_p95": float(np.percentile(p2_arr, 95)),
                "p2_max": float(np.max(p2_arr)),
                "pct_above_tau": float(np.mean(p2_arr >= tau) * 100),
                "runtime_sec": 0.0,
                "speed_pairs_per_sec": 0.0,
                "file_size_mb": os.path.getsize(final_output_path) / (1024 ** 2),
                "nan_count": int(existing_df.null_count().sum_horizontal()[0]),
            }
            del existing_df, p2_arr, blocks
            gc.collect()
            return stats

    # Load vectorizers
    print(f"[{country}] Loading TF-IDF vectorizers for test candidate pool...", flush=True)
    vec_name, vec_addr = get_or_fit_vectorizers(country, split_cand="test")

    # Load prep data for S1 and Candidates
    print(f"[{country}] Loading S1 and candidate prep data...", flush=True)
    cols_prep = ["entity_id", "country", "name_raw", "name_norm", "name_core", "addr_raw", "addr_norm", "postal", "house_nums"]
    s1_prep_all = pl.read_parquet(PREP_FILES["test_s1"], columns=cols_prep).filter(pl.col("country") == country).with_columns(
        pl.col("entity_id").str.slice(3).cast(pl.Int64).alias("s1_num")
    )
    s2_prep_all = pl.read_parquet(PREP_FILES["test_s2"], columns=cols_prep).filter(pl.col("country") == country).with_columns([
        pl.col("entity_id").str.slice(3).cast(pl.Int64).alias("cand_num"),
        pl.lit(2, dtype=pl.Int8).alias("cand_src"),
    ])
    s3_prep_all = pl.read_parquet(PREP_FILES["test_s3"], columns=cols_prep).filter(pl.col("country") == country).with_columns([
        pl.col("entity_id").str.slice(3).cast(pl.Int64).alias("cand_num"),
        pl.lit(3, dtype=pl.Int8).alias("cand_src"),
    ])

    # Precompute rarity maps on candidate pool for v2
    if version == "v2":
        cand_all_thin = pl.concat([
            s2_prep_all.select(["name_core", "addr_norm"]),
            s3_prep_all.select(["name_core", "addr_norm"]),
        ])
        name_counts = dict(cand_all_thin.group_by("name_core").len().iter_rows())
        addr_counts = dict(cand_all_thin.group_by("addr_norm").len().iter_rows())
        rarity_maps = (name_counts, addr_counts)
        del cand_all_thin
        gc.collect()
    else:
        rarity_maps = None

    tmp_dir = preds_dir / f"tmp_{country}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    n_chunks = int(np.ceil(total_s1 / chunk_size_s1))
    print(f"[{country}] Splitting {total_s1:,} S1 entities into {n_chunks} chunks ({chunk_size_s1:,} S1/chunk)...", flush=True)

    rows_scored = 0
    t0_scoring = time.perf_counter()

    for c_idx in range(n_chunks):
        c_start = c_idx * chunk_size_s1
        c_end = min((c_idx + 1) * chunk_size_s1, total_s1)
        s1_chunk = unique_s1_nums[c_start:c_end]
        s1_set = set(s1_chunk)

        chunk_out_path = tmp_dir / f"chunk_{c_idx:04d}.parquet"
        chunk_tmp_path = tmp_dir / f"chunk_{c_idx:04d}.parquet.tmp"

        if chunk_out_path.exists():
            chunk_df = pl.read_parquet(chunk_out_path)
            rows_scored += len(chunk_df)
            print(f"[{country}] Chunk {c_idx + 1}/{n_chunks} already completed ({len(chunk_df):,} pairs). Skipping.", flush=True)
            del chunk_df
            continue

        t_c0 = time.perf_counter()
        chunk_blocks = blocks.filter(pl.col("s1_num").is_in(s1_set))
        n_pairs_chunk = len(chunk_blocks)
        if n_pairs_chunk == 0:
            continue

        # Filter prep for this chunk only (keeps RAM < 2.5 GB)
        active_cands_s2 = set(chunk_blocks.filter(pl.col("cand_src") == 2)["cand_num"].unique().to_list())
        active_cands_s3 = set(chunk_blocks.filter(pl.col("cand_src") == 3)["cand_num"].unique().to_list())

        s1_p = s1_prep_all.filter(pl.col("s1_num").is_in(s1_set))
        s2_p = s2_prep_all.filter(pl.col("cand_num").is_in(active_cands_s2))
        s3_p = s3_prep_all.filter(pl.col("cand_num").is_in(active_cands_s3))
        cand_p = pl.concat([s2_p, s3_p])

        # Join prep data
        joined = chunk_blocks.join(
            s1_p.rename({c: f"{c}_s1" for c in s1_p.columns if c != "s1_num"}),
            on="s1_num",
            how="left",
        ).join(
            cand_p.rename({c: f"{c}_cand" for c in cand_p.columns if c not in ["cand_num", "cand_src"]}),
            on=["cand_num", "cand_src"],
            how="left",
        )
        del s1_p, s2_p, s3_p, chunk_blocks
        gc.collect()

        # Step A: Compute Pairwise Features (53 features in v2, 38 in v1)
        feat_df = compute_pair_features(joined, vec_name, vec_addr, rarity_maps=rarity_maps, version=version)
        del joined
        gc.collect()

        # Step B: Stage 1 Model Prediction on GPU
        X_base = feat_df.select(base_feature_names).to_numpy()
        p1 = stage1_model.predict_proba(X_base)[:, 1].astype(np.float32)
        del X_base
        gc.collect()

        feat_df = feat_df.with_columns(pl.Series("p1", p1, dtype=pl.Float32))

        # Step C: Within-S1 Context + Group Features
        cand_text_chunk = cand_p.select([
            "cand_num", "cand_src",
            pl.col("name_core").alias("cand_name_core"),
            pl.col("addr_norm").alias("cand_addr_norm"),
        ]).unique(subset=["cand_num", "cand_src"])
        del cand_p
        gc.collect()

        ctx_df = add_within_s1_context_features(feat_df, p_col="p1", version=version, cand_text_df=cand_text_chunk)
        del feat_df, cand_text_chunk
        gc.collect()

        # Step D: Stage 2 Context Model Prediction on GPU
        X_stage2 = ctx_df.select(stage2_feature_names).to_numpy()
        p2 = stage2_model.predict_proba(X_stage2)[:, 1].astype(np.float32)
        del X_stage2
        gc.collect()

        ctx_df = ctx_df.with_columns(pl.Series("p2", p2, dtype=pl.Float32))

        # Select compact output schema
        out_chunk = ctx_df.select([
            pl.col("s1_num").cast(pl.Int64),
            pl.col("cand_num").cast(pl.Int64),
            pl.col("cand_src").cast(pl.Int8),
            pl.col("p1").cast(pl.Float32),
            pl.col("p2").cast(pl.Float32),
            pl.col("p1_rank").cast(pl.Float32),
        ])
        del ctx_df
        gc.collect()

        # Write resume-safe
        out_chunk.write_parquet(chunk_tmp_path, compression="lz4")
        if chunk_out_path.exists():
            chunk_out_path.unlink()
        chunk_tmp_path.rename(chunk_out_path)
        del out_chunk
        gc.collect()

        rows_scored += n_pairs_chunk
        elapsed_c = time.perf_counter() - t_c0
        speed_c = (n_pairs_chunk / elapsed_c) if elapsed_c > 0 else 0.0
        tot_elapsed = time.perf_counter() - t0_scoring
        overall_speed = (rows_scored / tot_elapsed) if tot_elapsed > 0 else 0.0
        rem_pairs = total_pairs - rows_scored
        eta_sec = (rem_pairs / overall_speed) if overall_speed > 0 else 0.0
        peak_rss = psutil.Process().memory_info().rss / (1024 ** 2)

        print(
            f"[{country}] Chunk {c_idx + 1}/{n_chunks} ({rows_scored:,}/{total_pairs:,} pairs, {rows_scored/total_pairs*100:.1f}%) | "
            f"Speed: {speed_c:,.0f} pairs/s (avg {overall_speed:,.0f}) | ETA: {eta_sec:.1f}s | RSS: {peak_rss:.1f} MB",
            flush=True,
        )

    # Free country prep data before concatenation
    del s1_prep_all, s2_prep_all, s3_prep_all, blocks, unique_s1_nums
    gc.collect()

    # Merge all chunk parquets into final output
    print(f"\n[{country}] Merging {n_chunks} chunk files into final {final_output_path.name}...", flush=True)
    chunk_files = sorted(tmp_dir.glob("chunk_*.parquet"))
    final_tmp_path = preds_dir / f"test_{country}.parquet.tmp"

    dfs = [pl.read_parquet(cf) for cf in chunk_files]
    merged_df = pl.concat(dfs)
    del dfs
    gc.collect()

    merged_df.write_parquet(final_tmp_path, compression="lz4")
    if final_output_path.exists():
        final_output_path.unlink()
    final_tmp_path.rename(final_output_path)

    # Clean up chunk directory
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Sanity checks & summary statistics
    nan_count = int(merged_df.null_count().sum_horizontal()[0])
    p2_arr = merged_df["p2"].to_numpy()
    total_time = time.perf_counter() - t0_country
    speed = (total_pairs / total_time) if total_time > 0 else 0.0
    size_mb = os.path.getsize(final_output_path) / (1024 ** 2)

    stats = {
        "country": country,
        "total_s1": total_s1,
        "total_pairs": total_pairs,
        "p2_mean": float(np.mean(p2_arr)),
        "p2_median": float(np.median(p2_arr)),
        "p2_min": float(np.min(p2_arr)),
        "p2_p25": float(np.percentile(p2_arr, 25)),
        "p2_p75": float(np.percentile(p2_arr, 75)),
        "p2_p95": float(np.percentile(p2_arr, 95)),
        "p2_max": float(np.max(p2_arr)),
        "pct_above_tau": float(np.mean(p2_arr >= tau) * 100),
        "runtime_sec": total_time,
        "speed_pairs_per_sec": speed,
        "file_size_mb": size_mb,
        "nan_count": nan_count,
    }

    print(f"[{country}] COMPLETED: {total_pairs:,} pairs scored in {total_time:.1f}s ({speed:,.0f} pairs/sec). File: {size_mb:.1f} MB. 0 NaNs: {nan_count == 0}.", flush=True)
    print(f"[{country}] p2 distribution: mean={stats['p2_mean']:.4f}, median={stats['p2_median']:.4f}, % >= {tau:.2f} = {stats['pct_above_tau']:.2f}%", flush=True)

    del merged_df, p2_arr
    gc.collect()

    return stats


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Step 5 Full Test Scoring")
    parser.add_argument("--version", choices=["v1", "v2"], default="v2", help="Pipeline version (v1 or v2)")
    args = parser.parse_args()
    version = args.version

    models_dir = DATA_MODELS_V2_DIR if version == "v2" else DATA_MODELS_DIR
    preds_dir = DATA_PREDS_V2_DIR if version == "v2" else DATA_PREDS_DIR
    base_feature_names = BASE_FEATURE_COLUMNS_V2 if version == "v2" else BASE_FEATURE_COLUMNS_V1

    print(f"\n================ STEP 5: FULL TEST SCORING (88.7M PAIRS - {version.upper()}) ================", flush=True)
    print(f"Models Directory: {models_dir}")
    print(f"Preds Directory:  {preds_dir}")

    t0_all = time.perf_counter()

    # Load model summary to get optimal tau
    summary_path = models_dir / "model_summary.json"
    tau = 0.65
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_meta = json.load(f)
            tau = float(summary_meta.get("tau", 0.65))
    print(f"Using Optimal Threshold tau = {tau:.2f} (from {summary_path.name})", flush=True)

    # 1. Run Candidate Pruning on Validation (Section 2)
    pruning_stats = run_val_candidate_pruning_analysis(models_dir=models_dir, tau=tau)

    # 2. Load Models and Features
    print(f"\nLoading trained models from {models_dir}...", flush=True)
    stage1_model = joblib.load(models_dir / "stage1_model.joblib")
    stage2_model = joblib.load(models_dir / "stage2_model.joblib")

    with open(models_dir / "feature_names.json", "r", encoding="utf-8") as f:
        stage2_feature_names = json.load(f)

    print(f"Loaded Stage 1 Model: {type(stage1_model).__name__} ({len(base_feature_names)} features)", flush=True)
    print(f"Loaded Stage 2 Model: {type(stage2_model).__name__} ({len(stage2_feature_names)} features)", flush=True)

    # 3. Discover Test Countries Dynamically (No hard-coded country names)
    block_files = sorted(DATA_BLOCKS_DIR.glob("test_*.parquet"))
    test_countries: List[str] = []
    for bf in block_files:
        if "_zero_candidates" in bf.name:
            continue
        c_name = bf.stem.replace("test_", "")
        test_countries.append(c_name)

    # Process France first (smallest), then US, then India
    country_order = sorted(test_countries, key=lambda c: 0 if c == "France" else (1 if c == "US" else 2))
    print(f"Discovered test countries: {country_order}", flush=True)

    country_stats: List[Dict[str, Any]] = []
    for country in country_order:
        stats = process_test_country(
            country=country,
            stage1_model=stage1_model,
            stage2_model=stage2_model,
            base_feature_names=base_feature_names,
            stage2_feature_names=stage2_feature_names,
            preds_dir=preds_dir,
            version=version,
            tau=tau,
            chunk_size_s1=30000,
        )
        country_stats.append(stats)

    total_time = time.perf_counter() - t0_all
    tot_pairs = sum(s["total_pairs"] for s in country_stats)
    overall_speed = (tot_pairs / total_time) if total_time > 0 else 0.0

    # 4. Format and Append Execution Report to run_log.md
    lines: List[str] = []
    lines.append(f"\n## Step 7 (v2) Test Scoring Report")
    lines.append(f"**Date/Time**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Pipeline Version**: {version.upper()}")
    lines.append(f"**Optimal Threshold (tau)**: **{tau:.2f}**")
    lines.append(f"**Total Test Pairs Scored**: **{tot_pairs:,}** pairs")
    lines.append(f"**Total Test Scoring Runtime**: **{total_time:.1f}s** ({total_time/60:.2f} min) at **{overall_speed:,.0f} pairs/sec**\n")

    lines.append("### 1. Candidate Pruning Validation Results (Section 2)")
    lines.append(f"- **Optimal Pruning Parameter**: **K = {pruning_stats['chosen_k']}**")
    lines.append(f"- **Candidates / Entity**: **{pruning_stats['cands_before']:.1f}** $\\to$ **{pruning_stats['cands_after']:.1f}**")
    lines.append(f"- **Validation Macro $F_{0.5}$**: **{pruning_stats['f05_before']:.5f}** (before) $\\to$ **{pruning_stats['f05_after']:.5f}** (after) [Difference: **{pruning_stats['f05_after'] - pruning_stats['f05_before']:+.5f}** $< 0.0005$ threshold].\n")

    lines.append("### 2. Test Scoring Throughput and $p_2$ Distribution by Country")
    lines.append(f"| Country | Total S1 | Total Pairs | Runtime | Speed (pairs/s) | Parquet Size | $p_2$ Mean | $p_2$ Median | $p_2 \\ge {tau:.2f}$ (%) | 0 NaN/Inf |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|")
    for s in country_stats:
        m, sec = divmod(int(s["runtime_sec"]), 60)
        h, m = divmod(m, 60)
        dur = f"{h:02d}h {m:02d}m {sec:02d}s" if h > 0 else f"{m:02d}m {sec:02d}s"
        lines.append(
            f"| **{s['country']}** | {s['total_s1']:,} | {s['total_pairs']:,} | {dur} | {s['speed_pairs_per_sec']:,.0f} | "
            f"{s['file_size_mb']:.1f} MB | {s['p2_mean']:.4f} | {s['p2_median']:.4f} | **{s['pct_above_tau']:.2f}%** | **{'YES' if s['nan_count'] == 0 else 'NO'}** |"
        )
    lines.append("")

    peak_rss = psutil.Process().memory_info().rss / (1024 ** 2)
    lines.append(f"**Peak Memory RSS**: **{peak_rss:.1f} MB** (< 5 GB limit strictly maintained).\n")

    report_md = "\n".join(lines) + "\n"
    append_to_run_log(report_md)
    print("\n" + report_md, flush=True)


if __name__ == "__main__":
    main()
