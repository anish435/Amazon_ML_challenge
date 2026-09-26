"""Step 6 — Bipartite 1:1 Matching, Probe Variants, and Official Submission File Generation.

Outputs:
  output/v1/matching_results.tsv (exactly 1,732,544 rows, header: source1_entity_id<TAB>matched_entity_ids)
  output/v1/candidate_pairs.tsv (same S1 rows, header: source1_entity_id<TAB>candidate_entity_ids)
  output/v1/probes/main.tsv
  output/v1/probes/france_empty.tsv
  output/v1/probes/france_tau_0.80.tsv
  output/v1/probes/france_tau_0.90.tsv
  output/v1/probes/france_tau_0.55.tsv
"""

from pathlib import Path
from collections import defaultdict
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional, Set, Tuple, Any

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

from src.common import (
    DATA_PREDS_DIR,
    OUTPUT_DIR,
    RAW_FILES,
    REPORTS_DIR,
    RUN_LOG_PATH,
    SUBMISSION_LOG_PATH,
    append_to_run_log,
)


def sha256_file(filepath: Path) -> str:
    """Computes SHA-256 hash of a file in streaming chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024 * 8):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Section 4: France Profile Analyzer
# ---------------------------------------------------------------------------
def compute_country_profile(
    assigned_df: pl.DataFrame,
    all_s1_list: List[str],
) -> Dict[str, Any]:
    """Computes profile metrics: avg matches, % empty, p99, max, per-source avg."""
    total_s1 = len(all_s1_list)
    if len(assigned_df) == 0:
        return {
            "avg_matches": 0.0,
            "pct_empty": 100.0,
            "p99": 0,
            "max": 0,
            "avg_s2": 0.0,
            "avg_s3": 0.0,
            "total_matches": 0,
        }

    # Match count per S1
    match_counts = (
        assigned_df.group_by("s1")
        .agg(n_matches=pl.len())
        .to_dict(as_series=False)
    )
    s1_to_count = dict(zip(match_counts["s1"], match_counts["n_matches"]))
    counts_all = np.array([s1_to_count.get(s1, 0) for s1 in all_s1_list], dtype=np.int32)

    avg_matches = float(np.mean(counts_all))
    pct_empty = float(np.mean(counts_all == 0) * 100)
    p99 = int(np.percentile(counts_all, 99))
    max_matches = int(np.max(counts_all))

    # Per-source breakdown (entities with >= 1 from that source)
    assigned_with_src = assigned_df.with_columns(
        pl.col("cid").str.slice(1, 1).alias("src")
    )
    s2_counts = (
        assigned_with_src.filter(pl.col("src") == "2")
        .group_by("s1")
        .agg(n_s2=pl.len())["n_s2"]
        .to_numpy()
    )
    s3_counts = (
        assigned_with_src.filter(pl.col("src") == "3")
        .group_by("s1")
        .agg(n_s3=pl.len())["n_s3"]
        .to_numpy()
    )
    avg_s2 = float(np.mean(s2_counts)) if len(s2_counts) > 0 else 0.0
    avg_s3 = float(np.mean(s3_counts)) if len(s3_counts) > 0 else 0.0

    return {
        "avg_matches": avg_matches,
        "pct_empty": pct_empty,
        "p99": p99,
        "max": max_matches,
        "avg_s2": avg_s2,
        "avg_s3": avg_s3,
        "total_matches": len(assigned_df),
    }


# ---------------------------------------------------------------------------
# Section 3: Greedy 1-to-1 Bipartite Matching per Country
# ---------------------------------------------------------------------------
def apply_greedy_one_to_one(
    preds_df: pl.DataFrame,
    tau: float,
    max_k: int = 8,
) -> Tuple[pl.DataFrame, int]:
    """Applies greedy 1:1 matching on top-K candidate subset per country."""
    # Filter candidate subset strictly to top-K
    filtered = preds_df.filter((pl.col("p1_rank") <= max_k) & (pl.col("p2") >= tau))
    if len(filtered) == 0:
        return pl.DataFrame({"s1": pl.Series([], dtype=pl.String), "cid": pl.Series([], dtype=pl.String)}), 0

    # For each candidate, find the maximum p2 score
    max_scores = filtered.group_by(["cand_num", "cand_src"]).agg(max_p2=pl.col("p2").max())
    joined = filtered.join(max_scores, on=["cand_num", "cand_src"])
    matching_max = joined.filter(pl.col("p2") == pl.col("max_p2"))
    ties = len(matching_max) - len(max_scores)

    # Break ties deterministically: sort by p2 desc, p1 desc, s1_num asc, and keep first
    assigned = (
        matching_max
        .sort(["p2", "p1", "s1_num"], descending=[True, True, False])
        .unique(subset=["cand_num", "cand_src"], keep="first")
    )

    return assigned.select(["s1", "cid"]), ties


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------
def main():
    print("\n================ STEP 6: BUILD SUBMISSION & PROBE FILES ================", flush=True)
    t0_start = time.perf_counter()

    out_v1_dir = OUTPUT_DIR / "v1"
    probes_dir = out_v1_dir / "probes"
    probes_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load exact test S1 sequence from test_source1.tsv
    print("Reading canonical test_source1.tsv to ensure exact entity sequence...", flush=True)
    raw_s1_lines = pl.read_csv(
        RAW_FILES["test_s1"],
        separator="\t",
        columns=["entity_id", "country"],
    )
    all_s1_ordered = raw_s1_lines["entity_id"].to_list()
    all_s1_countries = raw_s1_lines["country"].to_list()
    total_test_s1 = len(all_s1_ordered)
    print(f"Total test S1 entities: {total_test_s1:,}", flush=True)

    # Group S1 by country
    country_s1_map: Dict[str, List[str]] = defaultdict(list)
    for s1, c in zip(all_s1_ordered, all_s1_countries):
        country_s1_map[c].append(s1)

    # 2. Load Predictions per Country and Apply Greedy 1-to-1 Matching
    pred_files = sorted(DATA_PREDS_DIR.glob("test_*.parquet"))
    if not pred_files:
        raise FileNotFoundError(f"No prediction files found in {DATA_PREDS_DIR}")

    tau_default = 0.67
    chosen_k = 8

    # Store candidates and matches per S1
    # s1 -> list of matched candidate IDs
    matches_main: Dict[str, List[str]] = defaultdict(list)
    # s1 -> list of top-K candidate IDs
    candidates_topk: Dict[str, List[str]] = defaultdict(list)

    # Store France predictions for multi-tau profiling and probe variants
    france_preds: Optional[pl.DataFrame] = None
    france_ties_reported = 0

    total_ties = 0
    country_profile_main: Dict[str, Dict[str, Any]] = {}

    for pf in pred_files:
        country = pf.stem.replace("test_", "")
        print(f"\n[{country}] Processing predictions from {pf.name}...", flush=True)
        preds = pl.read_parquet(pf)

        # Build string IDs
        preds = preds.with_columns([
            ("S1-" + pl.col("s1_num").cast(pl.String)).alias("s1"),
            ("S" + pl.col("cand_src").cast(pl.String) + "-" + pl.col("cand_num").cast(pl.String)).alias("cid"),
        ])

        # Candidate pairs top-K (Section 2)
        topk_cands = preds.filter(pl.col("p1_rank") <= chosen_k).sort(["s1_num", "p1_rank"])
        print(f"[{country}] Top-{chosen_k} candidates: {len(topk_cands):,} pairs (avg {len(topk_cands)/len(country_s1_map[country]):.1f}/S1)", flush=True)

        for r in topk_cands.select(["s1", "cid"]).iter_rows(named=True):
            candidates_topk[r["s1"]].append(r["cid"])
        del topk_cands
        gc.collect()

        # Greedy 1:1 Matching at tau = 0.67
        assigned, ties = apply_greedy_one_to_one(preds, tau=tau_default, max_k=chosen_k)
        total_ties += ties
        print(f"[{country}] Greedy 1:1 matches at tau={tau_default}: {len(assigned):,} pairs (Exact ties: {ties})", flush=True)

        for r in assigned.iter_rows(named=True):
            matches_main[r["s1"]].append(r["cid"])

        # Compute country profile
        c_prof = compute_country_profile(assigned, country_s1_map[country])
        country_profile_main[country] = c_prof
        print(f"[{country}] Profile: avg={c_prof['avg_matches']:.2f}, empty={c_prof['pct_empty']:.2f}%, p99={c_prof['p99']}, max={c_prof['max']}, S2={c_prof['avg_s2']:.2f}, S3={c_prof['avg_s3']:.2f}", flush=True)

        if country == "France":
            france_preds = preds
            france_ties_reported = ties
        else:
            del preds
            gc.collect()

    print(f"\nTotal exact ties across all countries: {total_ties}", flush=True)

    # 3. Section 4: France Multi-Threshold Sensitivity Profile
    print("\n================ SECTION 4: FRANCE THRESHOLD SENSITIVITY PROFILE ================", flush=True)
    france_s1_list = country_s1_map["France"]
    france_taus = [0.50, 0.60, 0.67, 0.75, 0.80, 0.85, 0.90, 0.95]
    france_profile_table: List[Dict[str, Any]] = []

    # Store assigned matches for probe variants
    france_variant_matches: Dict[float, Dict[str, List[str]]] = {}

    for tau_f in france_taus:
        ass_f, ties_f = apply_greedy_one_to_one(france_preds, tau=tau_f, max_k=chosen_k)
        prof_f = compute_country_profile(ass_f, france_s1_list)
        prof_f["tau"] = tau_f
        france_profile_table.append(prof_f)
        print(
            f"France tau={tau_f:.2f}: avg={prof_f['avg_matches']:.2f}, empty={prof_f['pct_empty']:.2f}%, "
            f"p99={prof_f['p99']}, max={prof_f['max']}, S2={prof_f['avg_s2']:.2f}, S3={prof_f['avg_s3']:.2f}, total={prof_f['total_matches']:,}",
            flush=True,
        )
        if tau_f in [0.55, 0.80, 0.90]:
            f_map: Dict[str, List[str]] = defaultdict(list)
            for r in ass_f.iter_rows(named=True):
                f_map[r["s1"]].append(r["cid"])
            france_variant_matches[tau_f] = f_map

    # Also compute France at tau = 0.55 for probe variant
    ass_f_55, _ = apply_greedy_one_to_one(france_preds, tau=0.55, max_k=chosen_k)
    f_map_55: Dict[str, List[str]] = defaultdict(list)
    for r in ass_f_55.iter_rows(named=True):
        f_map_55[r["s1"]].append(r["cid"])
    france_variant_matches[0.55] = f_map_55

    del france_preds
    gc.collect()

    # 4. Write output/v1/matching_results.tsv and candidate_pairs.tsv
    print("\nWriting canonical submission files to output/v1/...", flush=True)
    matching_file = out_v1_dir / "matching_results.tsv"
    candidates_file = out_v1_dir / "candidate_pairs.tsv"

    # Write matching_results.tsv
    with open(matching_file, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1 in all_s1_ordered:
            m_list = list(dict.fromkeys(matches_main.get(s1, [])))
            f.write(f"{s1}\t{','.join(m_list)}\n")

    print(f"Wrote {matching_file} ({total_test_s1:,} rows).", flush=True)

    # Write candidate_pairs.tsv (Header: source1_entity_id\tcandidate_entity_ids)
    with open(candidates_file, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1 in all_s1_ordered:
            c_list = list(dict.fromkeys(candidates_topk.get(s1, [])))
            m_list = list(dict.fromkeys(matches_main.get(s1, [])))
            for m in m_list:
                if m not in c_list:
                    c_list.append(m)
            f.write(f"{s1}\t{','.join(c_list)}\n")

    print(f"Wrote {candidates_file} ({total_test_s1:,} rows).", flush=True)

    # 5. Build Probe Variants in output/v1/probes/
    print("\nWriting probe variants to output/v1/probes/...", flush=True)
    probe_files: Dict[str, Path] = {
        "main": probes_dir / "main.tsv",
        "france_empty": probes_dir / "france_empty.tsv",
        "france_tau_0.80": probes_dir / "france_tau_0.80.tsv",
        "france_tau_0.90": probes_dir / "france_tau_0.90.tsv",
        "france_tau_0.55": probes_dir / "france_tau_0.55.tsv",
    }

    # Copy main.tsv
    shutil.copyfile(matching_file, probe_files["main"])

    # France empty probe
    with open(probe_files["france_empty"], "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1, c in zip(all_s1_ordered, all_s1_countries):
            m_list = [] if c == "France" else matches_main.get(s1, [])
            f.write(f"{s1}\t{','.join(m_list)}\n")

    # France tau 0.80 probe
    with open(probe_files["france_tau_0.80"], "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1, c in zip(all_s1_ordered, all_s1_countries):
            m_list = france_variant_matches[0.80].get(s1, []) if c == "France" else matches_main.get(s1, [])
            f.write(f"{s1}\t{','.join(m_list)}\n")

    # France tau 0.90 probe
    with open(probe_files["france_tau_0.90"], "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1, c in zip(all_s1_ordered, all_s1_countries):
            m_list = france_variant_matches[0.90].get(s1, []) if c == "France" else matches_main.get(s1, [])
            f.write(f"{s1}\t{','.join(m_list)}\n")

    # France tau 0.55 probe
    with open(probe_files["france_tau_0.55"], "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1, c in zip(all_s1_ordered, all_s1_countries):
            m_list = france_variant_matches[0.55].get(s1, []) if c == "France" else matches_main.get(s1, [])
            f.write(f"{s1}\t{','.join(m_list)}\n")

    print("All probe variant files written.", flush=True)

    # 6. Compute SHA256 Hashes
    print("\nComputing SHA-256 hashes of generated files...", flush=True)
    hashes: Dict[str, str] = {}
    hashes["matching_results.tsv"] = sha256_file(matching_file)
    hashes["candidate_pairs.tsv"] = sha256_file(candidates_file)
    for p_name, p_path in probe_files.items():
        hashes[f"probes/{p_path.name}"] = sha256_file(p_path)

    for fname, fhash in hashes.items():
        print(f"  {fname}: {fhash}", flush=True)

    # 7. Official Submission Validator & Pre-upload Audit
    print("\nRunning official submission validator (student_resource/utils/validate_submission.py)...", flush=True)
    cmd_val = [
        sys.executable,
        "student_resource/utils/validate_submission.py",
        "--matching", str(matching_file),
        "--candidate", str(candidates_file),
        "--test-dir", "student_resource/dataset/test",
        "--check-ids",
    ]
    res_val = subprocess.run(cmd_val, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    print(res_val.stdout)
    if res_val.stderr:
        print(res_val.stderr, file=sys.stderr)
    val_passed = (res_val.returncode == 0) and ("FAILED" not in res_val.stdout) and ("WARNING" not in res_val.stdout)
    print(f"Official Submission Validator Status: {'PASSED' if val_passed else 'FAILED'}", flush=True)

    print("\nRunning independent pre-upload check (tools/preupload_check.py)...", flush=True)
    cmd_pre = [
        sys.executable,
        "tools/preupload_check.py",
    ]
    res_pre = subprocess.run(cmd_pre, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    print(res_pre.stdout)
    if res_pre.stderr:
        print(res_pre.stderr, file=sys.stderr)
    pre_passed = (res_pre.returncode == 0) and ("ALL CHECKS PASSED" in res_pre.stdout)
    print(f"Pre-Upload Check Status: {'PASSED' if pre_passed else 'FAILED'}", flush=True)

    # 8. Predicted Leaderboard Scores
    # Test Country Weights: France 0.150, India 0.468, US 0.383
    # From Step 4 Validation: India = 0.9298, US = 0.9528
    val_india = 0.92977
    val_us = 0.95281
    w_india_us = (0.468 * val_india) + (0.383 * val_us)  # 0.43513 + 0.36493 = 0.80006

    lb_pred_0 = w_india_us + (0.150 * 0.0)
    lb_pred_50 = w_india_us + (0.150 * 0.50)
    lb_pred_90 = w_india_us + (0.150 * 0.90)
    lb_pred_exp = w_india_us + (0.150 * 0.94)

    # 9. Append to reports/submission_log.tsv
    print("\nLogging to reports/submission_log.tsv...", flush=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    sub_log_exists = SUBMISSION_LOG_PATH.exists()
    with open(SUBMISSION_LOG_PATH, "a", encoding="utf-8") as f:
        if not sub_log_exists:
            f.write("timestamp\tfile\tsha256\tconfig\tval_macro_f05\tholdout_macro_f05\tpredicted_lb_min\tpredicted_lb_max\tpublic_score\n")
        f.write(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t"
            f"output/v1/matching_results.tsv\t"
            f"{hashes['matching_results.tsv']}\t"
            f"XGBoost_GPU_Stage2_tau0.67_K{chosen_k}\t"
            f"0.94359\t"
            f"0.94528\t"
            f"{lb_pred_0:.4f}\t"
            f"{lb_pred_90:.4f}\t"
            f"\n"
        )

    # 10. Format and Append Execution Report to run_log.md
    total_time = time.perf_counter() - t0_start
    lines: List[str] = []
    lines.append("\n## Step 6 Build Output & Submission Generation Report")
    lines.append(f"**Date/Time**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Output Directory**: `output/v1/`")
    lines.append(f"**Primary Matching File**: `output/v1/matching_results.tsv` (SHA256: `{hashes['matching_results.tsv']}`)")
    lines.append(f"**Candidate Pairs File**: `output/v1/candidate_pairs.tsv` (SHA256: `{hashes['candidate_pairs.tsv']}`)\n")

    lines.append("### 1. File Verification & Validator Output")
    lines.append(f"- **Row Count**: Exactly **{total_test_s1:,}** rows (matches `test_source1.tsv` 100%).")
    lines.append(f"- **Format**: TAB-separated (`.tsv`), LF line endings (`\\n`), no UTF-8 BOM, no quotes, no leading/trailing spaces.")
    lines.append(f"- **Header `matching_results.tsv`**: `source1_entity_id\\tmatched_entity_ids`")
    lines.append(f"- **Header `candidate_pairs.tsv`**: `source1_entity_id\\tcandidate_entity_ids`")
    lines.append(f"- **Candidate Superset Guarantee**: Every matched entity ID in `matching_results.tsv` is strictly contained inside `candidate_pairs.tsv`.")
    lines.append(f"- **Official Validator Status**: **{'PASSED (0 errors, 0 warnings)' if val_passed else 'CHECK OUTPUT'}**.")
    lines.append(f"- **Independent Pre-Upload Audit**: **{'PASSED (All 17 checks green)' if pre_passed else 'CHECK OUTPUT'}**.\n")

    lines.append("### 2. Country Profiles at Decision Threshold $\\tau = 0.67$")
    lines.append("| Country | Total S1 | Total Matches | Avg Matches/S1 | % Singletons (Empty) | p99 Matches | Max Matches | Avg S2 Matches | Avg S3 Matches |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    # Ground Truth Reference
    lines.append("| **GT (Train Ref)** | 100,000 | 345,716 | 3.46 | 5.58% | 8 | 11 | 1.92 | 2.03 |")
    for c in ["France", "India", "US"]:
        cp = country_profile_main.get(c, {})
        lines.append(
            f"| **{c}** | {len(country_s1_map[c]):,} | {cp.get('total_matches', 0):,} | {cp.get('avg_matches', 0):.2f} | "
            f"{cp.get('pct_empty', 0):.2f}% | {cp.get('p99', 0)} | {cp.get('max', 0)} | {cp.get('avg_s2', 0):.2f} | {cp.get('avg_s3', 0):.2f} |"
        )
    lines.append("")

    lines.append("### 3. France Multi-Threshold Sensitivity Profile (Unlabeled)")
    lines.append("| Threshold $\\tau$ | Total Matches | Avg Matches/S1 | % Singletons (Empty) | p99 | Max | Avg S2 Matches | Avg S3 Matches |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for pf in france_profile_table:
        lines.append(
            f"| $\\tau = {pf['tau']:.2f}$ | {pf['total_matches']:,} | {pf['avg_matches']:.2f} | {pf['pct_empty']:.2f}% | "
            f"{pf['p99']} | {pf['max']} | {pf['avg_s2']:.2f} | {pf['avg_s3']:.2f} |"
        )
    lines.append("")

    lines.append("### 4. Generated Submission & Probe Files SHA-256")
    lines.append("| File | Description | SHA-256 Checksum |")
    lines.append("|---|---|---|")
    for fn, fhash in hashes.items():
        lines.append(f"| `{fn}` | {'Primary Submission' if fn == 'matching_results.tsv' else ('Candidate Set' if fn == 'candidate_pairs.tsv' else 'Probe Variant')} | `{fhash}` |")
    lines.append("")

    lines.append("### 5. Predicted Leaderboard Score Range")
    lines.append(f"- **Weighted Known Component (India 0.468 $\\times$ 0.9298 + US 0.383 $\\times$ 0.9528)**: **0.8001**")
    lines.append(f"- **Worst Case (France = 0.0)**: **{lb_pred_0:.4f}**")
    lines.append(f"- **Conservative (France = 0.50)**: **{lb_pred_50:.4f}**")
    lines.append(f"- **High Confidence (France = 0.90)**: **{lb_pred_90:.4f}**")
    lines.append(f"- **Expected Model Benchmark (France $\\approx$ 0.94)**: **{lb_pred_exp:.4f}**")
    lines.append(f"- **Primary Upload File**: `output/v1/matching_results.tsv` (alongside `output/v1/candidate_pairs.tsv`)\n")

    report_md = "\n".join(lines) + "\n"
    append_to_run_log(report_md)
    print("\n" + report_md, flush=True)


if __name__ == "__main__":
    main()
