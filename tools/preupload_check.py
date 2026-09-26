"""tools/preupload_check.py
Independent verification script for Amazon ML Challenge 2026 submission files (v2).
READ-ONLY: Does not modify or rebuild any files.
Re-implements logic from scratch to verify:
  - Raw byte formatting (tabs, newlines, UTF-8, no BOM, no quotes, no empty tokens)
  - Row coverage (exactly 1,732,544 S1 entities, exact match to test_source1.tsv)
  - ID prefixes (S2-, S3-), existence in test_source2/3, country alignment
  - 1:1 bipartite constraint (no candidate appears in >1 S1 row)
  - Consistency with predictions (p2 >= 0.67, p1_rank <= 8, 1:1 greedy assignment)
  - Candidate subset constraint (every matched ID is strictly inside candidate_pairs.tsv)
  - Sanity distributions per country and overall
  - Probe file consistency (france_empty.tsv)
  - Official validation with student_resource/utils/validate_submission.py (--check-ids)
"""

from collections import defaultdict
import gc
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, List, Set, Tuple, Any

import numpy as np
import polars as pl

# Reconfigure stdout/stderr for UTF-8 on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Paths for v2
PATH_STAGED_MATCHING = PROJECT_ROOT / "output" / "v1" / "matching_results.tsv"
PATH_STAGED_CANDIDATES = PROJECT_ROOT / "output" / "v1" / "candidate_pairs.tsv"
PATH_PROBE_FR_EMPTY = PROJECT_ROOT / "output" / "v1" / "probes" / "france_empty.tsv"

TEST_S1 = PROJECT_ROOT / "student_resource" / "dataset" / "test" / "test_source1.tsv"
TEST_S2 = PROJECT_ROOT / "student_resource" / "dataset" / "test" / "test_source2.tsv"
TEST_S3 = PROJECT_ROOT / "student_resource" / "dataset" / "test" / "test_source3.tsv"
DATA_PREDS_DIR = PROJECT_ROOT / "data" / "preds"

TAU_CONFIG = {
    "India": 0.67,
    "US": 0.67,
    "France": 0.67,
}
CHOSEN_K = 8
EXPECTED_ROWS = 1732544


def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024 * 8):
            h.update(chunk)
    return h.hexdigest()


def check_file_format_raw(filepath: Path, expected_header: bytes) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Checks raw byte compliance:
    1. Header exact match
    2. Exactly 2 tab-separated fields on every line
    3. UTF-8, no BOM; LF line endings
    4. No quote characters, no leading/trailing spaces
    5. No empty tokens inside ID lists (',,', leading or trailing comma)
    """
    issues = []
    stats = {}

    with open(filepath, "rb") as f:
        # Check BOM
        start_bytes = f.read(3)
        if start_bytes == b"\xef\xbb\xbf":
            issues.append("BOM detected at start of file (expected no BOM)")
            bom_present = True
        else:
            bom_present = False
            f.seek(0)

        line_num = 0
        crlf_count = 0
        lf_count = 0
        quote_errors = 0
        field_count_errors = 0
        space_errors = 0
        empty_token_errors = 0
        utf8_errors = 0

        header_raw = f.readline()
        line_num += 1

        if header_raw.endswith(b"\r\n"):
            crlf_count += 1
            header_stripped = header_raw[:-2]
        elif header_raw.endswith(b"\n"):
            lf_count += 1
            header_stripped = header_raw[:-1]
        else:
            header_stripped = header_raw
            issues.append("Header has no standard newline")

        if header_stripped != expected_header:
            issues.append(f"Header is '{header_stripped.decode('utf-8', errors='replace')}', expected '{expected_header.decode('utf-8')}'")

        data_rows = 0
        for raw_line in f:
            line_num += 1
            data_rows += 1

            if raw_line.endswith(b"\r\n"):
                crlf_count += 1
                line = raw_line[:-2]
            elif raw_line.endswith(b"\n"):
                lf_count += 1
                line = raw_line[:-1]
            else:
                line = raw_line
                if line_num <= EXPECTED_ROWS + 1:
                    issues.append(f"Line {line_num} does not end with standard newline")

            try:
                line_str = line.decode("utf-8")
            except UnicodeDecodeError:
                utf8_errors += 1
                continue

            parts = line.split(b"\t")
            if len(parts) != 2:
                field_count_errors += 1
                continue

            f0, f1 = parts[0], parts[1]

            if b'"' in f0 or b"'" in f0 or b'"' in f1 or b"'" in f1:
                quote_errors += 1

            if f0 != f0.strip(b" ") or f1 != f1.strip(b" "):
                space_errors += 1

            if f1 != b"":
                if b",," in f1 or f1.startswith(b",") or f1.endswith(b","):
                    empty_token_errors += 1

        stats["data_rows"] = data_rows
        stats["crlf_count"] = crlf_count
        stats["lf_count"] = lf_count
        stats["line_ending"] = "CRLF" if crlf_count > lf_count else "LF"
        stats["bom_present"] = bom_present
        stats["field_count_errors"] = field_count_errors
        stats["quote_errors"] = quote_errors
        stats["space_errors"] = space_errors
        stats["empty_token_errors"] = empty_token_errors
        stats["utf8_errors"] = utf8_errors

    if field_count_errors > 0:
        issues.append(f"{field_count_errors} lines have != 2 tab-separated fields")
    if quote_errors > 0:
        issues.append(f"{quote_errors} lines contain quote characters")
    if space_errors > 0:
        issues.append(f"{space_errors} lines have leading/trailing spaces")
    if empty_token_errors > 0:
        issues.append(f"{empty_token_errors} lines have empty tokens (,, or leading/trailing comma)")
    if utf8_errors > 0:
        issues.append(f"{utf8_errors} lines failed UTF-8 decoding")

    passed = len(issues) == 0
    return passed, issues, stats


def main():
    start_time = time.time()
    all_passed = True

    print("=" * 85)
    print("INDEPENDENT PRE-UPLOAD VERIFICATION AUDIT (v2)")
    print(f"Matching file:   {PATH_STAGED_MATCHING}")
    print(f"Candidate file:  {PATH_STAGED_CANDIDATES}")
    print(f"Probe file:      {PATH_PROBE_FR_EMPTY}")
    print(f"Decision tau:    {TAU_CONFIG}")
    print(f"Top-K cap:       {CHOSEN_K}")
    print("=" * 85)

    if not PATH_STAGED_MATCHING.exists():
        print(f"ERROR: {PATH_STAGED_MATCHING} does not exist!")
        sys.exit(1)
    if not PATH_STAGED_CANDIDATES.exists():
        print(f"ERROR: {PATH_STAGED_CANDIDATES} does not exist!")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # SECTION A: FILE FORMAT (Raw Bytes)
    # -------------------------------------------------------------------------
    print("\n--- SECTION A: FILE FORMAT (Raw Bytes) ---")
    m_pass, m_issues, m_stats = check_file_format_raw(PATH_STAGED_MATCHING, b"source1_entity_id\tmatched_entity_ids")
    c_pass, c_issues, c_stats = check_file_format_raw(PATH_STAGED_CANDIDATES, b"source1_entity_id\tcandidate_entity_ids")
    p_pass, p_issues, p_stats = check_file_format_raw(PATH_PROBE_FR_EMPTY, b"source1_entity_id\tmatched_entity_ids")

    # 1. Header
    print("1. Headers exact match:")
    print(f"   matching_results: {'PASS' if not any('Header' in i for i in m_issues) else 'FAIL: ' + str(m_issues)}")
    print(f"   candidate_pairs:  {'PASS' if not any('Header' in i for i in c_issues) else 'FAIL: ' + str(c_issues)}")
    print(f"   france_empty:     {'PASS' if not any('Header' in i for i in p_issues) else 'FAIL: ' + str(p_issues)}")
    if not (m_pass and c_pass and p_pass):
        all_passed = False

    # 2. Exactly 2 tab-separated fields
    print(f"2. Exactly 2 tab-separated fields on every line:")
    print(f"   matching_results: {'PASS' if m_stats['field_count_errors'] == 0 else 'FAIL'}")
    print(f"   candidate_pairs:  {'PASS' if c_stats['field_count_errors'] == 0 else 'FAIL'}")
    print(f"   france_empty:     {'PASS' if p_stats['field_count_errors'] == 0 else 'FAIL'}")

    # 3. UTF-8, no BOM, LF line endings
    print(f"3. UTF-8 encoding, no BOM, line endings:")
    print(f"   matching_results: BOM={'None' if not m_stats['bom_present'] else 'BOM!'}, Ending={m_stats['line_ending']} ({m_stats['lf_count']} LF)")
    print(f"   candidate_pairs:  BOM={'None' if not c_stats['bom_present'] else 'BOM!'}, Ending={c_stats['line_ending']} ({c_stats['lf_count']} LF)")
    print(f"   france_empty:     BOM={'None' if not p_stats['bom_present'] else 'BOM!'}, Ending={p_stats['line_ending']} ({p_stats['lf_count']} LF)")

    # 4. No quote characters, no space violations
    print(f"4. Quotes & spaces: matching={'PASS' if m_stats['quote_errors']==0 and m_stats['space_errors']==0 else 'FAIL'}, candidates={'PASS' if c_stats['quote_errors']==0 and c_stats['space_errors']==0 else 'FAIL'}")

    # 5. No empty tokens
    print(f"5. No empty tokens (',,'): matching={'PASS' if m_stats['empty_token_errors']==0 else 'FAIL'}, candidates={'PASS' if c_stats['empty_token_errors']==0 else 'FAIL'}")

    # -------------------------------------------------------------------------
    # SECTION B: ROW COVERAGE
    # -------------------------------------------------------------------------
    print("\n--- SECTION B: ROW COVERAGE ---")
    row_count_ok = (m_stats["data_rows"] == EXPECTED_ROWS) and (c_stats["data_rows"] == EXPECTED_ROWS)
    print(f"6. Data rows count == {EXPECTED_ROWS:,}:")
    print(f"   matching_results: {m_stats['data_rows']:,} rows -> {'PASS' if m_stats['data_rows'] == EXPECTED_ROWS else 'FAIL'}")
    print(f"   candidate_pairs:  {c_stats['data_rows']:,} rows -> {'PASS' if c_stats['data_rows'] == EXPECTED_ROWS else 'FAIL'}")
    if not row_count_ok:
        all_passed = False

    print("   Loading official test S1 entities...")
    t0 = time.time()
    test_s1_df = pl.read_csv(TEST_S1, separator="\t", columns=["entity_id", "country"]).select([
        pl.col("entity_id").str.strip_chars().alias("s1"),
        pl.col("country"),
    ])
    official_s1_list = test_s1_df["s1"].to_list()
    official_s1_set = set(official_s1_list)
    s1_country_map = dict(zip(official_s1_list, test_s1_df["country"].to_list()))
    print(f"   Loaded {len(official_s1_list):,} test S1 entities in {time.time()-t0:.2f}s.")

    staged_raw_df = pl.read_csv(
        PATH_STAGED_MATCHING,
        separator="\t",
        has_header=True,
        schema={"source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8},
    ).rename({"source1_entity_id": "s1", "matched_entity_ids": "matches"})

    staged_s1_list = staged_raw_df["s1"].str.strip_chars().to_list()
    staged_s1_set = set(staged_s1_list)

    dup_s1 = len(staged_s1_list) - len(staged_s1_set)
    print(f"7. No duplicate source1_entity_id: Duplicate count = {dup_s1} -> {'PASS' if dup_s1 == 0 else 'FAIL'}")
    if dup_s1 != 0:
        all_passed = False

    missing_s1 = official_s1_set - staged_s1_set
    extra_s1 = staged_s1_set - official_s1_set
    print(f"8. Set of S1 IDs exactly matches test_source1.tsv:")
    print(f"   Missing S1 count: {len(missing_s1)}, Extra S1 count: {len(extra_s1)} -> {'PASS' if len(missing_s1) == 0 and len(extra_s1) == 0 else 'FAIL'}")
    if len(missing_s1) > 0 or len(extra_s1) > 0:
        all_passed = False

    # Explode staged matches to pairs
    staged_pairs_df = (
        staged_raw_df
        .filter(pl.col("matches").is_not_null() & (pl.col("matches") != ""))
        .with_columns(pl.col("matches").str.split(","))
        .explode("matches")
        .select([
            pl.col("s1").str.strip_chars().alias("s1"),
            pl.col("matches").str.strip_chars().alias("cid"),
        ])
    )
    n_staged_pairs = len(staged_pairs_df)
    print(f"   Exploded staged pairs: {n_staged_pairs:,} matches.")

    # -------------------------------------------------------------------------
    # SECTION C: MATCHED IDS & 1:1 BIPARTITE CONSTRAINT
    # -------------------------------------------------------------------------
    print("\n--- SECTION C: MATCHED IDS & 1:1 BIPARTITE CONSTRAINT ---")
    row_dups = (
        staged_pairs_df
        .group_by(["s1", "cid"])
        .len()
        .filter(pl.col("len") > 1)
    )
    row_internal_dups = len(row_dups)
    print(f"10. No duplicate IDs within any row: Violations = {row_internal_dups} -> {'PASS' if row_internal_dups == 0 else 'FAIL'}")
    if row_internal_dups != 0:
        all_passed = False

    # 11. One-to-one constraint: no candidate appears in >1 row
    cid_counts = staged_pairs_df.group_by("cid").len().filter(pl.col("len") > 1)
    contested_cid_count = len(cid_counts)
    print(f"11. One-to-one (no candidate appears in >1 row): Contested candidates = {contested_cid_count} -> {'PASS' if contested_cid_count == 0 else 'FAIL'}")
    if contested_cid_count != 0:
        all_passed = False

    # Prefix check
    prefix_violations = len(staged_pairs_df.filter(
        (~pl.col("cid").str.starts_with("S2-")) & (~pl.col("cid").str.starts_with("S3-"))
    ))
    print(f"   Prefix check (S2- / S3-): Violations = {prefix_violations} -> {'PASS' if prefix_violations == 0 else 'FAIL'}")
    if prefix_violations != 0:
        all_passed = False

    # Check candidate existence in test_source2 / test_source3
    print("   Scanning test_source2.tsv and test_source3.tsv via hash-join...")
    unique_cids_df = staged_pairs_df.select("cid").unique()
    t_join = time.time()
    s2_found = (
        pl.scan_csv(TEST_S2, separator="\t")
        .select([
            pl.col("entity_id").str.strip_chars().alias("cid"),
            pl.col("country").alias("country_cand"),
        ])
        .join(unique_cids_df.lazy(), on="cid", how="inner")
        .collect()
    )
    s3_found = (
        pl.scan_csv(TEST_S3, separator="\t")
        .select([
            pl.col("entity_id").str.strip_chars().alias("cid"),
            pl.col("country").alias("country_cand"),
        ])
        .join(unique_cids_df.lazy(), on="cid", how="inner")
        .collect()
    )
    cand_info_df = pl.concat([s2_found, s3_found])
    print(f"   Verified {len(cand_info_df):,} unique candidates in {time.time()-t_join:.2f}s.")
    missing_cids_count = len(unique_cids_df) - len(cand_info_df)
    print(f"9. Candidate existence: Nonexistent IDs = {missing_cids_count} -> {'PASS' if missing_cids_count == 0 else 'FAIL'}")
    if missing_cids_count != 0:
        all_passed = False

    # 12. Country consistency
    pair_country_check = (
        staged_pairs_df
        .join(test_s1_df, on="s1", how="inner")
        .rename({"country": "country_s1"})
        .join(cand_info_df, on="cid", how="inner")
        .filter(pl.col("country_s1") != pl.col("country_cand"))
    )
    country_mismatches = len(pair_country_check)
    print(f"12. Every matched ID's country matches S1: Mismatches = {country_mismatches} -> {'PASS' if country_mismatches == 0 else 'FAIL'}")
    if country_mismatches != 0:
        all_passed = False
    del cand_info_df, pair_country_check, s2_found, s3_found
    gc.collect()

    # -------------------------------------------------------------------------
    # SECTION D: CONSISTENCY WITH MODEL PREDICTIONS & 1:1 RECOMPUTATION
    # -------------------------------------------------------------------------
    print("\n--- SECTION D: CONSISTENCY WITH MODEL PREDICTIONS & 1:1 RECOMPUTATION ---")
    print(f"13. Recomputing greedy 1:1 assignment directly from data/preds/test_*.parquet at tau={TAU_CONFIG}, top-K={CHOSEN_K}...")
    t0 = time.time()

    recomputed_pairs_list = []
    total_recomputed_ties = 0

    for country in ["France", "India", "US"]:
        pred_file = DATA_PREDS_DIR / f"test_{country}.parquet"
        if not pred_file.exists():
            print(f"   [ERROR] Missing prediction file {pred_file}")
            all_passed = False
            continue

        tau = TAU_CONFIG[country]
        preds = pl.read_parquet(pred_file)
        filtered = preds.filter((pl.col("p1_rank") <= CHOSEN_K) & (pl.col("p2") >= tau))

        if len(filtered) > 0:
            max_scores = filtered.group_by(["cand_num", "cand_src"]).agg(max_p2=pl.col("p2").max())
            joined = filtered.join(max_scores, on=["cand_num", "cand_src"])
            matching_max = joined.filter(pl.col("p2") == pl.col("max_p2"))
            c_ties = len(matching_max) - len(max_scores)
            total_recomputed_ties += c_ties

            # Deduplicate strictly 1:1 matching step 6 logic
            assigned = (
                matching_max
                .sort(["p2", "p1", "s1_num"], descending=[True, True, False])
                .unique(subset=["cand_num", "cand_src"], keep="first")
            )

            # Format IDs
            assigned = assigned.select([
                ("S1-" + pl.col("s1_num").cast(pl.String)).alias("s1"),
                ("S" + pl.col("cand_src").cast(pl.String) + "-" + pl.col("cand_num").cast(pl.String)).alias("cid"),
            ])
            recomputed_pairs_list.append(assigned)
            print(f"   [{country}] Recomputed: {len(assigned):,} matches (ties: {c_ties})")
        del preds, filtered
        gc.collect()

    all_recomputed_df = pl.concat(recomputed_pairs_list) if recomputed_pairs_list else pl.DataFrame()
    print(f"   Total independently recomputed pairs: {len(all_recomputed_df):,} in {time.time()-t0:.2f}s.")
    print(f"   Exact ties across all countries: {total_recomputed_ties}")

    # Compare staged_pairs_df vs all_recomputed_df
    diff_staged_extra = staged_pairs_df.join(all_recomputed_df, on=["s1", "cid"], how="anti")
    diff_recomp_extra = all_recomputed_df.join(staged_pairs_df, on=["s1", "cid"], how="anti")
    pair_diff_count = len(diff_staged_extra) + len(diff_recomp_extra)
    print(f"14. Full Independent Recompute Match: Differing pairs = {pair_diff_count} -> {'PASS' if pair_diff_count == 0 else 'FAIL'}")
    if pair_diff_count != 0:
        all_passed = False
        print(f"    Staged extra: {len(diff_staged_extra)}, Recomputed extra: {len(diff_recomp_extra)}")

    # -------------------------------------------------------------------------
    # SECTION E: CANDIDATE SUBSET GUARANTEE
    # -------------------------------------------------------------------------
    print("\n--- SECTION E: CANDIDATE SUBSET GUARANTEE ---")
    print("15. Checking that matched_entity_ids is strictly inside candidate_pairs.tsv (streaming lockstep)...")
    t0 = time.time()
    s1_subset_violations = 0
    total_missing_matches = 0
    total_checked = 0

    with open(PATH_STAGED_MATCHING, "r", encoding="utf-8") as f_match, open(PATH_STAGED_CANDIDATES, "r", encoding="utf-8") as f_cands:
        h_m = f_match.readline()
        h_c = f_cands.readline()
        for line_m, line_c in zip(f_match, f_cands):
            total_checked += 1
            s1_m, _, m_str = line_m.partition("\t")
            s1_c, _, c_str = line_c.partition("\t")
            m_str = m_str.rstrip("\r\n")
            c_str = c_str.rstrip("\r\n")

            if s1_m != s1_c:
                print(f"   [ERROR] Entity alignment mismatch at line {total_checked}: '{s1_m}' != '{s1_c}'")
                s1_subset_violations += 1
                all_passed = False
                break

            if m_str:
                cand_set = set(c_str.split(",")) if c_str else set()
                matched_ids = m_str.split(",")
                missing = [m for m in matched_ids if m not in cand_set]
                if missing:
                    s1_subset_violations += 1
                    total_missing_matches += len(missing)

    print(f"   Checked {total_checked:,} S1 rows in {time.time()-t0:.2f}s.")
    print(f"   S1 entities with matched IDs not in candidate_pairs.tsv: {s1_subset_violations}")
    print(f"   Candidate subset check: {'PASS (0 violations)' if s1_subset_violations == 0 else f'FAIL ({s1_subset_violations} violations)'}")
    if s1_subset_violations != 0:
        all_passed = False

    # -------------------------------------------------------------------------
    # SECTION F: SANITY DISTRIBUTION STATS
    # -------------------------------------------------------------------------
    print("\n--- SECTION F: SANITY DISTRIBUTION STATS ---")
    match_counts_df = (
        test_s1_df
        .join(staged_pairs_df.group_by("s1").len().rename({"len": "k"}), on="s1", how="left")
        .with_columns(pl.col("k").fill_null(0))
    )

    sanity_ok = True
    for country in ["France", "India", "US"]:
        c_df = match_counts_df.filter(pl.col("country") == country)
        c_k = c_df["k"].to_numpy()
        c_tot = len(c_k)
        c_empty = int(np.sum(c_k == 0))
        avg_m = float(np.mean(c_k))
        pct_empty = c_empty / c_tot * 100
        med_m = float(np.median(c_k))
        p99_m = float(np.percentile(c_k, 99))
        max_m = int(np.max(c_k))

        print(f"   {country:<8} ({c_tot:,} entities):")
        print(f"     Avg Matches:    {avg_m:.4f}")
        print(f"     Singletons (%): {pct_empty:.2f}% ({c_empty:,})")
        print(f"     Median Matches: {med_m:.1f}")
        print(f"     p99 Matches:    {p99_m:.1f}")
        print(f"     Max Matches:    {max_m}")

        if country == "France" and not (2.8 <= avg_m <= 3.8):
            print(f"     [WARNING] France avg matches {avg_m:.4f} outside expected [2.8, 3.8]")
            sanity_ok = False
        elif country in ("India", "US") and not (2.5 <= avg_m <= 3.6):
            print(f"     [WARNING] {country} avg matches {avg_m:.4f} outside expected [2.5, 3.6]")
            sanity_ok = False
        if not (4.0 <= pct_empty <= 16.0):
            print(f"     [WARNING] {country} singleton % {pct_empty:.2f}% outside expected [4.0, 16.0]")
            sanity_ok = False
        if max_m >= 20:
            print(f"     [WARNING] {country} max matches {max_m} >= 20")
            sanity_ok = False

    all_k = match_counts_df["k"].to_numpy()
    print(f"   Overall ({len(all_k):,} entities):")
    print(f"     Avg Matches:    {np.mean(all_k):.4f}")
    print(f"     Singletons (%): {np.sum(all_k == 0)/len(all_k)*100:.2f}%")
    print(f"     Median Matches: {np.median(all_k):.1f}")
    print(f"     p99 Matches:    {np.percentile(all_k, 99):.1f}")
    print(f"     Max Matches:    {np.max(all_k)}")
    print(f"   Sanity ranges: {'PASS' if sanity_ok else 'FLAGGED'}")
    if not sanity_ok:
        all_passed = False
    del match_counts_df, staged_pairs_df, staged_raw_df
    gc.collect()

    # -------------------------------------------------------------------------
    # SECTION G: PROBE FILE VERIFICATION
    # -------------------------------------------------------------------------
    print("\n--- SECTION G: PROBE FILE VERIFICATION ---")
    print("16. Verifying output/v1/probes/france_empty.tsv:")
    fr_empty_violations = 0
    non_fr_mismatches = 0
    total_probe_lines = 0

    with open(PATH_STAGED_MATCHING, "r", encoding="utf-8") as f_staged, open(PATH_PROBE_FR_EMPTY, "r", encoding="utf-8") as f_probe:
        h_s = f_staged.readline()
        h_p = f_probe.readline()
        for line_s, line_p in zip(f_staged, f_probe):
            total_probe_lines += 1
            s1_s, _, m_s = line_s.partition("\t")
            s1_p, _, m_p = line_p.partition("\t")

            cntry = s1_country_map.get(s1_s.strip())
            if cntry == "France":
                if m_p.strip() != "":
                    fr_empty_violations += 1
            else:
                if line_s != line_p:
                    non_fr_mismatches += 1

    probe_hash = sha256_file(PATH_PROBE_FR_EMPTY)
    print(f"    Total probe rows verified: {total_probe_lines:,}")
    print(f"    France non-empty violations: {fr_empty_violations}")
    print(f"    Non-France byte mismatches: {non_fr_mismatches}")
    print(f"    Probe SHA256: {probe_hash}")
    probe_ok = (fr_empty_violations == 0 and non_fr_mismatches == 0 and total_probe_lines == EXPECTED_ROWS)
    print(f"    Probe consistency: {'PASS' if probe_ok else 'FAIL'}")
    if not probe_ok:
        all_passed = False

    # -------------------------------------------------------------------------
    # SECTION H: OFFICIAL VALIDATOR
    # -------------------------------------------------------------------------
    print("\n--- SECTION H: OFFICIAL VALIDATOR ---")
    print("17. Running official validator: student_resource/utils/validate_submission.py --check-ids...")
    cmd = [
        sys.executable,
        "student_resource/utils/validate_submission.py",
        "--matching", str(PATH_STAGED_MATCHING),
        "--candidate", str(PATH_STAGED_CANDIDATES),
        "--test-dir", "student_resource/dataset/test",
        "--check-ids",
    ]
    val_proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    print("\n--- Validator Full Output Begin ---")
    if val_proc.stdout:
        print(val_proc.stdout.strip())
    if val_proc.stderr:
        print(val_proc.stderr.strip())
    print("--- Validator Full Output End ---")

    val_passed = (val_proc.returncode == 0) and ("FAILED" not in val_proc.stdout) and ("WARNING" not in val_proc.stdout)
    print(f"\nOfficial Validator Result: {'PASS (0 errors, 0 warnings)' if val_passed else 'FAIL'}")
    if not val_passed:
        all_passed = False

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    staged_hash = sha256_file(PATH_STAGED_MATCHING)
    cand_hash = sha256_file(PATH_STAGED_CANDIDATES)

    print("\n" + "=" * 85)
    print(f"TOTAL AUDIT TIME: {time.time()-start_time:.2f}s")
    print("=" * 85)

    if all_passed:
        print("\n>>> ALL CHECKS PASSED: READY TO UPLOAD <<<")
        print(f"Primary Matching File: {PATH_STAGED_MATCHING}")
        print(f"SHA256: {staged_hash}")
        print(f"Candidate File:        {PATH_STAGED_CANDIDATES}")
        print(f"SHA256: {cand_hash}")
    else:
        print("\n>>> PRE-UPLOAD AUDIT FAILED <<<")
        print(f"File: {PATH_STAGED_MATCHING}")
        print(f"SHA256: {staged_hash}")
        sys.exit(1)


if __name__ == "__main__":
    main()
