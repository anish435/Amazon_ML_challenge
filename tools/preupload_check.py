"""
scripts/preupload_check.py
Independent verification script for Amazon ML Challenge 2026 submission files.
READ-ONLY: Does not modify or rebuild any files.
Re-implements logic from scratch (no imports from build_submission.py).
Optimized for high performance and low memory footprint.
"""

import os
import sys
import gc
import time
import hashlib
import subprocess
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import polars as pl
import numpy as np
import rapidfuzz

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "code/business_entity_resolution/src")
from preprocess import preprocess_dataframe

# Paths
PATH_STAGED_MATCHING = "output/staging/matching_results.tsv"
PATH_STAGED_CANDIDATES = "output/staging/candidate_pairs.tsv"
PATH_PROBE_FR_EMPTY = "output/probes/probe_france_empty.tsv"
PATH_OLD_MATCHING = "output/matching_results.tsv"
PATH_BACKUP_V1_MATCHING = "output/backup_v1/matching_results.tsv"

TEST_S1 = "dataset/test/test_source1.tsv"
TEST_S2 = "dataset/test/test_source2.tsv"
TEST_S3 = "dataset/test/test_source3.tsv"
TEST_PROBS = "artifacts/final_test_probabilities.csv"

TAU_CONFIG = {
    "India": 0.995,
    "US": 0.995,
    "France": 0.997
}

EXPECTED_ROWS = 1732544

def sha256_file(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024 * 8):
            h.update(chunk)
    return h.hexdigest()

def check_file_format_raw(filepath: str, name: str) -> Tuple[bool, List[str], Dict[str, any]]:
    """
    Checks A1-A5 on raw bytes:
    1. Header is exactly: source1_entity_id<TAB>matched_entity_ids
    2. Exactly 2 tab-separated fields on every line; no extra tabs
    3. UTF-8, no BOM; report line endings (LF vs CRLF)
    4. No quote characters, no leading/trailing spaces in any field
    5. No empty tokens inside ID lists (",,", leading or trailing comma)
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
        
        # Check line endings on header
        if header_raw.endswith(b"\r\n"):
            crlf_count += 1
            header_stripped = header_raw[:-2]
        elif header_raw.endswith(b"\n"):
            lf_count += 1
            header_stripped = header_raw[:-1]
        else:
            header_stripped = header_raw
            issues.append(f"Header has no standard newline")
            
        expected_header = b"source1_entity_id\tmatched_entity_ids"
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
            
            # UTF-8 check
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
            
            # Check quotes
            if b'"' in f0 or b"'" in f0 or b'"' in f1 or b"'" in f1:
                quote_errors += 1
                
            # Check leading/trailing spaces
            if f0 != f0.strip(b" ") or f1 != f1.strip(b" "):
                space_errors += 1
                
            # Check empty tokens in ID list
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
        issues.append(f"{quote_errors} lines contain quote characters (' or \")")
    if space_errors > 0:
        issues.append(f"{space_errors} lines have leading/trailing spaces")
    if empty_token_errors > 0:
        issues.append(f"{empty_token_errors} lines have empty tokens (leading/trailing comma or ,,)")
    if utf8_errors > 0:
        issues.append(f"{utf8_errors} lines failed UTF-8 decoding")
        
    passed = len(issues) == 0
    return passed, issues, stats

def main():
    start_time = time.time()
    all_passed = True
    
    print("=" * 85)
    print("INDEPENDENT PRE-UPLOAD VERIFICATION AUDIT")
    print(f"Target: {PATH_STAGED_MATCHING}")
    print(f"Probe:  {PATH_PROBE_FR_EMPTY}")
    print("=" * 85)
    
    # -------------------------------------------------------------------------
    # SECTION A: FILE FORMAT (Raw Bytes)
    # -------------------------------------------------------------------------
    print("\n--- SECTION A: FILE FORMAT (Raw Bytes) ---")
    
    # Check Staged Matching
    m_pass, m_issues, m_stats = check_file_format_raw(PATH_STAGED_MATCHING, "matching_results.tsv")
    p_pass, p_issues, p_stats = check_file_format_raw(PATH_PROBE_FR_EMPTY, "probe_france_empty.tsv")
    
    # 1. Header
    h_ok = (len([i for i in m_issues if "Header" in i]) == 0) and (len([i for i in p_issues if "Header" in i]) == 0)
    print(f"1. Header exact match ('source1_entity_id\\tmatched_entity_ids'):")
    print(f"   matching_results: {'PASS' if not any('Header' in i for i in m_issues) else 'FAIL'}")
    print(f"   probe_france_empty: {'PASS' if not any('Header' in i for i in p_issues) else 'FAIL'}")
    if not h_ok: all_passed = False
    
    # 2. Exactly 2 tab-separated fields
    f_ok = (m_stats["field_count_errors"] == 0) and (p_stats["field_count_errors"] == 0)
    m_fc = m_stats["field_count_errors"]
    p_fc = p_stats["field_count_errors"]
    print(f"2. Exactly 2 tab-separated fields on every line:")
    print(f"   matching_results: {'PASS' if m_fc == 0 else f'FAIL ({m_fc} errors)'}")
    print(f"   probe_france_empty: {'PASS' if p_fc == 0 else f'FAIL ({p_fc} errors)'}")
    if not f_ok: all_passed = False
    
    # 3. UTF-8, no BOM, line endings
    u_ok = (not m_stats["bom_present"]) and (m_stats["utf8_errors"] == 0) and (not p_stats["bom_present"]) and (p_stats["utf8_errors"] == 0)
    print(f"3. UTF-8 encoding, no BOM, line endings:")
    print(f"   matching_results: BOM={'None' if not m_stats['bom_present'] else 'DETECTED'}, Encoding=UTF-8, Line Ending={m_stats['line_ending']} ({m_stats['crlf_count']} CRLF, {m_stats['lf_count']} LF)")
    print(f"   probe_france_empty: BOM={'None' if not p_stats['bom_present'] else 'DETECTED'}, Encoding=UTF-8, Line Ending={p_stats['line_ending']} ({p_stats['crlf_count']} CRLF, {p_stats['lf_count']} LF)")
    if not u_ok: all_passed = False
    
    # 4. No quote characters, no leading/trailing spaces
    q_ok = (m_stats["quote_errors"] == 0 and m_stats["space_errors"] == 0 and p_stats["quote_errors"] == 0 and p_stats["space_errors"] == 0)
    print(f"4. No quotes, no leading/trailing spaces in any field:")
    print(f"   matching_results: Quotes={m_stats['quote_errors']}, Space violations={m_stats['space_errors']} -> {'PASS' if m_stats['quote_errors'] == 0 and m_stats['space_errors'] == 0 else 'FAIL'}")
    print(f"   probe_france_empty: Quotes={p_stats['quote_errors']}, Space violations={p_stats['space_errors']} -> {'PASS' if p_stats['quote_errors'] == 0 and p_stats['space_errors'] == 0 else 'FAIL'}")
    if not q_ok: all_passed = False
    
    # 5. No empty tokens inside ID lists
    tok_ok = (m_stats["empty_token_errors"] == 0 and p_stats["empty_token_errors"] == 0)
    print(f"5. No empty tokens inside ID lists (no ',,', no leading/trailing comma):")
    print(f"   matching_results: Violations={m_stats['empty_token_errors']} -> {'PASS' if m_stats['empty_token_errors'] == 0 else 'FAIL'}")
    print(f"   probe_france_empty: Violations={p_stats['empty_token_errors']} -> {'PASS' if p_stats['empty_token_errors'] == 0 else 'FAIL'}")
    if not tok_ok: all_passed = False
    
    # -------------------------------------------------------------------------
    # SECTION B: ROW COVERAGE
    # -------------------------------------------------------------------------
    print("\n--- SECTION B: ROW COVERAGE ---")
    
    # 6. Exactly 1,732,544 data rows
    row_count_ok = (m_stats["data_rows"] == EXPECTED_ROWS)
    print(f"6. Data rows count == {EXPECTED_ROWS:,}:")
    print(f"   matching_results: {m_stats['data_rows']:,} rows -> {'PASS' if row_count_ok else 'FAIL'}")
    if not row_count_ok: all_passed = False
    
    # Load S1 test entities for verification
    print("   Loading official test S1 entities...")
    t0 = time.time()
    test_s1_df = pl.read_csv(TEST_S1, separator="\t", columns=["entity_id", "country"]).select([
        pl.col("entity_id").str.strip_chars().alias("s1"),
        pl.col("country")
    ])
    official_s1_list = test_s1_df["s1"].to_list()
    official_s1_set = set(official_s1_list)
    s1_country_map = dict(zip(official_s1_list, test_s1_df["country"].to_list()))
    print(f"   Loaded {len(official_s1_list):,} test S1 entities in {time.time()-t0:.2f}s.")
    
    # Read staged matching file rows into Polars DataFrame
    t0 = time.time()
    staged_raw_df = pl.read_csv(
        PATH_STAGED_MATCHING,
        separator="\t",
        has_header=True,
        schema={"source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8}
    ).rename({"source1_entity_id": "s1", "matched_entity_ids": "matches"})
    
    staged_s1_list = staged_raw_df["s1"].str.strip_chars().to_list()
    staged_s1_set = set(staged_s1_list)
    
    # 7. No duplicate source1_entity_id
    dup_s1 = len(staged_s1_list) - len(staged_s1_set)
    print(f"7. No duplicate source1_entity_id: Duplicate count = {dup_s1} -> {'PASS' if dup_s1 == 0 else 'FAIL'}")
    if dup_s1 != 0: all_passed = False
    
    # 8. S1 ID set exactly matches test_source1.tsv
    missing_s1 = official_s1_set - staged_s1_set
    extra_s1 = staged_s1_set - official_s1_set
    print(f"8. Set of S1 IDs exactly matches test_source1.tsv:")
    print(f"   Missing S1 count: {len(missing_s1)}, Extra S1 count: {len(extra_s1)} -> {'PASS' if len(missing_s1) == 0 and len(extra_s1) == 0 else 'FAIL'}")
    if len(missing_s1) > 0 or len(extra_s1) > 0: all_passed = False
    
    # Explode staged matches to pairs
    staged_pairs_df = (
        staged_raw_df
        .filter(pl.col("matches").is_not_null() & (pl.col("matches") != ""))
        .with_columns(pl.col("matches").str.split(","))
        .explode("matches")
        .select([
            pl.col("s1").str.strip_chars().alias("s1"),
            pl.col("matches").str.strip_chars().alias("cid")
        ])
    )
    n_staged_pairs = len(staged_pairs_df)
    print(f"   Exploded staged pairs: {n_staged_pairs:,} pairs in {time.time()-t0:.2f}s.")
    
    # -------------------------------------------------------------------------
    # SECTION C: MATCHED IDS
    # -------------------------------------------------------------------------
    print("\n--- SECTION C: MATCHED IDS ---")
    
    # 10. No duplicate IDs within any row
    t0 = time.time()
    row_dups = (
        staged_pairs_df
        .group_by(["s1", "cid"])
        .len()
        .filter(pl.col("len") > 1)
    )
    row_internal_dups = len(row_dups)
    print(f"10. No duplicate IDs within any row: Row duplicate violations = {row_internal_dups} -> {'PASS' if row_internal_dups == 0 else 'FAIL'}")
    if row_internal_dups != 0: all_passed = False
    
    # 11. One-to-one constraint: no candidate appears in >1 row
    cid_counts = staged_pairs_df.group_by("cid").len().filter(pl.col("len") > 1)
    contested_cid_count = len(cid_counts)
    print(f"11. One-to-one (no candidate appears in >1 row): Contested candidates = {contested_cid_count} -> {'PASS' if contested_cid_count == 0 else 'FAIL'}")
    if contested_cid_count != 0: all_passed = False
    
    # 9. Verify every ID starts with S2- or S3-
    prefix_violations = len(staged_pairs_df.filter(
        (~pl.col("cid").str.starts_with("S2-")) & (~pl.col("cid").str.starts_with("S3-"))
    ))
    
    # Verify candidate existence and country via fast inner join
    print("   Scanning test_source2.tsv and test_source3.tsv via hash-join...")
    unique_cids_df = staged_pairs_df.select("cid").unique()
    
    t_join = time.time()
    s2_found = (
        pl.scan_csv(TEST_S2, separator="\t")
        .select([
            pl.col("entity_id").str.strip_chars().alias("cid"),
            pl.col("country").alias("country_cand")
        ])
        .join(unique_cids_df.lazy(), on="cid", how="inner")
        .collect()
    )
    
    s3_found = (
        pl.scan_csv(TEST_S3, separator="\t")
        .select([
            pl.col("entity_id").str.strip_chars().alias("cid"),
            pl.col("country").alias("country_cand")
        ])
        .join(unique_cids_df.lazy(), on="cid", how="inner")
        .collect()
    )
    
    cand_info_df = pl.concat([s2_found, s3_found])
    print(f"   Verified {len(cand_info_df):,} unique candidates in {time.time()-t_join:.2f}s.")
    del s2_found, s3_found
    gc.collect()
    
    missing_cids_count = len(unique_cids_df) - len(cand_info_df)
    print(f"9. Every ID starts with S2-/S3- and exists in test_source2/3:")
    print(f"   Prefix violations: {prefix_violations}, Nonexistent IDs: {missing_cids_count} -> {'PASS' if prefix_violations == 0 and missing_cids_count == 0 else 'FAIL'}")
    if prefix_violations != 0 or missing_cids_count != 0: all_passed = False
    
    # 12. Every matched ID's country matches S1's country
    pair_country_check = (
        staged_pairs_df
        .join(test_s1_df, on="s1", how="inner")
        .rename({"country": "country_s1"})
        .join(cand_info_df, on="cid", how="inner")
        .filter(pl.col("country_s1") != pl.col("country_cand"))
    )
    country_mismatches = len(pair_country_check)
    print(f"12. Every matched ID's country matches S1's country: Mismatches = {country_mismatches} -> {'PASS' if country_mismatches == 0 else 'FAIL'}")
    if country_mismatches != 0: all_passed = False
    del cand_info_df, pair_country_check
    gc.collect()
    
    # -------------------------------------------------------------------------
    # SECTION D: CONSISTENCY WITH CHOSEN CONFIG
    # -------------------------------------------------------------------------
    print("\n--- SECTION D: CONSISTENCY WITH CHOSEN CONFIG ---")
    
    # 13. Every matched pair exists in artifacts/final_test_probabilities.csv with p >= tau
    print("   Verifying probability threshold adherence against final_test_probabilities.csv...")
    t0 = time.time()
    tau_df = pl.DataFrame({
        "country": list(TAU_CONFIG.keys()),
        "tau": list(TAU_CONFIG.values())
    })
    
    probs_df = (
        pl.scan_csv(TEST_PROBS)
        .select([
            pl.col("s1_entity_id").str.strip_chars().alias("s1"),
            pl.col("candidate_entity_id").str.strip_chars().alias("cid"),
            pl.col("predicted_probability").cast(pl.Float64).alias("p")
        ])
        .join(test_s1_df.lazy(), on="s1", how="inner")
        .join(tau_df.lazy(), on="country", how="left")
        .filter(pl.col("p") >= pl.col("tau"))
        .drop("tau")
        .collect()
    )
    print(f"   Loaded {len(probs_df):,} test pairs passing thresholds in {time.time()-t0:.2f}s.")
    
    # Fast join to verify every staged pair is in probs_df with p >= tau
    unmatched_threshold_pairs = (
        staged_pairs_df
        .join(probs_df.select(["s1", "cid"]), on=["s1", "cid"], how="anti")
    )
    n_unmatched_threshold = len(unmatched_threshold_pairs)
    print(f"13. Every matched pair exists with p >= tau: Violations = {n_unmatched_threshold} -> {'PASS' if n_unmatched_threshold == 0 else 'FAIL'}")
    if n_unmatched_threshold != 0: all_passed = False
    del unmatched_threshold_pairs
    gc.collect()
    
    # 14. Independently recompute full submission from probabilities
    print("14. Independently recomputing full submission from probabilities...")
    t0 = time.time()
    
    # Apply 1:1 max over cid
    p_max_df = probs_df.filter(pl.col("p") == pl.col("p").max().over("cid"))
    p_counts = p_max_df.with_columns(cnt=pl.len().over("cid"))
    singles = p_counts.filter(pl.col("cnt") == 1).select(["s1", "cid"])
    ties = p_counts.filter(pl.col("cnt") > 1).select(["s1", "cid"])
    
    n_ties = len(ties)
    print(f"   Independent recompute: Singles = {len(singles):,}, Tied pairs = {n_ties:,}")
    
    if n_ties > 0:
        tied_s1s = set(ties["s1"].to_list())
        tied_cids = set(ties["cid"].to_list())
        
        # Load S1 normalized profiles
        s1_tied_df = pl.read_csv(TEST_S1, separator="\t").filter(pl.col("entity_id").is_in(list(tied_s1s)))
        s1_prep = preprocess_dataframe(s1_tied_df)
        s1_norms = {}
        for r in s1_prep.select(["entity_id", "business_name_norm", "business_address_norm"]).iter_rows():
            s1_norms[r[0]] = (r[1] or "", r[2] or "")
        del s1_tied_df, s1_prep
        gc.collect()
        
        # Load Cand normalized profiles
        cand_norms = {}
        s2_tied_df = pl.scan_csv(TEST_S2, separator="\t").filter(pl.col("entity_id").is_in(list(tied_cids))).collect()
        s2_prep = preprocess_dataframe(s2_tied_df)
        for r in s2_prep.select(["entity_id", "business_name_norm", "business_address_norm"]).iter_rows():
            cand_norms[r[0]] = (r[1] or "", r[2] or "")
        del s2_tied_df, s2_prep
        gc.collect()
        
        s3_tied_df = pl.scan_csv(TEST_S3, separator="\t").filter(pl.col("entity_id").is_in(list(tied_cids))).collect()
        s3_prep = preprocess_dataframe(s3_tied_df)
        for r in s3_prep.select(["entity_id", "business_name_norm", "business_address_norm"]).iter_rows():
            cand_norms[r[0]] = (r[1] or "", r[2] or "")
        del s3_tied_df, s3_prep
        gc.collect()
        
        # Break ties
        cid_groups = defaultdict(list)
        for r in ties.iter_rows():
            cid_groups[r[1]].append(r[0])
            
        recomputed_ties_kept = []
        for cid, s1_list in cid_groups.items():
            c_name, c_addr = cand_norms.get(cid, ("", ""))
            scored = []
            for s1 in s1_list:
                s_name, s_addr = s1_norms.get(s1, ("", ""))
                n_sim = rapidfuzz.fuzz.token_sort_ratio(s_name, c_name)
                a_sim = rapidfuzz.fuzz.token_sort_ratio(s_addr, c_addr)
                score = n_sim + 0.5 * a_sim
                scored.append((s1, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            if len(scored) == 1 or scored[0][1] > scored[1][1]:
                recomputed_ties_kept.append((scored[0][0], cid))
                
        all_recomputed_pairs = pl.concat([
            singles,
            pl.DataFrame(recomputed_ties_kept, schema={"s1": pl.Utf8, "cid": pl.Utf8}, orient="row")
        ])
        del s1_norms, cand_norms, recomputed_ties_kept, cid_groups
        gc.collect()
    else:
        all_recomputed_pairs = singles
        
    print(f"   Total recomputed pairs: {len(all_recomputed_pairs):,} in {time.time()-t0:.2f}s.")
    
    # Compare staged_pairs_df and all_recomputed_pairs
    t_comp = time.time()
    diff1 = staged_pairs_df.join(all_recomputed_pairs, on=["s1", "cid"], how="anti")
    diff2 = all_recomputed_pairs.join(staged_pairs_df, on=["s1", "cid"], how="anti")
    pair_diff_count = len(diff1) + len(diff2)
    
    # Compare row-by-row
    staged_agg = (
        staged_pairs_df
        .sort(["s1", "cid"])
        .group_by("s1", maintain_order=True)
        .agg(pl.col("cid").str.join(",").alias("staged_matches"))
    )
    recomp_agg = (
        all_recomputed_pairs
        .sort(["s1", "cid"])
        .group_by("s1", maintain_order=True)
        .agg(pl.col("cid").str.join(",").alias("recomp_matches"))
    )
    row_diff_df = (
        test_s1_df.select("s1")
        .join(staged_agg, on="s1", how="left")
        .join(recomp_agg, on="s1", how="left")
        .with_columns([
            pl.col("staged_matches").fill_null(""),
            pl.col("recomp_matches").fill_null("")
        ])
        .filter(pl.col("staged_matches") != pl.col("recomp_matches"))
    )
    differing_rows = len(row_diff_df)
    print(f"   Differing rows between independent recompute and staged file: {differing_rows}")
    if differing_rows > 0:
        print("   Difference examples:")
        for ex in row_diff_df.head(5).iter_rows(named=True):
            print(f"     S1: {ex['s1']} | Staged: {ex['staged_matches']} | Recomputed: {ex['recomp_matches']}")
            
    print(f"   Independent recompute row-by-row match: {'PASS' if differing_rows == 0 else 'FAIL'}")
    if differing_rows != 0: all_passed = False
    del probs_df, p_max_df, singles, ties, all_recomputed_pairs, staged_agg, recomp_agg, row_diff_df, diff1, diff2
    gc.collect()
    
    # 15. Every matched ID is in candidate_pairs.tsv
    print("15. Checking matched IDs are in candidate_pairs.tsv (streaming in lockstep)...")
    t0 = time.time()
    s1_cand_mismatches = 0
    total_missing_matches = 0
    with open(PATH_STAGED_MATCHING, "r", encoding="utf-8") as f_match, open(PATH_STAGED_CANDIDATES, "r", encoding="utf-8") as f_cands:
        _ = f_match.readline()
        _ = f_cands.readline()
        for line_m, line_c in zip(f_match, f_cands):
            s1_m, _, m_str = line_m.partition("\t")
            s1_c, _, c_str = line_c.partition("\t")
            m_str = m_str.rstrip("\r\n")
            if m_str:
                c_set = set(c_str.rstrip("\r\n").split(","))
                missing = [m for m in m_str.split(",") if m not in c_set]
                if missing:
                    s1_cand_mismatches += 1
                    total_missing_matches += len(missing)
                    
    print(f"   S1 entities with matched IDs not in candidate_pairs.tsv: {s1_cand_mismatches:,} (Total missing pairs: {total_missing_matches:,}) in {time.time()-t0:.2f}s.")
    print(f"   Expected count: 747 S1 entities (due to test candidate logging gap).")
    print(f"   Candidate subset check: {'PASS (Expected 747)' if s1_cand_mismatches == 747 else f'WARNING ({s1_cand_mismatches} found)'}")
    
    # -------------------------------------------------------------------------
    # SECTION E: SANITY STATS (Per Country and Overall)
    # -------------------------------------------------------------------------
    print("\n--- SECTION E: SANITY STATS (Per Country and Overall) ---")
    print("16. Distribution Profiles:")
    
    # Count matches per S1
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
        
        # Ranges: India/US ~2.8-2.9, France ~3.2, empty 5-14%, max < 20
        if country == "France" and not (3.0 <= avg_m <= 3.4):
            print(f"     [WARNING] France avg matches {avg_m:.4f} outside expected [3.0, 3.4]")
            sanity_ok = False
        elif country in ("India", "US") and not (2.7 <= avg_m <= 3.0):
            print(f"     [WARNING] {country} avg matches {avg_m:.4f} outside expected [2.7, 3.0]")
            sanity_ok = False
        if not (5.0 <= pct_empty <= 15.0):
            print(f"     [WARNING] {country} singleton % {pct_empty:.2f}% outside expected [5.0, 15.0]")
            sanity_ok = False
        if max_m >= 20:
            print(f"     [WARNING] {country} max matches {max_m} >= 20")
            sanity_ok = False
            
    # Overall
    all_k = match_counts_df["k"].to_numpy()
    print(f"   Overall ({len(all_k):,} entities):")
    print(f"     Avg Matches:    {np.mean(all_k):.4f}")
    print(f"     Singletons (%): {np.sum(all_k == 0)/len(all_k)*100:.2f}%")
    print(f"     Median Matches: {np.median(all_k):.1f}")
    print(f"     p99 Matches:    {np.percentile(all_k, 99):.1f}")
    print(f"     Max Matches:    {np.max(all_k)}")
    print(f"   Sanity ranges: {'PASS' if sanity_ok else 'FLAGGED'}")
    if not sanity_ok: all_passed = False
    del match_counts_df, staged_pairs_df, staged_raw_df
    gc.collect()
    
    # -------------------------------------------------------------------------
    # SECTION F: RIGHT FILE, NOT AN OLD ONE
    # -------------------------------------------------------------------------
    print("\n--- SECTION F: RIGHT FILE, NOT AN OLD ONE ---")
    staged_hash = sha256_file(PATH_STAGED_MATCHING)
    staged_size = os.path.getsize(PATH_STAGED_MATCHING)
    print(f"17. Staged file attributes:")
    print(f"    Path:   {PATH_STAGED_MATCHING}")
    print(f"    Size:   {staged_size:,} bytes")
    print(f"    Rows:   {m_stats['data_rows']:,}")
    print(f"    SHA256: {staged_hash}")
    
    # Compare with old matching files
    print("18. Comparing with previous matching results files:")
    for comp_path, label in [(PATH_OLD_MATCHING, "Old France-0.99 (output/matching_results.tsv)"),
                             (PATH_BACKUP_V1_MATCHING, "Original 0.683 submission (output/backup_v1/matching_results.tsv)")]:
        if os.path.exists(comp_path):
            comp_diff_rows = 0
            comp_hash = sha256_file(comp_path)
            with open(PATH_STAGED_MATCHING, "r", encoding="utf-8") as f_staged, open(comp_path, "r", encoding="utf-8") as f_comp:
                for line_s, line_c in zip(f_staged, f_comp):
                    if line_s != line_c:
                        comp_diff_rows += 1
            print(f"    {label}:")
            print(f"      SHA256: {comp_hash}")
            print(f"      Differing rows: {comp_diff_rows:,} / {EXPECTED_ROWS:,} ({comp_diff_rows/EXPECTED_ROWS*100:.2f}%)")
            print(f"      Differs confirmed: {'PASS' if comp_diff_rows > 0 and comp_hash != staged_hash else 'FAIL'}")
            if comp_diff_rows == 0 or comp_hash == staged_hash:
                all_passed = False
        else:
            print(f"    {label}: File not found at {comp_path}")
            
    # -------------------------------------------------------------------------
    # SECTION G: PROBE FILE
    # -------------------------------------------------------------------------
    print("\n--- SECTION G: PROBE FILE ---")
    print("19. Verifying output/probes/probe_france_empty.tsv:")
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
    if not probe_ok: all_passed = False
    
    # -------------------------------------------------------------------------
    # SECTION H: OFFICIAL VALIDATOR
    # -------------------------------------------------------------------------
    print("\n--- SECTION H: OFFICIAL VALIDATOR ---")
    print("20. Running official validator: utils/validate_submission.py...")
    cmd = [
        sys.executable,
        "-X", "utf8",
        "utils/validate_submission.py",
        "--matching", PATH_STAGED_MATCHING,
        "--candidate", PATH_STAGED_CANDIDATES,
        "--test-dir", "dataset/test"
    ]
    val_proc = subprocess.run(cmd, capture_output=True)
    val_stdout = val_proc.stdout.decode("utf-8", errors="replace") if val_proc.stdout else ""
    val_stderr = val_proc.stderr.decode("utf-8", errors="replace") if val_proc.stderr else ""
    print("\n--- Validator Full Output Begin ---")
    if val_stdout:
        print(val_stdout.strip())
    if val_stderr:
        print(val_stderr.strip())
    print("--- Validator Full Output End ---")
    
    val_passed = (val_proc.returncode == 0) and ("PASS" in val_stdout)
    print(f"\nOfficial Validator Result: {'PASS' if val_passed else 'FAIL'}")
    if not val_passed: all_passed = False
    
    print("\n" + "=" * 85)
    print(f"TOTAL AUDIT TIME: {time.time()-start_time:.2f}s")
    print("=" * 85)
    
    if all_passed:
        print("READY TO UPLOAD")
        print(f"File to upload: {PATH_STAGED_MATCHING}")
        print(f"SHA256: {staged_hash}")
    else:
        print("NOT READY")
        print(f"File: {PATH_STAGED_MATCHING}")
        print(f"SHA256: {staged_hash}")

if __name__ == "__main__":
    main()
