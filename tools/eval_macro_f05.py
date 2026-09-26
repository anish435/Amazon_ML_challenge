"""
tools/eval_macro_f05.py
Standalone Macro-averaged per-entity F0.5 evaluation module for Amazon ML Challenge 2026.
Contains zero project-internal imports.

Metric Definition:
F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
Calculated per Source 1 entity, then averaged across ALL Source 1 entities.
Singletons:
- True empty, predicted empty -> 1.0
- True empty, predicted non-empty -> 0.0
- True non-empty, predicted empty -> 0.0
"""

from typing import Dict, List, Set, Tuple, Optional, Any, Union
import polars as pl
import numpy as np


def compute_entity_f05(pred_ids: Union[Set[str], List[str]], truth_ids: Union[Set[str], List[str]]) -> float:
    """
    Computes F0.5 score for a single Source 1 entity.
    """
    pred_set = set(pred_ids)
    true_set = set(truth_ids)
    np_count = len(pred_set)
    nt_count = len(true_set)
    
    # Singleton cases
    if np_count == 0 and nt_count == 0:
        return 1.0
    if np_count == 0 or nt_count == 0:
        return 0.0
        
    tp = len(pred_set & true_set)
    if tp == 0:
        return 0.0
        
    precision = tp / np_count
    recall = tp / nt_count
    denom = 0.25 * precision + recall
    return (1.25 * precision * recall) / denom if denom > 0 else 0.0


def macro_f05_from_maps(
    pred_map: Dict[str, Set[str]],
    truth_map: Dict[str, Set[str]],
    all_s1_entities: List[str],
    s1_country_map: Optional[Dict[str, str]] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Computes macro-averaged F0.5 across all S1 entities and per country.
    """
    f_scores = []
    country_scores = {}
    
    if s1_country_map:
        country_groups = {}
        for s1 in all_s1_entities:
            cntry = s1_country_map.get(s1, "UNKNOWN")
            country_groups.setdefault(cntry, []).append(s1)
            
    for s1 in all_s1_entities:
        p_set = pred_map.get(s1, set())
        t_set = truth_map.get(s1, set())
        f_scores.append(compute_entity_f05(p_set, t_set))
        
    overall_macro = float(np.mean(f_scores)) if f_scores else 0.0
    
    if s1_country_map:
        for cntry, s1_list in country_groups.items():
            c_scores = [compute_entity_f05(pred_map.get(s1, set()), truth_map.get(s1, set())) for s1 in s1_list]
            country_scores[cntry] = float(np.mean(c_scores)) if c_scores else 0.0
            
    return overall_macro, country_scores


def macro_f05_dataframe(pred: pl.DataFrame, truth: pl.DataFrame, s1c: pl.DataFrame) -> Tuple[float, Dict[str, float]]:
    """
    Polars DataFrame implementation directly matching diagnose_postproc.py.
    Columns expected:
      - pred: ['s1', 'cid']
      - truth: ['s1', 'cid']
      - s1c: ['s1', 'country']
    """
    p = pred.select(["s1", "cid"]).unique()
    t = truth.select(["s1", "cid"]).unique()
    tp = p.join(t, on=["s1", "cid"]).group_by("s1").agg(tp=pl.len())
    
    e = (s1c.join(p.group_by("s1").agg(np=pl.len()), on="s1", how="left")
            .join(t.group_by("s1").agg(nt=pl.len()), on="s1", how="left")
            .join(tp, on="s1", how="left")
            .with_columns(pl.col("np", "nt", "tp").fill_null(0)))
            
    pr = pl.col("tp") / pl.col("np")
    rc = pl.col("tp") / pl.col("nt")
    
    e = e.with_columns(
        f=pl.when((pl.col("np") == 0) & (pl.col("nt") == 0)).then(1.0)
            .when(pl.col("tp") == 0).then(0.0)
            .otherwise(1.25 * pr * rc / (0.25 * pr + rc))
    )
    
    by = {r["country"]: r["f"] for r in e.group_by("country").agg(pl.col("f").mean()).iter_rows(named=True)}
    return float(e["f"].mean()), by


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING EVAL_MACRO_F05 SELF-TEST SUITE")
    print("=" * 70)
    
    # 1. Problem Statement Example
    # Pred: S1-00001 -> [S2-00047, S2-00193, S3-00812]
    # Truth: S1-00001 -> [S2-00047, S3-00812]
    # Expected: Precision = 2/3, Recall = 1.0, F0.5 = 0.714
    pred_ex = ["S2-00047", "S2-00193", "S3-00812"]
    truth_ex = ["S2-00047", "S3-00812"]
    score_ex = compute_entity_f05(pred_ex, truth_ex)
    print(f"Test 1 (Problem Statement Example): score = {score_ex:.4f} (expected ~0.7143)")
    assert round(score_ex, 3) == 0.714, f"Expected 0.714, got {score_ex}"
    print("  -> PASS")
    
    # 2. Singleton Cases
    # empty/empty -> 1.0
    s_ee = compute_entity_f05([], [])
    print(f"Test 2a (empty pred, empty truth): score = {s_ee:.1f} (expected 1.0)")
    assert s_ee == 1.0, f"Expected 1.0, got {s_ee}"
    print("  -> PASS")
    
    # any/empty -> 0.0
    s_ae = compute_entity_f05(["S2-00001"], [])
    print(f"Test 2b (non-empty pred, empty truth): score = {s_ae:.1f} (expected 0.0)")
    assert s_ae == 0.0, f"Expected 0.0, got {s_ae}"
    print("  -> PASS")
    
    # empty/any -> 0.0
    s_ea = compute_entity_f05([], ["S2-00001"])
    print(f"Test 2c (empty pred, non-empty truth): score = {s_ea:.1f} (expected 0.0)")
    assert s_ea == 0.0, f"Expected 0.0, got {s_ea}"
    print("  -> PASS")
    
    # 3. Macro-average over multiple entities via DataFrame
    pred_df = pl.DataFrame({
        "s1": ["S1-1", "S1-1", "S1-1", "S1-2"],
        "cid": ["S2-00047", "S2-00193", "S3-00812", "S2-99999"]
    })
    truth_df = pl.DataFrame({
        "s1": ["S1-1", "S1-1"],
        "cid": ["S2-00047", "S3-00812"]
    })
    s1c_df = pl.DataFrame({
        "s1": ["S1-1", "S1-2", "S1-3"],
        "country": ["India", "India", "US"]
    })
    overall_m, by_c = macro_f05_dataframe(pred_df, truth_df, s1c_df)
    expected_m = (5/7 + 0.0 + 1.0) / 3
    print(f"Test 3 (DataFrame Macro Average): score = {overall_m:.4f} (expected {expected_m:.4f})")
    assert abs(overall_m - expected_m) < 1e-6, f"Expected {expected_m}, got {overall_m}"
    print(f"  By Country: {by_c}")
    print("  -> PASS")
    
    print("=" * 70)
    print("ALL SELF-TESTS PASSED SUCCESSFULLY (100% SPECIFICATION CONFORMANCE)")
    print("=" * 70)
