"""tools/eval_macro_f05.py
Standalone Macro-averaged per-entity F0.5 evaluation module for Amazon ML Challenge 2026.
Contains zero project-internal imports. Follows docs/problem_statement.md exactly.

Official Metric Definition:
- For EACH Source 1 entity:
    P = |pred ∩ true| / |pred|
    R = |pred ∩ true| / |true|
    F0.5 = (1.25 * P * R) / (0.25 * P + R)
- Singletons & edge cases:
    - true empty AND pred empty -> 1.0
    - true empty AND pred non-empty -> 0.0
    - true non-empty AND pred empty -> 0.0
    - no overlap (tp == 0) -> 0.0
    - perfect match (pred == true) -> 1.0
- Final score = plain unweighted average across ALL Source 1 entities in the set
  (singletons included, zero-candidate entities count as empty predictions).
"""

from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union
import argparse
import sys
import numpy as np
import polars as pl


def compute_entity_f05(
    pred_ids: Union[Set[str], List[str]],
    truth_ids: Union[Set[str], List[str]],
) -> float:
    """Computes F0.5 score for a single Source 1 entity according to official rules."""
    pred_set = set(pred_ids) if not isinstance(pred_ids, set) else pred_ids
    true_set = set(truth_ids) if not isinstance(truth_ids, set) else truth_ids
    np_count = len(pred_set)
    nt_count = len(true_set)

    # Singleton and empty cases
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


def parse_tsv_predictions(tsv_path: Union[str, Path]) -> Tuple[Dict[str, Set[str]], List[str]]:
    """Parses a submission-format or ground-truth TSV file strictly from disk.

    Format: tab-separated, header on line 1, entity_id in column 0, comma-separated IDs in column 1.
    Strips quotes, spaces, empty tokens (',,', leading/trailing comma).
    Returns (mapping: s1 -> set of cids, ordered_list_of_s1).
    """
    tsv_path = Path(tsv_path)
    if not tsv_path.exists():
        raise FileNotFoundError(f"File not found: {tsv_path}")

    s1_map: Dict[str, Set[str]] = {}
    s1_order: List[str] = []

    with open(tsv_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            line_str = line.rstrip("\r\n")
            if not line_str:
                continue
            parts = line_str.split("\t")
            if len(parts) < 1:
                continue
            s1_id = parts[0].strip().strip('"').strip("'")
            if not s1_id:
                continue

            s1_order.append(s1_id)
            if len(parts) >= 2:
                raw_cids = parts[1].strip().strip('"').strip("'")
                if raw_cids:
                    # Clean comma-separated tokens
                    cids = {c.strip() for c in raw_cids.split(",") if c.strip()}
                    s1_map[s1_id] = cids
                else:
                    s1_map[s1_id] = set()
            else:
                s1_map[s1_id] = set()

    return s1_map, s1_order


def evaluate_predictions(
    pred_map: Dict[str, Set[str]],
    truth_map: Dict[str, Set[str]],
    all_s1_entities: List[str],
    s1_country_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Computes comprehensive official evaluation metrics on parsed prediction and truth maps."""
    f05_scores: List[float] = []
    p_scores: List[float] = []
    r_scores: List[float] = []

    country_scores: Dict[str, List[float]] = {}
    country_p: Dict[str, List[float]] = {}
    country_r: Dict[str, List[float]] = {}

    total_tp = 0
    total_pred = 0
    total_true = 0

    true_singletons = 0
    correct_singletons = 0

    for s1 in all_s1_entities:
        p_set = pred_map.get(s1, set())
        t_set = truth_map.get(s1, set())
        np_c = len(p_set)
        nt_c = len(t_set)
        tp = len(p_set & t_set)

        total_tp += tp
        total_pred += np_c
        total_true += nt_c

        # Singleton tracking
        if nt_c == 0:
            true_singletons += 1
            if np_c == 0:
                correct_singletons += 1

        f05 = compute_entity_f05(p_set, t_set)
        f05_scores.append(f05)

        # Per-entity precision & recall (0 if empty/no-overlap, 1.0 if both empty)
        if np_c == 0 and nt_c == 0:
            p_val = 1.0
            r_val = 1.0
        elif np_c == 0 or nt_c == 0 or tp == 0:
            p_val = 0.0
            r_val = 0.0
        else:
            p_val = tp / np_c
            r_val = tp / nt_c

        p_scores.append(p_val)
        r_scores.append(r_val)

        if s1_country_map:
            cntry = s1_country_map.get(s1, "Unknown")
            country_scores.setdefault(cntry, []).append(f05)
            country_p.setdefault(cntry, []).append(p_val)
            country_r.setdefault(cntry, []).append(r_val)

    macro_f05 = float(np.mean(f05_scores)) if f05_scores else 0.0
    micro_p = float(total_tp / total_pred) if total_pred > 0 else 0.0
    micro_r = float(total_tp / total_true) if total_true > 0 else 0.0
    pct_singleton_correct = float(correct_singletons / true_singletons * 100.0) if true_singletons > 0 else 100.0

    per_country: Dict[str, Dict[str, float]] = {}
    if s1_country_map:
        for c in sorted(country_scores.keys()):
            per_country[c] = {
                "macro_f05": float(np.mean(country_scores[c])),
                "precision": float(np.mean(country_p[c])),
                "recall": float(np.mean(country_r[c])),
                "n_entities": len(country_scores[c]),
            }

    return {
        "macro_f05": macro_f05,
        "precision": micro_p,
        "recall": micro_r,
        "macro_precision": float(np.mean(p_scores)) if p_scores else 0.0,
        "macro_recall": float(np.mean(r_scores)) if r_scores else 0.0,
        "true_singletons": true_singletons,
        "correct_singletons": correct_singletons,
        "pct_singleton_correct": pct_singleton_correct,
        "n_entities": len(all_s1_entities),
        "total_tp": total_tp,
        "total_pred": total_pred,
        "total_true": total_true,
        "per_country": per_country,
    }


def evaluate_files(
    pred_tsv: Union[str, Path],
    truth_tsv: Union[str, Path],
    s1_country_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Reads prediction and ground truth TSV files from disk and evaluates official macro F0.5."""
    pred_map, pred_s1_list = parse_tsv_predictions(pred_tsv)
    truth_map, truth_s1_list = parse_tsv_predictions(truth_tsv)

    # Use the union of S1 entities or canonical list from truth
    all_s1 = truth_s1_list if truth_s1_list else pred_s1_list
    return evaluate_predictions(pred_map, truth_map, all_s1, s1_country_map=s1_country_map)


# Backwards compatibility helpers
def macro_f05_from_maps(
    pred_map: Dict[str, Set[str]],
    truth_map: Dict[str, Set[str]],
    all_s1_entities: List[str],
    s1_country_map: Optional[Dict[str, str]] = None,
) -> Tuple[float, Dict[str, float]]:
    res = evaluate_predictions(pred_map, truth_map, all_s1_entities, s1_country_map)
    country_f = {c: data["macro_f05"] for c, data in res["per_country"].items()}
    return res["macro_f05"], country_f


def macro_f05_dataframe(pred: pl.DataFrame, truth: pl.DataFrame, s1c: pl.DataFrame) -> Tuple[float, Dict[str, float]]:
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


# ---------------------------------------------------------------------------
# Unit Test Suite
# ---------------------------------------------------------------------------
def run_unit_tests() -> bool:
    """Executes the complete unit test suite matching the problem statement specification."""
    print("=" * 80)
    print("RUNNING OFFICIAL EVAL_MACRO_F05 UNIT TEST SUITE")
    print("=" * 80)
    tests_passed = True

    # 1. Problem Statement Example
    # Pred: {S2-00047, S2-00193, S3-00812}
    # True: {S2-00047, S3-00812}
    # P = 2/3, R = 1.0, F0.5 = 1.25 * (2/3) * 1.0 / (0.25 * (2/3) + 1.0) = (5/6)/(7/6) = 5/7 = 0.7142857...
    pred_ex = {"S2-00047", "S2-00193", "S3-00812"}
    true_ex = {"S2-00047", "S3-00812"}
    score_ex = compute_entity_f05(pred_ex, true_ex)
    exp_ex = 5.0 / 7.0
    print(f"Test 1 (Problem-Statement Example):")
    print(f"  Pred: {sorted(pred_ex)}")
    print(f"  True: {sorted(true_ex)}")
    print(f"  F0.5 Score = {score_ex:.6f} | Rounded to 3 decimals = {round(score_ex, 3):.3f} (Expected 0.714)")
    assert round(score_ex, 3) == 0.714, f"Test 1 Failed: expected 0.714, got {score_ex}"
    assert abs(score_ex - exp_ex) < 1e-7
    print("  -> PASS")

    # 2. Corner Cases:
    print("\nTest 2 (Corner Cases):")
    # a) empty/empty -> 1.0
    s_ee = compute_entity_f05(set(), set())
    print(f"  a) empty pred / empty true: score = {s_ee:.1f} (Expected 1.0)")
    assert s_ee == 1.0, f"Expected 1.0, got {s_ee}"

    # b) pred non-empty / empty true -> 0.0
    s_pe = compute_entity_f05({"S2-00001"}, set())
    print(f"  b) pred non-empty / empty true: score = {s_pe:.1f} (Expected 0.0)")
    assert s_pe == 0.0, f"Expected 0.0, got {s_pe}"

    # c) empty pred / true non-empty -> 0.0
    s_ep = compute_entity_f05(set(), {"S2-00001"})
    print(f"  c) empty pred / true non-empty: score = {s_ep:.1f} (Expected 0.0)")
    assert s_ep == 0.0, f"Expected 0.0, got {s_ep}"

    # d) no overlap (tp == 0) -> 0.0
    s_no = compute_entity_f05({"S2-00001"}, {"S2-00002"})
    print(f"  d) no overlap (tp == 0): score = {s_no:.1f} (Expected 0.0)")
    assert s_no == 0.0, f"Expected 0.0, got {s_no}"

    # e) perfect match -> 1.0
    s_pf = compute_entity_f05({"S2-00047", "S3-00812"}, {"S2-00047", "S3-00812"})
    print(f"  e) perfect match: score = {s_pf:.1f} (Expected 1.0)")
    assert s_pf == 1.0, f"Expected 1.0, got {s_pf}"
    print("  -> PASS")

    # 3. Tiny 3-Entity File (Hand-Computed Macro Average)
    print("\nTest 3 (Tiny 3-Entity File Macro Average):")
    # Entity 1: Problem statement example -> 5/7 (~0.7142857)
    # Entity 2: Pred {"S2-00001"}, True set() -> 0.0
    # Entity 3: Pred set(), True set() -> 1.0
    # Hand-computed macro average = (5/7 + 0.0 + 1.0) / 3 = (12/7) / 3 = 4/7 = 0.57142857...
    tiny_pred = {
        "S1-1": {"S2-00047", "S2-00193", "S3-00812"},
        "S1-2": {"S2-00001"},
        "S1-3": set(),
    }
    tiny_truth = {
        "S1-1": {"S2-00047", "S3-00812"},
        "S1-2": set(),
        "S1-3": set(),
    }
    tiny_s1 = ["S1-1", "S1-2", "S1-3"]
    tiny_country = {"S1-1": "India", "S1-2": "India", "S1-3": "US"}

    res_tiny = evaluate_predictions(tiny_pred, tiny_truth, tiny_s1, tiny_country)
    expected_macro = 4.0 / 7.0
    print(f"  Entities: S1-1 (F0.5 = 5/7), S1-2 (F0.5 = 0.0), S1-3 (F0.5 = 1.0)")
    print(f"  Evaluated Macro F0.5 = {res_tiny['macro_f05']:.8f} (Hand-computed Expected: {expected_macro:.8f})")
    assert abs(res_tiny["macro_f05"] - expected_macro) < 1e-7, f"Expected {expected_macro}, got {res_tiny['macro_f05']}"

    # Per-country checks:
    # India: (5/7 + 0)/2 = 5/14 = 0.35714286...
    exp_in = (5.0 / 7.0 + 0.0) / 2.0
    # US: 1.0
    exp_us = 1.0
    print(f"  India Macro F0.5 = {res_tiny['per_country']['India']['macro_f05']:.8f} (Expected: {exp_in:.8f})")
    print(f"  US Macro F0.5    = {res_tiny['per_country']['US']['macro_f05']:.8f} (Expected: {exp_us:.8f})")
    assert abs(res_tiny["per_country"]["India"]["macro_f05"] - exp_in) < 1e-7
    assert abs(res_tiny["per_country"]["US"]["macro_f05"] - exp_us) < 1e-7

    # Singleton recall check:
    # True singletons: S1-2, S1-3 (2 total). Correct: S1-3 (1 total). % correct = 50.0%
    print(f"  True Singletons: {res_tiny['true_singletons']} | Correct: {res_tiny['correct_singletons']} ({res_tiny['pct_singleton_correct']:.1f}%)")
    assert res_tiny["true_singletons"] == 2
    assert res_tiny["correct_singletons"] == 1
    assert abs(res_tiny["pct_singleton_correct"] - 50.0) < 1e-7
    print("  -> PASS")

    print("\n" + "=" * 80)
    print("ALL UNIT TESTS PASSED WITH 100% MATHEMATICAL CONFORMANCE")
    print("=" * 80)
    return True


def main():
    parser = argparse.ArgumentParser(description="Official Macro F0.5 Evaluator")
    parser.add_argument("--pred", type=str, help="Path to prediction TSV file")
    parser.add_argument("--truth", type=str, help="Path to ground truth TSV file")
    parser.add_argument("--s1-country", type=str, help="Optional path to S1-country TSV file")
    parser.add_argument("--test", action="store_true", help="Run unit tests")
    args = parser.parse_args()

    if args.test or (len(sys.argv) == 1):
        run_unit_tests()
        if len(sys.argv) == 1:
            return

    if args.pred and args.truth:
        s1_country_map = None
        if args.s1_country:
            df_s1c = pl.read_csv(args.s1_country, separator="\t")
            cols = df_s1c.columns
            s1_col = cols[0]
            c_col = cols[1] if len(cols) > 1 else cols[0]
            s1_country_map = dict(zip(df_s1c[s1_col].to_list(), df_s1c[c_col].to_list()))

        metrics = evaluate_files(args.pred, args.truth, s1_country_map=s1_country_map)
        print(f"\nEvaluation Results for {Path(args.pred).name}:")
        print(f"  Overall Macro F0.5: {metrics['macro_f05']:.5f}")
        print(f"  Precision:          {metrics['precision']:.4f}")
        print(f"  Recall:             {metrics['recall']:.4f}")
        print(f"  Singletons Correct: {metrics['pct_singleton_correct']:.2f}% ({metrics['correct_singletons']:,}/{metrics['true_singletons']:,})")
        print(f"  Entities Evaluated: {metrics['n_entities']:,}")
        if metrics["per_country"]:
            print("  Per Country:")
            for c, data in metrics["per_country"].items():
                print(f"    {c}: Macro F0.5 = {data['macro_f05']:.5f} | P = {data['precision']:.4f} | R = {data['recall']:.4f} (N={data['n_entities']:,})")


if __name__ == "__main__":
    main()
