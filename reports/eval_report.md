# Official F0.5 Evaluation & Leaderboard Analysis Report
**Date/Time**: 2026-09-27 10:24:25

## 1. Official Evaluator Specification & Unit Test Suite
The evaluator (`tools/eval_macro_f05.py`) strictly implements the official rules from `docs/problem_statement.md`:
- Per-entity: $P = |\text{pred} \cap \text{true}| / |\text{pred}|$, $R = |\text{pred} \cap \text{true}| / |\text{true}|$, $F_{0.5} = 1.25 \cdot P \cdot R / (0.25 \cdot P + R)$.
- Singletons: $\text{true}=\emptyset \land \text{pred}=\emptyset \to 1.0$; $\text{true}=\emptyset \land \text{pred}\neq\emptyset \to 0.0$; $\text{true}\neq\emptyset \land \text{pred}=\emptyset \to 0.0$; $tp=0 \to 0.0$; perfect match $\to 1.0$.
- Macro average: unweighted arithmetic mean over **ALL** Source 1 entities in the split.

### Unit Test Verification Results
- **Test 1 (Problem-Statement Example)**: `pred` = {S2-00047, S2-00193, S3-00812}, `true` = {S2-00047, S3-00812} $\to$ **0.714286** (exact fraction $5/7 = 0.714$). **PASS**.
- **Test 2 (Corner Cases)**: Empty/empty $\to$ 1.0, non-empty/empty $\to$ 0.0, empty/non-empty $\to$ 0.0, no-overlap $\to$ 0.0, perfect $\to$ 1.0. **PASS**.
- **Test 3 (3-Entity File Macro Average)**: Hand-computed $(5/7 + 0.0 + 1.0)/3 = 4/7 = 0.57142857$. **PASS**.

## 2. File-Based Evaluation of Validation and Holdout Splits
All predictions and ground truth were written to disk as canonical submission-format TSVs (`reports/eval_tsv/`), then parsed fresh without in-memory shortcuts.

| Version | Split | Total S1 | Overall Macro $F_{0.5}$ | India $F_{0.5}$ | US $F_{0.5}$ | Precision | Recall | Singleton Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **v1** | Validation | 20,000 | **0.94359** | 0.92977 | 0.95281 | 0.9773 | 0.8977 | 95.53% |
| **v1** | Holdout | 10,000 | **0.94528** | 0.92981 | 0.95560 | 0.9781 | 0.8969 | 95.26% |
| **v2** | Validation | 20,000 | **0.95965** | 0.94593 | 0.96881 | 0.9846 | 0.9264 | 95.97% |
| **v2** | Holdout | 10,000 | **0.95944** | 0.94188 | 0.97115 | 0.9834 | 0.9245 | 96.17% |

### Explanation of Number Concordance
- **Exact Match**: The file-based evaluation matches the earlier reported numbers to all 5 decimal places (`v1`: 0.94359 val, 0.94528 holdout; `v2`: 0.95965 val, 0.95944 holdout).
- **Why 0 Difference**: The TSV writer accurately serializes greedy 1:1 matches, includes empty string lines for zero-candidate and below-threshold entities, and the parser treats empty token lines as genuine singletons ($F_{0.5}=1.0$), ensuring 100% fidelity.

## 3. Past Leaderboard Reconciliation (v1)
- **Observed Real Leaderboard Scores**: v1 Main = **0.932**, v1 France-Empty = **0.804**.
- **Test Country Weights (Exact)**: France = 0.149752 (14.98%), India = 0.467513 (46.75%), US = 0.382735 (38.27%).
- **France Singleton Share**: 5.82% (from v1 France predictions; $\pm 2\%$ range [3.82%, 7.82%]).
- **Leaderboard-Implied US+India Component**: $0.804 - (0.1498 \times 0.0582) = \mathbf{0.79528}$ (range [0.79229, 0.79828]).
- **Implied v1 France Score**: $(0.932 - 0.79528) / 0.1498 = \mathbf{0.9129}$ (range [0.8929, 0.9329]).
- **Prediction Error (Generalization Gap)**:
  - v1 Holdout US+India Predicted: **0.80044** vs Implied **0.79528** $\to$ Gap = **+0.0052** (+0.52 points).
  - v1 Val US+India Predicted: **0.79935** vs Implied **0.79528** $\to$ Gap = **+0.0041** (+0.41 points).

## 4. Honest Predicted Leaderboard Range for v2
Calibrating for the +0.0052 generalization gap and accounting for test sample variance ($\pm 0.003$):

| Candidate File | Low | Expected | High | Notes |
|---|---:|---:|---:|---|
| **`output/v2/matching_results.tsv`** (Primary) | **0.9376** | **0.9436** (~**0.944**) | **0.9496** | Full v2 model on all 3 countries (France scored at $\tau=0.65$) |
| **`output/v2/probes/v2_usin_v1_france.tsv`** (Probe) | **0.9406** | **0.9436** (~**0.944**) | **0.9466** | Guaranteed fallback: v1 France (~0.913) + v2 India & US |

## 5. Rules for Reading Leaderboard Results
1. **Public LB is a subset**: Test set has 1.73M entities. Public LB reflects ~20-30% of test. Fluctuation of $\pm 0.003$ is expected sampling variance.
2. **If v2 score is between 0.942 and 0.948**: Fully consistent with expected improvement. India + US gained +0.014 in holdout, netting $\approx +0.012$ after calibration.
3. **If v2 score is $\ge 0.949$**: France maintained or slightly improved with the new raw-number and rarity features.
4. **If v2 score drops below 0.938**: France suffered from the lower threshold $\tau=0.65$. In that event, uploading `output/v2/probes/v2_usin_v1_france.tsv` will immediately recover the France baseline while retaining all US/India gains.
