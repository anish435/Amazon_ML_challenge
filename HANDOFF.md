# Pipeline v2 Handoff Specification & Manifest
**Generated**: 2026-09-26  
**Destination Root**: `D:/Projects/v2_handoff/`  
**Origin Workspace**: `D:/Projects/Amazon_ML_challenge/` (Preserved in strict READ-ONLY mode)

---

## 1. Directory Structure & Lineage

| Handoff File Path | Source in Origin Workspace | Purpose & Contents |
|---|---|---|
| `Documentation_template.md` | `datasets/student_resource/Documentation_template.md` | Official Amazon ML Challenge methodology write-up template required in final submission zip. |
| `utils/validate_submission.py` | `utils/validate_submission.py` | Official competition validator script checking schema, entity row count, singleton rules, and candidate subset constraints. |
| `docs/problem_statement.md` | `datasets/student_resource/README.md` | Full official challenge description, file schemas, evaluation metric definition, and submission guidelines. |
| `docs/lessons_learned.md` | `reports/lessons_learned.md` | Comprehensive post-mortem matrix detailing all 28 project mistakes, root cause analyses, numerical evidence, and v2 prevention rules. |
| `docs/v2_rules.md` | `reports/v2_rules.md` (with amendments R6, R21, R22, R23) | Non-negotiable operational checklist for v2 engineering (metric alignment, full-pool realism, fingerprint gating, atomic checkpoints). |
| `docs/reusable_assets.md` | `reports/reusable_assets.md` | Verified reusable components catalog (preprocess, features, blocking) and explicit blacklist of invalid artifacts to discard. |
| `docs/key_facts.md` | `docs/key_facts.md` (Newly compiled & verified) | Concrete verified numbers: entity distributions, ground truth distribution parameters, 1:1 law, pool sizes, and tie statistics. |
| `tools/preupload_check.py` | `scripts/preupload_check.py` | Standalone, read-only 20-point verification script validating raw byte encoding, CRLF, row coverage, and probability adherence in < 3 min. |
| `tools/eval_macro_f05.py` | `tools/eval_macro_f05.py` (Newly compiled standalone tool) | Self-contained macro-averaged per-entity F0.5 evaluator with self-tests for the problem-statement example (0.714) and singletons (1.0 / 0.0). |
| `reference/old_preprocess.py` | `code/business_entity_resolution/src/preprocess.py` | Reference text normalization module (~140k rows/s) with Latin diacritic removal and legal suffix stripping (read-only reference). |
| `reference/old_features.py` | `code/business_entity_resolution/src/features.py` | Reference 48-feature candidate-pair comparison module mathematically verified against inference (read-only reference). |
| `reference/old_blocking.py` | `code/business_entity_resolution/src/blocking.py` | Reference multi-pass candidate generation engine implementing Phase 6 Blocking V3 with CSR indexing (read-only reference). |
| `fallback/matching_results_0706_france_empty.tsv` | `output/probes/probe_france_empty.tsv` | Current best competition submission file (Leaderboard score: 0.706). |
| `fallback/matching_results_0697.tsv` | `output/staging/matching_results.tsv` | Calibrated 1:1 submission file with rapidfuzz tie resolution (Leaderboard score: 0.697). |
| `reports/submission_log.tsv` | `reports/submission_log.tsv` (Newly compiled ledger) | Central ledger tracking date, file name, SHA256 hash, hyperparameter configuration, and public leaderboard scores. |
| `HANDOFF.md` | `HANDOFF.md` | This document: comprehensive file origin specification and cryptographic SHA256 manifest. |

---

## 2. Cryptographic SHA256 Manifest
| File | Size (Bytes) | SHA256 Checksum |
|---|---:|---|
| `Documentation_template.md` | 2,175 | `599b661af92a8d5c239ba283ee9b2ffed050c0bc86360964430a2f9857e8c599` |
| `docs\key_facts.md` | 4,105 | `10e85de14495a5d9a833dc2cc8995f36a8752813ef02e299e790244566cfd013` |
| `docs\lessons_learned.md` | 13,276 | `a2a493088dea5815df22f86d1a3a85101eb6442d8b53d49316ac3bd433c24c5d` |
| `docs\problem_statement.md` | 14,130 | `be67cd7d9dbc48a58f93e1f61402360655debac64214115bf9d27d3301939285` |
| `docs\reusable_assets.md` | 4,888 | `fa1507fc148f0078c7616e573ab4e456162ec15548a8fa7f29dc1aa0d08c78f0` |
| `docs\v2_rules.md` | 4,082 | `0727b3ba82afa7aa20624dadc2a751d25cd64a15824267cbf3ea3b4d114b996c` |
| `fallback\matching_results_0697.tsv` | 88,687,642 | `21f2f12f3defa53c9e6270c751eb1642e5b125e167969300e22419f425398cd0` |
| `fallback\matching_results_0706_france_empty.tsv` | 78,339,734 | `4dc26a4ba0f8bab25b08a3ae5e008295fa790c8fa1959779fba05021e45e1f25` |
| `reference\old_blocking.py` | 26,874 | `78b7ba6b9d604c83ce49e2b464d9baffde2a33108f9f7e09b2ac5a7fdcc6d692` |
| `reference\old_features.py` | 13,066 | `cd68193a39c2ac4206933c1108d69f92e14c57abed1291687f1a1882ea9860d5` |
| `reference\old_preprocess.py` | 12,527 | `7a115cb1e41e3083b4f20564cf2487c14492134b9d9c4d6df71bc7494329a1ec` |
| `reports\submission_log.tsv` | 860 | `a7590bd7ab40b32b2d842ac744a62f98ea75367bdd9b23827854fe25cf203a24` |
| `tools\eval_macro_f05.py` | 6,056 | `e4daa4607b8b110edea3091692bee42281bd94a0543af24c0d07571425261c07` |
| `tools\preupload_check.py` | 31,465 | `f5c07f47d31473f7d7400b672b27ff4b3dbfe021f08c8b988bd3fec05ecb5b2d` |
| `utils\validate_submission.py` | 13,687 | `f96f59934383a15095914f507620c078c474e8f9c60e864b2d051ed173a22dfc` |
