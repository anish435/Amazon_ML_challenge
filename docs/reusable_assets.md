# Reusable Pipeline Assets & Artifact Catalog (v1 $\rightarrow$ v2)

This catalog details which existing code modules, cached artifacts, and tools are verified safe for reuse in v2, and which must be discarded or refactored.

---

## 1. Verified Safe to Reuse in v2

| Asset | Path | Justification |
|---|---|---|
| **Text Normalization Engine** | [`code/business_entity_resolution/src/preprocess.py`](file:///d:/Projects/Amazon_ML_challenge/code/business_entity_resolution/src/preprocess.py) | High-throughput (~140k rows/s) multilingual normalizer; preserves Indic matras while stripping Latin diacritics and legal suffixes with 0.49% safe collision rate. |
| **Feature Extraction Engine** | [`code/business_entity_resolution/src/features.py`](file:///d:/Projects/Amazon_ML_challenge/code/business_entity_resolution/src/features.py) | 48 pairwise similarity features; verified with 0.00000000 max difference against streaming inference and completely free of categorical country encoding. |
| **Multi-Pass Blocking V3 Engine** | [`code/business_entity_resolution/src/blocking.py`](file:///d:/Projects/Amazon_ML_challenge/code/business_entity_resolution/src/blocking.py) | Achieves 89.33% recall on the full 10.32M pool using direct CSR array indexing and argpartition (0.95s per 10k entities). |
| **Fitted Train Blocking Caches** | `artifacts/cache/engine_train_India.pkl`<br>`artifacts/cache/engine_train_US.pkl` | Pre-fitted BlockingEngine V3 instances across the full 10.32M training pool; serialised via uncompressed pickle protocol 5 for instant startup. |
| **Preprocessed Train Candidate Tables** | `artifacts/cache/cand_prep_train_India.parquet`<br>`artifacts/cache/cand_prep_train_US.parquet` | Clean, tokenized, and normalized representations of all 10.32M Source 2 and Source 3 training records; eliminates redundant preprocessing time. |
| **Competitor Simulation Module** | [`scripts/run_competitor_simulation.py`](file:///d:/Projects/Amazon_ML_challenge/scripts/run_competitor_simulation.py) | Accurately models open-pool S1 competition on validation sets, replicating test-level contested rates (~18%). |
| **Clean Feature Extraction Script** | [`scripts/build_realistic_pairs.py`](file:///d:/Projects/Amazon_ML_challenge/scripts/build_realistic_pairs.py) | Reusable pipeline for generating realistic non-injected candidate feature matrices against the full 10.32M pool. |
| **Production Submission Builder** | [`scripts/build_submission.py`](file:///d:/Projects/Amazon_ML_challenge/scripts/build_submission.py) | Memory-efficient builder applying 1:1 bipartite matching and rapidfuzz string-similarity tie breaking with 0.00% contested pairs remaining. |
| **Independent Pre-Upload Validator** | [`scripts/preupload_check.py`](file:///d:/Projects/Amazon_ML_challenge/scripts/preupload_check.py) | High-speed, read-only 20-point verification script (executes in < 3 minutes) auditing raw byte encoding, row coverage, and config consistency. |
| **Official Competition Validator** | [`utils/validate_submission.py`](file:///d:/Projects/Amazon_ML_challenge/utils/validate_submission.py) | Ground-truth verification tool checking schema, row cardinality, singleton format, and candidate-pair subset constraints. |

---

## 2. NOT Safe to Reuse in v2 (Must be Discarded or Re-engineered)

| Asset | Path | Root Cause & Why It Must Not Be Reused |
|---|---|---|
| **Legacy Model Checkpoint** | `artifacts/final_model.joblib` | Trained on an easy ~248k candidate pool with injected true matches and ad hoc `scale_pos_weight`; never learned to reject hard look-alike negatives. |
| **Legacy Training Script** | `scripts/train_final_model.py` | Contains hardcoded `head(15000)` S1 slicing, `head(100000)` background sampling, and lacks validation/probability calibration loops. |
| **Unversioned Output Submissions** | `output/matching_results.tsv`<br>`output/backup_v1/matching_results.tsv` | Overwritten by ad hoc scripts without version control; represents contaminated states that do not match current clean post-processing logic. |
| **Inference Stage 7 Checkpoints** | `output/temp_candidates_*.tsv`<br>`output/temp_matches_*.tsv` | Contain a 747-line desynchronization gap caused by non-atomic checkpoint resumption during the test inference run. |
| **Truncated Test Probabilities** | `artifacts/final_test_probabilities.csv` | Written with 4-decimal rounding (`{p:.4f}`) and truncated at $p < 0.50$, introducing 315k artificial ties and preventing sub-0.50 threshold optimization. |
| **Unpinned Requirements File** | `requirements.txt` | Lacks core pipeline packages (`lightgbm`, `rapidfuzz`, `joblib`) and uses unpinned `>=` bounds, posing migration and reproducibility risks. |
| **Stale Architectural Documentation** | `README.md` | Outdated documentation that halts at Phase 3 (Blocking), missing features, training, inference, and post-processing architecture. |
