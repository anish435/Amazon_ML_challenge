# Non-Negotiable Rules for Pipeline v2

1. **Exact Metric Alignment**: Optimize, tune, and evaluate strictly on the official macro-averaged per-entity {0.5}$ metric with the official singleton penalty.
2. **Full-Pool Realism**: Fit candidate blocking on the complete 10.32M Source 2/3 pool without artificial downsampling or true-match injection.
3. **No Biased Sampling**: Never use .head() for sampling; use seeded, stratified random sampling (seed=42) across the full dataset.
4. **Scale Training Size**: Train on at least 50k–100k stratified S1 entities to ensure robust negative coverage and rare-token diversity.
5. **Competitor-Aware Validation**: Validation must include simulated competitor competition from the full S1 pool to replicate test contested rates.
6. **Fingerprint Verification Gate (R6)**: Validation must match test on measurable fingerprints (candidates/entity ~55–65, contested rate ~15–20%, avg matches ~3.0–3.5, % singletons ~5–12%, p99 < 10) on the same model and config before any tuning.
7. **Entity-ID Hash Splitting**: Define all train/validation splits by deterministic hashing of S1 entity IDs (hash(entity_id) % 10), never by arbitrary file row slices.
8. **Open-Set Country Protocol**: Systematically identify unseen countries during EDA; build cross-lingual normalizers and dynamic adaptation mechanisms rather than hardcoded dicts.
9. **No Ad Hoc Loss Weights**: Never use arbitrary scale_pos_weight formulas; calibrate output probabilities using Platt scaling or isotonic regression on validation folds.
10. **Zero Direct Country Features**: Maintain purely relational country features (country_exact, country_missing); never feed raw country identifiers as categorical inputs.
11. **Lossless Probability Storage**: Write ALL evaluated candidate pairs with unrounded full float32/float64 precision and never truncate  < 0.50$.
12. **Atomic Checkpoint Synchronization**: Ensure candidate TSV, match TSV, and probability CSV files are written and truncated atomically during batch processing and resumption.
13. **High-Powered Tie Validation**: Validate tie-breaking policies on datasets with at least 10,000 contested candidate instances before deploying to test.
14. **Direct Array Computations**: Perform vector operations and top-K extraction using direct CSR buffer indexing (indptr, indices, data) and 
p.argpartition; ban per-row slicing objects.
15. **Uncompressed Fast Index Caching**: Cache precomputed indexes using raw uncompressed pickle protocol 5 or memory-mapped arrays/parquet; never use slow compression on large in-memory objects.
16. **Immutable Versioned Builds**: Never overwrite files in output/; every submission build must write to an immutable versioned directory (output/v{X}/).
17. **Central Submission Ledger**: Maintain an append-only eports/submission_log.tsv recording git commit, script name, hyperparameter config, SHA256 hashes, and leaderboard scores.
18. **Independent Pre-Upload Verification**: Run an independent, read-only verification script and the official competition validator before uploading any file.
19. **Strict Network Isolation**: Maintain zero external API or network dependencies to guarantee 100% compliance with competition rules.
20. **Strictly Pinned Dependencies**: Pin every dependency with exact versions (==) in equirements.txt and keep documentation synchronized with executable code.
21. **Performance Discrepancy Halt Gate (R21)**: If validation score and leaderboard score differ by more than 3 percentage points under the same configuration, STOP and investigate root cause rather than tuning through the gap.
22. **Country Parameters with Strict Fallback (R22)**: If country-specific thresholds or rules are used, they must have an automated data-driven derivation and a strict, verified default fallback for any unseen country.
23. **Skew Tests on True Matches (R23)**: Run automated train/serve skew unit tests asserting zero difference (max absolute diff == 0.0) between offline feature extraction and real-time inference on true matches and hard negatives.
