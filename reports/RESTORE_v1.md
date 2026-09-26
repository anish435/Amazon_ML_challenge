# RESTORE_v1: Emergency Rollback Procedure for v1 Submission

This document outlines the exact, step-by-step commands to completely restore the **v1 (LB = 0.932)** pipeline and submission files from the verified backup.

---

## 1. Primary Submission File Information (v1 Fallback)

- **File to Upload**: `output/v1/matching_results.tsv`
- **Candidate File**: `output/v1/candidate_pairs.tsv`
- **SHA-256 Checksum (`matching_results.tsv`)**: `a00b60958e2b4cb7f2ff61d33b6e34899b1bebc5f4df1e3585dff2a0482813ae`
- **SHA-256 Checksum (`candidate_pairs.tsv`)**: `5fc22dd160d2757bfca8dbffa13e36f604dee36309e684f2bc3a394bbce49b70`
- **Public Leaderboard Score**: **0.932** (Current Best Benchmark)
- **Validation Macro $F_{0.5}$**: **0.94359** | **Holdout Macro $F_{0.5}$**: **0.94528**
- **Decision Threshold ($\tau$)**: `0.67` across France, India, and US
- **Top Candidate Pruning ($K$)**: `8` candidates/entity ($6.45\times$ compression)

---

## 2. Step-by-Step Restoration Commands

### Step A: Restore Codebase via Git Tag
Execute the following in the repository root (`d:\Projects\Amazon_ML_Mark2`):

```powershell
# Fetch and switch to the immutable v1 tag
git checkout v1-lb0932
```

### Step B: Restore Output and Model Artifacts from Backup
Copy the verified backup directory back into the active tree:

```powershell
# Copy outputs back
New-Item -ItemType Directory -Force -Path "output/v1"
Copy-Item -Path "backups/v1_lb0932/output/v1/*" -Destination "output/v1/" -Recurse -Force

# Copy models and validation predictions back
New-Item -ItemType Directory -Force -Path "data/models"
Copy-Item -Path "backups/v1_lb0932/data/models/*" -Destination "data/models/" -Recurse -Force

# Copy test predictions back
New-Item -ItemType Directory -Force -Path "data/preds"
Copy-Item -Path "backups/v1_lb0932/data/preds/*" -Destination "data/preds/" -Recurse -Force

# Copy test vectorizers back
New-Item -ItemType Directory -Force -Path "data/features"
Copy-Item -Path "backups/v1_lb0932/data/features/*" -Destination "data/features/" -Recurse -Force
```

### Step C: Verify Checksums Against Manifest
Run the independent SHA-256 verification:

```powershell
.\.venv\Scripts\python.exe -c "
import hashlib
from pathlib import Path

def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while chunk := f.read(8*1024*1024):
            h.update(chunk)
    return h.hexdigest()

expected_match = 'a00b60958e2b4cb7f2ff61d33b6e34899b1bebc5f4df1e3585dff2a0482813ae'
actual_match = sha256('output/v1/matching_results.tsv')
assert actual_match == expected_match, f'Checksum mismatch: {actual_match} != {expected_match}'
print('VERIFIED: output/v1/matching_results.tsv is 100% bit-for-bit identical to v1!')
"
```

### Step D: Run Validation Audit
Verify submission integrity:

```powershell
.\.venv\Scripts\python.exe tools/preupload_check.py
```
Expected result: `>>> ALL CHECKS PASSED: READY TO UPLOAD <<<` (0 errors, 0 warnings).
