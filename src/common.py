"""Common constants, paths, splitting logic, and logging helpers for Pipeline v2."""

from pathlib import Path
import re
import time
from typing import Optional, Tuple
import polars as pl
import psutil

# ---------------------------------------------------------------------------
# Project Directories and Canonical Paths
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

DATA_RAW_DIR = PROJECT_ROOT / "student_resource" / "dataset"
TRAIN_RAW_DIR = DATA_RAW_DIR / "train"
TEST_RAW_DIR = DATA_RAW_DIR / "test"

DATA_PREP_DIR = PROJECT_ROOT / "data" / "prep"
DATA_BLOCKS_DIR = PROJECT_ROOT / "data" / "blocks"
DATA_FEATURES_DIR = PROJECT_ROOT / "data" / "features"
DATA_MODELS_DIR = PROJECT_ROOT / "data" / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
REPORTS_DIR = PROJECT_ROOT / "reports"
FALLBACK_DIR = PROJECT_ROOT / "fallback"
TOOLS_DIR = PROJECT_ROOT / "tools"
REFERENCE_DIR = PROJECT_ROOT / "reference"

RUN_LOG_PATH = REPORTS_DIR / "run_log.md"
FRANCE_DIAGNOSIS_PATH = REPORTS_DIR / "france_diagnosis.md"
SUBMISSION_LOG_PATH = REPORTS_DIR / "submission_log.tsv"

# Raw file paths
RAW_FILES = {
    "train_s1": TRAIN_RAW_DIR / "train_source1.tsv",
    "train_s2": TRAIN_RAW_DIR / "train_source2.tsv",
    "train_s3": TRAIN_RAW_DIR / "train_source3.tsv",
    "train_gt": TRAIN_RAW_DIR / "train_ground_truth.tsv",
    "test_s1": TEST_RAW_DIR / "test_source1.tsv",
    "test_s2": TEST_RAW_DIR / "test_source2.tsv",
    "test_s3": TEST_RAW_DIR / "test_source3.tsv",
}

# Prep parquet file paths
PREP_FILES = {
    "train_s1": DATA_PREP_DIR / "train_s1.parquet",
    "train_s2": DATA_PREP_DIR / "train_s2.parquet",
    "train_s3": DATA_PREP_DIR / "train_s3.parquet",
    "test_s1": DATA_PREP_DIR / "test_s1.parquet",
    "test_s2": DATA_PREP_DIR / "test_s2.parquet",
    "test_s3": DATA_PREP_DIR / "test_s3.parquet",
}

# ---------------------------------------------------------------------------
# Split Logic (Deterministic Entity-ID Modulo)
# ---------------------------------------------------------------------------
def split_id(entity_id: str) -> int:
    """Extract integer digits of entity_id and compute modulo 100.
    Maps: 0..79 -> train, 80..89 -> val, 90..99 -> holdout.
    """
    digits = re.sub(r"\D", "", entity_id)
    if not digits:
        return -1
    return int(digits) % 100


def split_expr(col_name: str = "entity_id") -> pl.Expr:
    """Polars expression to extract integer digits of entity_id and compute modulo 100."""
    return pl.col(col_name).str.slice(3).cast(pl.Int64) % 100


def get_split_name(bucket: int) -> str:
    """Returns the split partition name from the bucket integer."""
    if 0 <= bucket <= 79:
        return "train"
    elif 80 <= bucket <= 89:
        return "val"
    elif 90 <= bucket <= 99:
        return "holdout"
    return "unknown"


# ---------------------------------------------------------------------------
# Integer ID Conversion Helpers (Memory Optimization)
# ---------------------------------------------------------------------------
def s1_to_num(entity_id: str) -> int:
    """Extract integer digits from S1 entity_id ('S1-12345' -> 12345)."""
    return int(entity_id[3:])


def num_to_s1(s1_num: int) -> str:
    """Reconstruct S1 entity_id from integer (12345 -> 'S1-12345')."""
    return f"S1-{s1_num}"


def cand_to_num_src(cand_id: str) -> Tuple[int, int]:
    """Convert candidate ID ('S2-12345' or 'S3-67890') to (cand_num: int64, cand_src: int8)."""
    return int(cand_id[3:]), int(cand_id[1])


def num_src_to_cand(cand_num: int, cand_src: int) -> str:
    """Reconstruct candidate ID from integer pair (12345, 2 -> 'S2-12345')."""
    return f"S{cand_src}-{cand_num}"


def test_id_round_trip() -> bool:
    """Verifies round-trip fidelity across S1, S2, and S3 entity IDs."""
    samples = [
        ("S1-1", 1),
        ("S1-925783039", 925783039),
        ("S2-4887273", 4887273, 2),
        ("S3-5082316", 5082316, 3),
    ]
    for item in samples[:2]:
        s1_id, exp_num = item
        num = s1_to_num(s1_id)
        assert num == exp_num, f"s1_to_num failed: {num} != {exp_num}"
        assert num_to_s1(num) == s1_id, f"num_to_s1 failed: {num_to_s1(num)} != {s1_id}"

    for item in samples[2:]:
        cand_id, exp_num, exp_src = item
        num, src = cand_to_num_src(cand_id)
        assert num == exp_num and src == exp_src, f"cand_to_num_src failed on {cand_id}"
        assert num_src_to_cand(num, src) == cand_id, f"num_src_to_cand failed on {cand_id}"

    return True


# ---------------------------------------------------------------------------
# Logging & Timing Helpers
# ---------------------------------------------------------------------------
def append_to_run_log(content: str) -> None:
    """Appends markdown content to reports/run_log.md atomically."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUN_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(content)
        if not content.endswith("\n"):
            f.write("\n")


class Timer:
    """Context manager for measuring elapsed execution time."""
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.start_time


def get_system_hardware_info() -> dict:
    """Queries hardware specs (CPU cores, total/free RAM, free disk space)."""
    cpu_cores = psutil.cpu_count(logical=True)
    vm = psutil.virtual_memory()
    total_ram_gb = round(vm.total / (1024 ** 3), 2)
    free_ram_gb = round(vm.available / (1024 ** 3), 2)
    
    # Check disk containing PROJECT_ROOT
    disk = psutil.disk_usage(str(PROJECT_ROOT))
    free_disk_gb = round(disk.free / (1024 ** 3), 2)
    total_disk_gb = round(disk.total / (1024 ** 3), 2)

    return {
        "cpu_cores": cpu_cores,
        "total_ram_gb": total_ram_gb,
        "free_ram_gb": free_ram_gb,
        "free_disk_gb": free_disk_gb,
        "total_disk_gb": total_disk_gb,
    }
