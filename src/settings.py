from pathlib import Path
import os

from loguru import logger

# ──────────────────────────────────────────────
# Data settings
# ──────────────────────────────────────────────
BASE_DIR=Path(__file__).resolve().parent.parent
DATA_DIR=BASE_DIR / "data"
RAW_DATA_DIR=DATA_DIR / "raw"
GENRES_DATA_DIR=RAW_DATA_DIR / "topics"
PROCESSED_DATA_DIR=DATA_DIR / "processed"
CHECKPOINTS_DIR=BASE_DIR / "checkpoints"
LOGS_DIR=BASE_DIR / "logs"
REPORTS_DIR=BASE_DIR / "reports"