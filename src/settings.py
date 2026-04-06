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
REPORTS_DIR=BASE_DIR / "report"

for dir in [DATA_DIR, RAW_DATA_DIR, GENRES_DATA_DIR, PROCESSED_DATA_DIR, CHECKPOINTS_DIR, LOGS_DIR, REPORTS_DIR]:
    dir.mkdir(parents=True, exist_ok=True)





# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Log level
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()

# Log file paths
LOG_FILE = LOGS_DIR / 'app.log'
ERROR_LOG_FILE = LOGS_DIR / 'error.log'

# Log rotation
LOG_ROTATION = os.getenv('LOG_ROTATION', '10 MB')
LOG_RETENTION = os.getenv('LOG_RETENTION', '30 days')

# Log format
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)




from loguru import logger

logger.remove()
logger.add(
    LOG_FILE,
    level=LOG_LEVEL,
    rotation=LOG_ROTATION,
    retention=LOG_RETENTION,
    format=LOG_FORMAT
)