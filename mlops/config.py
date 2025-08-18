# mlops/config.py
from __future__ import annotations
from pathlib import Path
import os

# Fix for Docker: Use environment variable or fallback to app directory
if os.environ.get('AIRFLOW__CORE__EXECUTOR'):  # Running in Airflow Docker
    ROOT = Path('/opt/airflow/app')
else:  # Local development
    ROOT = Path(__file__).resolve().parents[1]

# Data paths
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "train.csv"
TRAIN_CSV = DATA_DIR / "train_split.csv"
TEST_CSV = DATA_DIR / "test_split.csv"
AUG_DIR = DATA_DIR / "augmented"
AUG_CSV = AUG_DIR / "train_aug.csv"

# MLflow local file store (no server needed)
MLRUNS_DIR = ROOT / "mlruns"  # used as file:// URI

# Train/Test split configuration
TEST_SIZE = 0.001  # 0.1% for test (very small as requested)
TRAIN_SIZE = 1 - TEST_SIZE

# Augmentation & IO
CHUNK_SIZE   = 200_000        # adjust for your RAM
AUGMENT_FRAC = 0.30           # ~30% extra rows per chunk
COORD_NOISE  = 0.0008         # lon/lat jitter
TARGET_NOISE = 0.10           # ±10% target noise
SEED         = 42

# Columns we need
USECOLS = [
    "pickup_datetime",
    "passenger_count",
    "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude",
    "trip_duration",
]
DTYPES = {
    "passenger_count": "int16",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "trip_duration": "int32",
}