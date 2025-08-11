# mlops/train.py
from __future__ import annotations
from pathlib import Path
import os
import numpy as np
import pandas as pd
import mlflow, mlflow.sklearn
from mlflow.models import infer_signature
import sklearn  # for version pinning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from . import config as cfg


def _features_from(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract features from dataframe"""
    df = df.copy()
    df["hour"] = df["pickup_datetime"].dt.hour
    df["day_of_week"] = df["pickup_datetime"].dt.dayofweek
    df["distance"] = np.sqrt(
        (df["dropoff_longitude"] - df["pickup_longitude"]) ** 2 +
        (df["dropoff_latitude"] - df["pickup_latitude"]) ** 2
    )
    feats = [
        "passenger_count", "hour", "day_of_week", "distance",
        "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"
    ]
    X, y = df[feats], df["trip_duration"]
    mask = y < 18_000  # < 5 hours
    return X[mask], y[mask]


def train_from_csv(csv_path: Path = None, experiment_name: str = "nyc-taxi-experiment") -> dict[str, float]:
    """Train RF on a CSV and log to MLflow (local file store)."""

    # Use train split by default
    if csv_path is None:
        csv_path = cfg.TRAIN_CSV
        if not csv_path.exists() and cfg.RAW_CSV.exists():
            print(f"[TRAIN] Warning: {csv_path.name} not found, using {cfg.RAW_CSV.name}")
            csv_path = cfg.RAW_CSV

    # Use explicit path for MLflow runs directory
    if os.environ.get('AIRFLOW__CORE__EXECUTOR'):  # Running in Airflow Docker
        mlruns_dir = Path('/opt/airflow/app/mlruns')
    else:  # Local development
        mlruns_dir = cfg.MLRUNS_DIR

    mlruns_dir.mkdir(parents=True, exist_ok=True)
    mlruns_dir_resolved = mlruns_dir.resolve()
    print(f"[TRAIN] Using MLflow runs directory: {mlruns_dir_resolved}")

    # Set tracking URI to the resolved path
    mlflow.set_tracking_uri(f"file://{mlruns_dir_resolved}")
    mlflow.set_experiment(experiment_name)

    # Read data
    print(f"[TRAIN] Loading data from {csv_path.name}...")
    df = pd.read_csv(
        csv_path,
        usecols=[col for col in cfg.USECOLS if col in pd.read_csv(csv_path, nrows=1).columns],
        dtype={k: v for k, v in cfg.DTYPES.items() if k in pd.read_csv(csv_path, nrows=1).columns},
        parse_dates=["pickup_datetime"],
        dayfirst=False,
    )

    print(f"[TRAIN] Loaded {len(df):,} samples")

    X, y = _features_from(df)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, random_state=cfg.SEED)

    with mlflow.start_run():
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 80)
        mlflow.log_param("max_depth", 12)
        mlflow.log_param("data_source", Path(csv_path).name)
        mlflow.log_param("n_training_samples", len(Xtr))
        mlflow.log_param("n_validation_samples", len(Xte))

        model = RandomForestRegressor(
            n_estimators=80, max_depth=12, n_jobs=-1, random_state=cfg.SEED
        )
        print(f"[TRAIN] Training model on {len(Xtr):,} samples...")
        model.fit(Xtr, ytr)

        pred = model.predict(Xte)
        rmse = float(np.sqrt(mean_squared_error(yte, pred)))
        mae = float(mean_absolute_error(yte, pred))
        r2 = float(r2_score(yte, pred))
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Add signature + input_example
        input_example = Xtr.iloc[:5].copy()
        signature = infer_signature(Xtr, model.predict(Xtr))

        # Pin requirements to avoid pip-version warning
        pip_reqs = [
            f"mlflow=={mlflow.__version__}",
            f"scikit-learn=={sklearn.__version__}",
            f"pandas=={pd.__version__}",
            f"numpy=={np.__version__}",
        ]
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example,
            signature=signature,
            pip_requirements=pip_reqs,
        )

    print(f"[TRAIN] {Path(csv_path).name}: RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.3f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}