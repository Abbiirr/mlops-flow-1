# mlops/train.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import mlflow, mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from . import config as cfg

def _features_from(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df["hour"]        = df["pickup_datetime"].dt.hour
    df["day_of_week"] = df["pickup_datetime"].dt.dayofweek
    df["distance"]    = np.sqrt(
        (df["dropoff_longitude"] - df["pickup_longitude"])**2 +
        (df["dropoff_latitude"]  - df["pickup_latitude"])**2
    )
    feats = [
        "passenger_count","hour","day_of_week","distance",
        "pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"
    ]
    X, y = df[feats], df["trip_duration"]
    mask = y < 18_000  # < 5 hours
    return X[mask], y[mask]

def train_from_csv(csv_path: Path, experiment_name: str) -> dict[str, float]:
    """Train RF on a CSV and log to MLflow (local file store)."""
    cfg.MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlruns_dir = Path(cfg.MLRUNS_DIR).resolve()  # must be absolute
    mlflow.set_tracking_uri(mlruns_dir.as_uri())
    mlflow.set_experiment(experiment_name)

    df = pd.read_csv(
        csv_path,
        usecols=cfg.USECOLS,
        dtype=cfg.DTYPES,
        parse_dates=["pickup_datetime"],
        infer_datetime_format=True,
        dayfirst=False,
    )
    X, y = _features_from(df)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, random_state=cfg.SEED)

    with mlflow.start_run():
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 80)
        mlflow.log_param("max_depth", 12)
        mlflow.log_param("augment_source", Path(csv_path).name)
        mlflow.log_param("n_training_samples", len(Xtr))

        model = RandomForestRegressor(
            n_estimators=80, max_depth=12, n_jobs=-1, random_state=cfg.SEED
        )
        model.fit(Xtr, ytr)

        pred = model.predict(Xte)
        rmse = float(np.sqrt(mean_squared_error(yte, pred)))
        mae  = float(mean_absolute_error(yte, pred))
        r2   = float(r2_score(yte, pred))

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae",  mae)
        mlflow.log_metric("r2",   r2)
        mlflow.sklearn.log_model(model, "model")

    print(f"[TRAIN] {Path(csv_path).name}: RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.3f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}
