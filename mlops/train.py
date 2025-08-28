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


def _load_clean_csv(csv_path: Path) -> pd.DataFrame:
    """
    Robust CSV loader:
      - reads header once to discover cols
      - loads without dtype enforcement
      - coerces numerics (invalid values -> NaN)
      - drops rows with NaN in required cols
      - casts to final dtypes
    """
    cols_present = pd.read_csv(csv_path, nrows=0).columns.tolist()
    usecols = [c for c in cfg.USECOLS if c in cols_present]
    parse_dates = ["pickup_datetime"] if "pickup_datetime" in usecols else None

    # Skip malformed rows with wrong number of fields
    df = pd.read_csv(
        csv_path,
        usecols=usecols,
        parse_dates=parse_dates,       # documented param for date parsing
        dayfirst=False,
        on_bad_lines="skip",           # pandas>=1.3; skips bad lines quietly
    )

    # Coerce numeric columns; junk like repeated header tokens -> NaN
    for c, target_dtype in cfg.DTYPES.items():
        if c in df.columns:
            try:
                # Only coerce if target is a numeric dtype
                if np.issubdtype(np.dtype(target_dtype), np.number):
                    df[c] = pd.to_numeric(df[c], errors="coerce")  # invalid -> NaN
            except TypeError:
                # If dtype is not interpretable by np.dtype (e.g., 'Int64'), still try
                df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows that failed coercion on required features/target
    required = [c for c in [
        "passenger_count", "trip_duration",
        "pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude"
    ] if c in df.columns]

    if required:
        before = len(df)
        df = df.dropna(subset=required)
        dropped = before - len(df)
        if dropped:
            print(f"[TRAIN] Dropped {dropped:,} rows with bad types/NaN in {required}")

    # Cast to final dtypes
    final_cast = {k: v for k, v in cfg.DTYPES.items() if k in df.columns}
    df = df.astype(final_cast)

    return df


def _features_from(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()

    has_dt = "pickup_datetime" in df.columns
    if has_dt:
        if not np.issubdtype(df["pickup_datetime"].dtype, np.datetime64):
            # tolerate mixed formats; NaT on bad rows
            df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
        df["hour"] = df["pickup_datetime"].dt.hour.astype("Int8")
        df["day_of_week"] = df["pickup_datetime"].dt.dayofweek.astype("Int8")
    else:
        # No datetime column? fall back to constants
        df["hour"] = 0
        df["day_of_week"] = 0

    df["distance"] = np.sqrt(
        (df["dropoff_longitude"] - df["pickup_longitude"]) ** 2 +
        (df["dropoff_latitude"]  - df["pickup_latitude"])  ** 2
    )

    base_feats = [
        "passenger_count", "distance",
        "pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude",
    ]
    # only include time features when available
    feats = (["hour", "day_of_week"] + base_feats) if has_dt else base_feats

    # guard against any other absent columns
    feats = [c for c in feats if c in df.columns]

    y = df["trip_duration"]
    mask = y < 18_000  # < 5 hours
    X = df[feats]
    return X[mask], y[mask]

def _log_sklearn_model(model, X_for_sig: pd.DataFrame, signature, input_example, pip_reqs):
    """
    MLflow 3.x deprecates `artifact_path` in favor of `name`.
    Use `name="model"` when available; fall back to `artifact_path="model"` for older MLflow.
    """
    try:
        # Newer MLflow (3.x): prefer `name`
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=input_example,
            signature=signature,
            pip_requirements=pip_reqs,
        )
    except TypeError:
        # Older MLflow: use artifact_path
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example,
            signature=signature,
            pip_requirements=pip_reqs,
        )

def train_from_csv(csv_path: Path = None, experiment_name: str = "nyc-taxi-experiment") -> dict[str, float]:
    """Train RF on a CSV and log to MLflow (local file store)."""
    MLFLOW_URI = "http://mlflow:5000"
    mlflow.set_tracking_uri(MLFLOW_URI)
    # Use train split by default
    if csv_path is None:
        csv_path = cfg.TRAIN_CSV
        if not csv_path.exists() and cfg.RAW_CSV.exists():
            print(f"[TRAIN] Warning: {csv_path.name} not found, using {cfg.RAW_CSV.name}")
            csv_path = cfg.RAW_CSV

        # Hard-coded MLflow URI — no env, no local file fallback
    mlflow.set_tracking_uri(MLFLOW_URI)
    print("[TRAIN] MLflow tracking:", mlflow.get_tracking_uri())
    mlflow.set_experiment(experiment_name)

    print(f"[TRAIN] Loading data from {Path(csv_path).name}...")
    df = _load_clean_csv(csv_path)
    print(f"[TRAIN] Loaded {len(df):,} clean samples")

    X, y = _features_from(df)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, random_state=cfg.SEED)

    with mlflow.start_run():
        run_id = mlflow.active_run().info.run_id
        _ = mlflow.get_artifact_uri()

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

        input_example = Xtr.iloc[:5].copy()
        signature = infer_signature(Xtr, model.predict(Xtr))

        pip_reqs = [
            f"mlflow=={mlflow.__version__}",
            f"scikit-learn=={sklearn.__version__}",
            f"pandas=={pd.__version__}",
            f"numpy=={np.__version__}",
        ]
        _log_sklearn_model(model, Xtr, signature, input_example, pip_reqs)

    print(f"[TRAIN] {Path(csv_path).name}: RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.3f}")

    from mlflow.tracking import MlflowClient
    client = MlflowClient(tracking_uri=MLFLOW_URI)
    exp_id = client.get_run(run_id).info.experiment_id

    import logging
    logger = logging.getLogger("airflow.task")

    logger.info("MLflow tracking URI: %s", mlflow.get_tracking_uri())
    logger.info("Experiment name: %s", experiment_name)
    logger.info("Run ID: %s", run_id)
    return {
        "rmse": rmse, "mae": mae, "r2": r2,
        "run_id": run_id,
        "experiment_id": exp_id,
        "model_uri": f"runs:/{run_id}/model"
    }



