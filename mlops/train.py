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
    """Extract features from dataframe"""
    df = df.copy()
    if "pickup_datetime" in df.columns and not np.issubdtype(df["pickup_datetime"].dtype, np.datetime64):
        df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
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

    # Select MLflow runs directory (Airflow container vs local dev)
    if os.environ.get("AIRFLOW__CORE__EXECUTOR"):
        mlruns_dir = Path("/opt/airflow/app/mlruns")
    else:
        mlruns_dir = Path(cfg.MLRUNS_DIR)

    mlruns_dir = mlruns_dir.resolve()
    mlruns_dir.mkdir(parents=True, exist_ok=True)

    # ✅ Correct, cross-platform file URI (file:///C:/... on Windows)
    mlflow.set_tracking_uri(mlruns_dir.as_uri())
    print(f"[TRAIN] Using MLflow runs directory: {mlruns_dir}")
    mlflow.set_experiment(experiment_name)

    # Read + clean data
    print(f"[TRAIN] Loading data from {Path(csv_path).name}...")
    df = _load_clean_csv(csv_path)
    print(f"[TRAIN] Loaded {len(df):,} clean samples")

    X, y = _features_from(df)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, random_state=cfg.SEED)

    with mlflow.start_run():
        run_id = mlflow.active_run().info.run_id
        artifact_root = mlflow.get_artifact_uri()  # for debugging / curiosity

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

    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    exp_id = client.get_run(run_id).info.experiment_id

    return {
        "rmse": rmse, "mae": mae, "r2": r2,
        "run_id": run_id,
        "experiment_id": exp_id,
        "model_uri": f"runs:/{run_id}/model"
    }
