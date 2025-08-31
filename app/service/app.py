# service/app.py
from __future__ import annotations
import os, json
from pathlib import Path
from datetime import datetime
import typing as T

import numpy as np
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from mlops import config as cfg

# ========== HIGHLIGHTED CHANGES START ==========
# CHANGE: Set MLflow server URL as constant
MLFLOW_TRACKING_URI = "http://mlflow:5000"
# ========== HIGHLIGHTED CHANGES END ==========

MODEL_URI_ENV = "MODEL_URI"
LOCAL_CHAMPION_DIR = Path("models/champion")
CHAMPION_JSON = Path(os.getenv("CHAMPION_JSON", "champion.json"))


# --------- request/response schema ----------
class PredictRequest(BaseModel):
    pickup_datetime: datetime
    passenger_count: int
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float

    @field_validator("passenger_count")
    @classmethod
    def _pc_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("passenger_count must be >= 1")
        return v


# --------- feature engineering (must mirror training) ----------
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df["pickup_datetime"])
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["distance"] = np.sqrt(
        (df["dropoff_longitude"] - df["pickup_longitude"]) ** 2 +
        (df["dropoff_latitude"] - df["pickup_latitude"]) ** 2
    )
    cols = [
        "passenger_count", "hour", "day_of_week", "distance",
        "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"
    ]
    X = df[cols].copy()

    # Enforce the model's signature types
    X["passenger_count"] = X["passenger_count"].astype("int32")
    X["hour"] = X["hour"].astype("int32")
    X["day_of_week"] = X["day_of_week"].astype("int32")
    float_cols = ["distance", "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]
    X[float_cols] = X[float_cols].astype("float32")

    return X


# ========== HIGHLIGHTED CHANGES START ==========
def _resolve_from_champion_json() -> tuple[str | None, dict]:
    """Return (model_uri, info_dict) if champion.json exists."""
    if not CHAMPION_JSON.exists():
        return None, {}

    data = json.loads(CHAMPION_JSON.read_text())

    # CHANGE: Handle the format from write_champion()
    # First try champion_model_uri (if we update write_champion)
    uri = data.get("champion_model_uri")

    # Fallback to model_uri (current write_champion format)
    if not uri:
        uri = data.get("model_uri")

    # Fallback to constructing from run_id
    if not uri and data.get("run_id"):
        uri = f"runs:/{data['run_id']}/model"

    # CHANGE: Always use MLflow server, not local file store
    if uri:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    return uri, data


# ========== HIGHLIGHTED CHANGES END ==========


def resolve_model_uri() -> tuple[str, dict]:
    """
    Resolution order:
      0) champion.json (preferred)
      1) $MODEL_URI environment variable
      2) models/champion/ (filesystem mirror)
      3) best r2 from MLflow server
    Returns (uri, context_info)
    """
    # ========== HIGHLIGHTED CHANGES START ==========
    # CHANGE: Always set MLflow tracking URI to server
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # ========== HIGHLIGHTED CHANGES END ==========

    # 0) champion.json
    uri, info = _resolve_from_champion_json()
    if uri:
        return uri, {"source": "champion.json", **info}

    # 1) explicit env
    env_uri = os.getenv(MODEL_URI_ENV)
    if env_uri:
        return env_uri, {"source": f"env:{MODEL_URI_ENV}"}

    # 2) filesystem mirror (local backup)
    if (LOCAL_CHAMPION_DIR / "MLmodel").exists():
        return str(LOCAL_CHAMPION_DIR.resolve()), {"source": "filesystem:models/champion"}

    # ========== HIGHLIGHTED CHANGES START ==========
    # 3) CHANGE: Search MLflow server (not local mlruns)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    # ========== HIGHLIGHTED CHANGES END ==========

    best_row = None
    best_r2 = float("-inf")

    for exp in client.search_experiments():
        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["metrics.r2 DESC"],
            max_results=1
        )
        if runs is not None and not runs.empty:
            r = runs.iloc[0]
            r2 = r.get("metrics.r2")
            if r2 is not None and float(r2) > best_r2:
                best_r2 = float(r2)
                best_row = r

    if best_row is None:
        raise RuntimeError(
            "No MLflow runs found with metric r2; ensure champion.json exists or set MODEL_URI or create models/champion/."
        )

    run_id = str(best_row["run_id"])
    return f"runs:/{run_id}/model", {"source": "search:max-r2", "run_id": run_id, "r2": best_r2}


# --------- loader ----------
class _Champion:
    def __init__(self):
        self.model = None
        self.uri = None
        self.info: dict = {}

    def load(self):
        # ========== HIGHLIGHTED CHANGES START ==========
        # CHANGE: Ensure MLflow is connected to server before loading
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        # ========== HIGHLIGHTED CHANGES END ==========

        self.uri, self.info = resolve_model_uri()

        # This will now fetch from MinIO via MLflow server
        self.model = mlflow.pyfunc.load_model(self.uri)

    def predict(self, rows: list[PredictRequest]) -> np.ndarray:
        df = pd.DataFrame([r.model_dump() for r in rows])
        X = make_features(df)
        return self.model.predict(X)


champion = _Champion()
champion.load()

# --------- app ----------
app = FastAPI(title="NYC Taxi — Champion API", version="1.0")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_uri": champion.uri,
        "source": champion.info.get("source"),
        # ========== HIGHLIGHTED CHANGES START ==========
        # CHANGE: Add MLflow tracking URI to health check
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI
        # ========== HIGHLIGHTED CHANGES END ==========
    }


@app.get("/where")
def where():
    return {"resolved_uri": champion.uri, "info": champion.info}


@app.post("/predict")
def predict_one(req: PredictRequest):
    try:
        val = float(champion.predict([req])[0])
        return {"trip_duration": val, "model_uri": champion.uri}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict_batch")
def predict_batch(reqs: list[PredictRequest]):
    try:
        vals = champion.predict(reqs).tolist()
        return {"predictions": vals, "model_uri": champion.uri}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/reload")
def reload_model():
    champion.load()
    return {"reloaded_from": champion.uri, "source": champion.info.get("source")}