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

from mlops import config as cfg  # local mlruns default

MODEL_URI_ENV = "MODEL_URI"                 # optional override
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
        (df["dropoff_latitude"]  - df["pickup_latitude"]) ** 2
    )
    cols = [
        "passenger_count", "hour", "day_of_week", "distance",
        "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"
    ]
    X = df[cols].copy()

    # 🔐 Enforce the model’s signature types (no narrowing at predict time)
    X["passenger_count"] = X["passenger_count"].astype("int32")
    X["hour"]            = X["hour"].astype("int32")
    X["day_of_week"]     = X["day_of_week"].astype("int32")
    float_cols = ["distance", "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]
    X[float_cols] = X[float_cols].astype("float32")

    return X



# --------- helpers ----------
def _maybe_set_tracking_from(path_like: str | None) -> None:
    """Set MLflow tracking URI to a local file store if provided."""
    if not path_like:
        return
    p = Path(path_like).resolve()
    p.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(p.as_uri())  # file:///... on all OSes

def _resolve_from_champion_json() -> tuple[str | None, dict]:
    """Return (model_uri, info_dict) if champion.json exists and is usable."""
    if not CHAMPION_JSON.exists():
        return None, {}
    data = json.loads(CHAMPION_JSON.read_text())
    # Prefer explicit champion_model_uri; else infer from champion key
    uri = data.get("champion_model_uri")
    if not uri:
        champ_key = data.get("champion")
        if champ_key and isinstance(data.get(champ_key), dict):
            uri = data[champ_key].get("model_uri") or (
                f"runs:/{data[champ_key].get('run_id')}/model" if data[champ_key].get("run_id") else None
            )
    # Set tracking if mlruns_dir present
    _maybe_set_tracking_from((data.get("artifacts") or {}).get("mlruns_dir"))
    return uri, data

def resolve_model_uri() -> tuple[str, dict]:
    """
    Resolution order:
      0) champion.json (preferred)
      1) $MODEL_URI
      2) models/champion/ (filesystem mirror)
      3) best r2 from local mlruns
    Returns (uri, context_info)
    """
    # 0) champion.json
    uri, info = _resolve_from_champion_json()
    if uri:
        return uri, {"source": "champion.json", **info}

    # 1) explicit env
    env_uri = os.getenv(MODEL_URI_ENV)
    if env_uri:
        return env_uri, {"source": f"env:{MODEL_URI_ENV}"}

    # 2) filesystem mirror
    if (LOCAL_CHAMPION_DIR / "MLmodel").exists():
        return str(LOCAL_CHAMPION_DIR.resolve()), {"source": "filesystem:models/champion"}

    # 3) best r2 in local mlruns
    mlflow.set_tracking_uri(cfg.MLRUNS_DIR.resolve().as_uri())
    client = MlflowClient()
    best_run = None
    best_r2 = float("-inf")
    for exp in client.search_experiments():
        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["metrics.r2 DESC"],
            max_results=1
        )
        if len(runs) == 1:
            r = runs.iloc[0]
            r2 = r.get("metrics.r2")
            if r2 is not None and r2 > best_r2:
                best_r2 = r2
                best_run = r
    if not best_run:
        raise RuntimeError(
            "No MLflow runs found with metric r2; ensure champion.json exists or set MODEL_URI or create models/champion/."
        )
    return f"runs:/{best_run.run_id}/model", {"source": "search:max-r2", "run_id": best_run.run_id, "r2": best_r2}

# --------- loader ----------
class _Champion:
    def __init__(self):
        self.model = None
        self.uri = None
        self.info: dict = {}

    def load(self):
        self.uri, self.info = resolve_model_uri()
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
    return {"status": "ok", "model_uri": champion.uri, "source": champion.info.get("source")}

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
