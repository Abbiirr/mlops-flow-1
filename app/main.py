# app/main.py
from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
import requests
from enum import Enum

# Import your modules
import sys

sys.path.append('/opt/airflow/app')
from mlops import config as cfg
from mlops.augment import augment_to_disk
from mlops.train import train_from_csv
from mlops.data_split import split_data_to_disk, get_test_sample, get_test_info, load_test_data

# Configuration
AIRFLOW_API_URL = os.getenv("AIRFLOW_API_URL", "http://airflow-apiserver:8080")
AIRFLOW_USERNAME = os.getenv("AIRFLOW_USERNAME", "admin")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD", "admin")


# Pydantic Models
class DataSplitRequest(BaseModel):
    test_size: float = Field(default=0.001, ge=0.0001, le=0.1)
    seed: int = Field(default=42)


class TrainingRequest(BaseModel):
    experiment_name: str = Field(default="nyc-taxi-experiment")
    use_augmentation: bool = Field(default=True)
    augment_frac: float = Field(default=0.3, ge=0, le=1)
    n_estimators: int = Field(default=80, ge=10, le=500)
    max_depth: int = Field(default=12, ge=3, le=50)


class TestPredictionRequest(BaseModel):
    test_id: int = Field(ge=0, description="ID of the test sample (0-based index)")
    experiment_name: str = Field(default="nyc-taxi-experiment")


class TestPredictionResponse(BaseModel):
    test_id: int
    predicted_duration: float
    actual_duration: float
    error: float
    error_percentage: float
    input_features: Dict[str, Any]
    model_info: Dict[str, str]


class ManualPredictionRequest(BaseModel):
    passenger_count: int = Field(ge=1, le=6)
    pickup_datetime: str
    pickup_longitude: float = Field(ge=-180, le=180)
    pickup_latitude: float = Field(ge=-90, le=90)
    dropoff_longitude: float = Field(ge=-180, le=180)
    dropoff_latitude: float = Field(ge=-90, le=90)


class DagTriggerRequest(BaseModel):
    dag_id: str = Field(default="nyc_taxi_pipeline")
    conf: Optional[Dict[str, Any]] = None


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class TrainingResponse(BaseModel):
    job_id: str
    status: JobStatus
    metrics: Optional[Dict[str, float]] = None
    model_uri: Optional[str] = None
    message: str


# In-memory job store (in production, use Redis or database)
jobs_store: Dict[str, Dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    print("Starting MLOps FastAPI application...")

    # Ensure directories exist
    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    cfg.AUG_DIR.mkdir(parents=True, exist_ok=True)
    cfg.MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(f"file://{cfg.MLRUNS_DIR.resolve()}")

    # Check if data needs to be split
    if cfg.RAW_CSV.exists() and not cfg.TEST_CSV.exists():
        print("Splitting data into train/test sets...")
        try:
            split_data_to_disk()
        except Exception as e:
            print(f"Warning: Could not split data: {e}")

    yield

    # Shutdown
    print("Shutting down MLOps FastAPI application...")


# Create FastAPI app
app = FastAPI(
    title="NYC Taxi MLOps API",
    description="API for NYC Taxi trip duration prediction pipeline with test set evaluation",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Helper Functions
def get_airflow_auth():
    """Get Airflow authentication"""
    return (AIRFLOW_USERNAME, AIRFLOW_PASSWORD)


def load_latest_model(experiment_name: str):
    """Load the latest MLflow model from an experiment"""
    mlflow.set_tracking_uri(f"file://{cfg.MLRUNS_DIR.resolve()}")

    # Get the experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_name}' not found")

    # Get the latest run
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.r2 DESC"],
        max_results=1
    )

    if runs.empty:
        raise HTTPException(status_code=404, detail="No runs found in experiment")

    run_id = runs.iloc[0]["run_id"]
    model_uri = f"runs:/{run_id}/model"

    try:
        model = mlflow.sklearn.load_model(model_uri)
        return model, run_id
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


def prepare_features_from_row(row: pd.Series) -> pd.DataFrame:
    """Prepare features from a test row"""
    pickup_dt = row["pickup_datetime"]
    if isinstance(pickup_dt, str):
        pickup_dt = pd.to_datetime(pickup_dt)

    data = {
        "passenger_count": [row["passenger_count"]],
        "hour": [pickup_dt.hour],
        "day_of_week": [pickup_dt.dayofweek],
        "distance": [np.sqrt(
            (row["dropoff_longitude"] - row["pickup_longitude"]) ** 2 +
            (row["dropoff_latitude"] - row["pickup_latitude"]) ** 2
        )],
        "pickup_longitude": [row["pickup_longitude"]],
        "pickup_latitude": [row["pickup_latitude"]],
        "dropoff_longitude": [row["dropoff_longitude"]],
        "dropoff_latitude": [row["dropoff_latitude"]]
    }

    return pd.DataFrame(data)


def prepare_features_manual(request: ManualPredictionRequest) -> pd.DataFrame:
    """Prepare features for manual prediction"""
    pickup_dt = pd.to_datetime(request.pickup_datetime)

    data = {
        "passenger_count": [request.passenger_count],
        "hour": [pickup_dt.hour],
        "day_of_week": [pickup_dt.dayofweek],
        "distance": [np.sqrt(
            (request.dropoff_longitude - request.pickup_longitude) ** 2 +
            (request.dropoff_latitude - request.pickup_latitude) ** 2
        )],
        "pickup_longitude": [request.pickup_longitude],
        "pickup_latitude": [request.pickup_latitude],
        "dropoff_longitude": [request.dropoff_longitude],
        "dropoff_latitude": [request.dropoff_latitude]
    }

    return pd.DataFrame(data)


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "NYC Taxi MLOps API v2.0",
        "endpoints": {
            "health": "/health",
            "data": {
                "split": "/data/split",
                "info": "/data/test-info",
                "samples": "/data/test-samples"
            },
            "model": {
                "train": "/train",
                "predict_test": "/predict/test",
                "predict_manual": "/predict/manual",
                "experiments": "/experiments",
                "models": "/models/{experiment_name}"
            },
            "pipeline": {
                "trigger": "/trigger-dag",
                "status": "/dag-status/{dag_id}/{dag_run_id}"
            }
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "mlflow": False,
            "airflow": False,
            "train_data": False,
            "test_data": False
        }
    }

    # Check MLflow
    try:
        mlflow.list_experiments()
        health_status["components"]["mlflow"] = True
    except:
        pass

    # Check Airflow
    try:
        response = requests.get(f"{AIRFLOW_API_URL}/health", auth=get_airflow_auth())
        health_status["components"]["airflow"] = response.status_code == 200
    except:
        pass

    # Check data availability
    health_status["components"]["train_data"] = cfg.TRAIN_CSV.exists() or cfg.RAW_CSV.exists()
    health_status["components"]["test_data"] = cfg.TEST_CSV.exists()

    # Overall status
    if not all(health_status["components"].values()):
        health_status["status"] = "degraded"

    return health_status


@app.post("/data/split")
async def split_data(request: DataSplitRequest):
    """Split the raw data into train and test sets"""
    try:
        train_size, test_size = split_data_to_disk(
            test_size=request.test_size,
            seed=request.seed
        )

        return {
            "success": True,
            "train_samples": train_size,
            "test_samples": test_size,
            "test_percentage": f"{request.test_size:.1%}",
            "message": f"Data split successfully: {train_size:,} train, {test_size:,} test samples"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/test-info")
async def test_info():
    """Get information about the test dataset"""
    try:
        info = get_test_info()
        return info
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/test-samples")
async def get_test_samples(
        start_id: int = Query(default=0, ge=0),
        limit: int = Query(default=10, ge=1, le=100)
):
    """Get a range of test samples with their IDs"""
    try:
        test_df = load_test_data()

        if start_id >= len(test_df):
            raise HTTPException(status_code=404, detail=f"start_id {start_id} exceeds test set size")

        end_id = min(start_id + limit, len(test_df))
        samples = []

        for i in range(start_id, end_id):
            row = test_df.iloc[i]
            samples.append({
                "test_id": i,
                "passenger_count": int(row["passenger_count"]),
                "pickup_datetime": row["pickup_datetime"].isoformat() if hasattr(row["pickup_datetime"],
                                                                                 'isoformat') else str(
                    row["pickup_datetime"]),
                "pickup_location": {
                    "longitude": float(row["pickup_longitude"]),
                    "latitude": float(row["pickup_latitude"])
                },
                "dropoff_location": {
                    "longitude": float(row["dropoff_longitude"]),
                    "latitude": float(row["dropoff_latitude"])
                },
                "actual_duration": float(row["trip_duration"]),
                "actual_duration_minutes": float(row["trip_duration"] / 60)
            })

        return {
            "samples": samples,
            "start_id": start_id,
            "end_id": end_id - 1,
            "count": len(samples),
            "total_test_samples": len(test_df)
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Test data not found. Please split the data first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/test", response_model=TestPredictionResponse)
async def predict_test_sample(request: TestPredictionRequest):
    """Make a prediction on a test sample and compare with actual value"""
    try:
        # Get the test sample
        test_row = get_test_sample(request.test_id)

        # Load model
        model, run_id = load_latest_model(request.experiment_name)

        # Prepare features
        features = prepare_features_from_row(test_row)

        # Make prediction
        predicted_duration = float(model.predict(features)[0])
        actual_duration = float(test_row["trip_duration"])

        # Calculate error
        error = predicted_duration - actual_duration
        error_percentage = (error / actual_duration) * 100 if actual_duration > 0 else 0

        return TestPredictionResponse(
            test_id=request.test_id,
            predicted_duration=predicted_duration,
            actual_duration=actual_duration,
            error=error,
            error_percentage=error_percentage,
            input_features={
                "passenger_count": int(test_row["passenger_count"]),
                "pickup_datetime": str(test_row["pickup_datetime"]),
                "pickup_longitude": float(test_row["pickup_longitude"]),
                "pickup_latitude": float(test_row["pickup_latitude"]),
                "dropoff_longitude": float(test_row["dropoff_longitude"]),
                "dropoff_latitude": float(test_row["dropoff_latitude"]),
                "distance": float(features["distance"].iloc[0]),
                "hour": int(features["hour"].iloc[0]),
                "day_of_week": int(features["day_of_week"].iloc[0])
            },
            model_info={
                "run_id": run_id,
                "experiment_name": request.experiment_name
            }
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/manual")
async def predict_manual(
        request: ManualPredictionRequest,
        experiment_name: str = Query(default="nyc-taxi-experiment")
):
    """Make a prediction using manually provided features"""
    # Load model
    model, run_id = load_latest_model(experiment_name)

    # Prepare features
    features = prepare_features_manual(request)

    # Make prediction
    prediction = model.predict(features)[0]

    return {
        "predicted_duration": float(prediction),
        "predicted_duration_minutes": float(prediction / 60),
        "model_info": {
            "run_id": run_id,
            "experiment_name": experiment_name
        },
        "input_features": features.iloc[0].to_dict()
    }


@app.post("/train", response_model=TrainingResponse)
async def train_model(
        request: TrainingRequest,
        background_tasks: BackgroundTasks
):
    """Train a new model"""
    job_id = f"train_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    # Initialize job
    jobs_store[job_id] = {
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow().isoformat(),
        "request": request.dict()
    }

    # Run training in background
    background_tasks.add_task(
        run_training_job,
        job_id,
        request
    )

    return TrainingResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message="Training job submitted successfully"
    )


async def run_training_job(job_id: str, request: TrainingRequest):
    """Background task to run training"""
    try:
        jobs_store[job_id]["status"] = JobStatus.RUNNING
        jobs_store[job_id]["started_at"] = datetime.utcnow().isoformat()

        # Ensure train/test split exists
        if not cfg.TRAIN_CSV.exists() and cfg.RAW_CSV.exists():
            print(f"[Job {job_id}] Creating train/test split...")
            split_data_to_disk()

        # Prepare data
        if request.use_augmentation:
            augment_to_disk(
                input_csv=cfg.TRAIN_CSV,
                frac=request.augment_frac
            )
            train_csv = cfg.AUG_CSV
        else:
            train_csv = cfg.TRAIN_CSV

        # Train model
        metrics = train_from_csv(train_csv, request.experiment_name)

        # Get model URI
        experiment = mlflow.get_experiment_by_name(request.experiment_name)
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )
        run_id = runs.iloc[0]["run_id"]
        model_uri = f"runs:/{run_id}/model"

        # Update job status
        jobs_store[job_id].update({
            "status": JobStatus.SUCCESS,
            "completed_at": datetime.utcnow().isoformat(),
            "metrics": metrics,
            "model_uri": model_uri
        })

    except Exception as e:
        jobs_store[job_id].update({
            "status": JobStatus.FAILED,
            "completed_at": datetime.utcnow().isoformat(),
            "error": str(e)
        })


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get job status"""
    if job_id not in jobs_store:
        raise HTTPException(status_code=404, detail="Job not found")

    return jobs_store[job_id]


@app.post("/trigger-dag")
async def trigger_dag(request: DagTriggerRequest):
    """Trigger an Airflow DAG"""
    try:
        dag_run_id = trigger_airflow_dag(request.dag_id, request.conf)
        return {
            "success": True,
            "dag_id": request.dag_id,
            "dag_run_id": dag_run_id,
            "message": "DAG triggered successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/experiments")
async def list_experiments():
    """List all MLflow experiments"""
    mlflow.set_tracking_uri(f"file://{cfg.MLRUNS_DIR.resolve()}")
    experiments = mlflow.list_experiments()

    return {
        "experiments": [
            {
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "artifact_location": exp.artifact_location,
                "lifecycle_stage": exp.lifecycle_stage
            }
            for exp in experiments
        ]
    }


@app.get("/models/{experiment_name}")
async def get_models(
        experiment_name: str,
        limit: int = Query(default=10, ge=1, le=100)
):
    """Get models from an experiment"""
    mlflow.set_tracking_uri(f"file://{cfg.MLRUNS_DIR.resolve()}")

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_name}' not found")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.r2 DESC"],
        max_results=limit
    )

    if runs.empty:
        return {"models": []}

    models = []
    for _, run in runs.iterrows():
        models.append({
            "run_id": run["run_id"],
            "metrics": {
                "rmse": run.get("metrics.rmse"),
                "mae": run.get("metrics.mae"),
                "r2": run.get("metrics.r2")
            },
            "start_time": run["start_time"],
            "status": run["status"]
        })

    return {"models": models}


def trigger_airflow_dag(dag_id: str, conf: Optional[Dict] = None) -> str:
    """Trigger an Airflow DAG via API"""
    url = f"{AIRFLOW_API_URL}/api/v1/dags/{dag_id}/dagRuns"

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    payload = {
        "logical_date": datetime.utcnow().isoformat(),
        "conf": conf or {}
    }

    try:
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            auth=get_airflow_auth()
        )
        response.raise_for_status()
        return response.json()["dag_run_id"]
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to trigger DAG: {str(e)}")


def get_dag_run_status(dag_id: str, dag_run_id: str) -> str:
    """Get the status of a DAG run"""
    url = f"{AIRFLOW_API_URL}/api/v1/dags/{dag_id}/dagRuns/{dag_run_id}"

    try:
        response = requests.get(url, auth=get_airflow_auth())
        response.raise_for_status()
        return response.json()["state"]
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to get DAG status: {str(e)}")


@app.get("/dag-status/{dag_id}/{dag_run_id}")
async def get_dag_status(dag_id: str, dag_run_id: str):
    """Get DAG run status"""
    status = get_dag_run_status(dag_id, dag_run_id)
    return {
        "dag_id": dag_id,
        "dag_run_id": dag_run_id,
        "status": status
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)