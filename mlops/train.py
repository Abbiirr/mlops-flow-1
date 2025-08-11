# mlops/train.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import mlflow, mlflow.sklearn
from mlflow.models import infer_signature
import sklearn  # for version pinning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from . import config as cfg

def train_from_csv(csv_path: Path, experiment_name: str) -> dict[str, float]:
    cfg.MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlruns_dir = Path(cfg.MLRUNS_DIR).resolve()
    mlflow.set_tracking_uri(mlruns_dir.as_uri())

    # ensure experiment points to the container-writable artifact root
    from mlflow.tracking import MlflowClient
    def ensure_experiment(name: str, root: Path):
        c = MlflowClient()
        if c.get_experiment_by_name(name) is None:
            c.create_experiment(name, artifact_location=root.resolve().as_uri())
        mlflow.set_experiment(name)
    ensure_experiment(experiment_name, mlruns_dir)

    df = pd.read_csv(csv_path, usecols=cfg.USECOLS, dtype=cfg.DTYPES,
                     parse_dates=["pickup_datetime"], dayfirst=False)
    X, y = _features_from(df)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, random_state=cfg.SEED)

    with mlflow.start_run():
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 80)
        mlflow.log_param("max_depth", 12)
        mlflow.log_param("augment_source", Path(csv_path).name)
        mlflow.log_param("n_training_samples", len(Xtr))

        model = RandomForestRegressor(n_estimators=80, max_depth=12,
                                      n_jobs=1, random_state=cfg.SEED)
        model.fit(Xtr, ytr)

        pred = model.predict(Xte)
        rmse = float(np.sqrt(mean_squared_error(yte, pred)))
        mae  = float(mean_absolute_error(yte, pred))
        r2   = float(r2_score(yte, pred))
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae",  mae)
        mlflow.log_metric("r2",   r2)

        input_example = Xtr.iloc[:5].copy()
        signature = infer_signature(Xtr, model.predict(Xtr))
        pip_reqs = [
            f"mlflow=={mlflow.__version__}",
            f"scikit-learn=={sklearn.__version__}",
            f"pandas=={pd.__version__}",
            f"numpy=={np.__version__}",
        ]
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",                      # modern param
            input_example=input_example,
            signature=signature,
            pip_requirements=pip_reqs,
        )

    print(f"[TRAIN] {Path(csv_path).name}: RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.3f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}