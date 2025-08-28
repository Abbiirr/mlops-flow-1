# mlops/champion.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json, os, shutil, mlflow
from mlflow.artifacts import download_artifacts  # copy artifacts locally if you want a FS champion  :contentReference[oaicite:3]{index=3}

CHAMPION_JSON = Path("champion.json")
FS_CHAMPION_DIR = Path("models/champion")

def write_champion(best_run_id: str, metric_name: str, metric_value: float, note: str = "") -> dict:
    """Persist champion info and (optionally) copy its model to models/champion/."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    payload = {
        "run_id": best_run_id,
        "model_uri": f"runs:/{best_run_id}/model",
        "metric": metric_name,
        "metric_value": metric_value,
        "saved_at": datetime.utcnow().isoformat() + "Z"
    }
    CHAMPION_JSON.write_text(json.dumps(payload, indent=2))

    # Optional: filesystem mirror for easy serving without registry
    shutil.rmtree(FS_CHAMPION_DIR, ignore_errors=True)
    download_artifacts(run_id=best_run_id, artifact_path="model", dst_path=str(FS_CHAMPION_DIR))
    payload["filesystem_champion_dir"] = str(FS_CHAMPION_DIR.resolve())
    return payload
