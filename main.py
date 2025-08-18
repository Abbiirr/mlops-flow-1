# main.py
from __future__ import annotations
from pathlib import Path
import os
import json
import datetime as dt

from mlops import config as cfg
from mlops.champion import write_champion
from mlops.data_split import split_data_to_disk
from mlops.augment import augment_to_disk
from mlops.train import train_from_csv


def _choose_champion(baseline: dict, augmented: dict) -> str:
    """
    Pick the better model. Primary metric: higher R² wins.
    Tie-breaker: lower RMSE wins.
    """
    if augmented["r2"] > baseline["r2"]:
        return "augmented"
    if augmented["r2"] < baseline["r2"]:
        return "baseline"
    # tie on R² → use RMSE
    return "augmented" if augmented["rmse"] <= baseline["rmse"] else "baseline"


def main():
    print("=== STEP 0: Split raw data into train/test (like the DAG) ===")
    train_n, test_n = split_data_to_disk(
        input_csv=cfg.RAW_CSV,
        train_csv=cfg.TRAIN_CSV,
        test_csv=cfg.TEST_CSV,
        test_size=cfg.TEST_SIZE,
        seed=cfg.SEED,
    )
    print(f"[SPLIT] Train={train_n:,} rows | Test={test_n:,} rows")

    print("\n=== STEP 1: Train baseline on TRAIN split ===")
    baseline_metrics = train_from_csv(
        cfg.TRAIN_CSV, experiment_name="nyc-taxi-baseline-v2"
    )

    print("\n=== STEP 2: Create augmented dataset from TRAIN split ===")
    orig_rows, aug_rows = augment_to_disk(
        input_csv=cfg.TRAIN_CSV,
        output_csv=cfg.AUG_CSV,
        frac=cfg.AUGMENT_FRAC,
        chunksize=cfg.CHUNK_SIZE,
        coord_noise=cfg.COORD_NOISE,
        target_noise=cfg.TARGET_NOISE,
        seed=cfg.SEED,
    )
    print(f"[AUGMENT] Wrote {orig_rows:,} original + {aug_rows:,} augmented rows → {cfg.AUG_CSV.name}")

    print("\n=== STEP 3: Train on AUGMENTED data ===")
    augmented_metrics = train_from_csv(
        cfg.AUG_CSV, experiment_name="nyc-taxi-augmented-v2"
    )

    print("\n=== STEP 4: Evaluate & pick CHAMPION (augmented vs baseline) ===")
    improvement = {
        "rmse": baseline_metrics["rmse"] - augmented_metrics["rmse"],
        "mae": baseline_metrics["mae"] - augmented_metrics["mae"],
        "r2": augmented_metrics["r2"] - baseline_metrics["r2"],
    }
    print("------------------------------------------------------------")
    print("Baseline:", baseline_metrics)
    print("Augmented:", augmented_metrics)
    print("Improvement (augmented - baseline):", improvement)

    champion = _choose_champion(baseline_metrics, augmented_metrics)
    print(f"\n>>> CHAMPION MODEL: {champion.upper()} <<<")

    if champion == "augmented":
        best_run_id, best_r2 = augmented_metrics["run_id"], augmented_metrics["r2"]
    else:
        best_run_id, best_r2 = baseline_metrics["run_id"], baseline_metrics["r2"]
    persisted = write_champion(best_run_id, "r2", best_r2)

    # Persist the decision for downstream tools
    mlruns_dir = (
        Path("/opt/airflow/app/mlruns")
        if os.environ.get("AIRFLOW__CORE__EXECUTOR")
        else cfg.MLRUNS_DIR
    ).resolve()

    decision = {
        "decided_at": dt.datetime.utcnow().isoformat() + "Z",
        "train_samples": train_n,
        "test_samples": test_n,
        "baseline": baseline_metrics,
        "augmented": augmented_metrics,
        "improvement": improvement,
        "champion": champion,
        "champion_model_uri": persisted["model_uri"],
        "artifacts": {
            "train_csv": str(cfg.TRAIN_CSV),
            "aug_csv": str(cfg.AUG_CSV),
            "mlruns_dir": str(mlruns_dir),
        },
    }
    out_path = (cfg.ROOT / "champion.json").resolve()
    out_path.write_text(json.dumps(decision, indent=2))
    print(f"[INFO] Wrote decision to {out_path}")

    print("\nView runs with:  mlflow ui   (open http://127.0.0.1:5000/)")
    print(f"MLflow runs directory: {mlruns_dir}")


if __name__ == "__main__":
    main()
