# main.py
from __future__ import annotations
from mlops import config as cfg
from mlops.augment import augment_to_disk
from mlops.train import train_from_csv

def main():
    print("=== STEP 1: Baseline training on raw CSV ===")
    train_from_csv(cfg.RAW_CSV, experiment_name="nyc-taxi-baseline")

    print("\n=== STEP 2: Create augmented CSV on disk (chunked) ===")
    augment_to_disk()  # uses cfg.* defaults

    print("\n=== STEP 3: Training on augmented CSV ===")
    train_from_csv(cfg.AUG_CSV, experiment_name="nyc-taxi-augmented")

    print("\nAll done. View runs with:  mlflow ui  (open http://127.0.0.1:5000/)")

if __name__ == "__main__":
    main()
