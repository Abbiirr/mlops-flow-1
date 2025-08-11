from __future__ import annotations
import pendulum

# Airflow 3 public interface:
from airflow.sdk import dag, task  # Airflow 3.0+ public namespace
# If you’re on Airflow 2.x, use: from airflow.decorators import dag, task

# Import your project code (mounted at /opt/airflow/app)
from mlops import config as cfg
from mlops.augment import augment_to_disk
from mlops.train import train_from_csv

@dag(
    dag_id="nyc_taxi_pipeline",
    schedule="0 2 * * *",  # run daily at 02:00 UTC (adjust as needed)
    start_date=pendulum.datetime(2025, 8, 1, tz="UTC"),
    catchup=True,
    tags=["mlops", "mlflow", "nyc-taxi"],
)
def nyc_taxi_pipeline():
    @task()
    def baseline_train():
        return train_from_csv(cfg.RAW_CSV, experiment_name="nyc-taxi-baseline")

    @task()
    def write_augmented_csv():
        augment_to_disk()  # uses cfg defaults
        return str(cfg.AUG_CSV)

    @task()
    def train_on_augmented(_aug_path: str):
        return train_from_csv(cfg.AUG_CSV, experiment_name="nyc-taxi-augmented")

    # Orchestration
    metrics_baseline = baseline_train()
    aug_path = write_augmented_csv()
    metrics_aug = train_on_augmented(aug_path)

    # Implicit dependencies: baseline -> augment -> train_aug
    metrics_baseline >> aug_path >> metrics_aug

dag_inst = nyc_taxi_pipeline()
