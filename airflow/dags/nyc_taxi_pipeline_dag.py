# dags/nyc_taxi_pipeline_dag.py
from __future__ import annotations
import pendulum

# Airflow 3 public interface:
from airflow.sdk import dag, task  # Airflow 3.0+ public namespace
# If you're on Airflow 2.x, use: from airflow.decorators import dag, task

# Import your project code (mounted at /opt/airflow/app)
from mlops import config as cfg
from mlops.data_split import split_data_to_disk
from mlops.augment import augment_to_disk
from mlops.train import train_from_csv


@dag(
    dag_id="nyc_taxi_pipeline_v2",
    schedule="0 2 * * *",  # run daily at 02:00 UTC
    start_date=pendulum.datetime(2025, 8, 1, tz="UTC"),
    catchup=False,
    tags=["mlops", "mlflow", "nyc-taxi", "v2"],
    description="NYC Taxi pipeline with train/test split"
)
def nyc_taxi_pipeline_v2():
    @task()
    def split_data():
        """Split raw data into train and test sets"""
        train_size, test_size = split_data_to_disk(
            test_size=0.001,  # 0.1% for test
            seed=42
        )
        return {
            "train_size": train_size,
            "test_size": test_size,
            "train_path": str(cfg.TRAIN_CSV),
            "test_path": str(cfg.TEST_CSV)
        }

    @task()
    def baseline_train(split_info: dict):
        """Train baseline model on train split"""
        return train_from_csv(
            csv_path=cfg.TRAIN_CSV,
            experiment_name="nyc-taxi-baseline-v2"
        )

    @task()
    def create_augmented_data(split_info: dict):
        """Create augmented dataset from train split"""
        augment_to_disk(
            input_csv=cfg.TRAIN_CSV,  # Use train split
            output_csv=cfg.AUG_CSV,
            frac=0.3  # 30% augmentation
        )
        return str(cfg.AUG_CSV)

    @task()
    def train_on_augmented(aug_path: str):
        """Train model on augmented data"""
        return train_from_csv(
            csv_path=cfg.AUG_CSV,
            experiment_name="nyc-taxi-augmented-v2"
        )

    @task()
    def evaluate_models(baseline_metrics: dict, augmented_metrics: dict, split_info: dict):
        """Compare model performance"""
        print("=" * 50)
        print("Model Evaluation Results")
        print("=" * 50)
        print(f"\nDataset Split:")
        print(f"  Train samples: {split_info['train_size']:,}")
        print(f"  Test samples: {split_info['test_size']:,}")

        print(f"\nBaseline Model Metrics:")
        print(f"  RMSE: {baseline_metrics['rmse']:.2f}")
        print(f"  MAE: {baseline_metrics['mae']:.2f}")
        print(f"  R²: {baseline_metrics['r2']:.3f}")

        print(f"\nAugmented Model Metrics:")
        print(f"  RMSE: {augmented_metrics['rmse']:.2f}")
        print(f"  MAE: {augmented_metrics['mae']:.2f}")
        print(f"  R²: {augmented_metrics['r2']:.3f}")

        improvement = {
            "rmse": baseline_metrics['rmse'] - augmented_metrics['rmse'],
            "mae": baseline_metrics['mae'] - augmented_metrics['mae'],
            "r2": augmented_metrics['r2'] - baseline_metrics['r2']
        }

        print(f"\nImprovement with Augmentation:")
        print(f"  RMSE: {improvement['rmse']:.2f} (lower is better)")
        print(f"  MAE: {improvement['mae']:.2f} (lower is better)")
        print(f"  R²: {improvement['r2']:.3f} (higher is better)")

        return {
            "baseline": baseline_metrics,
            "augmented": augmented_metrics,
            "improvement": improvement,
            "best_model": "augmented" if augmented_metrics['r2'] > baseline_metrics['r2'] else "baseline"
        }

    # Define pipeline flow
    split_info = split_data()

    # Train baseline model on train split
    baseline_metrics = baseline_train(split_info)

    # Create augmented data and train
    aug_path = create_augmented_data(split_info)
    augmented_metrics = train_on_augmented(aug_path)

    # Evaluate and compare models
    evaluation = evaluate_models(baseline_metrics, augmented_metrics, split_info)


# Instantiate the DAG
dag_inst = nyc_taxi_pipeline_v2()