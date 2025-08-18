# mlops/data_split.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from . import config as cfg


def split_data_to_disk(
        input_csv: Path = cfg.RAW_CSV,
        train_csv: Path = cfg.TRAIN_CSV,
        test_csv: Path = cfg.TEST_CSV,
        test_size: float = cfg.TEST_SIZE,
        seed: int = cfg.SEED,
) -> tuple[int, int]:
    """
    Split the raw data into train and test sets.
    Returns tuple of (train_size, test_size)
    """
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing {input_csv}")

    print(f"[SPLIT] Reading {input_csv.name}...")
    df = pd.read_csv(
        input_csv,
        usecols=cfg.USECOLS,
        dtype=cfg.DTYPES,
        parse_dates=["pickup_datetime"],
        dayfirst=False,
    )

    # Add an index column for easy reference
    df.reset_index(drop=True, inplace=True)
    df['row_id'] = df.index

    # Move row_id to first column
    cols = ['row_id'] + [col for col in df.columns if col != 'row_id']
    df = df[cols]

    # Split the data
    print(f"[SPLIT] Splitting data: test_size={test_size:.1%}")
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        shuffle=True
    )

    # Reset indices but keep row_id as reference to original
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    # Create test_id for easy API access (0-based sequential)
    test_df['test_id'] = range(len(test_df))

    # Save to disk
    train_csv.parent.mkdir(parents=True, exist_ok=True)
    test_csv.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"[SPLIT] Train set: {len(train_df):,} rows → {train_csv.name}")
    print(f"[SPLIT] Test set:  {len(test_df):,} rows → {test_csv.name}")
    print(f"[SPLIT] Test IDs range: 0 to {len(test_df) - 1}")

    return len(train_df), len(test_df)


def load_test_data() -> pd.DataFrame:
    """Load the test dataset"""
    if not cfg.TEST_CSV.exists():
        raise FileNotFoundError(
            f"Test data not found at {cfg.TEST_CSV}. "
            "Please run data splitting first."
        )

    return pd.read_csv(
        cfg.TEST_CSV,
        dtype=cfg.DTYPES,
        parse_dates=["pickup_datetime"],
        dayfirst=False,
    )


def get_test_sample(test_id: int) -> pd.Series:
    """Get a specific test sample by ID"""
    test_df = load_test_data()

    if test_id < 0 or test_id >= len(test_df):
        raise ValueError(f"Invalid test_id: {test_id}. Valid range: 0 to {len(test_df) - 1}")

    # Use test_id column if it exists, otherwise use index
    if 'test_id' in test_df.columns:
        row = test_df[test_df['test_id'] == test_id]
        if row.empty:
            raise ValueError(f"Test ID {test_id} not found")
        return row.iloc[0]
    else:
        return test_df.iloc[test_id]


def get_test_info() -> dict:
    """Get information about the test dataset"""
    test_df = load_test_data()

    return {
        "total_samples": len(test_df),
        "test_id_range": f"0 to {len(test_df) - 1}",
        "columns": list(test_df.columns),
        "sample_ids": list(range(min(10, len(test_df)))),  # First 10 IDs as example
        "stats": {
            "avg_trip_duration": float(test_df['trip_duration'].mean()),
            "min_trip_duration": float(test_df['trip_duration'].min()),
            "max_trip_duration": float(test_df['trip_duration'].max()),
        }
    }