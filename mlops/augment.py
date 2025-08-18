# mlops/augment.py
from __future__ import annotations
from pathlib import Path
import math
import os
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from . import config as cfg


def _present(cols: Iterable[str], in_df: pd.DataFrame) -> list[str]:
    cset = set(in_df.columns)
    return [c for c in cols if c in cset]


def augment_to_disk(
    input_csv: Path | None = None,            # default: TRAIN_CSV, fallback: RAW_CSV
    output_csv: Path = cfg.AUG_CSV,
    *,
    chunksize: int = cfg.CHUNK_SIZE,
    frac: float = cfg.AUGMENT_FRAC,
    coord_noise: float = cfg.COORD_NOISE,
    target_noise: float = cfg.TARGET_NOISE,
    seed: int = cfg.SEED,
    preserve_id_cols: tuple[str, ...] = ("row_id", "test_id", "test_row_id"),
) -> tuple[int, int]:
    """
    Stream-read input_csv and write original + augmented rows to output_csv.

    Returns
    -------
    (total_original_rows, total_augmented_rows)
    """

    # 1) Pick source: TRAIN_CSV by default, fallback to RAW_CSV
    if input_csv is None:
        input_csv = cfg.TRAIN_CSV
    if not input_csv.exists():
        if cfg.RAW_CSV.exists():
            print(f"[AUGMENT] Warning: {input_csv.name} not found; using {cfg.RAW_CSV.name}")
            input_csv = cfg.RAW_CSV
        else:
            raise FileNotFoundError(f"Missing {input_csv}")

    # 2) Prepare output
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists():
        output_csv.unlink()

    # 3) Peek one row to discover available columns (since usecols must match)
    #    Then build a safe usecols + dtype subset.
    peek = pd.read_csv(input_csv, nrows=1)
    available = set(peek.columns)

    usecols = [c for c in cfg.USECOLS if c in available]
    # keep any ID-ish columns that appear
    for idc in preserve_id_cols:
        if idc in available and idc not in usecols:
            usecols.append(idc)

    dtypes = {k: v for k, v in cfg.DTYPES.items() if k in usecols}
    parse_dates = ["pickup_datetime"] if "pickup_datetime" in usecols else False

    # 4) Chunked read
    reader = pd.read_csv(
        input_csv,
        usecols=usecols,         # only columns we actually have (pre-checked)
        dtype=dtypes,
        parse_dates=parse_dates, # only if present
        dayfirst=False,
        chunksize=chunksize,     # iterator of DataFrames
    )

    print(f"[AUGMENT] {input_csv.name} → {output_csv} | chunksize={chunksize}, frac={frac}")
    rng = np.random.default_rng(seed)  # per-run Generator (no global state)
    header = True
    total_orig = 0
    total_aug = 0

    # Columns we may jitter if present
    jitter_cols = [
        "pickup_longitude", "dropoff_longitude",
        "pickup_latitude", "dropoff_latitude",
    ]

    for i, chunk in enumerate(reader, start=1):
        # 5) Write originals
        chunk.to_csv(output_csv, mode="a", header=header, index=False)

        # 6) Augment a fraction
        n_aug = 0
        if frac > 0 and len(chunk) > 0:
            n_aug = int(math.floor(len(chunk) * frac))
            if n_aug > 0:
                idx = rng.integers(0, len(chunk), size=n_aug)
                aug = chunk.iloc[idx].copy()

                # Jitter lon/lat only if those columns exist in the current file
                for col in _present(jitter_cols, aug):
                    aug[col] = aug[col].astype("float64") + rng.normal(0, coord_noise, size=n_aug)

                # ± noise on target if present
                if "trip_duration" in aug.columns:
                    scale = rng.uniform(1 - target_noise, 1 + target_noise, size=n_aug)
                    aug["trip_duration"] = np.maximum(1, (aug["trip_duration"] * scale)).astype("int32")

                # Append augmented
                aug.to_csv(output_csv, mode="a", header=False, index=False)

        total_orig += len(chunk)
        total_aug += n_aug
        header = False
        print(f"  chunk {i:>3}: orig={len(chunk):>7}  aug={n_aug:>7}")

    print(f"[AUGMENT] done. original={total_orig}  augmented={total_aug}  total={total_orig + total_aug}")
    return total_orig, total_aug
