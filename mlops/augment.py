# mlops/augment.py
from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import pandas as pd
from . import config as cfg

def augment_to_disk(
    input_csv: Path = cfg.RAW_CSV,
    output_csv: Path = cfg.AUG_CSV,
    *,
    chunksize: int = cfg.CHUNK_SIZE,
    frac: float = cfg.AUGMENT_FRAC,
    coord_noise: float = cfg.COORD_NOISE,
    target_noise: float = cfg.TARGET_NOISE,
    seed: int = cfg.SEED,
) -> None:
    """Stream input_csv → write original + augmented rows into output_csv."""
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing {input_csv}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists():
        output_csv.unlink()

    print(f"[AUGMENT] {input_csv.name} → {output_csv} | chunksize={chunksize}, frac={frac}")
    rng = np.random.default_rng(seed)
    header = True
    total_orig = 0
    total_aug = 0

    reader = pd.read_csv(
        input_csv,
        usecols=cfg.USECOLS,
        dtype=cfg.DTYPES,
        parse_dates=["pickup_datetime"],
        infer_datetime_format=True,
        dayfirst=False,
        chunksize=chunksize,
    )

    for i, chunk in enumerate(reader, start=1):
        # write originals
        chunk.to_csv(output_csv, mode="a", header=header, index=False)

        n_aug = 0
        if frac > 0:
            n_aug = int(math.floor(len(chunk) * frac))
            if n_aug > 0:
                idx = rng.integers(0, len(chunk), size=n_aug)
                aug = chunk.iloc[idx].copy()

                # jitter lon/lat
                aug["pickup_longitude"]  += rng.normal(0, coord_noise, size=n_aug)
                aug["dropoff_longitude"] += rng.normal(0, coord_noise, size=n_aug)
                aug["pickup_latitude"]   += rng.normal(0, coord_noise, size=n_aug)
                aug["dropoff_latitude"]  += rng.normal(0, coord_noise, size=n_aug)

                # ± noise on target
                scale = rng.uniform(1 - target_noise, 1 + target_noise, size=n_aug)
                aug["trip_duration"] = np.maximum(1, (aug["trip_duration"] * scale)).astype("int32")

                # append augmented
                aug.to_csv(output_csv, mode="a", header=False, index=False)

        total_orig += len(chunk)
        total_aug  += n_aug
        header = False
        print(f"  chunk {i:>3}: orig={len(chunk):>7}  aug={n_aug:>7}")

    print(f"[AUGMENT] done. original={total_orig}  augmented={total_aug}  total={total_orig+total_aug}")
