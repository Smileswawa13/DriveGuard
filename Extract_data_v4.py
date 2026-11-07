"""
DriveGuard Extract v4
Creates drive-disjoint, one-snapshot-per-drive train/validation/test partitions
and separate real Backblaze website demonstration samples.

Usage:
    python Extract_data_v4.py --input harddrive.csv

Requires: pandas, numpy, scikit-learn
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
SPLIT_RATIOS = (0.70, 0.10, 0.15, 0.05)  # train, validation, test, website
SAMPLE_COUNT = 5
SAMPLE_ROWS = 200
SAMPLE_FAILURES = 2


def read_and_clean(path: Path) -> pd.DataFrame:
    print(f"Reading {path} ...")
    df = pd.read_csv(path, low_memory=False)
    required = {"date", "serial_number", "failure"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["failure"] = pd.to_numeric(df["failure"], errors="coerce")
    df = df.dropna(subset=["date", "serial_number", "failure"]).copy()
    df["failure"] = df["failure"].astype(int)
    df = df[df["failure"].isin([0, 1])].copy()

    before = len(df)
    # Exact duplicate full rows are safe to remove. If same drive/day has
    # conflicting records, stop rather than choosing one silently.
    df = df.drop_duplicates()
    exact_removed = before - len(df)

    dup_key = df.duplicated(["serial_number", "date"], keep=False)
    if dup_key.any():
        conflicts = (
            df.loc[dup_key]
            .groupby(["serial_number", "date"], dropna=False)
            .nunique(dropna=False)
            .max(axis=1)
        )
        bad = conflicts[conflicts > 1]
        if len(bad):
            raise ValueError(
                f"Found {len(bad)} conflicting drive-day groups. "
                "Resolve these before extraction; no rows were silently selected."
            )
        df = df.drop_duplicates(["serial_number", "date"], keep="first")

    # A physical drive is positive if any record marks its last operational
    # day as failure=1. Its model snapshot will be that failure-labelled row.
    return df, exact_removed


def one_snapshot_per_drive(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    positive_ids = set(df.loc[df["failure"] == 1, "serial_number"].astype(str))
    chosen = []

    for serial, group in df.groupby("serial_number", sort=False):
        serial = str(serial)
        failed_rows = group[group["failure"] == 1]
        if serial in positive_ids:
            # If multiple distinct failure dates exist, use the latest failure
            # day as the terminal labelled snapshot.
            row = failed_rows.sort_values("date").iloc[-1]
        else:
            idx = int(rng.integers(0, len(group)))
            row = group.iloc[idx]
        chosen.append(row)

    result = pd.DataFrame(chosen).reset_index(drop=True)
    result["failure"] = result["failure"].astype(int)
    return result


def stratified_partition(drives: pd.DataFrame):
    ids = drives["serial_number"].astype(str)
    y = drives["failure"].astype(int)

    trainval_ids, demo_ids = train_test_split(
        ids, test_size=0.05, random_state=RANDOM_STATE, stratify=y
    )
    # Remaining 95%: test is 15% of total, hence 15/95 of remainder.
    trainval_y = drives.set_index(drives["serial_number"].astype(str)).loc[trainval_ids, "failure"]
    trainval_ids, test_ids = train_test_split(
        trainval_ids, test_size=0.15 / 0.95, random_state=RANDOM_STATE,
        stratify=trainval_y
    )
    trainval_y = drives.set_index(drives["serial_number"].astype(str)).loc[trainval_ids, "failure"]
    train_ids, val_ids = train_test_split(
        trainval_ids, test_size=0.10 / 0.80, random_state=RANDOM_STATE,
        stratify=trainval_y
    )
    lookup = drives.assign(_sid=drives["serial_number"].astype(str)).set_index("_sid")
    return (
        lookup.loc[list(train_ids)].reset_index(drop=True),
        lookup.loc[list(val_ids)].reset_index(drop=True),
        lookup.loc[list(test_ids)].reset_index(drop=True),
        lookup.loc[list(demo_ids)].reset_index(drop=True),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="harddrive.csv")
    parser.add_argument("--output", default="driveguard_extracted_v4")
    args = parser.parse_args()

    source = Path(args.input)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    samples_dir = output / "website_samples"
    samples_dir.mkdir(exist_ok=True)

    df, exact_removed = read_and_clean(source)
    rng = np.random.default_rng(RANDOM_STATE)
    drives = one_snapshot_per_drive(df, rng)

    train, val, test, demo = stratified_partition(drives)

    # Remove metadata and keep SMART columns only. capacity_bytes is excluded.
    smart_cols = [c for c in drives.columns if c.startswith("smart_")]
    if not smart_cols:
        raise ValueError("No SMART columns found in source dataset.")

    metadata = ["date", "serial_number", "model", "capacity_bytes", "failure"]
    keep = [c for c in metadata if c in drives.columns] + smart_cols

    for name, part in [
        ("train", train), ("validation", val), ("test", test), ("demo", demo)
    ]:
        part = part[keep].copy()
        part.to_csv(output / f"driveguard_{name}_v4.csv", index=False)

    # Five independent sample files: 2 positive drives and 198 healthy drives.
    demo_pos = demo[demo["failure"] == 1]
    demo_neg = demo[demo["failure"] == 0]
    needed_pos = SAMPLE_COUNT * SAMPLE_FAILURES
    needed_neg = SAMPLE_COUNT * (SAMPLE_ROWS - SAMPLE_FAILURES)
    if len(demo_pos) < needed_pos or len(demo_neg) < needed_neg:
        raise ValueError(
            f"Demo partition too small: {len(demo_pos)} failures and "
            f"{len(demo_neg)} healthy drives; need {needed_pos} and {needed_neg}."
        )

    demo_pos = demo_pos.sample(n=needed_pos, random_state=RANDOM_STATE)
    demo_neg = demo_neg.sample(n=needed_neg, random_state=RANDOM_STATE)
    for i in range(SAMPLE_COUNT):
        p = demo_pos.iloc[i*SAMPLE_FAILURES:(i+1)*SAMPLE_FAILURES]
        n = demo_neg.iloc[i*(SAMPLE_ROWS-SAMPLE_FAILURES):(i+1)*(SAMPLE_ROWS-SAMPLE_FAILURES)]
        sample = pd.concat([p, n], ignore_index=True).sample(
            frac=1, random_state=RANDOM_STATE + i
        )
        sample[keep].to_csv(samples_dir / f"driveguard_sample_{i+1}.csv", index=False)

    summary = {
        "source_file": str(source.resolve()),
        "source_rows_after_key_validation": int(len(df)),
        "exact_full_duplicate_rows_removed": int(exact_removed),
        "unique_drives": int(len(drives)),
        "unique_failure_drives": int(drives["failure"].sum()),
        "snapshot_policy": "one row per drive; failure-labelled terminal day for failed drives; seeded random day for healthy drives",
        "split_ratios": {"train": 0.70, "validation": 0.10, "test": 0.15, "website_demo": 0.05},
        "partitions": {},
        "smart_feature_count": len(smart_cols),
        "website_samples": {
            "files": SAMPLE_COUNT, "rows_each": SAMPLE_ROWS,
            "failure_rows_each": SAMPLE_FAILURES,
            "source": "separate website-demo drive partition"
        },
        "warning": "failure=1 denotes the last operational day before failure; this dataset does not prove multi-day advance warning."
    }
    for name, part in [("train", train), ("validation", val), ("test", test), ("website_demo", demo)]:
        summary["partitions"][name] = {
            "drives": int(len(part)),
            "failure_drives": int(part["failure"].sum()),
            "healthy_drives": int((part["failure"] == 0).sum()),
            "failure_prevalence": float(part["failure"].mean())
        }

    # Explicitly verify pairwise drive disjointness.
    sets = {
        "train": set(train["serial_number"].astype(str)),
        "validation": set(val["serial_number"].astype(str)),
        "test": set(test["serial_number"].astype(str)),
        "website_demo": set(demo["serial_number"].astype(str))
    }
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            overlap = sets[a] & sets[b]
            if overlap:
                raise RuntimeError(f"Drive leakage: {a}/{b} overlap = {len(overlap)}")

    (output / "extraction_summary_v4.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (output / "extraction_summary_v4.txt").open("w", encoding="utf-8") as f:
        f.write("DRIVEGUARD EXTRACTION V4 SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nExtraction complete: {output.resolve()}")


if __name__ == "__main__":
    main()
