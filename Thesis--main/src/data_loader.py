"""
Step 1: Data Loading, Cleaning, and Partitioning.

Loads the Intel Berkeley Lab dataset (54 motes, ~2.3M records).
Cleans NaN/outlier values.
Extracts temporal features for spatio-temporal analysis.
Partitions into:
    - 60% -> server_train_60.csv  (server-side TinyML training)
    - 40% -> node_realtime_40.csv (simulated real-time node sensing)

Partition is CHRONOLOGICAL and STRICTLY INDEPENDENT.
No overlap. Node side cannot access server data and vice versa.
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    RAW_DATASET_PATH, SERVER_DATA_PATH, NODE_DATA_PATH,
    MOTE_LOCS_PATH, SERVER_FRACTION, NODE_FRACTION, DATA_DIR
)


def load_raw_dataset(path=RAW_DATASET_PATH):
    """Load the Intel Berkeley Lab raw dataset."""
    print(f"Loading raw dataset from {path}...")
    columns = ["date", "time", "epoch", "mote_id",
               "temperature", "humidity", "light", "voltage"]
    df = pd.read_csv(
        path, sep=r"\s+", header=None, names=columns,
        on_bad_lines="skip", engine="python"
    )
    print(f"  Loaded {len(df):,} raw records")
    return df


def clean_dataset(df):
    """
    Remove rows with missing sensor readings and physical outliers.
    Intel Lab known valid ranges:
      temperature: -10 to 60 C
      humidity: 0 to 100 %
      light: 0 to 2500 lux
      voltage: 1.5 to 3.0 V
    """
    print("Cleaning dataset...")
    n_before = len(df)

    # Drop rows missing critical sensor columns
    df = df.dropna(subset=["temperature", "humidity", "light", "voltage"])

    # Remove physical outliers
    df = df[
        (df["temperature"] >= -10) & (df["temperature"] <= 60) &
        (df["humidity"] >= 0) & (df["humidity"] <= 100) &
        (df["light"] >= 0) & (df["light"] <= 2500) &
        (df["voltage"] >= 1.5) & (df["voltage"] <= 3.0)
    ]

    n_after = len(df)
    print(f"  Before cleaning: {n_before:,}")
    print(f"  After cleaning:  {n_after:,}")
    print(f"  Removed: {n_before - n_after:,} rows "
          f"({(n_before - n_after) / n_before * 100:.2f}%)")
    return df.reset_index(drop=True)


def parse_datetime(df):
    """Parse date+time into proper datetime column."""
    print("Parsing datetime...")
    df["datetime"] = pd.to_datetime(
        df["date"] + " " + df["time"],
        format="%Y-%m-%d %H:%M:%S.%f",
        errors="coerce"
    )
    df = df.dropna(subset=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    print(f"  Date range: {df['datetime'].min()} -> {df['datetime'].max()}")
    print(f"  Duration: {(df['datetime'].max() - df['datetime'].min()).days} days")
    return df


def extract_temporal_features(df):
    """
    Extract temporal features for spatio-temporal analysis.
    Includes cyclical encoding to capture periodic patterns.
    """
    print("Extracting temporal features...")
    dt = df["datetime"]

    df["hour"] = dt.dt.hour
    df["minute"] = dt.dt.minute
    df["day_of_week"] = dt.dt.dayofweek        # 0=Mon, 6=Sun

    # Cyclical encoding for hour (captures daily periodicity)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Drop original string columns (no longer needed)
    df = df.drop(columns=["date", "time"])

    print(f"  Features added: hour, minute, day_of_week, hour_sin, hour_cos")
    return df


def partition_data(df, server_frac=SERVER_FRACTION):
    """
    Chronological split into two INDEPENDENT partitions.
    First 60% (by time order) -> Server training data.
    Last 40% -> Node real-time sensing simulation data.
    """
    print(f"Partitioning data: {server_frac*100:.0f}% server / "
          f"{(1-server_frac)*100:.0f}% node...")

    # Data is already sorted by datetime from parse_datetime()
    split_idx = int(len(df) * server_frac)

    server_df = df.iloc[:split_idx].copy()
    node_df = df.iloc[split_idx:].copy()

    # Verify no overlap
    server_max_time = server_df["datetime"].max()
    node_min_time = node_df["datetime"].min()
    assert node_min_time > server_max_time, \
        "PARTITION ERROR: Overlap detected between server and node data!"

    print(f"  Server partition: {len(server_df):,} samples "
          f"({len(server_df)/len(df)*100:.1f}%)")
    print(f"    Period: {server_df['datetime'].min()} -> {server_df['datetime'].max()}")
    print(f"  Node partition:   {len(node_df):,} samples "
          f"({len(node_df)/len(df)*100:.1f}%)")
    print(f"    Period: {node_df['datetime'].min()} -> {node_df['datetime'].max()}")
    print(f"  Gap between partitions: {node_min_time - server_max_time}")

    return server_df, node_df


def save_partitions(server_df, node_df):
    """Save partitions to CSV."""
    print("Saving partitions...")
    server_df.to_csv(SERVER_DATA_PATH, index=False)
    node_df.to_csv(NODE_DATA_PATH, index=False)
    print(f"  Server data -> {SERVER_DATA_PATH}")
    print(f"  Node data   -> {NODE_DATA_PATH}")


def load_mote_locations(path=MOTE_LOCS_PATH):
    """Load the 54 sensor node physical locations."""
    motes = pd.read_csv(
        path, sep=r"\s+", header=None, names=["mote_id", "x", "y"]
    )
    print(f"Loaded {len(motes)} mote locations")
    return motes


def print_summary(server_df, node_df):
    """Print dataset summary statistics."""
    print("\n" + "=" * 65)
    print("DATASET SUMMARY")
    print("=" * 65)
    total = len(server_df) + len(node_df)
    print(f"Total clean records:     {total:,}")
    print(f"Server training (60%):   {len(server_df):,}")
    print(f"Node real-time (40%):    {len(node_df):,}")
    print(f"Unique motes:            {server_df['mote_id'].nunique()}")
    print(f"\nSensor value ranges (server partition):")
    for col in ["temperature", "humidity", "light", "voltage"]:
        print(f"  {col:>12}: {server_df[col].min():8.2f} -> "
              f"{server_df[col].max():8.2f}  "
              f"(mean: {server_df[col].mean():.2f})")
    print("=" * 65)


def run_data_pipeline():
    """Execute the full data loading pipeline."""
    print("=" * 65)
    print("PHASE 1: DATA LOADING AND PARTITIONING")
    print("=" * 65 + "\n")

    df = load_raw_dataset()
    df = clean_dataset(df)
    df = parse_datetime(df)
    df = extract_temporal_features(df)
    server_df, node_df = partition_data(df)
    save_partitions(server_df, node_df)
    print_summary(server_df, node_df)

    return server_df, node_df


if __name__ == "__main__":
    server_df, node_df = run_data_pipeline()
