"""
Step 5 & 6: Parallel Node + Server Simulation Engine.

SERVER SIDE: predicts y_hat(t) using TinyML model (trained on 60%).
NODE SIDE: reads 40% data as real-time stream, compares |predicted - sensed|.
  - If within threshold: SUPPRESS (save energy).
  - If exceeds threshold: TRANSMIT. Buffer mismatch. Retrain when buffer full.

Vectorized implementation: batch-predicts, then loops only for retraining.
"""

import pandas as pd
import numpy as np
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    NODE_DATA_PATH, THRESHOLD_DEFAULT, BUFFER_MIN, PACKET_SIZE_BITS,
    E_INIT, BS_LOCATION, METRICS_DIR, SIM_SAMPLE_LIMIT
)
from energy_model import energy_node_epoch
from tinyml_model import prepare_features, retrain_model


def run_simulation(model, feat_cols, nodes, node_df=None,
                   threshold=THRESHOLD_DEFAULT, buffer_min=BUFFER_MIN):
    if node_df is None:
        node_df = pd.read_csv(NODE_DATA_PATH)

    # Apply sample limit
    if SIM_SAMPLE_LIMIT and len(node_df) > SIM_SAMPLE_LIMIT:
        node_df = node_df.iloc[:SIM_SAMPLE_LIMIT].copy()

    X_node, y_node, _ = prepare_features(node_df, feat_cols)
    X_arr = X_node.values
    y_arr = y_node.values
    mote_ids = node_df["mote_id"].values
    n_total = len(X_arr)

    # Precompute per-node distance to CH
    node_to_ch_dist = {}
    for c in sorted(nodes["cluster"].unique()):
        ch = nodes[(nodes["cluster"] == c) & (nodes["is_ch"])].iloc[0]
        for _, nrow in nodes[nodes["cluster"] == c].iterrows():
            mid = nrow["mote_id"]
            if nrow["is_ch"]:
                node_to_ch_dist[mid] = 0.0
            else:
                node_to_ch_dist[mid] = np.sqrt(
                    (nrow["x"] - ch["x"])**2 + (nrow["y"] - ch["y"])**2
                )

    # Map mote_id -> distance array for vectorized energy calc
    dist_arr = np.array([node_to_ch_dist.get(m, 5.0) for m in mote_ids])

    # Initialize logs
    log_transmitted = np.zeros(n_total, dtype=bool)
    log_error = np.zeros(n_total)
    log_energy_proposed = np.zeros(n_total)
    log_energy_baseline = np.zeros(n_total)
    retrain_events = []
    mismatch_rate_over_time = []
    node_energy = {mid: E_INIT for mid in nodes["mote_id"].values}

    current_model = model
    t0 = time.time()

    # Process in CHUNKS: predict a batch, find mismatches, retrain if needed
    chunk_start = 0
    cumulative_mismatch = 0
    buffer_X = []
    buffer_y = []

    print(f"Simulating {n_total:,} slots, threshold={threshold}, "
          f"buffer_min={buffer_min}")

    while chunk_start < n_total:
        # Determine chunk: from chunk_start to next retrain trigger or end
        # Predict entire remaining data with current model
        remaining_X = X_arr[chunk_start:]
        remaining_y = y_arr[chunk_start:]
        remaining_dist = dist_arr[chunk_start:]
        remaining_motes = mote_ids[chunk_start:]

        preds = current_model.predict(remaining_X)
        errors = np.abs(preds - remaining_y)
        mismatch_mask = errors > threshold

        # Walk through mismatches to fill buffer and find retrain point
        retrain_idx = None  # relative to chunk_start
        buf_count_before = len(buffer_X)

        mismatch_indices = np.where(mismatch_mask)[0]
        needed = buffer_min - len(buffer_X)

        if needed > 0 and len(mismatch_indices) >= needed:
            # Retrain will trigger at the 'needed'-th mismatch
            retrain_idx = mismatch_indices[needed - 1]
        else:
            retrain_idx = None  # no retrain in this chunk

        if retrain_idx is not None:
            # Process up to retrain_idx (inclusive)
            end = retrain_idx + 1
        else:
            # Process all remaining
            end = len(remaining_X)

        # Fill logs for this segment
        seg_slice = slice(chunk_start, chunk_start + end)
        seg_errors = errors[:end]
        seg_mismatch = mismatch_mask[:end]
        seg_dist = remaining_dist[:end]
        seg_motes = remaining_motes[:end]

        log_error[seg_slice] = seg_errors
        log_transmitted[seg_slice] = seg_mismatch

        # Energy: vectorized
        for j in range(end):
            gi = chunk_start + j
            d = seg_dist[j]
            tx = seg_mismatch[j]
            log_energy_proposed[gi] = energy_node_epoch(d, transmit=bool(tx))
            log_energy_baseline[gi] = energy_node_epoch(d, transmit=True)
            if tx:
                mid = seg_motes[j]
                if mid in node_energy:
                    node_energy[mid] -= log_energy_proposed[gi]
                buffer_X.append(remaining_X[j])
                buffer_y.append(remaining_y[j])
            else:
                mid = seg_motes[j]
                if mid in node_energy:
                    node_energy[mid] -= log_energy_proposed[gi]

        cumulative_mismatch += seg_mismatch.sum()
        chunk_end_global = chunk_start + end

        # Log mismatch rate
        rate = cumulative_mismatch / chunk_end_global
        mismatch_rate_over_time.append({
            "slot": chunk_end_global,
            "mismatch_rate": rate
        })

        # Retrain if buffer is full
        if retrain_idx is not None and len(buffer_X) >= buffer_min:
            buf_X_df = pd.DataFrame(np.array(buffer_X), columns=feat_cols)
            buf_y_arr = np.array(buffer_y)
            current_model = retrain_model(
                current_model, buf_X_df, buf_y_arr, feat_cols
            )
            retrain_events.append({
                "slot": chunk_end_global,
                "buffer_size": len(buffer_X),
                "cumulative_mismatch_rate": rate
            })
            buffer_X = []
            buffer_y = []

        chunk_start = chunk_end_global

    elapsed = time.time() - t0
    total_tx = log_transmitted.sum()
    cr = 1 - (total_tx / n_total)
    e_proposed = log_energy_proposed.sum()
    e_baseline = log_energy_baseline.sum()
    e_savings = 1 - (e_proposed / e_baseline) if e_baseline > 0 else 0

    print(f"  Done in {elapsed:.1f}s")

    return {
        "threshold": threshold,
        "buffer_min": buffer_min,
        "n_slots": n_total,
        "n_transmissions": int(total_tx),
        "communication_reduction": cr,
        "energy_proposed_J": e_proposed,
        "energy_baseline_J": e_baseline,
        "energy_savings": e_savings,
        "mismatch_rate": cumulative_mismatch / n_total,
        "mae": np.mean(log_error),
        "rmse": np.sqrt(np.mean(log_error**2)),
        "n_retrains": len(retrain_events),
        "retrain_frequency": len(retrain_events) / n_total,
        "node_energy_remaining": node_energy,
        "log_transmitted": log_transmitted,
        "log_error": log_error,
        "log_energy_proposed": log_energy_proposed,
        "log_energy_baseline": log_energy_baseline,
        "retrain_events": retrain_events,
        "mismatch_rate_over_time": mismatch_rate_over_time,
        "final_model": current_model
    }


def print_simulation_results(results):
    print("\n" + "=" * 65)
    print("SIMULATION RESULTS")
    print("=" * 65)
    print(f"  Time slots processed:   {results['n_slots']:,}")
    print(f"  Threshold:              {results['threshold']} C")
    print(f"  Buffer min:             {results['buffer_min']}")
    print(f"\n  COMMUNICATION:")
    print(f"    Transmissions:        {results['n_transmissions']:,} / "
          f"{results['n_slots']:,}")
    print(f"    Communication Red.:   {results['communication_reduction']*100:.2f}%")
    print(f"    Mismatch rate:        {results['mismatch_rate']*100:.2f}%")
    print(f"\n  ENERGY:")
    print(f"    Proposed total:       {results['energy_proposed_J']*1e3:.4f} mJ")
    print(f"    Baseline total:       {results['energy_baseline_J']*1e3:.4f} mJ")
    print(f"    Energy savings:       {results['energy_savings']*100:.2f}%")
    print(f"\n  PREDICTION QUALITY:")
    print(f"    MAE:                  {results['mae']:.4f} C")
    print(f"    RMSE:                 {results['rmse']:.4f} C")
    print(f"\n  RETRAINING:")
    print(f"    Retrain events:       {results['n_retrains']}")
    print(f"    Retrain frequency:    {results['retrain_frequency']*100:.4f}%")
    print("=" * 65)


def save_simulation_results(results, suffix=""):
    os.makedirs(METRICS_DIR, exist_ok=True)
    summary = {k: v for k, v in results.items()
               if not isinstance(v, (np.ndarray, list, dict))}
    summary.pop("final_model", None)
    summary.pop("node_energy_remaining", None)
    pd.DataFrame([summary]).to_csv(
        os.path.join(METRICS_DIR, f"simulation_results{suffix}.csv"),
        index=False
    )
