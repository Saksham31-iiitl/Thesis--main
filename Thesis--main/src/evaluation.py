"""
Step 10: Evaluation and Metrics Computation.

Computes all thesis evaluation metrics:
  - Communication Reduction (CR)
  - Energy Savings
  - MAE, RMSE, R2
  - Retraining Frequency
  - Network Lifetime (rounds until first node dies)
  - Data Staleness
  - Latency per cycle
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    T_SENSE, T_INFER, T_TX, E_INIT, METRICS_DIR,
    P_SERVER, T_RETRAIN, PACKET_SIZE_BITS
)


def compute_network_lifetime(node_energy_remaining, e_init=E_INIT):
    """
    Compute network lifetime metrics from remaining node energies.
    Returns: first node death, last node death, avg remaining.
    """
    energies = np.array(list(node_energy_remaining.values()))
    alive = energies > 0
    return {
        "alive_nodes": alive.sum(),
        "dead_nodes": (~alive).sum(),
        "min_energy_remaining": energies.min(),
        "max_energy_remaining": energies.max(),
        "avg_energy_remaining": energies.mean(),
        "energy_consumed_fraction": 1 - (energies.mean() / e_init)
    }


def compute_latency(results):
    """
    Compute per-cycle latency.
    T_cycle = t_sense + t_infer + t_compare + delta * t_tx
    """
    t_compare = 0.001  # comparison time (negligible)

    # For proposed: only transmitting slots have TX delay
    proposed_latencies = T_SENSE + T_INFER + t_compare + \
        results["log_transmitted"].astype(float) * T_TX

    # Baseline: every slot transmits
    baseline_latency = T_SENSE + t_compare + T_TX  # no inference in baseline

    return {
        "avg_latency_proposed": proposed_latencies.mean(),
        "avg_latency_baseline": baseline_latency,
        "max_latency_proposed": proposed_latencies.max(),
        "latency_per_slot": proposed_latencies
    }


def compute_staleness(log_transmitted, mote_ids):
    """
    Compute data staleness: how many slots since the server last
    received an update from each node.
    """
    unique_motes = np.unique(mote_ids)
    last_update = {m: 0 for m in unique_motes}
    staleness_all = []

    for i, (transmitted, mid) in enumerate(zip(log_transmitted, mote_ids)):
        if transmitted:
            last_update[mid] = i
        staleness = i - last_update[mid]
        staleness_all.append(staleness)

    staleness_arr = np.array(staleness_all)
    return {
        "avg_staleness": staleness_arr.mean(),
        "max_staleness": staleness_arr.max(),
        "median_staleness": np.median(staleness_arr),
        "staleness_array": staleness_arr
    }


def compute_server_energy(results):
    """
    Server-side energy for retraining (separate accounting from node energy).
    """
    n_retrains = results["n_retrains"]
    e_retrain_total = n_retrains * P_SERVER * T_RETRAIN
    return {
        "server_retrain_energy_J": e_retrain_total,
        "server_retrain_energy_mJ": e_retrain_total * 1e3,
        "n_retrains": n_retrains,
        "energy_per_retrain_J": P_SERVER * T_RETRAIN
    }


def generate_full_report(results, node_df):
    """Generate comprehensive evaluation report."""
    print("\n" + "=" * 65)
    print("COMPREHENSIVE EVALUATION REPORT")
    print("=" * 65)

    # Communication
    print("\n--- COMMUNICATION ---")
    print(f"  Total slots:            {results['n_slots']:,}")
    print(f"  Transmissions:          {results['n_transmissions']:,}")
    print(f"  Suppressed:             "
          f"{results['n_slots'] - results['n_transmissions']:,}")
    print(f"  Communication Red.:     "
          f"{results['communication_reduction']*100:.2f}%")

    # Energy (node-side)
    print("\n--- NODE ENERGY ---")
    print(f"  Proposed total:         "
          f"{results['energy_proposed_J']*1e3:.4f} mJ")
    print(f"  Baseline total:         "
          f"{results['energy_baseline_J']*1e3:.4f} mJ")
    print(f"  Energy savings:         "
          f"{results['energy_savings']*100:.2f}%")

    # Server energy
    server_e = compute_server_energy(results)
    print("\n--- SERVER ENERGY (separate accounting) ---")
    print(f"  Retraining events:      {server_e['n_retrains']}")
    print(f"  Energy per retrain:     "
          f"{server_e['energy_per_retrain_J']*1e3:.1f} mJ")
    print(f"  Total server energy:    "
          f"{server_e['server_retrain_energy_mJ']:.1f} mJ")

    # Prediction quality
    print("\n--- PREDICTION QUALITY ---")
    print(f"  MAE:                    {results['mae']:.4f} C")
    print(f"  RMSE:                   {results['rmse']:.4f} C")

    # Retraining
    print("\n--- CONTINUOUS LEARNING ---")
    print(f"  Retrain events:         {results['n_retrains']}")
    print(f"  Retrain frequency:      "
          f"{results['retrain_frequency']*100:.4f}%")

    # Latency
    latency = compute_latency(results)
    print("\n--- LATENCY ---")
    print(f"  Avg proposed:           "
          f"{latency['avg_latency_proposed']*1000:.3f} ms")
    print(f"  Baseline (always TX):   "
          f"{latency['avg_latency_baseline']*1000:.3f} ms")

    # Network lifetime
    if "node_energy_remaining" in results:
        lifetime = compute_network_lifetime(results["node_energy_remaining"])
        print("\n--- NETWORK LIFETIME ---")
        print(f"  Alive nodes:            {lifetime['alive_nodes']}")
        print(f"  Energy consumed:        "
              f"{lifetime['energy_consumed_fraction']*100:.4f}%")

    # Staleness
    if "log_transmitted" in results and node_df is not None:
        stale = compute_staleness(
            results["log_transmitted"], node_df["mote_id"].values
        )
        print("\n--- DATA STALENESS ---")
        print(f"  Avg staleness:          {stale['avg_staleness']:.1f} slots")
        print(f"  Max staleness:          {stale['max_staleness']} slots")
        print(f"  Median staleness:       {stale['median_staleness']:.1f} slots")

    print("\n" + "=" * 65)

    return {
        "latency": latency,
        "server_energy": server_e,
        "lifetime": lifetime if "node_energy_remaining" in results else None,
        "staleness": stale if "log_transmitted" in results else None
    }
