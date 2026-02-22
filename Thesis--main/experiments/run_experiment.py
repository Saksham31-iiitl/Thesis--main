"""
Step 12: Master Experiment Runner.

Orchestrates the full thesis simulation pipeline:
  Phase 1: Data loading and partitioning
  Phase 2: Spatio-temporal clustering + CH selection
  Phase 3: TinyML model training
  Phase 4: Parallel node-server simulation
  Phase 5: Energy and communication analysis
  Phase 6: Threshold analysis (static + adaptive)
  Phase 7: Comprehensive evaluation
  Phase 8: Visualization (all thesis plots)
"""

import pandas as pd
import numpy as np
import time
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.config import (
    SERVER_DATA_PATH, NODE_DATA_PATH, THRESHOLD_DEFAULT,
    BUFFER_MIN, METRICS_DIR
)
from src.data_loader import run_data_pipeline
from src.spatio_temporal_clustering import run_clustering
from src.cluster_head_selection import run_ch_selection
from src.tinyml_model import run_model_training
from src.node_simulation import (
    run_simulation, print_simulation_results, save_simulation_results
)
from src.threshold_analysis import run_threshold_analysis
from src.evaluation import generate_full_report, compute_staleness
from experiments.visualization import generate_all_plots


def main():
    """Run the complete thesis experiment."""
    print("=" * 65)
    print("  M.Tech THESIS EXPERIMENT")
    print("  Energy Efficient Data Transmission in WSN-assisted IoT")
    print("  using Spatio-Temporal Clustering and TinyML")
    print("=" * 65)

    t_start = time.time()

    # ── Phase 1: Data ────────────────────────────────────────────
    print("\n>>> PHASE 1: Data Loading and Partitioning")
    if os.path.exists(SERVER_DATA_PATH) and os.path.exists(NODE_DATA_PATH):
        print("  Partitioned data files already exist. Loading...")
        server_df = pd.read_csv(SERVER_DATA_PATH)
        node_df = pd.read_csv(NODE_DATA_PATH)
        print(f"  Server: {len(server_df):,} | Node: {len(node_df):,}")
    else:
        server_df, node_df = run_data_pipeline()

    # ── Phase 2: Clustering + CH ─────────────────────────────────
    print("\n>>> PHASE 2: Spatio-Temporal Clustering + CH Selection")
    motes, profiles, clustering_results = run_clustering(server_df)
    nodes = run_ch_selection(motes)

    # ── Phase 3: TinyML Model ───────────────────────────────────
    print("\n>>> PHASE 3: TinyML Model Training")
    model, feat_cols, train_metrics, eval_metrics = run_model_training(
        server_df, node_df
    )

    # ── Phase 4: Simulation ─────────────────────────────────────
    print("\n>>> PHASE 4: Parallel Node-Server Simulation")
    sim_results = run_simulation(
        model=model,
        feat_cols=feat_cols,
        nodes=nodes,
        node_df=node_df,
        threshold=THRESHOLD_DEFAULT,
        buffer_min=BUFFER_MIN
    )
    print_simulation_results(sim_results)
    save_simulation_results(sim_results)

    # ── Phase 5: Already computed in simulation ──────────────────
    # Energy and communication metrics are part of sim_results

    # ── Phase 6: Threshold Analysis ──────────────────────────────
    print("\n>>> PHASE 6: Threshold Analysis")
    sweep_df, adaptive_results = run_threshold_analysis(
        model, feat_cols, node_df
    )

    # ── Phase 7: Comprehensive Evaluation ────────────────────────
    print("\n>>> PHASE 7: Comprehensive Evaluation")
    eval_extras = generate_full_report(sim_results, node_df)

    # Compute staleness for plotting
    staleness_info = compute_staleness(
        sim_results["log_transmitted"],
        node_df["mote_id"].values
    )

    # ── Phase 8: Visualization ──────────────────────────────────
    print("\n>>> PHASE 8: Generating All Plots")
    generate_all_plots(
        sim_results=sim_results,
        nodes=nodes,
        sweep_df=sweep_df,
        adaptive_results=adaptive_results,
        staleness_array=staleness_info["staleness_array"]
    )

    # ── Final Summary ────────────────────────────────────────────
    elapsed = time.time() - t_start
    print("\n" + "=" * 65)
    print("  EXPERIMENT COMPLETE")
    print("=" * 65)
    print(f"  Total time: {elapsed:.1f}s")
    print(f"\n  KEY RESULTS:")
    print(f"    Communication Reduction: "
          f"{sim_results['communication_reduction']*100:.2f}%")
    print(f"    Energy Savings:          "
          f"{sim_results['energy_savings']*100:.2f}%")
    print(f"    MAE:                     {sim_results['mae']:.4f} C")
    print(f"    Retraining Events:       {sim_results['n_retrains']}")
    print(f"    Retrain Frequency:       "
          f"{sim_results['retrain_frequency']*100:.4f}%")
    print(f"\n  Output saved to:")
    print(f"    Metrics: {METRICS_DIR}")
    print(f"    Figures: {os.path.join(project_root, 'results', 'figures')}")
    print("=" * 65)


if __name__ == "__main__":
    main()
