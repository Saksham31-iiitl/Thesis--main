"""
Step 11: Visualization Module.

Generates all thesis-grade plots:
  1.  Communication Reduction vs Threshold
  2.  Energy Savings vs Threshold
  3.  MAE vs Threshold
  4.  Cumulative Energy: Proposed vs Baseline
  5.  Communication Events Over Time
  6.  Retraining Events Over Time
  7.  Mismatch Rate Over Time
  8.  Spatio-Temporal Cluster Visualization
  9.  Per-Cluster Energy Distribution
  10. Staleness Distribution
  11. Latency Comparison
  12. Adaptive vs Static Threshold
  13. Network Lifetime
  14. Spatial vs Spatio-Temporal Clustering Comparison
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.config import FIGURES_DIR, BS_LOCATION

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.bbox": "tight"
})


def save_fig(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, f"{name}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_threshold_sweep(sweep_df):
    """Plot 1, 2, 3: CR, Energy Savings, MAE vs Threshold."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # CR vs Threshold
    axes[0].plot(sweep_df["threshold"],
                 sweep_df["communication_reduction"] * 100,
                 "b-o", linewidth=2, markersize=6)
    axes[0].set_xlabel("Threshold (C)")
    axes[0].set_ylabel("Communication Reduction (%)")
    axes[0].set_title("Communication Reduction vs Threshold")
    axes[0].grid(True, alpha=0.3)

    # Mismatch Rate vs Threshold
    axes[1].plot(sweep_df["threshold"],
                 sweep_df["mismatch_rate"] * 100,
                 "r-s", linewidth=2, markersize=6)
    axes[1].set_xlabel("Threshold (C)")
    axes[1].set_ylabel("Mismatch Rate (%)")
    axes[1].set_title("Mismatch Rate vs Threshold")
    axes[1].grid(True, alpha=0.3)

    # MAE vs Threshold (constant since predictions don't change)
    axes[2].plot(sweep_df["threshold"],
                 sweep_df["mae"],
                 "g-^", linewidth=2, markersize=6)
    axes[2].set_xlabel("Threshold (C)")
    axes[2].set_ylabel("MAE (C)")
    axes[2].set_title("Prediction Error (MAE)")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Static Threshold Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "01_threshold_sweep")


def plot_cumulative_energy(results):
    """Plot 4: Cumulative energy proposed vs baseline."""
    fig, ax = plt.subplots(figsize=(10, 5))
    cum_proposed = np.cumsum(results["log_energy_proposed"]) * 1e3
    cum_baseline = np.cumsum(results["log_energy_baseline"]) * 1e3

    # Downsample for plotting
    step = max(1, len(cum_proposed) // 2000)
    x = np.arange(0, len(cum_proposed), step)

    ax.plot(x, cum_baseline[x], "r-", label="Baseline (always TX)", linewidth=1.5)
    ax.plot(x, cum_proposed[x], "b-", label="Proposed (TinyML + suppression)",
            linewidth=1.5)
    ax.fill_between(x, cum_proposed[x], cum_baseline[x],
                    alpha=0.15, color="green", label="Energy Saved")
    ax.set_xlabel("Time Slot")
    ax.set_ylabel("Cumulative Energy (mJ)")
    ax.set_title("Cumulative Energy: Proposed vs Baseline")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(fig, "02_cumulative_energy")


def plot_communication_events(results, window=500):
    """Plot 5: Communication events over time (smoothed)."""
    fig, ax = plt.subplots(figsize=(10, 4))
    transmitted = results["log_transmitted"].astype(float)
    # Sliding window average
    smoothed = pd.Series(transmitted).rolling(window=window, min_periods=1).mean()

    step = max(1, len(smoothed) // 2000)
    x = np.arange(0, len(smoothed), step)

    ax.plot(x, smoothed.values[x] * 100, "b-", linewidth=1)
    ax.set_xlabel("Time Slot")
    ax.set_ylabel("Transmission Rate (%, smoothed)")
    ax.set_title(f"Communication Rate Over Time (window={window})")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    # Mark retraining events
    for evt in results.get("retrain_events", []):
        ax.axvline(x=evt["slot"], color="red", alpha=0.3, linewidth=0.5)

    save_fig(fig, "03_communication_events")


def plot_retraining_events(results):
    """Plot 6: Cumulative retraining events over time."""
    fig, ax = plt.subplots(figsize=(10, 4))
    events = results.get("retrain_events", [])
    if not events:
        ax.text(0.5, 0.5, "No retraining events", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
    else:
        slots = [e["slot"] for e in events]
        cumulative = np.arange(1, len(slots) + 1)
        ax.step(slots, cumulative, "r-", linewidth=2)
        ax.set_xlabel("Time Slot")
        ax.set_ylabel("Cumulative Retraining Events")
        ax.set_title("Retraining Events Over Time")
        ax.grid(True, alpha=0.3)

    save_fig(fig, "04_retraining_events")


def plot_mismatch_over_time(results):
    """Plot 7: Mismatch rate over time."""
    fig, ax = plt.subplots(figsize=(10, 4))
    rate_data = results.get("mismatch_rate_over_time", [])
    if rate_data:
        slots = [r["slot"] for r in rate_data]
        rates = [r["mismatch_rate"] * 100 for r in rate_data]
        ax.plot(slots, rates, "b-o", markersize=3, linewidth=1.5)
        ax.set_xlabel("Time Slot")
        ax.set_ylabel("Cumulative Mismatch Rate (%)")
        ax.set_title("Mismatch Rate Evolution (with Retraining)")
        ax.grid(True, alpha=0.3)

    save_fig(fig, "05_mismatch_over_time")


def plot_clusters(nodes, title_suffix=""):
    """Plot 8: Spatio-temporal cluster visualization."""
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00",
              "#984ea3", "#a65628"]

    for c in sorted(nodes["cluster"].unique()):
        cluster = nodes[nodes["cluster"] == c]
        members = cluster[~cluster["is_ch"]]
        chs = cluster[cluster["is_ch"]]

        ax.scatter(members["x"], members["y"], c=colors[c % len(colors)],
                   s=80, label=f"Cluster {c} ({len(cluster)} nodes)",
                   alpha=0.7, edgecolors="black", linewidth=0.5)
        ax.scatter(chs["x"], chs["y"], c=colors[c % len(colors)],
                   s=250, marker="*", edgecolors="black", linewidth=1.5,
                   zorder=5)

    # Base station
    ax.scatter(BS_LOCATION[0], BS_LOCATION[1], c="black", s=200,
               marker="s", label="Base Station", zorder=5)

    ax.set_xlabel("X Coordinate (m)")
    ax.set_ylabel("Y Coordinate (m)")
    ax.set_title(f"Spatio-Temporal Clustering with Cluster Heads{title_suffix}")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    save_fig(fig, "06_cluster_visualization")


def plot_clustering_comparison(nodes):
    """Plot 14: Spatial vs spatio-temporal clustering side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00"]

    for ax, col, title in [
        (axes[0], "spatial_cluster", "Spatial-Only K-Means"),
        (axes[1], "cluster", "Spatio-Temporal K-Means (Proposed)")
    ]:
        if col not in nodes.columns:
            continue
        for c in sorted(nodes[col].unique()):
            cluster = nodes[nodes[col] == c]
            ax.scatter(cluster["x"], cluster["y"], c=colors[c % len(colors)],
                       s=80, label=f"Cluster {c}", alpha=0.7,
                       edgecolors="black", linewidth=0.5)
        ax.scatter(BS_LOCATION[0], BS_LOCATION[1], c="black", s=200,
                   marker="s", label="BS", zorder=5)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(title)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Clustering Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "07_clustering_comparison")


def plot_staleness_distribution(staleness_array):
    """Plot 10: Staleness distribution histogram."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(staleness_array, bins=50, color="steelblue", edgecolor="black",
            alpha=0.7)
    ax.axvline(np.mean(staleness_array), color="red", linestyle="--",
               label=f"Mean: {np.mean(staleness_array):.1f}")
    ax.axvline(np.median(staleness_array), color="green", linestyle="--",
               label=f"Median: {np.median(staleness_array):.1f}")
    ax.set_xlabel("Staleness (slots since last update)")
    ax.set_ylabel("Frequency")
    ax.set_title("Data Staleness Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(fig, "08_staleness_distribution")


def plot_energy_comparison_bar(results):
    """Plot: Energy and communication summary bars."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Energy
    labels = ["Baseline\n(Always TX)", "Proposed\n(TinyML)"]
    energies = [results["energy_baseline_J"] * 1e3,
                results["energy_proposed_J"] * 1e3]
    bars = axes[0].bar(labels, energies, color=["#e41a1c", "#377eb8"],
                       edgecolor="black")
    axes[0].set_ylabel("Total Energy (mJ)")
    axes[0].set_title("Energy Consumption Comparison")
    axes[0].grid(axis="y", alpha=0.3)
    savings_pct = results["energy_savings"] * 100
    axes[0].annotate(f"{savings_pct:.1f}% savings",
                     xy=(1, energies[1]), xytext=(1.3, max(energies) * 0.8),
                     fontsize=12, fontweight="bold", color="green",
                     arrowprops=dict(arrowstyle="->", color="green"))

    # Communication
    tx = results["n_transmissions"]
    suppressed = results["n_slots"] - tx
    axes[1].bar(["Transmitted", "Suppressed"], [tx, suppressed],
                color=["#e41a1c", "#4daf4a"], edgecolor="black")
    axes[1].set_ylabel("Number of Events")
    axes[1].set_title("Communication Events")
    axes[1].grid(axis="y", alpha=0.3)
    cr_pct = results["communication_reduction"] * 100
    axes[1].annotate(f"{cr_pct:.1f}% reduced",
                     xy=(1, suppressed), xytext=(0.5, suppressed * 0.9),
                     fontsize=12, fontweight="bold", color="green")

    plt.tight_layout()
    save_fig(fig, "09_energy_comm_summary")


def plot_adaptive_threshold(adaptive_results, n_points=2000):
    """Plot 12: Adaptive threshold behavior over time."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    thresholds = adaptive_results["log_threshold"]
    errors = adaptive_results["log_error"]
    step = max(1, len(thresholds) // n_points)
    x = np.arange(0, len(thresholds), step)

    # Threshold over time
    axes[0].plot(x, thresholds[x], "b-", linewidth=0.8, label="Adaptive Threshold")
    axes[0].plot(x, errors[x], "r.", alpha=0.1, markersize=1, label="Prediction Error")
    axes[0].set_ylabel("Value (C)")
    axes[0].set_title("Adaptive Threshold vs Prediction Error")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Transmission decisions
    transmitted = adaptive_results["log_transmitted"].astype(float)
    smoothed = pd.Series(transmitted).rolling(window=500, min_periods=1).mean()
    axes[1].plot(x, smoothed.values[x] * 100, "b-", linewidth=1)
    axes[1].set_xlabel("Time Slot")
    axes[1].set_ylabel("TX Rate (%, smoothed)")
    axes[1].set_title("Communication Rate with Adaptive Threshold")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, "10_adaptive_threshold")


def generate_all_plots(sim_results, nodes, sweep_df=None,
                       adaptive_results=None, staleness_array=None):
    """Generate all thesis plots."""
    print("\n" + "=" * 65)
    print("GENERATING THESIS PLOTS")
    print("=" * 65)

    if sweep_df is not None:
        plot_threshold_sweep(sweep_df)
    plot_cumulative_energy(sim_results)
    plot_communication_events(sim_results)
    plot_retraining_events(sim_results)
    plot_mismatch_over_time(sim_results)
    plot_clusters(nodes)
    if "spatial_cluster" in nodes.columns:
        plot_clustering_comparison(nodes)
    plot_energy_comparison_bar(sim_results)
    if staleness_array is not None:
        plot_staleness_distribution(staleness_array)
    if adaptive_results is not None:
        plot_adaptive_threshold(adaptive_results)

    print("\nAll plots saved to:", FIGURES_DIR)
    print("=" * 65)
