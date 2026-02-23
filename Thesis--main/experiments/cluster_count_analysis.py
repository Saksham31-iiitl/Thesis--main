"""
Cluster Count (K) Analysis for Spatio-Temporal Clustering.

Tests K = 2 through 8 and evaluates each with:
  - Silhouette Score        (higher = better compact, well-separated clusters)
  - Davies-Bouldin Index    (lower  = better)
  - Intra-cluster temperature variance  (lower = nodes in same cluster behave similarly)
  - Cluster size balance    (std of cluster sizes — lower = more balanced)
  - Spatial coherence violations after post-processing

Generates:
  results/figures/cluster_k_silhouette.png
  results/figures/cluster_k_davies_bouldin.png
  results/figures/cluster_k_variance.png
  results/figures/cluster_k_combined.png
  results/figures/cluster_k_layouts.png  (shows cluster maps for K=3,4,5,6)
  results/metrics/cluster_count_analysis.csv
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings

warnings.filterwarnings("ignore")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

from src.config import (
    MOTE_LOCS_PATH, SERVER_DATA_PATH,
    SPATIAL_WEIGHT_ALPHA, CLUSTERING_TEMPORAL_FEATURES,
    RANDOM_SEED, METRICS_DIR, FIGURES_DIR
)
from src.spatio_temporal_clustering import (
    load_mote_locations,
    extract_temporal_profiles,
    _enforce_spatial_coherence,
    compute_intra_cluster_temp_variance,
)

K_RANGE = range(2, 9)   # K = 2, 3, 4, 5, 6, 7, 8
K_SELECTED = 4          # Current choice


def build_combined_features(motes, profiles_df, alpha=SPATIAL_WEIGHT_ALPHA):
    """Rebuild combined feature matrix (same as main clustering pipeline)."""
    merged = motes.merge(profiles_df, on="mote_id")
    spatial_cols  = ["x", "y"]
    temporal_cols = CLUSTERING_TEMPORAL_FEATURES
    feature_cols  = spatial_cols + temporal_cols

    X = merged[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled[:, :len(spatial_cols)] *= alpha   # spatial weight

    return X_scaled, feature_cols


def run_k_analysis(server_df, motes, profiles_df):
    """Run K-Means for each K and collect metrics."""
    X_combined, feature_cols = build_combined_features(motes, profiles_df)
    coords = motes[["x", "y"]].values

    print(f"\n{'K':>3}  {'Silhouette':>11}  {'DaviesBouldin':>13}  "
          f"{'IntraVar':>10}  {'SizeStd':>8}  {'CoherFix':>9}")
    print("-" * 60)

    records = []
    labels_per_k = {}

    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=15)
        labels = km.fit_predict(X_combined)

        # Spatial coherence post-processing (same as main pipeline)
        labels_post = _enforce_spatial_coherence(labels, coords, k)

        # Clustering quality metrics
        sil = silhouette_score(X_combined, labels_post)
        db  = davies_bouldin_score(X_combined, labels_post)

        # Downstream task: intra-cluster temperature variance
        motes_tmp = motes.copy()
        motes_tmp["cluster"] = labels_post
        intra_var = compute_intra_cluster_temp_variance(server_df, motes_tmp,
                                                         labels_post)

        # Balance: std of cluster sizes (lower = more balanced)
        unique, counts = np.unique(labels_post, return_counts=True)
        size_std = counts.std()
        size_min = counts.min()
        size_max = counts.max()

        # How many nodes were reassigned by coherence enforcement
        n_fixed = int((labels_post != labels).sum())

        print(f"{k:>3}  {sil:>11.4f}  {db:>13.4f}  "
              f"{intra_var:>10.4f}  {size_std:>8.2f}  {n_fixed:>9}")

        records.append({
            "k":              k,
            "silhouette":     sil,
            "davies_bouldin": db,
            "intra_temp_var": intra_var,
            "size_std":       size_std,
            "size_min":       size_min,
            "size_max":       size_max,
            "n_coherence_fixed": n_fixed,
            "cluster_sizes":  counts.tolist(),
        })
        labels_per_k[k] = labels_post

    df = pd.DataFrame(records)
    return df, labels_per_k, X_combined


def _add_selected_marker(ax, df, y_col, k_sel=K_SELECTED):
    """Mark the selected K on an axis."""
    row = df[df["k"] == k_sel].iloc[0]
    ax.axvline(k_sel, color="#3498db", linestyle="--", linewidth=1.5,
               alpha=0.7, label=f"Selected K={k_sel}")
    ax.plot(k_sel, row[y_col], "o", color="#3498db", markersize=10, zorder=5)


def plot_combined(df):
    """4-panel combined figure: the main figure to show in thesis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Cluster Count (K) Analysis — Spatio-Temporal Clustering\n"
        "Tested on Intel Berkeley Lab dataset, 54 sensor nodes",
        fontsize=14, fontweight="bold"
    )

    ks = df["k"].values

    # ── Silhouette ────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(ks, df["silhouette"], "o-", color="#2ecc71", linewidth=2,
            markersize=7, label="Silhouette")
    _add_selected_marker(ax, df, "silhouette")
    ax.set_xlabel("Number of Clusters (K)", fontsize=11)
    ax.set_ylabel("Silhouette Score", fontsize=11)
    ax.set_title("Silhouette Score (higher = better)", fontsize=11, fontweight="bold")
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9)
    # Annotate best
    best_k = df.loc[df["silhouette"].idxmax(), "k"]
    ax.annotate(f"Best K={best_k}",
                xy=(best_k, df.loc[df["silhouette"].idxmax(), "silhouette"]),
                xytext=(best_k + 0.3, df["silhouette"].max() - 0.01),
                fontsize=8, color="green",
                arrowprops=dict(arrowstyle="->", color="green"))

    # ── Davies-Bouldin ────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(ks, df["davies_bouldin"], "s-", color="#e74c3c", linewidth=2,
            markersize=7, label="Davies-Bouldin")
    _add_selected_marker(ax, df, "davies_bouldin")
    ax.set_xlabel("Number of Clusters (K)", fontsize=11)
    ax.set_ylabel("Davies-Bouldin Index", fontsize=11)
    ax.set_title("Davies-Bouldin Index (lower = better)", fontsize=11, fontweight="bold")
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9)
    best_k = df.loc[df["davies_bouldin"].idxmin(), "k"]
    ax.annotate(f"Best K={best_k}",
                xy=(best_k, df.loc[df["davies_bouldin"].idxmin(), "davies_bouldin"]),
                xytext=(best_k + 0.3, df["davies_bouldin"].min() + 0.05),
                fontsize=8, color="red",
                arrowprops=dict(arrowstyle="->", color="red"))

    # ── Intra-cluster temperature variance ────────────────────────
    ax = axes[1, 0]
    ax.plot(ks, df["intra_temp_var"], "^-", color="#9b59b6", linewidth=2,
            markersize=7, label="Intra-cluster temp var")
    _add_selected_marker(ax, df, "intra_temp_var")
    ax.set_xlabel("Number of Clusters (K)", fontsize=11)
    ax.set_ylabel("Avg Intra-Cluster Temp Variance (°C²)", fontsize=11)
    ax.set_title("Intra-Cluster Temperature Variance\n(lower = nodes behave more similarly)",
                 fontsize=11, fontweight="bold")
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9)

    # ── Cluster size balance ──────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(ks, df["size_std"], "D-", color="#f39c12", linewidth=2,
            markersize=7, label="Size std dev")
    ax.fill_between(ks, df["size_min"], df["size_max"],
                    alpha=0.15, color="#f39c12", label="Min–Max cluster size")
    _add_selected_marker(ax, df, "size_std")
    ax.set_xlabel("Number of Clusters (K)", fontsize=11)
    ax.set_ylabel("Cluster Size (nodes)", fontsize=11)
    ax.set_title("Cluster Balance\n(std dev of cluster sizes, lower = more balanced)",
                 fontsize=11, fontweight="bold")
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "cluster_k_combined.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: cluster_k_combined.png")


def plot_cluster_layouts(motes, labels_per_k, show_ks=(3, 4, 5, 6)):
    """Side-by-side spatial maps of cluster assignments for selected K values."""
    n = len(show_ks)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    cmap = cm.get_cmap("tab10")

    for ax, k in zip(axes, show_ks):
        labels = labels_per_k[k]
        motes_tmp = motes.copy()
        motes_tmp["cluster"] = labels

        for c in sorted(np.unique(labels)):
            sub = motes_tmp[motes_tmp["cluster"] == c]
            centroid = sub[["x", "y"]].mean()
            ax.scatter(sub["x"], sub["y"], s=100, color=cmap(c),
                       edgecolors="black", linewidths=0.5,
                       zorder=3, label=f"Cluster {c}")
            ax.plot(centroid["x"], centroid["y"], "*",
                    color=cmap(c), markersize=14,
                    markeredgecolor="black", markeredgewidth=0.8, zorder=4)

            # Annotate each node with its mote_id
            for _, row in sub.iterrows():
                ax.annotate(str(int(row["mote_id"])),
                            (row["x"], row["y"]),
                            textcoords="offset points",
                            xytext=(4, 4), fontsize=5, alpha=0.7)

        ax.set_title(f"K = {k}", fontsize=12, fontweight="bold")
        ax.set_xlabel("X (m)", fontsize=9)
        ax.set_ylabel("Y (m)", fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        # Add cluster sizes to legend
        unique, counts = np.unique(labels, return_counts=True)
        handles = [plt.scatter([], [], s=60, color=cmap(c), edgecolors="black",
                               linewidths=0.5, label=f"C{c}: {cnt} nodes")
                   for c, cnt in zip(unique, counts)]
        ax.legend(handles=handles, fontsize=7, frameon=True, loc="best")

    fig.suptitle(
        "Cluster Spatial Layouts for Different K Values\n"
        "(Stars = cluster centroids, numbers = mote IDs)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "cluster_k_layouts.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: cluster_k_layouts.png")


def plot_recommendation(df):
    """Bar chart of composite cluster quality score across K values."""
    d = df.copy()

    # Normalize each metric (higher=better after normalization)
    def norm_higher(s): return (s - s.min()) / (s.max() - s.min() + 1e-9)
    def norm_lower(s):  return 1 - norm_higher(s)

    d["n_sil"]  = norm_higher(d["silhouette"])
    d["n_db"]   = norm_lower(d["davies_bouldin"])
    d["n_var"]  = norm_lower(d["intra_temp_var"])
    d["n_bal"]  = norm_lower(d["size_std"])

    # Weighted composite: quality (60%) + balance (40%)
    d["composite"] = (
        0.25 * d["n_sil"] +
        0.25 * d["n_db"]  +
        0.25 * d["n_var"] +
        0.25 * d["n_bal"]
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#3498db" if k == K_SELECTED else "#bdc3c7" for k in d["k"]]
    bars = ax.bar(d["k"].astype(str), d["composite"], color=colors,
                  edgecolor="white", linewidth=0.8, zorder=3)

    ax.set_xlabel("Number of Clusters (K)", fontsize=12)
    ax.set_ylabel("Composite Cluster Quality Score\n"
                  "(Silhouette 25% + Davies-Bouldin 25% + Temp-Var 25% + Balance 25%)",
                  fontsize=11)
    ax.set_title("Overall Cluster Quality Score Across K Values\n"
                 "(Higher = better balance of all criteria)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.35, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    for bar, val, k in zip(bars, d["composite"], d["k"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=9,
                fontweight="bold" if k == K_SELECTED else "normal")

    best_k_idx = d["composite"].idxmax()
    best_k     = d.loc[best_k_idx, "k"]
    ax.annotate(
        f"Best composite\nK = {best_k}",
        xy=(d.loc[best_k_idx, "k"] - d["k"].min(),
            d.loc[best_k_idx, "composite"]),
        xytext=(d.loc[best_k_idx, "k"] - d["k"].min() + 0.5,
                d.loc[best_k_idx, "composite"] + 0.12),
        arrowprops=dict(arrowstyle="->", color="navy"),
        fontsize=10, color="navy", fontweight="bold", ha="center"
    )

    if best_k != K_SELECTED:
        note = (f"Note: K={best_k} scores highest overall. "
                f"Current K={K_SELECTED} chosen for practical balance\n"
                f"(manageable cluster head overhead + sufficiently large clusters for reliable CH selection)")
    else:
        note = f"K={K_SELECTED} achieves the best composite score — selection confirmed."

    fig.text(0.5, -0.04, note, ha="center", fontsize=9,
             style="italic", wrap=True)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "cluster_k_recommendation.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: cluster_k_recommendation.png")

    return best_k, d[["k", "composite"]]


def run_cluster_count_analysis():
    print("=" * 65)
    print("CLUSTER COUNT (K) ANALYSIS")
    print("=" * 65)

    motes = load_mote_locations()
    print(f"  Loaded {len(motes)} mote locations")

    server_df = pd.read_csv(SERVER_DATA_PATH)
    print(f"  Loaded server data: {len(server_df):,} samples")

    from src.spatio_temporal_clustering import extract_temporal_profiles
    profiles_df = extract_temporal_profiles(server_df, motes)

    print(f"\nRunning K-Means for K = {list(K_RANGE)}...")
    df_metrics, labels_per_k, X_combined = run_k_analysis(
        server_df, motes, profiles_df
    )

    # ── Save CSV ──────────────────────────────────────────────────
    os.makedirs(METRICS_DIR, exist_ok=True)
    save_df = df_metrics.drop(columns=["cluster_sizes"])
    save_df.to_csv(
        os.path.join(METRICS_DIR, "cluster_count_analysis.csv"), index=False
    )
    print(f"\n  Saved: cluster_count_analysis.csv")

    # ── Plots ──────────────────────────────────────────────────────
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plot_combined(df_metrics)
    plot_cluster_layouts(motes, labels_per_k, show_ks=(3, 4, 5, 6))
    best_k, score_df = plot_recommendation(df_metrics)

    # ── Print conclusion ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("CONCLUSION")
    print("=" * 65)
    row4  = df_metrics[df_metrics["k"] == K_SELECTED].iloc[0]
    best  = df_metrics[df_metrics["k"] == best_k].iloc[0]

    print(f"\n  Current selection K={K_SELECTED}:")
    print(f"    Silhouette:          {row4['silhouette']:.4f}")
    print(f"    Davies-Bouldin:      {row4['davies_bouldin']:.4f}")
    print(f"    Intra-temp variance: {row4['intra_temp_var']:.4f}")
    print(f"    Cluster sizes:       {df_metrics[df_metrics['k']==K_SELECTED]['cluster_sizes'].values[0]}")

    if best_k != K_SELECTED:
        print(f"\n  Best composite score: K={best_k}")
        print(f"    Silhouette:          {best['silhouette']:.4f}")
        print(f"    Davies-Bouldin:      {best['davies_bouldin']:.4f}")
        print(f"    Intra-temp variance: {best['intra_temp_var']:.4f}")
        print(f"    Cluster sizes:       {df_metrics[df_metrics['k']==best_k]['cluster_sizes'].values[0]}")
        print(f"\n  Recommendation: Consider switching to K={best_k}.")
        print(f"  However with 54 nodes K={best_k} may produce clusters too small")
        print(f"  for reliable CH selection (need >=8 nodes per cluster).")
        min_size = df_metrics[df_metrics["k"] == best_k]["size_min"].values[0]
        if min_size < 8:
            print(f"  WARNING: K={best_k} has a cluster with only {min_size} nodes.")
            print(f"  This is too small for reliable CH rotation. K={K_SELECTED} is safer.")
        else:
            print(f"  K={best_k} has min cluster size {min_size} — acceptable.")
    else:
        print(f"\n  K={K_SELECTED} achieves best composite score. Selection confirmed.")

    print("=" * 65)
    return df_metrics, labels_per_k, best_k


if __name__ == "__main__":
    df_m, lbl_map, best = run_cluster_count_analysis()
