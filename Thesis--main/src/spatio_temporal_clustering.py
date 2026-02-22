"""
Step 2: Spatio-Temporal Clustering of WSN Nodes.

Clusters 54 sensor nodes using BOTH:
  - Spatial features: physical (x, y) coordinates
  - Temporal features: per-node behavioral patterns extracted from
    server training data (std_temp, diurnal_amplitude, light_mean)

This is the KEY NOVELTY vs. purely spatial clustering (LEACH, etc.).

Two-phase approach:
  Phase 1: Weighted K-Means on combined spatio-temporal features
  Phase 2: Spatial coherence enforcement — fixes isolated outlier nodes
           by majority-vote of their K nearest spatial neighbors

Fair evaluation: both methods evaluated in the SAME combined feature space.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    MOTE_LOCS_PATH, SERVER_DATA_PATH, NUM_CLUSTERS,
    RANDOM_SEED, FIGURES_DIR, METRICS_DIR,
    SPATIAL_WEIGHT_ALPHA, CLUSTERING_TEMPORAL_FEATURES
)

# Spatial coherence parameters
COHERENCE_KNN = 3          # Number of nearest spatial neighbors to check
COHERENCE_MIN_AGREE = 3    # Minimum neighbors that must agree to reassign
COHERENCE_MIN_CLUSTER = 8  # Don't shrink any cluster below this size


def load_mote_locations(path=MOTE_LOCS_PATH):
    """Load physical coordinates of 54 sensor nodes."""
    motes = pd.read_csv(
        path, sep=r"\s+", header=None, names=["mote_id", "x", "y"]
    )
    return motes


def extract_temporal_profiles(server_df, motes):
    """
    Extract per-node temporal behavior profiles from server training data.

    For each node, compute:
      - mean_temp: average temperature reading
      - std_temp: temperature variability (CV=27.9%)
      - mean_humidity: average humidity
      - diurnal_amplitude: day-night temperature swing (CV=35.4%)
      - light_mean: average light exposure (CV=44.6% — best discriminator)
    """
    print("Extracting per-node temporal profiles...")

    # Ensure numeric columns
    for col in ["temperature", "humidity", "light"]:
        if col in server_df.columns:
            server_df[col] = pd.to_numeric(server_df[col], errors="coerce")

    profiles = []
    for mote_id in motes["mote_id"].values:
        node_data = server_df[server_df["mote_id"] == mote_id]

        if len(node_data) < 10:
            profiles.append({
                "mote_id": mote_id,
                "mean_temp": np.nan,
                "std_temp": np.nan,
                "mean_humidity": np.nan,
                "diurnal_amplitude": np.nan,
                "light_mean": np.nan
            })
            continue

        mean_temp = node_data["temperature"].mean()
        std_temp = node_data["temperature"].std()
        mean_humidity = node_data["humidity"].mean()
        light_mean = node_data["light"].mean()

        # Diurnal amplitude: daytime (8-20h) vs nighttime (0-7, 21-23)
        day_mask = node_data["hour"].between(8, 20)
        day_temp = node_data.loc[day_mask, "temperature"].mean()
        night_temp = node_data.loc[~day_mask, "temperature"].mean()
        diurnal_amp = abs(day_temp - night_temp) if not (
            np.isnan(day_temp) or np.isnan(night_temp)
        ) else np.nan

        profiles.append({
            "mote_id": mote_id,
            "mean_temp": mean_temp,
            "std_temp": std_temp,
            "mean_humidity": mean_humidity,
            "diurnal_amplitude": diurnal_amp,
            "light_mean": light_mean
        })

    profiles_df = pd.DataFrame(profiles)

    # Impute missing values with median
    for col in ["mean_temp", "std_temp", "mean_humidity",
                 "diurnal_amplitude", "light_mean"]:
        median_val = profiles_df[col].median()
        n_missing = profiles_df[col].isna().sum()
        if n_missing > 0:
            print(f"  Imputed {n_missing} missing values in "
                  f"{col} with median={median_val:.2f}")
            profiles_df[col] = profiles_df[col].fillna(median_val)

    print(f"  Computed temporal profiles for {len(profiles_df)} nodes")
    return profiles_df


def _build_combined_features(motes, profiles_df, alpha=SPATIAL_WEIGHT_ALPHA):
    """
    Build the combined spatio-temporal feature matrix.
    Spatial features are weighted by alpha after standardization.
    """
    merged = motes.merge(profiles_df, on="mote_id")
    spatial_cols = ["x", "y"]
    temporal_cols = CLUSTERING_TEMPORAL_FEATURES
    feature_cols = spatial_cols + temporal_cols

    X = merged[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply spatial weight
    n_spatial = len(spatial_cols)
    X_scaled[:, :n_spatial] *= alpha

    return X_scaled, feature_cols, scaler


def _enforce_spatial_coherence(labels, coords, n_clusters=NUM_CLUSTERS):
    """
    Post-processing: fix spatially isolated outlier nodes.

    For each node, check its K nearest spatial neighbors. If ALL K
    neighbors belong to a different cluster (node is spatially isolated),
    reassign it to the majority cluster of its neighbors.

    Only reassigns if it won't shrink any cluster below minimum size.
    """
    labels = labels.copy()
    n = len(labels)
    total_fixed = 0

    for iteration in range(5):
        changed = 0
        for i in range(n):
            # Find K nearest spatial neighbors
            dists = np.sqrt(
                (coords[:, 0] - coords[i, 0]) ** 2 +
                (coords[:, 1] - coords[i, 1]) ** 2
            )
            dists[i] = np.inf
            knn_idx = np.argsort(dists)[:COHERENCE_KNN]
            knn_labels = labels[knn_idx]

            own_cluster = labels[i]
            unique, counts = np.unique(knn_labels, return_counts=True)
            majority_cluster = unique[np.argmax(counts)]
            majority_count = np.max(counts)

            # Reassign only if all/nearly-all neighbors agree
            if (majority_cluster != own_cluster and
                    majority_count >= COHERENCE_MIN_AGREE):
                if np.sum(labels == own_cluster) > COHERENCE_MIN_CLUSTER:
                    labels[i] = majority_cluster
                    changed += 1

        total_fixed += changed
        if changed == 0:
            break

    if total_fixed > 0:
        print(f"  Spatial coherence: fixed {total_fixed} outlier nodes")

    return labels


def spatial_clustering(motes, n_clusters=NUM_CLUSTERS):
    """Baseline: cluster using only (x, y) coordinates."""
    X = motes[["x", "y"]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    labels = km.fit_predict(X_scaled)

    return labels


def spatio_temporal_clustering(X_combined, coords, n_clusters=NUM_CLUSTERS):
    """
    Proposed: two-phase spatio-temporal clustering.
    Phase 1: Weighted K-Means on combined features.
    Phase 2: Spatial coherence enforcement.
    """
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    labels = km.fit_predict(X_combined)

    # Phase 2: enforce spatial coherence
    labels = _enforce_spatial_coherence(labels, coords, n_clusters)

    return labels


def evaluate_in_same_space(X_combined, labels_spatial, labels_st):
    """
    Fair evaluation: compute silhouette and Davies-Bouldin for BOTH
    label sets in the SAME combined feature space.
    """
    metrics = {}
    for name, labels in [("spatial", labels_spatial),
                          ("spatio_temporal", labels_st)]:
        sil = silhouette_score(X_combined, labels)
        db = davies_bouldin_score(X_combined, labels)
        metrics[name] = {"silhouette": sil, "davies_bouldin": db}
    return metrics


def compute_intra_cluster_temp_variance(server_df, motes, labels,
                                         label_col="cluster"):
    """
    Downstream task metric: average intra-cluster temperature variance.
    Lower = nodes in the same cluster behave more similarly.
    """
    motes_copy = motes.copy()
    motes_copy[label_col] = labels

    cluster_variances = []
    for c in sorted(motes_copy[label_col].unique()):
        member_ids = motes_copy.loc[
            motes_copy[label_col] == c, "mote_id"
        ].values
        cluster_data = server_df[server_df["mote_id"].isin(member_ids)]
        if len(cluster_data) > 0:
            cluster_variances.append(cluster_data["temperature"].var())

    return np.mean(cluster_variances)


def compute_cluster_stats(motes, labels, label_name="cluster"):
    """Compute per-cluster size and intra-cluster distance stats."""
    motes_copy = motes.copy()
    motes_copy["cluster"] = labels
    stats = []
    for c in sorted(motes_copy["cluster"].unique()):
        members = motes_copy[motes_copy["cluster"] == c]
        centroid = members[["x", "y"]].mean().values
        dists = np.sqrt(
            ((members[["x", "y"]].values - centroid) ** 2).sum(axis=1)
        )
        stats.append({
            "cluster": c,
            "size": len(members),
            "mean_intra_dist": dists.mean(),
            "max_intra_dist": dists.max(),
            "centroid_x": centroid[0],
            "centroid_y": centroid[1]
        })
    return pd.DataFrame(stats)


def run_clustering(server_df=None):
    """
    Execute clustering comparison:
      1. Pure spatial K-Means (baseline)
      2. Spatio-temporal K-Means + coherence (proposed)

    Both evaluated in the SAME combined feature space for fair comparison.
    Returns motes DataFrame with cluster assignments.
    """
    print("=" * 65)
    print("PHASE 2: SPATIO-TEMPORAL CLUSTERING")
    print("=" * 65 + "\n")

    motes = load_mote_locations()

    if server_df is None:
        server_df = pd.read_csv(SERVER_DATA_PATH)

    # Extract temporal profiles
    profiles_df = extract_temporal_profiles(server_df, motes)

    # Build combined feature matrix
    X_combined, feature_cols, scaler = _build_combined_features(
        motes, profiles_df
    )
    coords = motes[["x", "y"]].values
    print(f"\n  Combined features: {feature_cols}")
    print(f"  Spatial weight alpha: {SPATIAL_WEIGHT_ALPHA}")

    # --- Baseline: spatial only ---
    print("\n[BASELINE] Spatial-only K-Means clustering...")
    spatial_labels = spatial_clustering(motes)
    spatial_stats = compute_cluster_stats(motes, spatial_labels)
    print(f"  Cluster sizes: {spatial_stats['size'].tolist()}")

    # --- Proposed: spatio-temporal + coherence ---
    print("\n[PROPOSED] Spatio-Temporal K-Means + Spatial Coherence...")
    st_labels = spatio_temporal_clustering(X_combined, coords)
    st_stats = compute_cluster_stats(motes, st_labels)
    print(f"  Cluster sizes: {st_stats['size'].tolist()}")

    n_diff = np.sum(st_labels != spatial_labels)
    print(f"  Nodes different from spatial-only: {n_diff}")

    # --- Fair evaluation in same combined space ---
    print("\n[FAIR EVALUATION] Both methods evaluated in combined "
          "feature space")
    fair_metrics = evaluate_in_same_space(
        X_combined, spatial_labels, st_labels
    )

    spatial_metrics = fair_metrics["spatial"]
    st_metrics = fair_metrics["spatio_temporal"]

    print(f"  Spatial-Only    -> Silhouette: "
          f"{spatial_metrics['silhouette']:.4f}  "
          f"DB: {spatial_metrics['davies_bouldin']:.4f}")
    print(f"  Spatio-Temporal -> Silhouette: "
          f"{st_metrics['silhouette']:.4f}  "
          f"DB: {st_metrics['davies_bouldin']:.4f}")

    # --- Downstream metric: intra-cluster temperature variance ---
    print("\n[DOWNSTREAM] Intra-cluster temperature variance "
          "(lower = better)")
    spatial_var = compute_intra_cluster_temp_variance(
        server_df, motes, spatial_labels
    )
    st_var = compute_intra_cluster_temp_variance(
        server_df, motes, st_labels
    )
    print(f"  Spatial-Only:    {spatial_var:.4f}")
    print(f"  Spatio-Temporal: {st_var:.4f}")
    improvement = ((spatial_var - st_var) / spatial_var) * 100
    print(f"  Variance reduction: {improvement:.1f}%")

    # --- Summary ---
    print("\n" + "-" * 50)
    print("COMPARISON: Spatial vs Spatio-Temporal (fair eval)")
    print("-" * 50)
    sil_better = ("PROPOSED" if st_metrics["silhouette"] >
                  spatial_metrics["silhouette"] else "BASELINE")
    db_better = ("PROPOSED" if st_metrics["davies_bouldin"] <
                 spatial_metrics["davies_bouldin"] else "BASELINE")
    var_better = "PROPOSED" if st_var < spatial_var else "BASELINE"
    print(f"  Silhouette (higher=better):    {sil_better} wins")
    print(f"  Davies-Bouldin (lower=better): {db_better} wins")
    print(f"  Intra-cluster variance (lower=better): {var_better} wins")

    # Assign proposed cluster labels to motes
    motes["cluster"] = st_labels
    motes["spatial_cluster"] = spatial_labels

    # Save results
    comparison = pd.DataFrame({
        "method": ["Spatial-Only", "Spatio-Temporal"],
        "silhouette": [spatial_metrics["silhouette"],
                       st_metrics["silhouette"]],
        "davies_bouldin": [spatial_metrics["davies_bouldin"],
                           st_metrics["davies_bouldin"]],
        "intra_cluster_temp_variance": [spatial_var, st_var]
    })
    os.makedirs(METRICS_DIR, exist_ok=True)
    comparison.to_csv(
        os.path.join(METRICS_DIR, "clustering_comparison.csv"), index=False
    )

    print("\n" + "=" * 65)
    print("Clustering complete. Using SPATIO-TEMPORAL clusters "
          "for simulation.")
    print("=" * 65)

    return motes, profiles_df, {
        "spatial": {"labels": spatial_labels, "metrics": spatial_metrics,
                    "stats": spatial_stats},
        "spatio_temporal": {"labels": st_labels, "metrics": st_metrics,
                            "stats": st_stats}
    }


if __name__ == "__main__":
    motes, profiles, results = run_clustering()
    print(motes[["mote_id", "x", "y", "cluster"]].to_string())
