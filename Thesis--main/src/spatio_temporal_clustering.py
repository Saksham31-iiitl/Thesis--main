"""
Step 2: Spatio-Temporal Clustering of WSN Nodes.

Clusters 54 sensor nodes using BOTH:
  - Spatial features: physical (x, y) coordinates
  - Temporal features: per-node behavioral patterns extracted from
    server training data (std_temp, diurnal_amplitude)

This is the KEY NOVELTY vs. purely spatial clustering (LEACH, etc.).
Nodes with similar locations AND similar sensing patterns are grouped
together, enabling better prediction and less intra-cluster variance.

Provides comparison between:
  - Pure spatial K-Means (baseline)
  - Spatio-temporal K-Means (proposed)

Fair evaluation: both methods are evaluated in the SAME combined
feature space so silhouette / Davies-Bouldin scores are comparable.

Feature weighting: spatial features are scaled by SPATIAL_WEIGHT_ALPHA
to balance the influence of spatial vs temporal dimensions.
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
      - std_temp: temperature variability (high CV — good discriminator)
      - mean_humidity: average humidity
      - diurnal_amplitude: difference between daytime and nighttime avg temp
        (highest CV — best discriminator for indoor WSN)
    """
    print("Extracting per-node temporal profiles...")

    # Ensure numeric columns
    for col in ["temperature", "humidity"]:
        if col in server_df.columns:
            server_df[col] = pd.to_numeric(server_df[col], errors="coerce")

    profiles = []
    for mote_id in motes["mote_id"].values:
        node_data = server_df[server_df["mote_id"] == mote_id]

        if len(node_data) < 10:
            # Node has too few readings; mark for median imputation below
            profiles.append({
                "mote_id": mote_id,
                "mean_temp": np.nan,
                "std_temp": np.nan,
                "mean_humidity": np.nan,
                "diurnal_amplitude": np.nan
            })
            continue

        mean_temp = node_data["temperature"].mean()
        std_temp = node_data["temperature"].std()
        mean_humidity = node_data["humidity"].mean()

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
            "diurnal_amplitude": diurnal_amp
        })

    profiles_df = pd.DataFrame(profiles)

    # Impute missing values with median (instead of global mean / 0.0)
    for col in ["mean_temp", "std_temp", "mean_humidity", "diurnal_amplitude"]:
        median_val = profiles_df[col].median()
        n_missing = profiles_df[col].isna().sum()
        if n_missing > 0:
            print(f"  Imputed {n_missing} missing values in {col} with median={median_val:.2f}")
            profiles_df[col] = profiles_df[col].fillna(median_val)

    print(f"  Computed temporal profiles for {len(profiles_df)} nodes")
    return profiles_df


def _build_combined_features(motes, profiles_df, alpha=SPATIAL_WEIGHT_ALPHA):
    """
    Build the combined spatio-temporal feature matrix used for both
    clustering and fair evaluation.

    Returns (X_scaled, feature_cols) where spatial features have been
    weighted by alpha after standardization.
    """
    merged = motes.merge(profiles_df, on="mote_id")
    spatial_cols = ["x", "y"]
    temporal_cols = CLUSTERING_TEMPORAL_FEATURES
    feature_cols = spatial_cols + temporal_cols

    X = merged[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply spatial weight: multiply spatial columns by alpha
    n_spatial = len(spatial_cols)
    X_scaled[:, :n_spatial] *= alpha

    return X_scaled, feature_cols, scaler


def spatial_clustering(motes, n_clusters=NUM_CLUSTERS):
    """Baseline: cluster using only (x, y) coordinates."""
    X = motes[["x", "y"]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    labels = km.fit_predict(X_scaled)

    return labels


def spatio_temporal_clustering(X_combined, n_clusters=NUM_CLUSTERS):
    """
    Proposed: cluster using weighted spatial + temporal features.
    X_combined is the pre-built, weighted, scaled feature matrix.
    """
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    labels = km.fit_predict(X_combined)

    return labels


def evaluate_in_same_space(X_combined, labels_spatial, labels_st):
    """
    Fair evaluation: compute silhouette and Davies-Bouldin for BOTH
    label sets in the SAME combined feature space.
    """
    metrics = {}
    for name, labels in [("spatial", labels_spatial), ("spatio_temporal", labels_st)]:
        sil = silhouette_score(X_combined, labels)
        db = davies_bouldin_score(X_combined, labels)
        metrics[name] = {"silhouette": sil, "davies_bouldin": db}
    return metrics


def compute_intra_cluster_temp_variance(server_df, motes, labels, label_col="cluster"):
    """
    Downstream task metric: compute average intra-cluster temperature
    variance. Lower = nodes in the same cluster behave more similarly
    = better for TinyML prediction.
    """
    motes_copy = motes.copy()
    motes_copy[label_col] = labels

    cluster_variances = []
    for c in sorted(motes_copy[label_col].unique()):
        member_ids = motes_copy.loc[motes_copy[label_col] == c, "mote_id"].values
        cluster_data = server_df[server_df["mote_id"].isin(member_ids)]
        if len(cluster_data) > 0:
            # Variance of temperature readings within this cluster
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
        dists = np.sqrt(((members[["x", "y"]].values - centroid) ** 2).sum(axis=1))
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
      1. Pure spatial (baseline)
      2. Spatio-temporal (proposed)

    Both evaluated in the SAME combined feature space for fair comparison.
    Also computes downstream intra-cluster temperature variance.

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

    # Build combined feature matrix (used for ST clustering AND fair eval)
    X_combined, feature_cols, scaler = _build_combined_features(motes, profiles_df)
    print(f"\n  Combined features: {feature_cols}")
    print(f"  Spatial weight alpha: {SPATIAL_WEIGHT_ALPHA}")

    # --- Baseline: spatial only ---
    print("\n[BASELINE] Spatial-only K-Means clustering...")
    spatial_labels = spatial_clustering(motes)
    spatial_stats = compute_cluster_stats(motes, spatial_labels)
    print(f"  Cluster sizes: {spatial_stats['size'].tolist()}")

    # --- Proposed: spatio-temporal ---
    print("\n[PROPOSED] Spatio-Temporal K-Means clustering...")
    st_labels = spatio_temporal_clustering(X_combined)
    st_stats = compute_cluster_stats(motes, st_labels)
    print(f"  Cluster sizes: {st_stats['size'].tolist()}")

    # --- Fair evaluation in same combined space ---
    print("\n[FAIR EVALUATION] Both methods evaluated in combined feature space")
    fair_metrics = evaluate_in_same_space(X_combined, spatial_labels, st_labels)

    spatial_metrics = fair_metrics["spatial"]
    st_metrics = fair_metrics["spatio_temporal"]

    print(f"  Spatial-Only    -> Silhouette: {spatial_metrics['silhouette']:.4f}  "
          f"DB: {spatial_metrics['davies_bouldin']:.4f}")
    print(f"  Spatio-Temporal -> Silhouette: {st_metrics['silhouette']:.4f}  "
          f"DB: {st_metrics['davies_bouldin']:.4f}")

    # --- Downstream task metric: intra-cluster temperature variance ---
    print("\n[DOWNSTREAM] Intra-cluster temperature variance (lower = better)")
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
    sil_better = "PROPOSED" if st_metrics["silhouette"] > spatial_metrics["silhouette"] else "BASELINE"
    db_better = "PROPOSED" if st_metrics["davies_bouldin"] < spatial_metrics["davies_bouldin"] else "BASELINE"
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
    print("Clustering complete. Using SPATIO-TEMPORAL clusters for simulation.")
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
