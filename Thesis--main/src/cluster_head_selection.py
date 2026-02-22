"""
Step 3: Cluster Head Selection and Rotation.

Selects one Cluster Head (CH) per cluster using a multi-criteria
weighted score:

    Score(i) = alpha * E_norm(i) + beta * Deg_norm(i) + gamma * Prox_norm(i)

Where:
    E_norm    = residual energy (normalized)
    Deg_norm  = connectivity degree (normalized)
    Prox_norm = proximity to cluster centroid (inverted distance, normalized)

CH rotation: at each round, the node with the highest score in each
cluster becomes CH. As nodes expend energy, scores change and CHs rotate.
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    CONNECTIVITY_PATH, E_INIT, NUM_CLUSTERS, BS_LOCATION,
    CH_WEIGHT_ENERGY, CH_WEIGHT_DEGREE, CH_WEIGHT_DIST, RANDOM_SEED
)


def load_connectivity(path=CONNECTIVITY_PATH):
    """
    Load the connectivity matrix.
    Format: src_id dst_id probability
    Convert to a degree count per node (number of neighbors with prob > 0.5).
    """
    rows = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                rows.append({
                    "src": int(parts[0]),
                    "dst": int(parts[1]),
                    "prob": float(parts[2])
                })
    conn_df = pd.DataFrame(rows)

    # Count neighbors with connectivity probability > 0.5
    strong = conn_df[conn_df["prob"] > 0.5]
    degrees = strong.groupby("src").size().reset_index(name="degree")
    degrees.rename(columns={"src": "mote_id"}, inplace=True)
    return degrees


def initialize_nodes(motes):
    """
    Prepare the node state DataFrame:
      - Assign initial energy to all nodes
      - Compute connectivity degree
      - Compute distance to cluster centroid and to BS
    """
    nodes = motes.copy()
    nodes["residual_energy"] = E_INIT

    # Load connectivity degrees
    degrees = load_connectivity()
    nodes = nodes.merge(degrees, on="mote_id", how="left")
    nodes["degree"] = nodes["degree"].fillna(0).astype(int)

    # Distance to base station
    nodes["dist_to_bs"] = np.sqrt(
        (nodes["x"] - BS_LOCATION[0]) ** 2 +
        (nodes["y"] - BS_LOCATION[1]) ** 2
    )

    return nodes


def compute_ch_scores(nodes):
    """
    Compute multi-criteria CH score for each node.
    Higher score = better CH candidate.
    """
    alpha = CH_WEIGHT_ENERGY
    beta = CH_WEIGHT_DEGREE
    gamma = CH_WEIGHT_DIST

    scores_list = []
    for c in sorted(nodes["cluster"].unique()):
        cluster_nodes = nodes[nodes["cluster"] == c].copy()

        # Centroid of cluster
        centroid = cluster_nodes[["x", "y"]].mean().values

        # Distance to centroid (lower is better -> invert)
        dist_to_centroid = np.sqrt(
            (cluster_nodes["x"] - centroid[0]) ** 2 +
            (cluster_nodes["y"] - centroid[1]) ** 2
        )
        # Avoid division by zero
        max_dist = dist_to_centroid.max()
        if max_dist == 0:
            prox_norm = pd.Series(1.0, index=cluster_nodes.index)
        else:
            prox_norm = 1 - (dist_to_centroid / max_dist)

        # Normalize energy and degree within cluster
        e_max = cluster_nodes["residual_energy"].max()
        e_norm = cluster_nodes["residual_energy"] / e_max if e_max > 0 else 0

        d_max = cluster_nodes["degree"].max()
        deg_norm = cluster_nodes["degree"] / d_max if d_max > 0 else 0

        score = alpha * e_norm + beta * deg_norm + gamma * prox_norm
        cluster_nodes["ch_score"] = score.values
        cluster_nodes["dist_to_centroid"] = dist_to_centroid.values
        scores_list.append(cluster_nodes)

    return pd.concat(scores_list, ignore_index=False)


def select_cluster_heads(nodes):
    """
    Select one CH per cluster: the node with the highest score.
    Returns updated nodes DataFrame with 'is_ch' column.
    """
    nodes = compute_ch_scores(nodes)
    nodes["is_ch"] = False

    ch_ids = []
    for c in sorted(nodes["cluster"].unique()):
        cluster_mask = nodes["cluster"] == c
        best_idx = nodes.loc[cluster_mask, "ch_score"].idxmax()
        nodes.loc[best_idx, "is_ch"] = True
        ch_ids.append(nodes.loc[best_idx, "mote_id"])

    return nodes, ch_ids


def get_ch_for_node(nodes, mote_id):
    """Get the cluster head mote_id for a given node."""
    cluster = nodes.loc[nodes["mote_id"] == mote_id, "cluster"].values[0]
    ch_row = nodes[(nodes["cluster"] == cluster) & (nodes["is_ch"])]
    return ch_row["mote_id"].values[0]


def get_node_to_ch_distance(nodes, mote_id):
    """Get Euclidean distance from a member node to its CH."""
    node = nodes[nodes["mote_id"] == mote_id].iloc[0]
    cluster = node["cluster"]
    ch = nodes[(nodes["cluster"] == cluster) & (nodes["is_ch"])].iloc[0]
    return np.sqrt((node["x"] - ch["x"]) ** 2 + (node["y"] - ch["y"]) ** 2)


def get_ch_to_bs_distance(nodes, cluster_id):
    """Get Euclidean distance from a CH to the base station."""
    ch = nodes[(nodes["cluster"] == cluster_id) & (nodes["is_ch"])].iloc[0]
    return np.sqrt(
        (ch["x"] - BS_LOCATION[0]) ** 2 +
        (ch["y"] - BS_LOCATION[1]) ** 2
    )


def run_ch_selection(motes):
    """Execute CH selection pipeline."""
    print("=" * 65)
    print("PHASE 2b: CLUSTER HEAD SELECTION")
    print("=" * 65 + "\n")

    nodes = initialize_nodes(motes)
    nodes, ch_ids = select_cluster_heads(nodes)

    print("Cluster Head assignments:")
    for c in sorted(nodes["cluster"].unique()):
        ch = nodes[(nodes["cluster"] == c) & (nodes["is_ch"])].iloc[0]
        members = nodes[nodes["cluster"] == c]
        print(f"  Cluster {c}: CH = Mote {int(ch['mote_id'])} "
              f"(score={ch['ch_score']:.3f}, "
              f"energy={ch['residual_energy']:.3f}J, "
              f"members={len(members)})")

    print(f"\nTotal nodes: {len(nodes)}, Clusters: {nodes['cluster'].nunique()}")
    return nodes


if __name__ == "__main__":
    from spatio_temporal_clustering import run_clustering
    motes, _, _ = run_clustering()
    nodes = run_ch_selection(motes)
