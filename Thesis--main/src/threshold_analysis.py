"""
Step 9: Threshold Analysis.

Sweeps static threshold values from 0.25 to 3.0 C and evaluates:
  - Communication reduction (CR)
  - Energy savings
  - Prediction quality (MAE)
  - Mismatch rate
  - Retraining frequency

Also implements an ADAPTIVE threshold that adjusts based on
recent prediction error variance.
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    THRESHOLD_SWEEP, ADAPTIVE_WINDOW, ADAPTIVE_ALPHA, ADAPTIVE_BETA,
    NODE_DATA_PATH, FEATURE_COLUMNS, TARGET_VARIABLE, METRICS_DIR,
    BUFFER_MIN, SIM_SAMPLE_LIMIT
)
from tinyml_model import prepare_features, retrain_model


def threshold_sweep_fast(model, feat_cols, node_df=None,
                         thresholds=THRESHOLD_SWEEP):
    """
    Fast threshold sweep: predict ONCE, then evaluate at each threshold.
    No retraining — this shows the static model performance across thresholds.
    """
    if node_df is None:
        node_df = pd.read_csv(NODE_DATA_PATH)
    if SIM_SAMPLE_LIMIT and len(node_df) > SIM_SAMPLE_LIMIT:
        node_df = node_df.iloc[:SIM_SAMPLE_LIMIT]

    X_node, y_node, _ = prepare_features(node_df, feat_cols)
    predictions = model.predict(X_node)
    errors = np.abs(predictions - y_node.values)

    results = []
    for tau in thresholds:
        mismatch_mask = errors > tau
        mismatch_rate = mismatch_mask.mean()
        cr = 1 - mismatch_rate

        results.append({
            "threshold": tau,
            "mismatch_rate": mismatch_rate,
            "communication_reduction": cr,
            "mae": errors.mean(),
            "rmse": np.sqrt((errors ** 2).mean()),
            "n_transmissions": mismatch_mask.sum(),
            "n_suppressed": (~mismatch_mask).sum()
        })

    return pd.DataFrame(results)


def adaptive_threshold_simulation(model, feat_cols, node_df=None,
                                  window=ADAPTIVE_WINDOW,
                                  alpha=ADAPTIVE_ALPHA,
                                  beta=ADAPTIVE_BETA,
                                  buffer_min=BUFFER_MIN):
    """
    Simulate with adaptive threshold:
      tau(t) = alpha * std(recent_errors) + beta

    The threshold adapts to the recent prediction error distribution.
    When errors are low, threshold tightens (more aggressive suppression).
    When errors are high, threshold loosens (allows more communication
    until model catches up via retraining).
    """
    if node_df is None:
        node_df = pd.read_csv(NODE_DATA_PATH)
    if SIM_SAMPLE_LIMIT and len(node_df) > SIM_SAMPLE_LIMIT:
        node_df = node_df.iloc[:SIM_SAMPLE_LIMIT]

    X_node, y_node, _ = prepare_features(node_df, feat_cols)
    n = len(X_node)

    current_model = model
    recent_errors = []
    buffer_X = []
    buffer_y = []

    log_threshold = np.zeros(n)
    log_transmitted = np.zeros(n, dtype=bool)
    log_error = np.zeros(n)
    retrain_count = 0

    for i in range(n):
        x_i = X_node.iloc[i:i+1]
        y_true = y_node.iloc[i]

        y_pred = current_model.predict(x_i)[0]
        error = abs(y_pred - y_true)
        log_error[i] = error

        # Compute adaptive threshold
        recent_errors.append(error)
        if len(recent_errors) > window:
            recent_errors.pop(0)

        if len(recent_errors) >= 5:
            tau = alpha * np.std(recent_errors) + beta
        else:
            tau = beta + 1.0  # conservative start

        log_threshold[i] = tau

        transmit = error > tau
        log_transmitted[i] = transmit

        if transmit:
            buffer_X.append(x_i.values[0])
            buffer_y.append(y_true)

        # Retrain if buffer full
        if len(buffer_X) >= buffer_min:
            buf_X_df = pd.DataFrame(
                np.array(buffer_X), columns=feat_cols
            )
            current_model = retrain_model(
                current_model, buf_X_df, np.array(buffer_y), feat_cols
            )
            buffer_X = []
            buffer_y = []
            retrain_count += 1

    mismatch_rate = log_transmitted.mean()
    cr = 1 - mismatch_rate

    return {
        "mismatch_rate": mismatch_rate,
        "communication_reduction": cr,
        "mae": log_error.mean(),
        "rmse": np.sqrt((log_error ** 2).mean()),
        "n_retrains": retrain_count,
        "retrain_frequency": retrain_count / n,
        "log_threshold": log_threshold,
        "log_transmitted": log_transmitted,
        "log_error": log_error,
        "avg_threshold": log_threshold.mean()
    }


def run_threshold_analysis(model, feat_cols, node_df=None):
    """Run full threshold analysis: static sweep + adaptive."""
    print("=" * 65)
    print("PHASE 6: THRESHOLD ANALYSIS")
    print("=" * 65 + "\n")

    # Static sweep
    print("Running static threshold sweep...")
    sweep_df = threshold_sweep_fast(model, feat_cols, node_df)

    print("\nStatic Threshold Sweep Results:")
    print(sweep_df[["threshold", "communication_reduction",
                     "mismatch_rate", "mae"]].to_string(index=False))

    # Adaptive threshold
    print("\nRunning adaptive threshold simulation...")
    adaptive_results = adaptive_threshold_simulation(
        model, feat_cols, node_df
    )
    print(f"\nAdaptive Threshold Results:")
    print(f"  Avg threshold:        {adaptive_results['avg_threshold']:.3f} C")
    print(f"  Communication Red.:   "
          f"{adaptive_results['communication_reduction']*100:.2f}%")
    print(f"  Mismatch rate:        "
          f"{adaptive_results['mismatch_rate']*100:.2f}%")
    print(f"  MAE:                  {adaptive_results['mae']:.4f} C")
    print(f"  Retrains:             {adaptive_results['n_retrains']}")

    # Save
    os.makedirs(METRICS_DIR, exist_ok=True)
    sweep_df.to_csv(
        os.path.join(METRICS_DIR, "threshold_sweep.csv"), index=False
    )

    return sweep_df, adaptive_results
