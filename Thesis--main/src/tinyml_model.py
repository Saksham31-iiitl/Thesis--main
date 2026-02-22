"""
Step 4: TinyML Model Training and Prediction.

Trains a DecisionTreeRegressor (TinyML-friendly) on the 60% server
training data to predict temperature.

The model is:
  - Trained ONCE on the full server partition (initial model)
  - Can be retrained incrementally when buffer reaches B_min

Features used: mote_id, humidity, light, voltage, hour, minute,
               day_of_week, hour_sin, hour_cos
Target: temperature
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import sys
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    SERVER_DATA_PATH, NODE_DATA_PATH, FEATURE_COLUMNS,
    TARGET_VARIABLE, MODEL_MAX_DEPTH, MODEL_RANDOM_STATE,
    METRICS_DIR
)


def prepare_features(df, feature_cols=FEATURE_COLUMNS):
    """Extract feature matrix X and target y from dataframe."""
    available_cols = [c for c in feature_cols if c in df.columns]
    X = df[available_cols].copy()

    # Ensure all numeric
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df[TARGET_VARIABLE].astype(float)

    return X, y, available_cols


def train_initial_model(server_df=None, max_depth=MODEL_MAX_DEPTH):
    """
    Train the initial TinyML model on the full server (60%) dataset.
    Returns: trained model, feature columns used, training metrics.
    """
    print("=" * 65)
    print("PHASE 3: TinyML MODEL TRAINING")
    print("=" * 65 + "\n")

    if server_df is None:
        server_df = pd.read_csv(SERVER_DATA_PATH)

    X_train, y_train, feat_cols = prepare_features(server_df)

    print(f"Training DecisionTreeRegressor (max_depth={max_depth})...")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Features ({len(feat_cols)}): {feat_cols}")

    model = DecisionTreeRegressor(
        max_depth=max_depth,
        random_state=MODEL_RANDOM_STATE
    )
    model.fit(X_train, y_train)

    # Training set metrics
    y_pred_train = model.predict(X_train)
    train_metrics = {
        "mse": mean_squared_error(y_train, y_pred_train),
        "mae": mean_absolute_error(y_train, y_pred_train),
        "r2": r2_score(y_train, y_pred_train),
        "n_nodes": model.tree_.node_count,
        "max_depth_actual": model.tree_.max_depth,
        "model_size_kb": (model.tree_.node_count * 40) / 1024
    }

    print(f"\nTraining set performance:")
    print(f"  MAE:  {train_metrics['mae']:.4f} C")
    print(f"  MSE:  {train_metrics['mse']:.4f}")
    print(f"  R2:   {train_metrics['r2']:.4f}")
    print(f"\nModel characteristics:")
    print(f"  Tree nodes:    {train_metrics['n_nodes']}")
    print(f"  Actual depth:  {train_metrics['max_depth_actual']}")
    print(f"  Est. size:     {train_metrics['model_size_kb']:.2f} KB")

    return model, feat_cols, train_metrics


def evaluate_on_node_data(model, feat_cols, node_df=None, threshold=1.0):
    """
    Evaluate the trained model on the node (40%) partition.
    Computes: MAE, RMSE, R2, mismatch rate at given threshold.
    """
    if node_df is None:
        node_df = pd.read_csv(NODE_DATA_PATH)

    X_test, y_test, _ = prepare_features(node_df, feat_cols)
    predictions = model.predict(X_test)

    errors = np.abs(predictions - y_test.values)
    mismatch_mask = errors > threshold

    metrics = {
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
        "r2": r2_score(y_test, predictions),
        "mismatch_rate": mismatch_mask.mean(),
        "communication_reduction": 1 - mismatch_mask.mean(),
        "threshold": threshold,
        "n_samples": len(y_test)
    }

    print(f"\nNode data evaluation (threshold={threshold} C):")
    print(f"  MAE:    {metrics['mae']:.4f} C")
    print(f"  RMSE:   {metrics['rmse']:.4f} C")
    print(f"  R2:     {metrics['r2']:.4f}")
    print(f"  Mismatch rate:   {metrics['mismatch_rate']*100:.2f}%")
    print(f"  Comm. reduction: {metrics['communication_reduction']*100:.2f}%")

    return predictions, metrics


def retrain_model(model, buffer_X, buffer_y, feat_cols):
    """
    Retrain the model using buffered mismatch samples.
    Creates a new DecisionTree trained on the combined
    original + buffer data approach via warm-start concept:
    we retrain on the buffer samples which represent the
    distribution shift the model is failing on.
    """
    new_model = DecisionTreeRegressor(
        max_depth=MODEL_MAX_DEPTH,
        random_state=MODEL_RANDOM_STATE
    )
    new_model.fit(buffer_X, buffer_y)
    return new_model


def run_model_training(server_df=None, node_df=None):
    """Execute full model training and evaluation pipeline."""
    model, feat_cols, train_metrics = train_initial_model(server_df)

    # Quick evaluation on node data to see baseline mismatch
    predictions, eval_metrics = evaluate_on_node_data(model, feat_cols, node_df)

    # Save metrics
    os.makedirs(METRICS_DIR, exist_ok=True)
    metrics_combined = {**train_metrics, **eval_metrics}
    pd.DataFrame([metrics_combined]).to_csv(
        os.path.join(METRICS_DIR, "model_metrics.csv"), index=False
    )

    print("\n" + "=" * 65)
    print("Model training complete.")
    print("=" * 65)

    return model, feat_cols, train_metrics, eval_metrics


if __name__ == "__main__":
    model, feat_cols, train_m, eval_m = run_model_training()
