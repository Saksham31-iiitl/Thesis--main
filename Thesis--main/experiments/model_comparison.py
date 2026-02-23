"""
Model Comparison for TinyML Selection.

Trains and compares 9 regression models on the Intel Lab dataset.
Evaluates each on:
  - Prediction accuracy: MAE, RMSE, R2
  - TinyML feasibility: model size (KB), inference time (ms)
  - Deployability on microcontroller (MCU-viable flag)

Justifies selection of DecisionTreeRegressor (max_depth=8) as the
optimal model for on-device prediction in constrained WSN nodes.

Output:
  results/metrics/model_comparison.csv
  results/figures/model_comparison_accuracy.png
  results/figures/model_comparison_size_speed.png
  results/figures/model_comparison_tinyml_score.png
"""

import pandas as pd
import numpy as np
import time
import os
import sys
import pickle
import warnings

warnings.filterwarnings("ignore")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.config import (
    SERVER_DATA_PATH, NODE_DATA_PATH,
    FEATURE_COLUMNS, TARGET_VARIABLE,
    METRICS_DIR, FIGURES_DIR
)
from src.tinyml_model import prepare_features

# ── Subsample for speed (training on full 1M+ is slow for heavy models) ──
TRAIN_SAMPLE = 100_000   # samples used to train each model
TEST_SAMPLE  = 50_000    # samples used to evaluate each model
RANDOM_SEED  = 42

# ── MCU constraint: 48 KB RAM for model storage ──────────────────────────
MCU_RAM_LIMIT_KB = 48.0

# ── Candidate models ─────────────────────────────────────────────────────
# Each entry: (name, model_object, mcu_viable, size_estimate_fn)
# mcu_viable: True = can be hand-coded / converted to C on MCU
#             False = runtime/memory too large for bare-metal MCU

def get_models():
    return [
        {
            "name": "Linear Regression",
            "short": "LinReg",
            "model": LinearRegression(),
            "mcu_viable": True,
            "size_note": "~9 floats × 4B = trivial (<1 KB)",
        },
        {
            "name": "Ridge Regression",
            "short": "Ridge",
            "model": Ridge(alpha=1.0),
            "mcu_viable": True,
            "size_note": "~9 floats × 4B = trivial (<1 KB)",
        },
        {
            "name": "Lasso Regression",
            "short": "Lasso",
            "model": Lasso(alpha=0.01),
            "mcu_viable": True,
            "size_note": "Sparse coefficients, <1 KB",
        },
        {
            "name": "Decision Tree (depth=3)",
            "short": "DT-3",
            "model": DecisionTreeRegressor(max_depth=3, random_state=RANDOM_SEED),
            "mcu_viable": True,
            "size_note": "Max 15 nodes × 40B ~ 0.6 KB",
        },
        {
            "name": "Decision Tree (depth=5)",
            "short": "DT-5",
            "model": DecisionTreeRegressor(max_depth=5, random_state=RANDOM_SEED),
            "mcu_viable": True,
            "size_note": "Max 63 nodes × 40B ~ 2.5 KB",
        },
        {
            "name": "Decision Tree (depth=8)",
            "short": "DT-8",
            "model": DecisionTreeRegressor(max_depth=8, random_state=RANDOM_SEED),
            "mcu_viable": True,
            "size_note": "Max 511 nodes × 40B ~ 20 KB → fits 48KB RAM",
        },
        {
            "name": "Decision Tree (depth=10)",
            "short": "DT-10",
            "model": DecisionTreeRegressor(max_depth=10, random_state=RANDOM_SEED),
            "mcu_viable": False,
            "size_note": "Max 2047 nodes × 40B ~ 80 KB → exceeds 48KB RAM",
        },
        {
            "name": "Random Forest (100 trees)",
            "short": "RF-100",
            "model": RandomForestRegressor(
                n_estimators=100, max_depth=8, random_state=RANDOM_SEED, n_jobs=-1
            ),
            "mcu_viable": False,
            "size_note": "100 × 20KB = 2000 KB → far exceeds MCU RAM",
        },
        {
            "name": "k-NN (k=5)",
            "short": "kNN-5",
            "model": KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
            "mcu_viable": False,
            "size_note": "Stores all training points → 100K × 9 × 4B = 3.6 MB",
        },
    ]


def estimate_model_size_kb(entry, trained_model):
    """Estimate model storage size in KB for TinyML deployment."""
    name = entry["short"]
    m = trained_model

    if "LinReg" in name or "Ridge" in name or "Lasso" in name:
        # Just coefficients + intercept
        n_params = len(m.coef_) + 1
        return (n_params * 4) / 1024  # 4 bytes per float32

    elif "DT" in name:
        # Each node in sklearn Decision Tree uses ~40 bytes
        n_nodes = m.tree_.node_count
        return (n_nodes * 40) / 1024

    elif "RF" in name:
        total_nodes = sum(e.tree_.node_count for e in m.estimators_)
        return (total_nodes * 40) / 1024

    elif "kNN" in name:
        # Stores entire training set
        n_train = m._fit_X.shape[0]
        n_features = m._fit_X.shape[1]
        return (n_train * n_features * 4) / 1024

    return 0.0


def measure_inference_time_ms(model, X_sample, n_repeats=200):
    """Measure average inference time per single sample (ms)."""
    X_one = X_sample.iloc[:1]  # single sample
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        model.predict(X_one)
        times.append(time.perf_counter() - t0)
    return np.median(times) * 1000  # convert to ms


def run_model_comparison():
    print("=" * 65)
    print("MODEL COMPARISON FOR TinyML SELECTION")
    print("=" * 65)

    # ── Load data ─────────────────────────────────────────────────
    print(f"\nLoading server training data from {SERVER_DATA_PATH}...")
    server_df = pd.read_csv(SERVER_DATA_PATH)
    print(f"  Full server partition: {len(server_df):,} samples")

    print(f"Loading node test data from {NODE_DATA_PATH}...")
    node_df = pd.read_csv(NODE_DATA_PATH)
    print(f"  Full node partition:   {len(node_df):,} samples")

    # Subsample for tractable comparison
    rng = np.random.default_rng(RANDOM_SEED)
    train_idx = rng.choice(len(server_df), size=min(TRAIN_SAMPLE, len(server_df)),
                            replace=False)
    test_idx  = rng.choice(len(node_df),   size=min(TEST_SAMPLE,  len(node_df)),
                            replace=False)

    train_sub = server_df.iloc[sorted(train_idx)]
    test_sub  = node_df.iloc[sorted(test_idx)]

    print(f"\n  Training subsample: {len(train_sub):,}")
    print(f"  Test subsample:     {len(test_sub):,}")

    X_train, y_train, feat_cols = prepare_features(train_sub)
    X_test,  y_test,  _         = prepare_features(test_sub, feat_cols)

    print(f"  Features ({len(feat_cols)}): {feat_cols}")

    # ── Train & evaluate each model ───────────────────────────────
    models = get_models()
    results = []

    for entry in models:
        name = entry["name"]
        print(f"\n  [{name}]")

        # Train
        t0 = time.perf_counter()
        entry["model"].fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        print(f"    Training time: {train_time:.2f}s")

        # Predict
        y_pred = entry["model"].predict(X_test)

        # Accuracy metrics
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)

        # Size
        size_kb = estimate_model_size_kb(entry, entry["model"])

        # Inference time (single sample)
        infer_ms = measure_inference_time_ms(entry["model"], X_test)

        # MCU viable: must be viable AND fit in 48 KB
        mcu_ok = entry["mcu_viable"] and (size_kb <= MCU_RAM_LIMIT_KB or size_kb < 1)

        print(f"    MAE={mae:.4f}C  RMSE={rmse:.4f}C  R2={r2:.4f}")
        print(f"    Size~{size_kb:.2f} KB  Infer~{infer_ms:.3f}ms  MCU-viable={mcu_ok}")

        results.append({
            "model":        name,
            "short":        entry["short"],
            "mae":          mae,
            "rmse":         rmse,
            "r2":           r2,
            "size_kb":      size_kb,
            "infer_ms":     infer_ms,
            "mcu_viable":   mcu_ok,
            "train_time_s": train_time,
            "size_note":    entry["size_note"],
        })

    df = pd.DataFrame(results)

    # ── Save CSV ──────────────────────────────────────────────────
    os.makedirs(METRICS_DIR, exist_ok=True)
    out_csv = os.path.join(METRICS_DIR, "model_comparison.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n  Saved comparison table to {out_csv}")

    # ── Print summary table ───────────────────────────────────────
    print("\n" + "=" * 65)
    print("COMPARISON SUMMARY")
    print("=" * 65)
    print(f"{'Model':<30} {'MAE':>7} {'RMSE':>7} {'R2':>7} "
          f"{'Size KB':>9} {'Infer ms':>9} {'MCU?':>5}")
    print("-" * 65)
    for _, row in df.iterrows():
        mcu_str = "YES" if row["mcu_viable"] else "NO"
        print(f"{row['model']:<30} {row['mae']:>7.4f} {row['rmse']:>7.4f} "
              f"{row['r2']:>7.4f} {row['size_kb']:>9.2f} "
              f"{row['infer_ms']:>9.3f} {mcu_str:>5}")

    # ── Generate plots ────────────────────────────────────────────
    os.makedirs(FIGURES_DIR, exist_ok=True)
    _plot_accuracy(df)
    _plot_size_speed(df)
    _plot_tinyml_score(df)
    _plot_combined_radar(df)

    print("\n  All plots saved to", FIGURES_DIR)
    print("=" * 65)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    True:  "#2ecc71",   # MCU-viable: green
    False: "#e74c3c",   # Not viable: red
}
HIGHLIGHT = "#3498db"   # Decision Tree depth=8 highlight colour


def _bar_colors(df):
    """Return colour list: blue for DT-8, green for viable, red for not."""
    colors = []
    for _, row in df.iterrows():
        if row["short"] == "DT-8":
            colors.append(HIGHLIGHT)
        elif row["mcu_viable"]:
            colors.append("#27ae60")
        else:
            colors.append("#c0392b")
    return colors


def _plot_accuracy(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Model Accuracy Comparison (Test Set — 50K samples)",
                 fontsize=14, fontweight="bold", y=1.02)

    metrics = [
        ("mae",  "MAE (C)", "lower is better", True),
        ("rmse", "RMSE (C)", "lower is better", True),
        ("r2",   "R2", "higher is better", False),
    ]

    shorts = df["short"].tolist()
    colors = _bar_colors(df)

    for ax, (col, ylabel, note, lower_better) in zip(axes, metrics):
        vals = df[col].values
        bars = ax.bar(shorts, vals, color=colors, edgecolor="white",
                      linewidth=0.8, zorder=3)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{ylabel}\n({note})", fontsize=10)
        ax.set_xticklabels(shorts, rotation=40, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.35, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

        # Value labels on bars
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

        # Highlight DT-8 row
        dt8_idx = df[df["short"] == "DT-8"].index[0]
        ax.axhline(vals[dt8_idx], color=HIGHLIGHT, linestyle="--",
                   linewidth=1.2, alpha=0.6)

    # Legend
    viable_patch = mpatches.Patch(color="#27ae60", label="MCU-viable")
    nope_patch   = mpatches.Patch(color="#c0392b", label="Not MCU-viable")
    dt8_patch    = mpatches.Patch(color=HIGHLIGHT,  label="Selected: DT-8")
    fig.legend(handles=[viable_patch, nope_patch, dt8_patch],
               loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.06), fontsize=10)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "model_comparison_accuracy.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: model_comparison_accuracy.png")


def _plot_size_speed(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("TinyML Feasibility: Model Size & Inference Speed",
                 fontsize=14, fontweight="bold")

    shorts = df["short"].tolist()
    colors = _bar_colors(df)

    # ── Model size ────────────────────────────────────────────────
    sizes = df["size_kb"].values
    bars1 = ax1.bar(shorts, sizes, color=colors, edgecolor="white",
                    linewidth=0.8, zorder=3)
    ax1.axhline(MCU_RAM_LIMIT_KB, color="red", linestyle="--",
                linewidth=1.5, label=f"MCU RAM limit ({MCU_RAM_LIMIT_KB:.0f} KB)")
    ax1.set_ylabel("Estimated Model Size (KB)", fontsize=11)
    ax1.set_title("Model Storage Size\n(must fit ≤ 48 KB MCU RAM)", fontsize=10)
    ax1.set_xticklabels(shorts, rotation=40, ha="right", fontsize=8)
    ax1.grid(axis="y", alpha=0.35, zorder=0)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(fontsize=9)
    # Clip display for very large values so bars are readable
    cap = max(MCU_RAM_LIMIT_KB * 3, sizes[sizes < 1e6].max() * 1.1)
    ax1.set_ylim(0, cap)
    for bar, val in zip(bars1, sizes):
        label = f"{val:.1f}" if val < 1000 else f"{val/1024:.0f} MB"
        y_pos = min(bar.get_height(), cap * 0.95)
        ax1.text(bar.get_x() + bar.get_width() / 2, y_pos + cap * 0.01,
                 label, ha="center", va="bottom", fontsize=7)

    # ── Inference time ────────────────────────────────────────────
    infer = df["infer_ms"].values
    bars2 = ax2.bar(shorts, infer, color=colors, edgecolor="white",
                    linewidth=0.8, zorder=3)
    ax2.axhline(10.0, color="orange", linestyle="--",
                linewidth=1.5, label="10 ms real-time budget")
    ax2.set_ylabel("Inference Time per Sample (ms)", fontsize=11)
    ax2.set_title("Inference Latency\n(must complete before next sensing cycle)",
                  fontsize=10)
    ax2.set_xticklabels(shorts, rotation=40, ha="right", fontsize=8)
    ax2.grid(axis="y", alpha=0.35, zorder=0)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.legend(fontsize=9)
    for bar, val in zip(bars2, infer):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + infer.max() * 0.01,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    viable_patch = mpatches.Patch(color="#27ae60", label="MCU-viable")
    nope_patch   = mpatches.Patch(color="#c0392b", label="Not MCU-viable")
    dt8_patch    = mpatches.Patch(color=HIGHLIGHT,  label="Selected: DT-8")
    fig.legend(handles=[viable_patch, nope_patch, dt8_patch],
               loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.06), fontsize=10)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "model_comparison_size_speed.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: model_comparison_size_speed.png")


def _plot_tinyml_score(df):
    """
    Composite TinyML score combining accuracy + feasibility.
    Score = 0.4*(1-norm_MAE) + 0.3*(1-norm_size) + 0.3*(1-norm_infer)
    MCU-non-viable models are greyed out.
    """
    d = df.copy()

    # Normalise each metric 0-1 (0=worst, 1=best)
    d["n_mae"]   = 1 - (d["mae"]  - d["mae"].min())  / (d["mae"].max()  - d["mae"].min()  + 1e-9)
    d["n_size"]  = 1 - (d["size_kb"]  - d["size_kb"].min())  / (d["size_kb"].max()  - d["size_kb"].min()  + 1e-9)
    d["n_infer"] = 1 - (d["infer_ms"] - d["infer_ms"].min()) / (d["infer_ms"].max() - d["infer_ms"].min() + 1e-9)

    d["tinyml_score"] = 0.4 * d["n_mae"] + 0.3 * d["n_size"] + 0.3 * d["n_infer"]

    # Zero out non-viable models
    d.loc[~d["mcu_viable"], "tinyml_score"] = 0.0

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = _bar_colors(d)

    bars = ax.bar(d["short"], d["tinyml_score"], color=colors,
                  edgecolor="white", linewidth=0.8, zorder=3)

    ax.set_ylabel("Composite TinyML Score\n(Accuracy 40% + Size 30% + Speed 30%)",
                  fontsize=11)
    ax.set_title(
        "TinyML Deployment Score per Model\n"
        "Non-MCU-viable models scored 0 (cannot be deployed on sensor node)",
        fontsize=12, fontweight="bold"
    )
    ax.set_xticklabels(d["short"], rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.35, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    for bar, val in zip(bars, d["tinyml_score"]):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.015,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8,
                    fontweight="bold" if val == d["tinyml_score"].max() else "normal")

    # Annotate winner
    best_idx = d["tinyml_score"].idxmax()
    best_bar = bars[best_idx]
    ax.annotate("SELECTED\n(Best TinyML score)",
                xy=(best_bar.get_x() + best_bar.get_width() / 2,
                    d["tinyml_score"].iloc[best_idx]),
                xytext=(best_bar.get_x() + best_bar.get_width() / 2 + 0.5,
                        d["tinyml_score"].iloc[best_idx] + 0.12),
                arrowprops=dict(arrowstyle="->", color="navy"),
                fontsize=9, color="navy", fontweight="bold",
                ha="center")

    viable_patch = mpatches.Patch(color="#27ae60", label="MCU-viable")
    nope_patch   = mpatches.Patch(color="#c0392b", label="Not MCU-viable (score=0)")
    dt8_patch    = mpatches.Patch(color=HIGHLIGHT,  label="Selected: DT-8")
    ax.legend(handles=[viable_patch, nope_patch, dt8_patch],
              frameon=False, fontsize=10, loc="upper left")

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "model_comparison_tinyml_score.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: model_comparison_tinyml_score.png")


def _plot_combined_radar(df):
    """Radar/spider chart for MCU-viable models only."""
    viable = df[df["mcu_viable"]].copy().reset_index(drop=True)

    if len(viable) < 2:
        return

    # Normalise (higher = better for all axes)
    def norm(series, lower_better=True):
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series([0.5] * len(series))
        n = (series - mn) / (mx - mn)
        return 1 - n if lower_better else n

    viable["n_mae"]   = norm(viable["mae"],      lower_better=True)
    viable["n_rmse"]  = norm(viable["rmse"],     lower_better=True)
    viable["n_r2"]    = norm(viable["r2"],        lower_better=False)
    viable["n_size"]  = norm(viable["size_kb"],  lower_better=True)
    viable["n_infer"] = norm(viable["infer_ms"], lower_better=True)

    categories = ["MAE", "RMSE", "R2", "Compactness", "Speed"]
    score_cols  = ["n_mae", "n_rmse", "n_r2", "n_size", "n_infer"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"polar": True})
    cmap = plt.cm.get_cmap("tab10", len(viable))

    for i, row in viable.iterrows():
        vals = [row[c] for c in score_cols]
        vals += vals[:1]
        color = HIGHLIGHT if row["short"] == "DT-8" else cmap(i)
        lw    = 2.5 if row["short"] == "DT-8" else 1.2
        ax.plot(angles, vals, color=color, linewidth=lw, label=row["short"])
        ax.fill(angles, vals, color=color, alpha=0.1 if row["short"] == "DT-8" else 0.04)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title("MCU-Viable Models: Multi-criteria Radar\n"
                 "(All axes: higher = better)",
                 fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "model_comparison_radar.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: model_comparison_radar.png")


if __name__ == "__main__":
    df_results = run_model_comparison()
