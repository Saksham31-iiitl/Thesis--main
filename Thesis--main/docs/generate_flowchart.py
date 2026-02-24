"""
Generate a professional system architecture flow diagram for the
WSN-IoT TinyML thesis IEEE paper.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(14, 20))
ax.set_xlim(0, 14)
ax.set_ylim(0, 22)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── Color palette ──
C_PHASE    = '#1B4F72'   # Dark blue for phase headers
C_SERVER   = '#2980B9'   # Blue for server-side boxes
C_NODE     = '#27AE60'   # Green for node-side boxes
C_DECISION = '#E67E22'   # Orange for decision diamonds
C_DATA     = '#8E44AD'   # Purple for data/storage
C_RESULT   = '#C0392B'   # Red for results/outcomes
C_ARROW    = '#2C3E50'   # Dark grey arrows
C_LIGHT_BG = '#EBF5FB'   # Light blue background
C_LIGHT_GR = '#EAFAF1'   # Light green background
C_LIGHT_OR = '#FDF2E9'   # Light orange background

def draw_box(ax, x, y, w, h, text, color, text_color='white', fontsize=9,
             style='round', alpha=1.0, bold=False, linewidth=1.5):
    """Draw a rounded rectangle with centered text."""
    if style == 'round':
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="round,pad=0.15", linewidth=linewidth,
                             edgecolor=color, facecolor=color, alpha=alpha)
    elif style == 'square':
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="square,pad=0.08", linewidth=linewidth,
                             edgecolor=color, facecolor=color, alpha=alpha)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=text_color, fontweight=weight, wrap=True,
            linespacing=1.4)

def draw_diamond(ax, cx, cy, w, h, text, color, text_color='white', fontsize=8):
    """Draw a diamond (decision) shape."""
    diamond = plt.Polygon([
        (cx, cy + h/2),
        (cx + w/2, cy),
        (cx, cy - h/2),
        (cx - w/2, cy)
    ], closed=True, facecolor=color, edgecolor=color, linewidth=1.5)
    ax.add_patch(diamond)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize,
            color=text_color, fontweight='bold', linespacing=1.3)

def draw_arrow(ax, x1, y1, x2, y2, color=C_ARROW, style='->', linewidth=1.8, label='', label_side='right'):
    """Draw an arrow between two points."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=linewidth, connectionstyle='arc3,rad=0'))
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        offset = 0.3 if label_side == 'right' else -0.3
        ax.text(mx + offset, my, label, fontsize=7.5, color=color,
                fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor=color, alpha=0.9))

def draw_curved_arrow(ax, x1, y1, x2, y2, color=C_ARROW, rad=0.3, label='', label_pos=(0,0)):
    """Draw a curved arrow."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=1.8, connectionstyle=f'arc3,rad={rad}'))
    if label:
        ax.text(label_pos[0], label_pos[1], label, fontsize=7.5, color=color,
                fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor=color, alpha=0.9))


# ═══════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════
ax.text(7, 21.5, 'System Architecture: Energy-Efficient WSN-IoT Framework',
        ha='center', va='center', fontsize=14, fontweight='bold',
        color=C_PHASE, style='normal')
ax.text(7, 21.1, 'Spatio-Temporal Clustering + TinyML + Continuous Learning',
        ha='center', va='center', fontsize=10, color='#555555')

# ═══════════════════════════════════════════════════════════════
# PHASE 1: DATA COLLECTION & PREPROCESSING (Top)
# ═══════════════════════════════════════════════════════════════
# Phase header
draw_box(ax, 7, 20.4, 13, 0.45, 'PHASE 1: Data Collection & Preprocessing',
         C_PHASE, fontsize=10, bold=True, style='square')

# Sensor network
draw_box(ax, 3.5, 19.5, 4.5, 0.7,
         '54 Mica2Dot Sensor Nodes\n(Temp, Humidity, Light, Voltage)',
         C_NODE, fontsize=8.5)

# Raw data
draw_box(ax, 10.5, 19.5, 4.5, 0.7,
         'Intel Berkeley Dataset\n2.3M readings, 37 days',
         C_DATA, fontsize=8.5)

draw_arrow(ax, 5.75, 19.5, 8.25, 19.5, label='raw data')

# Cleaning + Feature Engineering
draw_box(ax, 7, 18.5, 5.5, 0.75,
         'Data Cleaning & Feature Engineering\n'
         'Valid: 1.06M records (45.8%) │ Cyclical encoding: sin/cos(2πh/24)',
         '#1A5276', fontsize=8, text_color='white')

draw_arrow(ax, 7, 19.1, 7, 18.9)

# Split
draw_box(ax, 3.5, 17.5, 4.8, 0.7,
         'Server Training Data (60%)\n424,752 samples',
         C_SERVER, fontsize=8.5)

draw_box(ax, 10.5, 17.5, 4.8, 0.7,
         'Node Simulation Data (40%)\n636,015 samples',
         C_NODE, fontsize=8.5)

draw_arrow(ax, 5.5, 18.1, 3.5, 17.9, label='chronological\nsplit', label_side='left')
draw_arrow(ax, 8.5, 18.1, 10.5, 17.9)


# ═══════════════════════════════════════════════════════════════
# PHASE 2: SPATIO-TEMPORAL CLUSTERING (Left branch)
# ═══════════════════════════════════════════════════════════════
draw_box(ax, 3.5, 16.55, 6, 0.45, 'PHASE 2: Spatio-Temporal Clustering',
         C_PHASE, fontsize=10, bold=True, style='square')

draw_box(ax, 3.5, 15.7, 5.5, 0.85,
         'Feature Vector per Node\n'
         'F = [x, y, σ_temp, A_diurnal, L_mean]\n'
         'CV-selected: light(44.6%), diurnal(35.4%)',
         C_SERVER, fontsize=8)

draw_arrow(ax, 3.5, 17.1, 3.5, 16.15)

# K-Means
draw_box(ax, 3.5, 14.6, 5.5, 0.75,
         'K-Means Clustering (K=4)\n'
         'StandardScaler │ Spatial weight α=1.5',
         C_SERVER, fontsize=8.5)

draw_arrow(ax, 3.5, 15.25, 3.5, 15.0)

# CH Selection
draw_box(ax, 3.5, 13.55, 5.5, 0.8,
         'Cluster Head Selection\n'
         'Score = 0.4·E + 0.3·Deg + 0.3·Prox\n'
         'Per-cluster best-scoring node → CH',
         C_SERVER, fontsize=8)

draw_arrow(ax, 3.5, 14.2, 3.5, 13.98)

# Clustering result box
draw_box(ax, 3.5, 12.5, 5.2, 0.7,
         'Result: Silhouette 0.385 (+2.9%)\n'
         'Davies-Bouldin 0.841 │ 4 clusters',
         C_RESULT, fontsize=8)

draw_arrow(ax, 3.5, 13.12, 3.5, 12.88)


# ═══════════════════════════════════════════════════════════════
# PHASE 3: TINYML MODEL TRAINING (Right branch)
# ═══════════════════════════════════════════════════════════════
draw_box(ax, 10.5, 16.55, 6, 0.45, 'PHASE 3: TinyML Model Training',
         C_PHASE, fontsize=10, bold=True, style='square')

draw_box(ax, 10.5, 15.7, 5.5, 0.85,
         'Model Comparison (9 models)\n'
         'DT, RF, k-NN, Linear, Ridge,\n'
         'Lasso, SGD, Gradient Boosting',
         C_NODE, fontsize=8)

draw_arrow(ax, 10.5, 17.1, 10.5, 16.15)

# Selected model
draw_box(ax, 10.5, 14.6, 5.5, 0.75,
         'Selected: Decision Tree (depth=8)\n'
         'MAE: 2.50°C │ Size: 19.88 KB │ Infer: 0.50 ms',
         '#1E8449', fontsize=8, bold=False)

draw_arrow(ax, 10.5, 15.25, 10.5, 15.0)

# Deploy
draw_box(ax, 10.5, 13.55, 5.5, 0.75,
         'Deploy to Sensor Nodes\n'
         'Fits 48KB MCU RAM │ <1ms real-time',
         C_NODE, fontsize=8.5)

draw_arrow(ax, 10.5, 14.2, 10.5, 13.95)

# Model result
draw_box(ax, 10.5, 12.5, 5.2, 0.7,
         'MCU-viable: ✓ (ARM Cortex-M0+/M4)\n'
         '509 tree nodes │ 9 input features',
         C_RESULT, fontsize=8)

draw_arrow(ax, 10.5, 13.15, 10.5, 12.88)


# ═══════════════════════════════════════════════════════════════
# PHASE 4: RUNTIME OPERATION (Center, bottom half)
# ═══════════════════════════════════════════════════════════════
draw_box(ax, 7, 11.45, 13, 0.45, 'PHASE 4: Runtime Mismatch-Driven Transmission',
         C_PHASE, fontsize=10, bold=True, style='square')

# Sense
draw_box(ax, 7, 10.55, 4.5, 0.65,
         'Sensor Node: Read Actual Value y(t)\n'
         'E_sense = 16.5 μJ │ E_proc = 142.5 μJ',
         C_NODE, fontsize=8)

draw_arrow(ax, 5, 12.1, 5, 11.7)
draw_arrow(ax, 9, 12.1, 9, 11.7)

# Predict
draw_box(ax, 7, 9.55, 4.5, 0.65,
         'TinyML Predict: ŷ(t) = DT(features)\n'
         'E_infer = 57 μJ │ t_infer < 1 ms',
         '#1E8449', fontsize=8)

draw_arrow(ax, 7, 10.2, 7, 9.9)

# Decision diamond
draw_diamond(ax, 7, 8.35, 4.5, 1.1,
             '|y(t) − ŷ(t)| > τ ?\n(τ = 1.0°C default)',
             C_DECISION, fontsize=8.5)

draw_arrow(ax, 7, 9.2, 7, 8.93)

# YES branch (left) - Transmit
draw_box(ax, 2.5, 7.0, 4.0, 0.85,
         'TRANSMIT to CH → Server\n'
         'E_Tx ≈ 1 mJ per packet\n'
         '(1600 bits, Heinzelman model)',
         C_RESULT, fontsize=8)

draw_arrow(ax, 4.75, 8.35, 3.7, 7.45, label='YES', label_side='left')

# Buffer mismatch
draw_box(ax, 2.5, 5.85, 4.0, 0.7,
         'Buffer Mismatch Sample\n'
         'Add (x, y_actual) to retrain buffer',
         C_DATA, fontsize=8)

draw_arrow(ax, 2.5, 6.55, 2.5, 6.23)

# NO branch (right) - Suppress
draw_box(ax, 11.5, 7.0, 4.0, 0.85,
         'SUPPRESS Transmission\n'
         'Server uses predicted ŷ(t)\n'
         'Save ~1 mJ (radio TX skipped)',
         '#27AE60', fontsize=8)

draw_arrow(ax, 9.25, 8.35, 10.3, 7.45, label='NO', label_side='right')

# Energy saved result
draw_box(ax, 11.5, 5.85, 4.0, 0.7,
         '72.25% Transmissions Suppressed\n'
         '19.73% Total Energy Saved',
         '#1E8449', fontsize=8, bold=True)

draw_arrow(ax, 11.5, 6.55, 11.5, 6.23)


# ═══════════════════════════════════════════════════════════════
# PHASE 5: CONTINUOUS LEARNING (Bottom)
# ═══════════════════════════════════════════════════════════════
draw_box(ax, 7, 4.85, 13, 0.45, 'PHASE 5: Continuous Learning Pipeline',
         C_PHASE, fontsize=10, bold=True, style='square')

# Buffer check diamond
draw_diamond(ax, 4, 3.75, 4.2, 1.0,
             'Buffer ≥ 50\nsamples?',
             C_DECISION, fontsize=8.5)

draw_arrow(ax, 2.5, 5.47, 2.5, 4.95)
draw_curved_arrow(ax, 2.5, 4.95, 4, 4.28, color=C_ARROW, rad=-0.3)

# YES - Retrain
draw_box(ax, 9.5, 3.75, 5.0, 0.8,
         'Server Retrains Decision Tree\n'
         'P_server=0.5W, t=2s → E_retrain=1J\n'
         'New model deployed to node',
         C_SERVER, fontsize=8)

draw_arrow(ax, 6.1, 3.75, 6.95, 3.75, label='YES')

# NO - Wait
draw_box(ax, 4, 2.55, 3.5, 0.6,
         'Continue sensing\n(buffer accumulating)',
         '#7D8C8E', fontsize=8, text_color='white')

draw_arrow(ax, 4, 3.22, 4, 2.88, label='NO', label_side='left')

# Adaptive threshold
draw_box(ax, 9.5, 2.55, 5.0, 0.65,
         'Adaptive Threshold Update\n'
         'τ(t) = 1.5 · std(errors) + 0.3',
         C_DATA, fontsize=8)

draw_arrow(ax, 9.5, 3.32, 9.5, 2.9)

# Feedback loop arrow back to top of Phase 4
draw_curved_arrow(ax, 12.5, 2.55, 13, 10.55, color='#C0392B', rad=0.4,
                  label='updated model\n& threshold', label_pos=(13.4, 6.5))

# Result summary box
draw_box(ax, 7, 1.3, 12.5, 0.9,
         'RESULTS: 72.25% Comm. Reduction │ 19.73% Energy Savings │ MAE 0.805°C │ '
         '277 Retrains (0.554%) │ Configurable: 5.5%–95.7% suppression via τ sweep',
         '#1B4F72', fontsize=9, bold=True, alpha=0.95)

# ── Legend ──
legend_y = 0.25
legend_items = [
    (C_SERVER, 'Server-Side'),
    (C_NODE, 'Node-Side'),
    (C_DECISION, 'Decision'),
    (C_DATA, 'Data/Storage'),
    (C_RESULT, 'Result/Output'),
]
start_x = 1.5
for i, (color, label) in enumerate(legend_items):
    bx = start_x + i * 2.5
    rect = FancyBboxPatch((bx - 0.35, legend_y - 0.15), 0.7, 0.3,
                          boxstyle="round,pad=0.05", facecolor=color, edgecolor=color)
    ax.add_patch(rect)
    ax.text(bx + 0.55, legend_y, label, fontsize=8, va='center', color='#333333')

plt.tight_layout()
plt.savefig('/home/user/Thesis--main/Thesis--main/results/figures/system_architecture_flowchart.png',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('/home/user/Thesis--main/Thesis--main/docs/system_architecture_flowchart.png',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print("Flow diagram saved successfully!")
