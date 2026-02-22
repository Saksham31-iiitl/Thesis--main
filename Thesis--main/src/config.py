"""
Central configuration for the WSN-IoT TinyML thesis simulation.
All constants, hardware parameters, energy model coefficients, and paths.
"""

import os
import numpy as np

# ──────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")

RAW_DATASET_PATH = os.path.join(RAW_DIR, "data.txt")
MOTE_LOCS_PATH = os.path.join(DATA_DIR, "mote_locs.txt")
CONNECTIVITY_PATH = os.path.join(DATA_DIR, "connectivity.txt")
SERVER_DATA_PATH = os.path.join(DATA_DIR, "server_train_60.csv")
NODE_DATA_PATH = os.path.join(DATA_DIR, "node_realtime_40.csv")

# ──────────────────────────────────────────────────────────────────────
# DATASET PARTITIONING
# ──────────────────────────────────────────────────────────────────────
SERVER_FRACTION = 0.60   # 60% for server-side TinyML training
NODE_FRACTION = 0.40     # 40% for simulated real-time node sensing

# ──────────────────────────────────────────────────────────────────────
# NETWORK TOPOLOGY
# ──────────────────────────────────────────────────────────────────────
NUM_NODES = 54                          # Intel Lab dataset: 54 motes
NUM_CLUSTERS = 4                        # Number of spatial-temporal clusters
BS_LOCATION = np.array([20.0, 16.0])    # Base station at center of lab

# ──────────────────────────────────────────────────────────────────────
# HEINZELMAN FIRST-ORDER RADIO ENERGY MODEL
# ──────────────────────────────────────────────────────────────────────
E_ELEC = 50e-9          # Energy to run radio electronics (J/bit) = 50 nJ/bit
E_FS = 10e-12           # Free-space amplifier (J/bit/m^2) = 10 pJ/bit/m^2
E_MP = 0.0013e-12       # Multi-path amplifier (J/bit/m^4) = 0.0013 pJ/bit/m^4
E_DA = 5e-9             # Data aggregation energy (J/bit/signal) = 5 nJ/bit/signal

# Crossover distance: below d0 use free-space, above use multi-path
D0 = np.sqrt(E_FS / E_MP)  # approx 87.7 meters

# ──────────────────────────────────────────────────────────────────────
# HARDWARE PARAMETERS (Mica2 / CC2420 compatible)
# ──────────────────────────────────────────────────────────────────────
V_SUPPLY = 3.0           # Supply voltage (V)
I_SENSE = 0.55e-3        # Sensor current draw (A)
I_PROC = 9.5e-3          # MCU active processing current (A)
I_TX = 17.4e-3           # Radio TX current (A)
I_RX = 18.8e-3           # Radio RX current (A)

T_SENSE = 0.010          # Sensing duration (s)
T_PROC = 0.005           # Processing duration (s)
T_INFER = 0.002          # TinyML inference time (s) — Decision Tree is fast

# ──────────────────────────────────────────────────────────────────────
# COMMUNICATION PARAMETERS
# ──────────────────────────────────────────────────────────────────────
PACKET_SIZE_BYTES = 200       # Payload size per transmission (bytes)
PACKET_SIZE_BITS = PACKET_SIZE_BYTES * 8  # 1600 bits
RADIO_BITRATE = 250e3         # IEEE 802.15.4 radio bitrate (bps)
T_TX = PACKET_SIZE_BITS / RADIO_BITRATE   # Transmission time per packet (s)

# ──────────────────────────────────────────────────────────────────────
# INITIAL NODE ENERGY
# ──────────────────────────────────────────────────────────────────────
E_INIT = 0.5             # Initial energy per node (Joules) — typical for 2xAA

# ──────────────────────────────────────────────────────────────────────
# THRESHOLD PARAMETERS
# ──────────────────────────────────────────────────────────────────────
THRESHOLD_DEFAULT = 1.0              # Default static threshold (degrees C)
THRESHOLD_SWEEP = np.arange(0.25, 3.25, 0.25)  # For threshold sweep analysis
ADAPTIVE_WINDOW = 50                 # Sliding window size for adaptive threshold
ADAPTIVE_ALPHA = 1.5                 # Multiplier for adaptive threshold
ADAPTIVE_BETA = 0.3                  # Offset for adaptive threshold

# ──────────────────────────────────────────────────────────────────────
# RETRAINING / CONTINUOUS LEARNING
# ──────────────────────────────────────────────────────────────────────
BUFFER_MIN = 50           # Minimum mismatch samples before retraining
RETRAIN_LEARNING_RATE = 0.01
P_SERVER = 0.5            # Server power consumption during retrain (W)
T_RETRAIN = 2.0           # Retraining duration (s)

# ──────────────────────────────────────────────────────────────────────
# TINYML MODEL
# ──────────────────────────────────────────────────────────────────────
MODEL_MAX_DEPTH = 8       # Decision Tree max depth
MODEL_RANDOM_STATE = 42
TARGET_VARIABLE = "temperature"

# Features used for prediction
FEATURE_COLUMNS = [
    "mote_id", "humidity", "light", "voltage",
    "hour", "minute", "day_of_week",
    "hour_sin", "hour_cos"
]

# ──────────────────────────────────────────────────────────────────────
# SPATIO-TEMPORAL CLUSTERING
# ──────────────────────────────────────────────────────────────────────
# Spatial weight multiplier: balances 2 spatial vs N temporal features.
# Equal weight per feature after StandardScaler; temporal features get
# sufficient influence to differentiate behavioral patterns.
SPATIAL_WEIGHT_ALPHA = 1.5

# Temporal features to use for clustering (only high-CV, discriminative ones)
# Selected by Coefficient of Variation across 54 nodes:
#   light_mean: CV=44.6% (best — sensors near windows vs corners)
#   diurnal_amplitude: CV=35.4% (day-night temperature swing)
#   std_temp: CV=27.9% (temperature variability)
# Dropped: mean_temp (CV=4.3%), mean_humidity (CV=5.2%) — too uniform indoors
CLUSTERING_TEMPORAL_FEATURES = ["std_temp", "diurnal_amplitude", "light_mean"]

# ──────────────────────────────────────────────────────────────────────
# CLUSTER HEAD SELECTION WEIGHTS
# ──────────────────────────────────────────────────────────────────────
CH_WEIGHT_ENERGY = 0.4    # alpha: weight for residual energy
CH_WEIGHT_DEGREE = 0.3    # beta:  weight for connectivity degree
CH_WEIGHT_DIST = 0.3      # gamma: weight for proximity to centroid

# ──────────────────────────────────────────────────────────────────────
# SIMULATION
# ──────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
SIM_SAMPLE_LIMIT = 10000    # Max node-side samples to simulate (None = all)
