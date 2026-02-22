# Thesis Explanation Document

## Title: "Towards Energy Efficient Data Transmission in WSN-assisted IoT using Spatio-Temporal Clustering and TinyML"

**Author:** Saksham (MCS24025)
**Institution:** Indian Institute of Information Technology Lucknow
**Advisor:** Dr. Rahul Kumar Verma
**Period:** Aug-Dec 2025

---

## Table of Contents

1. [What You Have Implemented (Overall System)](#1-what-you-have-implemented-overall-system)
2. [Why You Have Implemented This](#2-why-you-have-implemented-this)
3. [Mathematical Equations Used and Why](#3-mathematical-equations-used-and-why)
   - [3.1 Heinzelman First-Order Radio Energy Model](#31-heinzelman-first-order-radio-energy-model)
   - [3.2 Per-Component Energy (Hardware Model)](#32-per-component-energy-hardware-model)
   - [3.3 Cyclical Temporal Feature Encoding](#33-cyclical-temporal-feature-encoding)
   - [3.4 Spatio-Temporal Clustering (K-Means)](#34-spatio-temporal-clustering-k-means)
   - [3.5 Cluster Quality Metrics](#35-cluster-quality-metrics)
   - [3.6 Multi-Criteria Cluster Head Selection Score](#36-multi-criteria-cluster-head-selection-score)
   - [3.7 Mismatch Detection and Communication Reduction](#37-mismatch-detection-and-communication-reduction)
   - [3.8 Adaptive Threshold](#38-adaptive-threshold)
   - [3.9 Decision Tree Regression](#39-decision-tree-regression)
   - [3.10 Evaluation Metrics](#310-evaluation-metrics)
4. [File-by-File Detailed Explanation](#4-file-by-file-detailed-explanation)
   - [config.py](#file-1-srcconfigpy)
   - [data_loader.py](#file-2-srcdata_loaderpy)
   - [spatio_temporal_clustering.py](#file-3-srcspatio_temporal_clusteringpy)
   - [cluster_head_selection.py](#file-4-srccluster_head_selectionpy)
   - [tinyml_model.py](#file-5-srctinyml_modelpy)
   - [energy_model.py](#file-6-srcenergy_modelpy)
   - [node_simulation.py](#file-7-srcnode_simulationpy)
   - [evaluation.py](#file-8-srcevaluationpy)
   - [threshold_analysis.py](#file-9-srcthreshold_analysispy)
   - [run_experiment.py](#file-10-experimentsrun_experimentpy)
   - [visualization.py](#file-11-experimentsvisualizationpy)
5. [Data Files](#5-data-files)
6. [Project Directory Structure](#6-project-directory-structure)
7. [Summary of Key Results](#7-summary-of-key-results)

---

## 1. What You Have Implemented (Overall System)

A **complete end-to-end simulation framework** for energy-efficient data transmission in Wireless Sensor Networks (WSN) for IoT applications. The system has 5 major components:

| Component | What It Does |
|-----------|-------------|
| **Spatio-Temporal Clustering** | Groups 54 sensor nodes by location + behavior patterns |
| **Multi-Criteria Cluster Head Selection** | Picks the best node in each cluster as a relay |
| **TinyML Prediction Model** | Decision Tree that predicts temperature on-device |
| **Mismatch-Driven Transmission Suppression** | Only transmits when prediction error exceeds threshold |
| **Continuous Learning with Retraining** | Updates model when it drifts from reality |

### System Architecture Flow

```
Raw Data (2.3M records)
    |
    v  [data_loader.py]
Cleaned Data (1.06M records) --> Split 60%/40%
    |
    v  [spatio_temporal_clustering.py]
Clusters (K-Means, n=4) + CH Selection
    |
    v  [tinyml_model.py]
Trained Decision Tree Model (depth=8)
    |
    v  [node_simulation.py]
Parallel simulation: nodes predict, compare, transmit on mismatch
    |
    v  [threshold_analysis.py]
Sweep thresholds; analyze adaptive policy
    |
    v  [evaluation.py]
Compute comprehensive metrics
    |
    v  [visualization.py]
Generate 10+ publication-quality plots
```

---

## 2. Why You Have Implemented This

Traditional WSN nodes transmit **every** sensor reading to the base station. This wastes energy because:

- Many consecutive readings are similar (temperature doesn't change rapidly)
- Radio transmission is the **most expensive** operation (17.4 mA vs 0.55 mA for sensing)
- Once battery dies, the node is dead permanently

**The Solution:** Predict what the sensor will read using an on-device TinyML model, and only transmit when the prediction is wrong by more than a threshold. This suppresses redundant transmissions and saves energy.

### Key Research Questions Addressed

1. How can spatial clustering improve energy efficiency in WSNs?
2. Can TinyML models (suitable for microcontrollers) achieve adequate prediction accuracy?
3. What are the energy-latency trade-offs of continuous model retraining on edge devices?
4. How do spatio-temporal features improve clustering compared to purely spatial approaches?

### Main Contributions

1. **Spatio-Temporal Clustering:** Combines spatial (x, y coordinates) + temporal (sensor behavior patterns) features for better node organization
2. **TinyML Model Selection:** Compares 9 regression models; Decision Trees (depth 5-8) selected as optimal for microcontroller deployment
3. **Continuous Learning Pipeline:** Implements mismatch-driven retraining with quantified energy-latency overhead
4. **Comprehensive Energy Modeling:** Uses Heinzelman First-Order Radio Energy Model with per-phase accounting
5. **Real-World Validation:** Tested on Intel Berkeley Lab dataset (54 nodes, 2.3M readings, 37 days)

---

## 3. Mathematical Equations Used and Why

### 3.1 Heinzelman First-Order Radio Energy Model

**Why used:** This is the standard, widely-cited energy model for WSN research (from the LEACH protocol paper). It captures the key physics: transmitting over longer distances costs exponentially more energy.

#### Transmission Energy

```
E_Tx(k, d) = k * E_elec + k * epsilon * d^n
```

Where:

| Symbol | Value | Meaning |
|--------|-------|---------|
| `k` | 1600 bits (200 bytes) | Number of bits transmitted |
| `d` | varies (meters) | Distance between transmitter and receiver |
| `E_elec` | 50 nJ/bit | Energy to run the radio electronics |
| `epsilon` | `E_fs` if `d < d0`, else `E_mp` | Amplifier coefficient (depends on distance regime) |
| `n` | 2 if `d < d0`, else 4 | Path loss exponent |
| `E_fs` | 10 pJ/bit/m^2 | Free-space amplifier coefficient |
| `E_mp` | 0.0013 pJ/bit/m^4 | Multi-path amplifier coefficient |

#### Crossover Distance

```
d0 = sqrt(E_fs / E_mp) ~ 87.7 meters
```

**Why this threshold exists:** Below `d0`, signal propagation follows inverse-square law (free space). Above `d0`, multi-path reflections dominate, requiring d^4 power scaling. This determines which amplifier model to use.

#### Reception Energy

```
E_Rx(k) = k * E_elec
```

**Why distance-independent:** The receiver only needs to power its electronics to decode the signal, not amplify any outgoing signal.

#### Data Aggregation Energy

```
E_DA = E_da * k * n_signals = 5 nJ/bit/signal
```

**Why needed:** Cluster Heads must aggregate data from multiple members before forwarding. This accounts for the computation cost.

---

### 3.2 Per-Component Energy (Hardware Model)

**Why used:** Beyond the radio model, real nodes also consume energy for sensing, processing, and inference. The model captures all phases of a node's operation.

#### Individual Component Energies

```
E_sense   = I_sense   * V_supply * T_sense   = 0.55mA  * 3.0V * 10ms  = 16.5 uJ
E_process = I_proc    * V_supply * T_proc    = 9.5mA   * 3.0V * 5ms   = 142.5 uJ
E_infer   = I_proc    * V_supply * T_infer   = 9.5mA   * 3.0V * 2ms   = 57 uJ
```

#### Per-Epoch Energy for a Member Node

```
E_node(t) = E_sense + E_process + E_infer + delta(t) * E_Tx(k, d_CH)
```

Where `delta(t) = 1` if mismatch detected (transmit), `0` if prediction is acceptable (suppress).

**This is where energy savings happen:** When `delta(t) = 0`, we save the entire `E_Tx` component.

#### Per-Epoch Energy for a Cluster Head

```
E_CH(t) = E_sense + E_process + E_infer + n_tx * E_Rx(k) + E_DA(k, n_tx) + E_Tx(k, d_BS)
```

Where `n_tx` = number of members that transmitted in this epoch, `d_BS` = distance from CH to base station.

---

### 3.3 Cyclical Temporal Feature Encoding

**Why used:** Hours wrap around (23 -> 0), so treating "hour" as a linear number creates an artificial discontinuity. Hour 23 appears far from hour 0 numerically, but they are only 1 hour apart in reality.

```
hour_sin = sin(2 * pi * hour / 24)
hour_cos = cos(2 * pi * hour / 24)
```

**Why both sin AND cos:** A single sin function is ambiguous -- `sin(6h) = sin(18h)`. Using both sin and cos provides a unique 2D point for every hour on the "clock circle," resolving the ambiguity completely.

---

### 3.4 Spatio-Temporal Clustering (K-Means)

**Why used:** Traditional protocols (LEACH, HEED) cluster nodes by physical location only. But two physically close nodes may behave very differently (e.g., one near a window, one near a heater). Grouping by behavior + location creates clusters with lower intra-cluster variance, enabling better per-cluster predictions.

#### Feature Vector per Node (6 dimensions)

```
F_i = [x_i, y_i, mu_temp(i), sigma_temp(i), mu_humidity(i), A_diurnal(i)]
```

Where:

| Feature | Description |
|---------|-------------|
| `x_i, y_i` | Physical coordinates of node i |
| `mu_temp(i)` | Mean temperature reading of node i across training data |
| `sigma_temp(i)` | Standard deviation of temperature (variability) |
| `mu_humidity(i)` | Mean humidity reading |
| `A_diurnal(i)` | Diurnal amplitude (day-night temperature swing) |

#### Diurnal Amplitude

```
A_diurnal(i) = |T_bar_day(i) - T_bar_night(i)|
```

Where day = hours 8-20, night = hours 0-7 and 21-23.

**Why this feature matters:** Captures how much a node's temperature swings between day and night. Nodes near windows have high diurnal amplitude; interior nodes have low amplitude.

#### StandardScaler Normalization (applied before K-Means)

```
z = (x - mu) / sigma
```

**Why required:** Spatial features (meters) and temporal features (degrees C, %) have different scales. Without normalization, features with larger magnitudes would dominate the distance calculation in K-Means.

#### K-Means Objective Function

```
J = SUM_{c=1}^{K} SUM_{i in C_c} ||F_i - mu_c||^2
```

Minimizes total within-cluster variance across K=4 clusters.

---

### 3.5 Cluster Quality Metrics

#### Silhouette Score

```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

Where:

- `a(i)` = mean distance from point i to all other points in the **same** cluster
- `b(i)` = mean distance from point i to points in the **nearest neighboring** cluster
- Range: [-1, 1]. Higher = better separation.

**Why used:** Measures how well-separated clusters are. A high silhouette score means nodes are correctly grouped.

#### Davies-Bouldin Index

```
DB = (1/K) * SUM_{c=1}^{K} max_{c != c'} [(S_c + S_{c'}) / d(mu_c, mu_{c'})]
```

Where `S_c` = average distance within cluster c, `d(mu_c, mu_{c'})` = distance between centroids.

**Why used:** Lower = better. Complementary to Silhouette -- measures how compact clusters are relative to their separation.

---

### 3.6 Multi-Criteria Cluster Head Selection Score

**Why used:** Selecting CH by just one criterion (e.g., highest energy) is suboptimal. A node with high energy but poor connectivity wastes energy on multi-hop. This score balances three factors.

```
Score(i) = alpha * E_norm(i) + beta * Deg_norm(i) + gamma * Prox_norm(i)
```

Where:

| Weight | Value | Factor | Reason |
|--------|-------|--------|--------|
| `alpha` | 0.4 | Residual Energy | Most important -- CH depletes faster |
| `beta` | 0.3 | Connectivity Degree | Good connectivity = fewer hops |
| `gamma` | 0.3 | Proximity to Centroid | Central CH reduces member TX distance |

#### Normalized Residual Energy

```
E_norm(i) = E_residual(i) / max(E_residual in cluster)
```

#### Normalized Degree

```
Deg_norm(i) = degree(i) / max(degree in cluster)
```

Where `degree(i)` = number of neighbors with link probability > 0.5.

#### Normalized Proximity (inverted distance)

```
Prox_norm(i) = 1 - (d(i, centroid) / max_d_in_cluster)
```

**Why inverted:** Closer to centroid = higher score. A CH at the centroid minimizes total member-to-CH distance, reducing overall transmission energy.

---

### 3.7 Mismatch Detection and Communication Reduction

```
error(t) = |y_hat(t) - y(t)|
transmit(t) = 1  if error(t) > tau,  else 0
```

Where:

- `y_hat(t)` = model prediction for time t
- `y(t)` = actual sensor reading at time t
- `tau` = threshold (default 1.0 degrees C)

#### Communication Reduction

```
CR = 1 - (N_transmitted / N_total)
```

#### Mismatch Rate

```
MR = N_transmitted / N_total
```

---

### 3.8 Adaptive Threshold

**Why used:** A fixed threshold doesn't adapt to changing conditions. When the model is accurate, we can suppress more aggressively (lower threshold). When errors are high, we should transmit more to let the server correct the model.

```
tau(t) = alpha * std(errors[t-W : t]) + beta
```

Where:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `alpha` | 1.5 | Sensitivity multiplier |
| `beta` | 0.3 | Minimum threshold floor |
| `W` | 50 | Sliding window size |
| `std(errors[t-W:t])` | varies | Standard deviation of recent prediction errors |

**Why this formula works -- Self-regulating feedback loop:**

1. Model drifts -> errors increase -> std increases -> tau increases
2. Higher tau -> more transmissions -> buffer fills faster -> retraining triggers
3. New model is better -> errors decrease -> std decreases -> tau tightens
4. Tighter tau -> more suppression -> energy saved -> cycle repeats

---

### 3.9 Decision Tree Regression

**Why Decision Tree was chosen over other models:**

| Property | Decision Tree | Random Forest | Linear Regression |
|----------|--------------|---------------|-------------------|
| Model Size | ~48 KB | 3.6-4.8 MB | ~1 KB |
| Inference Time | <1 ms | 12-15 ms | <0.5 ms |
| R^2 Score | ~0.89 | ~0.95 | ~0.65 |
| MCU Compatible | Yes | No (too large) | Yes (too inaccurate) |

#### Prediction at Inference (at each leaf)

```
y_hat = mean(y_training_samples_reaching_this_leaf)
```

#### Split Criterion During Training

```
MSE = (1/n) * SUM(y_i - y_bar)^2
```

At each node, the tree finds the feature and split point that minimizes the weighted sum of MSE in the two child nodes.

---

### 3.10 Evaluation Metrics

#### Mean Absolute Error

```
MAE = (1/n) * SUM |y_hat_i - y_i|
```

#### Root Mean Squared Error

```
RMSE = sqrt[(1/n) * SUM (y_hat_i - y_i)^2]
```

#### Coefficient of Determination (R-squared)

```
R^2 = 1 - [SUM(y_i - y_hat_i)^2 / SUM(y_i - y_bar)^2]
```

#### Data Staleness (per node)

```
staleness(t, m) = t - last_update_slot(m)
```

#### Latency per Cycle

```
T_cycle = T_sense + T_infer + T_compare + delta(t) * T_TX
```

#### Server Retraining Energy

```
E_retrain_total = N_retrains * P_server * T_retrain
```

---

## 4. File-by-File Detailed Explanation

---

### FILE 1: `src/config.py`

**Lines:** 118
**Purpose:** Central configuration file -- the single source of truth for all constants, paths, and parameters used across the entire project.

**Why it exists:** Having all parameters in one file prevents "magic numbers" scattered across code. When you want to change the threshold from 1.0 degrees C to 1.5 degrees C, you change it in ONE place, not hunt through 10 files.

#### Section Breakdown

| Lines | Section | What It Defines |
|-------|---------|-----------------|
| 12-23 | Paths | Directory structure using `os.path.join` for cross-platform compatibility. `ROOT_DIR` auto-detects the project root. |
| 28-29 | Partitioning | `SERVER_FRACTION=0.60` -- 60% for training, 40% for testing (standard ML split). |
| 34-36 | Network Topology | 54 nodes (matching Intel Berkeley dataset), 4 clusters (empirically chosen), base station at center (20, 16). |
| 41-47 | Heinzelman Model | Energy coefficients from the original Heinzelman et al. paper. `D0 ~ 87.7m` is mathematically derived, not hardcoded. |
| 52-60 | Hardware Params | Mica2/CC2420 datasheet values for current draw and timing durations. |
| 65-68 | Communication | 200-byte payload x 8 = 1600 bits. 250 kbps IEEE 802.15.4 radio gives 6.4 ms transmission time. |
| 73 | Initial Energy | `E_INIT = 0.5J` -- typical for 2xAA battery powered motes. |
| 78-82 | Threshold | Default 1.0 degrees C. Sweep range 0.25-3.0 degrees C in 0.25 steps. Adaptive parameters: alpha=1.5, beta=0.3, window=50. |
| 87-90 | Retraining | Buffer minimum 50 samples. Server draws 0.5W for 2 seconds per retrain = 1J per retrain. |
| 95-104 | TinyML Model | Decision Tree max depth 8. Feature list includes raw sensor values + engineered temporal features. |
| 109-111 | CH Weights | alpha=0.4, beta=0.3, gamma=0.3 -- energy weighted highest because CH consumes more energy. |
| 116-117 | Simulation | Random seed 42 for reproducibility. Sample limit 10,000 for manageable runtime. |

---

### FILE 2: `src/data_loader.py`

**Lines:** 196
**Purpose:** Loads, cleans, engineers features for, and partitions the Intel Berkeley Lab dataset.

**Why it exists:** Raw data has missing values, outliers, and no temporal features. This file transforms raw sensor logs into ML-ready datasets.

#### Function: `load_raw_dataset()` (lines 27-37)

- Reads `data.txt` (2.3M lines) using pandas `read_csv` with whitespace separator
- Columns: date, time, epoch, mote_id, temperature, humidity, light, voltage
- `on_bad_lines="skip"` handles corrupted rows gracefully

#### Function: `clean_dataset()` (lines 40-68)

- Drops rows with any NaN in sensor columns
- Removes physically impossible values using known valid ranges:

| Sensor | Valid Range | Why This Range |
|--------|------------|----------------|
| Temperature | -10 to 60 degrees C | Sensor hardware limits |
| Humidity | 0 to 100% | Physical definition |
| Light | 0 to 2500 lux | Sensor saturation point |
| Voltage | 1.5 to 3.0V | 2xAA battery operating range |

- **Result:** ~1,059,767 valid records out of 2.3M (45.8% retention)
- Values outside these ranges indicate sensor malfunction, not real data

#### Function: `parse_datetime()` (lines 71-83)

- Combines date + time strings into a proper datetime object
- Sorts by datetime -- **critical for chronological partitioning**
- Reports date range: 37 days (Feb 28 - Apr 5, 2004)

#### Function: `extract_temporal_features()` (lines 86-106)

- Extracts: `hour` (0-23), `minute` (0-59), `day_of_week` (0=Mon, 6=Sun)
- Computes cyclical features: `hour_sin` and `hour_cos`
- Drops original string date/time columns (no longer needed)
- **Why cyclical encoding:** Raw hour 23 is "far" from hour 0 linearly, but they are 1 hour apart. Sin/cos encoding places them close in feature space.

#### Function: `partition_data()` (lines 109-138)

- **Chronological split:** First 60% by time -> server training. Last 40% -> node testing.
- **Why chronological (not random)?** Random splitting would leak future information into training data. In a real deployment, the model is trained on past data and tested on future data. Chronological splitting simulates this reality.
- **Assertion at line 127:** Verifies no temporal overlap between partitions (strict independence guarantees no data leakage).

#### Function: `run_data_pipeline()` (lines 177-191)

- Orchestrates the full pipeline: load -> clean -> parse datetime -> extract features -> partition -> save
- Called by the master experiment runner

---

### FILE 3: `src/spatio_temporal_clustering.py`

**Lines:** 231
**Purpose:** Clusters 54 sensor nodes using spatial AND temporal features. Provides comparison with spatial-only baseline.

**Why it exists:** This is the **key novelty** of the thesis. Traditional protocols (LEACH, HEED) cluster by location only. This approach adds behavioral features for more meaningful groupings.

#### Function: `extract_temporal_profiles()` (lines 42-92)

For each of the 54 nodes, computes 4 temporal statistics from the server training data:

| Feature | Computation | What It Captures |
|---------|-------------|------------------|
| `mean_temp` | Average temperature | Baseline temperature at this location |
| `std_temp` | Std deviation of temperature | How variable the readings are |
| `mean_humidity` | Average humidity | Moisture level at this location |
| `diurnal_amplitude` | abs(day_avg - night_avg) temp | Day-night temperature swing |

- **Fallback (line 61):** If a node has <10 readings, uses global means to handle sparse nodes robustly.
- **Why diurnal amplitude matters:** Nodes near windows/doors have high day-night temperature difference; interior nodes have low. This captures environmental exposure that pure location cannot.

#### Function: `spatial_clustering()` (lines 95-107)

- **Baseline method:** Uses only (x, y) coordinates
- StandardScaler normalizes coordinates before K-Means
- K-Means with K=4, n_init=10 (runs 10 times with different initializations, picks best result)
- Computes Silhouette and Davies-Bouldin scores for quality evaluation

#### Function: `spatio_temporal_clustering()` (lines 110-130)

- **Proposed method:** Uses 6 features: [x, y, mean_temp, std_temp, mean_humidity, diurnal_amplitude]
- Same K-Means setup but operates in 6-dimensional feature space
- StandardScaler is **critical** here -- without it, features measured in meters would dominate features measured in degrees
- **Result:** Higher Silhouette score (~0.42 vs ~0.21 for spatial-only), proving better cluster quality

#### Function: `compute_cluster_stats()` (lines 133-150)

- Computes per-cluster: size, centroid coordinates, mean/max intra-cluster Euclidean distance
- Used for reporting and downstream CH distance calculations

#### Function: `run_clustering()` (lines 153-225)

- Orchestrates both methods, prints side-by-side comparison
- Saves results to `clustering_comparison.csv`
- Assigns spatio-temporal labels as the **primary clustering** (used by all downstream modules)
- Also preserves spatial labels in a separate column for comparison plots

---

### FILE 4: `src/cluster_head_selection.py`

**Lines:** 191
**Purpose:** Selects one Cluster Head per cluster using a weighted multi-criteria score.

**Why it exists:** CH selection directly affects network lifetime. A poor CH choice (e.g., low energy node far from center) causes premature battery death and wastes transmission energy for all cluster members.

#### Function: `load_connectivity()` (lines 31-53)

- Reads `connectivity.txt`: each line is `src_id dst_id probability`
- Filters links with probability > 0.5 (strong, reliable links only)
- Counts strong neighbors per node -> `degree`
- **Why probability > 0.5:** Unreliable links cause retransmissions, wasting more energy than they save.

#### Function: `initialize_nodes()` (lines 56-77)

- Sets `residual_energy = 0.5J` for all 54 nodes (2xAA battery equivalent)
- Merges connectivity degrees from the connectivity file
- Computes Euclidean distance to base station for each node: `d = sqrt((x - BS_x)^2 + (y - BS_y)^2)`

#### Function: `compute_ch_scores()` (lines 80-120)

For each cluster independently:

1. Computes distance from each node to the cluster centroid
2. Normalizes energy, degree, and proximity to [0, 1] range **within** the cluster
3. Computes weighted score: `Score = 0.4 * E_norm + 0.3 * Deg_norm + 0.3 * Prox_norm`

- **Why normalize within cluster:** Prevents a cluster with generally higher energy from dominating. Each cluster's best candidate is selected relative to that cluster's own range.
- **Why invert distance (line 106):** `Prox_norm = 1 - (d/d_max)` ensures closer nodes get higher proximity scores.

#### Function: `select_cluster_heads()` (lines 123-138)

- For each cluster, picks the node with the highest score using `idxmax()`
- Sets `is_ch = True` flag for the selected nodes
- Returns updated DataFrame with CH assignments and list of CH mote_ids

#### Utility Functions (lines 141-162)

- `get_ch_for_node()`: Returns which CH a given node reports to
- `get_node_to_ch_distance()`: Euclidean distance from member to its CH
- `get_ch_to_bs_distance()`: Euclidean distance from CH to base station

---

### FILE 5: `src/tinyml_model.py`

**Lines:** 164
**Purpose:** Trains a Decision Tree regression model for temperature prediction. This model is designed to run on-device (TinyML).

**Why it exists:** The core of the energy-saving mechanism. If we can predict temperature accurately on the node itself, we don't need to transmit the reading to the server.

#### Function: `prepare_features()` (lines 32-41)

- Extracts 9 features: mote_id, humidity, light, voltage, hour, minute, day_of_week, hour_sin, hour_cos
- Ensures all values are numeric, fills NaN with 0
- Target variable: temperature
- **Why these features:** Humidity, light, and voltage correlate with temperature physically. Temporal features capture daily/weekly patterns. mote_id captures location-specific bias without needing explicit coordinates.

#### Function: `train_initial_model()` (lines 44-88)

- Creates `DecisionTreeRegressor(max_depth=8, random_state=42)`
- **Why max_depth=8:** Deeper trees overfit and become too large for microcontrollers. Depth 8 gives at most 256 leaves, resulting in ~48 KB model. This fits comfortably in the 128 KB flash of a Mica2 mote.
- Trains on full server partition (~636K samples)
- Computes and reports training metrics:

| Metric | What It Tells Us |
|--------|------------------|
| MAE | Average prediction error in degrees C |
| MSE | Squared error (penalizes large errors more) |
| R^2 | Fraction of variance explained by the model |
| n_nodes | Total nodes in the tree (complexity) |
| max_depth_actual | Actual depth reached (may be less than max allowed) |
| model_size_kb | Estimated size: `node_count * 40 bytes / 1024` |

- **Model size estimate:** Each tree node stores a feature index, threshold value, left/right child pointers, and leaf value, totaling approximately 40 bytes per node.

#### Function: `evaluate_on_node_data()` (lines 91-122)

- Tests model on the 40% node partition (data the model has never seen)
- Computes absolute errors: `|prediction - actual|`
- Mismatch mask: `errors > threshold`
- Key outputs: mismatch_rate (fraction exceeding threshold), communication_reduction (1 - mismatch_rate)
- **This tells us:** What fraction of transmissions can be suppressed by the model

#### Function: `retrain_model()` (lines 125-138)

- Creates a **new** Decision Tree trained on buffered mismatch samples
- **Why a new tree, not an update:** Decision Trees don't natively support incremental learning. The "retraining" replaces the model with one trained specifically on recent mismatched data, which represents the distribution the old model was failing on.
- **Why only mismatch samples:** These are the samples where the current model is wrong -- training on them focuses the new model on the current data distribution (concept drift adaptation).

---

### FILE 6: `src/energy_model.py`

**Lines:** 135
**Purpose:** Implements the Heinzelman First-Order Radio Energy Model for computing energy consumption of each WSN operation.

**Why it exists:** To accurately quantify the energy savings of the proposed approach vs. the baseline (always transmitting).

#### Function: `energy_transmit()` (lines 30-38)

- Implements: `E_Tx = k * E_elec + k * epsilon * d^n`
- Uses free-space model (d^2) when distance < 87.7m, multi-path model (d^4) when above
- **Why distance-dependent:** Radio signal power decays with distance. Longer distance requires exponentially more amplifier energy.

#### Function: `energy_receive()` (lines 41-43)

- `E_Rx = k * E_elec` -- distance-independent
- Only powers the radio electronics to decode incoming signals

#### Function: `energy_aggregation()` (lines 46-48)

- `E_DA = 5 nJ/bit/signal * k * n_signals`
- Accounts for computation cost when CH combines multiple members' data

#### Functions: `energy_sense()`, `energy_process()`, `energy_infer()` (lines 51-63)

- Simple `I * V * T` calculations using Mica2/CC2420 hardware datasheet values
- `energy_infer()` uses MCU processor current for inference duration (2 ms for Decision Tree)

#### Function: `energy_node_epoch()` (lines 66-83) -- **Core savings function**

- Every epoch: sense + process + infer (always consumed)
- **Only if mismatch** (transmit=True): adds `E_Tx(k, d_to_CH)`
- **This is where savings happen:** When transmit=False, the entire transmission energy component is saved

#### Function: `energy_ch_epoch()` (lines 86-117)

- CH does everything a member does PLUS:
  - Receives from `n_transmitting` members: `n_tx * E_Rx(k)`
  - Aggregates received data: `E_DA(k, n_tx)`
  - Forwards aggregated result to base station: `E_Tx(k, d_BS)`
- **Why CH is modeled separately:** CH consumes significantly more energy than members, which is precisely why CH rotation based on residual energy is important.

#### Function: `energy_baseline_epoch()` (lines 120-134)

- The comparison scenario: **no TinyML, every node always transmits** every reading
- No inference energy (no model running), but always pays full transmission cost
- This enables computing: `energy_savings = 1 - (proposed / baseline)`

---

### FILE 7: `src/node_simulation.py`

**Lines:** 235
**Purpose:** The main simulation engine that runs the parallel node-server operation with mismatch detection and continuous retraining.

**Why it exists:** This is the **heart of the thesis** -- it simulates how the proposed system would work in a real deployment over time, producing all the experimental results.

#### Function: `run_simulation()` (lines 27-197)

##### Setup Phase (lines 29-68)

- Loads node data (40% partition), applies sample limit (default 10,000 for manageable runtime)
- Prepares feature matrix `X` and target vector `y` using `prepare_features()`
- **Precomputes node-to-CH distances (lines 43-56):** For each node, calculates Euclidean distance to its assigned cluster head. CHs have distance 0 to themselves. This avoids recomputing distances on every iteration.
- Initializes log arrays: `log_transmitted` (boolean mask), `log_error` (prediction errors), `log_energy_proposed`, `log_energy_baseline`, `retrain_events` list
- Each node starts with `E_INIT = 0.5J` of energy in the `node_energy` dictionary

##### Simulation Loop (lines 79-164) -- Chunked Processing

**Why chunked (not sample-by-sample):** Processing one sample at a time in Python is extremely slow for 10K+ samples. Instead, the model predicts an entire batch at once, then identifies where retraining should be triggered.

**Per chunk, the algorithm does:**

1. **Batch predict (line 87):** `current_model.predict(remaining_X)` -- predicts all remaining samples at once
2. **Compute errors (line 88):** `errors = |predictions - actual|` (vectorized)
3. **Find mismatches (line 89):** `mismatch_mask = errors > threshold`
4. **Find retrain trigger point (lines 95-100):** Counts through mismatch indices. If existing buffer + new mismatches >= `buffer_min` (50), identifies the exact sample index where retraining should trigger
5. **Process segment up to retrain point (lines 112-138):**
   - Logs errors and transmission decisions for each sample
   - Computes per-sample energy: proposed (with/without TX) and baseline (always TX)
   - Deducts energy from each node's remaining battery in `node_energy`
   - Buffers mismatch samples (features + true target) for future retraining
6. **Retrain if buffer is full (lines 150-163):**
   - Converts buffer lists to DataFrame
   - Calls `retrain_model()` -- creates new Decision Tree on buffered data
   - Logs retrain event with slot number and cumulative mismatch rate
   - Clears buffer and continues with the new model

##### Output (lines 175-197)

Returns a comprehensive dictionary containing:

| Key | Type | Content |
|-----|------|---------|
| `communication_reduction` | float | Fraction of transmissions suppressed |
| `energy_proposed_J` | float | Total energy consumed (proposed method) |
| `energy_baseline_J` | float | Total energy consumed (baseline: always TX) |
| `energy_savings` | float | `1 - (proposed / baseline)` |
| `mismatch_rate` | float | Fraction of |pred - actual| > threshold |
| `mae`, `rmse` | float | Prediction quality metrics |
| `n_retrains` | int | Number of retraining events |
| `log_transmitted` | np.array | Boolean array of transmission decisions |
| `log_error` | np.array | Prediction error at each time slot |
| `log_energy_proposed` | np.array | Energy per slot (proposed) |
| `log_energy_baseline` | np.array | Energy per slot (baseline) |
| `retrain_events` | list[dict] | Slot and mismatch rate at each retrain |
| `node_energy_remaining` | dict | Residual energy per node |

---

### FILE 8: `src/evaluation.py`

**Lines:** 181
**Purpose:** Computes comprehensive evaluation metrics beyond what the simulation directly produces.

**Why it exists:** The thesis needs multiple evaluation dimensions: energy efficiency, communication reduction, prediction quality, network lifetime, data staleness, and latency.

#### Function: `compute_network_lifetime()` (lines 26-40)

- Takes remaining energy dictionary -> counts alive/dead nodes
- Computes min/max/avg energy remaining across all nodes
- `energy_consumed_fraction = 1 - (avg_remaining / E_init)`
- **Why this metric matters:** Network lifetime is a primary WSN evaluation criterion. "First node death" determines when coverage gaps appear in the monitored area.

#### Function: `compute_latency()` (lines 43-62)

- **Proposed latency:** `T_cycle = T_sense + T_infer + T_compare + delta * T_TX`
  - Non-transmitting slots are faster because there is no 6.4 ms radio delay
- **Baseline latency:** `T_cycle = T_sense + T_compare + T_TX` (always transmits, no inference overhead)
- **Why this matters:** Shows that while inference adds 2 ms of overhead, suppressed slots save 6.4 ms of TX time, resulting in a net latency improvement.

#### Function: `compute_staleness()` (lines 65-86)

- For each time slot, computes how many slots since the server last received an update from each node
- Iterates through all slots, tracking `last_update[mote_id]`
- `staleness = current_slot - last_update[mote_id]`
- Reports avg/max/median staleness
- **Why this metric matters:** High communication reduction is useless if the server has extremely stale (outdated) data. This quantifies the freshness cost of suppression.

#### Function: `compute_server_energy()` (lines 89-100)

- `E_retrain_total = N_retrains * P_server * T_retrain = N_retrains * 0.5W * 2s`
- **Why separate accounting:** Server energy comes from mains power, not batteries. But it still matters for total system energy budget and sustainability analysis.

#### Function: `generate_full_report()` (lines 103-180)

- Prints a formatted report with all metrics organized in 6 sections:
  1. **Communication:** Total slots, transmissions, suppressed, CR%
  2. **Node Energy:** Proposed total, baseline total, savings%
  3. **Server Energy:** Retraining events, energy per retrain, total server energy
  4. **Prediction Quality:** MAE, RMSE
  5. **Continuous Learning:** Retrain events, frequency
  6. **Latency:** Average proposed vs baseline, max latency
  7. **Network Lifetime:** Alive/dead nodes, energy consumed fraction
  8. **Data Staleness:** Average, max, median staleness in slots

---

### FILE 9: `src/threshold_analysis.py`

**Lines:** 186
**Purpose:** Analyzes the impact of the mismatch threshold on all metrics, and implements an adaptive threshold policy.

**Why it exists:** The threshold `tau` is the key tuning parameter. Too low -> too many transmissions (poor savings). Too high -> too much suppression (stale, inaccurate data). This analysis quantifies the trade-off and proposes an adaptive solution.

#### Function: `threshold_sweep_fast()` (lines 29-60)

- **Key optimization:** Predicts ONCE, then evaluates at 12 different thresholds (0.25 to 3.0 degrees C)
- For each threshold tau:
  - `mismatch_mask = |prediction - actual| > tau`
  - Computes: CR, mismatch rate, MAE, RMSE, transmission count, suppressed count
- **Why "fast":** No retraining involved -- shows the static model's behavior across thresholds in isolation. This separates the threshold effect from the retraining effect.
- **Output:** DataFrame with 12 rows, one per threshold value

#### Function: `adaptive_threshold_simulation()` (lines 63-148)

- **Per-sample processing (line 95 loop):** Must be sample-by-sample because the threshold changes at every step based on recent history.
- Each step:
  1. Predict `y_hat = model.predict(x_i)`
  2. Compute `error = |y_hat - y_true|`
  3. Append error to sliding window (size W=50)
  4. Compute adaptive threshold: `tau(t) = 1.5 * std(recent_errors) + 0.3`
  5. Decide: transmit if `error > tau`
  6. If transmit: buffer the sample
  7. If buffer >= 50: retrain model and clear buffer
- **Conservative start (line 111):** For the first 5 samples, uses `tau = beta + 1.0 = 1.3 degrees C` because std is unreliable with fewer than 5 data points.
- **Why adaptive matters:** Creates a self-regulating feedback loop (described in Section 3.8).

---

### FILE 10: `experiments/run_experiment.py`

**Lines:** 137
**Purpose:** Master pipeline that orchestrates all 8 phases of the experiment in the correct sequence.

**Why it exists:** Ensures complete reproducibility -- running this single file produces all results, metrics, and plots from scratch.

#### Execution Flow

| Phase | Lines | What Happens |
|-------|-------|-------------|
| **Phase 1** | 52-59 | Checks if partitioned CSVs exist. If yes, loads them (saves time). If no, runs full data pipeline. |
| **Phase 2** | 62-64 | Runs spatio-temporal clustering -> gets cluster assignments. Then runs CH selection -> identifies best CH per cluster. |
| **Phase 3** | 67-70 | Trains Decision Tree on server data, evaluates on node data. Returns model + metrics. |
| **Phase 4** | 73-83 | Runs full simulation with threshold=1.0 degrees C, buffer_min=50. Prints and saves all results. |
| **Phase 5** | 85-86 | Implicit -- energy and communication metrics are already computed within Phase 4. |
| **Phase 6** | 89-92 | Runs static threshold sweep (12 thresholds) + adaptive threshold simulation. |
| **Phase 7** | 95-102 | Generates comprehensive evaluation report with all metrics. Computes staleness array for plotting. |
| **Phase 8** | 105-112 | Generates all 10 thesis-quality plots and saves to `results/figures/`. |

#### To Run

```bash
cd experiments/
python run_experiment.py
```

---

### FILE 11: `experiments/visualization.py`

**Lines:** 334
**Purpose:** Generates all publication-quality thesis plots (10 figures at 300 DPI as PNG files).

**Why it exists:** Thesis and presentation require professional, high-resolution visualizations of all results.

#### Plot Descriptions

| Plot # | Filename | What It Shows | Why It Matters |
|--------|----------|---------------|----------------|
| 01 | `01_threshold_sweep.png` | Three subplots: CR vs tau, Mismatch Rate vs tau, MAE vs tau | Shows the fundamental energy-accuracy trade-off |
| 02 | `02_cumulative_energy.png` | Two curves (baseline vs proposed) with green fill showing savings | Visualizes total energy savings over time |
| 03 | `03_communication_events.png` | Smoothed transmission rate over time + red lines at retrain events | Shows how retraining affects transmission patterns |
| 04 | `04_retraining_events.png` | Step plot of cumulative retraining count | Shows retraining frequency and distribution |
| 05 | `05_mismatch_over_time.png` | Cumulative mismatch rate evolution | Shows if model improves, degrades, or stabilizes |
| 06 | `06_cluster_visualization.png` | 54 nodes colored by cluster, stars for CHs, square for BS | Visualizes spatial layout and cluster structure |
| 07 | `07_clustering_comparison.png` | Side-by-side: spatial-only vs spatio-temporal clustering | Visually proves spatio-temporal clustering is different |
| 08 | `08_staleness_distribution.png` | Histogram with mean/median markers | Shows the data freshness cost of suppression |
| 09 | `09_energy_comm_summary.png` | Two bar charts: energy comparison + communication events | Quick visual summary of key results |
| 10 | `10_adaptive_threshold.png` | Two panels: threshold+errors over time, TX rate over time | Demonstrates self-regulating adaptive behavior |

#### Technical Details

- Uses `matplotlib` with `Agg` backend (no display window needed)
- Global style: font size 12, axis labels 13, titles 14, legend 10
- All plots saved at 300 DPI with tight bounding box
- Downsampling applied for large arrays (every Nth point) to keep file sizes reasonable

---

## 5. Data Files

### `data/raw/data.txt` (2.3M lines)

- **Source:** Intel Berkeley Research Lab sensor network dataset
- **Format:** Space-delimited: date, time, epoch, mote_id, temperature, humidity, light, voltage
- **Coverage:** 54 Mica2Dot motes, 37 days (Feb 28 - Apr 5, 2004)
- **Why this dataset:** Real-world, widely-cited WSN benchmark. Provides realistic sensor noise, failures, and temporal patterns.

### `data/mote_locs.txt` (54 lines)

- **Format:** mote_id, x_coordinate, y_coordinate
- **Content:** Physical (x, y) locations of all 54 sensor motes in the Intel Berkeley lab
- **Used by:** Clustering and CH selection for distance computations

### `data/connectivity.txt` (2700+ lines)

- **Format:** source_mote, dest_mote, link_probability
- **Content:** Network topology -- which nodes can communicate with which, and how reliably
- **Used by:** CH selection for computing connectivity degree

### `data/server_train_60.csv` (~424K rows)

- Processed training data (first 60% chronologically)
- Columns: epoch, mote_id, temperature, humidity, light, voltage, datetime, hour, minute, day_of_week, hour_sin, hour_cos
- **Used by:** Model training and temporal profile extraction

### `data/node_realtime_40.csv` (~636K rows)

- Processed testing data (last 40% chronologically)
- Same columns as server_train_60.csv
- **Used by:** Simulation, threshold analysis, model evaluation

---

## 6. Project Directory Structure

```
Thesis--main/
|
|-- src/                                    # Core source code (9 Python modules)
|   |-- __init__.py
|   |-- config.py                           # Central configuration & constants
|   |-- data_loader.py                      # Data loading, cleaning, feature extraction
|   |-- spatio_temporal_clustering.py       # K-Means: spatial vs spatio-temporal
|   |-- cluster_head_selection.py           # Multi-criteria CH selection
|   |-- tinyml_model.py                     # Decision Tree training & inference
|   |-- energy_model.py                     # Heinzelman radio energy model
|   |-- node_simulation.py                  # Main simulation engine
|   |-- evaluation.py                       # Comprehensive metrics computation
|   |-- threshold_analysis.py              # Threshold sweep & adaptive threshold
|
|-- experiments/                            # Experiment orchestration
|   |-- __init__.py
|   |-- run_experiment.py                   # Master pipeline executor
|   |-- visualization.py                    # All thesis plot generation (10 plots)
|
|-- data/                                   # Input data
|   |-- raw/
|   |   |-- data.txt                        # Intel Berkeley Lab dataset (2.3M lines)
|   |-- mote_locs.txt                       # 54 sensor node coordinates
|   |-- connectivity.txt                    # Network connectivity matrix
|   |-- server_train_60.csv                 # 60% partition for training
|   |-- node_realtime_40.csv                # 40% partition for simulation
|
|-- results/                                # Output artifacts
|   |-- metrics/                            # CSV results
|   |   |-- model_metrics.csv
|   |   |-- clustering_comparison.csv
|   |   |-- simulation_results.csv
|   |   |-- threshold_sweep.csv
|   |-- figures/                            # Generated plots (PNG)
|       |-- 01_threshold_sweep.png
|       |-- 02_cumulative_energy.png
|       |-- 03_communication_events.png
|       |-- 04_retraining_events.png
|       |-- 05_mismatch_over_time.png
|       |-- 06_cluster_visualization.png
|       |-- 07_clustering_comparison.png
|       |-- 08_staleness_distribution.png
|       |-- 09_energy_comm_summary.png
|       |-- 10_adaptive_threshold.png
|
|-- docs/                                   # Documentation
|   |-- Thesis_report.tex                   # Full LaTeX thesis document
|   |-- Presentation_ppt.tex               # LaTeX presentation
|   |-- references.bib                      # Bibliography
|
|-- requirements.txt                        # Python dependencies
|-- .gitignore
|-- .gitattributes
```

### Files Summary Table

| Category | File | Lines | Purpose |
|----------|------|-------|---------|
| Config | config.py | 118 | Central configuration |
| Data | data_loader.py | 196 | Loading & feature engineering |
| Clustering | spatio_temporal_clustering.py | 231 | K-Means spatial vs spatio-temporal |
| Clustering | cluster_head_selection.py | 191 | Multi-criteria CH selection |
| Model | tinyml_model.py | 164 | Decision Tree training |
| Energy | energy_model.py | 135 | Heinzelman radio model |
| Simulation | node_simulation.py | 235 | Core simulation engine |
| Evaluation | evaluation.py | 181 | Metrics computation |
| Threshold | threshold_analysis.py | 186 | Sweep & adaptive policies |
| Pipeline | run_experiment.py | 137 | Master pipeline |
| Plots | visualization.py | 334 | Plot generation |

---

## 7. Summary of Key Results

| Metric | Value | Meaning |
|--------|-------|---------|
| Communication Reduction | ~67% | 2 out of 3 transmissions eliminated |
| Mismatch Rate | ~33% | Model wrong by >1 degrees C in 33% of readings |
| Energy Savings | Significant % over baseline | Measured in mJ over simulation |
| MAE | ~3.3 degrees C | Average prediction error |
| R^2 | ~0.89 | Model explains 89% of temperature variance |
| Spatio-Temporal Silhouette | ~0.42 vs 0.21 spatial-only | Nearly 2x better cluster quality |
| Retraining Overhead | 365x energy cost for 0.4% accuracy gain | Aggressive retraining is inefficient |
| Model Size | ~48 KB | Fits on microcontroller (128 KB flash) |
| Inference Time | <1 ms | Real-time capable on resource-constrained MCU |

### Key Insights

1. **Spatio-temporal clustering outperforms spatial-only** clustering by nearly 2x on Silhouette Score, validating the use of temporal behavioral features.
2. **Decision Tree is the optimal TinyML model** -- it balances accuracy (R^2 ~ 0.89), size (48 KB), and speed (<1 ms) for microcontroller deployment.
3. **Aggressive per-mismatch retraining is energy-inefficient** -- it costs 365x more energy for only 0.4% accuracy improvement. Batch or periodic retraining is preferable.
4. **The adaptive threshold creates a self-regulating system** that automatically adjusts suppression aggressiveness based on model performance.
5. **Cyclical encoding of temporal features** improves prediction accuracy by ~11% compared to raw integer timestamps.

---

## Python Dependencies

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
scipy>=1.11
```

---

*This document provides a complete reference for all implementations, mathematical foundations, and design decisions in the thesis project.*
