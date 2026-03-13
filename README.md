# MSARNet: A Privacy-Preserving Multimodal Deep Tabular Network for Federated Physiological Fatigue Regression

**Aadarsh Ramakrishna ·Senthil Prakash P.N · Beneta Johnson · Tarun Parthiban **  
School of Computer Science and Engineering, Vellore Institute of Technology, Vellore, India

---

## Overview

MSARNet is a deep tabular architecture for simultaneous regression of **physical and mental fatigue scores** from multimodal wearable physiological signals. The model is designed to operate under **federated learning**, enabling privacy-preserving collaborative training across clients without sharing raw physiological data.

Five sensor modalities — heart rate (HR), inter-beat interval (IBI), accelerometer (ACC), electrodermal activity (EDA), and skin temperature (TEMP) — are reduced to a 14-dimensional tabular window representation, then processed through four parallel pathways (cross-feature attention, feature interaction + squeeze-excitation, self-attention, deep MLP), fused by adaptive gating, and decoded into joint fatigue predictions.

**Best result:** MSARNet + FedAdam → **RMSE = 0.2293, MAE = 0.2135** — 5.9% lower RMSE than the next-best federated model (TabNet + FedAvg: RMSE = 0.2436) and better than all centralized baselines including the strongest centralized model (TabNet, RMSE = 0.2332).

---

## Repository Structure

```
.
├── Dataset_PreProcess.ipynb
│     Raw FatigueSet ingestion: sensor file parsing, adaptive resampling,
│     30-second windowing (15-second step), mean/std feature extraction,
│     fatigue score interpolation → final_feature_label_dataset_normalized_interpolated.xlsx
│
├── SPTBA_Centralized_WeightedMetrics.ipynb
│     Centralized training for all 8 models (25 epochs each).
│     Evaluation uses evaluate_weighted() — per-client RMSE/MAE aggregated
│     by sample weight, identical formula to FL weighted_average().
│     Outputs → All_Centralized_Combined.csv
│
├── FL_SPTBA1_Performed_Final.ipynb
│     Federated training for all 8 models × 3 strategies (FedAvg, FedProx, FedAdam).
│     12 clients, 25 rounds, 1 local epoch per round, fraction_fit = 1.0.
│     Outputs → Combined_FL_Results_Final.csv
│
├── final_feature_label_dataset_normalized_interpolated.xlsx
│   │  2,819 windows × 19 columns (14 features + 2 labels + person/session/window_start)
├── All_Centralized_Combined.csv
│   │  Columns: round, model, rmse, mae  [8 models × 25 epochs = 200 rows]
└── Combined_FL_Results_Final.csv
│      Columns: algorithm, model, round, rmse, mae, r2  [3 × 8 × 25 = 600 rows]
│
└── README.md
```

---

## Dataset

**Source:** [FatigueSet](https://www.kaggle.com/datasets/tanjemahamed/mental-fatigue-level-detection-fatigueset-data) — publicly available multimodal physiological fatigue benchmark.

### Statistics

| Property | Value |
|---|---|
| Participants | 12 |
| Sessions per participant | Up to 3 |
| Physiological signals | HR (`wrist_hr.csv`), IBI (`wrist_ibi.csv`), ACC (`wrist_acc.csv`), EDA (`wrist_eda.csv`), TEMP (`wrist_skin_temperature.csv`) |
| Window size | 30 seconds |
| Step size | 15 seconds (50% overlap) |
| Features per window | 14 — mean and std of each of the 7 signal channels (hr, duration, ax, ay, az, eda, temp) |
| Total windows | **2,819** |
| Target variables | `physicalFatigueScore` ∈ [0, 1], `mentalFatigueScore` ∈ [0, 1] |

### Per-Participant Window Counts

| Person | Total | Train (80%) | Test (20%) |
|---|---|---|---|
| 1 | 277 | 221 | 56 |
| 2 | 181 | 144 | 37 |
| 3 | 223 | 178 | 45 |
| 4 | 263 | 210 | 53 |
| 5 | 255 | 204 | 51 |
| 6 | 221 | 176 | 45 |
| 7 | 257 | 205 | 52 |
| 8 | 200 | 160 | 40 |
| 9 | 258 | 206 | 52 |
| 10 | 196 | 156 | 40 |
| 11 | 248 | 198 | 50 |
| 12 | 240 | 192 | 48 |
| **Total** | **2,819** | **2,250** | **569** |

### Preprocessing Pipeline (`Dataset_PreProcess.ipynb`)

1. **Sensor file loading** — each of the five wrist sensor CSV files is read per participant per session, timestamps parsed as milliseconds-since-epoch
2. **Adaptive resampling** — sampling interval determined per session as `max(min_interval across sensors)`; all channels resampled to this common grid via `resample().mean().interpolate()`
3. **Windowing** — 30-second sliding windows with 15-second step; windows spanning session boundaries are discarded
4. **Feature extraction** — mean and std computed per channel per window → 14 features total
5. **Target interpolation** — `exp_fatigue.csv` questionnaire scores linearly interpolated between measurement points to assign a continuous score to every window
6. **Normalization** — `StandardScaler` fitted exclusively on each participant's training portion (first 80% of windows in chronological order); test windows normalized using the same participant-specific parameters; scaler is never fit on test data

---

## Experimental Setup

### Evaluation Protocol

All RMSE and MAE values — centralized and federated — use the same **client-weighted aggregation formula**:

$$\text{RMSE}_{\text{weighted}} = \sum_{k=1}^{12} \frac{n_k}{N} \cdot \text{RMSE}_k \qquad \text{MAE}_{\text{weighted}} = \sum_{k=1}^{12} \frac{n_k}{N} \cdot \text{MAE}_k$$

- **Federated** — implemented as Flower's `weighted_average` callback applied at the server after each round
- **Centralized** — implemented as `evaluate_weighted()` in `SPTBA_Centralized_WeightedMetrics.ipynb`, called after each epoch; loops over all 12 clients, computes per-client RMSE/MAE, applies the same formula


### Training Configuration

| Parameter | Centralized | Federated |
|---|---|---|
| Optimizer | Adam, lr = 1e-3 | Adam, lr = 1e-3 (local) |
| Loss | MSELoss | MSELoss |
| Batch size | 32 | 32 |
| Epochs / Rounds | 25 | 25 |
| Local epochs per round | — | 1 |
| Clients per round | — | 12 / 12 (fraction_fit = 1.0) |
| Seed | 42 | 42 |

### Federated Strategy Parameters

| Strategy | Key parameters |
|---|---|
| **FedAvg** | Weighted parameter averaging; no proximal term |
| **FedProx** | Proximal term µ = 0.01 added to local loss |
| **FedAdam** | Server-side: η = 0.01, β₁ = 0.9, β₂ = 0.99, τ = 1e-9 |

> **Note:** TabNet under FedAdam uses slightly different server parameters (β₂ = 0.999, τ = 1e-8) . All other models use the values above.

---

## Model Architectures

### Baseline Models

**Linear** — `Linear(14 → 2)`, no nonlinearity. Lower bound of performance.

**MLP** — `Linear(14→64) → ReLU → Linear(64→64) → ReLU → Linear(64→2)`. Two hidden layers.

**Residual Network** — Three-layer MLP with additive skip connections:
```
h1 = ReLU(fc1(x))              fc1: Linear(14→64)
h2 = ReLU(fc2(h1)) + h1        fc2: Linear(64→64)
h3 = ReLU(fc3(h2)) + h2        fc3: Linear(64→64)
out = Linear(64→2)(h3)
```

**Dual-Head Network** — Shared encoder with task-specific output heads:
```
Shared:   Linear(14→64) → ReLU → Linear(64→64) → ReLU
Physical: Linear(64→1)
Mental:   Linear(64→1)
Output:   torch.cat([physical, mental], dim=1)   →  (batch, 2)
```

### Benchmark Tabular Architectures

---

**TabNet** (adapted from Arik & Pfister, AAAI 2021)

The original TabNet uses sparsemax sequential attention masks, GLU activations, Ghost BatchNorm, and sparsity regularization. This adaptation makes three simplifications for small-scale regression:

1. Sparsemax attention replaced with `nn.Sigmoid()` — produces per-dimension soft weights in (0, 1)
2. GLU replaced with GELU
3. Ghost BatchNorm replaced with LayerNorm; scale accumulation and sparsity regularization removed

```
Input → Linear(14→64) → LayerNorm → GELU → TabNetBlock × 2 → Linear(64→2)

TabNetBlock(dim=64):
  (i)  Feature transform:   Linear(64→128) → GELU → Linear(128→64)
  (ii) Sigmoid gate:        Linear(64→64) → Sigmoid  [applied to transformed output]
  (iii) Gated output:       sigmoid_weights * transformed_h
  (iv) Residual + LN:       LayerNorm(gated + x)
```

---

**AutoInt** (adapted from Song et al., CIKM 2019)

A single shared linear projection maps all 14 features jointly to a 14×64 token matrix (replacing the per-feature embedding tables of the original). Three stacked 4-head self-attention layers follow, each with learned linear residual projections (not identity skips) and LayerNorm.

```
Linear(14 → 896) → reshape(batch, 14, 64)
→ [ 4-Head MHA(64, dropout=0.1) → Linear(64→64) residual + LayerNorm ] × 3
→ Flatten(896)
→ Linear(896→128) → ReLU → Dropout(0.1)
→ Linear(128→64)  → ReLU → Dropout(0.1)
→ Linear(64→2)

Verified parameter count: 199,426  (~2.0 × 10⁵)
```

---

**DeepFM** (adapted from Guo et al., IJCAI 2017)

Three components computed in parallel over the raw 14-dimensional input (deep component consumes normalized features directly, not stacked embeddings as in the original):

```
Linear term:  Linear(14→1)
FM term:      feature_embeddings ∈ ℝ^{14×32};  FM = 0.5·(‖Σᵢ vᵢxᵢ‖² − Σᵢ ‖vᵢxᵢ‖²) → scalar
Deep term:    Linear(14→128) → BatchNorm → ReLU → Dropout(0.1)
              → Linear(128→64) → BatchNorm → ReLU → Dropout(0.1)

Output:       concat(linear[1], FM[1], deep[64]) = 66-dim → Linear(66→2)
```

---

**MSARNet — Proposed Architecture**

```
Stage 1 — Input Projection
  Linear(14→28) → LayerNorm → GELU          [output: x₀ = xproj ∈ ℝ²⁸]

Stage 2 — Pathway 1: Cross-Feature Attention
  14 × Linear(1→32)                          [14 independent per-feature embeddings]
  → Token Matrix (batch, 14, 32)
  → 4-Head MHA(32, dropout=0.0) → Residual + LayerNorm
  → Flatten(448) → Linear(448→28)            [p1 ∈ ℝ²⁸]

Stage 3 — Pathway 2A: Feature Interaction (stacked × 2)
  xout = x₀ · (x · w + b) + x               [learned w ∈ ℝ^{28×1}, b ∈ ℝ²⁸]

Stage 4 — Pathway 2B: Squeeze-Excitation
  Linear(28→7) → ReLU → Linear(7→28) → Sigmoid
  → element-wise × x                         [p2 ∈ ℝ²⁸, channel-wise recalibration]

Stage 5 — Pathway 3: Self-Attention
  xproj → unsqueeze(1)
  → 4-Head MHA(28, dropout=0.1) → squeeze
  → Residual + LayerNorm                     [p3 ∈ ℝ²⁸]

Stage 6 — Pathway 4: Deep MLP
  Linear(28→56) → LayerNorm → GELU → Dropout(0.15)
  → Linear(56→28) → LayerNorm → GELU        [p4 ∈ ℝ²⁸]

Stage 7 — Adaptive Gating
  Linear(28→14) → LayerNorm → GELU → Dropout(0.1) → Linear(14→4)
  → Softmax(logits / t)                      [t: learned temperature, init=1]
  combined = g₁·p1 + g₂·p2 + g₃·p3 + g₄·p4
  combined = combined + x₀                   [global residual anchor]

Stage 8 — Output Regression Network
  Linear(28→256) → LayerNorm → GELU → Dropout(0.15)
  → Linear(256→128) → LayerNorm → GELU → Dropout(0.10)
  → Linear(128→64) → GELU → Linear(64→2)

Output: (batch, 2) = [physicalFatigueScore, mentalFatigueScore]
```

---

## Results

All values are verified directly from `All_Centralized_Combined.csv` and `Combined_FL_Results_Final.csv`.

### Table 1 — Centralized Training, Epoch 25

| Rank | Model | RMSE ↓ | MAE ↓ | R² | Initial RMSE (E1) | Improvement |
|---|---|---|---|---|---|---|
| 1 | TabNet | **0.2332** | 0.1990 | −0.150 | 0.3316 | +29.7% |
| 2 | MSARNet | **0.2490** | 0.2162 | −0.310 | 0.2791 | +10.8% |
| 3 | AutoInt | 0.2717 | 0.2516 | −0.843 | 0.2430 | **−11.8%** |
| 4 | Residual | 0.2998 | 0.2503 | −1.286 | 0.4961 | +39.6% |
| 5 | MLP | 0.3033 | 0.2627 | −1.621 | 0.3827 | +20.8% |
| 6 | Linear | 0.3169 | 0.2849 | −2.254 | 1.2106 | +73.8% |
| 7 | Dual-Head | 0.3750 | 0.3234 | −3.609 | 0.5032 | +25.5% |
| 8 | DeepFM | 0.4029 | 0.3282 | −3.155 | 0.5186 | +22.3% |

AutoInt's −11.8% improvement indicates performance **degradation** from epoch 1 to 25 — consistent with over-parameterisation: 199,426 parameters trained on ~2,250 samples. All negative R² values reflect inter-individual physiological variability rather than model failure; no model produces a consistent cross-subject signal under centralized pooling.

### Table 2 — Best Federated (Round 25) vs. Centralized (Epoch 25)

| Model | CL RMSE | Best FL RMSE | Best Strategy | Δ% | FL wins? |
|---|---|---|---|---|---|
| MSARNet | 0.2490 | **0.2293** | FedAdam | −7.9% | ✅ |
| TabNet | 0.2332 | 0.2436 | FedAvg | +4.5% | ❌ |
| AutoInt | 0.2717 | 0.2595 | FedAdam | −4.5% | ✅ |
| Residual | 0.2998 | 0.3119 | FedAdam | +4.0% | ❌ |
| MLP | 0.3033 | 0.2899 | FedAdam | −4.4% | ✅ |
| Linear | 0.3169 | 0.3770 | FedAdam | +19.0% | ❌ |
| Dual-Head | 0.3750 | 0.3910 | FedAdam | +4.3% | ❌ |
| DeepFM | 0.4029 | 0.2875 | FedAvg | −28.7% | ✅ |

**4 of 8 architectures achieve lower RMSE under their best federated configuration.** The best federated result overall — MSARNet + FedAdam (0.2293) — is lower than every centralized result including the best centralized model (TabNet, 0.2332).

### Table 3 — FedAvg Convergence Trajectory

| Rank | Model | R1 | R5 | R10 | R25 | MAE | Trend |
|---|---|---|---|---|---|---|---|
| 1 | MSARNet | 0.2477 | 0.2223 | 0.2287 | 0.2360 | 0.2133 | Min at R3 (0.2218), gradual rise |
| 2 | TabNet | 0.4027 | 0.2601 | 0.2442 | 0.2436 | 0.2183 | Rapid early descent, plateau |
| 3 | AutoInt | 0.2352 | 0.2591 | 0.2639 | 0.2709 | 0.2479 | Monotonic divergence |
| 4 | DeepFM | 6.6721 | 2.7009 | 0.3778 | 0.2875 | 0.2374 | Slow recovery from instability |
| 5 | Dual-Head | 0.3237 | 0.3669 | 0.3911 | 0.3962 | 0.3480 | Divergence |
| 6 | MLP | 0.3884 | 0.4301 | 0.4354 | 0.4228 | 0.3751 | Gradual divergence |
| 7 | Linear | 0.8592 | 0.7820 | 0.6943 | 0.5026 | 0.3830 | Slow monotonic improvement |
| 8 | Residual | 0.6542 | 0.5999 | 0.5640 | 0.5167 | 0.4538 | Slow monotonic improvement |

### Table 4 — FedProx Results (µ = 0.01), Round 25

| Rank | Model | RMSE | MAE | vs. FedAvg R25 |
|---|---|---|---|---|
| 1 | MSARNet | **0.2496** | 0.2303 | +0.0136 worse — proximal suppresses per-client personalization |
| 2 | TabNet | 0.2628 | 0.2331 | +0.0192 worse — proximal constrains beneficial attention drift |
| 3 | AutoInt | 0.2720 | 0.2512 | −0.0017 better — proximal prevents divergence |
| 4 | Residual | 0.3724 | 0.3256 | −0.1443 better — significant improvement |
| 5 | MLP | 0.3911 | 0.3475 | −0.0317 better — proximal helps avoid drift |
| 6 | Dual-Head | 0.4044 | 0.3579 | +0.0082 worse — minimal impact |
| 7 | DeepFM | 0.5560 | 0.4280 | +0.2686 worse — proximal disrupts FM recovery |
| 8 | Linear | 0.6200 | 0.4919 | +0.1174 worse — proximal too restrictive for linear models |

### Table 5 — FedAdam Convergence Trajectory

| Rank | Model | R1 | R15 | R25 | MAE | Pattern |
|---|---|---|---|---|---|---|
| 1 | MSARNet | 2.5119 | 0.3499 | **0.2293** | 0.2135 | Slow start, strong late convergence |
| 2 | TabNet | 1.3045 | 0.2585 | 0.2464 | 0.2279 | Early instability, stabilizes |
| 3 | AutoInt | 765.508 | 0.2448 | 0.2595 | 0.2411 | Extreme early instability, rapid recovery |
| 4 | MLP | 0.3980 | 0.4064 | 0.2899 | 0.2529 | Consistent gradual improvement |
| 5 | Residual | 0.6870 | 0.2702 | 0.3119 | 0.2723 | Strong mid-run improvement |
| 6 | Linear | 1.3442 | 0.3585 | 0.3770 | 0.3216 | Recovers well from early instability |
| 7 | Dual-Head | 0.4226 | 0.3677 | 0.3910 | 0.3500 | Minor improvement |
| 8 | DeepFM | 13.241 | 83.013 | 86.515 | — | **Catastrophic divergence** |

DeepFM diverged under FedAdam due to structural incompatibility between FM gradient dynamics and server-side adaptive variance scaling. Excluded from FedAdam comparisons throughout the paper.

### Table 6 — MSARNet Across All Training Paradigms

| Paradigm | RMSE | MAE | vs. Centralized |
|---|---|---|---|
| Centralized | 0.2490 | 0.2162 | — |
| FedAvg (R25) | 0.2360 | 0.2133 | −5.2% |
| FedProx (R25) | 0.2496 | 0.2303 | +0.2% |
| **FedAdam (R25)** | **0.2293** | **0.2135** | **−7.9%** |

MSARNet FedAvg reaches its minimum at **Round 3 (RMSE = 0.2218)** before rising to 0.2360 by Round 25 — indicating early per-client adaptation that is gradually eroded by repeated global averaging without adaptive rates.

## Reproducibility

### Requirements

```bash
pip install torch flwr numpy pandas scikit-learn openpyxl
```

Tested with: Python 3.10, PyTorch 2.x, Flower (flwr) 1.x, run on Google Colab (CPU runtime).

### Execution Order

```
1. Dataset_PreProcess.ipynb
   → Produces: final_feature_label_dataset_normalized_interpolated.xlsx

2. SPTBA_Centralized_WeightedMetrics.ipynb
   → Input:  final_feature_label_dataset_normalized_interpolated.xlsx
   → Produces: All_Centralized_Combined.csv  (8 models × 25 epochs)

3. FL_SPTBA1_Performed_Final.ipynb
   → Input:  final_feature_label_dataset_normalized_interpolated.xlsx
   → Produces: Combined_FL_Results_Final.csv  (8 models × 3 strategies × 25 rounds)
```

### Key Implementation Notes

- Global random seed = 42 set in `seed_everything()` before all experiments (Python, NumPy, PyTorch, CUDA, CuDNN deterministic mode)
- CL training uses pooled global train loader (`shuffle=False`) but evaluates per-client — evaluation is not pooled
- FL simulation uses `fraction_fit=1.0` and `fraction_evaluate=1.0` — all 12 clients participate every round
- DeepFM is present in all FL result files under FedAdam but its Round 25 RMSE = 86.515 (diverged) — it is excluded from FedAdam analysis and comparisons throughout the paper
- MSARNet is named `MSARv3` in all CSV files and notebook code

---

## Citation

```bibtex
@article{ramakrishna2026msarnet,
  title     = {MSARNet: A Privacy-Preserving Multimodal Deep Tabular Network
               for Federated Physiological Fatigue Regression},
  author    = {Ramakrishna, Aadarsh and Prakash, Senthil P.N. and Johnson, Beneta and
               Parthiban, Tarun },
  institution = {Vellore Institute of Technology},
  year      = {2025}
}
```

---

[13] Rieke, N. et al.: The future of digital health with federated learning. *npj Digital Medicine* 3, 119 (2020)  
[14] Kairouz, P. et al.: Advances and open problems in federated learning. *Foundations and Trends in ML* 14(1–2), 1–210 (2021)  
[15] Hu, J., Shen, L., Sun, G.: Squeeze-and-excitation networks. *CVPR*, 7132–7141 (2018)
