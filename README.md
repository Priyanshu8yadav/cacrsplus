# 🛡️ CA-CRS+: Real-Time Causal Crowd-Risk Intelligence

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.23691416-blue.svg)](https://doi.org/10.5281/zenodo.23691416)
[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.23691416.svg)](https://doi.org/10.5281/zenodo.23691416)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io)

> **A Zone-Aware, Resource-Informed Crowd Risk Scoring Framework with Cause-Specific Gate Intervention Recommendation for Stampede Prevention**

**Authors:** [Priyanshu Kumar Yadav](mailto:priyanshu322006@gmail.com) & [Abhinav Kumar Mishra](mailto:mishrakabhinav34@gmail.com)  
*Department of Computer Science and Engineering, Rajkiya Engineering College, Mainpuri, Uttar Pradesh, India*  

**Permanent DOI:** [https://doi.org/10.5281/zenodo.23691416](https://doi.org/10.5281/zenodo.23691416) (`10.5281/zenodo.23691416`)  
**Zenodo Record:** [https://zenodo.org/records/23691416](https://zenodo.org/records/23691416)

---

## 📌 Project Notice & Provenance

This research was initially developed and submitted to the *IEEE Region 10 Conference (TENCON 2026)*. Following peer review, the research was comprehensively rebuilt and expanded end-to-end to address reviewer feedback:
- Whole-body detection heuristics were replaced by benchmark-validated density-map crowd counters (DM-Count VGG-19, APGCC).
- Directional conflict was reformulated as the magnitude-weighted circular variance of optical flow.
- Non-linear crush scoring with exponential penalty was benchmarked against matched linear models on dense imagery.
- All experiments were evaluated on public ground-truth benchmarks (**ShanghaiTech Parts A & B** and **PETS2009 S2.L1** multi-camera views).
- Formal 5-parameter sensitivity analysis and measured edge throughput were incorporated.

**Status:** The manuscript was voluntarily withdrawn from TENCON 2026 and is published here as an **open-access research preprint and reproducible artifact archive on Zenodo**:
- **Permanent DOI:** [https://doi.org/10.5281/zenodo.23691416](https://doi.org/10.5281/zenodo.23691416)
- **Zenodo Archive:** [https://zenodo.org/records/23691416](https://zenodo.org/records/23691416)
- **Research Paper PDF:** [`docs/CA-CRS-Plus-Research-Paper.pdf`](docs/CA-CRS-Plus-Research-Paper.pdf)

---

## 🚨 Motivation & The Four Core Gaps

Human stampedes at mass gatherings (e.g., Itaewon 2022, Hathras 2024, Maha Kumbh Mela 2025) remain a persistent, fatal category of disaster. Roughly 80% of Indian stampedes occur at religious gatherings that lack thermal cameras, cloud bandwidth, and specialized surveillance infrastructure.

Existing intelligent crowd-safety systems suffer from four systemic shortcomings:

| Limitation | Prior Systems | CA-CRS+ Framework |
|---|---|---|
| **(L1) Detection without Cause** | Issue one generic venue alert regardless of whether risk stems from density, panic speed, or opposing flows. | **Dominant-Factor Causal Attribution ($r_f$)** identifying whether Density, Speed, or Conflict drives danger. Refuses action when ambiguous (`MIXED → HOLD`). |
| **(L2) Global Alert without Space** | No localization of which physical corridor or plaza is critical. | **Venue-Graph Zone Model** treating each camera feed as a localized zone node with spatial capacities and adjacency. |
| **(L3) No Gate-Specific Action** | Do not specify which named physical gate to actuate. | **Cause-to-Gate Mapping with Ripple-Effect Prevention**: Density $\to$ `OPEN`, Speed $\to$ `CLOSE`, Conflict $\to$ `REDIRECT`. Downgrades `REDIRECT → HOLD` if downstream zone $CRS > 0.50$. |
| **(L4) No Resource Validation** | Alert operators without checking if sufficient responders exist. | **Dynamic Logarithmic Resource Sizing**: Scales security marshals as $D_{\text{mar}} = \lceil \alpha \log_{10}(1 + N) \rceil$, avoiding linear $3\times$ overstaffing. |

### Systematic Comparison with Existing Systems

| System | Multi-Zone | Causal Attribution | Gate Command | Resource Check | RGB-Only (No Thermal/Cloud) |
|---|:---:|:---:|:---:|:---:|:---:|
| Kim et al. (2024) [4] | ❌ | ❌ | ❌ | ❌ | ❌ (Thermal) |
| Ding et al. (2025) [6] | ❌ | ❌ | ❌ | ❌ | ✔️ |
| Mallick et al. (2024) [10] | ❌ | ❌ | ❌ | ❌ | ✔️ |
| SAFEST (2015) [9] | ✔️ | ❌ | ❌ | ❌ | ❌ (Infrared) |
| Cob-Parro et al. (2025) [5] | ❌ | ❌ | ❌ | ❌ | ✔️ |
| **CA-CRS+ (Ours)** | **✔️** | **✔️** | **✔️** | **✔️** | **✔️** |

---

## 🧠 The Mathematical Framework

```
                          ┌────────────────────────┐
                          │ Standard RGB CCTV Feed │
                          └───────────┬────────────┘
                                      │
            ┌─────────────────────────┴────────────────────────┐
            ▼                                                  ▼
┌───────────────────────┐                          ┌───────────────────────┐
│ Module 2: DM-Count    │                          │ Module 2: Farneback   │
│ Density Map (VGG-19)  │                          │ Dense Optical Flow    │
│  N = ∫ density_map    │                          │  Moving pixels M      │
│  D_norm = min(N/C, 1) │                          │  Speed S_norm (Eq. 1) │
└───────────┬───────────┘                          │  Conflict C_norm (2)  │
            │                                      └───────────┬───────────┘
            └─────────────────────────┬────────────────────────┘
                                      ▼
                        ┌───────────────────────────┐
                        │ Module 3: Non-Linear CRS  │
                        │ CRS = w1·D + Φ(D,S) + w3·C│
                        │ Φ = w2·S(1-D) + γ·e^λ(D-S)│
                        └─────────────┬─────────────┘
                                      │
            ┌─────────────────────────┴────────────────────────┐
            ▼                                                  ▼
┌───────────────────────────┐                      ┌───────────────────────────┐
│ Module 3: Causal Vector   │                      │ Module 7: Predictive Holt │
│  r_f = w_f·X_f / Σ w_j·X_j│                      │  Linear-Trend Forecaster  │
│  Dominant cause > 0.40    │                      │  Time-to-DANGER countdown │
└───────────┬───────────────┘                      └───────────────────────────┘
            ▼
┌───────────────────────────┐
│ Module 4: Cause-to-Gate   │
│  Density  → OPEN          │
│  Speed    → CLOSE         │
│  Conflict → REDIRECT      │
│  (Ripple Check: downstream│
│   CRS > 0.50 → HOLD)      │
└───────────┬───────────────┘
            ▼
┌───────────────────────────┐      ┌───────────────────────────┐
│ Module 5: Log Personnel   │      │ Module 6: Urgency GRS     │
│ D_mar = ⌈α·log10(1+N)⌉    │ ───► │ GRS = Σ ω_k·CRS_k / Σ ω_k │
│ α ∈ {0, 4, 10}            │      │ ω_k = 2 if DANGER else 1  │
└───────────────────────────┘      └───────────────────────────┘
```

### 1. Motion & Directional Conflict (Module 2)
- **Resolution-Independent Speed ($S_{\text{norm}}$):**
  $$S_{\text{norm}} = \min\left(\frac{1}{v_{\max}} \cdot \frac{1}{|M|} \sum_{i \in M} \frac{\|f_i\|}{W \Delta t},\; 1\right)$$
- **Circular-Variance Conflict Index ($C_{\text{norm}}$):**
  Unlike naive "fraction opposing dominant direction" metrics that collapse under balanced 50/50 counter-flow, $C_{\text{norm}}$ uses the magnitude-weighted circular variance of flow vectors:
  $$C_{\text{norm}} = 1 - \frac{\left\|\sum_{i \in M} m_i e^{j \theta_i}\right\|}{\sum_{i \in M} m_i}$$
  *Aligned flow $\to C_{\text{norm}} \to 0$; Balanced counter-flow $\to C_{\text{norm}} \to 1$.*

### 2. Non-Linear Crush Scoring & Gridlock Paradox (Module 3)
When crowd density reaches saturation and velocity decays toward zero, linear models paradoxically register a **decrease** in risk. CA-CRS+ resolves this via an exponential crush penalty:
$$\text{CA-CRS+}^{(k)} = \text{clip}_{[0,1]}\left(w_1 D_{\text{norm}}^{(k)} + \Phi(D_{\text{norm}}^{(k)}, S_{\text{norm}}) + w_3 C_{\text{norm}}\right)$$
$$\Phi(D, S) = w_2 S(1 - D) + \gamma e^{\lambda (D - S)}$$
*Calibrated parameters: $w_1 = 0.40, w_2 = 0.30, w_3 = 0.30, \gamma = 0.05, \lambda = 4.0$.*

### 3. Cause-Specific Gate Actuation & Ripple Prevention (Module 4)
| Dominant Cause | Primary Gate Command | Secondary Gate | Rationale |
|---|:---:|:---:|---|
| **Density** | `OPEN` (High) | `OPEN` (Med) | Relieve accumulation & increase outflow |
| **Speed** | `CLOSE` (High) | `HOLD` (Low) | Throttle rapid inflow & prevent surge |
| **Conflict** | `REDIRECT` (High) | `OPEN` (Med) | Segregate opposing pedestrian streams |
| **Mixed** | `HOLD` | `HOLD` | Refuse hazardous action; escalate to operator |

**Ripple-Effect Prevention:** Before issuing `REDIRECT`, the system evaluates the receiving downstream zone $Z_g^+$. If $\text{CRS}(Z_g^+) > 0.50$, the command is downgraded to `HOLD`, preventing pressure relief in one zone from triggering a secondary crush elsewhere.

### 4. Dynamic Logarithmic Staffing (Module 5)
Personnel demand scales logarithmically with zone population $N_k$, avoiding overestimation:
$$D_{\text{mar}}^{(k)} = \left\lceil \alpha_{l_k} \cdot \log_{10}(1 + N_k) \right\rceil, \quad \alpha \in \{0, 4, 10\} \text{ for SAFE / WARNING / DANGER}$$

### 5. Multi-Camera Urgency-Weighted GRS (Module 6)
Aggregates distributed zone risk scalars into a venue-wide Global Risk Score with a 2:1 urgency weighting:
$$\text{GRS} = \frac{\sum_k \omega_k \cdot \text{CA-CRS+}^{(k)}}{\sum_k \omega_k}, \quad \omega_k = \mathbb{I}\left[\text{Risk}^{(k)} = \text{DANGER}\right] + 1$$

---

## 📊 Key Benchmark Results

### 1. Counting Accuracy vs. Ground Truth (Table IV)
Evaluated on public ground-truth benchmarks against state-of-the-art counters:
- **ShanghaiTech Part A (Dense):** APGCC attains **MAE 48.7 / RMSE 76.7** (close to published SOTA).
- **ShanghaiTech Part B (Moderate):** DM-Count attains **MAE 14.4 / RMSE 27.7**.
- **PETS2009 S2.L1 (Tracking GT):** Density regression attains **MAE 0.55** against multi-object tracking ground truth.

### 2. Density Pathway: Non-Linear vs. Linear Baseline (Table V)
| Safe Capacity | % DANGER (CA-CRS+) | % DANGER (Linear Baseline) | Dominant Cause | Gate Action |
|:---:|:---:|:---:|:---:|:---:|
| 300 | **84%** | 0% | 100% Density | 100% OPEN |
| 500 | **55%** | 0% | 100% Density | 100% OPEN |
| 800 | **31%** | 0% | 100% Density | 100% OPEN |
| 1200 | **16%** | 0% | 100% Density | 100% OPEN |
| 2000 | **8%** | 0% | 100% Density | 100% OPEN |

*Crucial finding: The weight-matched linear baseline flags 0% DANGER at every capacity because its density term saturates at $w_1 = 0.40 < 0.70$. CA-CRS+ detects high-density crush states that linear models miss completely.*

### 3. Five-Parameter Sensitivity Analysis (Table VI)
| Parameter | Tested Range | % DANGER Range | Swing (pts) | Effect |
|:---:|:---:|:---:|:---:|:---:|
| $w_1$ | 0.20 – 0.60 | 49 – 59 | 10 | Moderate |
| $w_2$ | 0.10 – 0.50 | 55 – 55 | 0 | None (Robust) |
| $w_3$ | 0.10 – 0.50 | 55 – 55 | 0 | None (Robust) |
| $\gamma$ | 0.01 – 0.20 | 42 – 75 | **34** | Strong |
| $\lambda$ | 1.0 – 8.0 | 30 – 78 | **48** | Strong |

*Classification is robust to linear weights ($w_2, w_3$ swing 0 pts) and sensitive only to exponential crush parameters $(\gamma, \lambda)$, motivating their empirical calibration.*

### 4. Edge Throughput on Commodity GPU (Table VII)
Measured end-to-end latency on a single **NVIDIA Tesla T4 GPU**:
- Counting (DM-Count): `0.096 s/frame` (10.5 FPS)
- Motion (Farneback flow): `0.155 s/frame` (6.5 FPS)
- Scoring, Attribution, Gate Mapping, GRS: `< 0.001 s/frame` (> 25,000 FPS)
- **Total Pipeline:** `0.250 s/frame` = **4.0 FPS** (supporting a 1–2 Hz risk update cadence appropriate for human crowd dynamics).

---

## 🚀 Quickstart & Reproduction

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/Priyanshu8yadav/cacrsplus.git
cd cacrsplus

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Live Interactive Dashboard
```bash
streamlit run dashboard/app.py
```
- Real-time multi-zone CCTV / video stream monitoring.
- Density heatmap overlays, circular-variance optical flow vectors.
- Live causal attribution radar and Holt linear-trend time-to-DANGER countdown.
- Modbus TCP virtual PLC gate command actuation.

### 3. Run Standalone Benchmark Reproductions (No GPU Required)
Every key claim from the paper can be reproduced immediately with standalone scripts:

```bash
# Table V: Non-linear crush scoring vs. linear baseline
python3 experiments/reproduce_table5_linear_vs_nonlinear.py

# Fig. 1 & Section IV-D: Circular-variance conflict & ripple-effect downgrade
python3 experiments/reproduce_fig1_conflict_redirect.py

# Fig. 2 & Section IV-E: Multi-camera view fusion & GRS orchestration
python3 experiments/reproduce_fig2_multiview_fusion.py

# Table VI: 5-parameter sensitivity sweep
python3 experiments/reproduce_table6_parameter_sensitivity.py

# Verify live re-fit of occlusion calibration κ(ρ)
python3 verify_kappa.py

# Run test suite
python3 tests/test_engine_upgrades.py
```

---

## 📁 Repository Structure

```
cacrsplus/
├── dashboard/                  # Live Streamlit application & core engine
│   ├── app.py                  # Dashboard UI, MJPEG streaming, multi-zone layout
│   ├── ca_crs_engine.py        # Modules M1–M6: Non-linear scoring, causes, gates, GRS
│   ├── dmcount_counter.py      # VGG-19 density-map regression & heatmap blender
│   ├── dmcount_model.py        # PyTorch DM-Count network architecture
│   ├── forecasting.py          # Module 7: Holt linear-trend time-to-DANGER forecaster
│   ├── responders.py           # Virtual PLC Modbus TCP gate actuation
│   ├── accessibility.py        # HUD high-contrast rendering & WCAG compliance
│   └── threaded_stream.py      # Non-blocking multi-camera video reader
├── experiments/                # Paper reproduction scripts
│   ├── reproduce_table5_linear_vs_nonlinear.py
│   ├── reproduce_table6_parameter_sensitivity.py
│   ├── reproduce_fig1_conflict_redirect.py
│   └── reproduce_fig2_multiview_fusion.py
├── calibration/                # κ(ρ) occlusion calibration tools (SAHI + head detection)
├── sensitivity/                # Sobol parameter sensitivity & PETS2009 trajectory validation
├── parsers/                    # Dataset loaders (ShanghaiTech, UCF-QNRF, PETS2009)
├── outputs/                    # Curated benchmark plots, fitted JSON parameters
│   ├── calibration/            # κ fitted parameters & calibration curves
│   ├── sensitivity/            # Sobol heatmaps, parameter sweeps, classification CSV
│   └── tables/                 # Benchmark summary JSONs
├── tests/                      # Synthetic engine & forecasting unit tests
├── models/                     # Weights directory (see models/README.md)
├── config.yaml                 # Central physics, thresholds & parameter configuration
├── verify_kappa.py             # Live re-fit of κ(ρ) from raw data
├── requirements.txt            # Python dependencies
├── .zenodo.json                # Zenodo archive metadata configuration
├── CITATION.cff                # Citation File Format 1.2.0 metadata
├── ZENODO_DEPOSIT_GUIDE.md     # Step-by-step instructions for uploading to Zenodo
└── LICENSE                     # MIT Open Source License
```

---

## 📖 Citation

If you use this codebase, framework, or benchmark artifacts in your research, please cite:

- **Permanent DOI:** [https://doi.org/10.5281/zenodo.23691416](https://doi.org/10.5281/zenodo.23691416)
- **Zenodo Record:** [https://zenodo.org/records/23691416](https://zenodo.org/records/23691416)

```bibtex
@misc{yadav2026cacrsplus,
  title={{CA-CRS+}: A Zone-Aware, Resource-Informed Crowd Risk Scoring Framework with Cause-Specific Gate Intervention Recommendation for Stampede Prevention},
  author={Yadav, Priyanshu Kumar and Mishra, Abhinav Kumar},
  year={2026},
  howpublished={Preprint and Reproducible Research Archive, Zenodo},
  doi={10.5281/zenodo.23691416},
  url={https://doi.org/10.5281/zenodo.23691416}
}
```

---

## 📄 License
This project is released under the **[MIT License](LICENSE)**. Benchmark documentation and preprints are released under **Creative Commons Attribution 4.0 International (CC-BY-4.0)**.
