
# SEM-ANN Hybrid Analysis: TQM & Lean Manufacturing Impact on Sustainability Performance
### Saudi Food Manufacturing Sector | Python Simulation Study

## Overview

This project implements a full **Structural Equation Modeling (SEM) + Artificial Neural Network (ANN)** hybrid analysis pipeline to investigate how **Total Quality Management (TQM)** and **Lean Manufacturing** practices influence **Sustainability Performance (SP)** across Environmental, Social, and Economic dimensions — contextualized within the Saudi Arabian food manufacturing sector.

The entire workflow is self-contained in a single Python script: from synthetic survey data generation to statistical validation, path estimation, deep-learning prediction, and strategic managerial insight maps.

---

## 🔬 Methodology

| Stage | Technique | Output |
|---|---|---|
| Survey Simulation | 5-point Likert scale (n=350) | `survey_data.csv` |
| Reliability & Validity | Cronbach's α, CR, AVE | `cfa_validity.csv` |
| Structural Modeling | OLS-based SEM path estimation | `sem_path_results.csv` |
| Predictive Modeling | MLP Neural Network (64→32→16) | `ann_importance.csv` |
| Strategic Analysis | IPMA (Importance-Performance Map) | Figure visualization |
| Demographics | Firm size, region, years of operation | Figure visualization |

---

## 📊 Figures Generated (8 total)

1. **Pearson Correlation Matrix** – Latent construct correlations
2. **CFA Bar Chart** – Cronbach's α, Composite Reliability, AVE per construct
3. **SEM Beta Coefficients** – Path strengths (TQM/LEAN → ENV/SOC/ECO → SP)
4. **SEM Path Diagram** – Visual structural model with beta annotations
5. **ANN Performance** – Training loss curve + Actual vs. Predicted scatter
6. **Variable Importance** – Permutation importance (%) for each predictor
7. **IPMA Map** – Importance-Performance quadrant chart for prioritization
8. **Demographics** – Firm size distribution, regional spread, SP by firm size

---

## 🛠️ Tech Stack

- **Python 3.x**
- `numpy`, `pandas` — data generation & manipulation
- `scikit-learn` — MLPRegressor, StandardScaler, permutation importance
- `matplotlib`, `seaborn` — all visualizations (TkAgg interactive backend)

---

## 🚀 Usage

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
python sem_ann_analysis_windows.py
```

All 8 figures will pop up interactively on screen and are saved as `.png` files alongside 4 `.csv` result files in the script's directory.

---


