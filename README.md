# EPL Probabilistic Match Forecasting (H / D / A)

A robust probabilistic forecasting system for predicting **EPL full-time results**: **Home Win (H), Draw (D), Away Win (A)**.  
The framework’s core strength is a **Stacked Ensemble** that integrates three complementary modeling paradigms, with **strict temporal validation** and **probability calibration** to optimize **Log-loss** and improve reliability (especially for **Draw**).

---

## 🌟 Project Overview

This project develops a highly robust probabilistic forecasting system for predicting the full-time results of EPL matches (Home Win, Draw, Away Win).

### Core Design Principles

- **Multi-Model Integration**  
  Combines **XGBoost**, **Dixon-Coles**, and a **Transformer** to capture:
  - feature interactions (structured features)
  - goal-scoring dynamics (Poisson-based modeling)
  - temporal momentum (sequence learning)

- **Temporal Integrity**  
  Uses strict **Time Series Cross-Validation (TimeSeriesSplit)** and rolling/back-tested feature engineering to avoid look-ahead bias.

- **Probabilistic Calibration (Isotonic Calibration)**  
  Focuses on optimizing **Log-loss**, improving probability reliability—particularly for hard-to-predict draw outcomes.

---

## ⚙️ Methodology & Architecture

The pipeline is a structured multi-stage process where all modeling decisions respect temporal ordering.

### 1) Feature Engineering & Data Processing

- **130+ engineered features** across **13 categories**
- Features computed via **rolling window** so they are available before kick-off.
- Key techniques:
  - **Dynamic Strength**: Elo-style ratings for attack/defense, updated sequentially
  - **Short-Term Form**: rolling stats over last **L** matches (goals, win rates, etc.)
  - **Time Decay Weighting**:  
    \[
    w_t = \exp(-\alpha \Delta t)
    \]
    with half-life approximately \(\tau \approx 308\) days to emphasize recent data

### 2) Base Models

| Model | Paradigm | Core Functionality |
|------|----------|--------------------|
| XGBoost Classifier | Gradient Boosting Trees | Captures complex non-linear interactions across structured features (Elo, H2H, referee stats, etc.) |
| Bayesian Dixon-Coles | Statistical Goal Modeling | Calibrated probabilities via Poisson goal process + low-score dependence correction |
| Transformer Sequence Encoder | Self-Attention | Learns momentum/short-term form from fixed-length past-match sequences |

### 3) Stacked Ensemble (Meta-Model)

Base-model **Out-Of-Fold (OOF)** predicted probabilities are fed into a **Two-Stage Logistic Regression Fusion**:

- **Stage 1 (Draw vs. No-Draw)**  
  Predicts **D** vs **H/A**, incorporating draw-fusion features (e.g., competitive closeness indicators) to boost draw signal.

- **Stage 2 (Home vs. Away | Non-Draw)**  
  Predicts conditional **A vs H** given a non-draw outcome.

Final probability synthesis includes:
- global draw proportion scaling \(\alpha\)
- temperature scaling \(\tau\)
- **Isotonic Calibration** for best Log-loss

---

## 📊 Validation Results

Performance on held-out validation set:

| Model | Accuracy (%) | Log-loss | RPS |
|------|--------------|----------|-----|
| XGBoost | 54.2 | 0.186 | 0.184 |
| Dixon-Coles | 52.8 | 0.191 | 0.188 |
| Transformer | 53.0 | 0.188 | 0.186 |
| **Stacked Ensemble** | **55.1** | **0.179** | **0.176** |

---

## 🛠️ Environment Setup & Dependencies

### Reference Compute Environment

| Spec | Detail |
|------|--------|
| CPU | Intel i7-14650HX or equivalent |
| Memory | 32 GB RAM |
| Language | Python 3.13.10 |
| Core Frameworks | PyTorch 2.9.1, XGBoost 3.1.2 |

### Install Dependencies

Recommended: use a virtual environment (**Conda** or **venv**).

```bash
# Core Data Processing and Numerical Operations
pip install pandas numpy joblib tqdm

# Machine Learning, Optimization, and Calibration
pip install scikit-learn scipy scikit-optimize

# Gradient Boosting Library
pip install xgboost

# Deep Learning (Transformer) - CPU Version
pip install torch==2.9.1+cpu

# Statistical Modeling
pip install statsmodels

# Optional: Plotting and Visualization
pip install matplotlib seaborn
