# Customer Lifetime Value Prediction | Analytics Vidhya Data Scientist Hiring Hackathon

> **Rank: #62 / 6800+ participants | Top 1%**

## Problem Statement

Predict the **Customer Lifetime Value (CLTV)** for customers of an e-commerce/retail business based on their historical transaction and behavioral data.evaluated on ***RMSE.***

---

## Approach

### 1. Exploratory Data Analysis
- Target distribution analysis (heavy right skew → log transformation)
- Correlation heatmaps and outlier profiling
- Segment-level aggregation to understand high-value vs. low-value customer patterns

### 2. Feature Engineering
- **RFM features**: Recency, Frequency, Monetary value per customer
- **Behavioral signals**: Purchase intervals, category diversity, return rates
- **Temporal features**: Day-of-week, month, seasonality flags
- **Aggregated statistics**: Mean, std, max transaction values per customer
- Interaction features between high-importance variables

### 3. Modeling
Trained a **weighted Out-of-Fold (OOF) ensemble** of three gradient boosting models:

| Model | Role |
|---|---|
| LightGBM | Fast training, handles high cardinality |
| XGBoost | Strong baseline, tree depth tuning |
| CatBoost | Native categorical handling, robust to outliers |

- 5-fold Stratified K-Fold cross-validation
- Weights optimized by minimizing OOF RMSE
- Final prediction: weighted average of all three models

### 4. Post-processing
- Inverse log-transform on predictions
- Clipping negative predictions to zero
- Threshold analysis on validation fold

## Tech Stack

- **Python 3.10**
- `lightgbm`, `xgboost`, `catboost`
- `scikit-learn`, `pandas`, `numpy`
- `matplotlib`, `seaborn`

> **Note:** Raw competition data is not included due to Analytics Vidhya's data usage policy. Download it from the [competition page](https://www.analyticsvidhya.com/datahack/contest/data-scientist-hiring-hackathon/).
---

## Key Takeaways

- Log-transforming a heavily skewed CLTV target significantly improved model performance
- Weighted OOF ensemble consistently outperformed any single model
- RFM-based features were the strongest predictors of lifetime value

---

## Connect

**Satyam Sharma**  
[LinkedIn](https://linkedin.com/in/sharmasatyam01) | [GitHub](https://github.com/baneXP)
