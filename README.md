# Hydro Raw Material Receival Forecasting

Quantile loss optimization (q=0.2) for cumulative material receival prediction in manufacturing logistics

## Overview

Forecasts raw material receivals (cumulative weight) for Hydro's manufacturing facilities. Ensemble model (CatBoost 60% + LightGBM 40%) with 120+ engineered features (temporal, lag-based, aggregate statistics) and rolling cross-validation. Heavily penalizes overestimation (4:1 loss ratio).

**Best validation score**: ~10,800 Quantile Loss

## Data

```
data/
├── kernel/
│   ├── receivals.csv          # Historical receival records
│   └── purchase_orders.csv    # Purchase order data
├── extended/
│   ├── materials.csv          # Material metadata
│   └── transportation.csv     # Logistics attributes
└── prediction_mapping.csv     # Submission mapping
```

## Structure

- `notebooks/short_notebook_1.ipynb` — Baseline & EDA
- `notebooks/short_notebook_2.ipynb` — Ensemble & optimization
- `material_receival_visualization.ipynb` — Visualization
- `Report.pdf` — Technical report
- `submissions/submission_best.csv` — Best submission

## Quickstart

```bash
jupyter notebook notebooks/short_notebook_1.ipynb
jupyter notebook notebooks/short_notebook_2.ipynb
```

## Requirements

```
python>=3.9
pandas numpy scikit-learn
catboost lightgbm optuna
matplotlib seaborn
```

## References

- [Dataset_definitions_and_explanation.pdf](Dataset_definitions_and_explanation.pdf)
- [Machine_learning_task_for_TDT4173.pdf](Machine_learning_task_for_TDT4173.pdf)
- [kaggle_metric.ipynb](kaggle_metric.ipynb)

**Author**: Andrea Richichi — NTNU TDT4173

