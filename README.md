# Boston Housing Price Prediction - Comprehensive ML Project

## Project Overview
This project implements a complete machine learning system for predicting Boston housing prices, demonstrating advanced ML techniques including ensemble methods, neural networks, hyperparameter optimization, and model interpretation.

## Project Structure
```
Boston/
│
├── data/                       # Dataset storage
│   └── housing.csv
│
├── notebooks/                  # Jupyter notebooks for analysis
│   ├── 01_eda_univariate.ipynb
│   ├── 02_eda_multivariate.ipynb
│   ├── 03_preprocessing_pipeline.ipynb
│   ├── 04_baseline_models.ipynb
│   ├── 05_advanced_tree_models.ipynb
│   ├── 06_neural_networks.ipynb
│   ├── 07_hyperparameter_tuning.ipynb
│   ├── 08_ensemble_stacking.ipynb
│   ├── 09_evaluation_analysis.ipynb
│   └── 10_interpretation_explainability.ipynb
│
├── src/                        # Source code modules
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   └── interpretation.py
│
├── models/                     # Saved model artifacts
│
├── figures/                    # Generated visualizations
│
├── reports/                    # Final report and documentation
│   └── final_report.md
│
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

1. Clone or navigate to the project directory
2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Information

The Boston Housing dataset contains information about houses in Boston suburbs with the following features:

- **CRIM**: Per capita crime rate by town
- **ZN**: Proportion of residential land zoned for lots over 25,000 sq.ft
- **INDUS**: Proportion of non-retail business acres per town
- **CHAS**: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- **NOX**: Nitric oxides concentration (parts per 10 million)
- **RM**: Average number of rooms per dwelling
- **AGE**: Proportion of owner-occupied units built prior to 1940
- **DIS**: Weighted distances to five Boston employment centres
- **RAD**: Index of accessibility to radial highways
- **TAX**: Full-value property-tax rate per $10,000
- **PTRATIO**: Pupil-teacher ratio by town
- **B**: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
- **LSTAT**: % lower status of the population
- **MEDV**: Median value of owner-occupied homes in $1000's (TARGET)

## Project Workflow

### 1. Exploratory Data Analysis (EDA)
- Univariate analysis with distributions and statistical tests
- Bivariate analysis with correlation and scatter plots
- Multivariate analysis with PCA and clustering
- Missing value and outlier detection

### 2. Data Preprocessing Pipeline
- Train/validation/test split (70/15/15)
- Feature engineering and transformations
- Scaling and normalization
- Pipeline creation with scikit-learn

### 3. Baseline Models
- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree

### 4. Advanced Models
**Tree-based Ensembles:**
- XGBoost (with Optuna tuning helper)
- LightGBM
- CatBoost

**Neural Networks:**
- Deep MLP
- TabNet (implemented)

**Ensembling:**
- Stacking regressor blending Ridge + LightGBM + XGBoost

### 5. Model Optimization
- Hyperparameter tuning with Optuna
- Cross-validation
- Ensemble and stacking approaches

### 6. Evaluation & Analysis
- Multiple metrics (RMSE, MAE, R², MAPE)
- Learning curves
- Bias-variance analysis
- Statistical significance testing

### 7. Interpretation & Explainability
- SHAP values (global and local)
- Permutation importance
- Feature importance comparison
- Individual prediction explanations

## Results Summary

(To be filled after model training)

## Key Findings

(To be filled after analysis)

## Reproducibility

- Use `set_global_seed(42)` in [src/models.py](src/models.py) before any training to align NumPy/torch/CUDA seeds and deterministic CuDNN settings.
- Requirements are pinned in [requirements.txt](requirements.txt); use a fresh virtual environment.
- Pipelines/models are modular in `src/`; saved artifacts live under `models/` for repeatable runs.
- Optuna tuning helper (`optuna_tune_xgboost`) returns the best fitted model and the study for auditability.

## Author

MOUAD IDRISSI ZAKI
Created as part of a comprehensive ML systems course project.

## License

Educational use only.
