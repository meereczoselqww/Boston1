# Quick Start Guide

This guide helps you get started with the Boston Housing Price Prediction project in 5 minutes.

## Prerequisites

- Python 3.9, 3.10, or 3.11
- pip or conda package manager
- Git (for cloning)

## 1. Clone and Setup (2 minutes)

```bash
# Clone repository
git clone https://github.com/meereczoselqww/Boston1.git
cd Boston1

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Alternative (Conda):**
```bash
conda env create -f environment.yml
conda activate boston-housing-ml
```

## 2. Verify Installation (30 seconds)

```bash
# Run tests to verify everything works
pytest tests/ -v

# Expected: 25 passed tests
```

## 3. Explore the Project (2 minutes)

### Option A: Jupyter Notebook (Recommended for beginners)

```bash
# Start Jupyter
jupyter notebook

# Open: notebooks/Boston_Housing_Analysis.ipynb
# Run cells sequentially to see full analysis
```

### Option B: Python Scripts (For advanced users)

```python
# Load data
from src.data_loader import load_boston_data

df = load_boston_data('data/housing.csv')
print(df.head())

# Split and preprocess
from src.preprocessing import create_train_val_test_split, create_preprocessing_pipeline

X = df.drop('MEDV', axis=1)
y = df['MEDV']

X_train, X_val, X_test, y_train, y_val, y_test = create_train_val_test_split(X, y)

pipeline = create_preprocessing_pipeline(scaler_type='standard')
X_train_scaled = pipeline.fit_transform(X_train)
X_val_scaled = pipeline.transform(X_val)
X_test_scaled = pipeline.transform(X_test)

# Train a model
from src.models import get_baseline_models
from src.evaluation import evaluate_model

models = get_baseline_models()
ridge = models['Ridge Regression']
ridge.fit(X_train_scaled, y_train)

# Evaluate
metrics = evaluate_model(ridge, X_test_scaled, y_test, "Ridge Regression")
print(metrics)
```

## 4. Key Files to Review

### For Understanding the Project:
1. **README.md** - Project overview and structure
2. **reports/final_report.md** - Complete analysis and results
3. **REQUIREMENTS_COMPLIANCE.md** - Detailed requirements checklist

### For Learning the Code:
1. **src/data_loader.py** - How to load and process data
2. **src/preprocessing.py** - Feature engineering and scaling
3. **src/models.py** - Model definitions and training
4. **src/evaluation.py** - Metrics and evaluation
5. **src/interpretation.py** - Model explanation methods

### For Contributing:
1. **CONTRIBUTING.md** - Development guidelines
2. **tests/** - Test examples for new code

## 5. Common Tasks

### Train a Model

```python
from src.models import get_baseline_models, set_global_seed
from src.data_loader import load_boston_data
from src.preprocessing import create_train_val_test_split, create_preprocessing_pipeline

# Set seed for reproducibility
set_global_seed(42)

# Load and prepare data
df = load_boston_data('data/housing.csv')
X = df.drop('MEDV', axis=1)
y = df['MEDV']

X_train, X_val, X_test, y_train, y_val, y_test = create_train_val_test_split(X, y)

pipeline = create_preprocessing_pipeline(scaler_type='standard')
X_train_scaled = pipeline.fit_transform(X_train)
X_test_scaled = pipeline.transform(X_test)

# Train
models = get_baseline_models()
model = models['Ridge Regression']
model.fit(X_train_scaled, y_train)

# Evaluate
from src.evaluation import calculate_metrics
predictions = model.predict(X_test_scaled)
metrics = calculate_metrics(y_test, predictions)
print(f"Test RMSE: {metrics['RMSE']:.2f}")
print(f"Test R²: {metrics['R²']:.3f}")
```

### Generate SHAP Explanations

```python
from src.models import XGBRegressor
from src.interpretation import explain_with_shap_tree
import matplotlib.pyplot as plt

# Train XGBoost
model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Generate SHAP explanations
explainer, shap_values = explain_with_shap_tree(
    model, 
    X_test_scaled,
    feature_names=X.columns.tolist(),
    plot_type='summary'
)

plt.show()
```

### Run Hyperparameter Optimization

```python
from src.models import optuna_tune_xgboost

# This takes 5-10 minutes
best_model, study = optuna_tune_xgboost(
    X_train_scaled, y_train,
    X_val_scaled, y_val,
    n_trials=30,
    random_state=42
)

print(f"Best parameters: {study.best_params}")
print(f"Best validation RMSE: {study.best_value:.3f}")
```

## 6. Project Structure Quick Reference

```
Boston1/
├── data/                        # Dataset
│   └── housing.csv             # Boston housing data (506 samples)
│
├── notebooks/                   # Analysis notebooks
│   └── Boston_Housing_Analysis.ipynb  # Main notebook with all steps
│
├── src/                         # Source code
│   ├── data_loader.py          # Data loading (103 lines)
│   ├── preprocessing.py        # Preprocessing (163 lines)
│   ├── models.py              # Models (309 lines)
│   ├── evaluation.py          # Evaluation (300 lines)
│   └── interpretation.py      # Explanations (418 lines)
│
├── tests/                       # Unit tests
│   ├── test_data_loader.py    # 6 tests
│   ├── test_preprocessing.py  # 12 tests
│   └── test_evaluation.py     # 7 tests
│
├── models/                      # Saved models
│   ├── preprocessor.joblib
│   ├── xgboost_optimized.joblib
│   ├── lightgbm.joblib
│   ├── stacking_regressor.joblib
│   └── deep_mlp_state_dict.pth
│
├── reports/                     # Documentation
│   ├── final_report.md         # Main report (1,150 lines)
│   └── *.docx                  # Additional documentation
│
├── .github/workflows/           # CI/CD
│   └── ci.yml                  # GitHub Actions config
│
├── README.md                    # Project overview
├── CONTRIBUTING.md              # Development guide
├── REQUIREMENTS_COMPLIANCE.md   # Requirements checklist
├── requirements.txt             # Python dependencies
└── environment.yml              # Conda environment
```

## 7. Performance Benchmarks

Expected results after running the full pipeline:

| Model | Test RMSE | Test R² | Training Time |
|-------|-----------|---------|---------------|
| Ridge Regression | 4.75 | 0.74 | < 1 second |
| XGBoost (default) | 3.84 | 0.84 | ~ 2 seconds |
| LightGBM | 3.67 | 0.85 | ~ 1 second |
| XGBoost (tuned) | 3.21 | 0.88 | ~ 5 minutes |
| Deep MLP | 3.89 | 0.83 | ~ 2 minutes |
| **Stacking Ensemble** | **3.18** | **0.89** | ~ 5 seconds |

**Best Model:** Stacking Ensemble (RMSE=3.18, R²=0.887)

## 8. Troubleshooting

### Tests Fail
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Clear pytest cache
rm -rf .pytest_cache

# Run tests again
pytest tests/ -v
```

### Import Errors
```bash
# Ensure you're in the project root
cd Boston1

# Add src to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or use the package structure
pip install -e .
```

### Missing Data
```bash
# Verify data file exists
ls -l data/housing.csv

# Should be 49,082 bytes, 506 lines
wc -l data/housing.csv
```

### Jupyter Kernel Issues
```bash
# Install kernel
python -m ipykernel install --user --name=boston-ml

# Select kernel in Jupyter: Kernel > Change Kernel > boston-ml
```

## 9. Next Steps

1. **Read the Report**: Start with `reports/final_report.md` for comprehensive understanding
2. **Run the Notebook**: Open `notebooks/Boston_Housing_Analysis.ipynb` and run all cells
3. **Explore the Code**: Review `src/` modules to understand implementation
4. **Experiment**: Try different models, hyperparameters, or feature engineering
5. **Contribute**: See `CONTRIBUTING.md` for guidelines

## 10. Getting Help

- **Documentation**: Check `reports/final_report.md` for detailed explanations
- **Code Examples**: See `notebooks/Boston_Housing_Analysis.ipynb` for usage examples
- **Requirements**: Review `REQUIREMENTS_COMPLIANCE.md` for what's implemented
- **Issues**: Open an issue on GitHub for bugs or questions

## 11. Key Commands Summary

```bash
# Setup
pip install -r requirements.txt

# Testing
pytest tests/ -v                          # Run all tests
pytest tests/ --cov=src --cov-report=html # With coverage

# Code Quality
black src/ tests/                         # Format code
flake8 src/ tests/                        # Lint code
mypy src/                                 # Type check

# Jupyter
jupyter notebook                          # Start notebook server

# Documentation
cat README.md                             # Project overview
cat reports/final_report.md              # Full report
cat REQUIREMENTS_COMPLIANCE.md            # Compliance checklist
```

---

**Time to Productivity:**
- Quick exploration: 5 minutes
- Basic understanding: 30 minutes
- Deep dive: 2-3 hours
- Full mastery: Read all documentation

**Happy coding!** 🚀
