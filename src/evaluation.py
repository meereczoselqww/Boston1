"""
Model Evaluation Module
Comprehensive evaluation metrics and analysis functions
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
from sklearn.model_selection import learning_curve, cross_val_score
from scipy import stats
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_metrics(y_true, y_pred) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        
    Returns:
        Dictionary of metric names and values
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    # Additional metrics
    mse = mean_squared_error(y_true, y_pred)
    
    # Adjusted R²
    n = len(y_true)
    p = 1  # Will be updated if feature count is provided
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'Adjusted R²': adj_r2,
        'MAPE (%)': mape,
        'MSE': mse
    }


def evaluate_model(model, X, y, model_name: str = "Model") -> Dict[str, float]:
    """
    Evaluate a trained model on given data
    
    Args:
        model: Trained model
        X: Features
        y: True targets
        model_name: Name of the model for display
        
    Returns:
        Dictionary of metrics
    """
    y_pred = model.predict(X)
    metrics = calculate_metrics(y, y_pred)
    
    print(f"\n{model_name} Performance:")
    print("-" * 50)
    for metric, value in metrics.items():
        print(f"{metric:15s}: {value:.4f}")
    
    return metrics


def compare_models(
    models: Dict[str, Any],
    X_train, y_train,
    X_val, y_val,
    X_test, y_test
) -> pd.DataFrame:
    """
    Compare multiple models across train, validation, and test sets
    
    Args:
        models: Dictionary of model names and instances
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate on all sets
        train_metrics = calculate_metrics(y_train, model.predict(X_train))
        val_metrics = calculate_metrics(y_val, model.predict(X_val))
        test_metrics = calculate_metrics(y_test, model.predict(X_test))
        
        # Store results
        results.append({
            'Model': name,
            'Train RMSE': train_metrics['RMSE'],
            'Val RMSE': val_metrics['RMSE'],
            'Test RMSE': test_metrics['RMSE'],
            'Train R²': train_metrics['R²'],
            'Val R²': val_metrics['R²'],
            'Test R²': test_metrics['R²'],
            'Train MAE': train_metrics['MAE'],
            'Val MAE': val_metrics['MAE'],
            'Test MAE': test_metrics['MAE']
        })
    
    results_df = pd.DataFrame(results)
    return results_df


def plot_predictions(y_true, y_pred, model_name: str = "Model", ax=None):
    """
    Plot predicted vs actual values
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        model_name: Name of the model
        ax: Matplotlib axis object (optional)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    ax.set_title(f'{model_name}: Predicted vs Actual', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add R² score
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.95, f'R² = {r2:.4f}', 
            transform=ax.transAxes, 
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def plot_residuals(y_true, y_pred, model_name: str = "Model", ax=None):
    """
    Plot residuals analysis
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        model_name: Name of the model
        ax: Matplotlib axis object (optional)
    """
    residuals = y_true - y_pred
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Predicted Values', fontsize=12)
    ax.set_ylabel('Residuals', fontsize=12)
    ax.set_title(f'{model_name}: Residual Plot', fontsize=14)
    ax.grid(True, alpha=0.3)


def plot_learning_curves(
    model, X, y, 
    cv: int = 5,
    scoring: str = 'neg_root_mean_squared_error',
    model_name: str = "Model"
):
    """
    Plot learning curves for bias-variance analysis
    
    Args:
        model: Model to evaluate
        X: Features
        y: Target
        cv: Number of cross-validation folds
        scoring: Scoring metric
        model_name: Name of the model
    """
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        cv=cv,
        scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1,
        random_state=42
    )
    
    train_mean = -train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = -val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score', marker='o')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
    
    plt.plot(train_sizes, val_mean, label='Validation score', marker='s')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)
    
    plt.xlabel('Training Set Size', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title(f'{model_name}: Learning Curves', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)


def statistical_comparison(
    model1_scores: np.ndarray,
    model2_scores: np.ndarray,
    model1_name: str = "Model 1",
    model2_name: str = "Model 2"
) -> Dict[str, Any]:
    """
    Perform statistical significance test between two models
    
    Args:
        model1_scores: Cross-validation scores for model 1
        model2_scores: Cross-validation scores for model 2
        model1_name: Name of model 1
        model2_name: Name of model 2
        
    Returns:
        Dictionary with test results
    """
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(model1_scores, model2_scores)
    
    results = {
        f'{model1_name} mean': model1_scores.mean(),
        f'{model1_name} std': model1_scores.std(),
        f'{model2_name} mean': model2_scores.mean(),
        f'{model2_name} std': model2_scores.std(),
        't-statistic': t_stat,
        'p-value': p_value,
        'significant (α=0.05)': p_value < 0.05
    }
    
    print(f"\nStatistical Comparison: {model1_name} vs {model2_name}")
    print("-" * 60)
    for key, value in results.items():
        if isinstance(value, bool):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.4f}")
    
    return results


def cross_validate_model(
    model, X, y,
    cv: int = 5,
    scoring: str = 'neg_root_mean_squared_error'
) -> Dict[str, Any]:
    """
    Perform cross-validation and return detailed results
    
    Args:
        model: Model to evaluate
        X: Features
        y: Target
        cv: Number of folds
        scoring: Scoring metric
        
    Returns:
        Dictionary with CV results
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    scores = -scores  # Convert back to positive RMSE
    
    results = {
        'mean_score': scores.mean(),
        'std_score': scores.std(),
        'min_score': scores.min(),
        'max_score': scores.max(),
        'all_scores': scores
    }
    
    return results
