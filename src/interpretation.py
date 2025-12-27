"""
Model Interpretation and Explainability Module
SHAP values, feature importance, and local explanations
Using PyTorch instead of sklearn for permutation importance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import torch
from typing import Any, Optional, List


def get_feature_importance_tree(model, feature_names: List[str]) -> pd.DataFrame:
    """
    Get feature importance from tree-based models
    
    Args:
        model: Trained tree-based model
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importances sorted
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        raise ValueError("Model does not have feature_importances_ attribute")
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    return importance_df


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    model_name: str = "Model",
    figsize: tuple = (10, 8)
):
    """
    Plot feature importance
    
    Args:
        importance_df: DataFrame with Feature and Importance columns
        top_n: Number of top features to show
        model_name: Name of the model
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Select top N features
    plot_df = importance_df.head(top_n)
    
    plt.barh(range(len(plot_df)), plot_df['Importance'])
    plt.yticks(range(len(plot_df)), plot_df['Feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'{model_name}: Top {top_n} Feature Importances', fontsize=14)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()


def calculate_permutation_importance(
    model, X, y,
    feature_names: List[str],
    n_repeats: int = 10,
    random_state: int = 42,
    scoring_fn=None
) -> pd.DataFrame:
    """
    Calculate permutation importance using PyTorch
    
    Args:
        model: Trained model (can be PyTorch model or sklearn-like with predict method)
        X: Features (array, DataFrame, or torch.Tensor)
        y: Target (array, Series, or torch.Tensor)
        feature_names: List of feature names
        n_repeats: Number of times to permute each feature
        random_state: Random seed
        scoring_fn: Custom scoring function (default: negative MSE)
        
    Returns:
        DataFrame with permutation importances
    """
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    
    # Convert to numpy arrays if needed
    if isinstance(X, pd.DataFrame):
        X_arr = X.values.copy()
    elif isinstance(X, torch.Tensor):
        X_arr = X.cpu().numpy().copy()
    else:
        X_arr = np.array(X).copy()
    
    if isinstance(y, pd.Series):
        y_arr = y.values
    elif isinstance(y, torch.Tensor):
        y_arr = y.cpu().numpy()
    else:
        y_arr = np.array(y)
    
    n_features = X_arr.shape[1]
    
    # Define default scoring function (negative MSE - higher is better)
    if scoring_fn is None:
        def scoring_fn(model, X_data, y_true):
            # Check if it's a PyTorch model
            if hasattr(model, 'forward'):
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_data)
                    predictions = model(X_tensor).numpy().flatten()
            else:
                predictions = model.predict(X_data)
            mse = np.mean((y_true - predictions) ** 2)
            return -mse  # negative MSE so higher is better
    
    # Calculate baseline score
    baseline_score = scoring_fn(model, X_arr, y_arr)
    
    # Calculate permutation importance for each feature
    importances = np.zeros((n_features, n_repeats))
    
    for feat_idx in range(n_features):
        for rep in range(n_repeats):
            # Copy the data
            X_permuted = X_arr.copy()
            
            # Permute the feature
            perm_indices = torch.randperm(X_permuted.shape[0]).numpy()
            X_permuted[:, feat_idx] = X_permuted[perm_indices, feat_idx]
            
            # Calculate score with permuted feature
            permuted_score = scoring_fn(model, X_permuted, y_arr)
            
            # Importance is the decrease in performance
            importances[feat_idx, rep] = baseline_score - permuted_score
    
    # Calculate mean and std
    importance_mean = np.mean(importances, axis=1)
    importance_std = np.std(importances, axis=1)
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_mean,
        'Std': importance_std
    }).sort_values('Importance', ascending=False)
    
    return importance_df


def plot_permutation_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    model_name: str = "Model",
    figsize: tuple = (10, 8)
):
    """
    Plot permutation importance with error bars
    
    Args:
        importance_df: DataFrame with Feature, Importance, and Std columns
        top_n: Number of top features to show
        model_name: Name of the model
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Select top N features
    plot_df = importance_df.head(top_n)
    
    plt.barh(range(len(plot_df)), plot_df['Importance'], xerr=plot_df['Std'])
    plt.yticks(range(len(plot_df)), plot_df['Feature'])
    plt.xlabel('Permutation Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'{model_name}: Top {top_n} Permutation Importances', fontsize=14)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()


def explain_with_shap_tree(
    model, X,
    feature_names: Optional[List[str]] = None,
    plot_type: str = 'summary'
):
    """
    Generate SHAP explanations for tree-based models
    
    Args:
        model: Trained tree-based model
        X: Features (array or DataFrame)
        feature_names: List of feature names
        plot_type: Type of plot ('summary', 'bar', 'waterfall', 'force')
    """
    # Create explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Convert to DataFrame if needed
    if feature_names is not None and not isinstance(X, pd.DataFrame):
        X_df = pd.DataFrame(X, columns=feature_names)
    else:
        X_df = X
    
    # Create plots
    if plot_type == 'summary':
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_df, show=False)
        plt.tight_layout()
    elif plot_type == 'bar':
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_df, plot_type='bar', show=False)
        plt.tight_layout()
    
    return explainer, shap_values


def explain_with_shap_kernel(
    model, X_background, X_explain,
    feature_names: Optional[List[str]] = None,
    n_samples: int = 100
):
    """
    Generate SHAP explanations using KernelExplainer (model-agnostic)
    
    Args:
        model: Trained model
        X_background: Background data for explainer
        X_explain: Data to explain
        feature_names: List of feature names
        n_samples: Number of background samples to use
    """
    # Sample background data if too large
    if len(X_background) > n_samples:
        background_idx = np.random.choice(len(X_background), n_samples, replace=False)
        X_background_sample = X_background[background_idx] if isinstance(X_background, np.ndarray) else X_background.iloc[background_idx]
    else:
        X_background_sample = X_background
    
    # Create explainer
    explainer = shap.KernelExplainer(model.predict, X_background_sample)
    shap_values = explainer.shap_values(X_explain)
    
    return explainer, shap_values


def plot_shap_waterfall(
    explainer, shap_values, X, 
    instance_idx: int,
    feature_names: Optional[List[str]] = None
):
    """
    Plot SHAP waterfall plot for a single instance
    
    Args:
        explainer: SHAP explainer object
        shap_values: SHAP values
        X: Feature data
        instance_idx: Index of instance to explain
        feature_names: List of feature names
    """
    if hasattr(shap, 'waterfall_plot'):
        # For newer SHAP versions
        if isinstance(X, pd.DataFrame):
            X_instance = X.iloc[instance_idx]
        else:
            X_instance = X[instance_idx]
        
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[instance_idx],
                base_values=explainer.expected_value,
                data=X_instance,
                feature_names=feature_names
            )
        )
    else:
        # Fallback for older versions
        shap.force_plot(
            explainer.expected_value,
            shap_values[instance_idx],
            X[instance_idx] if isinstance(X, np.ndarray) else X.iloc[instance_idx],
            feature_names=feature_names,
            matplotlib=True
        )


def plot_shap_dependence(
    shap_values, X,
    feature_name: str,
    interaction_feature: Optional[str] = None,
    feature_names: Optional[List[str]] = None
):
    """
    Plot SHAP dependence plot
    
    Args:
        shap_values: SHAP values
        X: Feature data
        feature_name: Name of feature to plot
        interaction_feature: Name of interaction feature
        feature_names: List of all feature names
    """
    plt.figure(figsize=(10, 6))
    
    if isinstance(X, pd.DataFrame):
        feature_idx = list(X.columns).index(feature_name)
        interaction_idx = list(X.columns).index(interaction_feature) if interaction_feature else 'auto'
    else:
        feature_idx = feature_names.index(feature_name) if feature_names else feature_name
        interaction_idx = feature_names.index(interaction_feature) if interaction_feature and feature_names else 'auto'
    
    shap.dependence_plot(
        feature_idx,
        shap_values,
        X,
        interaction_index=interaction_idx,
        show=False
    )
    plt.tight_layout()


def compare_feature_importance_methods(
    model, X, y,
    feature_names: List[str],
    model_name: str = "Model"
):
    """
    Compare feature importance from different methods
    
    Args:
        model: Trained model
        X: Features
        y: Target
        feature_names: List of feature names
        model_name: Name of the model
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Tree-based importance
    if hasattr(model, 'feature_importances_'):
        tree_importance = get_feature_importance_tree(model, feature_names)
        top_features = tree_importance.head(15)
        
        axes[0].barh(range(len(top_features)), top_features['Importance'])
        axes[0].set_yticks(range(len(top_features)))
        axes[0].set_yticklabels(top_features['Feature'])
        axes[0].set_xlabel('Importance')
        axes[0].set_title(f'{model_name}: Feature Importance (Built-in)')
        axes[0].invert_yaxis()
        axes[0].grid(axis='x', alpha=0.3)
    
    # Permutation importance
    perm_importance = calculate_permutation_importance(model, X, y, feature_names)
    top_perm = perm_importance.head(15)
    
    axes[1].barh(range(len(top_perm)), top_perm['Importance'], xerr=top_perm['Std'])
    axes[1].set_yticks(range(len(top_perm)))
    axes[1].set_yticklabels(top_perm['Feature'])
    axes[1].set_xlabel('Importance')
    axes[1].set_title(f'{model_name}: Permutation Importance')
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()


def local_explanation_examples(
    model, X, y,
    explainer, shap_values,
    feature_names: List[str],
    n_examples: int = 3
):
    """
    Show local explanations for several examples
    
    Args:
        model: Trained model
        X: Features
        y: True targets
        explainer: SHAP explainer
        shap_values: SHAP values
        feature_names: List of feature names
        n_examples: Number of examples to show
    """
    # Select diverse examples
    predictions = model.predict(X)
    residuals = np.abs(y - predictions)
    
    # Best prediction, worst prediction, median prediction
    best_idx = np.argmin(residuals)
    worst_idx = np.argmax(residuals)
    median_idx = np.argsort(residuals)[len(residuals)//2]
    
    indices = [best_idx, median_idx, worst_idx]
    labels = ['Best Prediction', 'Median Prediction', 'Worst Prediction']
    
    for idx, label in zip(indices[:n_examples], labels[:n_examples]):
        print(f"\n{label}:")
        print(f"True value: {y.iloc[idx] if hasattr(y, 'iloc') else y[idx]:.2f}")
        print(f"Predicted value: {predictions[idx]:.2f}")
        print(f"Error: {residuals[idx]:.2f}")
        
        # Plot waterfall
        plt.figure(figsize=(10, 6))
        plot_shap_waterfall(explainer, shap_values, X, idx, feature_names)
        plt.title(f'{label} - SHAP Explanation')
        plt.tight_layout()
