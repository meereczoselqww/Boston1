"""
Model Building Module
Contains model definitions and training utilities
"""

import numpy as np
import pandas as pd
import random
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import joblib

# Optional dependencies
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    CatBoostRegressor = None

try:
    from pytorch_tabnet.tab_model import TabNetRegressor
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
    TabNetRegressor = None


def set_global_seed(seed: int = 42) -> None:
    """Set seeds across libraries for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class DeepMLPRegressor(nn.Module):
    """Deep Multi-Layer Perceptron for Regression"""
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: list = [256, 128, 64, 32],
        dropout_rate: float = 0.3
    ):
        """
        Initialize Deep MLP
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
        """
        super(DeepMLPRegressor, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """Forward pass"""
        return self.network(x)


class MLPWrapper:
    """Scikit-learn style wrapper for PyTorch MLP"""
    
    def __init__(
        self,
        hidden_dims: list = [256, 128, 64, 32],
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        device: str = 'cpu',
        random_state: int = 42
    ):
        """
        Initialize MLP Wrapper
        
        Args:
            hidden_dims: Hidden layer dimensions
            dropout_rate: Dropout probability
            learning_rate: Learning rate for optimizer
            epochs: Number of training epochs
            batch_size: Batch size for training
            device: Device to use ('cpu' or 'cuda')
            random_state: Random seed
        """
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.random_state = random_state
        self.model = None
        self.loss_history = []
        
        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
    def fit(self, X, y):
        """Train the model"""
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y.values if hasattr(y, 'values') else y).reshape(-1, 1).to(self.device)
        
        input_dim = X.shape[1]
        self.model = DeepMLPRegressor(input_dim, self.hidden_dims, self.dropout_rate).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            self.loss_history.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy().flatten()
        return predictions


def get_baseline_models() -> Dict[str, Any]:
    """
    Get dictionary of baseline models
    
    Returns:
        Dictionary with model names and instances
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0, random_state=42),
        'Lasso Regression': Lasso(alpha=1.0, random_state=42),
        'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42)
    }
    return models


def get_tree_ensemble_models() -> Dict[str, Any]:
    """
    Get dictionary of tree-based ensemble models
    
    Returns:
        Dictionary with model names and instances
    """
    models = {
        'XGBoost': XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            verbosity=0
        ),
        'LightGBM': LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            verbosity=-1
        ),
        'CatBoost': CatBoostRegressor(
            iterations=100,
            learning_rate=0.1,
            depth=5,
            random_state=42,
            verbose=False
        )
    }
    return models


def get_tabnet_model(random_state: int = 42) -> TabNetRegressor:
    """Return a default TabNet regressor configured for tabular data."""
    return TabNetRegressor(
        n_d=32,
        n_a=32,
        n_steps=5,
        gamma=1.5,
        n_independent=2,
        n_shared=2,
        momentum=0.02,
        lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-3),
        scheduler_params={"step_size": 50, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type="entmax",
        verbose=0,
        seed=random_state
    )


def build_stacking_regressor(random_state: int = 42) -> StackingRegressor:
    """Create a stacking regressor that blends linear and tree ensemble bases."""
    base_estimators = [
        ("ridge", Ridge(alpha=1.0, random_state=random_state)),
        ("lgbm", LGBMRegressor(random_state=random_state, n_estimators=300, learning_rate=0.05)),
        ("xgb", XGBRegressor(random_state=random_state, n_estimators=300, learning_rate=0.05, max_depth=5, verbosity=0))
    ]
    final_estimator = Lasso(alpha=1e-3, random_state=random_state)
    return StackingRegressor(
        estimators=base_estimators,
        final_estimator=final_estimator,
        passthrough=True,
        n_jobs=-1
    )


def optuna_tune_xgboost(X_train, y_train, X_val, y_val, n_trials: int = 30, random_state: int = 42):
    """Lightweight Optuna tuner for XGBoost; returns best model and study."""
    import optuna

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "random_state": random_state,
            "verbosity": 0,
        }
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)
        preds = model.predict(X_val)
        rmse = np.sqrt(((preds - y_val) ** 2).mean())
        return rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_params.update({"random_state": random_state, "verbosity": 0})
    best_model = XGBRegressor(**best_params)
    best_model.fit(X_train, y_train, verbose=False)
    return best_model, study


def save_model(model, filepath: str):
    """Save a trained model to disk"""
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath: str):
    """Load a trained model from disk"""
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model
