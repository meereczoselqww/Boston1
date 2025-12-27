"""
Preprocessing Pipeline Module
Handles feature engineering, scaling, and transformation pipelines
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from typing import Tuple, List, Optional
import joblib


def create_train_val_test_split(
    X: pd.DataFrame, 
    y: pd.Series,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train, validation, and test sets
    
    Args:
        X: Feature dataframe
        y: Target series
        test_size: Proportion for test set
        val_size: Proportion for validation set (from training data)
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Second split: separate validation from training
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
    )
    
    print(f"Train set size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Validation set size: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test set size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_preprocessing_pipeline(
    scaler_type: str = 'standard',
    feature_names: Optional[List[str]] = None
) -> Pipeline:
    """
    Create a preprocessing pipeline with scaling
    
    Args:
        scaler_type: Type of scaler ('standard', 'robust', or 'minmax')
        feature_names: List of feature names (optional)
        
    Returns:
        Sklearn Pipeline object
    """
    scalers = {
        'standard': StandardScaler(),
        'robust': RobustScaler(),
        'minmax': MinMaxScaler()
    }
    
    if scaler_type not in scalers:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    pipeline = Pipeline([
        ('scaler', scalers[scaler_type])
    ])
    
    return pipeline


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to the dataset
    
    Args:
        df: Input dataframe
        
    Returns:
        DataFrame with additional engineered features
    """
    df_copy = df.copy()
    
    # Interaction features
    df_copy['RM_LSTAT'] = df_copy['RM'] * df_copy['LSTAT']
    df_copy['DIS_NOX'] = df_copy['DIS'] * df_copy['NOX']
    
    # Polynomial features for key variables
    df_copy['RM_squared'] = df_copy['RM'] ** 2
    df_copy['LSTAT_squared'] = df_copy['LSTAT'] ** 2
    
    # Log transforms for skewed features
    df_copy['log_CRIM'] = np.log1p(df_copy['CRIM'])
    df_copy['log_DIS'] = np.log1p(df_copy['DIS'])
    
    # Binned features
    df_copy['RM_bins'] = pd.cut(df_copy['RM'], bins=3, labels=['Small', 'Medium', 'Large'])
    df_copy['RM_bins'] = df_copy['RM_bins'].astype(str)
    
    return df_copy


class FeatureEngineering:
    """Class for managing feature engineering transformations"""
    
    def __init__(self):
        self.feature_names_in_ = None
        self.feature_names_out_ = None
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the feature engineering (learn feature names)"""
        self.feature_names_in_ = list(X.columns)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering transformations"""
        X_copy = X.copy()
        
        # Interaction features
        X_copy['RM_LSTAT'] = X_copy['RM'] * X_copy['LSTAT']
        X_copy['DIS_NOX'] = X_copy['DIS'] * X_copy['NOX']
        
        # Polynomial features
        X_copy['RM_squared'] = X_copy['RM'] ** 2
        X_copy['LSTAT_squared'] = X_copy['LSTAT'] ** 2
        
        # Log transforms
        X_copy['log_CRIM'] = np.log1p(X_copy['CRIM'])
        X_copy['log_DIS'] = np.log1p(X_copy['DIS'])
        
        self.feature_names_out_ = list(X_copy.columns)
        
        return X_copy
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)


def save_pipeline(pipeline, filepath: str):
    """Save a fitted pipeline to disk"""
    joblib.dump(pipeline, filepath)
    print(f"Pipeline saved to {filepath}")


def load_pipeline(filepath: str):
    """Load a fitted pipeline from disk"""
    pipeline = joblib.load(filepath)
    print(f"Pipeline loaded from {filepath}")
    return pipeline
