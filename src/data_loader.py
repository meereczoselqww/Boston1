"""
Data Loading and Initial Processing Module
Handles reading the Boston Housing dataset and basic data operations
"""

import pandas as pd
import numpy as np
from typing import Tuple
from pathlib import Path


class DataLoader:
    """Class for loading and initial processing of Boston Housing data"""
    
    def __init__(self, data_path: str):
        """
        Initialize DataLoader with path to data file
        
        Args:
            data_path: Path to the housing.csv file
        """
        self.data_path = Path(data_path)
        self.feature_names = [
            'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
            'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
        ]
        self.target_name = 'MEDV'
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the Boston Housing dataset from CSV
        
        Returns:
            DataFrame with loaded data
        """
        # Load data with space-separated values
        df = pd.read_csv(
            self.data_path,
            delim_whitespace=True,
            names=self.feature_names,
            header=None
        )
        
        print(f"Data loaded successfully!")
        print(f"Shape: {df.shape}")
        print(f"Features: {len(self.feature_names) - 1}")
        print(f"Target: {self.target_name}")
        
        return df
    
    def get_feature_descriptions(self) -> dict:
        """
        Get detailed descriptions of all features
        
        Returns:
            Dictionary with feature names and descriptions
        """
        descriptions = {
            'CRIM': 'Per capita crime rate by town',
            'ZN': 'Proportion of residential land zoned for lots over 25,000 sq.ft',
            'INDUS': 'Proportion of non-retail business acres per town',
            'CHAS': 'Charles River dummy variable (1 if bounds river; 0 otherwise)',
            'NOX': 'Nitric oxides concentration (parts per 10 million)',
            'RM': 'Average number of rooms per dwelling',
            'AGE': 'Proportion of owner-occupied units built prior to 1940',
            'DIS': 'Weighted distances to five Boston employment centres',
            'RAD': 'Index of accessibility to radial highways',
            'TAX': 'Full-value property-tax rate per $10,000',
            'PTRATIO': 'Pupil-teacher ratio by town',
            'B': '1000(Bk - 0.63)^2 where Bk is proportion of blacks by town',
            'LSTAT': '% lower status of the population',
            'MEDV': 'Median value of owner-occupied homes in $1000s (TARGET)'
        }
        return descriptions
    
    def split_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split dataframe into features and target
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (features_df, target_series)
        """
        X = df.drop(columns=[self.target_name])
        y = df[self.target_name]
        return X, y


def load_boston_data(data_path: str = "../data/housing.csv") -> pd.DataFrame:
    """
    Convenience function to load Boston Housing data
    
    Args:
        data_path: Path to the housing.csv file
        
    Returns:
        DataFrame with loaded data
    """
    loader = DataLoader(data_path)
    return loader.load_data()
