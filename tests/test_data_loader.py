"""
Unit tests for data_loader module
Tests data loading, feature descriptions, and data splitting functionality
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import DataLoader, load_boston_data


class TestDataLoader:
    """Test suite for DataLoader class"""
    
    @pytest.fixture
    def data_path(self):
        """Return path to test data"""
        return Path(__file__).parent.parent / 'data' / 'housing.csv'
    
    @pytest.fixture
    def loader(self, data_path):
        """Create DataLoader instance"""
        return DataLoader(str(data_path))
    
    def test_init(self, data_path):
        """Test DataLoader initialization"""
        loader = DataLoader(str(data_path))
        assert loader.data_path == data_path
        assert len(loader.feature_names) == 14
        assert loader.target_name == 'MEDV'
    
    def test_load_data(self, loader):
        """Test data loading functionality"""
        df = loader.load_data()
        
        # Check shape
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 506
        assert df.shape[1] == 14
        
        # Check no missing values
        assert df.isnull().sum().sum() == 0
        
        # Check all columns present
        expected_cols = [
            'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
            'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
        ]
        assert list(df.columns) == expected_cols
    
    def test_get_feature_descriptions(self, loader):
        """Test feature descriptions retrieval"""
        descriptions = loader.get_feature_descriptions()
        
        assert isinstance(descriptions, dict)
        assert len(descriptions) == 14
        assert 'CRIM' in descriptions
        assert 'MEDV' in descriptions
        assert isinstance(descriptions['CRIM'], str)
    
    def test_split_features_target(self, loader):
        """Test splitting features and target"""
        df = loader.load_data()
        X, y = loader.split_features_target(df)
        
        # Check shapes
        assert X.shape == (506, 13)
        assert y.shape == (506,)
        
        # Check MEDV not in features
        assert 'MEDV' not in X.columns
        
        # Check y is Series
        assert isinstance(y, pd.Series)
        assert y.name == 'MEDV'


def test_load_boston_data():
    """Test convenience function for loading data"""
    data_path = Path(__file__).parent.parent / 'data' / 'housing.csv'
    df = load_boston_data(str(data_path))
    
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (506, 14)


def test_data_value_ranges():
    """Test that data values are in expected ranges"""
    data_path = Path(__file__).parent.parent / 'data' / 'housing.csv'
    loader = DataLoader(str(data_path))
    df = loader.load_data()
    
    # CHAS should be binary
    assert df['CHAS'].isin([0, 1]).all()
    
    # RM (rooms) should be positive
    assert (df['RM'] > 0).all()
    
    # MEDV (target) should be positive
    assert (df['MEDV'] > 0).all()
    
    # PTRATIO should be reasonable (student-teacher ratio)
    assert (df['PTRATIO'] > 0).all()
    assert (df['PTRATIO'] < 100).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
