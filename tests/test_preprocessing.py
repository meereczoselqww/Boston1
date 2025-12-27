"""
Unit tests for preprocessing module
Tests train/val/test splitting, pipeline creation, and feature engineering
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing import (
    create_train_val_test_split,
    create_preprocessing_pipeline,
    FeatureEngineering
)


class TestTrainValTestSplit:
    """Test suite for data splitting"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        n_samples = 100
        X = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples)
        })
        y = pd.Series(np.random.randn(n_samples), name='target')
        return X, y
    
    def test_split_sizes(self, sample_data):
        """Test that split sizes are correct"""
        X, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = create_train_val_test_split(
            X, y, test_size=0.15, val_size=0.15, random_state=42
        )
        
        # Check approximate sizes (70/15/15)
        total = len(X)
        assert abs(len(X_train) / total - 0.70) < 0.05
        assert abs(len(X_val) / total - 0.15) < 0.05
        assert abs(len(X_test) / total - 0.15) < 0.05
        
        # Check sum equals total
        assert len(X_train) + len(X_val) + len(X_test) == total
    
    def test_split_no_overlap(self, sample_data):
        """Test that splits have no overlapping indices"""
        X, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = create_train_val_test_split(
            X, y, random_state=42
        )
        
        train_idx = set(X_train.index)
        val_idx = set(X_val.index)
        test_idx = set(X_test.index)
        
        # No overlap
        assert len(train_idx & val_idx) == 0
        assert len(train_idx & test_idx) == 0
        assert len(val_idx & test_idx) == 0
    
    def test_reproducibility(self, sample_data):
        """Test that split is reproducible with same random_state"""
        X, y = sample_data
        
        split1 = create_train_val_test_split(X, y, random_state=42)
        split2 = create_train_val_test_split(X, y, random_state=42)
        
        # Check train sets are identical
        pd.testing.assert_frame_equal(split1[0], split2[0])
        pd.testing.assert_series_equal(split1[3], split2[3])


class TestPreprocessingPipeline:
    """Test suite for preprocessing pipeline"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data"""
        np.random.seed(42)
        return pd.DataFrame({
            'feature1': np.random.randn(100) * 10 + 5,
            'feature2': np.random.randn(100) * 2 + 10,
            'feature3': np.random.randn(100) * 100
        })
    
    def test_standard_scaler_pipeline(self, sample_data):
        """Test pipeline with standard scaler"""
        pipeline = create_preprocessing_pipeline(scaler_type='standard')
        
        # Fit and transform
        X_scaled = pipeline.fit_transform(sample_data)
        
        # Check mean is approximately 0
        assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-10)
        
        # Check std is approximately 1
        assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-10)
    
    def test_robust_scaler_pipeline(self, sample_data):
        """Test pipeline with robust scaler"""
        pipeline = create_preprocessing_pipeline(scaler_type='robust')
        
        # Fit and transform
        X_scaled = pipeline.fit_transform(sample_data)
        
        # Check median is approximately 0
        assert np.allclose(np.median(X_scaled, axis=0), 0, atol=0.1)
    
    def test_minmax_scaler_pipeline(self, sample_data):
        """Test pipeline with minmax scaler"""
        pipeline = create_preprocessing_pipeline(scaler_type='minmax')
        
        # Fit and transform
        X_scaled = pipeline.fit_transform(sample_data)
        
        # Check values are in [0, 1]
        assert np.all(X_scaled >= 0)
        assert np.all(X_scaled <= 1)
    
    def test_invalid_scaler(self):
        """Test that invalid scaler type raises error"""
        with pytest.raises(ValueError):
            create_preprocessing_pipeline(scaler_type='invalid')


class TestFeatureEngineering:
    """Test suite for feature engineering"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data with feature names matching Boston dataset"""
        np.random.seed(42)
        return pd.DataFrame({
            'CRIM': np.random.rand(100) * 10,
            'RM': np.random.rand(100) * 5 + 4,
            'LSTAT': np.random.rand(100) * 30,
            'DIS': np.random.rand(100) * 10 + 1,
            'NOX': np.random.rand(100) * 0.5 + 0.3
        })
    
    def test_feature_engineering_creates_new_features(self, sample_data):
        """Test that feature engineering creates expected features"""
        fe = FeatureEngineering()
        X_transformed = fe.fit_transform(sample_data)
        
        # Check new features are created
        assert 'RM_LSTAT' in X_transformed.columns
        assert 'DIS_NOX' in X_transformed.columns
        assert 'RM_squared' in X_transformed.columns
        assert 'LSTAT_squared' in X_transformed.columns
        assert 'log_CRIM' in X_transformed.columns
        assert 'log_DIS' in X_transformed.columns
        
        # Check original features are preserved
        assert 'CRIM' in X_transformed.columns
        assert 'RM' in X_transformed.columns
    
    def test_interaction_features_correct(self, sample_data):
        """Test that interaction features are calculated correctly"""
        fe = FeatureEngineering()
        X_transformed = fe.fit_transform(sample_data)
        
        # Check RM_LSTAT calculation
        expected_rm_lstat = sample_data['RM'] * sample_data['LSTAT']
        pd.testing.assert_series_equal(
            X_transformed['RM_LSTAT'],
            expected_rm_lstat,
            check_names=False
        )
        
        # Check DIS_NOX calculation
        expected_dis_nox = sample_data['DIS'] * sample_data['NOX']
        pd.testing.assert_series_equal(
            X_transformed['DIS_NOX'],
            expected_dis_nox,
            check_names=False
        )
    
    def test_polynomial_features_correct(self, sample_data):
        """Test that polynomial features are calculated correctly"""
        fe = FeatureEngineering()
        X_transformed = fe.fit_transform(sample_data)
        
        # Check RM_squared
        expected_rm_sq = sample_data['RM'] ** 2
        pd.testing.assert_series_equal(
            X_transformed['RM_squared'],
            expected_rm_sq,
            check_names=False
        )
    
    def test_log_features_correct(self, sample_data):
        """Test that log features are calculated correctly"""
        fe = FeatureEngineering()
        X_transformed = fe.fit_transform(sample_data)
        
        # Check log_CRIM
        expected_log_crim = np.log1p(sample_data['CRIM'])
        pd.testing.assert_series_equal(
            X_transformed['log_CRIM'],
            expected_log_crim,
            check_names=False
        )
    
    def test_fit_transform_consistency(self, sample_data):
        """Test that fit_transform gives same result as fit then transform"""
        fe1 = FeatureEngineering()
        fe2 = FeatureEngineering()
        
        X_fit_transform = fe1.fit_transform(sample_data)
        X_fit_then_transform = fe2.fit(sample_data).transform(sample_data)
        
        pd.testing.assert_frame_equal(X_fit_transform, X_fit_then_transform)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
