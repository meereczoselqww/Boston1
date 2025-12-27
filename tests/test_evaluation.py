"""
Unit tests for evaluation module
Tests metric calculations, model comparison, and statistical testing
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluation import (
    calculate_metrics,
    evaluate_model,
    statistical_comparison,
    cross_validate_model
)


class TestMetricsCalculation:
    """Test suite for metric calculation functions"""
    
    @pytest.fixture
    def perfect_predictions(self):
        """Create perfect predictions for testing"""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        return y_true, y_pred
    
    @pytest.fixture
    def imperfect_predictions(self):
        """Create imperfect predictions for testing"""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.9, 3.8, 5.2])
        return y_true, y_pred
    
    def test_perfect_predictions(self, perfect_predictions):
        """Test metrics with perfect predictions"""
        y_true, y_pred = perfect_predictions
        metrics = calculate_metrics(y_true, y_pred)
        
        # RMSE should be 0
        assert np.isclose(metrics['RMSE'], 0, atol=1e-10)
        
        # MAE should be 0
        assert np.isclose(metrics['MAE'], 0, atol=1e-10)
        
        # R² should be 1
        assert np.isclose(metrics['R²'], 1, atol=1e-10)
        
        # MAPE should be 0
        assert np.isclose(metrics['MAPE (%)'], 0, atol=1e-10)
    
    def test_imperfect_predictions(self, imperfect_predictions):
        """Test metrics with imperfect predictions"""
        y_true, y_pred = imperfect_predictions
        metrics = calculate_metrics(y_true, y_pred)
        
        # Check all metrics are present
        assert 'RMSE' in metrics
        assert 'MAE' in metrics
        assert 'R²' in metrics
        assert 'MAPE (%)' in metrics
        
        # RMSE should be positive
        assert metrics['RMSE'] > 0
        
        # MAE should be positive
        assert metrics['MAE'] > 0
        
        # R² should be less than 1
        assert metrics['R²'] < 1
        assert metrics['R²'] > 0  # Still good predictions
    
    def test_metrics_relationships(self, imperfect_predictions):
        """Test mathematical relationships between metrics"""
        y_true, y_pred = imperfect_predictions
        metrics = calculate_metrics(y_true, y_pred)
        
        # RMSE >= MAE (equality when all errors are same magnitude)
        assert metrics['RMSE'] >= metrics['MAE']
        
        # MSE = RMSE^2
        assert np.isclose(metrics['MSE'], metrics['RMSE'] ** 2)


class TestModelEvaluation:
    """Test suite for model evaluation"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample regression data"""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X[:, 0] * 2 + X[:, 1] * 3 + np.random.randn(100) * 0.1
        return X, y
    
    def test_evaluate_model(self, sample_data):
        """Test model evaluation function"""
        X, y = sample_data
        
        # Train a simple model
        model = LinearRegression()
        model.fit(X, y)
        
        # Evaluate
        metrics = evaluate_model(model, X, y, model_name="Test Model")
        
        # Check metrics are returned
        assert isinstance(metrics, dict)
        assert 'RMSE' in metrics
        assert 'R²' in metrics
        
        # Check model performs well (linear relationship)
        assert metrics['R²'] > 0.9


class TestStatisticalComparison:
    """Test suite for statistical comparison"""
    
    def test_statistical_comparison_identical_models(self):
        """Test comparison of identical models"""
        scores1 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        scores2 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        
        results = statistical_comparison(
            scores1, scores2,
            model1_name="Model A",
            model2_name="Model B"
        )
        
        # p-value should be 1.0 (or NaN for identical values)
        assert results['p-value'] == 1.0 or np.isnan(results['p-value'])
        
        # Should not be significant
        if not np.isnan(results['p-value']):
            assert results['significant (α=0.05)'] == False
    
    def test_statistical_comparison_different_models(self):
        """Test comparison of clearly different models"""
        scores1 = np.array([1.0, 1.1, 0.9, 1.0, 1.1])
        scores2 = np.array([2.0, 2.1, 1.9, 2.0, 2.1])
        
        results = statistical_comparison(
            scores1, scores2,
            model1_name="Model A",
            model2_name="Model B"
        )
        
        # Should be significant (clearly different)
        assert results['significant (α=0.05)'] == True
        
        # p-value should be small
        assert results['p-value'] < 0.05


class TestCrossValidation:
    """Test suite for cross-validation"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data"""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X[:, 0] * 2 + X[:, 1] * 3 + np.random.randn(100) * 0.1
        return X, y
    
    def test_cross_validate_model(self, sample_data):
        """Test cross-validation function"""
        X, y = sample_data
        model = LinearRegression()
        
        results = cross_validate_model(model, X, y, cv=5)
        
        # Check results structure
        assert 'mean_score' in results
        assert 'std_score' in results
        assert 'all_scores' in results
        
        # Check all_scores has correct length
        assert len(results['all_scores']) == 5
        
        # Check mean and std are positive
        assert results['mean_score'] > 0
        assert results['std_score'] >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
