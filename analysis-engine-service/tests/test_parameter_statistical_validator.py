"""
Unit tests for the Parameter Statistical Validator.

This module tests the statistical validation functionality that determines
if parameter changes are statistically significant.
"""

import pytest
import numpy as np
from analysis_engine.adaptive_layer.parameter_statistical_validator import ParameterStatisticalValidator

class TestParameterStatisticalValidator:
    """Test suite for the Parameter Statistical Validator"""

    def test_validate_parameter_change_significant(self):
        """Test statistical validation with data showing significant change"""
        validator = ParameterStatisticalValidator()
        
        # Baseline performance data
        baseline = [{"profit_loss": 1.2}, {"profit_loss": 1.5}, {"profit_loss": 1.1}, 
                    {"profit_loss": 1.3}, {"profit_loss": 1.4}, {"profit_loss": 1.2},
                    {"profit_loss": 1.3}, {"profit_loss": 1.2}, {"profit_loss": 1.4}, 
                    {"profit_loss": 1.3}, {"profit_loss": 1.2}, {"profit_loss": 1.4}]
        
        # New values with significant improvement
        new_values = [{"profit_loss": 1.8}, {"profit_loss": 2.0}, {"profit_loss": 1.9}, 
                     {"profit_loss": 2.1}, {"profit_loss": 1.7}, {"profit_loss": 2.0},
                     {"profit_loss": 1.9}, {"profit_loss": 1.8}, {"profit_loss": 2.1}, 
                     {"profit_loss": 1.9}, {"profit_loss": 2.0}, {"profit_loss": 1.8}]
                     
        result = validator.validate_parameter_change(baseline, new_values)
        
        # Assert the results
        assert result["is_significant"] == True
        assert result["confidence"] > 0.7  # High confidence
        assert result["p_value"] < 0.05  # Statistically significant
        assert result["effect_size"] > 0.5  # At least medium effect size
        assert "baseline_mean" in result
        assert "new_mean" in result
        assert result["new_mean"] > result["baseline_mean"]

    def test_validate_parameter_change_not_significant(self):
        """Test statistical validation with data not showing significant change"""
        validator = ParameterStatisticalValidator()
        
        # Baseline performance data
        baseline = [{"profit_loss": 1.2}, {"profit_loss": 1.5}, {"profit_loss": 1.1}, 
                    {"profit_loss": 1.3}, {"profit_loss": 1.4}, {"profit_loss": 1.2},
                    {"profit_loss": 1.3}, {"profit_loss": 1.2}, {"profit_loss": 1.4}, 
                    {"profit_loss": 1.3}, {"profit_loss": 1.2}, {"profit_loss": 1.4}]
        
        # New values with minor differences (not significant)
        new_values = [{"profit_loss": 1.25}, {"profit_loss": 1.45}, {"profit_loss": 1.15}, 
                     {"profit_loss": 1.35}, {"profit_loss": 1.4}, {"profit_loss": 1.25},
                     {"profit_loss": 1.3}, {"profit_loss": 1.25}, {"profit_loss": 1.35}, 
                     {"profit_loss": 1.3}, {"profit_loss": 1.35}, {"profit_loss": 1.25}]
                     
        result = validator.validate_parameter_change(baseline, new_values)
        
        # Assert the results
        assert result["is_significant"] == False
        assert result["p_value"] > 0.05  # Not statistically significant
    
    def test_insufficient_data(self):
        """Test handling of insufficient data"""
        validator = ParameterStatisticalValidator({"min_sample_size": 10})
        
        # Too few samples
        baseline = [{"profit_loss": 1.2}, {"profit_loss": 1.5}, {"profit_loss": 1.1}]
        new_values = [{"profit_loss": 1.8}, {"profit_loss": 2.0}, {"profit_loss": 1.9}]
                     
        result = validator.validate_parameter_change(baseline, new_values)
        
        # Assert the results
        assert result["is_significant"] == False
        assert "Sample size below threshold" in result["reason"]
        assert result["sample_sizes"] == (3, 3)
    
    def test_effect_size_calculation(self):
        """Test effect size calculation"""
        validator = ParameterStatisticalValidator()
        
        # Test small effect size
        group1 = [10, 11, 9, 10, 10, 11, 9, 10, 11, 10]
        group2 = [11, 12, 10, 11, 11, 12, 10, 11, 12, 11]
        
        effect_size = validator._calculate_cohens_d(group1, group2)
        interpretation = validator._interpret_effect_size(effect_size)
        
        assert 0.2 <= effect_size < 0.8
        assert interpretation in ["small", "medium"]
        
        # Test large effect size
        group1 = [10, 11, 9, 10, 10, 11, 9, 10, 11, 10]
        group2 = [15, 16, 14, 15, 15, 16, 14, 15, 16, 15]
        
        effect_size = validator._calculate_cohens_d(group1, group2)
        interpretation = validator._interpret_effect_size(effect_size)
        
        assert effect_size >= 0.8
        assert interpretation == "large"
    
    def test_required_sample_estimation(self):
        """Test estimation of required samples"""
        validator = ParameterStatisticalValidator()
        
        # Small effect size requires more samples
        estimated_samples = validator.estimate_required_samples(
            observed_effect=0.3,  # Small effect
            baseline_std=1.0,
            new_std=1.0
        )
        
        assert estimated_samples > 50  # Small effects need larger samples
        
        # Large effect size requires fewer samples
        estimated_samples = validator.estimate_required_samples(
            observed_effect=1.0,  # Large effect
            baseline_std=1.0,
            new_std=1.0
        )
        
        assert estimated_samples < 50  # Large effects need fewer samples
        
        # Ensure minimum sample size is respected
        estimated_samples = validator.estimate_required_samples(
            observed_effect=5.0,  # Very large effect
            baseline_std=1.0,
            new_std=1.0
        )
        
        assert estimated_samples >= validator.min_sample_size
    
    def test_different_metric_keys(self):
        """Test validation with different metric keys"""
        validator = ParameterStatisticalValidator()
        
        baseline = [{"win_rate": 0.5}, {"win_rate": 0.55}, {"win_rate": 0.48}, 
                    {"win_rate": 0.52}, {"win_rate": 0.51}, {"win_rate": 0.49},
                    {"win_rate": 0.53}, {"win_rate": 0.5}, {"win_rate": 0.54}, 
                    {"win_rate": 0.52}]
        
        new_values = [{"win_rate": 0.6}, {"win_rate": 0.65}, {"win_rate": 0.62}, 
                      {"win_rate": 0.63}, {"win_rate": 0.59}, {"win_rate": 0.61},
                      {"win_rate": 0.64}, {"win_rate": 0.62}, {"win_rate": 0.6}, 
                      {"win_rate": 0.63}]
                     
        result = validator.validate_parameter_change(baseline, new_values, metric_key="win_rate")
        
        assert result["is_significant"] == True
        assert result["p_value"] < 0.05
