#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 23:38:16 2020

@author: afranio
"""

import bibmon
import pandas as pd
import pytest
import numpy as np
from datetime import datetime
from bibmon import comparative_table
from sklearn.metrics import r2_score, mean_absolute_error

def test_complete_analysis():
    
    # load data
    data = bibmon.load_real_data()
    data = data.apply(pd.to_numeric, errors='coerce')
    
    # preprocessing pipeline
    
    preproc_tr = ['remove_empty_variables',
                  'ffill_nan',
                  'remove_frozen_variables',
                  'normalize']
    
    preproc_ts = ['ffill_nan','normalize']
    
    # define training set
        
    (X_train, X_validation, 
     X_test, Y_train, 
     Y_validation, Y_test) = bibmon.train_val_test_split(data, 
                                            start_train = '2017-12-24T12:00', 
                                            end_train = '2018-01-01T00:00', 
                                           end_validation = '2018-01-02T00:00', 
                                            end_test = '2018-01-04T00:00',
                                            tags_Y = 'tag100')
                                                         
    # define the model
                                                         
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    
    model = bibmon.sklearnRegressor(reg)                                                          

    # define regression metrics
                                                         
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_absolute_error
    
    mtr = [r2_score, mean_absolute_error]
                           
    # complete analysis!
                              
    bibmon.complete_analysis(model, X_train, X_validation, X_test, 
                            Y_train, Y_validation, Y_test,
                            f_pp_train = preproc_tr,
                            f_pp_test = preproc_ts,
                            metrics = mtr, 
                            count_window_size = 3, count_limit = 2,
                            fault_start = '2018-01-02 06:00:00',
                            fault_end = '2018-01-02 09:00:00') 
    
    model.plot_importances()                                                                             

# Fixtures for test data
@pytest.fixture
def sample_data():
    """Generate synthetic data for training, validation and testing."""
    X_train = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100)
    })
    Y_train = pd.Series(np.random.randn(100))
    X_validation = pd.DataFrame({
        'feature1': np.random.randn(50),
        'feature2': np.random.randn(50)
    })
    Y_validation = pd.Series(np.random.randn(50))
    X_test = pd.DataFrame({
        'feature1': np.random.randn(30),
        'feature2': np.random.randn(30)
    })
    Y_test = pd.Series(np.random.randn(30))
    return X_train, X_validation, X_test, Y_train, Y_validation, Y_test

@pytest.fixture
def model_with_y():
    """Mock model with Y variable (regression)."""
    class MockModel:
        def __init__(self):
            self.has_Y = True
            self.name = "Model with Y"
            self.lim_conf = 0.99
            self.Y_train_orig = None
            self.X_train_orig = None
            self.Y_train_pred_orig = None
            self.X_train_pred_orig = None
            self.train_time = 0.0
            self.test_time = 0.0
            self.Y_test_orig = None
            self.Y_test_pred_orig = None
            self.X_test_orig = None
            self.X_test_pred_orig = None
            self.alarms = {}
        def predict(self, X, Y=None, *args, **kwargs):
            pred = pd.Series(np.random.randn(len(X)), index=X.index)
            if Y is not None:
                self.Y_test_orig = Y
                self.Y_test_pred_orig = pred
            self.X_test_orig = X
            self.X_test_pred_orig = pred
            self.test_time = 0.1
            return pred
        def fit(self, X_train, Y_train, f_pp=None, a_pp=None, f_pp_test=None, a_pp_test=None, lim_conf=0.99, redefine_limit=False):
            self.lim_conf = lim_conf
            self.Y_train_orig = Y_train
            self.X_train_orig = X_train
            self.Y_train_pred_orig = pd.Series(np.random.randn(len(Y_train)), index=Y_train.index)
            self.X_train_pred_orig = pd.Series(np.random.randn(len(X_train)), index=X_train.index)
            self.train_time = 0.1
            return self
    return MockModel()

@pytest.fixture
def model_without_y():
    """Mock model without Y variable (reconstruction)."""
    class MockModel:
        def __init__(self):
            self.has_Y = False
            self.name = "Model without Y"
            self.lim_conf = 0.99
            self.X_train_orig = None
            self.X_train_pred_orig = None
            self.train_time = 0.0
            self.test_time = 0.0
            self.X_test_orig = None
            self.X_test_pred_orig = None
            self.alarms = {}
        def predict(self, X, Y=None, *args, **kwargs):
            pred = pd.DataFrame(np.random.randn(*X.shape), index=X.index, columns=X.columns)
            self.X_test_orig = X
            self.X_test_pred_orig = pred
            self.test_time = 0.1
            return pred
        def fit(self, X_train, Y_train, f_pp=None, a_pp=None, f_pp_test=None, a_pp_test=None, lim_conf=0.99, redefine_limit=False):
            self.lim_conf = lim_conf
            self.X_train_orig = X_train
            self.X_train_pred_orig = pd.DataFrame(np.random.randn(*X_train.shape), index=X_train.index, columns=X_train.columns)
            self.train_time = 0.1
            return self
    return MockModel()

# MC/DC Test Cases
def test_comparative_table_with_y_and_metrics(sample_data, model_with_y):
    """Test Case 1: C1=True, C2=True, C3=False - Model with Y and metrics."""
    X_train, X_validation, X_test, Y_train, Y_validation, Y_test = sample_data
    metrics = [r2_score, mean_absolute_error]
    X_pred_to_plot = None
    result = comparative_table(
        models=[model_with_y],
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        Y_train=Y_train,
        Y_validation=Y_validation,
        Y_test=Y_test,
        metrics=metrics,
        X_pred_to_plot=X_pred_to_plot,
        plot_SPE=False,
        plot_predictions=False
    )
    assert len(result) >= 1
    assert any('mean_absolute_error' in str(df) for df in result)

def test_comparative_table_without_y_with_xpred(sample_data, model_without_y):
    """Test Case 2: C1=True, C2=False, C3=True - Model without Y, with X_pred_to_plot."""
    X_train, X_validation, X_test, Y_train, Y_validation, Y_test = sample_data
    metrics = [r2_score, mean_absolute_error]
    X_pred_to_plot = 'feature1'
    result = comparative_table(
        models=[model_without_y],
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        metrics=metrics,
        X_pred_to_plot=X_pred_to_plot,
        plot_SPE=False,
        plot_predictions=False
    )
    assert len(result) >= 1
    assert any('mean_absolute_error' in str(df) for df in result)

def test_comparative_table_without_y_without_xpred(sample_data, model_without_y):
    """Test Case 3: C1=True, C2=False, C3=False - Model without Y and without X_pred_to_plot."""
    X_train, X_validation, X_test, Y_train, Y_validation, Y_test = sample_data
    metrics = [r2_score, mean_absolute_error]
    X_pred_to_plot = None
    result = comparative_table(
        models=[model_without_y],
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        metrics=metrics,
        X_pred_to_plot=X_pred_to_plot,
        plot_SPE=False,
        plot_predictions=False,
        times=True
    )
    assert len(result) == 1
    assert 'Train' in result[0].columns

def test_comparative_table_without_metrics(sample_data, model_with_y):
    """Test Case 4: C1=False - Without metrics, only time table."""
    X_train, X_validation, X_test, Y_train, Y_validation, Y_test = sample_data
    metrics = None
    X_pred_to_plot = None
    result = comparative_table(
        models=[model_with_y],
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        Y_train=Y_train,
        Y_validation=Y_validation,
        Y_test=Y_test,
        metrics=metrics,
        X_pred_to_plot=X_pred_to_plot,
        plot_SPE=False,
        plot_predictions=False
    )
    assert len(result) == 1
    assert 'Train' in result[0].columns

def test_comparative_table_with_fault_period(sample_data, model_with_y):
    """Test Case 5: C4=True, C5=True - Fault with defined start and end."""
    X_train, X_validation, X_test, Y_train, Y_validation, Y_test = sample_data
    metrics = [r2_score, mean_absolute_error]
    fault_start = '2023-01-01 00:00:00'
    fault_end = '2023-01-02 00:00:00'
    result = comparative_table(
        models=[model_with_y],
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        Y_train=Y_train,
        Y_validation=Y_validation,
        Y_test=Y_test,
        metrics=metrics,
        fault_start=fault_start,
        fault_end=fault_end,
        plot_SPE=False,
        plot_predictions=False
    )
    assert len(result) >= 2
    assert any('FDR' in str(df) for df in result)
    assert any('FAR' in str(df) for df in result)

def test_comparative_table_with_fault_start_only(sample_data, model_with_y):
    """Test Case 6: C4=True, C5=False - Fault with only start defined."""
    X_train, X_validation, X_test, Y_train, Y_validation, Y_test = sample_data
    metrics = [r2_score, mean_absolute_error]
    fault_start = '2023-01-01 00:00:00'
    fault_end = None
    result = comparative_table(
        models=[model_with_y],
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        Y_train=Y_train,
        Y_validation=Y_validation,
        Y_test=Y_test,
        metrics=metrics,
        fault_start=fault_start,
        fault_end=fault_end,
        plot_SPE=False,
        plot_predictions=False
    )
    assert len(result) >= 2
    assert any('FDR' in str(df) for df in result)
    assert any('FAR' in str(df) for df in result)

def test_comparative_table_with_mask(sample_data, model_with_y):
    """Test Case 7: C4=False, C6=False - With detection mask."""
    X_train, X_validation, X_test, Y_train, Y_validation, Y_test = sample_data
    metrics = [r2_score, mean_absolute_error]
    mask = np.array([0, 1, 1, 0, 1])
    model_with_y.fit(X_train, Y_train)
    result = comparative_table(
        models=[model_with_y],
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        Y_train=Y_train,
        Y_validation=Y_validation,
        Y_test=Y_test,
        metrics=metrics,
        mask=mask,
        plot_SPE=False,
        plot_predictions=False,
        fit_model=False
    )
    assert len(result) >= 2
    assert any('FDR' in str(df) for df in result)
    assert any('FAR' in str(df) for df in result)

def test_comparative_table_without_fault_and_mask(sample_data, model_with_y):
    """Test Case 8: C4=False, C6=True - Without fault and mask, only prediction table."""
    X_train, X_validation, X_test, Y_train, Y_validation, Y_test = sample_data
    metrics = [r2_score, mean_absolute_error]
    fault_start = None
    fault_end = None
    mask = None
    result = comparative_table(
        models=[model_with_y],
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        Y_train=Y_train,
        Y_validation=Y_validation,
        Y_test=Y_test,
        metrics=metrics,
        fault_start=fault_start,
        fault_end=fault_end,
        mask=mask,
        plot_SPE=False,
        plot_predictions=False,
        times=False
    )
    assert len(result) == 1
    assert 'Train' in result[0].columns
    assert 'Validation' in result[0].columns
    assert 'Test' in result[0].columns

def test_detect_drift_bias():
    """Test for drift/bias detection in a time series."""
    from bibmon import _alarms
    # Time series with clear drift
    data = np.concatenate([np.ones(50), np.ones(50)*10])
    window = 10
    threshold = 2.0
    # The function should return 1 (or True) if drift/bias is detected
    alarm = _alarms.detect_drift_bias(data, window=window, threshold=threshold)
    assert alarm == 1 or alarm is True

def test_detect_nelson_rule1():
    """Test for Nelson Rule 1: one point above 3 standard deviations from the mean."""
    from bibmon import _alarms
    import numpy as np
    # Series with one outlier above 3 sigma
    data = np.concatenate([np.zeros(20), np.array([10]), np.zeros(20)])
    # The function should return 1 (or True) if Nelson Rule 1 is detected
    alarm = _alarms.detect_nelson_rule1(data)
    assert alarm == 1 or alarm is True

def test_detect_nelson_rule2():
    """Test for Nelson Rule 2: nine consecutive points on the same side of the mean."""
    from bibmon import _alarms
    import numpy as np
    # Series with nine consecutive points above the mean
    data = np.concatenate([np.zeros(10), np.ones(9)*5, np.zeros(10)])
    # The function should return 1 (or True) if Nelson Rule 2 is detected
    alarm = _alarms.detect_nelson_rule2(data)
    assert alarm == 1 or alarm is True

def test_detect_nelson_rule3():
    """Test for Nelson Rule 3: six consecutive points all increasing or all decreasing."""
    from bibmon import _alarms
    import numpy as np
    # Series with six consecutive increasing values
    data = np.concatenate([np.zeros(10), np.arange(1, 7), np.zeros(10)])
    # The function should return 1 (or True) if Nelson Rule 3 is detected
    alarm = _alarms.detect_nelson_rule3(data)
    assert alarm == 1 or alarm is True

def test_detect_nelson_rule4():
    """Test for Nelson Rule 4: fourteen points in a row alternating up and down."""
    from bibmon import _alarms
    import numpy as np
    # Series with fourteen points alternating above and below the mean
    data = np.array([1, -1] * 7 + [0]*10)  # 14 alternations, then zeros
    # The function should return 1 (or True) if Nelson Rule 4 is detected
    alarm = _alarms.detect_nelson_rule4(data)
    assert alarm == 1 or alarm is True

def test_detect_nelson_rule5():
    """Test for Nelson Rule 5: two out of three consecutive points above 2 standard deviations from the mean, all on the same side."""
    from bibmon import _alarms
    import numpy as np
    # Series with three points far above +2 sigma
    data = np.concatenate([np.ones(30), np.array([30, 35, 40]), np.ones(30)])
    # The function should return 1 (or True) if Nelson Rule 5 is detected
    alarm = _alarms.detect_nelson_rule5(data)
    assert alarm == 1 or alarm is True

def test_detect_nelson_rule6():
    """Test for Nelson Rule 6: four out of five consecutive points above 1 standard deviation from the mean, all on the same side."""
    from bibmon import _alarms
    import numpy as np
    # Series with five points far above +1 sigma
    data = np.concatenate([np.ones(30), np.array([10, 12, 14, 16, 18]), np.ones(30)])
    # The function should return 1 (or True) if Nelson Rule 6 is detected
    alarm = _alarms.detect_nelson_rule6(data)
    assert alarm == 1 or alarm is True

def test_detect_nelson_rule7():
    """Test for Nelson Rule 7: fifteen consecutive points within 1 standard deviation of the mean, in both directions."""
    from bibmon import _alarms
    import numpy as np
    # Series with 15 points clearly within 1 sigma of the mean
    data = np.concatenate([np.ones(10), np.array([0.8, 1.2, 0.9, 1.1, 0.7, 1.3, 0.6, 1.4, 0.5, 1.5, 0.4, 1.6, 0.3, 1.7, 0.2]), np.ones(10)])
    # The function should return 1 (or True) if Nelson Rule 7 is detected
    alarm = _alarms.detect_nelson_rule7(data)
    assert alarm == 1 or alarm is True

def test_detect_nelson_rule8():
    """Test for Nelson Rule 8: eight consecutive points outside 1 standard deviation of the mean, all on the same side."""
    from bibmon import _alarms
    import numpy as np
    # Series with eight consecutive points above +1 sigma
    data = np.concatenate([np.ones(30), np.array([5, 6, 7, 8, 9, 10, 11, 12]), np.ones(30)])
    # The function should return 1 (or True) if Nelson Rule 8 is detected
    alarm = _alarms.detect_nelson_rule8(data)
    assert alarm == 1 or alarm is True

def test_detect_variance_change():
    """Test for sudden variance change detection."""
    from bibmon import _alarms
    import numpy as np
    # Series with sudden variance change
    data = np.concatenate([np.random.normal(0, 0.1, 50), np.random.normal(0, 2.0, 50)])
    # The function should return 1 (or True) if variance change is detected
    alarm = _alarms.detect_variance_change(data, window_size=20, threshold=1.5)
    assert alarm == 1 or alarm is True

def test_detect_outlier_frequency_change():
    """Test for outlier frequency change detection."""
    from bibmon import _alarms
    import numpy as np
    # Series with change in outlier frequency
    data = np.concatenate([np.random.normal(0, 1, 50), np.random.normal(0, 1, 50) + np.random.choice([0, 5], 50, p=[0.8, 0.2])])
    # The function should return 1 (or True) if outlier frequency change is detected
    alarm = _alarms.detect_outlier_frequency_change(data, window_size=20, threshold=0.1)
    assert alarm == 1 or alarm is True
    