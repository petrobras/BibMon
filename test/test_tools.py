#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 23:38:16 2020

@author: afranio
"""

import bibmon
import pandas as pd
import pytest

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

def test_calculate_timestamps_with_preserve_periods():
    df = pd.DataFrame({
        'var1': range(100),
        'var2': range(100, 200)
    }, index=pd.date_range('2021-01-01', periods=50, freq='h').append(
        pd.date_range('2021-01-03', periods=50, freq='h')))
    
    start_train, end_train, end_validation, end_test = bibmon.calculate_timestamps(
        df, train_pct=0.6, validation_pct=0.2, test_pct=0.2, preserve_periods=True, time_tolerance='1h'
    )
    
    assert end_train == df.index[99], "Train end does not preserve periods correctly"

def test_calculate_timestamps_invalid_percentages():
    df = pd.DataFrame({
        'var1': range(10),
        'var2': range(10, 20)
    }, index=pd.date_range('2021-01-01', periods=10, freq='h'))

    with pytest.raises(ValueError, match="Train, validation, and test percentages must add up to 1"):
        bibmon.calculate_timestamps(df, train_pct=0.5, validation_pct=0.3, test_pct=0.3)

def test_calculate_timestamps_non_datetime_index():
    df = pd.DataFrame({
        'var1': range(10),
        'var2': range(10, 20)
    })

    with pytest.raises(ValueError, match="The dataframe index must be a DatetimeIndex"):
        bibmon.calculate_timestamps(df, train_pct=0.6, validation_pct=0.2, test_pct=0.2)

def test_calculate_timestamps_small_time_tolerance():
    df = pd.DataFrame({
        'var1': range(100),
        'var2': range(100, 200)
    }, index=pd.date_range('2021-01-01', periods=50, freq='1min').append(
        pd.date_range('2021-01-01 01:00:00', periods=50, freq='1min')))
    
    start_train, end_train, end_validation, end_test = bibmon.calculate_timestamps(
        df, train_pct=0.6, validation_pct=0.2, test_pct=0.2, preserve_periods=True, time_tolerance='1min'
    )
    
    assert end_train == df.index[49], "Train end timestamp is incorrect"