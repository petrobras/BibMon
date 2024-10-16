#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 23:38:16 2020

@author: afranio
"""

import bibmon
import pandas as pd

import bibmon.three_w

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


def test_find_df_transitions():
    data = bibmon.load_real_data()

    transitions = bibmon._bibmon_tools.find_df_transitions(data, 1, "number", "tag101")

    assert transitions == [99, 101, 102, 103, 104, 106, 107, 108, 243]

def test_split_df_percentages():
    data = bibmon.load_real_data()

    splitted = bibmon._bibmon_tools.split_df_percentages(data, [0.6, 0.2, 0.2])

    assert splitted[0].shape[0] == 1901
    assert splitted[1].shape[0] == 633
    assert splitted[2].shape[0] == 633

def test_split_df_percentages_error():
    data = bibmon.load_real_data()

    try:
        _ = bibmon._bibmon_tools.split_df_percentages(data, [0.6, 0.2])
    except ValueError:
        assert True

def test_3w_load_dataset_ini():
    config = bibmon.three_w.tools.load_dataset_ini()

    assert config["VERSION"]["DATASET"] == "2.0.0"

def test_3w_split_dataset():
    data, conf, _ = bibmon.load_3w()

    train_df, validation_df, test_df = bibmon.three_w.tools.split_dataset(data, conf)
    
    assert (train_df.shape[0], validation_df.shape[0], test_df.shape[0]) == (85112, 21278, 136746)