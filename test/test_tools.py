#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 23:38:16 2020

@author: afranio
"""

import bibmon
import pandas as pd
import os
import pickle

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

    # Filename for saving the model

    model_filename = 'test_model.pkl'
                           
    # complete analysis!
                              
    bibmon.complete_analysis(model, X_train, X_validation, X_test, 
                            Y_train, Y_validation, Y_test,
                            f_pp_train = preproc_tr,
                            f_pp_test = preproc_ts,
                            metrics = mtr, 
                            count_window_size = 3, count_limit = 2,
                            fault_start = '2018-01-02 06:00:00',
                            fault_end = '2018-01-02 09:00:00',
                            save_model=True,
                            model_filename=model_filename) 
    
    model.plot_importances()

    assert os.path.exists(model_filename), "Model file was not saved."

    loaded_model = bibmon.load_model(model_filename)

    assert isinstance(loaded_model, type(model)), "Loaded model is not the same type as the original model."

    if os.path.exists(model_filename):
        os.remove(model_filename)