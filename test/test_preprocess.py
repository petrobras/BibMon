#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 00:04:38 2020

@author: afranio
"""

import pandas as pd
import bibmon

#%%
def test_preprocess():

    # load data
    data = bibmon.load_real_data()
    data = data.apply(pd.to_numeric, errors='coerce')
    
    # define functions to be tested
    
    pp = bibmon.PreProcess()
    
    funcs = [f for f in dir(pp) if (f[0] != '_' and callable(getattr(pp,f)))]
    funcs = [f for f in funcs if f not in ['apply','back_to_units']]
    
    # define training and test sets
        
    (X_train, _, 
     X_test, Y_train, 
     _, Y_test) = bibmon.train_val_test_split(data, 
                                           start_train = '2017-12-24T12:00', 
                                           end_train = '2018-01-01T00:00', 
                                           end_validation = '2018-01-02T00:00', 
                                           end_test = '2018-01-04T00:00',
                                           tags_Y = 'tag100')
                                                         
    for f in funcs:
        
        print(f)
        
        ppX = bibmon.PreProcess(f_pp = [f])
        ppY = bibmon.PreProcess(f_pp = [f], is_Y = True)

        X_train_proc = ppX.apply(X_train, train_or_test = 'train')
        Y_train_proc = ppY.apply(Y_train, train_or_test = 'train')

        assert(isinstance(X_train_proc, pd.DataFrame))
        assert(isinstance(Y_train_proc, pd.DataFrame))
        assert(Y_train_proc.columns == Y_train.columns)
        
        X_test_proc = ppX.apply(X_test, train_or_test = 'test')
        Y_test_proc = ppY.apply(Y_test, train_or_test = 'test')
      
        assert(isinstance(X_test_proc, pd.DataFrame))
        assert(isinstance(Y_test_proc, pd.DataFrame))
        assert(Y_test_proc.columns == Y_test.columns)
        
        if 'nan' in f:
            assert(X_train_proc.isnull().sum().sum()==0)
            assert(Y_train_proc.isnull().sum().sum()==0)
            assert(X_test_proc.isnull().sum().sum()==0)
            assert(Y_test_proc.isnull().sum().sum()==0)