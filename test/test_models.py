#%%
import bibmon
import pandas as pd
import numpy as np

#%%

def test_models_with_df_inputs():

    # load data
    data = bibmon.load_real_data()
    data = data.apply(pd.to_numeric, errors='coerce')
    
    # preprocessing pipeline
    f_pp_tr = ['remove_empty_variables',
               'ffill_nan',
               'remove_frozen_variables',
               'normalize']
    
    f_pp_ts = ['ffill_nan','normalize']
    
    # define training set
    (X_train, X_validation, 
     X_test, Y_train, 
     Y_validation, Y_test) = bibmon.train_val_test_split(data, 
                                           start_train = '2017-12-24T12:00', 
                                           end_train = '2018-01-01T00:00', 
                                           end_validation = '2018-01-02T00:00', 
                                           end_test = '2018-01-04T00:00',
                                           tags_Y = 'tag100')
    
    # running the unit tests
                                                             
    for attr in bibmon.__all__:             
        a = getattr(bibmon,attr)     
        if isinstance(a, type):         
            if a.__base__ == bibmon._generic_model.GenericModel:   
                if a == bibmon.sklearnRegressor:                 
                    from sklearn.linear_model import LinearRegression
                    m = a(LinearRegression())
                else:                    
                    m = a()        
                              
                # TRAINING
                    
                if m.has_Y:    
                    # X DF with one column
                    m.fit(X_train.loc[:,['tag102']], Y_train, 
                          f_pp = f_pp_tr,
                          delete_training_data = False)
                    
                    # X series with one column
                    m.fit(X_train.loc[:,['tag102']].T.squeeze(), Y_train, 
                          f_pp = f_pp_tr,
                          delete_training_data = False)
                    
                # X complete DF
                m.fit(X_train, Y_train, 
                      f_pp = f_pp_tr,
                      delete_training_data = False)
                                            
                dfs_train = [m.X_train, m.X_train_orig]
                series_train = [m.SPE_train]                

                if m.has_Y:
                    dfs_train.extend([m.Y_train_pred, m.Y_train_pred_orig])
                else:
                    dfs_train.extend([m.X_train_pred, m.X_train_pred_orig])
                       
                if m.has_Y:
                    dfs_train.extend([m.Y_train, m.Y_train_orig])
                
                for df in dfs_train:
                    assert(isinstance(df, pd.DataFrame))
                    
                for s in series_train:
                    assert(isinstance(s, pd.Series))
                    
                assert(isinstance(m.limSPE, float))

                # PREDICTION 1 - VALIDATION AND REDEFINITION OF LIMIT
                
                m.predict(X_validation, Y_validation, redefine_limit = True,
                          f_pp = f_pp_ts)
                
                dfs_val = [m.X_test, m.X_test_orig]
                series_val = [m.SPE_test]                

                if m.has_Y:
                    dfs_val.extend([m.Y_test_pred, m.Y_test_pred_orig])
                else:
                    dfs_val.extend([m.X_test_pred, m.X_test_pred_orig])
                       
                if m.has_Y:
                    dfs_val.extend([m.Y_test, m.Y_test_orig])
                
                for df in dfs_val:
                    assert(isinstance(df, pd.DataFrame))
                    
                for s in series_val:
                    assert(isinstance(s, pd.Series))
                
                # PREDICTION 2 - TEST WITH WINDOWS FOR ALARMS
                    
                m.predict(X_test, Y_test,                           
                          count_window_size = 10,
                          redefine_limit = False, count_limit = 5, 
                          f_pp = f_pp_ts)        
                                
                series_alarms = [m.alarmOutlier, m.alarmCount]

                for s in series_alarms:
                    assert(isinstance(s, pd.Series))
                    
                # PREDICTION 3 - TEST WITH A POINT (SERIES AND DF)

                f_pp_ts =['replace_nan_with_values']
                a_pp_ts = {'replace_nan_with_values__val': X_train.median()}
        
                # SERIES
                m.predict(X_test.iloc[0:1], Y_test.iloc[0:1],
                          f_pp = f_pp_ts,
                          a_pp = a_pp_ts)
                
                # DF
                m.predict(X_test.iloc[0:1], Y_test.iloc[0:1],
                          f_pp = f_pp_ts,
                          a_pp = a_pp_ts)               

#%%
                
def test_models_with_np_array_inputs():

    # load data
    train_df, test_df = bibmon.load_tennessee_eastman(train_id = 0, 
                                                      test_id = 1)
    
    X_train = train_df.drop('XMEAS(35)',axis=1)
    Y_train = train_df['XMEAS(35)']

    X_test = test_df.drop('XMEAS(35)',axis=1)
    Y_test = test_df['XMEAS(35)']

    # preprocessing pipeline    
    f_pp = ['normalize']
    
    for attr in bibmon.__all__:             
        a = getattr(bibmon,attr)     
        if isinstance(a, type):         
            if a.__base__ == bibmon._generic_model.GenericModel:   
                if a == bibmon.sklearnRegressor:                 
                    from sklearn.linear_model import LinearRegression
                    m = a(LinearRegression())
                else:                    
                    m = a()        
                              
                # TRAINING

                if m.has_Y:    
                    # X with one column
                    m.fit(np.array(X_train.iloc[:,0]), np.array(Y_train), 
                          f_pp = f_pp,
                          delete_training_data = False)
                    
                # complete X
                m.fit(np.array(X_train), np.array(Y_train), 
                      f_pp = f_pp,
                      delete_training_data = False)
    
                dfs_train = [m.X_train, m.X_train_orig]
                series_train = [m.SPE_train]                

                if m.has_Y:
                    dfs_train.extend([m.Y_train_pred, m.Y_train_pred_orig])
                else:
                    dfs_train.extend([m.X_train_pred, m.X_train_pred_orig])
                       
                if m.has_Y:
                    dfs_train.extend([m.Y_train, m.Y_train_orig])
                
                for df in dfs_train:
                    assert(isinstance(df, pd.DataFrame))
                    
                for s in series_train:
                    assert(isinstance(s, pd.Series))
                    
                assert(isinstance(m.limSPE, float))  
                
                if isinstance(m, bibmon.PCA):
                    m.plot_cumulative_variance()
                
                # TEST
                
                m.predict(np.array(X_test), np.array(Y_test), 
                          f_pp = f_pp)
                
                dfs_val = [m.X_test, m.X_test_orig]
                series_val = [m.SPE_test]                

                if m.has_Y:
                    dfs_val.extend([m.Y_test_pred, m.Y_test_pred_orig])
                else:
                    dfs_val.extend([m.X_test_pred, m.X_test_pred_orig])
                       
                if m.has_Y:
                    dfs_val.extend([m.Y_test, m.Y_test_orig])
                
                for df in dfs_val:
                    assert(isinstance(df, pd.DataFrame))
                    
                for s in series_val:
                    assert(isinstance(s, pd.Series))
                    
                # PREDICTION 3 - TEST WITH A POINT (SERIES AND DF)

                f_pp_ts =['normalize','replace_nan_with_values']
                a_pp_ts = {'replace_nan_with_values__val': X_train.median()}
        
                # SERIES
                m.predict(np.array(X_test.iloc[0]), np.array(Y_test.iloc[0]),
                          f_pp = f_pp_ts,
                          a_pp = a_pp_ts)
                
                # DF
                m.predict(np.array(X_test.iloc[0:1]), 
                          np.array(Y_test.iloc[0:1]),
                          f_pp = f_pp_ts,
                          a_pp = a_pp_ts)           
                