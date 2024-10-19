import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

###############################################################################

def create_df_with_dates (array, 
                          start = '2020-01-01 00:00:00', 
                          freq = '1min'):
    """
    Parameters
    ----------
    array: pandas.DataFrame or numpy.array
        Original data.
    start: string, optional
        Start timestamp.
    freq: string, optional
        Sampling interval.
    Returns
    ----------                
    df: pandas.DataFrame
        Processed data.
    """    
    df = pd.DataFrame(array)
    df.index = pd.date_range(start = start, 
                             periods = df.shape[0],
                             freq=freq)
    return df

###############################################################################

def create_df_with_noise (array, 
                          noise_frac, 
                          max_index_for_noise):
    """
    Adds artificial measurement noise to data.
    
    Parameters
    ----------
    array: pandas.DataFrame or numpy.array
        Original data.
    noise_frac: float 
        Fraction (between 0 and 1) of the total amplitude of the variable
        that will be used as the noise standard deviation.
    max_index_for_noise: int
        Maximum index to consider the amplitude 
        in the standard deviation calculation.
    Returns
    ----------                
    df: pandas.DataFrame
        Processed data.
    """    

    df = pd.DataFrame(array)

    sigma_noise = noise_frac*(np.amax(array[:max_index_for_noise,:], 
                                      axis=0)-
                              np.amin(array[:max_index_for_noise,:], 
                                      axis=0))

    for i in range(df.shape[1]):
        df.iloc[:,i] = (df.iloc[:,i] + 
                        sigma_noise[i]*np.random.randn(df.shape[0]))
        
    return df
    
###############################################################################

def align_dfs_by_rows (df1, df2):
    """
    Aligns DataFrames by rows.

    Parameters
    ----------
    df1, df2: pandas.DataFrame
        Original data.
    Returns
    ----------                
    new_df1, new_df2: pandas.DataFrame
        Processed data.
    """
    
    new_df1 = df1.loc[df1.index.isin(df2.index)]
    new_df2 = df2.loc[df2.index.isin(df1.index)]

    return new_df1, new_df2

###############################################################################

def spearmanr_dendrogram(df, figsize = (18,8)):
    """
    Generates a dendrogram of Spearman correlations.
    
    Parameters
    ----------
    df: pandas.DataFrame
        Dados to be analyzed.
    figsize: tuple of ints, optional
        Figure dimensions.
    """   
    import scipy.cluster.hierarchy
    import scipy.stats
    
    df = pd.DataFrame(df)
    
    corr = np.round(scipy.stats.spearmanr(df).correlation, 4)
    corr_condensed = scipy.cluster.hierarchy.distance.squareform(1-corr)
    z = scipy.cluster.hierarchy.linkage(corr_condensed, method='average')
    plt.figure(figsize=figsize)
    scipy.cluster.hierarchy.dendrogram(z, labels = df.columns.tolist(), 
                                       orientation='left', leaf_font_size=16)
    plt.xlabel('Spearman Correlation')
    plt.ylabel('Variable clusters')
    plt.show()

###############################################################################

def train_val_test_split (data, start_train, end_train, 
                          end_validation, end_test, 
                          tags_X = None, tags_Y = None):
    """
    Separates the data into consecutive portions of 
    train, validation, and test, returning 3 DataFrames.
    It can also separate into predictor variables (X) and 
    predicted variables (Y), which in this case will return 6 DataFrames.
        
    Parameters
    ----------
    data: pandas.DataFrame
        Data to be separated.
    start_train: string
        Start timestamp of the train portion.
    end_train: string
        End timestamp of the train portion.
    end_validation: string
        End timestamp of the validation portion.
    end_test: string
        End timestamp of the test portion.
    tags_X: list of strings
        Variables to be considered in the X set.
    tags_Y: list of strings
        Variables to be considered in the Y set.
    Returns
    ----------                
    : pandas.DataFrames
        Separated data.
    """               
    train_data = data.loc[start_train:end_train]
    validation_data = data.loc[end_train:end_validation].iloc[1:,:]
    test_data = data.loc[end_validation:end_test].iloc[1:,:]

    if tags_Y is not None:

        if not isinstance(tags_Y, list): tags_Y = [tags_Y]
        
        train_data_Y = train_data.loc[:,tags_Y]
        validation_data_Y = validation_data.loc[:,tags_Y]
        test_data_Y = test_data.loc[:,tags_Y]

        if tags_X is not None:

            if not isinstance(tags_X, list): tags_X = [tags_X]
            
            train_data_X = train_data.loc[:,tags_X]
            validation_data_X = validation_data.loc[:,tags_X]
            test_data_X = test_data.loc[:,tags_X]

        else:

            dif = train_data.columns.difference(train_data_Y.columns)
            train_data_X = train_data[dif]
            validation_data_X = validation_data[dif]
            test_data_X = test_data[dif]
            
        return (train_data_X, validation_data_X, test_data_X, 
                train_data_Y, validation_data_Y, test_data_Y)
            
    else:
        
        if tags_X is not None:

            if not isinstance(tags_X, list): tags_Y = [tags_X]
            
            train_data = train_data.loc[:,tags_X]
            validation_data = validation_data.loc[:,tags_X]
            test_data = test_data.loc[:,tags_X]           
            
        return (train_data, validation_data, test_data)
            
###############################################################################

def complete_analysis (model, X_train, X_validation, X_test, 
                       Y_train = None , Y_validation = None, Y_test = None,
                       lim_conf = 0.99,
                       f_pp_train = ['remove_empty_variables',
                                     'ffill_nan',
                                     'remove_frozen_variables',
                                     'normalize'],
                       a_pp_train = None,
                       f_pp_test = ['replace_nan_with_values',
                                    'normalize'],
                       a_pp_test = None,
                       logy = True, 
                       metrics = None, 
                       X_pred_to_plot = None,
                       count_limit = 1,
                       count_window_size = 0,
                       fault_start = None,
                       fault_end = None):
    """
    Performs a complete monitoring analysis, with train, validation, and test.

    Parameters
    ----------
    model: BibMon model
        Model to be considered in the analysis.
    X_train: pandas.DataFrame or numpy.array
        Training data X.
    X_validation: pandas.DataFrame or numpy.array
        Validation data X.
    X_test: pandas.DataFrame or numpy.array
        Test data X.
    Y_train: pandas.DataFrame or numpy.array, optional
        Training data Y.
    Y_validation: pandas.DataFrame or numpy.array, optional
        Validation data Y.
    Y_test: pandas.DataFrame or numpy.array, optional
        Test data Y.
    lim_conf: float, optional
        Confidence limit for the detection index.
    f_pp_train: list, optional
        List containing strings with names of functions to be used 
        in pre-processing the train data (the functions are defined in the 
        PreProcess class, in the BibMon_Tools.py file).
    a_pp_train: dict, optional
        Dictionary containing the parameters to be provided
        to each function to perform pre-processing of the train data, in
        the format {'functionname__argname': argvalue, ...}.
    f_pp_test: list, optional
        List containing strings with names of functions to be used 
        in pre-processing the test data (the functions are defined in the 
        PreProcess class, in the BibMon_Tools.py file).
    a_pp_test: dict, optional
        Dictionary containing the parameters to be provided
        to each function to perform pre-processing of the test data, in
        the format {'functionname__argname': argvalue, ...}.
    logy: boolean, optional
        If use logarithmic scale in the SPE plots.
    metrics: list of functions, optional
        Functions for calculating metrics to be displayed in the title of 
        the graph.
    X_pred_to_plot: string, optional
        In case the model is a reconstruction model (i.e., self.has_Y = False),
        indicates which column of X to plot along with the prediction.
    count_limit: int, optional
        Limit of points to be considered in the window 
        for the count alarm to sound.
    count_window_size: int, optional
        Window sizes used in count alarm calculation. 
    fault_start: string, optional
        Start timestamp of the fault.
    fault_end: string, optional
        End timestamp of the fault.
    """               
    fig, ax = plt.subplots(3,2, figsize = (15,12))

    cond_to_plot_pred = (model.has_Y or 
                        ((not model.has_Y) and (X_pred_to_plot is not None)))

    ######## TRAINING ########

    model.fit(X_train, Y_train, 
              f_pp = f_pp_train,
              a_pp = a_pp_train,
              f_pp_test = f_pp_test,
              a_pp_test = a_pp_test,
              lim_conf = lim_conf,
              redefine_limit = False)
                                
    # PLOTTING SPE
                
    model.plot_SPE(ax = ax[0,0], logy = logy)
    ax[0,0].set_title('Training')

    # PLOTTING PREDICTIONS

    if cond_to_plot_pred:
        model.plot_predictions(ax = ax[0,1], train_or_test = 'train', 
                               X_pred_to_plot = X_pred_to_plot,
                               metrics = metrics)

    ######## VALIDATION ########

    model.predict(X_validation, Y_validation, 
                  count_window_size = count_window_size, 
                  redefine_limit = True)

    # PLOTTING SPE

    model.plot_SPE(ax = ax[1,0], train_or_test = 'test', logy = logy)
    ax[1,0].set_title('Validation')

    # PLOTTING PREDICTIONS

    if cond_to_plot_pred:
        model.plot_predictions(ax = ax[1,1], train_or_test = 'test', 
                               X_pred_to_plot = X_pred_to_plot, 
                               metrics = metrics)
        
    ######## TEST ########
        
    model.predict(X_test, Y_test, 
                  count_window_size = count_window_size, 
                  count_limit = count_limit,
                  redefine_limit = False)

    # PLOTTING SPE

    model.plot_SPE(ax = ax[2,0], train_or_test = 'test', logy = logy)
    ax[2,0].set_title('Test')

    if fault_start is not None:
        ax[2,0].axvline(datetime.strptime(str(fault_start),
                                          '%Y-%m-%d %H:%M:%S'), ls = '--')
    if fault_end is not None:
        ax[2,0].axvline(datetime.strptime(str(fault_end),
                                          '%Y-%m-%d %H:%M:%S'), ls = '--')

    # PLOTTING PREDICTIONS

    if cond_to_plot_pred:
        model.plot_predictions(ax = ax[2,1], train_or_test = 'test', 
                               X_pred_to_plot = X_pred_to_plot,
                               metrics = metrics)

        if fault_start is not None:
            ax[2,1].axvline(datetime.strptime(str(fault_start),
                                              '%Y-%m-%d %H:%M:%S'), ls = '--')
        if fault_end is not None:
            ax[2,1].axvline(datetime.strptime(str(fault_end),
                                              '%Y-%m-%d %H:%M:%S'), ls = '--')
        
    fig.tight_layout();
            
##############################################################################

def comparative_table (models, X_train, X_validation, X_test, 
                       Y_train = None , Y_validation = None, Y_test = None,
                       lim_conf = 0.99,
                       f_pp_train = ['remove_empty_variables',
                                     'ffill_nan',
                                     'remove_frozen_variables',
                                     'normalize'],
                       a_pp_train = None,
                       f_pp_test = ['replace_nan_with_values',
                                    'normalize'],
                       a_pp_test = None,
                       logy = True, metrics = None,
                       X_pred_to_plot = None,
                       count_limit = 1,
                       count_window_size = 0,
                       fault_start = None,
                       fault_end = None,
                       mask = None,
                       times = True,
                       plot_SPE = True,
                       plot_predictions = True,
                       fit_model = True):

    """
    Performs complete monitoring analysis of multiple models and builds
    comparative result tables.

    Parameters
    ----------
    models: list of BibMon models
        Models to be considered in the analysis.
    X_train: pandas.DataFrame or numpy.array
        Training data X.
    X_validation: pandas.DataFrame or numpy.array
        Validation data X.
    X_test: pandas.DataFrame or numpy.array
        Test data X.
    Y_train: pandas.DataFrame or numpy.array, optional
        Training data Y.
    Y_validation: pandas.DataFrame or numpy.array, optional
        Validation data Y.
    Y_test: pandas.DataFrame or numpy.array, optional
        Test data Y.
    lim_conf: float, optional
        Confidence limit for the detection index.
    f_pp_train: list, optional
        List containing strings with names of functions to be used 
        in pre-processing the train data (the functions are defined in the 
        PreProcess class, in the BibMon_Tools.py file).
    a_pp_train: dict, optional
        Dictionary containing the parameters to be provided
        to each function to perform pre-processing of the train data, in
        the format {'functionname__argname': argvalue, ...}.
    f_pp_test: list, optional
        List containing strings with names of functions to be used 
        in pre-processing the test data (the functions are defined in the 
        PreProcess class, in the BibMon_Tools.py file).
    a_pp_test: dict, optional
        Dictionary containing the parameters to be provided
        to each function to perform pre-processing of the test data, in
        the format {'functionname__argname': argvalue, ...}.
    logy: boolean, optional
        If use logarithmic scale in the SPE plots.
    metrics: list of functions, optional
        Functions for calculating metrics to be displayed in the title of 
        the graph.
    X_pred_to_plot: string, optional
        In case the model is a reconstruction model (i.e., self.has_Y = False),
        indicates which column of X to plot along with the prediction.
    count_limit: int, optional
        Limit of points to be considered in the window 
        for the count alarm to sound.
    count_window_size: int, optional
        Window sizes used in count alarm calculation.
    fault_start: string, optional
        Start timestamp of the fault.
    fault_end: string, optional
        End timestamp of the fault.
    mask: numpy.array, optional
        Boolean array indicating the indices where the process is
        in fault.
    times: boolean, optional
        If execution times should be calculated.
    plot_SPE: boolean, optional
        If SPE plots should be plotted.
    plot_predictions: boolean, optional
        If prediction plots should be plotted.
    fit_model: boolean, optional
        If models should be trained.
    Returns
    ----------
    : list of pandas.DataFrames
        List with the generated tables (prediction and/or detection).
    """
    
    n = len(models)

    if plot_SPE:
        fig_spe, ax_spe = plt.subplots(n, 3, figsize = (15, 4*n))

    if plot_predictions:
        fig_pred, ax_pred = plt.subplots(n, 3, figsize = (15, 4*n))
        
    train_metrics = {}     
    validation_metrics = {}        
    test_metrics = {}        

    detection_alarms = {}
    false_detection_rates = {}
    false_alarm_rates = {}

    for i in range(len(models)):
        
        model = models[i]
        
        cond_to_plot_pred = (model.has_Y or 
                            ((not model.has_Y) and 
                            (X_pred_to_plot is not None)))

        ######## TRAINING ########
        
        if fit_model:
            model.fit(X_train, Y_train, 
                      f_pp = f_pp_train,
                      a_pp = a_pp_train, 
                      f_pp_test = f_pp_test,
                      a_pp_test = a_pp_test,
                      lim_conf = lim_conf,
                      redefine_limit = False)
        
        # TERMS FOR PREDICTION TABLE
        
        if metrics is not None:
            if model.has_Y:
                true = model.Y_train_orig
                pred = model.Y_train_pred_orig
            else:
                if X_pred_to_plot is not None:
                    true = model.X_train_orig[X_pred_to_plot]
                    pred = model.X_train_pred_orig[X_pred_to_plot]                
            for mr in metrics:
                true, pred = align_dfs_by_rows(true.dropna(), pred)
                train_metrics[model.name+': '+mr.__name__] = mr(true, pred)
                                    
        # PLOTTING SPE
                
        if plot_SPE:
            model.plot_SPE(ax = ax_spe[i,0], logy = logy)
            ax_spe[i,0].set_title('Training')

        # PLOTTING PREDICTIONS
            
        if plot_predictions:
            if cond_to_plot_pred:
                model.plot_predictions(ax = ax_pred[i,0], 
                                        train_or_test = 'train', 
                                        X_pred_to_plot = X_pred_to_plot,
                                        metrics = metrics)

        ######## VALIDATION ########
                
        model.predict(X_validation, Y_validation, 
                      count_window_size = count_window_size, 
                      redefine_limit = True)

        # TERMS FOR PREDICTION TABLE
        
        if metrics is not None:
            if model.has_Y:
                true = model.Y_test_orig
                pred = model.Y_test_pred_orig
            else:
                if X_pred_to_plot is not None:
                    true = model.X_test_orig[X_pred_to_plot]
                    pred = model.X_test_pred_orig[X_pred_to_plot]                
            for mr in metrics:
                true, pred = align_dfs_by_rows(true.dropna(), pred)
                validation_metrics[model.name+': '+mr.__name__]= mr(true, pred)

        # PLOTTING SPE
                
        if plot_SPE:
            model.plot_SPE(ax = ax_spe[i,1], train_or_test = 'test', 
                           logy=logy)
            ax_spe[i,1].set_title(model.name+'\n\nValidation')

        # PLOTTING PREDICTIONS
            
        if plot_predictions:
            if cond_to_plot_pred:
                model.plot_predictions(ax = ax_pred[i,1], 
                                       train_or_test = 'test', 
                                       X_pred_to_plot = X_pred_to_plot, 
                                       metrics = metrics)
                ax_pred[i,1].set_title(model.name+'\n\n'+
                                       ax_pred[i,1].get_title())
                
        ######## TEST ########
                
        model.predict(X_test, Y_test, 
                      count_window_size = count_window_size, 
                      count_limit = count_limit,
                      redefine_limit = False)

        # TERMS FOR PREDICTION TABLE
        
        if metrics is not None:
            if model.has_Y:
                true = model.Y_test_orig
                pred = model.Y_test_pred_orig
            else:
                if X_pred_to_plot is not None:
                    true = model.X_test_orig[X_pred_to_plot]
                    pred = model.X_test_pred_orig[X_pred_to_plot]                
            for mr in metrics:
                true, pred = align_dfs_by_rows(true.dropna(), pred)
                test_metrics[model.name+': '+mr.__name__] =  mr(true, pred)

        # TERMS FOR DETECTION TABLE
        
        if mask is None:
            if fault_start is not None:            
                for key, value in model.alarms.items():
                    if fault_end is not None:
                        detection_alarms[model.name+': '+key] = \
                            value[fault_start:fault_end][:-1].mean()
                        false_detection_rates[model.name+': '+key] = \
                            pd.concat([value[:fault_start][:-1], 
                                    value[fault_end:]]).mean()
                    else:
                        detection_alarms[model.name+': '+key] = \
                            value[fault_start:fault_end].mean()
                        false_detection_rates[model.name+': '+key] = \
                            value[:fault_start][:-1].mean()
        else:
            for key, value in model.alarms.items():
                detection_alarms[model.name+': '+key] = \
                    mask[mask==1].eq(value[value==1]).astype(int).mean()
                false_detection_rates[model.name+': '+key] = \
                    1-mask[mask==0].eq(value[value==0]).astype(int).mean()

        # PLOTTING SPE
                
        if plot_SPE:

            model.plot_SPE(ax = ax_spe[i,2], train_or_test='test',
                           logy = logy)
            ax_spe[i,2].set_title('Test')
            
            if fault_start is not None:
                ax_spe[i,2].axvline(datetime.strptime(str(fault_start),
                                                      '%Y-%m-%d %H:%M:%S'), 
                                                      ls='--')
            if fault_end is not None:
                ax_spe[i,2].axvline(datetime.strptime(str(fault_end),
                                                      '%Y-%m-%d %H:%M:%S'), 
                                                      ls='--')

        # PLOTTING PREDICTIONS
                
        if plot_predictions:
            if cond_to_plot_pred:
                model.plot_predictions(ax = ax_pred[i,2], 
                                        train_or_test = 'test', 
                                        X_pred_to_plot = X_pred_to_plot,
                                        metrics = metrics)
                if fault_start is not None:
                    ax_pred[i,2].axvline(datetime.strptime(str(fault_start),
                                                 '%Y-%m-%d %H:%M:%S'), ls='--')
                if fault_end is not None:
                    ax_pred[i,2].axvline(datetime.strptime(str(fault_end),
                                                 '%Y-%m-%d %H:%M:%S'), ls='--')

        if plot_SPE:            
            fig_spe.tight_layout();
        if plot_predictions:
            fig_pred.tight_layout();
            
    ######## GENERATING FINAL TABLES ########
        
    return_tables = []
        
    # PREDICTION

    if metrics is not None:

        prediction_table = pd.DataFrame([train_metrics, validation_metrics,
                                         test_metrics], 
                                        index = ['Train',
                                                 'Validation',
                                                 'Test']).T
                                                            
        models_names = [models[i].name for i in range(len(models))]
        metrics_names = [metrics[i].__name__ for i in range(len(metrics))]
        
        iterables = [models_names, metrics_names]

        index = pd.MultiIndex.from_product(iterables, 
                                           names = ['Models', 'Metrics'])
        
        message = ("The MultiIndex table is not of the right size. "+
                   "This could happen when two models have the same name.")
        
        try:
            prediction_table.index = index
        except ValueError as err:
            raise ValueError(message) from err
        
        return_tables.append(prediction_table.swaplevel().sort_index(axis=0))

    # DETECTION     
        
    if fault_start is not None:
        
        detection_table = pd.DataFrame([detection_alarms, 
                                        false_detection_rates],
                                       index = ['FDR','FAR']).T
        
        models_names = [models[i].name for i in range(len(models))]
        alarms_names = [list(model.alarms.keys())[i] for i in 
                range(len(list(model.alarms.keys())))]  
        
        iterables = [models_names, alarms_names]

        index = pd.MultiIndex.from_product(iterables, 
                                        names = ['Models', 'Alarms'])
        
        detection_table.index = index

        return_tables.append(detection_table.swaplevel().sort_index(axis=0))
        
    # COMPUTATIONAL TIMES
        
    if times:
        
        times_dict = {}
        
        times_dict['Train'] = [1e6*models[i].train_time/X_train.shape[0]
                        for i in range(len(models))]
        times_dict['Test'] = [1e6*models[i].test_time/X_test.shape[0] 
                        for i in range(len(models))]
        
        times_df = pd.DataFrame(times_dict, 
                        index=[models[i].name for i in range(len(models))])
        
        return_tables.append(times_df)
        
    return return_tables