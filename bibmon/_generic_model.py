import copy
import time
import optuna
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.model_selection
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod

from ._alarms import detecOutlier
from ._preprocess import PreProcess
from ._bibmon_tools import align_dfs_by_rows

###############################################################################

class GenericModel (ABC):
    """
    Abstract class used as a base for creating child classes with
    specific models (PCA, sklearnRegressor, etc).

    Attributes
    ----------
            
        * Data sets:
        ----------
        X_train, X_test, Y_train, Y_test: pandas.DataFrame
            Normalized training and testing data sets.
            X refers to the input and Y to the output (when there is Y).
            
        X_train_orig, X_test_orig, 
        Y_train_orig, Y_test_orig: pandas.DataFrame
            Original training and testing data sets (before pre-processing). 
            X refers to the input and Y to the output (when there is Y).       
            
        * Predictions:
        ----------
        X_train_pred, X_test_pred 
        (or Y_train_pred, Y_test_pred): pandas.DataFrame
            Reconstructions (or predictions) 
            provided by the model in training and testing.
            
        X_train_pred_orig, X_test_pred_orig
        (or Y_train_pred_orig, Y_test_pred_orig): pandas.DataFrame
            Reconstructions (or predictions) 
            provided by the model in training and testing
            in the original units.
        
        * Detection, contribution and prediction metrics:
        ----------
        SPE_train, SPE_test: pd.Series
            Square Prediction Error. One-dimensional vectors containing
            the SPE values calculated at each time step in 
            training and testing.  
                
        * Training parameters:
        ----------
        Mux, SDx, Muy, SDy: pandas.Series
            Means and standard deviations of the variables 
            in the training period.
            
        limSPE_train, SPE_mean: float
            Limit for detection and mean of the SPE obtained in training.

        limSPE: float
            Potentially redefined detection limit in the validation stage.
            
        lim_conf: float
            Confidence limit for the detection index.

        train_time: float
            Computational training time.
            
        index_train: pandas.DatetimeIndex
            Indexes of the training set.
            
        * Testing parameters:
        ----------            
        count_window_size: int
            Window sizes used in alarm calculations.
        
        outlier_alarm,
        count_alarm: numpy.array
            One-dimensional arrays that store the various alarms for each
            time step.
            
        alarms: dict
            Dictionary containing the mentioned alarms above.
        
        count_limit: int
            Limit of points to be considered for the count_alarm to trigger.

        test_time: float
            Computational testing time.

        has_alarm_windows: boolean
            Indicator of testing step with or without windows for the alarms.
            
        * Other attributes:
        ----------
        preproc_X, preproc_Y: PreProcess
            Object of the PreProcess class (defined in the _bibmon_tools.py 
            file) that contains the preprocessing functions and arguments
            to be applied in training and testing.
        tags_X, tags_Y: list of str
            tags kept in the data set, obtained in pre-training.
        test_tags_X, test_tags_Y: list of str
            tags used in testing.
        has_Y: boolean
            indicator of the existence or absence of 
            a predicted dataset (Y) in the model.
            
    Methods
    ----------
        train_core (@abstractmethod)
        map_from_X (@abstractmethod)
        set_hyperparameters
        load_model
        pre_train
        train
        pre_test
        test
        plot_SPE
        fit
        predict       
    """

    ###########################################################################
    
    @abstractmethod
    def train_core (self):
        """
        The core of the training algorithm, that is,
        all the necessary steps between `pre_train()` and
        the calculation of the prediction or reconstruction 
        in training by `map_from_X()`.
        """
        pass

    ###########################################################################
            
    @abstractmethod
    def map_from_X (self, X):
        """
        Receives a data matrix X and returns a matrix of predicted 
        or reconstructed values.

        Parameters
        ----------
        X: numpy.array
            Window X of data for prediction or reconstruction.  
            
        Returns
        ----------                
        : numpy.array
        Reconstructed X (or predicted Y).
        """  
        pass

    ###########################################################################
            
    def set_hyperparameters (self, params_dict):
        """
        Receives a dict with hyperparameters to be assigned in the model.

        Parameters
        ----------
        params_dict: dict
            Dictionary with hyperparameter values

        """          
        for key, value in params_dict.items():
            setattr(self, key, value)

    ###########################################################################
        
    def load_model (self, limSPE, SPE_mean, count_window_size,
                    Mux, SDx, Muy = None, SDy = None):
        """
        Receives parameters from a previously trained model for
        making predictions and tests without the need for training.

        Parameters
        ----------                     
        limSPE: float
            Detection limit and mean of the SPE.
        SPE_mean: float
            Mean of the SPE.
        count_window_size: int
            Window sizes used in count alarms calculation.
        Mux: pandas.Series
            Means of the X variables in the training period.
        SDx: pandas.Series
            Standard deviations of the X variables in
            the training period.
        Muy: pandas.Series, optional
            Means of the Y variables in the training period.
        SDy: pandas.Series, optional
            Standard deviations of the Y variables in the training period.    
        """  
    
        self.MuX = pd.Series(Mux)
        self.SDX = pd.Series(SDx)
        self.MuY = pd.Series(Muy)
        self.SDY = pd.Series(SDy)
        self.limSPE = limSPE
        self.SPE_mean = SPE_mean
        self.count_window_size = count_window_size

    ###########################################################################

    def pre_train (self, X_train, Y_train = None, 
                   f_pp = ['remove_empty_variables',
                           'ffill_nan',
                           'remove_frozen_variables',
                           'normalize'], 
                   a_pp = None,
                   f_pp_test = ['replace_nan_with_values',
                                'normalize'], 
                   a_pp_test = None):
        """
        Receives the data for model training and prepares them for training.

        Parameters
        ----------
        X_train: pandas.DataFrame or numpy.ndarray
            Window of X data used in training.
        Y_train: pandas.DataFrame or numpy.ndarray, optional
            Window of Y data used in training.
        f_pp: list, optional
            List containing strings with names of functions to be used 
            in pre-processing the training data (the functions are defined
            in the PreProcess class, in the BibMon_Tools.py file)
        f_pp_test: list, optional
            List containing strings with names of functions to be used 
            in pre-processing the testing data (the functions are defined 
            in the PreProcess class, in the BibMon_Tools.py file)
        a_pp: dict, optional
            Dictionary containing the parameters to be provided
            to each function to perform pre-processing of the training data, 
            in the format {'functionname__argname': argvalue, ...}
        a_pp_test: dict, optional
            Dictionary containing the parameters to be provided
            to each function to perform pre-processing of the testing data, ]
            in the format {'functionname__argname': argvalue, ...}
        """
                
        # storing complete training set
        
        self.X_train = pd.DataFrame(X_train)
        self.Y_train = pd.DataFrame(Y_train)

        # data set that will keep the original units

        self.X_train_orig = copy.deepcopy(self.X_train)
        if self.has_Y:
            self.Y_train_orig = copy.deepcopy(self.Y_train)

        # defining the training medians as the default values 
        # to replace NaNs during testing
        
        if f_pp_test is not None:
            if 'replace_nan_with_values' in f_pp_test:
                if a_pp_test is None:
                    a_pp_test = {}
                a_pp_test['replace_nan_with_values__val'] = \
                    self.X_train.median()

        # applying pre-processing functions

        if f_pp is not None:
            self.preproc_X = PreProcess(f_pp, a_pp)
            self.X_train = self.preproc_X.apply(self.X_train)
            if self.has_Y:
                self.preproc_Y = PreProcess(f_pp,  a_pp, is_Y = True)
                self.Y_train = self.preproc_Y.apply(self.Y_train)
                
        # preparing the testing pre-processing functions
                            
        if f_pp_test is not None:
            self.preproc_X.f_pp = f_pp_test
            self.preproc_X.a_pp = a_pp_test
            if self.has_Y:
                self.preproc_Y.f_pp = f_pp_test
                self.preproc_Y.a_pp = a_pp_test
                
        # aligning the rows of X and Y, if necessary            
        if self.has_Y:
            self.X_train, self.Y_train = align_dfs_by_rows(self.X_train,
                                                            self.Y_train)
        
        # saving the tags
        self.tags_X = self.X_train.columns.to_list()
        if self.has_Y:
            self.tags_Y = self.Y_train.columns.to_list()
          
    ###########################################################################

    def train (self, lim_conf = 0.99, delete_training_data = False):
        """
        Performs the model training. 
        Must be called after the pre_train function.

        Parameters
        ----------
        lim_conf: float, optional
            Confidence limit for the detection index.
        delete_training_data: boolean, optional
            If True, the data is deleted at the end of training.
            Useful to save memory.
        """        
        
        self.lim_conf = lim_conf

        start_train = time.time()
        
        # core of the training algorithm!
        self.train_core()
        
        # predicting! (or reconstructing!)
        train_pred = self.map_from_X (self.X_train.values)

        end_train = time.time()  
        
        self.train_time = end_train - start_train
        
        # storing results and calculating SPE
        
        if self.has_Y:        
            self.Y_train_pred = pd.DataFrame(train_pred,
                                            index=self.Y_train.index,
                                            columns = self.Y_train.columns)  
            
            self.SPE_train = np.sum((self.Y_train.values-
                                    self.Y_train_pred.values)**2,
                                    axis=1)
        else:
            self.X_train_pred = pd.DataFrame(train_pred,
                                            index=self.X_train.index,
                                            columns = self.X_train.columns) 

            self.SPE_train = np.sum((self.X_train.values-
                                    self.X_train_pred.values)**2,
                                    axis=1)
            
        # calculating indices and detection limits

        self.SPE_mean = self.SPE_train.mean()
        
        iSPE = np.sort(self.SPE_train)
        self.limSPE = iSPE[int(lim_conf*self.X_train.shape[0])]
        
        self.limSPE_train = copy.deepcopy(self.limSPE)
        
        # denormalizing
        
        if self.has_Y and self.preproc_Y is not None:
            if 'normalize' in self.preproc_Y.f_pp:
                self.Y_train_pred_orig = \
                    self.preproc_Y.back_to_units(self.Y_train_pred)
            else:
                self.Y_train_pred_orig = self.Y_train_pred
        else:
            if 'normalize' in self.preproc_X.f_pp:
                self.X_train_pred_orig = \
                    self.preproc_X.back_to_units(self.X_train_pred)
            else:
                self.X_train_pred_orig = self.X_train_pred

        # indexes of the training set
        self.index_train = self.X_train.index
        
        self.SPE_train = pd.Series(self.SPE_train, index = self.index_train)
        
        # deleting training data, if applicable
        if delete_training_data:
            del self.X_train, self.X_train_orig, \
                self.SPE_train, self.index_train
            if self.has_Y:
                del self.Y_train, self.Y_train_orig, \
                    self.Y_train_pred, self.Y_train_pred_orig
            else:
                del self.X_train_pred, self.X_train_pred_orig    
    
    ###########################################################################
    
    def hyperparameter_tuning (self, params, n_trials = 20, lim_conf = 0.99,
                               percent_validation = 0.2,
                               n_splits = None, delete_training_data = False):
        """
        Performs hyperparameter tuning using the Optuna library.

        Parameters
        ----------
        params: pandas.DataFrame
            Contains the possibilities to be tested and the types of 
            parameters. Must be defined as: 
            pd.DataFrame({'possibilities': [list_possibilities1, 
                                            list_possibilities2, ...],
                          'types': [str_type1, 
                                    str_type2, ...]},
                         index = [str_param1_name, str_param2_name, ...])
        n_trials: int, optional
            Number of trials in the optimization performed by Optuna.
        lim_conf: float, optional
            Confidence limit for the detection index.
        percent_validation: float (0<value<1), optional
            Percentage of the data to be separated for use 
            in internal validation, if no cross-validation is performed.
        n_splits: int, optional
            Number of sets to be used in cross-validation.
            If not None, the value of percent_validation is disregarded.
        delete_training_data: boolean, optional
            If True, the data is deleted at the end of training.
            Useful to save memory.
        """  
        def objective (trial):

            suggestions = {}
            
            for index, row in params.iterrows():
                if row['type'] == 'uniform':
                    sug = trial.suggest_uniform(index, 
                                                row['possibilities'][0], 
                                                row['possibilities'][1])                  
                if row['type'] == 'loguniform':
                    sug = trial.suggest_loguniform(index, 
                                                row['possibilities'][0], 
                                                row['possibilities'][1])  
                if row['type'] == 'discrete_uniform':
                    sug = trial.suggest_discrete_uniform(index, 
                                                       row['possibilities'][0], 
                                                       row['possibilities'][1],
                                                       row['possibilities'][2])   
                if row['type'] == 'int':
                    sug = trial.suggest_int(index, 
                                            int(row['possibilities'][0]), 
                                            int(row['possibilities'][1]))
                if row['type'] == 'categorical':
                    sug = trial.suggest_categorical(index, 
                                                    row['possibilities']) 
                suggestions[index] = sug
            
            self.set_hyperparameters(suggestions)
            
            if n_splits is not None:
            
                kf = sklearn.model_selection.KFold(n_splits = n_splits)
                            
                SPEs = []
                
                for train_index, test_index in kf.split(self.X_train):
                    autocopy = copy.deepcopy(self)
                    autocopy.pre_train(X_train =
                                       self.X_train.iloc[train_index,:],
                                       Y_train = 
                                       self.Y_train.iloc[train_index,:])
                    autocopy.train(lim_conf, delete_training_data = True)
                    
                    autocopy.pre_test(X_test = self.X_train.iloc[test_index,:],
                                      Y_test = self.Y_train.iloc[test_index,:])
                    
                    autocopy.test()
                    SPEs.append(np.mean(autocopy.SPE_test))

                del autocopy
                return np.mean(SPEs)
            
            else:
                
                n = int(self.X_train.shape[0]*percent_validation)
                
                autocopy = copy.deepcopy(self)
                
                autocopy.pre_train(X_train = self.X_train.iloc[n:,:],
                                    Y_train = self.Y_train.iloc[n:,:])
                autocopy.train(lim_conf, delete_training_data = True)
                autocopy.pre_test(X_test = self.X_train.iloc[:n,:],
                                    Y_test = self.Y_train.iloc[:n,:])
                autocopy.test()
            
                mean_spe = np.mean(autocopy.SPE_test)
                del autocopy
                return mean_spe
        
        self.hyperparemeter_study = optuna.create_study()
        self.hyperparemeter_study.optimize(objective, n_trials = n_trials)
        
        print(self.hyperparemeter_study.best_params)
        
        self.set_hyperparameters(self.hyperparemeter_study.best_params)
                   
    ###########################################################################

    def pre_test (self, X_test, Y_test = None, 
                  count_window_size = 0, count_limit = 1,
                  f_pp = None, a_pp = None):
        '''   
        Receives a window of data and prepares it for the model testing.
        
        Parameters
        ----------
        X_test: pandas.DataFrame, pandas.Series or numpy.ndarray
            Window of data or observation X needed to perform the analysis.
        Y_test: pandas.DataFrame, pandas.Series or numpy.ndarray, optional
            Window of data or observation Y needed to perform the analysis.
        count_window_size: int, optional
            Window size used in the count alarm calculation.
        count_limit: int, optional
            Limit of points to be considered in the window 
            for the count alarm to trigger. 
        f_pp: list, optional
            List containing strings with names of functions to be used 
            in pre-processing (the functions are defined in the 
            PreProcess class).
        a_pp: dict, optional
            Dictionary containing the parameters to be provided
            to each function to perform pre-processing of the data, in
            the format {'functionname__argname': argvalue, ...}
        '''
        
        # window sizes for alarms:
                        
        self.count_limit = count_limit
        self.count_window_size = count_window_size

        # storing complete testing set
                                
        if (isinstance(X_test, np.ndarray) and X_test.shape == ()):
            self.X_test = pd.DataFrame(X_test[None])
        elif len(X_test.shape)==1 and ((isinstance(X_test, np.ndarray))):
                self.X_test = pd.DataFrame(X_test).T
        else:
            self.X_test = pd.DataFrame(X_test)  
            
        if self.has_Y:
            if (isinstance(Y_test, np.ndarray) and Y_test.shape == ()):
                self.Y_test = pd.DataFrame(Y_test[None])
            elif len(Y_test.shape)==1 and ((isinstance(Y_test, np.ndarray))):
                    self.Y_test = pd.DataFrame(Y_test).T
            else:
                self.Y_test = pd.DataFrame(Y_test)

        # data set that will keep the original units       
                
        self.X_test_orig = copy.deepcopy(self.X_test)
        if self.has_Y:
            self.Y_test_orig = copy.deepcopy(self.Y_test) 

        # applying pre-processing functions

        if f_pp is not None:

            if not hasattr(self, 'preproc_X'):
                self.preproc_X = PreProcess(f_pp, a_pp)
            else:
                self.preproc_X.f_pp = f_pp
                self.preproc_X.a_pp = a_pp

            if self.has_Y:

                if not hasattr(self, 'preproc_Y'):
                    self.preproc_Y = PreProcess(f_pp, a_pp, is_Y = True)
                else:
                    self.preproc_Y.f_pp = f_pp
                    self.preproc_Y.a_pp = a_pp

        if hasattr(self, 'preproc_X'):
            self.X_test = self.preproc_X.apply(self.X_test, 
                                                train_or_test = 'test')

        if hasattr(self, 'preproc_Y'):
            self.Y_test = self.preproc_Y.apply(self.Y_test, 
                                                train_or_test = 'test')  

        # aligning the rows of X and Y, if necessary
        if self.has_Y:
            self.X_test, self.Y_test = align_dfs_by_rows(self.X_test,
                                                         self.Y_test)

        # test tags   
            
        self.tags_test_X = [t for t in self.tags_X
                            if t in self.X_test.columns.tolist()]
        
        self.X_test = self.X_test[self.tags_test_X]

        if self.has_Y:

            self.tags_test_Y = [t for t in self.tags_Y 
                                if t in self.Y_test.columns.tolist()]
                
            self.Y_test = self.Y_test[self.tags_test_Y]
                        
    ###########################################################################

    def test (self, redefine_limit = False, delete_testing_data = False):
        """
        Analyzes a window of data, applying a model test.
        Must be called after the pre_test function.
        
        Parameters
        ----------
        redefine_limit: boolean, optional
            Indicator of redefinition or not of the detection limit during
            testing.
        delete_testing_data: boolean, optional
            If True, the data is deleted at the end of testing.
            Useful to save memory.
        """

        start_test = time.time()
        
        # predicting!
        test_pred = self.map_from_X(self.X_test.values)

        end_test = time.time()
        
        self.test_time = end_test - start_test

        # storing results and calculating SPE
    
        if self.has_Y:
            self.Y_test_pred = pd.DataFrame(test_pred,
                                            index=self.Y_test.index,
                                            columns = self.Y_test.columns)
            self.SPE_test = np.sum((self.Y_test.values-
                                    self.Y_test_pred.values)**2,
                                    axis=1)
            
        else:
            self.X_test_pred = pd.DataFrame(test_pred,
                                            index=self.X_test.index,
                                            columns = self.X_test.columns)
            self.SPE_test = np.sum((self.X_test.values-
                                    self.X_test_pred.values)**2,
                                    axis=1)   
        
        # redefining the limit, for the validation case
        
        if redefine_limit:
            iSPE = np.sort(self.SPE_test)
            self.limSPE = iSPE[int(self.lim_conf*self.X_test.shape[0])]
        
        # calculations of the alarms
        
        if self.count_window_size != 0:
            self.alarmCount = np.zeros(len(self.SPE_test))

        self.alarmOutlier = detecOutlier(self.SPE_test, 
                                         self.limSPE)

        # if more window alarms are defined in the package, 
        # i_min must be the minumum size among them
        i_min = self.count_window_size
        
        for i in range(i_min, len(self.SPE_test)):
                        
            if self.count_window_size != 0:
                if i >= self.count_window_size:
                    self.alarmCount[i] = \
                    detecOutlier(self.SPE_test[i-self.count_window_size:(i+1)], 
                                 self.limSPE, count = True, 
                                 count_limit = self.count_limit)
            
        # adjusting the size of the prediction and denormalizing    
        if self.has_Y:
            if 'normalize' in self.preproc_Y.f_pp:
                self.Y_test_pred_orig = \
                    self.preproc_Y.back_to_units(self.Y_test_pred)
            else:
                self.Y_test_pred_orig = self.Y_test_pred
        else:
            if 'normalize' in self.preproc_X.f_pp:
                self.X_test_pred_orig = \
                    self.preproc_X.back_to_units(self.X_test_pred)
            else:
                self.X_test_pred_orig = self.X_test_pred
        
        self.SPE_test = pd.Series(self.SPE_test, 
                                index = self.X_test.index)

        self.alarms = {}

        self.alarmOutlier = pd.Series(self.alarmOutlier,
                                    index = self.X_test.index)
        self.alarms['alarmOutlier'] = self.alarmOutlier
        if self.count_window_size != 0:
            self.alarmCount = pd.Series(self.alarmCount,
                                        index = self.X_test.index)
            str_count = f'alarmCount WS={self.count_window_size}, '\
                    f'limCount = {self.count_limit}'
            self.alarms[str_count] = self.alarmCount

        # deleting testing data, if applicable
        if delete_testing_data:
            del self.X_test, self.X_test_orig, \
                self.SPE_test, self.alarms, self.alarmOutlier
            if self.has_Y:
                del self.Y_test, self.Y_test_orig, \
                    self.Y_test_pred, self.Y_test_pred_orig
            else:
                del self.X_test_pred, self.X_test_pred_orig
            if hasattr(self, 'alarmCount'): del self.alarmCount

    ###########################################################################
        
    def plot_SPE (self, ax = None, train_or_test = 'train', logy = True,
                  legends = True, plot_alarm_outlier = True):
        """
        Plotting the temporal evolution of SPE.

        Parameters
        ----------
        ax: matplotlib.axes._subplots.AxesSubplot, optional
            Axis on which the graph will be plotted.
        train_or_test: string, optional
            Indicates whether to plot the graph for 'train' or 'test'.
        logy: boolean, optional
            Indicates whether the y-axis scale should be logarithmic (True) or 
            linear (False).
        legends: boolean, optional
            If the graph should display legends.
        plot_alarm_outlier: boolean, optional
            If the alarmOutlier should be plotted.
        """
        
        if ax is not None:
            pass
        else:
            fig, ax = plt.subplots()
            
        if train_or_test == 'train':
            SPE = self.SPE_train
            limSPE = self.limSPE_train
        elif train_or_test == 'test':
            SPE = self.SPE_test
            limSPE = self.limSPE
        
        SPE.plot(ax=ax, logy = logy, ls='',
                marker='.', label='SPE')
        
        if train_or_test == 'test':
            for label, alarm in self.alarms.items():
                if not plot_alarm_outlier and label=='alarmOutlier':
                    continue
                (ax.get_ylim()[1]*alarm).replace(0, np.nan).plot(ax = ax, 
                                                                logy = logy, 
                                                                ls = '',
                                                                marker = '.', 
                                                                alpha = 0.5,
                                                                label = label)
                    
        ax.axhline(limSPE, color='red', ls = '--',
                label='%.0f%% Confidence Limit' %(self.lim_conf*100))
        
        if legends:
            ax.legend(fontsize=12);
                        
    ###########################################################################

    def plot_predictions (self, ax = None, train_or_test = 'train', 
                          X_pred_to_plot = None, metrics = None):
        """
        Plotting the temporal evolution of the predictions along with the
        respective true values.

        Parameters
        ----------
        ax: matplotlib.axes._subplots.AxesSubplot, optional
            Axis on which the graph will be plotted.
        train_or_test: string, optional
            Indicates whether to plot the graph for 'train' or 'test'.
        X_pred_to_plot: string, optional
            In case the model is a reconstruction model 
            (i.e., self.has_Y = False), indicates which column of X to plot 
            along with the prediction.
        metrics: list of functions, optional
            Functions for calculating metrics to be displayed in the title of 
            the graph.
        """ 

        if ax is not None:
            pass
        else:
            fig, ax = plt.subplots()
        
        if self.has_Y:
        
            if train_or_test=='train':
                true = self.Y_train_orig
                pred = self.Y_train_pred_orig
            elif train_or_test=='test':
                true = self.Y_test_orig
                pred = self.Y_test_pred_orig
            
            regression_or_reconstruction = 'Regression'
            
        else:
            
            if X_pred_to_plot is not None:

                if train_or_test=='train':
                    true = self.X_train_orig[X_pred_to_plot]
                    pred = self.X_train_pred_orig[X_pred_to_plot]
                elif train_or_test=='test':
                    true = self.X_test_orig[X_pred_to_plot]
                    pred = self.X_test_pred_orig[X_pred_to_plot]
                
                regression_or_reconstruction = 'Reconstruction'
                
            else:

                raise RuntimeError('The model has no Y. \
                                Specify which variable in X should be plotted')
                
        true.plot(ax=ax)
        pred.plot(ax=ax,ls='--')
        
        ax.legend(['Measurement',regression_or_reconstruction])
        
        tags = pd.DataFrame(true).columns.to_list()
                
        title = str(tags)
        
        if metrics is not None:
            
            if not isinstance(metrics, list): 
                metrics = [metrics]
                                
            true, pred = align_dfs_by_rows(true.dropna(), pred)
            
            for metric in metrics:
                title = (title+
                        f'\n{metric.__name__}: {metric(true, pred):.2f}')
        
        ax.set_title(title)
    
    ###########################################################################

    def fit (self, X_train, Y_train = None,
             f_pp = ['remove_empty_variables',
                     'ffill_nan',
                     'remove_frozen_variables',
                     'normalize'],
             a_pp = None,
             f_pp_test = ['replace_nan_with_values',
                          'normalize'],
             a_pp_test = None,
             lim_conf = 0.99,
             delete_training_data = False,
             redefine_limit = False,
             frac_val = 0.15,
             tune = False,
             params = None,
             params_types = None,
             params_possibilities = None,
             n_trials = 20):
        """
        Performs the complete pipeline of model training,
        sequentially executing the 'pre_train' and 'train' methods.

        Parameters
        ----------
        X_train: pandas.DataFrame or numpy.ndarray
            Window of X data used in training.
        Y_train: pandas.DataFrame or numpy.ndarray, optional
            Window of Y data used in training.
        f_pp: list, optional
            List containing strings with names of functions to be used 
            in pre-processing the training data (the functions are defined
            in the PreProcess class, in the BibMon_Tools.py file).
        f_pp_test: list, optional
            List containing strings with names of functions to be used 
            in pre-processing the testing data (the functions are defined 
            in the PreProcess class, in the BibMon_Tools.py file).
        a_pp: dict, optional
            Dictionary containing the parameters to be provided
            to each function to perform pre-processing of the training data, in
            the format {'functionname__argname': argvalue, ...}.
        a_pp_test: dict, optional
            Dictionary containing the parameters to be provided
            to each function to perform pre-processing of the testing data, in
            the format {'functionname__argname': argvalue, ...}.
        lim_conf: float, optional
            Confidence limit for the detection index.
        delete_training_data: boolean, optional
            If True, the data is deleted at the end of training.
            Useful to save memory.
        redefine_limit: boolean, optional
            Indicator of redefinition or not of the detection limit using
            a validation period taken from the training data itself.
        frac_val: float, optional
            Fraction of the data used for validation. 
            0<frac_val<1.
            Only used if redefine_limit==True.
        tune: boolean, optional
            Indicator of automatic hyperparameter tuning.
        params: string or list of strings, optional
            Name(s) of the parameter(s) to be tuned.
        params_types:  string or list of strings
            Type(s) of the parameter(s) to be tuned.
        params_possibilities: list, optional
            Possibilities to be tested for each parameter.
            It must be a list containing the possibilities 
            (in case of only one parameter), or a list containing the lists
            for each possibility.
            Possibilities must be provided according to the type of
            the parameter, as specified in the Optuna library API.
        n_trials: int, optional
            Number of iterations in the hyperparameter search optimization.
        """

        X_train = pd.DataFrame(X_train)
        Y_train = pd.DataFrame(Y_train)

        if redefine_limit:
            
            n_trn =  int(X_train.shape[0]*(1-frac_val))
            X_val = X_train.iloc[n_trn:]
            X_train = X_train.iloc[:n_trn]
            
            if Y_train is not None:

                Y_val = Y_train.iloc[n_trn:]
                Y_train = Y_train.iloc[:n_trn]
            
            else:
                
                Y_val = None
            
        if tune:
            self.pre_train(X_train, Y_train, 
                f_pp = f_pp,
                a_pp = a_pp,
                f_pp_test = f_pp_test,
                a_pp_test  = a_pp_test)
            if not isinstance(params, list):
                params = [params]
            if not isinstance(params_types, list):
                params_types = [params_types]
            if not isinstance(params_possibilities, list):
                print('params_possibilities must be a list!')
            else:
                if not isinstance(params_possibilities[0], list):
                    params_possibilities = [params_possibilities]
            params_df = pd.DataFrame({'possibilities': params_possibilities,
                                      'type': params_types},
                                     index = params) 
            self.tuning(n_trials = n_trials, params = params_df, 
                        lim_conf = lim_conf, 
                        delete_training_data = delete_training_data)
            
        self.pre_train(X_train, Y_train, 
                        f_pp = f_pp,
                        a_pp = a_pp,
                        f_pp_test = f_pp_test,
                        a_pp_test  = a_pp_test)

        self.train(lim_conf = lim_conf, 
                   delete_training_data = delete_training_data)

        if redefine_limit:
            self.pre_test(X_val, Y_val)
            self.test(redefine_limit = True)
        if f_pp_test:   
            # save value for use in predict if necessary
            self.f_pp_test = f_pp_test 
        if a_pp_test:  
             # save value for use in predict if necessary
            self.a_pp_test = a_pp_test

    ###########################################################################
        
    def predict (self, 
                 X_test, 
                 Y_test = None,  
                 count_window_size = 0, 
                 count_limit = 1, 
                 f_pp = 'fit', 
                 a_pp = 'fit', 
                 delete_testing_data = False,
                 redefine_limit = False):
        '''   
        Performs the complete pipeline of model prediction (testing), 
        sequentially executing the 'pre_test' and 'test' methods.
        
        Parameters
        ----------
        X_test: pandas.DataFrame, pandas.Series or numpy.ndarray
            Window of data or observation X needed to perform the analysis.
        Y_test: pandas.DataFrame, pandas.Series or numpy.ndarray, optional
            Window of data or observation Y needed to perform the analysis.
        count_window_size: int, optional
            Window size used in count alarm calculation. 
        count_limit: int, optional
            Limit of points to be considered in the window 
            for the count alarm to trigger.            
        f_pp: list, optional
            List containing strings with names of functions to be used 
            in pre-processing (the functions are defined in the 
            PreProcess class).
        a_pp: dict, optional
            Dictionary containing the parameters to be provided
            to each function to perform pre-processing of the data, in
            the format {'functionname__argname': argvalue, ...}
        delete_testing_data: boolean, optional
            If True, the data is deleted at the end of testing.
            Useful to save memory.   
        redefine_limit: boolean, optional
            Indicator of redefinition or not of the detection limit during
            testing.
        '''
        
        if f_pp == 'fit':
            if hasattr(self,'f_pp_test'):
                f_pp = self.f_pp_test
            else:
                f_pp = None
        if a_pp == 'fit':
            if hasattr(self,'a_pp_test'):
                a_pp = self.a_pp_test
            else:
                a_pp = None
            
        self.pre_test(X_test, Y_test, count_limit = count_limit, 
                      count_window_size = count_window_size, 
                      f_pp = f_pp, a_pp = a_pp)
        self.test(redefine_limit = redefine_limit,
                  delete_testing_data = delete_testing_data)
        