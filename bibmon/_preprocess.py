import copy
import pandas as pd
import statsmodels.tsa.tsatools

###############################################################################

class PreProcess ():
    """
    Class used to encapsulate data preprocessing methods.
    
    Parameters
    ----------
            
        f_pp: list, optional
            List containing strings with names of methods to be used 
            in the preprocessing of the train data. The list of methods 
            is shown below.
        a_pp: dict, optional
            Dictionary containing the parameters to be provided
            to each function to perform preprocessing of the train data, in
            the format {'functionname__argname': argvalue, ...}
        is_Y: boolean, optional
            If the data being preprocessed is Y (that is, to be predicted).
        
    Methods:
    
    * Variable selection:
        remove_empty_variables();
        remove_frozen_variables()
        
    * Missing values imputation:
        ffill()
        remove_observations_with_nan();
        replace_nan_with_values()
        
    * Normalization:
        back_to_units();
        normalize()
        
    * Adding dynamics:
        apply_lag();
        add_moving_average()
        
    * Noise treatment:
        moving_average_filter()       
    
    """
            
    ###########################################################################
    
    def __init__(self, f_pp = None, a_pp = None, is_Y = False):
 
        self.is_Y = is_Y
        self.f_pp = f_pp        
        self._a_pp = a_pp     
        if self.f_pp is not None:
            self.params_per_func = {f: {} for f in f_pp}        

    ###########################################################################

    @property
    def a_pp(self):
        return self._a_pp

    ###########################################################################
    
    @a_pp.setter
    def a_pp(self, a_pp):

        self._a_pp = a_pp
        
        if self.f_pp is not None:
        
            self.params_per_func = {f: {} for f in self.f_pp}
            
            if a_pp is not None:
                                
                for pname, pval in a_pp.items():
                    func, param = pname.split('__',1)
                    self.params_per_func[func][param] = pval

    ###########################################################################
    
    def apply(self, df, train_or_test = 'train'):
        """
        Sequentially applies the preprocessing functions 
        defined during initialization.
    
        Parameters
        ----------
        df: pandas.DataFrame
            Data to be processed.
        train_or_test: string, optional
            Indicates which step the data corresponds to.
        Returns
        ----------                
        : pandas.DataFrame
           Processed data.
        """         
        
        df_processed = df
        
        for i in range(len(self.f_pp)):
            f = self.f_pp[i]
            df_processed = getattr(self, f)(df_processed,
                                            train_or_test, 
                                            **self.params_per_func[f])
                
        return df_processed
    
    ######################### 
    # VARIABLE SELECTION 
    #########################

    ###########################################################################

    def remove_empty_variables (self, df, train_or_test = 'train'):
        """
        Removes variables with no values.
    
        Parameters
        ----------
        df: pandas.DataFrame
            Data to be processed.
        train_or_test: string, optional
            Indicates which step the data corresponds to.
        Returns
        ----------                
        : pandas.DataFrame
           Processed data.
        """           
        if train_or_test == 'train':
            return df.dropna(axis=1, how='all')
        elif train_or_test == 'test':
            return df

    ###########################################################################

    def remove_frozen_variables (self, df, train_or_test = 'train',
                                 threshold = 1e-6): 
        """
        Removes variables whose variation falls below a given limit.
    
        Parameters
        ----------
        df: pandas.DataFrame
            Data to be processed.
        train_or_test: string, optional
            Indicates which step the data corresponds to.
        threshold: float, optional
            Variance limit to consider a variable as frozen.
        Returns
        ----------                
        : pandas.DataFrame
           Processed data.
        """                                              
        if not self.is_Y:
            if train_or_test == 'train':
                return df.loc[:, df.var(ddof=1) > threshold]
            elif train_or_test == 'test':
                return df
        else:
            return df
        
    ################################## 
    # MISSING VALUES IMPUTATION
    ##################################
        
    ###########################################################################
    
    def ffill_nan (self, df, train_or_test = 'train'):
        """
        Fills missing (NaN) values with the last valid value.
        Uses the next valid value if there is no last available.
    
        Parameters
        ----------
        df: pandas.DataFrame
            Data to be processed.
        train_or_test: string, optional
            Indicates which step the data corresponds to.
        Returns
        ----------                
        : pandas.DataFrame
           Processed data.
        """    
        return df.ffill().bfill()

    ###########################################################################
    
    def remove_observations_with_nan (self, df, train_or_test = 'train'):
        """
        Removes observations with missing data (NaN).
    
        Parameters
        ----------
        df: pandas.DataFrame
            Data to be processed.
        train_or_test: string, optional
            Indicates which step the data corresponds to.
        Returns
        ----------                
        : pandas.DataFrame
           Processed data.
        """    
        return df.dropna(axis=0, how='any')
    
    ###########################################################################

    def replace_nan_with_values (self, df, train_or_test = 'train', val = 0):
        """
        Replaces missing data (NaN) with a predefined value.

        Parameters
        ----------
        df: pandas.DataFrame
            Data to be processed.
        train_or_test: string, optional
            Indicates which step the data corresponds to.
        val: int or float
            Value to be used in the replacement.
        Returns
        ----------                
        : pandas.DataFrame
        Processed data.
        """    
                                    
        return df.fillna(val)

    ###############
    # NORMALIZATION
    ###############

    ###########################################################################

    def back_to_units (self, df):
        """
        Returns the variables to the original scale, 
        reverting effects of a normalization.

        Parameters
        ----------
        df: pandas.DataFrame
            Data to be processed.
        Returns
        ----------                
        : pandas.DataFrame
        Processed data.
        """    
        return df*self.SD + self.Mu
    
    ###########################################################################

    def normalize (self, df, train_or_test = 'train', mode = 'standard'):
        """
        Variable normalization.

        Parameters
        ----------
        df: pandas.DataFrame
            Data to be processed.
        train_or_test: string, optional
            Indicates which step the data corresponds to.
        mode: string, optional
            Type of normalization (standard, robust, m-robust or s-robust).
        Returns
        ----------                
        : pandas.DataFrame
        Processed data.
        """    
        if train_or_test == 'train':
            
            if mode == 'standard':
                self.Mu = df.mean()
                self.SD = df.std(ddof=1)
            elif mode == 'robust':
                self.Mu = df.median()
                self.SD = df.mad()               
            elif mode == 'm-robust':
                self.Mu = df.median()
                self.SD = df.std(ddof=1)
            elif mode == 's-robust':
                self.Mu = df.mean()
                self.SD = df.mad()               
            
            return (df - self.Mu)/self.SD
        
        elif train_or_test == 'test':
        
            return (df - self.Mu)/self.SD

    ##############################
    # ADDING DYNAMICS
    ##############################
        
    ###########################################################################

    def apply_lag (self, df, train_or_test = 'train', lag = 1):
        """
        Generation of time-delayed variables.

        Parameters
        ----------
        df: pandas.DataFrame
            Data to be processed.
        train_or_test: string, optional
            Indicates which step the data corresponds to.
        lag: int, optional
            Number of delays to be considered.
        Returns
        ----------                
        : pandas.DataFrame
        Processed data.
        """    
                        
        if self.is_Y:
            return df.iloc[lag:,:]
        else:    
            array_lagged = statsmodels.tsa.tsatools.lagmat(df, maxlag = lag, 
                                                           trim = "forward", 
                                                       original = 'in')[lag:,:]   
            new_columns = []
            for l in range(lag):
                new_columns.append(df.columns+' - lag '+str(l+1))
            columns_lagged = df.columns.append(new_columns)
            index_lagged = df.index[lag:]
            df_lagged = pd.DataFrame(array_lagged, index = index_lagged,
                                     columns = columns_lagged)
            
            return df_lagged  
        
    ###########################################################################

    def add_moving_average (self, df, train_or_test = 'train', WS = 10):
        """
        Adding variables filtered by moving average.
        Attention! Do not confuse with moving_average_filter, in which
        the original variables are not kept in the dataset.

        Parameters
        ----------
        df: pandas.DataFrame
            Data to be processed.
        train_or_test: string, optional
            Indicates which step the data corresponds to.
        WS: int, optional
            Window size of the filter.
        Returns
        ----------                
        : pandas.DataFrame
        Processed data.
        """    
        if self.is_Y:
            return df
                
        new_df = copy.deepcopy(df)
                
        for column in df:
            new_df[column+' MA'] = new_df[column].rolling(WS).mean()
        
        return new_df.drop(df.index[:WS])

    ##############################
    # NOISE TREATMENT
    ##############################

    ###########################################################################

    def moving_average_filter (self, df,  train_or_test = 'train', WS = 10):
        """
        Moving average noise filter.

        Parameters
        ----------
        df: pandas.DataFrame
            Data to be processed.
        train_or_test: string, optional
            Indicates which step the data corresponds to.
        WS: int, optional
            Window size of the filter.
        Returns
        ----------                
        : pandas.DataFrame
        Processed data.
        """    
        new_df = copy.deepcopy(df)
                
        for column in df:
            new_df[column] = new_df[column].rolling(WS).mean()
            
        if hasattr(df,'name'):
            new_df.name = df.name
                        
        return new_df.drop(df.index[:WS])