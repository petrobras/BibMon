import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

from ._generic_model import GenericModel

###############################################################################

class sklearnRegressor (GenericModel):
    """
    Interface for sklearn regressors.
            
    Parameters
    ----------
    regressor: any regressor that uses the sklearn interface. 
        For example:
            * sklearn.svm.classes.SVR,
            * sklearn.ensemble.forest.RandomForestRegressor,
            * sklearn.neural_network.multilayer_perceptron.MLPRegressor,
            * etc....
    permutation_importance: boolean, optional
        Whether permutation variable importance should be calculated.    
        """     

    ###########################################################################

    def __init__ (self, regressor, permutation_importance = False):

        self.has_Y = True       
        self.regressor = regressor
	
        self.name = self.regressor.__class__.__name__
        
        self.permutation_importance = permutation_importance
    
    ###########################################################################
        
    def train_core (self):
        
        self.regressor.fit(self.X_train.values,
                           self.Y_train.values.squeeze())
        
        if self.permutation_importance:
            
            res = permutation_importance(self.regressor, 
                                         self.X_train.values, 
                                         self.Y_train.values.squeeze(),
                                         n_repeats=10)
            
            self.regressor.perm_feature_importances_ = res.importances_mean

    ###########################################################################

    def map_from_X (self, X):
        
        return self.regressor.predict(X)
    
    ###########################################################################
    
    def set_hyperparameters (self, params_dict):   
        
        for key, value in params_dict.items():
            setattr(self.regressor, key, value)
            
    ###########################################################################
            
    def update_importances(self):
        """
        Calculates permutation importances of the variables.
        """          

        res = permutation_importance(self.regressor, 
                                     self.X_test.values, 
                                     self.Y_test.values.squeeze(),
                                     n_repeats = 10)
            
        self.regressor.perm_feature_importances_ = res.importances_mean        
            
    ###########################################################################

    def plot_importances(self, n = None, permutation_importance = False):
        
        """
        Plots the permutation importances of the variables.
    
        Parameters
        ----------
        n: int, optional
            Maximum number of variables to be plotted.
        permutation_importance: boolean, optional
            If permutation importances should be prioritized over coefficients
            in linear models.
        """          
        
        model = self.regressor
        
        if hasattr(model,'coef_'):
            imp = model.coef_
        elif hasattr(model,'feature_importances_'):
            imp = model.feature_importances_
        elif (hasattr(model,'perm_feature_importances_')):
            imp = model.perm_feature_importances_

        if ((hasattr(model,'coef_') or
            hasattr(model,'feature_importances_')) and permutation_importance): 
            if hasattr(model, 'perm_feature_importances_'):
                imp = model.perm_feature_importances_
          
        if not (hasattr(model,'coef_') or
                hasattr(model,'feature_importances_') or
                hasattr(model, 'perm_feature_importances_')):
            print('There are no importances calculated for this model.')
            return
        
        tags = self.X_train.columns
        
        if n is not None:
            pass
        else:
            n = len(self.X_train.columns)
    
        fig, ax = plt.subplots(1,2, figsize = (20,4))
    
        coefs = []
        abs_coefs = []
    
        coefs = (pd.Series(imp, index = tags))
        coefs.plot(use_index=False, ax=ax[0]);
        abs_coefs = (abs(coefs)/(abs(coefs).sum()))
        abs_coefs.sort_values(ascending=False).plot(use_index=False, ax=ax[1],
                                                    marker='.')
    
        ax[0].set_title('Relative variable importances')
        ax[1].set_title('Relative variable importances - \
                        descending order')
    
        abs_coefs_df = pd.DataFrame(np.array(abs_coefs).T,
                                    columns = ['Importances'],
                                    index = tags)
    
        df = abs_coefs_df['Importances'].sort_values(ascending=False)
        
        plt.figure()
        df.iloc[0:n].plot(kind='barh', figsize=(15,0.25*n), legend=False)
        
        return df