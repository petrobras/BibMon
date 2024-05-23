import numpy as np
import matplotlib.pyplot as plt

from ._generic_model import GenericModel

###############################################################################

class PCA (GenericModel):
    """
    Principal Component Analysis.
    
    For details on the technique, see https://doi.org/10.3390/pr12020251
    
    Parameters
    ----------
    ncomp: int or float
           float: number between 0.0 and 1.0 that corresponds to the minimum 
                  fraction of accumulated variance for component selection;
           int: defines the number of components.
    """

    ###########################################################################
            
    def __init__ (self, ncomp=0.9):

        self.has_Y = False
        self.ncomp = ncomp
        self.name = 'PCA'
        
    ###########################################################################
    
    def load_model (self,limSPE, SPE_mean, count_window_size,
                    Mux, SDx, S, V, n):
        
        """
        Receives parameters from a previously trained model to
        perform predictions and tests without the need for training.
    
        Parameters
        ----------
        Mux: pandas.Series
            Means of the X variables in the training period.
        SDx: pandas.Series
            Standard deviations of the X variables in the training period.
        Mux: pandas.Series, optional
            Means of the Y variables in the training period.
        SDx: pandas.Series, optional
            Standard deviations of the Y variables in the training period.                         
        limSPE: float
            Detection limit and mean of the SPE.
        SPE_mean: float
            Mean of the SPE.
        count_window_size: int
            Window sizes used in the count alarms calculation.        
        S: numpy.array
            Specific parameter of PCA.
        V: numpy.array
            Specific parameter of PCA.
        n: int
            Specific parameter of PCA.        
        """            
                
        super().load_model (limSPE, SPE_mean,
                            count_window_size, Mux, SDx)
        
        self.S = np.array(S)
        self.V = np.array(V)
        self.n = n
        
        self.pv = S/np.sum(S)
        self.pva = np.cumsum(S)/sum(S)

    ###########################################################################
        
    def train_core (self):

        self.Sxx_train = np.cov(self.X_train, rowvar=False)
        _, self.S, self.Vh = np.linalg.svd(self.Sxx_train)
        self.V = self.Vh.T
        self.pv = self.S/np.sum(self.S)
        self.pva = np.cumsum(self.S)/sum(self.S)

        if self.ncomp> 0 and self.ncomp < 1:
            self.n = np.where(self.pva>self.ncomp)[0][0]+1
        elif self.ncomp >= 1:
            self.n = self.ncomp
        else:
            # Error: unkown self.n
            self.n = None
                
    ###########################################################################

    def map_from_X (self, X): 

        X_train_proj = X@self.V[:,:self.n]
        return X_train_proj@self.V[:,:self.n].T
    
    ###########################################################################
        
    def plot_cumulative_variance (self, ax = None):
        """
        Plots the cumulative variance.
    
        Parameters
        ----------
        ax: matplotlib.axes._subplots.AxesSubplot, optional
            Axis on which the graph will be plotted.
        """                
        if ax is not None:
            pass
        else:
            fig, ax = plt.subplots()
        
        ax.bar(np.arange(len(self.pv)), self.pv)
        ax.plot(np.arange(len(self.pv)), self.pva)
        ax.set_xlabel('Number of components')
        ax.set_ylabel('Data variance')