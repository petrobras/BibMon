from .generic_model import GenericModel

class NewModel (GenericModel):
    
    ###########################################################################
            
    def __init__ (self):
        """
        Constructor.

        Here, the variable has_Y should be specified.
        This variable indicates whether the model has a 
        separate set of prediction variables Y or not.

        It is recommended to inform model initialization parameters here.
        """
        self.has_Y = False # or True!
        
    ###########################################################################
    
    def load_model (self,
                         limSPE, SPE_mean, lagX, count_window_size,
                         Mux, SDx, Muy = None, SDy = None,
                         # potential additional parameters,
                         ):
        
        super().load_model (limSPE, SPE_mean, count_window_size, 
                            Mux, SDx, Muy, SDy)
        
        """
        Receives parameters from a previously trained model for
        making predictions and tests without the need for training.
        
        !!
        Below, you should store additional parameters as attributes 
        of the class; if there are no additional parameters, 
        this method can be deleted.
        !!
    
        Parameters:
        ----------
        Mux: pandas.Series
            Means of the X variables in the training period.
        SDx: pandas.Series
            Standard deviations of the X variables in
            the training period.
        Muy: pandas.Series, optional
            Means of the Y variables in the training period.
        SDy: pandas.Series, optional
            Standard deviations of the Y variables in the training period.                         
        limSPE: float
            Detection limit and mean of the SPE.
        SPE_mean: float
            Mean of the SPE.
        count_window_size: int
            Window sizes used in count alarms calculation.
        """

    ###########################################################################
        
    def train_core (self):
        """
        The core of the training algorithm, that is,
        all the necessary steps between pre_train() and
        the calculation of the prediction in training.
        """

    ###########################################################################

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