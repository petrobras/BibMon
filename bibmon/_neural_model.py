from ._generic_model import GenericModel

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam

###############################################################################

class NeuralModel(GenericModel):
    """
    Model that uses Keras to apply Deep Learning to find anomaly.
            
    Parameters
    ----------
    outputCount: int
        The quantity of possible states for the process (normal, anomaly, possible anomaly, etc)
    columCount: int
        The number of columns/features this model can handle
    lstmshapes: list, optional
        Whether permutation variable importance should be calculated.
    dropout: float, optional
        The dropout of each LSTM layer.
        """     

    ###########################################################################

    def __init__ (self, columCount,
                  lstmshapes=[128, 64],
                  denseshapes=[16, 32],
                  dropout = 0.2):

        self.model = Sequential()
        
        self.model.add(LSTM(lstmshapes[0], stateful=True, return_sequences=True, batch_input_shape=(1, 1, columCount)))
        self.model.add(Dropout(dropout))

        for shape in lstmshapes[1:-1]:
            self.model.add(LSTM(shape, stateful=True, return_sequences=True))
            self.model.add(Dropout(dropout))

        if len(lstmshapes) > 1:
            self.model.add(LSTM(lstmshapes[-1], stateful=True, return_sequences=False))
            self.model.add(Dropout(dropout))

        for shape in denseshapes:
            self.model.add(Dense(shape, activation='relu'))
        
        self.model.add(Dense(1, activation='softmax'))

        optimizer = Adam(learning_rate=0.001)
        self.model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        
    ###########################################################################
        
    def train_core (self):
        self.model.fit(
            self.X_train.values,
            self.Y_train.values.squeeze(),
            epochs=20,
            batch_size=64,
            validation_split=0.2
        )

    ###########################################################################

    def map_from_X(self, X):
        return self.model.predict(X)
    
    ###########################################################################
    
    def set_hyperparameters (self, params_dict):   
        for key, value in params_dict.items():
            setattr(self.regressor, key, value)