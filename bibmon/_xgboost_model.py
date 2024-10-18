from xgboost import XGBRegressor
from _generic_model import GenericModel  # Assuming GenericModel is in the _generic_model.py file
from sklearn.metrics import r2_score

class XGBModel(GenericModel):
    def __init__(self, model_params=None, *args, **kwargs):
        """
        Initializes the XGBModel class, which is a subclass of GenericModel.
        
        Parameters
        ----------
        model_params: dict, optional
            Parameters for the XGBRegressor to be used. If not provided, 
            the default values of XGBRegressor will be used.
        """
        super().__init__(*args, **kwargs)
        if model_params is None:
            model_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
        # Setting the model as XGBRegressor with the provided parameters
        self.model = XGBRegressor(**model_params)

    def train(self, X_train, Y_train, **kwargs):
        """
        Trains the XGBRegressor model on the provided data.
        
        Parameters
        ----------
        X_train: pandas.DataFrame or numpy.ndarray
            Input dataset for training.
        Y_train: pandas.DataFrame or numpy.ndarray
            Target labels or values for training.
        
        Returns
        -------
        str
            Confirmation message that training is completed.
        """
        self.model.fit(X_train, Y_train, **kwargs)
        return "Training completed."

    def predict(self, X_test):
        """
        Makes predictions with the XGBRegressor model on the provided test data.
        
        Parameters
        ----------
        X_test: pandas.DataFrame or numpy.ndarray
            Input dataset to make predictions.
        
        Returns
        -------
        numpy.ndarray
            Predicted values by the model.
        """
        Y_pred = self.model.predict(X_test)
        return Y_pred

    def evaluate(self, X_test, Y_test, metric_func=None):
        """
        Evaluates the model's performance on the test data.

        Parameters
        ----------
        X_test: pandas.DataFrame or numpy.ndarray
            Input dataset for evaluation.
        Y_test: pandas.DataFrame or numpy.ndarray
            Target labels or values for evaluation.
        metric_func: callable, optional
            Metric function to be used to evaluate the model. If not provided, uses R2 score.
        
        Returns
        -------
        float
            Result of the evaluation metric.
        """
        Y_pred = self.predict(X_test)
        if metric_func is None:
            
            score = r2_score(Y_test, Y_pred)
        else:
            score = metric_func(Y_test, Y_pred)
        return score
