from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
import xgboost as xgb

class GridSearchModel:
    """
    Class used to encapsulate the grid search process for hyperparameter tuning.
    
    Parameters
    ----------
    model : estimator object
        The base model to perform grid search on (e.g., RandomForestRegressor).
        
    param_grid : dict, optional
        The grid of parameters to search over. If not provided, a default
        parameter grid for RandomForestRegressor will be used.
        
    scoring : string, optional
        The scoring method to use for evaluating the models.
        
    cv : int, optional
        Number of cross-validation folds.
    
    test_size : float, optional
        The percentage of data used for testing, default is 0.1 (10%).
    
    Attributes
    ----------
    best_params_ : dict
        The best parameters found by the grid search.
    
    best_estimator_ : estimator object
        The estimator that was chosen by the grid search.
    
    Methods
    ----------
    fit(df, target_column):
        Fits the grid search using the provided data.
    """

    def __init__(self, model=None, param_grid=None, scoring='neg_mean_absolute_error', cv=5, test_size=0.01):
        self.model = model if model is not None else RandomForestRegressor(n_jobs=-1, random_state=42)
        
        # Param_grid default
        default_param_grid = {
            'n_estimators': [10, 100],
            'max_depth': [None, 10],
            'min_samples_split': [2, 4],
        }
        self.param_grid = param_grid if param_grid is not None else default_param_grid
        
        self.scoring = scoring
        self.cv = cv
        self.test_size = test_size  # Using only 10% of the data for testing
        self.grid_search = None
        self.best_params_ = None
        self.best_estimator_ = None

    def fit(self, df: pd.DataFrame, target_column: str):
        """
        Fit the grid search model on the provided data.
        
        Parameters
        ----------
        df : pandas.DataFrame
            The dataset including features and target column.
            
        target_column : str
            The name of the column to be predicted (target variable).
        """
        X = df.drop(columns=[target_column])
        Y = df[target_column]
        
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=self.test_size, random_state=42)
        
        self.grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=-1
        )
        
        self.grid_search.fit(X_train, Y_train)
        
        self.best_params_ = self.grid_search.best_params_
        self.best_estimator_ = self.grid_search.best_estimator_
        
        return self.best_estimator_
    
    def predict(self, X):
        """
        Make predictions using the best estimator found by grid search.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Input data for making predictions.

        Returns
        ----------
        y_pred : numpy.ndarray
            Predicted values.
        """
        if self.best_estimator_ is None:
            raise ValueError("The model has not been fitted yet. Call the `fit` method first.")
        return self.best_estimator_.predict(X)

    def score(self, X_test, y_test):
        """
        Evaluate the best estimator's performance on the test set.

        Parameters
        ----------
        X_test : pandas.DataFrame or numpy.ndarray
            Test features.
        y_test : pandas.Series or numpy.ndarray
            True values for the test set.

        Returns
        ----------
        score : float
            The score of the best model on the test set based on the defined scoring metric.
        """
        if self.best_estimator_ is None:
            raise ValueError("The model has not been fitted yet. Call the `fit` method first.")
        return self.best_estimator_.score(X_test, y_test)
    
    def get_best_params(self):
        """
        Get the best hyperparameters found by grid search.

        Returns
        ----------
        best_params : dict
            Dictionary of the best hyperparameters.
        """
        if self.best_params_ is None:
            raise ValueError("Grid search has not been performed yet. Call the `fit` method first.")
        return self.best_params_


    def save_model(self, filename):
        """
        Save the best model to a file.

        Parameters
        ----------
        filename : str
            Path where the model should be saved.

        Returns
        ----------
        None
        """
        if self.best_estimator_ is None:
            raise ValueError("The model has not been fitted yet. Call the `fit` method first.")
        joblib.dump(self.best_estimator_, filename)
        
    def load_model(self, filename):
        """
        Load a model from a file.
        
        Parameters
        ----------
        filename : str
            Path to the saved model file.
            
        Returns
        ----------
        None
        """
        if self.best_estimator_ is None:
            raise ValueError("The model has not been fitted yet. Call the `fit` method first.")
        self.best_estimator_ = joblib.load(filename)
        
    
    def plot_feature_importance(self, feature_names):
        """
        Plot the feature importance from the best estimator.

        Parameters
        ----------
        feature_names : list
            List of feature names in the same order as they appear in the dataset.

        Returns
        ----------
        None
        """
        if self.best_estimator_ is None:
            raise ValueError("The model has not been fitted yet. Call the `fit` method first.")
        
        importances = self.best_estimator_.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance")
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.show()
        
    def cross_val_score_summary(self, X, Y):
        """
        Get a summary of cross-validation scores for the best model.

        Parameters
        ----------
        X : pandas.DataFrame
            Input data (features).
        Y : pandas.Series
            Target variable.

        Returns
        ----------
        summary : dict
            Dictionary with mean and standard deviation of cross-validation scores.
        """
        if self.best_estimator_ is None:
            raise ValueError("The model has not been fitted yet. Call the `fit` method first.")
        
        scores = cross_val_score(self.best_estimator_, X, Y, cv=self.cv, scoring=self.scoring)
        return {
            'mean_score': scores.mean(),
            'std_dev': scores.std(),
            'scores': scores
        }

    def xgboost_best(self, df: pd.DataFrame, target_column: str):
        """Perform grid search to find the best parameters for XGBRegressor."""
        X = df.drop(columns=[target_column])
        Y = df[target_column]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=self.test_size, random_state=42)

        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [ 0.05, 0.1],
            'max_depth': [None, 6],
            'subsample': [0.1,0.5,],
            'colsample_bytree': [0.1,0.5],
            'gamma': [0, 0.1],
            'reg_alpha': [0, 0.1],
            'reg_lambda': [0.1, 0.5]
        }

        model = xgb.XGBRegressor(tree_method='gpu_hist', predictor='gpu_predictor', random_state=42)
        self.grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=self.scoring, cv=self.cv, n_jobs=-1)
        self.grid_search.fit(X_train, Y_train)
        self.best_params_ = self.grid_search.best_params_
        self.best_estimator_ = self.grid_search.best_estimator_

        return self.best_estimator_