import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.feature_selection import mutual_info_regression

    
def bestColumn_pearson_spearman(df):
    """
    This function calculates the Pearson and Spearman correlation coefficients
    between all columns in the dataframe. It identifies the column with the highest
    average correlation (positive) with other columns using both correlation methods.
    
    Pearson measures linear relationships, while Spearman measures monotonic relationships.
    
    Returns:
        dict: A dictionary containing the best column for each correlation method.
    """
    pearson_corr = df.corr(method="pearson")
    spearman_corr = df.corr(method="spearman")
    
    # Calculate the average correlation of each column with the others
    spearman_correlation_mean = spearman_corr.mean().sort_values(ascending=False)
    y_column_spearman = spearman_correlation_mean.index[1]
    
    peasron_correlation_mean = pearson_corr.mean().sort_values(ascending=False)
    y_column_pearson = peasron_correlation_mean.index[1]
    
    #return a dictionary with the best column for each correlation method
    return {'pearson': y_column_pearson, 'spearman': y_column_spearman}

def bestColumn_with_least_mae_or_r2(df):
    """
    This function evaluates each column as a target variable for regression
    and calculates the Mean Absolute Error (MAE) and R-squared (R²) scores
    for predictions made by an XGBoost regressor.
    
    It identifies which column minimizes MAE and maximizes R², providing insights
    on the best target variable based on predictive performance.
    
    Returns:
        dict: A dictionary with sorted results for MAE and R².
    """
    performance_mae = {}
    performance_r2 = {}
    
    for columns in df.columns:
        X = df.drop(columns, axis=1)
        Y = df[columns]
        
        Y = Y.fillna(Y.mean())
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        #using XGBRegressor to predict the column, because it is faster than the other regressors that dosent not use GPU
        model = xgb.XGBRegressor(random_state=42, n_jobs=-1, max_depth=None, tree_method='gpu_hist', predictor='gpu_predictor')
        model.fit(X_train, Y_train)
        
        Y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(Y_test, Y_pred)
        r2 = r2_score(Y_test, Y_pred)
        
        performance_mae[columns] = mae 
        performance_r2[columns] = r2  

    
    performance_mae_sorted = sorted(performance_mae.items(), key=lambda x: x[1], reverse=False)
    performance_r2_sorted = sorted(performance_r2.items(), key=lambda x: x[1], reverse=True)

    return {'r2': performance_r2_sorted,'mae': performance_mae_sorted}

def bestColumn_feature_importance(df):
    
    """
    This function evaluates the importance of each feature by training an XGBoost regressor
    for each column and computing the average feature importance. This helps to identify
    which columns contribute most to predicting the target variable.

    Returns:
        list: A sorted list of feature importances for each column.
    """
    
    feature_importances = {}

    for columns in df.columns:
        X = df.drop(columns, axis=1)
        Y = df[columns]
        
        Y = Y.fillna(Y.mean())
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        model = xgb.XGBRegressor(random_state=42, n_jobs=-1, max_depth=None, tree_method='gpu_hist', predictor='gpu_predictor')
        model.fit(X_train, Y_train)
        
        # Importance of the features
        feature_importance = model.feature_importances_
        feature_importances[columns] = feature_importance.mean()

    feature_importances_sorted = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    
    return feature_importances_sorted


def bestColumn_mutual_information(df):
    
    """
    This function calculates the mutual information scores between each column
    and the other columns in the dataframe. Mutual information quantifies the
    amount of information obtained about one variable through the other, helping
    to determine which features are most informative for predicting the target variable.

    Returns:
        list: A sorted list of mutual information scores for each column.
    """
    
    mi_scores = {}
    
    for columns in df.columns:
        X = df.drop(columns, axis=1)
        Y = df[columns]

        mi = mutual_info_regression(X, Y)
        mi_scores[columns] = mi.mean()  #

    
    mi_sorted = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)
    
    return mi_sorted