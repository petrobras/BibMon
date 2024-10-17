import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize

def detect_outliers_iqr(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Detects outliers in a DataFrame using the IQR (Interquartile Range) method.

    Args:
        df (pd.DataFrame): DataFrame with the data.
        cols (list): List of columns for which outliers will be detected.

    Returns:
        pd.DataFrame: DataFrame with outliers flagged as 1 and other points as 0.
    """

    df_outliers = df.copy()
    for col in cols:
        Q1 = df_outliers[col].quantile(0.25)
        Q3 = df_outliers[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_outliers[col] = ((df_outliers[col] < lower_bound) | (df_outliers[col] > upper_bound)).astype(int)
    return df_outliers

def remove_outliers(df: pd.DataFrame, cols: list, method: str = 'remove') -> pd.DataFrame:
    """
    Removes or handles outliers in a DataFrame using the IQR (Interquartile Range) method.

    Args:
        df (pd.DataFrame): DataFrame with the data.
        cols (list): List of columns for which outliers will be removed or handled.
        method (str): Method for handling outliers. Can be 'remove' (removes outliers),
                      'median' (replaces outliers with the median), or 'winsorize' (applies winsorization).
                      Default: 'remove'.

    Returns:
        pd.DataFrame: DataFrame with outliers removed or handled.
    """

    df_outliers = df.copy()
    for col in cols:
        Q1 = df_outliers[col].quantile(0.25)
        Q3 = df_outliers[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        if method == 'remove':
            df_outliers = df_outliers[(df_outliers[col] >= lower_bound) & (df_outliers[col] <= upper_bound)]
        elif method == 'median':
            median = df_outliers[col].median()
            df_outliers.loc[(df_outliers[col] < lower_bound) | (df_outliers[col] > upper_bound), col] = median
        elif method == 'winsorize':
            df_outliers[col] = winsorize(df_outliers[col], limits=[0.05, 0.05])
        else:
            raise ValueError("Invalid method. Choose between 'remove', 'median', or 'winsorize'.")

    return df_outliers