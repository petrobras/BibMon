import numpy as np

###############################################################################

def detecOutlier(data, lim, count = False, count_limit = 1):
    """
    Detects outliers in the given data using a specified limit.

    Parameters
    ----------
    data: array-like
        The input data to be analyzed.
    lim: float
        The limit value used to detect outliers.
    count: bool, optional
        If True, counts the number of outliers 
        exceeding the limit. Default is False.
    count_limit: int, optional
        The maximum number of outliers allowed. 
        Default is 1.

    Returns
    ----------
    alarm: ndarray or int
        If count is False, returns an array indicating 
        the outliers (0 for values below or equal to lim,
        1 for values above lim).
        If count is True, returns the number of outliers 
        exceeding the limit.
    """

    if np.isnan(data).any():
        data = np.nan_to_num(data)

    if count == False:
        alarm = np.copy(data)
        alarm = np.where(alarm<=lim, 0, alarm)
        alarm = np.where(alarm>lim, 1, alarm)
        return alarm
    else:
        alarm = 0
        local_count = np.count_nonzero(data > lim)
        if local_count > count_limit:
            alarm = +1
        return alarm
        
