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
        
def detectStdDevOutlier(data, threshold=2):
    """
    Detects outliers based on the standard deviation method.

    Parameters
    ----------
    data: array-like
        The input data to be analyzed.
    threshold: float, optional
        The number of standard deviations from the mean to consider 
        as an outlier. Default is 2.

    Returns
    ----------
    alarm: ndarray
        An array indicating the outliers (0 for values within 
        threshold * stddev, 1 for outliers).
    """
    mean = np.nanmean(data)
    std_dev = np.nanstd(data)
    upper_limit = mean + threshold * std_dev
    lower_limit = mean - threshold * std_dev
    return np.where((data > upper_limit) | (data < lower_limit), 1, 0)


def detectTrend(data, window_size=5):
    """
    Detects trend in the data using a moving average.

    Parameters
    ----------
    data: array-like
        The input data to be analyzed.
    window_size: int
        The size of the moving window to calculate the trend.

    Returns
    ----------
    alarm: ndarray
        An array indicating the trend (1 for positive trend, -1 for negative trend, 0 for no trend).
    """
    if len(data) < window_size:
        return np.zeros(len(data))

    moving_avg = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    trend = np.diff(moving_avg)
    trend_alarm = np.where(trend > 0, 1, np.where(trend < 0, -1, 0))
    
    # Pad the result with zeros for the beginning of the array
    trend_alarm = np.pad(trend_alarm, (window_size-1, 0), mode='constant')
    
    return trend_alarm


def detectMovingWindowOutlier(data, window_size=10, count_limit=1):
    """
    Detects outliers in a moving window.

    Parameters
    ----------
    data: array-like
        The input data to be analyzed.
    window_size: int
        The size of the moving window to analyze.
    count_limit: int
        The maximum number of outliers allowed within the window.

    Returns
    ----------
    alarm: ndarray
        An array indicating if the count of outliers exceeds 
        the count_limit within each window (1 for alarm, 0 for no alarm).
    """
    alarm = np.zeros(len(data))
    for i in range(len(data) - window_size + 1):
        window = data[i:i + window_size]
        if np.count_nonzero(window > np.nanmean(window)) > count_limit:
            alarm[i + window_size - 1] = 1  # Mark the last element in the window
    return alarm

def detectBias(data, expected_value, threshold=0.1):
    """
    Detects bias in the data by comparing the mean to an expected value.

    Parameters
    ----------
    data: array-like
        The input data to be analyzed.
    expected_value: float
        The expected mean value to compare against.
    threshold: float, optional
        The threshold for deviation from the expected value. Default is 0.1.

    Returns
    ----------
    alarm: ndarray
        An array indicating if the bias exceeds the threshold (1 for alarm, 0 for no alarm).
    """
    mean_value = np.nanmean(data)
    return np.where(np.abs(mean_value - expected_value) > threshold, 1, 0)


def detectNelsonRules(data,threshold=1):
    """
    Detects anomalies in the data based on Nelson Rules 1, 2, and 3.

    Parameters
    ----------
    data: array-like
        The input data to be analyzed.

    Returns
    ----------
    alarms: dict
        A dictionary with alarms for each rule (1 for alarm, 0 for no alarm).
    """
    mean_value = np.nanmean(data)
    std_dev = np.nanstd(data)

    #rule 1 = 1 point exceeding the threshold
    rule_1_alarms = np.where(np.abs(data - mean_value) > threshold * std_dev, 1, 0)

    #rule 2 = 2 points in a row exceeding the threshold
    rule_2_alarms = np.zeros_like(data)
    for i in range(1, len(data)):
        if (np.abs(data[i] - mean_value) > threshold * std_dev) and (np.abs(data[i-1] - mean_value) > threshold * std_dev):
            rule_2_alarms[i-1:i+1] = 1  

    # rule 3 = 6 points in a row increasing or decreasing
    rule_3_alarms = np.zeros_like(data)
    count = 0
    for i in range(len(data)):
        if data[i] > mean_value + threshold :
            count += 1
            if count >= 6:
                rule_3_alarms[i-5:i+1] = 1  
        else:
            count = 0  
    alarms = {
        'rule_1': rule_1_alarms,
        'rule_2': rule_2_alarms,
        'rule_3': rule_3_alarms
    }
    
    return alarms

     
