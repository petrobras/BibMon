import os
import pandas as pd
import importlib.resources as pkg_resources

from ._bibmon_tools import create_df_with_dates
from . import real_process_data, tennessee_eastman 

###############################################################################

def load_tennessee_eastman (train_id = 0, test_id = 0):
    """
    Load the 'Tennessee Eastman Process' benchmark data.

    Parameters
    ----------
    train_id: int, optional
        Identifier of the training data.
        No fault: 0. With faults: 1 to 20.
    test_id: int, optional
        Identifier of the test data.
        No fault: 0. With faults: 1 to 20.
    Returns
    ----------                
    train_df: pandas.DataFrame
        Training data.
    test_df: pandas.DataFrame
        Test data.
    """    
        
    tags1 = ['XMEAS('+str(ii)+')' for ii in range(1,42)]
    tags2 = ['XMV('+str(ii)+')' for ii in range(1,12)]
    tags = tags1 + tags2
    
    file_train = f'd{train_id}.dat'
    file_test = f'd{test_id}_te.dat'

    if len(file_train) == 6:
        file_train = file_train[:2]+'0'+file_train[2:]
        
    if len(file_test) == 9:
        file_test = file_test[:1]+'0'+file_test[1:]

    with pkg_resources.path(tennessee_eastman, file_train) as filepath:

        if file_train == 'd00.dat':
            
            tmp1 = pd.read_csv(filepath,sep='\t',
                               names=['0'])
            tmp2 = pd.DataFrame([tmp1.T.iloc[0,i].strip() for 
                                i in range(tmp1.shape[0])])
            train_df = pd.DataFrame()

            for ii in range(52):
                train_df[tags[ii]]=[float(s) for s in tmp2[0][ii].split('  ')]
                
            train_df = create_df_with_dates(train_df, 
                                            freq = '3min')
            
        else:

            train_df = create_df_with_dates(filepath, sep='\s+',names=tags,
                                            freq = '3min')

    with pkg_resources.path(tennessee_eastman, file_test) as filepath:

        test_df = create_df_with_dates(pd.read_csv(filepath, 
                                                   sep = '\s+', 
                                                   names = tags),
                                       start = '2020-02-01 00:00:00',
                                       freq = '3min')

    return train_df, test_df

###############################################################################

def load_real_data ():
    """
    Load a sample of real process data.
    The variables have been anonymized for availability in the library.
    
    Returns
    ----------                
    : pandas.DataFrame
        Process data.
    """    

    with pkg_resources.path(real_process_data,'real_process_data.csv') as file:
        return pd.read_csv(file,index_col = 0, parse_dates = True)