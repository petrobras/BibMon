import os
import pandas as pd
import importlib.resources as pkg_resources
from typing import Literal

from . import _bibmon_tools as b_tools
from . import real_process_data, tennessee_eastman, three_w

###############################################################################


def load_tennessee_eastman(train_id=0, test_id=0):
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

    tags1 = ["XMEAS(" + str(ii) + ")" for ii in range(1, 42)]
    tags2 = ["XMV(" + str(ii) + ")" for ii in range(1, 12)]
    tags = tags1 + tags2

    file_train = f"d{train_id}.dat"
    file_test = f"d{test_id}_te.dat"

    if len(file_train) == 6:
        file_train = file_train[:2] + "0" + file_train[2:]

    if len(file_test) == 9:
        file_test = file_test[:1] + "0" + file_test[1:]

    with pkg_resources.path(tennessee_eastman, file_train) as filepath:

        if file_train == "d00.dat":

            tmp1 = pd.read_csv(filepath, sep="\t", names=["0"])
            tmp2 = pd.DataFrame(
                [tmp1.T.iloc[0, i].strip() for i in range(tmp1.shape[0])]
            )
            train_df = pd.DataFrame()

            for ii in range(52):
                train_df[tags[ii]] = [float(s) for s in tmp2[0][ii].split("  ")]

            train_df = b_tools.create_df_with_dates(train_df, freq="3min")

        else:

            train_df = b_tools.create_df_with_dates(
                pd.read_csv(filepath, sep="\s+", names=tags), freq="3min"
            )

    with pkg_resources.path(tennessee_eastman, file_test) as filepath:

        test_df = b_tools.create_df_with_dates(
            pd.read_csv(filepath, sep="\s+", names=tags),
            start="2020-02-01 00:00:00",
            freq="3min",
        )

    return train_df, test_df


###############################################################################


def load_real_data():
    """
    Load a sample of real process data.
    The variables have been anonymized for availability in the library.

    Returns
    ----------
    : pandas.DataFrame
        Process data.
    """

    with pkg_resources.path(real_process_data, "real_process_data.csv") as file:
        return pd.read_csv(file, index_col=0, parse_dates=True)


###############################################################################

AVAILABLE_3W_CLASSES = ["8"]


def load_3w(dataset_class: Literal["8"] = "8"):
    """
    Load the '3W-8' benchmark data.

    Parameters
    ----------
    dataset_class: string
        Identifier of the dataset class.
        Available classes: '8'.
    Returns
    ----------
    : pandas.DataFrame
        Process data.
    : configparser.ConfigParser
        Configuration file.
    """

    if dataset_class != "8":
        raise ValueError(
            f"Dataset class not available. Available classes: {AVAILABLE_3W_CLASSES}"
        )

    with pkg_resources.path(three_w, "WELL-00019_20120601165020.parquet") as file:
        with pkg_resources.path(three_w, "dataset.ini") as path:
            ini = three_w.tools.load_dataset_ini(path)
            return (
                pd.read_parquet(
                    file,
                    engine=ini.get("PARQUET_SETTINGS", "PARQUET_ENGINE"),
                ),
                ini,
            )
