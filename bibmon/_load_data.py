import os
import pandas as pd
import importlib.resources as pkg_resources
from typing import Tuple
import requests
import io
import configparser
from tqdm import tqdm


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

def load_3w(dataset_class: int = 8, dataset_name: str = "WELL-00019_20120601165020.parquet") -> Tuple[pd.DataFrame, configparser.ConfigParser, int]:
    """
    Load the '3W-8' benchmark data. If it receives a different class or dataset name, it will try to download from the repository.

    Warning: This assumes that the dataset is available in the repository. If the dataset is not available, the function will raise an error. This will not download a new config file.

    Parameters
    ----------
    dataset_class: int, optional
        Identifier of the dataset class.
    dataset_name: str, optional
        Name of the dataset file.
    Returns
    ----------
    : pandas.DataFrame
        Process data.
    : configparser.ConfigParser
        Configuration file.
    : int
        Identifier of the dataset class.

    """
    data_frame: pd.DataFrame = None
    ini = three_w.tools.load_dataset_ini()

    if dataset_class == 8 and dataset_name == "WELL-00019_20120601165020.parquet":
        with pkg_resources.path(three_w, dataset_name) as file:
            data_frame = pd.read_parquet(
                file,
                engine=ini.get("PARQUET_SETTINGS", "PARQUET_ENGINE"),
            )
    else:
        data_set_url = f'https://github.com/petrobras/3W/raw/refs/heads/main/dataset/{dataset_class}/{dataset_name}'

        print(f"Downloading dataset from {data_set_url}")

        # Send a head request to know the total file size in advance
        response = requests.head(data_set_url)
        file_size = int(response.headers.get('content-length', 0))  # Get file size from headers

        # Download the file with a progress bar
        response = requests.get(data_set_url, stream=True)
        response.raise_for_status()  # Check if the request was successful

        parquet_file = io.BytesIO()

        chunk_size = 1024
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=dataset_name) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                parquet_file.write(chunk)
                pbar.update(len(chunk))

        # Reset the BytesIO buffer's position to the beginning
        parquet_file.seek(0)

        data_frame = pd.read_parquet(
            parquet_file,
            engine=ini.get("PARQUET_SETTINGS", "PARQUET_ENGINE"),
        )

    if data_frame is None:
        raise ValueError("The dataset could not be loaded.")
    if ini is None:
        raise ValueError("The dataset configuration file could not be loaded.")

    return (
        data_frame,
        ini,
        int(dataset_class),
    )
