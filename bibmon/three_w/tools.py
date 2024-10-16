import configparser
import pandas as pd
from .. import _preprocess as preproc
from .. import _bibmon_tools as b_tools
from typing import Literal, Tuple


def load_dataset_ini(dataset_ini_path) -> configparser.ConfigParser:
    """
    Loads the dataset.ini file.

    Parameters
    ----------
    dataset_ini_path: string
        Path to the dataset.ini file.
    Returns
    ----------
    : configparser.ConfigParser
        Configuration file.
    """

    config = configparser.ConfigParser()
    config.read(dataset_ini_path)

    return config

###############################################################################

def split_dataset(
    dataFrame: pd.DataFrame,
    config_file: configparser.ConfigParser,
    training_percentage: float = 0.8,
    validation_percentage: float = 0.2,
    split_on: Literal["transition_state", "error_state"] = "transition_state",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a DataFrame into training, validation and test sets.

    The split is done according to the transition_state or error_state, the sum of training and validation percentages must equal 1.

    Parameters
    ----------
    DataFrame: pandas.DataFrame
        Data to be split.
    config_file: configparser.ConfigParser
        Configuration file.
    training_percentage: float
        Percentage of the training set.
    validation_percentage: float
        Percentage of the validation set.
    split_on: string
        Split the data on the transition_state or error_state.
    Returns
    ----------
    : tuple of pandas.DataFrames
        Training, validation and test sets.
    """

    if training_percentage + validation_percentage != 1:
        raise ValueError(
            "The sum of training and validation percentages must equals 1."
        )

    df_processed = preproc.PreProcess(f_pp=["ffill_nan"]).apply(dataFrame)

    transitions = b_tools.find_df_transitions(df_processed, 1, "number", "class")

    split_index = 0

    for t in transitions:
        if t == 0:
            continue

        transition_class = int(df_processed.iloc[t]["class"])
        transient_offset = int(config_file.get("EVENTS", "TRANSIENT_OFFSET"))

        if transition_class > transient_offset and split_on == "transition_state":
            split_index = t
            break
        if transition_class < transient_offset and split_on == "error_state":
            split_index = t
            break

    if split_index == 0:
        raise ValueError("No split index found.")

    train_and_validation_df = df_processed.iloc[:split_index]
    test_df = df_processed.iloc[split_index:]

    splitted_df = b_tools.split_df_percentages(
        train_and_validation_df, [training_percentage, validation_percentage]
    )

    train_df = splitted_df[0]
    validation_df = splitted_df[1]

    return train_df, validation_df, test_df

###############################################################################

def format_for_llm_prediction(df: pd.DataFrame, config_file: configparser.ConfigParser):
    pass