import configparser
import pandas as pd
from typing import Literal, Tuple
import os
import json

from .. import _preprocess as preproc
from .. import _bibmon_tools as b_tools


def load_dataset_ini() -> configparser.ConfigParser:
    """
    Loads the dataset.ini file.

    Parameters
    ----------
    Returns
    ----------
    : configparser.ConfigParser
        Configuration file.
    """

    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), "dataset.ini"))

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


def event_formatter(
    df: pd.DataFrame, config_file: configparser.ConfigParser, n_of_rows: int = 30
) -> dict:
    df_class = df.iloc[0]["class"]
    event_name: str = None
    if df_class == 0:
        event_name = "normal"
    elif df_class < 100:
        event_name = "error"
    else:
        event_name = "transition"

    # Round numeric values to 4 decimal places before converting them to JSON
    rounded_df = df.round(4)

    average_values = json.dumps(rounded_df.mean().apply(lambda x: f"{x:.4f}").to_dict())
    standard_deviation = json.dumps(
        rounded_df.std().apply(lambda x: f"{x:.4f}").to_dict()
    )

    # Convert to string with rounded values
    head = json.dumps(rounded_df.head(n_of_rows).reset_index().map(str).to_dict())
    tail = json.dumps(rounded_df.tail(n_of_rows).reset_index().map(str).to_dict())

    return {
        "event_name": event_name,
        "average_values": average_values,
        "standard_deviation": standard_deviation,
        "head": head,
        "tail": tail,
    }


###############################################################################


def format_for_llm_prediction(
    df: pd.DataFrame,
    config_file: configparser.ConfigParser,
    class_id: int,
    n_of_rows: int = 30,
) -> dict:
    """
    Formats the dataset for LLM prediction.

    The output is:
    {
        "event_name": str,
        "event_description": str,
        "columns_and_description": dict,
        "data": [
            {'event_name': str, 'average_values': str, 'standard_deviation': str, 'head': str, 'tail': str},
            {'event_name': str, 'average_values': str, 'standard_deviation': str, 'head': str, 'tail': str},
            ...
        ]
    }

    Parameters
    ----------
    df: pandas.DataFrame
        Data to be formatted.

    config_file: configparser.ConfigParser
        Configuration file.

    class_id: int
        Class ID of the dataset.

    n_of_rows: int
        Number of rows to be displayed in the head and tail.

    Returns
    ----------
    : dict
        Formatted data.
    """

    event_names = config_file.get("EVENTS", "NAMES")

    event_names_list = [
        event.strip() for event in event_names.replace("\n", "").split(",")
    ]

    event_name = event_names_list[class_id]
    event_description = config_file.get(event_name, "DESCRIPTION")

    properties_section = config_file["PARQUET_FILE_PROPERTIES"]

    # Create a dictionary with {name: description}
    properties_dict = {key: value.strip() for key, value in properties_section.items()}

    # remove columns from the properties_dict that are not in the dataset
    columns_to_remove = df.keys()
    properties_dict = {
        key: value for key, value in properties_dict.items() if key in columns_to_remove
    }

    df_processed = preproc.PreProcess(f_pp=["ffill_nan"]).apply(df)

    transitions = b_tools.find_df_transitions(df_processed, 1, "number", "class")

    dataset_data: list[pd.DataFrame] = []

    last_transition_offset = 0
    for t in transitions:
        target_df = df_processed.iloc[last_transition_offset:t]
        dataset_data.append(event_formatter(target_df, config_file, n_of_rows))
        last_transition_offset = t

    # Last split
    target_df = df_processed.iloc[last_transition_offset:]
    dataset_data.append(event_formatter(target_df, config_file, n_of_rows))

    return {
        "event_name": event_name,
        "event_description": event_description,
        "columns_and_description": properties_dict,
        "data": dataset_data,
    }
