import json
from .types import *


def three_w_find_linked_column(data: dict):
    """
    Preset for LLM using the Three W dataset to find the column linked to a specific error.

    Tip: spread the returned tuple into the chat_completion_json_tool method with the * operator.

    Parameters
    ----------
    data: dict
        The dataset in JSON format, including an error event and its description, as well as multiple columns and their descriptions.

    Returns
    ----------
    tuple
        A tuple containing the tool schema, system message, and messages to be used for the LLM completion.
    """

    prompt = """
    You are a highly specialized assistant tasked with analyzing datasets to identify the column linked to a specific error. You will receive the dataset in JSON format, including an error event and its description, as well as multiple columns and their descriptions. Your role is to find and return only the column directly associated with the error. To do so, follow these strict guidelines:
    Failure to follow these instructions will result in an invalid response.

    1. **Do not consider the "class" or "state" columns**—these are manually labeled with operational status and should never be used for error prediction.
    2. **Examine all provided columns** and determine which one correlates with the provided error event, based on the descriptions and any available data patterns.
    e. Do **not** provide any additional information, context, or outputs beyond what is required by the schema. Ensure your response is concise and follows the JSON format exactly.

    4. **Respond only in JSON format**, adhering strictly to the following schema: {"column": "column_name", "extra": "extra_information"}.

    5. **Do not provide any additional information** beyond the required schema. Your response should only include the column name and any extra information, if applicable.
    Here is the dataset you need to analyze:

    The possible columns are: 
    """

    tool_schema = None
    system_message = LlmMessage(
        "system",
        prompt,
    )

    data_as_string = json.dumps(data)

    # removes \
    data_as_string = data_as_string.replace("\\", "")

    messages = [
        LlmMessage(
            "user",
            '{"event_name": "PIPE_OVERHEAT", "event_description": "Pipe Overheat Fault in System", "columns_and_description": {"timestamp": "Time when the observation was recorded", "pipe_temp": "Temperature of the pipe [oC]", "flow_rate": "Flow rate through the pipe [m3/s]", "pressure_upstream": "Upstream pressure [Pa]", "pressure_downstream": "Downstream pressure [Pa]", "valve_status": "Status of the control valve [0=closed, 1=open]", "overheat_flag": "Flag indicating pipe overheat condition [0=no, 1=yes]", "ambient_temp": "Ambient temperature [oC]", "system_status": "Overall system operational state", "class": "Observation label (fault or normal)"}, "data": [{"event_name": "normal", "average_values": "{"pipe_temp": "150.25", "flow_rate": "0.85", "pressure_upstream": "500000", "pressure_downstream": "400000", "valve_status": "1.0", "ambient_temp": "25.7", "class": "0"}", "standard_deviation": "{"pipe_temp": "5.34", "flow_rate": "0.02", "pressure_upstream": "10000", "pressure_downstream": "9500", "valve_status": "0.0", "ambient_temp": "0.5", "class": "0"}", "head": "{"timestamp": {"0": "2024-01-01 00:00:01", "1": "2024-01-01 00:00:02", "2": "2024-01-01 00:00:03", "3": "2024-01-01 00:00:04", "4": "2024-01-01 00:00:05"}, "pipe_temp": {"0": "149.5", "1": "149.7", "2": "149.8", "3": "149.9", "4": "150.0"}, "pressure_upstream": {"0": "500100", "1": "500000", "2": "499900", "3": "499800", "4": "499700"}, "flow_rate": {"0": "0.85", "1": "0.85", "2": "0.85", "3": "0.85", "4": "0.85"}, "class": {"0": "0", "1": "0", "2": "0", "3": "0", "4": "0"}}", "tail": "{"timestamp": {"0": "2024-01-01 23:59:56", "1": "2024-01-01 23:59:57", "2": "2024-01-01 23:59:58", "3": "2024-01-01 23:59:59"}, "pipe_temp": {"0": "150.2", "1": "150.4", "2": "150.5", "3": "150.6"}, "pressure_upstream": {"0": "499600", "1": "499500", "2": "499400", "3": "499300"}, "class": {"0": "0", "1": "0", "2": "0", "3": "0"}}"}, {"event_name": "fault", "average_values": "{"pipe_temp": "175.30", "flow_rate": "0.60", "pressure_upstream": "480000", "pressure_downstream": "370000", "valve_status": "1.0", "ambient_temp": "26.1", "class": "1"}", "standard_deviation": "{"pipe_temp": "7.21", "flow_rate": "0.01", "pressure_upstream": "15000", "pressure_downstream": "14000", "valve_status": "0.0", "ambient_temp": "0.4", "class": "1"}", "head": "{"timestamp": {"0": "2024-01-02 00:00:01", "1": "2024-01-02 00:00:02", "2": "2024-01-02 00:00:03", "3": "2024-01-02 00:00:04", "4": "2024-01-02 00:00:05"}, "pipe_temp": {"0": "175.1", "1": "175.2", "2": "175.3", "3": "175.4", "4": "175.5"}, "pressure_upstream": {"0": "480100", "1": "480000", "2": "479900", "3": "479800", "4": "479700"}, "flow_rate": {"0": "0.60", "1": "0.60", "2": "0.60", "3": "0.60", "4": "0.60"}, "class": {"0": "1", "1": "1", "2": "1", "3": "1", "4": "1"}}"}]}',
        ),
        LlmMessage(
            "assistant",
            '{"column": "pipe_temp", "extra": "The pipe_temp column is directly associated with the PIPE_OVERHEAT event."}',
        ),
        LlmMessage(
            "user",
            data_as_string,
        ),
    ]
    return tool_schema, system_message, messages
