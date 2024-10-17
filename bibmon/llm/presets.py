import json
from .types import *


def three_w_find_linked_column(data: dict):

    tool_schema = LlmToolSchema(
        [
            LlmToolSchemaAtribute(
                "column", "string", "The column that is linked to the error"
            ),
            LlmToolSchemaAtribute(
                "extra",
                "string",
                "Information of the suspected error and the linked column",
            ),
        ]
    )
    system_message = LlmMessage(
        "system",
        "You are an assistant that will receive information on a dataset as JSON and the error reason and will find the column that is linked to the error. This column will be used to train a model to predict the error. You shall not use the class or state columns as those are hand labeled with the current condition, 0 = normal, 1-99 = error, 101-199 = transition.",
    )


    data_as_string = json.dumps(data)

    # remove \
    data_as_string = data_as_string.replace("\\", "")
    messages = [
        LlmMessage(
            "user",
            data_as_string,
        )
    ]
    return tool_schema, system_message, messages
