from .types import *

def three_w_find_linked_column():

    tool_schema = LlmToolSchema([
        LlmToolSchemaAtribute("column", "string", "The column that is linked to the error"),
        LlmToolSchemaAtribute("extra", "string", "Information of the suspected error and the linked column"),
    ])
    system_message = LlmMessage("system", "You are an assistant that will receive information on a dataset as JSON and the error reason and will find the column that is linked to the error. This column will be used to train a model to predict the error.")
    messages = []
    return tool_schema, system_message, messages