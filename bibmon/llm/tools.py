from openai.types.chat.chat_completion import ChatCompletion
from .types import *


def parse_chat_completion(completion: ChatCompletion) -> List[LlmMessage]:
    """
    Parses a chat completion response from the language model into a list of messages.

    Parameters
    ----------
    completion: ChatCompletion
        The chat completion response from the language model.

    Returns
    ----------
    list of dict
        A list of messages (in dictionary format) forming the conversation history. Each dictionary has 'role' and 'content' keys.
        Example: [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Tell me a joke."}]
    """
    return completion.choices[0].message.content