from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
import json

from .types import *


class LLM:
    def __init__(
        self,
        endpoint_url: str = "http://127.0.0.1:5000/v1",
        api_key: str = "NO_API_KEY",
        model: str = "llama3",
    ):
        self.client = OpenAI(base_url=endpoint_url, api_key=api_key)
        self._model = model

    def chat_completion(
        self,
        system_message: LlmMessage = LlmMessage(
            "system", "You are a helpful assistant."
        ),
        messages: List[LlmMessage] = [],
        temperature: float = 0.1,
        top_p: float = 0.3,
        frequency_penalty: float = 0.0,
        presence_penalty: float = -1.0,
        **other_params,
    ) -> ChatCompletion:
        """
        Generates a chat completion using a language model, with adjustable parameters
        for controlling the model's behavior, output length, and randomness.

        Parameters
        ----------
        system_message: str, optional
            The initial system message to start the conversation. Default is "You are a helpful assistant."

        messages: list of dict
            A list of messages (in dictionary format) forming the conversation history. Each dictionary must have 'role' and 'content' keys.
            Example: [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Tell me a joke."}]

        temperature: float, optional
            Controls the randomness of the response. Lower values make output more deterministic, while higher values make it more creative.
            Default is 0.7.

        top_p: float, optional
            Nucleus sampling. Limits the token pool to those with cumulative probability up to this value. Lower values make output more focused.
            Default is 0.9.

        frequency_penalty: float, optional
            A number that reduces the model's tendency to repeat the same phrases or words. A higher value reduces repetition.
            Range: -2.0 to 2.0. Default is 0.0 (no penalty).

        presence_penalty: float, optional
            Encourages the model to introduce new topics that haven’t been mentioned in the conversation. A higher value increases topic diversity.
            Range: -2.0 to 2.0. Default is 0.0 (no penalty).

        **other_params: dict, optional
            Additional parameters to forward to the completion call. Examples include `stop` (a list of stop sequences) or `logit_bias` (a dictionary for token biasing).

        Returns
        ----------
        ChatCompletion
            The generated chat completion response from the language model, typically containing the generated message and metadata.

        Raises
        ----------
        ValueError
            If the system_message is empty.
        """

        if not system_message:
            raise ValueError("The system message cannot be empty.")

        dict_messages = [message.to_dict() for message in [system_message] + messages]

        print(dict_messages)
        
        return self.client.chat.completions.create(
            messages=dict_messages,
            model=self._model,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            **other_params,  # Forward other optional parameters
        )

    def chat_completion_json_tool(
        self,
        tool_schema: LlmToolSchema,
        system_message: LlmMessage = LlmMessage(
            "system", "You are a helpful assistant."
        ),
        messages: List[LlmMessage] = [],
        **other_params,
    ):

        system_message.content += (
            "You will only respond using the following JSON schema: "
            + json.dumps(tool_schema.to_dict())
        )

        return self.chat_completion(
            system_message, messages, response_format="json", **other_params
        )

    def parse_chat_completion(self, completion: ChatCompletion) -> List[LlmMessage]:
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
