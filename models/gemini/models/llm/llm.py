import base64
import json
from collections.abc import Generator
from typing import Any, Iterator, Mapping, Optional, Sequence, Union

from dify_plugin.entities.model.llm import (
    LLMResult,
    LLMResultChunk,
    LLMResultChunkDelta,
)
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    AudioPromptMessageContent,
    DocumentPromptMessageContent,
    ImagePromptMessageContent,
    PromptMessage,
    PromptMessageContent,
    PromptMessageContentType,
    PromptMessageTool,
    SystemPromptMessage,
    TextPromptMessageContent,
    ToolPromptMessage,
    UserPromptMessage,
    VideoPromptMessageContent,
)
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeBadRequestError,
    InvokeConnectionError,
    InvokeError,
    InvokeServerUnavailableError,
)
from dify_plugin.interfaces.model.large_language_model import LargeLanguageModel
from google import genai
from google.genai import errors, types

from .utils import FileCache

file_cache = FileCache()


class GoogleLargeLanguageModel(LargeLanguageModel):
    def _invoke(
        self,
        model: str,
        credentials: Mapping[str, Any],
        prompt_messages: Sequence[PromptMessage],
        model_parameters: Mapping[str, Any],
        tools: Optional[Sequence[PromptMessageTool]] = None,
        stop: Optional[Sequence[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        _ = user
        return self._generate(
            model=model,
            credentials=credentials,
            prompt_messages=prompt_messages,
            model_parameters=model_parameters,
            tools=tools,
            stop=stop,
            stream=stream,
        )

    def get_num_tokens(
        self,
        model: str,
        credentials: Mapping[str, Any],
        prompt_messages: Sequence[PromptMessage],
        tools: Optional[Sequence[PromptMessageTool]] = None,
    ) -> int:
        """
        Get number of tokens for given prompt messages

        :param model: model name
        :param credentials: model credentials
        :param prompt_messages: prompt messages
        :param tools: tools for tool calling
        :return:md = genai.GenerativeModel(model)
        """
        prompt = self._convert_messages_to_prompt(prompt_messages)
        return self._get_num_tokens_by_gpt2(prompt)

    def _convert_messages_to_prompt(self, messages: Sequence[PromptMessage]) -> str:
        """
        Format a list of messages into a full prompt for the Google model

        :param messages: List of PromptMessage to combine.
        :return: Combined string with necessary human_prompt and ai_prompt tags.
        """
        messages = list(messages)
        text = "".join(
            (self._convert_one_message_to_text(message) for message in messages)
        )
        return text.rstrip()

    def _convert_to_tools(self, tools: Sequence[PromptMessageTool]) -> types.Tool:
        function_declarations = []
        for tool in tools:
            properties = {}
            for key, value in tool.parameters.get("properties", {}).items():
                properties[key] = types.Schema(
                    type=types.Type.STRING,
                    description=value.get("description", ""),
                    enum=value.get("enum", []),
                )
            if properties:
                parameters = types.Schema(
                    properties=properties,
                    required=tool.parameters.get("required", []),
                    type=types.Type.OBJECT,
                )
            else:
                parameters = None
            function_declaration = types.FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters=parameters,
            )
            function_declarations.append(function_declaration)
        return types.Tool(function_declarations=function_declarations)

    def validate_credentials(self, model: str, credentials: Mapping[str, Any]) -> None:
        try:
            ping_message = UserPromptMessage(content="ping")
            self._generate(
                model=model,
                credentials=credentials,
                prompt_messages=[ping_message],
                stream=False,
                model_parameters={"max_output_tokens": 5},
            )
        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex))

    def _generate(
        self,
        model: str,
        credentials: Mapping[str, Any],
        prompt_messages: Sequence[PromptMessage],
        model_parameters: Mapping[str, Any],
        tools: Optional[Sequence[PromptMessageTool]] = None,
        stop: Optional[Sequence[str]] = None,
        stream: bool = True,
    ) -> Union[LLMResult, Generator]:
        # Copy model parameters to avoid modifying the original dict
        model_configuration = dict(model_parameters)

        # Create a google genai client
        api_key = credentials.get("google_api_key")
        if not api_key or not isinstance(api_key, str):
            raise InvokeError("The google api key is required.")
        client = genai.Client(api_key=api_key)

        # JSON Schema
        if json_schema := model_configuration.pop("json_schema", None):
            try:
                json_schema = json.loads(json_schema)
            except Exception:
                raise InvokeError("Invalid JSON Schema")
            if tools:
                raise InvokeError(
                    "gemini not support use Tools and JSON Schema at same time"
                )
            model_configuration["response_mime_type"] = "application/json"
            model_configuration["response_schema"] = json_schema

        # Stop
        if stop:
            model_configuration["stop_sequences"] = stop

        # System promtp
        system_instruction = self._get_system_instruction(
            prompt_messages=prompt_messages
        )
        if system_instruction:
            model_configuration["system_instruction"] = system_instruction

        # Build contents
        contents = self._convert_to_contents(prompt_messages=prompt_messages)
        if len(contents) == 0:
            raise InvokeError(
                "The user prompt message is required. You only add a system prompt message."
            )

        # Tools
        if tools:
            model_configuration["tools"] = self._convert_to_tools(tools)

        config = types.GenerateContentConfig(
            response_modalities=["Text", "Image"],
            **model_configuration,
        )

        if stream:
            response = client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=config,
            )
            return self._handle_generate_stream_response(
                model=model,
                credentials=credentials,
                response=response,
                prompt_messages=prompt_messages,
            )
        else:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
                # request_options={"timeout": 600},
            )
            return self._handle_generate_response(
                model=model,
                credentials=credentials,
                response=response,
                prompt_messages=prompt_messages,
            )

    def _convert_to_contents(
        self, prompt_messages: Sequence[PromptMessage]
    ) -> Sequence[types.ContentDict]:
        contents: list[types.ContentDict] = []
        for prompt in filter(
            lambda prompt: not isinstance(prompt, SystemPromptMessage), prompt_messages
        ):
            content = self._convert_to_content_dict(prompt)
            if len(contents) > 0 and contents[-1].get("role") == content.get("role"):
                parts = contents[-1].get("parts") or []
                parts.extend(content.get("parts") or [])
                contents[-1] = types.ContentDict(
                    parts=parts,
                    role=content.get("role"),
                )
                continue
            contents.append(content)
        return contents

    def _get_system_instruction(
        self, *, prompt_messages: Sequence[PromptMessage]
    ) -> str:
        system_instruction = ""
        if len(prompt_messages) > 0 and isinstance(
            prompt_messages[0], SystemPromptMessage
        ):
            prompt = prompt_messages[0]
            if isinstance(prompt.content, str):
                system_instruction = prompt.content
            elif isinstance(prompt.content, list):
                system_instruction = ""
                for content in prompt.content:
                    if isinstance(content, TextPromptMessageContent):
                        system_instruction += content.data
                    else:
                        raise InvokeError(
                            "system prompt content does not support image, document, video, audio"
                        )
            else:
                raise InvokeError("system prompt content must be a string or a list")
        return system_instruction

    def _handle_generate_response(
        self,
        model: str,
        credentials: Mapping[str, Any],
        response: types.GenerateContentResponse,
        prompt_messages: Sequence[PromptMessage],
    ) -> LLMResult:
        assistant_prompt_message = AssistantPromptMessage(content=response.text)
        if response.usage_metadata:
            prompt_tokens = response.usage_metadata.prompt_token_count
            completion_tokens = response.usage_metadata.candidates_token_count
            if prompt_tokens is None:
                raise ValueError("prompt_tokens is None")
            if completion_tokens is None:
                raise ValueError("completion_tokens is None")
        else:
            prompt_tokens = self.get_num_tokens(
                model=model,
                credentials=credentials,
                prompt_messages=prompt_messages,
            )
            completion_tokens = self.get_num_tokens(
                model=model,
                credentials=credentials,
                prompt_messages=[assistant_prompt_message],
            )
        usage = self._calc_response_usage(
            model=model,
            credentials=dict(credentials),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        result = LLMResult(
            model=model,
            prompt_messages=list(prompt_messages),
            message=assistant_prompt_message,
            usage=usage,
        )
        return result

    def _handle_generate_stream_response(
        self,
        model: str,
        credentials: Mapping[str, Any],
        response: Iterator[types.GenerateContentResponse],
        prompt_messages: Sequence[PromptMessage],
    ) -> Generator:
        index = -1
        prompt_tokens = 0
        completion_tokens = 0
        for chunk in response:
            if not chunk.candidates:
                continue
            for candidate in chunk.candidates:
                if not candidate.content or not candidate.content.parts:
                    continue

                message = self._parse_parts(candidate.content.parts)
                index += len(candidate.content.parts)

                if not message.content and not message.tool_calls:
                    continue

                if chunk.usage_metadata:
                    prompt_tokens += chunk.usage_metadata.prompt_token_count or 0
                    completion_tokens += (
                        chunk.usage_metadata.candidates_token_count or 0
                    )

                # if the stream is not finished, yield the chunk
                if not candidate.finish_reason:
                    yield LLMResultChunk(
                        model=model,
                        prompt_messages=list(prompt_messages),
                        delta=LLMResultChunkDelta(
                            index=index,
                            message=message,
                        ),
                    )
                # if the stream is finished, yield the chunk and the finish reason
                else:
                    if prompt_tokens == 0 or completion_tokens == 0:
                        prompt_tokens = self.get_num_tokens(
                            model=model,
                            credentials=credentials,
                            prompt_messages=prompt_messages,
                        )
                        completion_tokens = self.get_num_tokens(
                            model=model,
                            credentials=credentials,
                            prompt_messages=[message],
                        )
                    usage = self._calc_response_usage(
                        model=model,
                        credentials=dict(credentials),
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )
                    yield LLMResultChunk(
                        model=model,
                        prompt_messages=list(prompt_messages),
                        delta=LLMResultChunkDelta(
                            index=index,
                            message=message,
                            finish_reason=candidate.finish_reason,
                            usage=usage,
                        ),
                    )

    def _convert_one_message_to_text(self, message: PromptMessage) -> str:
        """
        Convert a single message to a string.

        :param message: PromptMessage to convert.
        :return: String representation of the message.
        """
        human_prompt = "\n\nuser:"
        ai_prompt = "\n\nmodel:"
        content = message.content
        if isinstance(content, list):
            content = "".join(
                (c.data for c in content if c.type != PromptMessageContentType.IMAGE)
            )
        if isinstance(message, UserPromptMessage):
            message_text = f"{human_prompt} {content}"
        elif isinstance(message, AssistantPromptMessage):
            message_text = f"{ai_prompt} {content}"
        elif isinstance(message, SystemPromptMessage | ToolPromptMessage):
            message_text = f"{human_prompt} {content}"
        else:
            raise ValueError(f"Got unknown type {message}")
        return message_text

    def _convert_to_content_dict(self, prompt: PromptMessage) -> types.ContentDict:
        role = "user" if isinstance(prompt, UserPromptMessage) else "model"
        parts = []

        prompt_contents = prompt.content
        if isinstance(prompt_contents, str):
            parts.append(types.Part.from_text(text=prompt_contents))
        elif isinstance(prompt_contents, list):
            for content in prompt_contents:
                if isinstance(content, TextPromptMessageContent):
                    parts.append(types.Part.from_text(text=content.data))
                elif isinstance(
                    content,
                    ImagePromptMessageContent
                    | DocumentPromptMessageContent
                    | VideoPromptMessageContent
                    | AudioPromptMessageContent,
                ):
                    if content.url:
                        parts.append(
                            types.Part.from_uri(
                                file_uri=content.url,
                                mime_type=content.mime_type,
                            )
                        )
                    else:
                        parts.append(
                            types.Part.from_bytes(
                                data=base64.b64decode(content.base64_data),
                                mime_type=content.mime_type,
                            )
                        )
                else:
                    raise InvokeError(f"unknown content type {content.type}")
        else:
            raise InvokeError("prompt content must be a string or a list")

        return types.ContentDict(parts=parts, role=role)

    @property
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        """
        Map model invoke error to unified error
        The key is the ermd = genai.GenerativeModel(model) error type thrown to the caller
        The value is the md = genai.GenerativeModel(model) error type thrown by the model,
        which needs to be converted into a unified error type for the caller.

        :return: Invoke emd = genai.GenerativeModel(model) error mapping
        """
        return {
            InvokeConnectionError: [
                errors.ClientError,
            ],
            InvokeServerUnavailableError: [
                errors.ServerError,
            ],
            InvokeBadRequestError: [
                errors.UnknownFunctionCallArgumentError,
                errors.UnsupportedFunctionError,
                errors.FunctionInvocationError,
            ],
        }

    def _parse_parts(self, parts: Sequence[types.Part], /) -> AssistantPromptMessage:
        contents: list[PromptMessageContent] = []
        function_calls = []
        for part in parts:
            if part.text:
                contents.append(TextPromptMessageContent(data=part.text))
            if part.function_call:
                function_call = part.function_call
                function_call_id = function_call.id
                function_call_name = function_call.name
                function_call_args = function_call.args
                if not isinstance(function_call_id, str):
                    raise InvokeError("function_call_id received is not a string")
                if not isinstance(function_call_name, str):
                    raise InvokeError("function_call_name received is not a string")
                if not isinstance(function_call_args, dict):
                    raise InvokeError("function_call_args received is not a dict")
                function_call = AssistantPromptMessage.ToolCall(
                    id=function_call_id,
                    type="function",
                    function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                        name=function_call_name,
                        arguments=json.dumps(function_call_args),
                    ),
                )
                function_calls.append(function_call)
            if part.inline_data:
                inline_data = part.inline_data
                mime_type = inline_data.mime_type
                data = inline_data.data
                if mime_type is None:
                    raise InvokeError("receive inline_data with no mime_type")
                if data is None:
                    raise InvokeError("receive inline_data with no data")
                if "image" in mime_type:
                    contents.append(
                        ImagePromptMessageContent(
                            format=mime_type.split("/")[-1],
                            base64_data=base64.b64encode(data).decode(),
                            mime_type=mime_type,
                        )
                    )
                else:
                    raise InvokeError(f"unsupported mime_type {mime_type}")

        # FIXME: This is a workaround to fix the typing issue in the dify_plugin
        # https://github.com/langgenius/dify-plugin-sdks/issues/41
        # fixed_contents = [content.model_dump(mode="json") for content in contents]
        message = AssistantPromptMessage(
            content=contents, tool_calls=function_calls  # type: ignore
        )
        return message
