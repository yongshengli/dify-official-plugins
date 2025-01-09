import json
import time
from collections.abc import Generator
from copy import deepcopy
from typing import Any, Optional, cast

from pydantic import BaseModel, Field

from dify_plugin.entities.agent import AgentInvokeMessage
from dify_plugin.entities.model.llm import LLMModelConfig, LLMResult, LLMResultChunk, LLMUsage
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    PromptMessage,
    PromptMessageContentType,
    SystemPromptMessage,
    ToolPromptMessage,
    UserPromptMessage,
)
from dify_plugin.entities.tool import ToolInvokeMessage, ToolProviderType
from dify_plugin.interfaces.agent import AgentModelConfig, AgentStrategy, ToolEntity


class FunctionCallingParams(BaseModel):
    query: str
    instruction: str | None
    model: AgentModelConfig
    tools: list[ToolEntity] | None
    maximum_iterations: int = 3


class ToolInvokeMeta(BaseModel):
    """
    Tool invoke meta
    """

    time_cost: float = Field(..., description="The time cost of the tool invoke")
    error: Optional[str] = None
    tool_config: Optional[dict] = None

    @classmethod
    def empty(cls) -> "ToolInvokeMeta":
        """
        Get an empty instance of ToolInvokeMeta
        """
        return cls(time_cost=0.0, error=None, tool_config={})

    @classmethod
    def error_instance(cls, error: str) -> "ToolInvokeMeta":
        """
        Get an instance of ToolInvokeMeta with error
        """
        return cls(time_cost=0.0, error=error, tool_config={})

    def to_dict(self) -> dict:
        return {
            "time_cost": self.time_cost,
            "error": self.error,
            "tool_config": self.tool_config,
        }


class FunctionCallingAgentStrategy(AgentStrategy):
    def __init__(self, session):
        super().__init__(session)
        self.query = ""

    def _invoke(self, parameters: dict[str, Any]) -> Generator[AgentInvokeMessage]:
        """
        Run FunctionCall agent application
        """
        fc_params = FunctionCallingParams(**parameters)
        query = fc_params.query
        self.query = query
        instruction = fc_params.instruction
        init_prompt_messages = [
            SystemPromptMessage(content=instruction),
            UserPromptMessage(content=query),
        ]
        tools = fc_params.tools
        tool_instances = {tool.identity.name: tool for tool in tools} if tools else {}
        model = fc_params.model
        stop = fc_params.model.completion_params.get("stop", []) if fc_params.model.completion_params else []
        # convert tools into ModelRuntime Tool format
        prompt_messages_tools = self._init_prompt_tools(tools)

        iteration_step = 1
        max_iteration_steps = fc_params.maximum_iterations
        current_thoughts: list[PromptMessage] = []
        # continue to run until there is not any tool call
        function_call_state = True
        llm_usage: dict[str, Optional[LLMUsage]] = {"usage": None}
        final_answer = ""

        while function_call_state and iteration_step <= max_iteration_steps:
            function_call_state = False
            round_started_at = time.perf_counter()
            round_log = self.create_log_message(
                label=f"ROUND {iteration_step}",
                data={},
                metadata={
                    "started_at": round_started_at,
                },
                status=ToolInvokeMessage.LogMessage.LogStatus.START,
            )
            yield round_log
            if iteration_step == max_iteration_steps:
                # the last iteration, remove all tools
                prompt_messages_tools = []

            # recalc llm max tokens
            prompt_messages = self._organize_prompt_messages(
                history_prompt_messages=init_prompt_messages, current_thoughts=current_thoughts
            )
            if model.completion_params:
                self.recalc_llm_max_tokens(model.entity, prompt_messages, model.completion_params)
            # invoke model
            model_started_at = time.perf_counter()
            model_log = self.create_log_message(
                label=f"{model.model} Thought",
                data={},
                metadata={"start_at": model_started_at, "provider": model.provider},
                parent=round_log,
                status=ToolInvokeMessage.LogMessage.LogStatus.START,
            )
            yield model_log
            chunks: Generator[LLMResultChunk, None, None] | LLMResult = self.session.model.llm.invoke(
                model_config=LLMModelConfig(**model.model_dump(mode="json")),
                prompt_messages=prompt_messages,
                stream=True,
                stop=stop,
                tools=prompt_messages_tools,
            )

            tool_calls: list[tuple[str, str, dict[str, Any]]] = []

            # save full response
            response = ""

            # save tool call names and inputs
            tool_call_names = ""
            tool_call_inputs = ""

            current_llm_usage = None

            if isinstance(chunks, Generator):
                is_first_chunk = True
                for chunk in chunks:
                    if is_first_chunk:
                        is_first_chunk = False
                    # check if there is any tool call
                    if self.check_tool_calls(chunk):
                        function_call_state = True
                        tool_calls.extend(self.extract_tool_calls(chunk) or [])
                        tool_call_names = ";".join([tool_call[1] for tool_call in tool_calls])
                        try:
                            tool_call_inputs = json.dumps(
                                {tool_call[1]: tool_call[2] for tool_call in tool_calls}, ensure_ascii=False
                            )
                        except json.JSONDecodeError:
                            # ensure ascii to avoid encoding error
                            tool_call_inputs = json.dumps({tool_call[1]: tool_call[2] for tool_call in tool_calls})

                    if chunk.delta.message and chunk.delta.message.content:
                        if isinstance(chunk.delta.message.content, list):
                            for content in chunk.delta.message.content:
                                response += content.data
                        else:
                            response += str(chunk.delta.message.content)

                    if chunk.delta.usage:
                        self.increase_usage(llm_usage, chunk.delta.usage)
                        current_llm_usage = chunk.delta.usage

            else:
                result = chunks
                # check if there is any tool call
                if self.check_blocking_tool_calls(result):
                    function_call_state = True
                    tool_calls.extend(self.extract_blocking_tool_calls(result) or [])
                    tool_call_names = ";".join([tool_call[1] for tool_call in tool_calls])
                    try:
                        tool_call_inputs = json.dumps(
                            {tool_call[1]: tool_call[2] for tool_call in tool_calls}, ensure_ascii=False
                        )
                    except json.JSONDecodeError:
                        # ensure ascii to avoid encoding error
                        tool_call_inputs = json.dumps({tool_call[1]: tool_call[2] for tool_call in tool_calls})

                if result.usage:
                    self.increase_usage(llm_usage, result.usage)
                    current_llm_usage = result.usage

                if result.message and result.message.content:
                    if isinstance(result.message.content, list):
                        for content in result.message.content:
                            response += content.data
                    else:
                        response += str(result.message.content)

                if not result.message.content:
                    result.message.content = ""
            yield self.finish_log_message(
                log=model_log,
                data={
                    "output": response,
                    "tool_name": tool_call_names,
                    "tool_input": tool_call_inputs,
                },
                metadata={
                    "started_at": model_started_at,
                    "finished_at": time.perf_counter(),
                    "elapsed_time": time.perf_counter() - model_started_at,
                    "provider": model.provider,
                    "total_price": current_llm_usage.total_price if current_llm_usage else 0,
                    "currency": current_llm_usage.currency if current_llm_usage else "",
                    "total_tokens": current_llm_usage.total_tokens if current_llm_usage else 0,
                },
            )
            assistant_message = AssistantPromptMessage(content="", tool_calls=[])
            if tool_calls:
                assistant_message.tool_calls = [
                    AssistantPromptMessage.ToolCall(
                        id=tool_call[0],
                        type="function",
                        function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                            name=tool_call[1], arguments=json.dumps(tool_call[2], ensure_ascii=False)
                        ),
                    )
                    for tool_call in tool_calls
                ]
            else:
                assistant_message.content = response

            current_thoughts.append(assistant_message)

            final_answer += response + "\n"

            # call tools
            tool_responses = []
            for tool_call_id, tool_call_name, tool_call_args in tool_calls:
                tool_instance = tool_instances[tool_call_name]
                tool_call_started_at = time.perf_counter()
                tool_call_log = self.create_log_message(
                    label=f"CALL {tool_call_name}",
                    data={},
                    metadata={
                        "started_at": time.perf_counter(),
                        "provider": tool_instance.identity.provider,
                    },
                    parent=round_log,
                    status=ToolInvokeMessage.LogMessage.LogStatus.START,
                )
                yield tool_call_log
                if not tool_instance:
                    tool_response = {
                        "tool_call_id": tool_call_id,
                        "tool_call_name": tool_call_name,
                        "tool_response": f"there is not a tool named {tool_call_name}",
                        "meta": ToolInvokeMeta.error_instance(f"there is not a tool named {tool_call_name}").to_dict(),
                    }
                else:
                    # invoke tool
                    tool_invoke_responses = self.session.tool.invoke(
                        provider_type=ToolProviderType.BUILT_IN,
                        provider=tool_instance.identity.provider,
                        tool_name=tool_instance.identity.name,
                        parameters={**tool_instance.runtime_parameters, **tool_call_args},
                    )
                    result = ""
                    for response in tool_invoke_responses:
                        if response.type == ToolInvokeMessage.MessageType.TEXT:
                            result += cast(ToolInvokeMessage.TextMessage, response.message).text
                        elif response.type == ToolInvokeMessage.MessageType.LINK:
                            result += (
                                f"result link: {cast(ToolInvokeMessage.TextMessage, response.message).text}."
                                + " please tell user to check it."
                            )
                        elif response.type in {
                            ToolInvokeMessage.MessageType.IMAGE_LINK,
                            ToolInvokeMessage.MessageType.IMAGE,
                        }:
                            result += (
                                "image has been created and sent to user already, "
                                + "you do not need to create it, just tell the user to check it now."
                            )
                        elif response.type == ToolInvokeMessage.MessageType.JSON:
                            text = json.dumps(
                                cast(ToolInvokeMessage.JsonMessage, response.message).json_object, ensure_ascii=False
                            )
                            result += f"tool response: {text}."
                        else:
                            result += f"tool response: {response.message!r}."
                    tool_invoke_meta = ToolInvokeMeta.error_instance("")
                    tool_response = {
                        "tool_call_id": tool_call_id,
                        "tool_call_name": tool_call_name,
                        "tool_response": result,
                        "meta": tool_invoke_meta.to_dict(),
                    }

                yield self.finish_log_message(
                    log=tool_call_log,
                    data={
                        "output": tool_response,
                    },
                    metadata={
                        "started_at": tool_call_started_at,
                        "provider": tool_instance.identity.provider,
                        "finished_at": time.perf_counter(),
                        "elapsed_time": time.perf_counter() - tool_call_started_at,
                    },
                )
                tool_responses.append(tool_response)
                if tool_response["tool_response"] is not None:
                    current_thoughts.append(
                        ToolPromptMessage(
                            content=str(tool_response["tool_response"]),
                            tool_call_id=tool_call_id,
                            name=tool_call_name,
                        )
                    )

            # update prompt tool
            for prompt_tool in prompt_messages_tools:
                self.update_prompt_message_tool(tool_instances[prompt_tool.name], prompt_tool)
            yield self.finish_log_message(
                log=round_log,
                data={
                    "output": {
                        "llm_response": response,
                        "tool_responses": tool_responses,
                    },
                },
                metadata={
                    "started_at": round_started_at,
                    "finished_at": time.perf_counter(),
                    "elapsed_time": time.perf_counter() - round_started_at,
                    "total_price": current_llm_usage.total_price if current_llm_usage else 0,
                    "currency": current_llm_usage.currency if current_llm_usage else "",
                    "total_tokens": current_llm_usage.total_tokens if current_llm_usage else 0,
                },
            )
            iteration_step += 1

        yield self.create_text_message(final_answer)
        yield self.create_json_message(
            {
                "execution_metadata": {
                    "total_price": llm_usage["usage"].total_price if llm_usage["usage"] is not None else 0,
                    "currency": llm_usage["usage"].currency if llm_usage["usage"] is not None else "",
                    "total_tokens": llm_usage["usage"].total_tokens if llm_usage["usage"] is not None else 0,
                }
            }
        )

    def check_tool_calls(self, llm_result_chunk: LLMResultChunk) -> bool:
        """
        Check if there is any tool call in llm result chunk
        """
        return bool(llm_result_chunk.delta.message.tool_calls)

    def check_blocking_tool_calls(self, llm_result: LLMResult) -> bool:
        """
        Check if there is any blocking tool call in llm result
        """
        return bool(llm_result.message.tool_calls)

    def extract_tool_calls(self, llm_result_chunk: LLMResultChunk) -> list[tuple[str, str, dict[str, Any]]]:
        """
        Extract tool calls from llm result chunk

        Returns:
            List[Tuple[str, str, Dict[str, Any]]]: [(tool_call_id, tool_call_name, tool_call_args)]
        """
        tool_calls = []
        for prompt_message in llm_result_chunk.delta.message.tool_calls:
            args = {}
            if prompt_message.function.arguments != "":
                args = json.loads(prompt_message.function.arguments)

            tool_calls.append(
                (
                    prompt_message.id,
                    prompt_message.function.name,
                    args,
                )
            )

        return tool_calls

    def extract_blocking_tool_calls(self, llm_result: LLMResult) -> list[tuple[str, str, dict[str, Any]]]:
        """
        Extract blocking tool calls from llm result

        Returns:
            List[Tuple[str, str, Dict[str, Any]]]: [(tool_call_id, tool_call_name, tool_call_args)]
        """
        tool_calls = []
        for prompt_message in llm_result.message.tool_calls:
            args = {}
            if prompt_message.function.arguments != "":
                args = json.loads(prompt_message.function.arguments)

            tool_calls.append(
                (
                    prompt_message.id,
                    prompt_message.function.name,
                    args,
                )
            )

        return tool_calls

    def _init_system_message(self, prompt_template: str, prompt_messages: list[PromptMessage]) -> list[PromptMessage]:
        """
        Initialize system message
        """
        if not prompt_messages and prompt_template:
            return [
                SystemPromptMessage(content=prompt_template),
            ]

        if prompt_messages and not isinstance(prompt_messages[0], SystemPromptMessage) and prompt_template:
            prompt_messages.insert(0, SystemPromptMessage(content=prompt_template))

        return prompt_messages or []

    def _organize_user_query(self, query: str, prompt_messages: list[PromptMessage]) -> list[PromptMessage]:
        """
        Organize user query
        """

        prompt_messages.append(UserPromptMessage(content=query))

        return prompt_messages

    def _clear_user_prompt_image_messages(self, prompt_messages: list[PromptMessage]) -> list[PromptMessage]:
        """
        As for now, gpt supports both fc and vision at the first iteration.
        We need to remove the image messages from the prompt messages at the first iteration.
        """
        prompt_messages = deepcopy(prompt_messages)

        for prompt_message in prompt_messages:
            if isinstance(prompt_message, UserPromptMessage) and isinstance(prompt_message.content, list):
                prompt_message.content = "\n".join(
                    [
                        content.data
                        if content.type == PromptMessageContentType.TEXT
                        else "[image]"
                        if content.type == PromptMessageContentType.IMAGE
                        else "[file]"
                        for content in prompt_message.content
                    ]
                )

        return prompt_messages

    def _organize_prompt_messages(
        self, current_thoughts: list[PromptMessage], history_prompt_messages: list[PromptMessage]
    ) -> list[PromptMessage]:
        prompt_template = ""
        history_prompt_messages = self._init_system_message(prompt_template, history_prompt_messages)
        query_prompt_messages = self._organize_user_query(self.query or "", [])

        prompt_messages = [*history_prompt_messages, *query_prompt_messages, *current_thoughts]
        if len(current_thoughts) != 0:
            # clear messages after the first iteration
            prompt_messages = self._clear_user_prompt_image_messages(prompt_messages)
        return prompt_messages
