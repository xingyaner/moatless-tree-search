import logging
from typing import List

import tenacity
from litellm.exceptions import (
    BadRequestError,
    NotFoundError,
    AuthenticationError,
    APIError,
)
from pydantic import BaseModel, ValidationError

from moatless.completion.completion import CompletionModel, CompletionResponse
from moatless.completion.model import Completion, StructuredOutput, Usage
from moatless.exceptions import CompletionRejectError, CompletionRuntimeError

logger = logging.getLogger(__name__)


class ToolCallCompletionModel(CompletionModel):
    def create_completion(
            self,
            messages: List[dict],
            system_prompt: str,
            response_model: List[type[StructuredOutput]] | type[StructuredOutput],
    ) -> CompletionResponse:
        tools = []

        if isinstance(response_model, list):
            tools.extend(
                [
                    r.openai_schema(thoughts_in_action=self.thoughts_in_action)
                    for r in response_model
                ]
            )
        elif response_model:
            tools.append(response_model.openai_schema())
        else:
            tools = None

        total_usage = Usage()
        retry_count = 0

        messages.insert(0, {"role": "system", "content": system_prompt})

        retries = tenacity.Retrying(
            retry=tenacity.retry_if_not_exception_type(
                (APIError, BadRequestError, NotFoundError, AuthenticationError)
            ),
            stop=tenacity.stop_after_attempt(3),
        )

        def _do_completion():
            nonlocal total_usage, retry_count
            llm_completion_response = None
            try:
                if self.thoughts_in_action:
                    tool_choice = "required"
                else:
                    tool_choice = "auto"

                llm_completion_response = self._litellm_base_completion(
                    messages=messages, tools=tools, tool_choice=tool_choice
                )

                if not llm_completion_response or not llm_completion_response.choices:
                    raise CompletionRuntimeError("No choices returned")

                # 获取原始消息对象
                raw_msg = llm_completion_response.choices[0].message

                # --- 【DeepSeek-Chat 协议鲁棒性处理】 ---
                # 1. 强制空值占位
                content = raw_msg.content if raw_msg.content is not None else ""

                # 2. 构造符合协议的消息节点
                assistant_entry = {
                    "role": "assistant",
                    "content": content
                }

                # 保留思维链（如果是 Chat 模型返回了 CoT）
                reasoning = getattr(raw_msg, 'reasoning_content', None)
                if reasoning:
                    assistant_entry["reasoning_content"] = reasoning

                # 必须将 tool_calls 写回消息历史，否则后续 role: tool 消息会因找不到来源而报错
                if raw_msg.tool_calls:
                    assistant_entry["tool_calls"] = [
                        tc.model_dump() if hasattr(tc, 'model_dump') else tc
                        for tc in raw_msg.tool_calls
                    ]

                messages.append(assistant_entry)

                # 后续解析逻辑保持原样
                total_usage += Usage.from_completion_response(llm_completion_response, self.model)

                content = llm_completion_response.choices[0].message.content

                def get_response_model(tool_name: str):
                    if isinstance(response_model, list):
                        for r in response_model:
                            if r.name == tool_name:
                                return r
                    else:
                        return response_model

                response_objects = []
                invalid_function_names = []
                seen_arguments = set()
                flags = set()

                if llm_completion_response.choices[0].message.tool_calls:
                    for tool_call in llm_completion_response.choices[
                        0
                    ].message.tool_calls:
                        action = get_response_model(tool_call.function.name)

                        if not action:
                            logger.warning(
                                f"Invalid action name: {tool_call.function.name}"
                            )
                            invalid_function_names.append(tool_call.function.name)
                            continue

                        # Check for duplicate arguments
                        if tool_call.function.arguments in seen_arguments:
                            logger.warning(
                                f"Duplicate tool call arguments found for {tool_call.function.name}"
                            )
                            flags.add("duplicate_tool_call")
                            continue

                        seen_arguments.add(tool_call.function.arguments)
                        response_object = action.model_validate_json(
                            tool_call.function.arguments
                        )
                        response_objects.append(response_object)

                    if invalid_function_names:
                        available_actions = [r.name for r in response_model]
                        raise ValueError(
                            f"Unknown functions {invalid_function_names}. Available functions: {available_actions}"
                        )

                if not content and not response_objects:
                    raise ValueError("No tool call or content in message.")

                completion = Completion.from_llm_completion(
                    input_messages=messages,
                    completion_response=llm_completion_response,
                    model=self.model,
                    retries=retry_count,
                    usage=total_usage,
                    flags=list(flags),
                )

                return CompletionResponse.create(text=content, output=response_objects, completion=completion)

            except (ValidationError, ValueError) as e:
                logger.warning(
                    f"Completion attempt failed with error: {e}. Will retry."
                )
                messages.append(
                    {
                        "role": "user",
                        "content": f"The response was invalid. Fix the errors, exceptions found\n{e}",
                    }
                )
                raise CompletionRejectError(
                    message=str(e),
                    last_completion=llm_completion_response,
                    messages=messages,
                ) from e

        try:
            return retries(_do_completion)
        except tenacity.RetryError as e:
            raise e.reraise()
