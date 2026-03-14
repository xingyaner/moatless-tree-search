import json
import logging
import os
from enum import Enum
from textwrap import dedent
from typing import Optional, Union, List, Any, Dict

import litellm
import tenacity
from litellm.exceptions import (
    BadRequestError,
    NotFoundError,
    AuthenticationError,
    APIError,
)
from pydantic import BaseModel, Field, model_validator, ValidationError

from moatless.completion.model import Completion, StructuredOutput
from moatless.exceptions import CompletionRejectError, CompletionRuntimeError

# =================================================================
# --- 硬编码 API 配置区 (Hardcoded API Config) ---
# =================================================================
# 已经按照您的指令填入 DeepSeek API 信息
DEFAULT_API_KEY = "sk-"
DEFAULT_BASE_URL = "https://api.deepseek.com"
# =================================================================

logger = logging.getLogger(__name__)


class LLMResponseFormat(str, Enum):
    TOOLS = "tool_call"
    JSON = "json"
    ANTHROPIC_TOOLS = "anthropic_tools"
    REACT = "react"


class CompletionResponse(BaseModel):
    """Container for completion responses that can include multiple structured outputs and text"""

    structured_outputs: List[StructuredOutput] = Field(default_factory=list)
    text_response: Optional[str] = Field(default=None)
    completion: Optional[Completion] = Field(default=None)

    @classmethod
    def create(cls, text: str | None = None, output: List[StructuredOutput] | StructuredOutput | None = None,
               completion: Completion | None = None) -> "CompletionResponse":
        if isinstance(output, StructuredOutput):
            outputs = [output]
        elif isinstance(output, list):
            outputs = output
        else:
            outputs = None

        return cls(text_response=text, structured_outputs=outputs, completion=completion)

    @property
    def structured_output(self) -> Optional[StructuredOutput]:
        """Get the first structured output"""
        if len(self.structured_outputs) > 1:
            ignored_outputs = [
                output.__class__.__name__ for output in self.structured_outputs[1:]
            ]
            logger.warning(
                f"Multiple structured outputs found in completion response, returning {self.structured_outputs[0].__class__.__name__} and ignoring: {ignored_outputs}"
            )
        return self.structured_outputs[0] if self.structured_outputs else None


class CompletionModel(BaseModel):
    model: str = Field(..., description="The model to use for completion")
    temperature: float = Field(0.0, description="The temperature to use for completion")
    max_tokens: int = Field(2000, description="The maximum number of tokens to generate")
    timeout: float = Field(120.0, description="The timeout in seconds for completion requests")
    # 默认值直接指向硬编码的 Base URL
    model_base_url: Optional[str] = Field(default=DEFAULT_BASE_URL, description="The base URL for the model API")
    # 默认值直接指向硬编码的 API Key
    model_api_key: Optional[str] = Field(default=DEFAULT_API_KEY, description="The API key for the model", exclude=True)
    response_format: Optional[LLMResponseFormat] = Field(None, description="The response format expected from the LLM")
    stop_words: Optional[list[str]] = Field(default=None, description="The stop words to use for completion")

    metadata: Optional[dict] = Field(default=None, description="Additional metadata for the completion model")
    enable_thinking: bool = Field(default=False, description="Whether to enable DeepSeek thinking mode")
    thoughts_in_action: bool = Field(default=False,
                                     description="Whether to include thoughts in the action or in the message", )

    def clone(self, **kwargs) -> "CompletionModel":
        """Create a copy of the completion model with optional parameter overrides."""
        model_data = self.model_dump()
        model_data.update(kwargs)
        # 确保 clone 时也保留 API key
        if "model_api_key" not in kwargs:
            model_data["model_api_key"] = self.model_api_key
        return CompletionModel.model_validate(model_data)

    def create_completion(self, messages: List[dict], system_prompt: str,
                          response_model: List[type[StructuredOutput]] | type[StructuredOutput]) -> CompletionResponse:
        if not response_model:
            raise CompletionRuntimeError(f"Response model is required for completion")

        if isinstance(response_model, list) and len(response_model) > 1:
            avalabile_actions = [a for a in response_model if hasattr(a, "name")]
            if not avalabile_actions:
                raise CompletionRuntimeError(f"No actions found in {response_model}")

            class TakeAction(StructuredOutput):
                action: Union[tuple(response_model)] = Field(...)
                action_type: str = Field(..., description="The type of action being taken")

                @model_validator(mode="before")
                def validate_action(cls, data: dict) -> dict:
                    if not isinstance(data, dict): return data
                    action_type = data.get("action_type")
                    if not action_type: return data
                    action_class = next((a for a in avalabile_actions if a.name == action_type), None)
                    if not action_class: raise ValidationError(f"Unknown action {action_type}")
                    data["action"] = action_class.model_validate(data.get("action"))
                    return data

            response_model = TakeAction

        # --- 【关键修正 1】防止 System 消息重复插入 ---
        current_messages = messages.copy()
        if not current_messages or current_messages[0].get("role") != "system":
            full_system = system_prompt + dedent(
                f"""\n# Response format\nYou must respond with only a JSON object matching the schema:\n{json.dumps(response_model.model_json_schema(), indent=2, ensure_ascii=False)}\n""")
            current_messages.insert(0, {"role": "system", "content": full_system})

        retries = tenacity.Retrying(
            retry=tenacity.retry_if_not_exception_type((APIError, BadRequestError, NotFoundError, AuthenticationError)),
            stop=tenacity.stop_after_attempt(3),
        )

        def _do_completion():
            completion_response = None
            try:
                # 开启 DeepSeek 官方 JSON 模式
                completion_response = self._litellm_base_completion(messages=current_messages,
                                                                    response_format={"type": "json_object"})

                if not completion_response or not completion_response.choices:
                    raise CompletionRuntimeError("No completion response or choices returned")

                msg_obj = completion_response.choices[0].message
                content = msg_obj.content if msg_obj.content is not None else ""
                reasoning = getattr(msg_obj, 'reasoning_content', None) or ""

                tool_args = ""
                if hasattr(msg_obj, 'tool_calls') and msg_obj.tool_calls:
                    tool_args = msg_obj.tool_calls[0].function.arguments

                # 【核心拦截点】：打印绝对原始的 LLM 响应，不加任何处理
                print("\n" + "█" * 40 + " [ABSOLUTE RAW LLM START] " + "█" * 40)
                # 使用 repr() 打印，可以看清换行符、不可见字符和特殊标记
                print(f"RAW_CONTENT: {repr(content)}")
                print(f"RAW_REASONING: {repr(reasoning)}")
                print(f"RAW_TOOL_CALLS: {repr(tool_args)}")
                print("█" * 40 + " [ABSOLUTE RAW LLM END] " + "█" * 40 + "\n")


                # --- 合并思维链和正文，确保 JSON 不被遗漏 ---
                combined_input = f"{reasoning}\n{content}\n{tool_args}".strip()
                combined_input = combined_input + "\n[RESPONSE_END]"


                # --- 【核心修复】：拦截纯空格回复 ---
                if not combined_input or combined_input.isspace():
                    logger.warning("DeepSeek returned whitespace. Injecting error to trigger retry.")
                    combined_input = "ERROR: Your response was empty. Please provide a valid JSON action."
                # --- 【新增：证据拦截打印】 ---
                print("\n" + "!"*30 + " [RAW LLM DATA START] " + "!"*30)
                print(f"MODEL: {self.model}")
                print(f"REASONING LENGTH: {len(reasoning)}")
                print(f"CONTENT LENGTH: {len(content)}")
                print(f"RAW COMBINED INPUT:\n{combined_input}")
                print("!"*30 + " [RAW LLM DATA END] " + "!"*30 + "\n")

                assistant_history_entry = {"role": "assistant", "content": content}
                if reasoning:
                    assistant_history_entry["reasoning_content"] = reasoning
                current_messages.append(assistant_history_entry)

                # 将合并后的文本交给增强型解析器，如果为空则传 "{}" 触发解析器内部的重试诱导
                response = response_model.model_validate_json(combined_input if combined_input else "{}")
                completion_obj = Completion.from_llm_completion(input_messages=current_messages,
                                                                completion_response=completion_response,
                                                                model=self.model)

                if hasattr(response, "action"):
                    return CompletionResponse.create(output=response.action, completion=completion_obj)
                return CompletionResponse.create(output=response, completion=completion_obj)

            except (ValidationError, json.JSONDecodeError, ValueError) as e:
                # 这就是闭环的关键：将解析器的报错发回给 LLM
                feedback_msg = f"FORMAT ERROR: {str(e)}"
                logger.warning(f"Validation failed: {feedback_msg}. Retrying with feedback.")

                # 更新消息历史，让模型知道哪里错了
                current_messages.append({"role": "user", "content": feedback_msg})

                # 重新抛出，触发 tenacity 重试
                raise CompletionRejectError(message=str(e), last_completion=completion_response,
                                            messages=current_messages) from e


        try:
            return retries(_do_completion)
        except tenacity.RetryError as e:
            raise e.reraise()

    def _litellm_base_completion(self, messages: list[dict], **kwargs) -> Any:
        # 允许 litellm 自动处理不同模型的参数映射
        litellm.drop_params = True

        @tenacity.retry(
            stop=tenacity.stop_after_attempt(2),
            wait=tenacity.wait_exponential(multiplier=2),
            reraise=True,
            before_sleep=lambda retry_state: logger.warning(
                f"Retrying litellm completion after error: {retry_state.outcome.exception()}"
            ),
        )
        def _do_call():
            # 显式初始化 extra_body 为 None，防止 NameError
            # 如果未来需要再次开启 Reasoner 的思考模式，可以在此进行条件赋值
            extra_body = None

            # 如果模型是 reasoner 或显式启用了思考模式，则配置相应的参数
            if self.enable_thinking or (self.model and "reasoner" in self.model.lower()):
                extra_body = {"thinking": {"type": "enabled"}}

            return litellm.completion(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=messages,
                metadata=self.metadata or {},
                timeout=self.timeout,
                api_base=self.model_base_url,
                api_key=self.model_api_key,
                stop=self.stop_words,
                tools=kwargs.get('tools'),
                tool_choice=kwargs.get('tool_choice'),
                response_format=kwargs.get('response_format'),
                request_timeout=self.timeout,
                extra_body=extra_body  # 此时 extra_body 已定义，不再报错
            )

        try:
            return _do_call()
        except tenacity.RetryError as e:
            last_exception = e.last_attempt.exception()
            if isinstance(last_exception, litellm.APIError):
                logger.error("LiteLLM API Error: %s", last_exception.message)
            raise last_exception

    def model_dump(self, **kwargs):
        dump = super().model_dump(**kwargs)
        if "response_format" in dump and dump["response_format"]:
            dump["response_format"] = dump["response_format"].value
        return dump

    @classmethod
    def model_validate(cls, obj):
        if isinstance(obj, dict) and obj.get("response_format"):
            # DeepSeek Reasoner 应该直接使用 CompletionModel，因为其工具调用不是通过 'tool_call' 格式
            # 而是通过 'content' 中的 JSON + extra_body.thinking
            if "deepseek-reasoner" in obj["model"] and obj["response_format"] == LLMResponseFormat.TOOLS.value:
                # 对于 DeepSeek Reasoner 的 tool_call，我们不使用 ToolCallCompletionModel
                # 而是直接使用 CompletionModel，因为它需要通过 content 返回 JSON
                obj["response_format"] = LLMResponseFormat.JSON.value  # 强制改为 JSON
                return cls(**obj)  # 直接用 CompletionModel 实例化

            response_format = LLMResponseFormat(obj["response_format"])
            obj["response_format"] = response_format

            if response_format == LLMResponseFormat.TOOLS:
                from moatless.completion.tool_call import ToolCallCompletionModel
                return ToolCallCompletionModel(**obj)
            elif response_format == LLMResponseFormat.REACT:
                from moatless.completion.react import ReActCompletionModel
                return ReActCompletionModel(**obj)

        return cls(**obj)

    @model_validator(mode="after")
    def set_api_key(self) -> "CompletionModel":
        if not self.model_api_key or "sk-" not in self.model_api_key: self.model_api_key = DEFAULT_API_KEY
        if not self.model_base_url: self.model_base_url = DEFAULT_BASE_URL
        return self
