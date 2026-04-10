# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import json
from collections.abc import Sequence
from typing import Any

import regex as re

from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    DeltaMessage,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import ToolParser
from vllm.entrypoints.openai.tool_parsers.pythonic_tool_parser import (
    _handle_single_tool,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike

logger = init_logger(__name__)


class MultiFormatToolParser(ToolParser):
    """Tool parser that dispatches on ``chat_template_kwargs['tool_format']``."""

    _MINIMAX_START_TOKEN = "<tool_calls>"
    _MINIMAX_BLOCK_REGEX = re.compile(
        r"<tool_calls>(.*?)</tool_calls>",
        re.DOTALL,
    )
    _MINIMAX_INVOKE_REGEX = re.compile(
        r'<invoke\s+name="([^"]+)"\s*>(.*?)</invoke>',
        re.DOTALL,
    )
    _MINIMAX_PARAMETER_REGEX = re.compile(
        r'<parameter\s+name="([^"]+)"'
        r'(?:\s+string="(true|false)")?\s*>(.*?)</parameter>',
        re.DOTALL,
    )

    _GPTOSS_BLOCK_REGEX = re.compile(
        r"<tool_call>\s*(?:assistant\s+)?to=functions\.(\S+?)"
        r"(?:\s+json)?\s*\n(.*?)\n?\s*</tool_call>",
        re.DOTALL,
    )

    _PYTHON_BLOCK_REGEX = re.compile(
        r"<tool_call>(.*?)</tool_call>",
        re.DOTALL,
    )

    def __init__(
        self,
        tokenizer: TokenizerLike,
        chat_template_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(tokenizer)

        self.tool_format = str(
            (chat_template_kwargs or {}).get("tool_format") or "default"
        )
        self._delegate: ToolParser | None = None

        if self.tool_format == "default":
            from vllm.entrypoints.openai.tool_parsers.hermes_tool_parser import (
                Hermes2ProToolParser,
            )

            self._delegate = Hermes2ProToolParser(tokenizer)
        elif self.tool_format == "qwen3":
            from vllm.entrypoints.openai.tool_parsers.qwen3xml_tool_parser import (
                Qwen3XMLToolParser,
            )

            self._delegate = Qwen3XMLToolParser(tokenizer)
        elif self.tool_format == "glm":
            from vllm.entrypoints.openai.tool_parsers.glm4_moe_tool_parser import (
                Glm4MoeModelToolParser,
            )

            self._delegate = Glm4MoeModelToolParser(tokenizer)

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        if self._delegate is not None:
            return self._delegate.adjust_request(request)
        return super().adjust_request(request)

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        if self._delegate is not None:
            return self._delegate.extract_tool_calls(model_output, request)

        try:
            if self.tool_format == "minimax":
                return self._extract_minimax_tool_calls(model_output)
            if self.tool_format == "dsv32":
                return self._extract_dsv32_tool_calls(model_output)
            if self.tool_format == "gptoss":
                return self._extract_gptoss_tool_calls(model_output)
            if self.tool_format == "python":
                return self._extract_python_tool_calls(model_output)
        except Exception:
            logger.exception(
                "Error extracting tool calls for tool_format=%s.",
                self.tool_format,
            )

        return ExtractedToolCallInformation(
            tools_called=False,
            tool_calls=[],
            content=model_output,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        if self._delegate is not None:
            return self._delegate.extract_tool_calls_streaming(
                previous_text,
                current_text,
                delta_text,
                previous_token_ids,
                current_token_ids,
                delta_token_ids,
                request,
            )

        return None

    @staticmethod
    def _json_or_string(value: str) -> Any:
        value = value.strip()
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    @staticmethod
    def _prefix_content(model_output: str, first_tool_index: int | None) -> str | None:
        if first_tool_index is None or first_tool_index <= 0:
            return None
        content = model_output[:first_tool_index]
        return content if content.strip() else None

    @staticmethod
    def _tool_call(function_name: str, arguments: dict[str, Any]) -> ToolCall:
        return ToolCall(
            type="function",
            function=FunctionCall(
                name=function_name,
                arguments=json.dumps(arguments, ensure_ascii=False),
            ),
        )

    def _extract_minimax_tool_calls(
        self,
        model_output: str,
    ) -> ExtractedToolCallInformation:
        if self._MINIMAX_START_TOKEN not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        tool_calls: list[ToolCall] = []
        for block in self._MINIMAX_BLOCK_REGEX.findall(model_output):
            for function_name, invoke_body in self._MINIMAX_INVOKE_REGEX.findall(block):
                arguments = {
                    param_name: self._json_or_string(param_value)
                    for param_name, _, param_value in (
                        self._MINIMAX_PARAMETER_REGEX.findall(invoke_body)
                    )
                }
                tool_calls.append(self._tool_call(function_name, arguments))

        if not tool_calls:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=tool_calls,
            content=self._prefix_content(
                model_output,
                model_output.find(self._MINIMAX_START_TOKEN),
            ),
        )

    def _extract_dsv32_tool_calls(
        self,
        model_output: str,
    ) -> ExtractedToolCallInformation:
        if self._MINIMAX_START_TOKEN not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        tool_calls: list[ToolCall] = []
        for block in self._MINIMAX_BLOCK_REGEX.findall(model_output):
            for function_name, invoke_body in self._MINIMAX_INVOKE_REGEX.findall(block):
                arguments: dict[str, Any] = {}
                for (
                    param_name,
                    string_flag,
                    param_value,
                ) in self._MINIMAX_PARAMETER_REGEX.findall(invoke_body):
                    arguments[param_name] = (
                        param_value
                        if string_flag == "true"
                        else self._json_or_string(param_value)
                    )
                tool_calls.append(self._tool_call(function_name, arguments))

        if not tool_calls:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=tool_calls,
            content=self._prefix_content(
                model_output,
                model_output.find(self._MINIMAX_START_TOKEN),
            ),
        )

    def _extract_gptoss_tool_calls(
        self,
        model_output: str,
    ) -> ExtractedToolCallInformation:
        matches = list(self._GPTOSS_BLOCK_REGEX.finditer(model_output))

        if not matches:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        tool_calls: list[ToolCall] = []
        for match in matches:
            function_name = match.group(1)
            arguments = json.loads(match.group(2).strip())
            tool_calls.append(self._tool_call(function_name, arguments))

        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=tool_calls,
            content=self._prefix_content(model_output, matches[0].start()),
        )

    def _extract_python_tool_calls(
        self,
        model_output: str,
    ) -> ExtractedToolCallInformation:
        matches = self._PYTHON_BLOCK_REGEX.findall(model_output)
        if not matches:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        tool_calls: list[ToolCall] = []
        for block in matches:
            module = ast.parse(block.strip())
            for statement in module.body:
                if not isinstance(statement, ast.Expr) or not isinstance(
                    statement.value,
                    ast.Call,
                ):
                    raise ValueError(
                        "Expected Python function call(s) inside <tool_call> tags."
                    )
                tool_calls.append(_handle_single_tool(statement.value))

        if not tool_calls:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=tool_calls,
            content=self._prefix_content(
                model_output,
                model_output.find("<tool_call>"),
            ),
        )
