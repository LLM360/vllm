# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import json
from collections.abc import Sequence
from typing import Any

import regex as re

from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    DeltaMessage,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import ToolParser
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike

logger = init_logger(__name__)


class MultiFormatToolParser(ToolParser):
    """Tool parser that dispatches on chat template tool-call format kwargs."""

    _SUPPORTED_TOOL_FORMATS = frozenset(
        {
            "qwen3",
            "minimax",
            "dsv32",
            "glm",
            "gptoss",
            "python",
            "json",
            "xml",
            "xml_typed",
        }
    )

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

    _IFM_TOOL_CALLS_START_TOKEN = "<ifm|tool_calls>"
    _IFM_TOOL_CALL_START_TOKEN = "<ifm|tool_call>"
    _IFM_BLOCK_REGEX = re.compile(
        r"<ifm\|tool_call>(.*?)</ifm\|tool_call>",
        re.DOTALL,
    )
    _IFM_ARG_REGEX = re.compile(
        r"<ifm\|arg_key>(.*?)</ifm\|arg_key>\s*"
        r"(?:<ifm\|arg_type>(.*?)</ifm\|arg_type>\s*)?"
        r"<ifm\|arg_value>(.*?)</ifm\|arg_value>",
        re.DOTALL,
    )
    _IFM_REASONING_PREFIX_REGEX = re.compile(
        r"\A\s*(?:"
        r"<ifm\|think>.*?</ifm\|think>|"
        r"<ifm\|think_fast>.*?</ifm\|think_fast>|"
        r"<ifm\|think_faster>.*?</ifm\|think_faster>"
        r")\s*",
        re.DOTALL,
    )

    _GLM_BLOCK_REGEX = re.compile(
        r"<tool_call>(.*?)</tool_call>",
        re.DOTALL,
    )
    _GLM_ARG_REGEX = re.compile(
        r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
        re.DOTALL,
    )

    def __init__(
        self,
        tokenizer: TokenizerLike,
        chat_template_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(tokenizer)

        chat_template_kwargs = chat_template_kwargs or {}
        if "tool_calling_format" in chat_template_kwargs:
            raise ValueError(
                "Unsupported argument: tool_calling_format. "
                "Use tool_call_format with one of: json, xml, xml_typed."
            )
        if "tool_format" in chat_template_kwargs:
            raise ValueError(
                "Unsupported argument: tool_format. "
                "Use tool_call_format with one of: json, xml, xml_typed."
            )
        raw_tool_format = chat_template_kwargs.get("tool_call_format", "xml")
        self.tool_format = self._validate_tool_format(raw_tool_format)
        self._delegate: ToolParser | None = None

        if self.tool_format == "qwen3":
            from vllm.entrypoints.openai.tool_parsers.qwen3xml_tool_parser import (
                Qwen3XMLToolParser,
            )

            self._delegate = Qwen3XMLToolParser(tokenizer)

    @classmethod
    def _validate_tool_format(cls, tool_format: Any) -> str:
        if not isinstance(tool_format, str):
            raise ValueError(
                "tool_call_format must be a string. "
                f"Got {type(tool_format).__name__}."
            )
        if tool_format not in cls._SUPPORTED_TOOL_FORMATS:
            supported_formats = ", ".join(sorted(cls._SUPPORTED_TOOL_FORMATS))
            raise ValueError(
                f"Unsupported tool_call_format '{tool_format}'. "
                "Use one of these exact values: "
                f"{supported_formats}."
            )
        return tool_format

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
            if self.tool_format == "json":
                return self._extract_ifm_json_tool_calls(model_output, request)
            if self.tool_format in {"xml", "xml_typed"}:
                return self._extract_ifm_xml_tool_calls(model_output, request)
            if self.tool_format == "minimax":
                return self._extract_minimax_tool_calls(model_output)
            if self.tool_format == "dsv32":
                return self._extract_dsv32_tool_calls(model_output)
            if self.tool_format == "glm":
                if self._IFM_TOOL_CALL_START_TOKEN in model_output:
                    return self._extract_ifm_xml_tool_calls(model_output, request)
                return self._extract_glm_tool_calls(model_output, request)
            if self.tool_format == "gptoss":
                return self._extract_gptoss_tool_calls(model_output)
            if self.tool_format == "python":
                return self._extract_python_tool_calls(model_output)
        except Exception:
            logger.exception(
                "Error extracting tool calls for tool_call_format=%s.",
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

    @classmethod
    def _strip_ifm_reasoning_prefix(cls, content: str) -> str:
        while match := cls._IFM_REASONING_PREFIX_REGEX.match(content):
            content = content[match.end() :]
        return content

    @classmethod
    def _prefix_content(
        cls,
        model_output: str,
        first_tool_index: int | None,
    ) -> str | None:
        if first_tool_index is None or first_tool_index <= 0:
            return None
        content = cls._strip_ifm_reasoning_prefix(model_output[:first_tool_index])
        return content if content else None

    @staticmethod
    def _tool_call(function_name: str, arguments: dict[str, Any]) -> ToolCall:
        return ToolCall(
            type="function",
            function=FunctionCall(
                name=function_name,
                arguments=json.dumps(arguments, ensure_ascii=False),
            ),
        )

    @staticmethod
    def _schema_arg_type(
        tool_name: str,
        arg_name: str,
        tools: list[ChatCompletionToolsParam] | None,
    ) -> Any | None:
        if tools is None:
            return None
        for tool in tools:
            if tool.function.name != tool_name or tool.function.parameters is None:
                continue
            properties = tool.function.parameters.get("properties", {})
            arg_spec = properties.get(arg_name, {})
            if not isinstance(arg_spec, dict):
                return None
            return arg_spec.get("type")
        return None

    @staticmethod
    def _arg_type_is_string(arg_type: Any | None) -> bool:
        if isinstance(arg_type, str):
            return arg_type == "string"
        if isinstance(arg_type, list):
            return "string" in arg_type
        return False

    @staticmethod
    def _json_stringify(value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)

    @classmethod
    def _coerce_argument_value(
        cls,
        value: Any,
        tool_name: str,
        arg_name: str,
        tools: list[ChatCompletionToolsParam] | None,
        *,
        arg_type: str | None = None,
        from_text: bool = False,
    ) -> Any:
        target_type = cls._schema_arg_type(tool_name, arg_name, tools) or arg_type
        if cls._arg_type_is_string(target_type):
            return cls._json_stringify(value)
        if target_type == "any" and from_text:
            return value

        if isinstance(value, str) and (from_text or target_type is not None):
            return cls._deserialize_glm_value(value)
        return value

    @classmethod
    def _coerce_arguments(
        cls,
        tool_name: str,
        arguments: dict[str, Any],
        tools: list[ChatCompletionToolsParam] | None,
    ) -> dict[str, Any]:
        return {
            arg_name: cls._coerce_argument_value(
                arg_value,
                tool_name,
                arg_name,
                tools,
            )
            for arg_name, arg_value in arguments.items()
        }

    @staticmethod
    def _json_arguments_to_dict(arguments: Any) -> dict[str, Any]:
        if arguments is None:
            return {}
        if isinstance(arguments, str):
            arguments = json.loads(arguments) if arguments.strip() else {}
        if not isinstance(arguments, dict):
            raise ValueError("Tool call arguments must be a JSON object.")
        return arguments

    @classmethod
    def _ifm_prefix_index(cls, model_output: str, first_match_index: int) -> int:
        group_index = model_output.find(cls._IFM_TOOL_CALLS_START_TOKEN)
        if group_index != -1:
            return group_index
        return first_match_index

    def _extract_ifm_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        matches = list(self._IFM_BLOCK_REGEX.finditer(model_output))
        if not matches:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        first_block = matches[0].group(1).strip()
        if first_block.startswith(("{", "[")):
            return self._extract_ifm_json_tool_calls(model_output, request)
        return self._extract_ifm_xml_tool_calls(model_output, request)

    def _extract_ifm_json_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        matches = list(self._IFM_BLOCK_REGEX.finditer(model_output))
        if not matches:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        tool_calls: list[ToolCall] = []
        for match in matches:
            raw_tool_call = json.loads(match.group(1).strip())
            raw_tool_calls = (
                raw_tool_call if isinstance(raw_tool_call, list) else [raw_tool_call]
            )
            for tool_call in raw_tool_calls:
                function = tool_call.get("function", tool_call)
                function_name = function.get("name")
                if not function_name:
                    raise ValueError("Tool call JSON is missing a function name.")
                arguments = self._json_arguments_to_dict(
                    function.get("arguments", {})
                )
                arguments = self._coerce_arguments(
                    function_name,
                    arguments,
                    request.tools,
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
                self._ifm_prefix_index(model_output, matches[0].start()),
            ),
        )

    def _extract_ifm_xml_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        matches = list(self._IFM_BLOCK_REGEX.finditer(model_output))
        if not matches:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        tool_calls: list[ToolCall] = []
        for match in matches:
            block = match.group(1)
            first_arg_idx = block.find("<ifm|arg_key>")
            if first_arg_idx == -1:
                function_name = block.strip()
                arguments: dict[str, Any] = {}
            else:
                function_name = block[:first_arg_idx].strip()
                arg_block = block[first_arg_idx:]
                arguments = {}
                for key, arg_type, value in self._IFM_ARG_REGEX.findall(arg_block):
                    arg_key = key.strip()
                    explicit_arg_type = arg_type.strip() or None
                    schema_arg_type = self._schema_arg_type(
                        function_name,
                        arg_key,
                        request.tools,
                    )
                    target_type = schema_arg_type or explicit_arg_type
                    value_for_coercion = (
                        value
                        if self._arg_type_is_string(target_type)
                        or target_type == "any"
                        else value.strip()
                    )
                    arg_value = self._coerce_argument_value(
                        value_for_coercion,
                        function_name,
                        arg_key,
                        request.tools,
                        arg_type=explicit_arg_type,
                        from_text=True,
                    )
                    arguments[arg_key] = arg_value

            if function_name:
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
                self._ifm_prefix_index(model_output, matches[0].start()),
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

    @staticmethod
    def _deserialize_glm_value(value: str) -> Any:
        value = value.strip()
        try:
            return json.loads(value)
        except Exception:
            pass

        try:
            return ast.literal_eval(value)
        except Exception:
            pass

        return value

    def _extract_glm_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        matches = list(self._GLM_BLOCK_REGEX.finditer(model_output))
        if not matches:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        tool_calls: list[ToolCall] = []
        for match in matches:
            block = match.group(1)
            first_arg_idx = block.find("<arg_key>")
            if first_arg_idx == -1:
                function_name = block.strip()
                arguments: dict[str, Any] = {}
            else:
                function_name = block[:first_arg_idx].strip()
                arg_block = block[first_arg_idx:]
                arguments = {}
                for key, value in self._GLM_ARG_REGEX.findall(arg_block):
                    arg_key = key.strip()
                    arg_value = self._coerce_argument_value(
                        value.strip(),
                        function_name,
                        arg_key,
                        request.tools,
                        from_text=True,
                    )
                    arguments[arg_key] = arg_value

            if function_name:
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
            content=self._prefix_content(model_output, matches[0].start()),
        )

    @staticmethod
    def _get_python_value(val: ast.expr) -> Any:
        if isinstance(val, ast.Constant):
            return val.value
        if isinstance(val, ast.Name):
            if val.id in {"true", "True"}:
                return True
            if val.id in {"false", "False"}:
                return False
            if val.id in {"null", "None"}:
                return None
        if isinstance(val, ast.Dict):
            if not all(isinstance(k, ast.Constant) for k in val.keys):
                raise ValueError("Dict tool call arguments must have literal keys")
            return {
                k.value: MultiFormatToolParser._get_python_value(v)  # type: ignore
                for k, v in zip(val.keys, val.values)
            }
        if isinstance(val, ast.List):
            return [MultiFormatToolParser._get_python_value(v) for v in val.elts]
        if isinstance(val, ast.Tuple):
            return [MultiFormatToolParser._get_python_value(v) for v in val.elts]
        if (
            isinstance(val, ast.UnaryOp)
            and isinstance(val.op, (ast.USub, ast.UAdd))
            and isinstance(val.operand, ast.Constant)
            and isinstance(val.operand.value, (int, float))
        ):
            operand = val.operand.value
            return -operand if isinstance(val.op, ast.USub) else operand
        raise ValueError("Tool call arguments must be literals")

    @staticmethod
    def _handle_python_tool(call: ast.Call) -> ToolCall:
        if not isinstance(call.func, ast.Name):
            raise ValueError("Invalid tool call name")
        function_name = call.func.id
        arguments = {}
        for keyword in call.keywords:
            arguments[keyword.arg] = MultiFormatToolParser._get_python_value(
                keyword.value
            )
        return ToolCall(
            type="function",
            function=FunctionCall(
                name=function_name,
                arguments=json.dumps(arguments, ensure_ascii=False),
            ),
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
                tool_calls.append(self._handle_python_tool(statement.value))

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


class K2V3ToolParser(MultiFormatToolParser):
    """K2-V3 parser for BBQ 0518 IFM tool-call and reasoning tokens."""
