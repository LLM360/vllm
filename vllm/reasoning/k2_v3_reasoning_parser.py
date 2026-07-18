# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    DeltaMessage,
    ResponsesRequest,
)
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
from vllm.tokenizers import TokenizerLike


class K2V3ReasoningParser(DeepSeekR1ReasoningParser):
    """
    Reasoning parser for the K2-v3 model family.

    K2-v3 supports three reasoning effort levels, each using different
    IFM think tokens:
      - high (default): <ifm|think> / </ifm|think>
      - medium:         <ifm|think_fast> / </ifm|think_fast>
      - low:            <ifm|think_faster> / </ifm|think_faster>

    The effort level is selected via the ``reasoning_effort`` parameter
    in ``chat_template_kwargs``.  The chat template inserts the start
    token into the prompt, so the generated output typically only
    contains the end token.
    """

    _EFFORT_TOKENS: dict[str, tuple[str, str]] = {
        "high": ("<ifm|think>", "</ifm|think>"),
        "medium": ("<ifm|think_fast>", "</ifm|think_fast>"),
        "low": ("<ifm|think_faster>", "</ifm|think_faster>"),
    }
    _TOOL_CALLS_START_TOKEN = "<ifm|tool_calls>"

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        effort = chat_kwargs.get("reasoning_effort") or "high"
        if effort == "none":
            effort = "high"
        self._start_token, self._end_token = self._EFFORT_TOKENS.get(
            effort, self._EFFORT_TOKENS["high"]
        )
        super().__init__(tokenizer, *args, **kwargs)

    @property
    def start_token(self) -> str:
        return self._start_token

    @property
    def end_token(self) -> str:
        return self._end_token

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str, str]:
        # The generated start token is optional because the chat template
        # normally inserts it into the prompt.
        _, start_token, output = model_output.partition(self.start_token)
        if not start_token:
            output = model_output

        # An explicit reasoning boundary always takes precedence. Tool-like
        # markup before it is reasoning and must never be executed.
        if self.end_token in output:
            reasoning, _, content = output.partition(self.end_token)
            return reasoning, content

        # If the model omitted the reasoning boundary but started a tool-call
        # section, treat <ifm|tool_calls> as an implicit boundary so the
        # existing tool parser receives the tool section.
        tool_calls_start_index = output.find(self._TOOL_CALLS_START_TOKEN)
        if tool_calls_start_index != -1:
            return (
                output[:tool_calls_start_index],
                output[tool_calls_start_index:],
            )

        # Without either boundary, prefer a visible answer over an empty
        # response caused by classifying the entire output as reasoning.
        return "", output

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        if len(delta_token_ids) == 1 and delta_token_ids[0] == self.end_token_id:
            return DeltaMessage(reasoning="")

        return super().extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )
