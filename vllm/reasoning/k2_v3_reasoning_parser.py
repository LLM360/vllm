# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    DeltaMessage,
    ResponsesRequest,
)
from vllm.reasoning.abs_reasoning_parsers import (
    ReasoningParserStreamingFinalization,
    ReasoningParserStreamingMetadataPartition,
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
        self._streaming_state = "reasoning"
        self._streaming_buffer = ""
        self._streaming_token_buffer: list[int] = []
        self._streaming_metadata_partition: (
            ReasoningParserStreamingMetadataPartition | None
        ) = None

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
        return self._split_model_output(model_output)

    def _split_model_output(self, model_output: str) -> tuple[str, str]:
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
        del previous_text, current_text, previous_token_ids, current_token_ids
        self._streaming_metadata_partition = None

        if self._streaming_state == "content":
            self._streaming_metadata_partition = (
                ReasoningParserStreamingMetadataPartition(
                    content_token_count=len(delta_token_ids)
                )
            )
            return DeltaMessage(content=delta_text)

        self._streaming_buffer += delta_text
        self._streaming_token_buffer.extend(delta_token_ids)

        # An explicit close always wins, including when a plural tool wrapper
        # appeared earlier. Until that close arrives, all output is ambiguous
        # and cannot be streamed without risking an incorrect classification.
        if self.end_token in self._streaming_buffer:
            reasoning, content = self._split_model_output(self._streaming_buffer)
            if self.end_token_id in self._streaming_token_buffer:
                end_index = self._streaming_token_buffer.index(self.end_token_id)
                reasoning_token_count = end_index + 1
                self._streaming_metadata_partition = (
                    ReasoningParserStreamingMetadataPartition(
                        reasoning_token_count=reasoning_token_count,
                        content_token_count=(
                            len(self._streaming_token_buffer) - reasoning_token_count
                        ),
                    )
                )
            self._streaming_buffer = ""
            self._streaming_token_buffer = []
            self._streaming_state = "content"
            return DeltaMessage(
                reasoning=reasoning,
                content=content if content else None,
            )

        return None

    def finalize_reasoning_streaming(
        self,
    ) -> ReasoningParserStreamingFinalization | None:
        if self._streaming_state != "reasoning":
            return None

        reasoning, content = self._split_model_output(self._streaming_buffer)
        start_token_count = int(
            bool(self._streaming_token_buffer)
            and self._streaming_token_buffer[0] == self.start_token_id
        )
        tool_start_token_id = self.vocab.get(self._TOOL_CALLS_START_TOKEN)
        if (
            self._TOOL_CALLS_START_TOKEN in self._streaming_buffer
            and tool_start_token_id in self._streaming_token_buffer
        ):
            content_start = self._streaming_token_buffer.index(tool_start_token_id)
            self._streaming_metadata_partition = (
                ReasoningParserStreamingMetadataPartition(
                    reasoning_token_count=content_start,
                    content_token_count=len(self._streaming_token_buffer)
                    - content_start,
                )
            )
        elif not reasoning:
            self._streaming_metadata_partition = (
                ReasoningParserStreamingMetadataPartition(
                    reasoning_token_count=start_token_count,
                    content_token_count=len(self._streaming_token_buffer)
                    - start_token_count,
                )
            )
        else:
            # If a tokenizer does not expose the plural wrapper as one token,
            # the serving layer cannot split its logprobs without guessing.
            self._streaming_metadata_partition = None
        self._streaming_buffer = ""
        self._streaming_token_buffer = []
        self._streaming_state = "content"
        return ReasoningParserStreamingFinalization(
            delta=DeltaMessage(reasoning=reasoning, content=content),
            reasoning_ended=True,
        )

    def take_reasoning_streaming_metadata_partition(
        self,
    ) -> ReasoningParserStreamingMetadataPartition | None:
        partition = self._streaming_metadata_partition
        self._streaming_metadata_partition = None
        return partition
