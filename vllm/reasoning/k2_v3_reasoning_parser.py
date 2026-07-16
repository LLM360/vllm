# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

from vllm.entrypoints.openai.protocol import DeltaMessage
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
