# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
from vllm.tokenizers import TokenizerLike


class K2V3ReasoningParser(DeepSeekR1ReasoningParser):
    """
    Reasoning parser for the K2-v3 model family.

    K2-v3 supports three reasoning effort levels, each using different
    think tokens:
      - high (default): <think> / </think>
      - medium:         <think_fast> / </think_fast>
      - low:            <think_faster> / </think_faster>

    The effort level is selected via the ``reasoning_effort`` parameter
    in ``chat_template_kwargs``.  The chat template inserts the start
    token into the prompt, so the generated output typically only
    contains the end token.
    """

    _EFFORT_TOKENS: dict[str, tuple[str, str]] = {
        "high": ("<think>", "</think>"),
        "medium": ("<think_fast>", "</think_fast>"),
        "low": ("<think_faster>", "</think_faster>"),
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
