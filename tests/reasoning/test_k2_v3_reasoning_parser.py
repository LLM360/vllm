# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.reasoning.utils import run_reasoning_extraction
from vllm.reasoning import ReasoningParser, ReasoningParserManager

PARSER_NAME = "k2_v3"

EFFORT_TOKENS = {
    "high": ("<ifm|think>", "</ifm|think>"),
    "medium": ("<ifm|think_fast>", "</ifm|think_fast>"),
    "low": ("<ifm|think_faster>", "</ifm|think_faster>"),
}

TOOL_CALL = (
    "<ifm|tool_call>get_weather"
    "<ifm|arg_key>city</ifm|arg_key>"
    "<ifm|arg_value>Tokyo</ifm|arg_value>"
    "</ifm|tool_call>"
)


class FakeTokenizer:
    _SPECIAL_TOKENS = sorted(
        {
            token
            for token_pair in EFFORT_TOKENS.values()
            for token in token_pair
        },
        key=len,
        reverse=True,
    )

    def __init__(self):
        self._vocab: dict[str, int] = {
            token: token_id
            for token_id, token in enumerate(self._SPECIAL_TOKENS, start=1)
        }
        self._next_token_id = len(self._vocab) + 1

    def get_vocab(self):
        return self._vocab

    def tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        i = 0
        while i < len(text):
            for special_token in self._SPECIAL_TOKENS:
                if text.startswith(special_token, i):
                    tokens.append(special_token)
                    i += len(special_token)
                    break
            else:
                tokens.append(text[i])
                i += 1
        return tokens

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens)

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self._token_to_id(tokens)
        return [self._token_to_id(token) for token in tokens]

    def _token_to_id(self, token: str) -> int:
        if token not in self._vocab:
            self._vocab[token] = self._next_token_id
            self._next_token_id += 1
        return self._vocab[token]


@pytest.fixture(scope="module")
def k2_v3_tokenizer():
    return FakeTokenizer()


def _make_parser(tokenizer, effort="high") -> ReasoningParser:
    return ReasoningParserManager.get_reasoning_parser(PARSER_NAME)(
        tokenizer, chat_template_kwargs={"reasoning_effort": effort}
    )


# ---------------------------------------------------------------------------
# Test cases parameterised by effort level
# ---------------------------------------------------------------------------


def _cases_for_effort(start: str, end: str):
    """Build test-case dicts for a given start/end token pair."""
    return {
        "simple_reasoning": {
            "output": f"This is reasoning{end}This is content",
            "reasoning": "This is reasoning",
            "content": "This is content",
            "is_reasoning_end": True,
        },
        "complete_reasoning": {
            "output": f"This is reasoning{end}",
            "reasoning": "This is reasoning",
            "content": None,
            "is_reasoning_end": True,
        },
        "no_end_token": {
            "output": "This is reasoning only",
            "reasoning": "This is reasoning only",
            "content": None,
            "is_reasoning_end": False,
        },
        "with_start_token": {
            "output": f"{start}This is reasoning{end}This is content",
            "reasoning": "This is reasoning",
            "content": "This is content",
            "is_reasoning_end": True,
        },
        "with_start_no_end": {
            "output": f"{start}Still thinking",
            "reasoning": "Still thinking",
            "content": None,
            "is_reasoning_end": False,
        },
        "multiple_lines": {
            "output": f"Line1\nLine2{end}Content1\nContent2",
            "reasoning": "Line1\nLine2",
            "content": "Content1\nContent2",
            "is_reasoning_end": True,
        },
    }


_EFFORTS = ["high", "medium", "low"]
_CASE_NAMES = [
    "simple_reasoning",
    "complete_reasoning",
    "no_end_token",
    "with_start_token",
    "with_start_no_end",
    "multiple_lines",
]


def _build_params():
    params = []
    for effort in _EFFORTS:
        start, end = EFFORT_TOKENS[effort]
        cases = _cases_for_effort(start, end)
        for case_name in _CASE_NAMES:
            for streaming in [False, True]:
                mode = "streaming" if streaming else "nonstreaming"
                test_id = f"{effort}_{case_name}_{mode}"
                params.append(
                    pytest.param(effort, streaming, cases[case_name], id=test_id)
                )
    return params


@pytest.mark.parametrize("effort, streaming, param_dict", _build_params())
def test_reasoning(
    effort: str,
    streaming: bool,
    param_dict: dict,
    k2_v3_tokenizer,
):
    output = k2_v3_tokenizer.tokenize(param_dict["output"])
    output_tokens: list[str] = [
        k2_v3_tokenizer.convert_tokens_to_string([token]) for token in output
    ]
    parser = _make_parser(k2_v3_tokenizer, effort)

    reasoning, content = run_reasoning_extraction(
        parser, output_tokens, streaming=streaming
    )

    assert reasoning == param_dict["reasoning"]
    assert content == param_dict["content"]

    # Test is_reasoning_end
    output_ids = k2_v3_tokenizer.convert_tokens_to_ids(output)
    assert parser.is_reasoning_end(output_ids) == param_dict["is_reasoning_end"]

    # Test extract_content_ids
    if param_dict["content"] is not None:
        content_ids = parser.extract_content_ids(output_ids)
        expected_ids = k2_v3_tokenizer.convert_tokens_to_ids(
            k2_v3_tokenizer.tokenize(param_dict["content"])
        )
        assert content_ids == expected_ids
    else:
        assert parser.extract_content_ids(output_ids) == []


# ---------------------------------------------------------------------------
# Default effort / edge cases
# ---------------------------------------------------------------------------


def test_default_effort_is_high(k2_v3_tokenizer):
    """Parser with no reasoning_effort should use <ifm|think>/</ifm|think>."""
    parser = ReasoningParserManager.get_reasoning_parser(PARSER_NAME)(k2_v3_tokenizer)
    assert parser.start_token == "<ifm|think>"
    assert parser.end_token == "</ifm|think>"


def test_none_effort_falls_back_to_high(k2_v3_tokenizer):
    """reasoning_effort='none' should fall back to high tokens."""
    parser = _make_parser(k2_v3_tokenizer, "none")
    assert parser.start_token == "<ifm|think>"
    assert parser.end_token == "</ifm|think>"


def test_unknown_effort_falls_back_to_high(k2_v3_tokenizer):
    """Unknown effort value should fall back to high tokens."""
    parser = _make_parser(k2_v3_tokenizer, "ultra")
    assert parser.start_token == "<ifm|think>"
    assert parser.end_token == "</ifm|think>"


@pytest.mark.parametrize("effort", _EFFORTS)
def test_streaming_standalone_end_token_emits_empty_reasoning(
    effort: str, k2_v3_tokenizer
):
    parser = _make_parser(k2_v3_tokenizer, effort)
    end_token = EFFORT_TOKENS[effort][1]
    end_token_id = k2_v3_tokenizer.convert_tokens_to_ids(end_token)

    delta = parser.extract_reasoning_streaming(
        previous_text="",
        current_text=end_token,
        delta_text=end_token,
        previous_token_ids=[],
        current_token_ids=[end_token_id],
        delta_token_ids=[end_token_id],
    )

    assert delta is not None
    assert delta.reasoning == ""
    assert delta.content is None


@pytest.mark.parametrize("effort", _EFFORTS)
@pytest.mark.parametrize(
    "reasoning",
    [
        pytest.param("", id="without_reasoning"),
        pytest.param("Need lookup", id="with_reasoning"),
    ],
)
def test_streaming_end_token_routes_following_tool_call_to_content(
    effort: str, reasoning: str, k2_v3_tokenizer
):
    parser = _make_parser(k2_v3_tokenizer, effort)
    end_token = EFFORT_TOKENS[effort][1]
    model_deltas = [f"{reasoning}{end_token}", TOOL_CALL]
    emitted_deltas: list[tuple[str | None, str | None]] = []
    previous_text = ""
    previous_token_ids: list[int] = []

    for delta_text in model_deltas:
        delta_tokens = k2_v3_tokenizer.tokenize(delta_text)
        delta_token_ids = k2_v3_tokenizer.convert_tokens_to_ids(delta_tokens)
        current_text = previous_text + delta_text
        current_token_ids = previous_token_ids + delta_token_ids

        delta = parser.extract_reasoning_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=previous_token_ids,
            current_token_ids=current_token_ids,
            delta_token_ids=delta_token_ids,
        )
        if delta is not None:
            emitted_deltas.append((delta.reasoning, delta.content))

        previous_text = current_text
        previous_token_ids = current_token_ids

    expected_deltas = [(reasoning, None), (None, TOOL_CALL)]
    assert emitted_deltas == expected_deltas
