# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import AutoTokenizer

from tests.reasoning.utils import run_reasoning_extraction
from vllm.reasoning import ReasoningParser, ReasoningParserManager

PARSER_NAME = "k2_v2"
REASONING_MODEL_NAME = "LLM360/K2-V2-Instruct"

EFFORT_TOKENS = {
    "high": ("<think>", "</think>"),
    "medium": ("<think_fast>", "</think_fast>"),
    "low": ("<think_faster>", "</think_faster>"),
}


@pytest.fixture(scope="module")
def k2_v2_tokenizer():
    return AutoTokenizer.from_pretrained(REASONING_MODEL_NAME, trust_remote_code=True)


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
    k2_v2_tokenizer,
):
    output = k2_v2_tokenizer.tokenize(param_dict["output"])
    output_tokens: list[str] = [
        k2_v2_tokenizer.convert_tokens_to_string([token]) for token in output
    ]
    parser = _make_parser(k2_v2_tokenizer, effort)

    reasoning, content = run_reasoning_extraction(
        parser, output_tokens, streaming=streaming
    )

    assert reasoning == param_dict["reasoning"]
    assert content == param_dict["content"]

    # Test is_reasoning_end
    output_ids = k2_v2_tokenizer.convert_tokens_to_ids(output)
    assert parser.is_reasoning_end(output_ids) == param_dict["is_reasoning_end"]

    # Test extract_content_ids
    if param_dict["content"] is not None:
        content_ids = parser.extract_content_ids(output_ids)
        expected_ids = k2_v2_tokenizer.convert_tokens_to_ids(
            k2_v2_tokenizer.tokenize(param_dict["content"])
        )
        assert content_ids == expected_ids
    else:
        assert parser.extract_content_ids(output_ids) == []


# ---------------------------------------------------------------------------
# Default effort / edge cases
# ---------------------------------------------------------------------------


def test_default_effort_is_high(k2_v2_tokenizer):
    """Parser with no reasoning_effort should use <think>/<\/think>."""
    parser = ReasoningParserManager.get_reasoning_parser(PARSER_NAME)(k2_v2_tokenizer)
    assert parser.start_token == "<think>"
    assert parser.end_token == "</think>"


def test_none_effort_falls_back_to_high(k2_v2_tokenizer):
    """reasoning_effort='none' should fall back to high tokens."""
    parser = _make_parser(k2_v2_tokenizer, "none")
    assert parser.start_token == "<think>"
    assert parser.end_token == "</think>"


def test_unknown_effort_falls_back_to_high(k2_v2_tokenizer):
    """Unknown effort value should fall back to high tokens."""
    parser = _make_parser(k2_v2_tokenizer, "ultra")
    assert parser.start_token == "<think>"
    assert parser.end_token == "</think>"
