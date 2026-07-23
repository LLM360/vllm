# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.reasoning.utils import run_reasoning_extraction
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatMessage,
    ToolCall,
)
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.entrypoints.openai.tool_parsers import ToolParserManager
from vllm.reasoning import ReasoningParser, ReasoningParserManager

PARSER_NAME = "k2_v3"

EFFORT_TOKENS = {
    "high": ("<ifm|think>", "</ifm|think>"),
    "medium": ("<ifm|think_fast>", "</ifm|think_fast>"),
    "low": ("<ifm|think_faster>", "</ifm|think_faster>"),
}


def _tool_call(name: str) -> str:
    return (
        f"<ifm|tool_call>{name}"
        "<ifm|arg_key>city</ifm|arg_key>"
        "<ifm|arg_value>Tokyo</ifm|arg_value>"
        "</ifm|tool_call>"
    )


TOOL_CALL = _tool_call("get_weather")
GROUPED_TOOL_CALL = f"<ifm|tool_calls>{TOOL_CALL}</ifm|tool_calls>"
SECOND_TOOL_CALL = _tool_call("get_time")
GROUPED_TOOL_CALLS = f"<ifm|tool_calls>{TOOL_CALL}{SECOND_TOOL_CALL}</ifm|tool_calls>"


def _make_nonstreaming_message(
    model_output: str,
    tokenizer,
    effort: str = "high",
) -> ChatMessage:
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        tool_choice="auto",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": name,
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
            for name in ("get_weather", "get_time")
        ],
    )
    reasoning_parser = _make_parser(tokenizer, effort)
    reasoning, content = reasoning_parser.extract_reasoning(model_output, request)
    function_calls, parsed_content = OpenAIServing._parse_tool_calls_from_content(
        request=request,
        tokenizer=tokenizer,
        content=content,
        enable_auto_tools=True,
        tool_parser_cls=ToolParserManager.get_tool_parser(PARSER_NAME),
        chat_template_kwargs={"tool_call_format": "xml"},
    )
    tool_calls = [ToolCall(function=call) for call in function_calls or []]

    return ChatMessage(
        role="assistant",
        reasoning=reasoning,
        content=parsed_content,
        tool_calls=tool_calls,
    )


class FakeTokenizer:
    _SPECIAL_TOKENS = sorted(
        {token for token_pair in EFFORT_TOKENS.values() for token in token_pair}
        | {"<ifm|tool_calls>"},
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


def _run_streaming_deltas(parser, tokenizer, deltas: list[str]):
    emitted = []
    previous_text = ""
    previous_token_ids: list[int] = []
    for delta_text in deltas:
        delta_tokens = tokenizer.tokenize(delta_text)
        delta_token_ids = tokenizer.convert_tokens_to_ids(delta_tokens)
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
            emitted.append(delta)
        previous_text = current_text
        previous_token_ids = current_token_ids
    return emitted


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
            "nonstreaming_content": "",
            "is_reasoning_end": True,
        },
        "no_end_token": {
            "output": "This is reasoning only",
            "reasoning": "This is reasoning only",
            "content": None,
            "nonstreaming_reasoning": "",
            "nonstreaming_content": "This is reasoning only",
            "streaming_reasoning": "",
            "streaming_content": "This is reasoning only",
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
            "nonstreaming_reasoning": "",
            "nonstreaming_content": "Still thinking",
            "streaming_reasoning": "",
            "streaming_content": "Still thinking",
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

    if streaming:
        emitted = _run_streaming_deltas(parser, k2_v3_tokenizer, output_tokens)
        finalization = parser.finalize_reasoning_streaming()
        if finalization is not None and finalization.delta is not None:
            emitted.append(finalization.delta)
        reasoning_parts = [
            delta.reasoning for delta in emitted if delta.reasoning is not None
        ]
        content_parts = [
            delta.content for delta in emitted if delta.content is not None
        ]
        reasoning = "".join(reasoning_parts) if reasoning_parts else None
        content = "".join(content_parts) if content_parts else None
    else:
        reasoning, content = run_reasoning_extraction(
            parser, output_tokens, streaming=False
        )

    expected_reasoning = param_dict.get(
        "nonstreaming_reasoning" if not streaming else "streaming_reasoning",
        param_dict["reasoning"],
    )
    expected_content = param_dict.get(
        "nonstreaming_content" if not streaming else "streaming_content",
        param_dict["content"],
    )
    assert reasoning == expected_reasoning
    assert content == expected_content

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
@pytest.mark.parametrize(
    "output",
    [
        pytest.param("", id="empty_output"),
        pytest.param("The answer is 42.", id="content"),
    ],
)
def test_nonstreaming_without_boundary_returns_content(
    effort: str,
    output: str,
    k2_v3_tokenizer,
):
    message = _make_nonstreaming_message(
        output,
        k2_v3_tokenizer,
        effort,
    )

    assert message.reasoning == ""
    assert message.reasoning_content == ""
    assert isinstance(message.reasoning, str)
    assert isinstance(message.reasoning_content, str)
    assert message.content == output
    assert message.tool_calls == []


@pytest.mark.parametrize("effort", _EFFORTS)
@pytest.mark.parametrize(
    "tool_section, expected_tool_names",
    [
        pytest.param(GROUPED_TOOL_CALL, ["get_weather"], id="grouped"),
        pytest.param(
            GROUPED_TOOL_CALLS,
            ["get_weather", "get_time"],
            id="multiple_grouped",
        ),
    ],
)
@pytest.mark.parametrize(
    "reasoning",
    [
        pytest.param("", id="empty_reasoning"),
        pytest.param("Need lookup. ", id="with_reasoning"),
    ],
)
def test_nonstreaming_tool_calls_wrapper_implicitly_ends_unclosed_reasoning(
    effort: str,
    tool_section: str,
    expected_tool_names: list[str],
    reasoning: str,
    k2_v3_tokenizer,
):
    message = _make_nonstreaming_message(
        f"{reasoning}{tool_section}",
        k2_v3_tokenizer,
        effort,
    )

    assert message.reasoning == reasoning
    assert message.reasoning_content == reasoning
    assert isinstance(message.reasoning, str)
    assert isinstance(message.reasoning_content, str)
    assert message.content == ""
    assert [tool.function.name for tool in message.tool_calls] == expected_tool_names


@pytest.mark.parametrize("effort", _EFFORTS)
def test_nonstreaming_generated_start_with_unclosed_tool_call(
    effort: str, k2_v3_tokenizer
):
    start_token = EFFORT_TOKENS[effort][0]
    message = _make_nonstreaming_message(
        f"{start_token}Need lookup. {GROUPED_TOOL_CALL}",
        k2_v3_tokenizer,
        effort,
    )

    assert message.reasoning == "Need lookup. "
    assert message.reasoning_content == "Need lookup. "
    assert message.content == ""
    assert [tool.function.name for tool in message.tool_calls] == ["get_weather"]


@pytest.mark.parametrize("effort", _EFFORTS)
@pytest.mark.parametrize(
    "incomplete_tool",
    [
        pytest.param("<ifm|tool_calls>", id="wrapper_only"),
        pytest.param(
            "<ifm|tool_calls><ifm|tool_call>get_weather",
            id="incomplete_inner_call",
        ),
        pytest.param(
            f"<ifm|tool_calls>{TOOL_CALL}",
            id="complete_inner_missing_wrapper_close",
        ),
    ],
)
def test_nonstreaming_incomplete_tool_calls_wrapper_is_implicit_boundary(
    effort: str,
    incomplete_tool: str,
    k2_v3_tokenizer,
):
    output = f"Need lookup. {incomplete_tool}"
    message = _make_nonstreaming_message(output, k2_v3_tokenizer, effort)

    assert message.reasoning == "Need lookup. "
    assert message.reasoning_content == "Need lookup. "
    assert message.content == incomplete_tool
    assert message.tool_calls == []


@pytest.mark.parametrize("effort", _EFFORTS)
def test_nonstreaming_singular_tool_tag_is_not_an_implicit_boundary(
    effort: str,
    k2_v3_tokenizer,
):
    output = f"Need lookup. {TOOL_CALL}"
    request = ChatCompletionRequest(model="test-model", messages=[])
    parser = _make_parser(k2_v3_tokenizer, effort)

    reasoning, content = parser.extract_reasoning(output, request)

    assert reasoning == ""
    assert content == output


@pytest.mark.parametrize("effort", _EFFORTS)
def test_nonstreaming_singular_tool_tag_remains_content_in_serving(
    effort: str,
    k2_v3_tokenizer,
):
    output = f"Need lookup. {TOOL_CALL}"

    message = _make_nonstreaming_message(output, k2_v3_tokenizer, effort)

    assert message.reasoning == ""
    assert message.reasoning_content == ""
    assert message.content == output
    assert message.tool_calls == []


@pytest.mark.parametrize("effort", _EFFORTS)
@pytest.mark.parametrize(
    "tool_markup",
    [
        pytest.param(TOOL_CALL, id="singular"),
        pytest.param(
            f"<ifm|tool_calls>{TOOL_CALL}",
            id="incomplete_plural_wrapper",
        ),
    ],
)
def test_nonstreaming_explicit_close_requires_complete_plural_tool_wrapper(
    effort: str,
    tool_markup: str,
    k2_v3_tokenizer,
):
    end_token = EFFORT_TOKENS[effort][1]

    message = _make_nonstreaming_message(
        f"Need lookup.{end_token}{tool_markup}",
        k2_v3_tokenizer,
        effort,
    )

    assert message.reasoning == "Need lookup."
    assert message.reasoning_content == "Need lookup."
    assert message.content == tool_markup
    assert message.tool_calls == []


@pytest.mark.parametrize("effort", _EFFORTS)
@pytest.mark.parametrize(
    "reasoning, tail, expected_content, expected_tool_names",
    [
        pytest.param("", "", "", [], id="close_only"),
        pytest.param("Need lookup.", "", "", [], id="reasoning_only"),
        pytest.param(
            "", "The answer is 42.", "The answer is 42.", [], id="content_only"
        ),
        pytest.param(
            "Need lookup.",
            GROUPED_TOOL_CALL,
            "",
            ["get_weather"],
            id="single_tool_only",
        ),
        pytest.param(
            "Need two lookups.",
            GROUPED_TOOL_CALLS,
            "",
            ["get_weather", "get_time"],
            id="multiple_tools_only",
        ),
        pytest.param(
            "Need lookup.",
            f"Calling the tool.\n{GROUPED_TOOL_CALL}",
            "Calling the tool.\n",
            ["get_weather"],
            id="content_and_tool",
        ),
    ],
)
def test_nonstreaming_explicit_close_response_matrix(
    effort: str,
    reasoning: str,
    tail: str,
    expected_content: str,
    expected_tool_names: list[str],
    k2_v3_tokenizer,
):
    end_token = EFFORT_TOKENS[effort][1]
    message = _make_nonstreaming_message(
        f"{reasoning}{end_token}{tail}",
        k2_v3_tokenizer,
        effort,
    )

    assert message.reasoning == reasoning
    assert message.reasoning_content == reasoning
    assert isinstance(message.reasoning, str)
    assert isinstance(message.reasoning_content, str)
    assert message.content == expected_content
    assert [tool.function.name for tool in message.tool_calls] == expected_tool_names


@pytest.mark.parametrize("effort", _EFFORTS)
def test_nonstreaming_explicit_close_takes_precedence_over_tool_marker(
    effort: str, k2_v3_tokenizer
):
    end_token = EFFORT_TOKENS[effort][1]
    reasoning_tool = _tool_call("consider_weather")
    output_tool = GROUPED_TOOL_CALL
    expected_reasoning = f"Maybe call this tool: {reasoning_tool}"

    message = _make_nonstreaming_message(
        f"{expected_reasoning}{end_token}{output_tool}",
        k2_v3_tokenizer,
        effort,
    )

    assert message.reasoning == expected_reasoning
    assert message.reasoning_content == expected_reasoning
    assert message.content == ""
    assert [tool.function.name for tool in message.tool_calls] == ["get_weather"]


@pytest.mark.parametrize("effort", _EFFORTS)
@pytest.mark.parametrize(
    "tail, expected_content, expected_tool_names",
    [
        pytest.param("The answer is 42.", "The answer is 42.", [], id="content"),
        pytest.param(GROUPED_TOOL_CALL, "", ["get_weather"], id="tool"),
    ],
)
def test_nonstreaming_multiple_close_tokens_preserve_extra_close_as_content(
    effort: str,
    tail: str,
    expected_content: str,
    expected_tool_names: list[str],
    k2_v3_tokenizer,
):
    end_token = EFFORT_TOKENS[effort][1]
    message = _make_nonstreaming_message(
        f"Need lookup.{end_token}{end_token}{tail}",
        k2_v3_tokenizer,
        effort,
    )

    assert message.reasoning == "Need lookup."
    assert message.reasoning_content == "Need lookup."
    assert message.content == f"{end_token}{expected_content}"
    assert [tool.function.name for tool in message.tool_calls] == expected_tool_names


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


@pytest.mark.parametrize("effort", _EFFORTS)
@pytest.mark.parametrize(
    "tool_section",
    [
        pytest.param(GROUPED_TOOL_CALL, id="single"),
        pytest.param(GROUPED_TOOL_CALLS, id="multiple"),
    ],
)
def test_streaming_missing_close_quarantines_plural_wrapper_until_finalization(
    effort: str, tool_section: str, k2_v3_tokenizer
):
    parser = _make_parser(k2_v3_tokenizer, effort)
    marker = "<ifm|tool_calls>"
    emitted = _run_streaming_deltas(
        parser,
        k2_v3_tokenizer,
        ["Need lookup. ", marker[:8], marker[8:] + tool_section[len(marker) :]],
    )

    assert emitted == []
    finalization = parser.finalize_reasoning_streaming()
    assert finalization is not None
    assert finalization.reasoning_ended
    assert finalization.delta is not None
    assert finalization.delta.reasoning == "Need lookup. "
    assert finalization.delta.content == tool_section


@pytest.mark.parametrize("effort", _EFFORTS)
def test_streaming_failed_plural_marker_candidate_is_released_as_reasoning(
    effort: str, k2_v3_tokenizer
):
    parser = _make_parser(k2_v3_tokenizer, effort)
    emitted = _run_streaming_deltas(
        parser,
        k2_v3_tokenizer,
        ["Need ", "<ifm|tool_call", ">not grouped"],
    )

    assert emitted == []
    finalization = parser.finalize_reasoning_streaming()
    assert finalization is not None and finalization.reasoning_ended
    assert finalization.delta is not None
    assert finalization.delta.reasoning == ""
    assert finalization.delta.content == "Need <ifm|tool_call>not grouped"


@pytest.mark.parametrize("effort", _EFFORTS)
def test_streaming_explicit_close_releases_quarantined_wrapper_as_reasoning(
    effort: str, k2_v3_tokenizer
):
    parser = _make_parser(k2_v3_tokenizer, effort)
    end_token = EFFORT_TOKENS[effort][1]
    emitted = _run_streaming_deltas(
        parser,
        k2_v3_tokenizer,
        ["Maybe ", GROUPED_TOOL_CALL, f" reconsider{end_token}Answer"],
    )

    assert [(delta.reasoning, delta.content) for delta in emitted] == [
        (f"Maybe {GROUPED_TOOL_CALL} reconsider", "Answer"),
    ]
    assert parser.finalize_reasoning_streaming() is None


@pytest.mark.parametrize("effort", _EFFORTS)
def test_streaming_only_post_close_plural_wrapper_becomes_content(
    effort: str, k2_v3_tokenizer
):
    parser = _make_parser(k2_v3_tokenizer, effort)
    end_token = EFFORT_TOKENS[effort][1]
    emitted = _run_streaming_deltas(
        parser,
        k2_v3_tokenizer,
        [GROUPED_TOOL_CALL, end_token, GROUPED_TOOL_CALLS],
    )

    assert [(delta.reasoning, delta.content) for delta in emitted] == [
        (GROUPED_TOOL_CALL, None),
        (None, GROUPED_TOOL_CALLS),
    ]


@pytest.mark.parametrize("effort", _EFFORTS)
@pytest.mark.parametrize(
    "incomplete_tool",
    [
        pytest.param("<ifm|tool_calls>", id="wrapper"),
        pytest.param("<ifm|tool_calls><ifm|tool_call>get_weather", id="inner_call"),
    ],
)
def test_streaming_incomplete_plural_wrapper_is_released_at_finalization(
    effort: str, incomplete_tool: str, k2_v3_tokenizer
):
    parser = _make_parser(k2_v3_tokenizer, effort)
    emitted = _run_streaming_deltas(
        parser, k2_v3_tokenizer, [f"Need lookup. {incomplete_tool}"]
    )

    assert emitted == []
    finalization = parser.finalize_reasoning_streaming()
    assert finalization is not None and finalization.reasoning_ended
    assert finalization.delta is not None
    assert finalization.delta.reasoning == "Need lookup. "
    assert finalization.delta.content == incomplete_tool


@pytest.mark.parametrize("effort", _EFFORTS)
def test_streaming_without_boundary_finalizes_as_content(effort: str, k2_v3_tokenizer):
    parser = _make_parser(k2_v3_tokenizer, effort)
    emitted = _run_streaming_deltas(
        parser,
        k2_v3_tokenizer,
        ["Answer ", "without a close token."],
    )

    assert emitted == []
    finalization = parser.finalize_reasoning_streaming()
    assert finalization is not None and finalization.reasoning_ended
    assert finalization.delta is not None
    assert finalization.delta.reasoning == ""
    assert finalization.delta.content == "Answer without a close token."


@pytest.mark.parametrize("effort", _EFFORTS)
@pytest.mark.parametrize(
    "model_deltas",
    [
        pytest.param(["{start}", "Reasoning"], id="standalone_start"),
        pytest.param(["{start}Reasoning"], id="start_with_text"),
    ],
)
def test_streaming_optional_generated_start_is_not_emitted(
    effort: str, model_deltas: list[str], k2_v3_tokenizer
):
    parser = _make_parser(k2_v3_tokenizer, effort)
    start_token = EFFORT_TOKENS[effort][0]
    emitted = _run_streaming_deltas(
        parser,
        k2_v3_tokenizer,
        [delta.format(start=start_token) for delta in model_deltas],
    )

    assert emitted == []
    finalization = parser.finalize_reasoning_streaming()
    assert finalization is not None and finalization.reasoning_ended
    assert finalization.delta is not None
    assert finalization.delta.reasoning == ""
    assert finalization.delta.content == "Reasoning"


@pytest.mark.parametrize("effort", _EFFORTS)
def test_streaming_multiple_close_tokens_use_first_boundary(
    effort: str, k2_v3_tokenizer
):
    parser = _make_parser(k2_v3_tokenizer, effort)
    end_token = EFFORT_TOKENS[effort][1]
    emitted = _run_streaming_deltas(
        parser,
        k2_v3_tokenizer,
        [f"Reasoning{end_token}{end_token}Answer"],
    )

    assert [(delta.reasoning, delta.content) for delta in emitted] == [
        ("Reasoning", f"{end_token}Answer")
    ]


@pytest.mark.parametrize("effort", _EFFORTS)
def test_streaming_metadata_partition_matches_explicit_boundary(
    effort: str, k2_v3_tokenizer
):
    parser = _make_parser(k2_v3_tokenizer, effort)
    end_token = EFFORT_TOKENS[effort][1]
    reasoning = "Need lookup."
    content = "Answer"

    emitted = _run_streaming_deltas(
        parser,
        k2_v3_tokenizer,
        [reasoning, f"{end_token}{content}"],
    )
    partition = parser.take_reasoning_streaming_metadata_partition()

    assert len(emitted) == 1
    assert partition is not None
    assert partition.reasoning_token_count == len(
        k2_v3_tokenizer.tokenize(f"{reasoning}{end_token}")
    )
    assert partition.content_token_count == len(k2_v3_tokenizer.tokenize(content))
    assert parser.take_reasoning_streaming_metadata_partition() is None


@pytest.mark.parametrize("effort", _EFFORTS)
def test_streaming_metadata_partition_matches_missing_close_tool_boundary(
    effort: str, k2_v3_tokenizer
):
    parser = _make_parser(k2_v3_tokenizer, effort)
    reasoning = "Need lookup. "

    emitted = _run_streaming_deltas(
        parser,
        k2_v3_tokenizer,
        [reasoning, GROUPED_TOOL_CALL],
    )
    finalization = parser.finalize_reasoning_streaming()
    partition = parser.take_reasoning_streaming_metadata_partition()

    assert emitted == []
    assert finalization is not None and finalization.delta is not None
    assert partition is not None
    assert partition.reasoning_token_count == len(k2_v3_tokenizer.tokenize(reasoning))
    assert partition.content_token_count == len(
        k2_v3_tokenizer.tokenize(GROUPED_TOOL_CALL)
    )


@pytest.mark.parametrize("effort", _EFFORTS)
def test_streaming_terminal_partial_plural_marker_is_finalized_as_content(
    effort: str, k2_v3_tokenizer
):
    parser = _make_parser(k2_v3_tokenizer, effort)
    emitted = _run_streaming_deltas(
        parser,
        k2_v3_tokenizer,
        ["Need lookup. <ifm|tool_"],
    )

    assert emitted == []
    finalization = parser.finalize_reasoning_streaming()
    assert finalization is not None
    assert finalization.reasoning_ended
    assert finalization.delta is not None
    assert finalization.delta.reasoning == ""
    assert finalization.delta.content == "Need lookup. <ifm|tool_"
