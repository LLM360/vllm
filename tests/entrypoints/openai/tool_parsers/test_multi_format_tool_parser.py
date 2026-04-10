# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from tests.entrypoints.openai.tool_parsers.utils import run_tool_extraction_nonstreaming
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.tool_parsers import ToolParser, ToolParserManager

pytestmark = pytest.mark.cpu_test


class FakeTokenizer:
    def __init__(self):
        self._vocab: dict[str, int] = {}
        self._next_token_id = 1

    def get_vocab(self):
        return self._vocab

    def encode(self, text: str, add_special_tokens: bool = False):
        if text not in self._vocab:
            self._vocab[text] = self._next_token_id
            self._next_token_id += 1
        return [self._vocab[text]]

    def decode(self, token_ids):
        reverse_vocab = {token_id: token for token, token_id in self._vocab.items()}
        return "".join(reverse_vocab[token_id] for token_id in token_ids)


def make_parser(tool_format: str) -> ToolParser:
    return ToolParserManager.get_tool_parser("multi_format")(
        FakeTokenizer(),
        chat_template_kwargs={"tool_format": tool_format},
    )


def make_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test-model",
        messages=[],
    )


def test_default_format_delegates_to_hermes():
    parser = make_parser("default")

    extracted = run_tool_extraction_nonstreaming(
        parser,
        '<tool_call>\n{"name":"get_weather","arguments":{"city":"Tokyo"}}\n</tool_call>',
        make_request(),
    )

    assert extracted.tools_called
    assert extracted.content is None
    assert extracted.tool_calls[0].function.name == "get_weather"
    assert json.loads(extracted.tool_calls[0].function.arguments) == {"city": "Tokyo"}


def test_qwen3_format_delegates_to_qwen3xml():
    parser = make_parser("qwen3")

    extracted = run_tool_extraction_nonstreaming(
        parser,
        "<tool_call>\n<function=get_weather>\n"
        "<parameter=city>Tokyo</parameter>\n"
        "</function>\n</tool_call>",
        make_request(),
    )

    assert extracted.tools_called
    assert extracted.tool_calls[0].function.name == "get_weather"
    assert json.loads(extracted.tool_calls[0].function.arguments) == {"city": "Tokyo"}


def test_glm_format_delegates_to_glm47():
    parser = make_parser("glm")

    extracted = run_tool_extraction_nonstreaming(
        parser,
        "<tool_call>get_weather\n<arg_key>city</arg_key>"
        "<arg_value>Beijing</arg_value></tool_call>",
        make_request(),
    )

    assert extracted.tools_called
    assert extracted.tool_calls[0].function.name == "get_weather"
    assert json.loads(extracted.tool_calls[0].function.arguments) == {
        "city": "Beijing"
    }


def test_minimax_format_extracts_inline_invokes():
    parser = make_parser("minimax")

    extracted = run_tool_extraction_nonstreaming(
        parser,
        "Checking."
        '<tool_calls><invoke name="get_weather">'
        '<parameter name="city">Tokyo</parameter>'
        '<parameter name="days">5</parameter>'
        '<parameter name="enabled">true</parameter>'
        '<parameter name="filters">{"units":"metric"}</parameter>'
        "</invoke></tool_calls>",
        make_request(),
    )

    assert extracted.tools_called
    assert extracted.content == "Checking."
    assert json.loads(extracted.tool_calls[0].function.arguments) == {
        "city": "Tokyo",
        "days": 5,
        "enabled": True,
        "filters": {"units": "metric"},
    }


def test_dsv32_format_honors_string_attribute():
    parser = make_parser("dsv32")

    extracted = run_tool_extraction_nonstreaming(
        parser,
        "Prefix"
        "<tool_calls>"
        '<invoke name="get_weather">'
        '<parameter name="city" string="true">Tokyo</parameter>'
        '<parameter name="days" string="false">5</parameter>'
        '<parameter name="enabled" string="false">false</parameter>'
        "</invoke>"
        "</tool_calls>",
        make_request(),
    )

    assert extracted.tools_called
    assert extracted.content == "Prefix"
    assert json.loads(extracted.tool_calls[0].function.arguments) == {
        "city": "Tokyo",
        "days": 5,
        "enabled": False,
    }


def test_gptoss_format_extracts_multiple_calls():
    parser = make_parser("gptoss")

    extracted = run_tool_extraction_nonstreaming(
        parser,
        "Planning...\n"
        "<tool_call>to=functions.get_weather json\n"
        '{"location":"SF"}\n'
        "</tool_call>"
        "<tool_call>to=functions.get_time json\n"
        '{"timezone":"UTC"}\n'
        "</tool_call>",
        make_request(),
    )

    assert extracted.tools_called
    assert extracted.content == "Planning...\n"
    assert [tool_call.function.name for tool_call in extracted.tool_calls] == [
        "get_weather",
        "get_time",
    ]
    assert json.loads(extracted.tool_calls[0].function.arguments) == {"location": "SF"}
    assert json.loads(extracted.tool_calls[1].function.arguments) == {
        "timezone": "UTC"
    }


def test_gptoss_format_with_assistant_prefix():
    parser = make_parser("gptoss")

    extracted = run_tool_extraction_nonstreaming(
        parser,
        '<tool_call>assistant to=functions.get_weather json\n'
        '{"location": "San Francisco, CA", "unit": "celsius"}\n'
        "</tool_call>",
        make_request(),
    )

    assert extracted.tools_called
    assert extracted.tool_calls[0].function.name == "get_weather"
    assert json.loads(extracted.tool_calls[0].function.arguments) == {
        "location": "San Francisco, CA",
        "unit": "celsius",
    }


def test_python_format_extracts_single_call():
    parser = make_parser("python")

    extracted = run_tool_extraction_nonstreaming(
        parser,
        '<tool_call>\nget_weather(city="SF")\n</tool_call>',
        make_request(),
    )

    assert extracted.tools_called
    assert extracted.content is None
    assert extracted.tool_calls[0].function.name == "get_weather"
    assert json.loads(extracted.tool_calls[0].function.arguments) == {"city": "SF"}


def test_python_format_extracts_multiple_calls():
    parser = make_parser("python")

    extracted = run_tool_extraction_nonstreaming(
        parser,
        '<tool_call>\nget_weather(city="SF")\n</tool_call>'
        "\n"
        '<tool_call>\nget_time(timezone="UTC")\n</tool_call>',
        make_request(),
    )

    assert extracted.tools_called
    assert [tool_call.function.name for tool_call in extracted.tool_calls] == [
        "get_weather",
        "get_time",
    ]
    assert json.loads(extracted.tool_calls[0].function.arguments) == {"city": "SF"}
    assert json.loads(extracted.tool_calls[1].function.arguments) == {
        "timezone": "UTC"
    }


def test_custom_formats_do_not_stream_yet():
    parser = make_parser("python")

    delta = parser.extract_tool_calls_streaming(
        previous_text="",
        current_text="<tool_call>",
        delta_text="<tool_call>",
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=make_request(),
    )

    assert delta is None


def test_readme_default_example():
    parser = make_parser("default")
    extracted = run_tool_extraction_nonstreaming(
        parser,
        '<tool_call>\n'
        '{"name": "get_weather", "arguments": {"location": "San Francisco, CA"}}\n'
        "</tool_call>",
        make_request(),
    )
    assert extracted.tools_called
    assert extracted.tool_calls[0].function.name == "get_weather"
    assert json.loads(extracted.tool_calls[0].function.arguments) == {
        "location": "San Francisco, CA"
    }


def test_readme_qwen3_example():
    parser = make_parser("qwen3")
    extracted = run_tool_extraction_nonstreaming(
        parser,
        '<tool_call>\n'
        '<function=get_weather>\n'
        '<parameter=location>\n'
        'San Francisco, CA\n'
        '</parameter>\n'
        '</function>\n'
        '</tool_call>',
        make_request(),
    )
    assert extracted.tools_called
    assert extracted.tool_calls[0].function.name == "get_weather"
    assert json.loads(extracted.tool_calls[0].function.arguments) == {
        "location": "San Francisco, CA"
    }


def test_readme_minimax_example():
    parser = make_parser("minimax")
    extracted = run_tool_extraction_nonstreaming(
        parser,
        '<tool_calls>\n'
        '<invoke name="get_weather">\n'
        '<parameter name="location">San Francisco, CA</parameter>\n'
        '<parameter name="unit">celsius</parameter>\n'
        '</invoke>\n'
        '</tool_calls>',
        make_request(),
    )
    assert extracted.tools_called
    assert extracted.tool_calls[0].function.name == "get_weather"
    assert json.loads(extracted.tool_calls[0].function.arguments) == {
        "location": "San Francisco, CA",
        "unit": "celsius",
    }


def test_readme_glm_example():
    parser = make_parser("glm")
    extracted = run_tool_extraction_nonstreaming(
        parser,
        "<tool_call>get_weather\n"
        "<arg_key>location</arg_key><arg_value>San Francisco, CA</arg_value>"
        "</tool_call>",
        make_request(),
    )
    assert extracted.tools_called
    assert extracted.tool_calls[0].function.name == "get_weather"
    assert json.loads(extracted.tool_calls[0].function.arguments) == {
        "location": "San Francisco, CA"
    }


def test_readme_dsv32_example():
    parser = make_parser("dsv32")
    extracted = run_tool_extraction_nonstreaming(
        parser,
        '<tool_calls>\n'
        '<invoke name="get_weather">\n'
        '<parameter name="location" string="true">San Francisco, CA</parameter>\n'
        '<parameter name="unit" string="true">celsius</parameter>\n'
        '</invoke>\n'
        '</tool_calls>',
        make_request(),
    )
    assert extracted.tools_called
    assert extracted.tool_calls[0].function.name == "get_weather"
    args = json.loads(extracted.tool_calls[0].function.arguments)
    assert args == {"location": "San Francisco, CA", "unit": "celsius"}
    assert isinstance(args["location"], str)
    assert isinstance(args["unit"], str)


def test_readme_gptoss_example():
    parser = make_parser("gptoss")
    extracted = run_tool_extraction_nonstreaming(
        parser,
        '<tool_call>assistant to=functions.get_weather json\n'
        '{"location": "San Francisco, CA", "unit": "celsius"}\n'
        "</tool_call>",
        make_request(),
    )
    assert extracted.tools_called
    assert extracted.tool_calls[0].function.name == "get_weather"
    assert json.loads(extracted.tool_calls[0].function.arguments) == {
        "location": "San Francisco, CA",
        "unit": "celsius",
    }


def test_readme_python_example():
    parser = make_parser("python")
    extracted = run_tool_extraction_nonstreaming(
        parser,
        '<tool_call>\n'
        'get_weather(location="San Francisco, CA", unit="celsius")\n'
        '</tool_call>',
        make_request(),
    )
    assert extracted.tools_called
    assert extracted.tool_calls[0].function.name == "get_weather"
    assert json.loads(extracted.tool_calls[0].function.arguments) == {
        "location": "San Francisco, CA",
        "unit": "celsius",
    }
