# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import AsyncExitStack
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from openai.types.responses.tool import (
    CodeInterpreterContainerCodeInterpreterToolAuto,
    LocalShell,
    Mcp,
    Tool,
)

from vllm.entrypoints.context import ConversationContext, SimpleContext
from vllm.entrypoints.openai.protocol import (
    DeltaMessage,
    ErrorResponse,
    ResponsesRequest,
)
from vllm.entrypoints.openai.serving_responses import (
    OpenAIServingResponses,
    _extract_allowed_tools_from_mcp_requests,
    extract_tool_types,
)
from vllm.entrypoints.tool_server import ToolServer
from vllm.inputs.data import TokensPrompt as EngineTokensPrompt
from vllm.reasoning import (
    ReasoningParserStreamingFinalization,
    ReasoningParserStreamingMetadataPartition,
)


class MockConversationContext(ConversationContext):
    """Mock conversation context for testing"""

    def __init__(self):
        self.init_tool_sessions_called = False
        self.init_tool_sessions_args = None
        self.init_tool_sessions_kwargs = None

    def append_output(self, output) -> None:
        pass

    def append_tool_output(self, output) -> None:
        pass

    async def call_tool(self):
        return []

    def need_builtin_tool_call(self) -> bool:
        return False

    def render_for_completion(self):
        return []

    async def init_tool_sessions(self, tool_server, exit_stack, request_id, mcp_tools):
        self.init_tool_sessions_called = True
        self.init_tool_sessions_args = (tool_server, exit_stack, request_id, mcp_tools)

    async def cleanup_session(self) -> None:
        pass


@pytest.fixture
def mock_serving_responses():
    """Create a mock OpenAIServingResponses instance"""
    serving_responses = MagicMock(spec=OpenAIServingResponses)
    serving_responses.tool_server = MagicMock(spec=ToolServer)
    return serving_responses


@pytest.fixture
def mock_context():
    """Create a mock conversation context"""
    return MockConversationContext()


@pytest.fixture
def mock_exit_stack():
    """Create a mock async exit stack"""
    return MagicMock(spec=AsyncExitStack)


def test_extract_tool_types(monkeypatch: pytest.MonkeyPatch) -> None:
    tools: list[Tool] = []
    assert extract_tool_types(tools) == set()

    tools.append(LocalShell(type="local_shell"))
    assert extract_tool_types(tools) == {"local_shell"}

    tools.append(CodeInterpreterContainerCodeInterpreterToolAuto(type="auto"))
    assert extract_tool_types(tools) == {"local_shell", "auto"}

    tools.extend(
        [
            Mcp(type="mcp", server_label="random", server_url=""),
            Mcp(type="mcp", server_label="container", server_url=""),
            Mcp(type="mcp", server_label="code_interpreter", server_url=""),
            Mcp(type="mcp", server_label="web_search_preview", server_url=""),
        ]
    )
    # When envs.VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS is not set,
    # mcp tool types are all ignored.
    assert extract_tool_types(tools) == {"local_shell", "auto"}

    # container is allowed, it would be extracted
    monkeypatch.setenv("VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS", "container")
    assert extract_tool_types(tools) == {"local_shell", "auto", "container"}

    # code_interpreter and web_search_preview are allowed,
    # they would be extracted
    monkeypatch.setenv(
        "VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS", "code_interpreter,web_search_preview"
    )
    assert extract_tool_types(tools) == {
        "local_shell",
        "auto",
        "code_interpreter",
        "web_search_preview",
    }


async def _collect_simple_streaming_events(
    deltas: list[DeltaMessage | None],
    finalization: ReasoningParserStreamingFinalization | None = None,
    terminal_finish_reason: str = "stop",
    metadata_calls: list[dict] | None = None,
    metadata_partitions: (
        list[ReasoningParserStreamingMetadataPartition | None] | None
    ) = None,
    token_ids_per_delta: list[list[int]] | None = None,
):
    from vllm.logprobs import Logprob

    include_logprobs = metadata_calls is not None
    metadata_calls = metadata_calls if metadata_calls is not None else []
    parser = MagicMock()
    parser_calls = []
    pending_deltas = iter(deltas)

    def extract_reasoning_streaming(**kwargs):
        parser_calls.append(
            {
                "previous_text": kwargs["previous_text"],
                "previous_token_ids": list(kwargs["previous_token_ids"]),
            }
        )
        return next(pending_deltas)

    parser.extract_reasoning_streaming.side_effect = extract_reasoning_streaming
    parser.finalize_reasoning_streaming.return_value = finalization
    if metadata_partitions is None:
        parser.take_reasoning_streaming_metadata_partition.return_value = None
    else:
        parser.take_reasoning_streaming_metadata_partition.side_effect = iter(
            metadata_partitions
        )

    serving_responses = MagicMock(spec=OpenAIServingResponses)
    serving_responses.reasoning_parser = lambda _tokenizer: parser

    def create_stream_response_logprobs(**kwargs):
        metadata_calls.append({**kwargs, "token_ids": list(kwargs["token_ids"])})
        return [
            {
                "token": f"token-{token_id}",
                "logprob": -0.1,
                "top_logprobs": [],
            }
            for token_id in kwargs["token_ids"]
        ]

    serving_responses._create_stream_response_logprobs.side_effect = (
        create_stream_response_logprobs
    )

    async def result_generator():
        token_ids_per_delta_ = token_ids_per_delta or [
            [token_id] for token_id in range(1, len(deltas) + 1)
        ]
        for delta_index, token_ids in enumerate(token_ids_per_delta_):
            context = SimpleContext()
            output = MagicMock()
            output.text = f"delta-{delta_index + 1}"
            output.token_ids = token_ids
            output.logprobs = [
                {
                    token_id: Logprob(
                        logprob=-0.1,
                        decoded_token=f"token-{token_id}",
                    )
                }
                for token_id in token_ids
            ]
            output.finish_reason = (
                terminal_finish_reason
                if delta_index == len(token_ids_per_delta_) - 1
                else None
            )
            context.last_output = MagicMock(outputs=[output])
            yield context

    events = [
        event
        async for event in OpenAIServingResponses._process_simple_streaming_events(
            serving_responses,
            request=ResponsesRequest(
                input="test",
                stream=True,
                store=False,
                include=(
                    ["message.output_text.logprobs"] if include_logprobs else None
                ),
            ),
            sampling_params=MagicMock(),
            result_generator=result_generator(),
            context=SimpleContext(),
            model_name="model",
            tokenizer=MagicMock(),
            request_metadata=MagicMock(),
            created_time=0,
            _increment_sequence_number_and_return=lambda event: event,
        )
    ]
    return events, parser_calls


@pytest.mark.asyncio
async def test_simple_streaming_skips_empty_reasoning_boundary():
    events, parser_calls = await _collect_simple_streaming_events(
        [
            DeltaMessage(reasoning=""),
            DeltaMessage(content="The answer is 42."),
        ]
    )

    assert [event.type for event in events] == [
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
    ]
    assert events[0].item.type == "message"
    assert events[2].delta == "The answer is 42."

    assert parser_calls[1]["previous_text"] == "delta-1"
    assert parser_calls[1]["previous_token_ids"] == [1]


@pytest.mark.asyncio
async def test_simple_streaming_treats_combined_empty_reasoning_as_content():
    events, _ = await _collect_simple_streaming_events(
        [DeltaMessage(reasoning="", content="The answer is 42.")]
    )

    assert not any("reasoning" in event.type for event in events)
    text_delta = next(
        event for event in events if event.type == "response.output_text.delta"
    )
    assert text_delta.delta == "The answer is 42."


@pytest.mark.asyncio
async def test_simple_streaming_preserves_nonempty_reasoning_before_boundary():
    events, _ = await _collect_simple_streaming_events(
        [
            DeltaMessage(reasoning="Need to calculate."),
            DeltaMessage(reasoning=""),
            DeltaMessage(content="The answer is 42."),
        ]
    )

    reasoning_deltas = [
        event.delta for event in events if event.type == "response.reasoning_text.delta"
    ]
    reasoning_done = next(
        event for event in events if event.type == "response.reasoning_text.done"
    )
    text_deltas = [
        event.delta for event in events if event.type == "response.output_text.delta"
    ]

    assert reasoning_deltas == ["Need to calculate."]
    assert reasoning_done.text == "Need to calculate."
    assert text_deltas == ["The answer is 42."]


@pytest.mark.asyncio
async def test_simple_streaming_terminal_finalization_preserves_held_content():
    wrapper = "<ifm|tool_calls>held tool markup</ifm|tool_calls>"
    events, _ = await _collect_simple_streaming_events(
        [DeltaMessage(reasoning="Need lookup."), None],
        ReasoningParserStreamingFinalization(
            delta=DeltaMessage(reasoning="", content=wrapper),
            reasoning_ended=True,
        ),
    )

    reasoning_deltas = [
        event.delta for event in events if event.type == "response.reasoning_text.delta"
    ]
    text_deltas = [
        event.delta for event in events if event.type == "response.output_text.delta"
    ]
    assert reasoning_deltas == ["Need lookup."]
    assert text_deltas == [wrapper]


@pytest.mark.asyncio
async def test_simple_streaming_plain_unclosed_output_finalizes_as_content():
    answer = "Plain answer without a reasoning close."
    events, _ = await _collect_simple_streaming_events(
        [None],
        ReasoningParserStreamingFinalization(
            delta=DeltaMessage(reasoning="", content=answer),
            reasoning_ended=True,
        ),
    )

    assert not any("reasoning" in event.type for event in events)
    text_deltas = [
        event.delta for event in events if event.type == "response.output_text.delta"
    ]
    assert text_deltas == [answer]


@pytest.mark.asyncio
async def test_simple_streaming_finalized_content_uses_all_pending_logprobs():
    answer = "Buffered answer."
    metadata_calls: list[dict] = []
    events, _ = await _collect_simple_streaming_events(
        [None, None],
        ReasoningParserStreamingFinalization(
            delta=DeltaMessage(reasoning="", content=answer),
            reasoning_ended=True,
        ),
        metadata_calls=metadata_calls,
    )

    assert len(metadata_calls) == 1
    assert metadata_calls[0]["token_ids"] == [1, 2]
    text_delta = next(
        event for event in events if event.type == "response.output_text.delta"
    )
    assert [logprob.token for logprob in text_delta.logprobs] == [
        "token-1",
        "token-2",
    ]


@pytest.mark.asyncio
async def test_simple_streaming_unbuffered_content_keeps_per_delta_logprobs():
    metadata_calls: list[dict] = []
    await _collect_simple_streaming_events(
        [DeltaMessage(content="First"), DeltaMessage(content="Second")],
        metadata_calls=metadata_calls,
    )

    assert [call["token_ids"] for call in metadata_calls] == [[1], [2]]


@pytest.mark.asyncio
async def test_simple_streaming_combined_delta_keeps_reasoning_and_content():
    metadata_calls: list[dict] = []
    events, _ = await _collect_simple_streaming_events(
        [DeltaMessage(reasoning="Held reasoning", content="Final answer")],
        metadata_calls=metadata_calls,
    )

    reasoning_deltas = [
        event.delta for event in events if event.type == "response.reasoning_text.delta"
    ]
    text_deltas = [
        event.delta for event in events if event.type == "response.output_text.delta"
    ]
    assert reasoning_deltas == ["Held reasoning"]
    assert text_deltas == ["Final answer"]
    # Responses reasoning events have no logprobs field. Without an explicit
    # parser partition, reasoning-token metadata must not be mislabeled as
    # belonging to the following output-text event.
    assert metadata_calls == []


@pytest.mark.asyncio
async def test_simple_streaming_combined_delta_uses_exact_content_partition():
    metadata_calls: list[dict] = []
    events, _ = await _collect_simple_streaming_events(
        [DeltaMessage(reasoning="Held reasoning", content="Final answer")],
        metadata_calls=metadata_calls,
        metadata_partitions=[
            ReasoningParserStreamingMetadataPartition(
                reasoning_token_count=2,
                content_token_count=1,
            )
        ],
        token_ids_per_delta=[[1, 2, 3]],
    )

    assert [call["token_ids"] for call in metadata_calls] == [[3]]
    text_delta = next(
        event for event in events if event.type == "response.output_text.delta"
    )
    assert [logprob.token for logprob in text_delta.logprobs] == ["token-3"]


@pytest.mark.asyncio
async def test_simple_streaming_abort_does_not_finalize_held_content():
    metadata_calls: list[dict] = []
    events, _ = await _collect_simple_streaming_events(
        [None],
        ReasoningParserStreamingFinalization(
            delta=DeltaMessage(
                reasoning="",
                content="<ifm|tool_calls>held</ifm|tool_calls>",
            ),
            reasoning_ended=True,
        ),
        terminal_finish_reason="abort",
        metadata_calls=metadata_calls,
    )

    assert events == []
    assert metadata_calls == []


class TestInitializeToolSessions:
    """Test class for _initialize_tool_sessions method"""

    @pytest_asyncio.fixture
    async def serving_responses_instance(self):
        """Create a real OpenAIServingResponses instance for testing"""
        # Create minimal mocks for required dependencies
        engine_client = MagicMock()

        model_config = MagicMock()
        model_config.hf_config.model_type = "test"
        model_config.get_diff_sampling_param.return_value = {}
        engine_client.model_config = model_config

        engine_client.input_processor = MagicMock()
        engine_client.io_processor = MagicMock()

        models = MagicMock()

        tool_server = MagicMock(spec=ToolServer)

        # Create the actual instance
        instance = OpenAIServingResponses(
            engine_client=engine_client,
            models=models,
            request_logger=None,
            chat_template=None,
            chat_template_content_format="auto",
            tool_server=tool_server,
        )

        return instance

    @pytest.mark.asyncio
    async def test_initialize_tool_sessions(
        self, serving_responses_instance, mock_context, mock_exit_stack
    ):
        """Test that method works correctly with only MCP tools"""

        request = ResponsesRequest(input="test input", tools=[])

        # Call the method
        await serving_responses_instance._initialize_tool_sessions(
            request, mock_context, mock_exit_stack
        )
        assert mock_context.init_tool_sessions_called is False

        # Create only MCP tools
        tools = [
            {"type": "web_search_preview"},
            {"type": "code_interpreter", "container": {"type": "auto"}},
        ]

        request = ResponsesRequest(input="test input", tools=tools)

        # Call the method
        await serving_responses_instance._initialize_tool_sessions(
            request, mock_context, mock_exit_stack
        )

        # Verify that init_tool_sessions was called
        assert mock_context.init_tool_sessions_called

    def test_validate_create_responses_input(
        self, serving_responses_instance, mock_context, mock_exit_stack
    ):
        request = ResponsesRequest(
            input="test input",
            previous_input_messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What is my horoscope? I am an Aquarius.",
                        }
                    ],
                }
            ],
            previous_response_id="lol",
        )
        error = serving_responses_instance._validate_create_responses_input(request)
        assert error is not None
        assert error.error.type == "invalid_request_error"


class TestValidateGeneratorInput:
    """Test class for _validate_generator_input method"""

    @pytest_asyncio.fixture
    async def serving_responses_instance(self):
        """Create a real OpenAIServingResponses instance for testing"""
        # Create minimal mocks for required dependencies
        engine_client = MagicMock()

        model_config = MagicMock()
        model_config.hf_config.model_type = "test"
        model_config.get_diff_sampling_param.return_value = {}
        engine_client.model_config = model_config

        engine_client.input_processor = MagicMock()
        engine_client.io_processor = MagicMock()

        models = MagicMock()

        # Create the actual instance
        instance = OpenAIServingResponses(
            engine_client=engine_client,
            models=models,
            request_logger=None,
            chat_template=None,
            chat_template_content_format="auto",
        )

        # Set max_model_len for testing
        instance.max_model_len = 100

        return instance

    def test_validate_generator_input(self, serving_responses_instance):
        """Test _validate_generator_input with valid prompt length"""
        # Create an engine prompt with valid length (less than max_model_len)
        valid_prompt_token_ids = list(range(5))  # 5 tokens < 100 max_model_len
        engine_prompt = EngineTokensPrompt(prompt_token_ids=valid_prompt_token_ids)

        # Call the method
        result = serving_responses_instance._validate_generator_input(engine_prompt)

        # Should return None for valid input
        assert result is None

        # create an invalid engine prompt
        invalid_prompt_token_ids = list(range(200))  # 100 tokens >= 100 max_model_len
        engine_prompt = EngineTokensPrompt(prompt_token_ids=invalid_prompt_token_ids)

        # Call the method
        result = serving_responses_instance._validate_generator_input(engine_prompt)

        # Should return an ErrorResponse
        assert result is not None
        assert isinstance(result, ErrorResponse)


class TestExtractAllowedToolsFromMcpRequests:
    """Test class for _extract_allowed_tools_from_mcp_requests function"""

    def test_extract_allowed_tools_basic_formats(self):
        """Test extraction with list format, object format, and None."""
        from openai.types.responses.tool import McpAllowedToolsMcpToolFilter

        tools = [
            # List format
            Mcp(
                type="mcp",
                server_label="server1",
                allowed_tools=["tool1", "tool2"],
            ),
            # Object format
            Mcp(
                type="mcp",
                server_label="server2",
                allowed_tools=McpAllowedToolsMcpToolFilter(
                    tool_names=["tool3", "tool4"]
                ),
            ),
            # None (no filter)
            Mcp(
                type="mcp",
                server_label="server3",
                allowed_tools=None,
            ),
        ]
        result = _extract_allowed_tools_from_mcp_requests(tools)
        assert result == {
            "server1": ["tool1", "tool2"],
            "server2": ["tool3", "tool4"],
            "server3": None,
        }

    def test_extract_allowed_tools_star_normalization(self):
        """Test that '*' wildcard is normalized to None (select all tools).

        This is the key test requested by reviewers to explicitly demonstrate
        that the "*" select-all scenario is handled correctly.
        """
        from openai.types.responses.tool import McpAllowedToolsMcpToolFilter

        tools = [
            # Star in list format
            Mcp(
                type="mcp",
                server_label="server1",
                allowed_tools=["*"],
            ),
            # Star mixed with other tools in list
            Mcp(
                type="mcp",
                server_label="server2",
                allowed_tools=["tool1", "*"],
            ),
            # Star in object format
            Mcp(
                type="mcp",
                server_label="server3",
                allowed_tools=McpAllowedToolsMcpToolFilter(tool_names=["*"]),
            ),
        ]
        result = _extract_allowed_tools_from_mcp_requests(tools)
        # All should be normalized to None (allows all tools)
        assert result == {
            "server1": None,
            "server2": None,
            "server3": None,
        }

    def test_extract_allowed_tools_filters_non_mcp(self):
        """Test that non-MCP tools are ignored during extraction."""
        tools = [
            Mcp(
                type="mcp",
                server_label="server1",
                allowed_tools=["tool1"],
            ),
            LocalShell(type="local_shell"),  # Non-MCP tool should be ignored
            Mcp(
                type="mcp",
                server_label="server2",
                allowed_tools=["tool2"],
            ),
        ]
        result = _extract_allowed_tools_from_mcp_requests(tools)
        # Non-MCP tools should be ignored
        assert result == {
            "server1": ["tool1"],
            "server2": ["tool2"],
        }
