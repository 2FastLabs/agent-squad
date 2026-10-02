from unittest.mock import AsyncMock, Mock, call

import pytest
from openai.types.chat import ChatCompletionChunk

from agent_squad.agents import OpenAIAgent, OpenAIAgentOptions


def _chunk(content=None, *, usage_only=False):
    return ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "test-model",
            "choices": [] if usage_only else [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3} if usage_only else None,
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("usage_position", ["start", "middle", "end", "only"])
async def test_usage_only_chunks_preserve_stream_and_final_message(usage_position):
    client = Mock()
    agent = OpenAIAgent(
        OpenAIAgentOptions(
            name="TestAgent", description="Test streaming", api_key="test-key", client=client, streaming=True
        )
    )
    agent.callbacks.on_llm_new_token = AsyncMock()
    chunks = [_chunk(), _chunk("hello "), _chunk("world")]
    if usage_position == "only":
        chunks = [_chunk(usage_only=True)]
        expected_text = ""
        expected_calls = []
    else:
        chunks.insert({"start": 0, "middle": 2, "end": 3}[usage_position], _chunk(usage_only=True))
        expected_text = "hello world"
        expected_calls = [call("hello "), call("world")]
    client.chat.completions.create.return_value = iter(chunks)

    stream = await agent.process_request("hello", "user", "session", [])
    responses = [response async for response in stream]

    assert "".join(response.text or "" for response in responses) == expected_text
    assert responses[-1].final_message.content == [{"text": expected_text}]
    assert responses[-1].final_message.role == "assistant"
    assert agent.callbacks.on_llm_new_token.await_args_list == expected_calls
    client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_stream_errors_after_usage_chunks_still_propagate():
    client = Mock()
    agent = OpenAIAgent(
        OpenAIAgentOptions(
            name="TestAgent", description="Test streaming", api_key="test-key", client=client, streaming=True
        )
    )

    def failing_stream():
        yield _chunk(usage_only=True)
        raise RuntimeError("stream disconnected")

    client.chat.completions.create.return_value = failing_stream()
    with pytest.raises(RuntimeError, match="stream disconnected"):
        async for _ in agent.handle_streaming_response({"stream": True}):
            pass
