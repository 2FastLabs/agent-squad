"""
Integration tests for MiniMaxAgent.
These tests verify the agent works with the actual MiniMax API.
They are skipped by default unless MINIMAX_API_KEY is set.
"""
import os
import pytest
from agent_squad.types import ConversationMessage, ParticipantRole
from agent_squad.agents import AgentStreamResponse


MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY")
skip_no_key = pytest.mark.skipif(
    not MINIMAX_API_KEY,
    reason="MINIMAX_API_KEY not set"
)


@skip_no_key
class TestMiniMaxAgentIntegration:
    @pytest.fixture
    def agent(self):
        from agent_squad.agents.minimax_agent import MiniMaxAgent, MiniMaxAgentOptions
        return MiniMaxAgent(MiniMaxAgentOptions(
            name="TestAgent",
            description="A helpful test agent",
            api_key=MINIMAX_API_KEY,
            model="MiniMax-M2.7",
            streaming=False,
            inference_config={
                'maxTokens': 100,
                'temperature': 0.5,
            }
        ))

    @pytest.fixture
    def streaming_agent(self):
        from agent_squad.agents.minimax_agent import MiniMaxAgent, MiniMaxAgentOptions
        return MiniMaxAgent(MiniMaxAgentOptions(
            name="TestStreamingAgent",
            description="A helpful test agent with streaming",
            api_key=MINIMAX_API_KEY,
            model="MiniMax-M2.7",
            streaming=True,
            inference_config={
                'maxTokens': 100,
                'temperature': 0.5,
            }
        ))

    @pytest.mark.asyncio
    async def test_single_response(self, agent):
        result = await agent.process_request(
            "What is 2 + 2? Answer with just the number.",
            "test_user",
            "test_session",
            []
        )

        assert isinstance(result, ConversationMessage)
        assert result.role == ParticipantRole.ASSISTANT.value
        assert result.content
        assert "4" in result.content[0]['text']

    @pytest.mark.asyncio
    async def test_streaming_response(self, streaming_agent):
        result = await streaming_agent.process_request(
            "Say 'hello world' and nothing else.",
            "test_user",
            "test_session",
            []
        )

        chunks = []
        final_message = None
        async for chunk in result:
            assert isinstance(chunk, AgentStreamResponse)
            if chunk.text:
                chunks.append(chunk.text)
            if chunk.final_message:
                final_message = chunk.final_message

        assert len(chunks) > 0
        assert final_message is not None
        assert final_message.role == ParticipantRole.ASSISTANT.value

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, agent):
        result1 = await agent.process_request(
            "My name is Alice.",
            "test_user",
            "test_session",
            []
        )
        assert isinstance(result1, ConversationMessage)

        chat_history = [
            ConversationMessage(
                role=ParticipantRole.USER.value,
                content=[{"text": "My name is Alice."}]
            ),
            result1
        ]

        result2 = await agent.process_request(
            "What is my name?",
            "test_user",
            "test_session",
            chat_history
        )
        assert isinstance(result2, ConversationMessage)
        assert "Alice" in result2.content[0]['text']
