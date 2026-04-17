"""
Integration tests for MiniMaxClassifier.
These tests verify the classifier works with the actual MiniMax API.
They are skipped by default unless MINIMAX_API_KEY is set.
"""
import os
import pytest
from agent_squad.agents import Agent, AgentOptions
from agent_squad.types import ConversationMessage, ParticipantRole


MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY")
skip_no_key = pytest.mark.skipif(
    not MINIMAX_API_KEY,
    reason="MINIMAX_API_KEY not set"
)


class MockAgent(Agent):
    """A minimal agent for testing the classifier."""
    async def process_request(self, input_text, user_id, session_id, chat_history, additional_params=None):
        return ConversationMessage(
            role=ParticipantRole.ASSISTANT.value,
            content=[{"text": f"Response from {self.name}"}]
        )


@skip_no_key
class TestMiniMaxClassifierIntegration:
    @pytest.fixture
    def classifier(self):
        from agent_squad.classifiers.minimax_classifier import MiniMaxClassifier, MiniMaxClassifierOptions
        classifier = MiniMaxClassifier(MiniMaxClassifierOptions(
            api_key=MINIMAX_API_KEY,
            model_id="MiniMax-M2.7",
            inference_config={
                'max_tokens': 200,
                'temperature': 0.1,
            }
        ))

        tech_agent = MockAgent(AgentOptions(
            name="TechAgent",
            description="Handles technology and programming questions"
        ))
        health_agent = MockAgent(AgentOptions(
            name="HealthAgent",
            description="Handles health and medical questions"
        ))

        classifier.set_agents({tech_agent.id: tech_agent, health_agent.id: health_agent})
        return classifier

    @pytest.mark.asyncio
    async def test_classify_tech_question(self, classifier):
        result = await classifier.classify(
            "How do I write a Python function?",
            []
        )
        assert result is not None
        assert result.selected_agent is not None
        assert result.confidence > 0
