import boto3
import pytest
from botocore.stub import Stubber

from agent_squad.agents import Agent, AgentOptions
from agent_squad.classifiers import BedrockClassifier, BedrockClassifierOptions


class MockAgent(Agent):
    async def process_request(self, *args, **kwargs):
        raise NotImplementedError


@pytest.mark.asyncio
async def test_converse_request_passes_botocore_validation():
    # Stubber validates parameters like the real client, so extra keys on the
    # message (e.g. ConversationMessage.citations) would fail here.
    client = boto3.client(
        'bedrock-runtime',
        region_name='us-east-1',
        aws_access_key_id='test',
        aws_secret_access_key='test',
    )
    classifier = BedrockClassifier(BedrockClassifierOptions(
        model_id='anthropic.claude-test', client=client))
    agent = MockAgent(AgentOptions(name='Billing Agent', description='Handles billing'))
    classifier.set_agents({agent.id: agent})

    with Stubber(client) as stubber:
        stubber.add_response('converse', {
            'output': {'message': {'role': 'assistant', 'content': [{'toolUse': {
                'toolUseId': 'tool-1',
                'name': 'analyzePrompt',
                'input': {'userinput': 'refund', 'selected_agent': 'billing-agent', 'confidence': 0.8},
            }}]}},
            'stopReason': 'tool_use',
            'usage': {'inputTokens': 10, 'outputTokens': 5, 'totalTokens': 15},
            'metrics': {'latencyMs': 1},
        })

        result = await classifier.classify('I want a refund', [])

    assert result.selected_agent is agent
    assert result.confidence == 0.8


def _stubbed_classifier(inference_config=None):
    client = boto3.client(
        'bedrock-runtime',
        region_name='us-east-1',
        aws_access_key_id='test',
        aws_secret_access_key='test',
    )
    classifier = BedrockClassifier(BedrockClassifierOptions(
        model_id='anthropic.claude-test', client=client, inference_config=inference_config))
    agent = MockAgent(AgentOptions(name='Billing Agent', description='Handles billing'))
    classifier.set_agents({agent.id: agent})
    return classifier, client


_TOOL_RESPONSE = {
    'output': {'message': {'role': 'assistant', 'content': [{'toolUse': {
        'toolUseId': 'tool-1',
        'name': 'analyzePrompt',
        'input': {'userinput': 'refund', 'selected_agent': 'billing-agent', 'confidence': 0.8},
    }}]}},
    'stopReason': 'tool_use',
    'usage': {'inputTokens': 10, 'outputTokens': 5, 'totalTokens': 15},
    'metrics': {'latencyMs': 1},
}


async def _sent_inference_config(inference_config=None):
    classifier, client = _stubbed_classifier(inference_config)
    captured = {}
    original = client.converse

    def converse(**kwargs):
        captured.update(kwargs)
        return original(**kwargs)

    client.converse = converse
    with Stubber(client) as stubber:
        stubber.add_response('converse', _TOOL_RESPONSE)
        await classifier.classify('I want a refund', [])
    return captured['inferenceConfig']


@pytest.mark.asyncio
async def test_sampling_defaults_are_sent():
    config = await _sent_inference_config()
    assert config['temperature'] == 0.0
    # Claude Haiku 4.5 and newer reject temperature and topP together.
    assert 'topP' not in config


@pytest.mark.asyncio
async def test_top_p_is_sent_when_set():
    config = await _sent_inference_config({'top_p': 0.9})
    assert config['topP'] == 0.9


@pytest.mark.asyncio
async def test_sampling_params_set_to_none_are_omitted():
    # Models such as Claude Opus 5 reject temperature/topP with a 400.
    config = await _sent_inference_config({'temperature': None, 'top_p': None})
    assert 'temperature' not in config
    assert 'topP' not in config
    assert config['maxTokens'] == 1000
