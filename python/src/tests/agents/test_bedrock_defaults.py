"""Request-level checks for the Bedrock default model and sampling settings.

Every request the Bedrock agents and the classifier build is validated against
botocore's Converse input shape, so a misnamed or malformed field fails here
instead of at the Bedrock endpoint.
"""

from copy import deepcopy
from unittest.mock import Mock

import boto3
import pytest
from botocore.validate import validate_parameters

from agent_squad.agents import (
    BedrockInlineAgent,
    BedrockInlineAgentOptions,
    BedrockLLMAgent,
    BedrockLLMAgentOptions,
    BedrockTranslatorAgent,
    BedrockTranslatorAgentOptions,
)
from agent_squad.classifiers import BedrockClassifier, BedrockClassifierOptions
from agent_squad.types import (
    BEDROCK_MODEL_ID_CLAUDE_3_HAIKU,
    BEDROCK_MODEL_ID_CLAUDE_3_SONNET,
    BEDROCK_MODEL_ID_CLAUDE_3_5_SONNET,
    BEDROCK_MODEL_ID_CLAUDE_HAIKU_4_5,
)

MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
KINDS = ["llm", "inline", "translator", "classifier"]


@pytest.fixture
def runtime_client():
    client = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    converse_shape = client.meta.service_model.operation_model("Converse").input_shape
    stream_shape = client.meta.service_model.operation_model("ConverseStream").input_shape

    def converse(**params):
        validate_parameters(params, converse_shape)
        content = [{"text": "hello"}]
        tool_name = params.get("toolConfig", {}).get("toolChoice", {}).get("tool", {}).get("name")
        if tool_name == "Translate":
            content = [{"toolUse": {"name": "Translate", "input": {"translation": "hello"}}}]
        elif "analyzePrompt" in str(params.get("toolConfig")):
            content = [{"toolUse": {"input": {"userinput": "hello", "selected_agent": "", "confidence": 0.8}}}]
        return {"output": {"message": {"role": "assistant", "content": content}}}

    def converse_stream(**params):
        validate_parameters(params, stream_shape)
        return {"stream": iter([])}

    client.converse = Mock(side_effect=converse)
    client.converse_stream = Mock(side_effect=converse_stream)
    yield client
    client.close()


def make_agent(kind, client, inference_config=None, model_id=None):
    options = {"client": client, "inference_config": inference_config, "model_id": model_id}
    if kind == "classifier":
        classifier = BedrockClassifier(BedrockClassifierOptions(**options))
        classifier.set_agents({})
        return classifier
    options.update(name="Test", description="Test requests")
    if kind == "inline":
        return BedrockInlineAgent(BedrockInlineAgentOptions(**options, bedrock_agent_client=Mock()))
    if kind == "translator":
        return BedrockTranslatorAgent(BedrockTranslatorAgentOptions(**options))
    return BedrockLLMAgent(BedrockLLMAgentOptions(**options))


async def send_request(agent):
    if isinstance(agent, BedrockClassifier):
        return await agent.classify("hello", [])
    return await agent.process_request("hello", "user", "session", [])


def top_p_only_config(kind):
    # The classifier reads snake_case keys, the agents take the Converse names.
    return {"top_p": 0.8} if kind == "classifier" else {"topP": 0.8}


def full_config(kind):
    if kind == "classifier":
        return {"temperature": 0.4, "top_p": 0.8, "maxTokens": 500, "stop_sequences": ["END"]}
    return {"temperature": 0.4, "topP": 0.8, "maxTokens": 500, "stopSequences": ["END"]}


def test_default_model_constant_is_the_global_haiku_4_5_profile():
    assert BEDROCK_MODEL_ID_CLAUDE_HAIKU_4_5 == MODEL_ID


def test_retired_model_constants_keep_their_values():
    assert BEDROCK_MODEL_ID_CLAUDE_3_HAIKU == "anthropic.claude-3-haiku-20240307-v1:0"
    assert BEDROCK_MODEL_ID_CLAUDE_3_SONNET == "anthropic.claude-3-sonnet-20240229-v1:0"
    assert BEDROCK_MODEL_ID_CLAUDE_3_5_SONNET == "anthropic.claude-3-5-sonnet-20240620-v1:0"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_default_request_uses_haiku_4_5_and_one_sampling_parameter(kind, runtime_client):
    await send_request(make_agent(kind, runtime_client))

    sent = runtime_client.converse.call_args.kwargs
    assert sent["modelId"] == MODEL_ID
    # Claude Haiku 4.5 rejects temperature and topP in the same request.
    assert "topP" not in sent["inferenceConfig"]
    assert sent["inferenceConfig"]["temperature"] == 0.0


@pytest.mark.asyncio
async def test_default_streaming_request_uses_haiku_4_5(runtime_client):
    agent = BedrockLLMAgent(
        BedrockLLMAgentOptions(name="Test", description="Test requests", client=runtime_client, streaming=True)
    )

    async for _ in await send_request(agent):
        pass

    sent = runtime_client.converse_stream.call_args.kwargs
    assert sent["modelId"] == MODEL_ID
    assert "topP" not in sent["inferenceConfig"]
    assert sent["inferenceConfig"]["temperature"] == 0.0


@pytest.mark.asyncio
async def test_inline_agent_foundation_model_defaults_to_haiku_4_5(runtime_client):
    assert make_agent("inline", runtime_client).foundation_model == MODEL_ID


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_top_p_without_temperature_does_not_add_the_default_temperature(kind, runtime_client):
    config = top_p_only_config(kind)
    original = deepcopy(config)

    await send_request(make_agent(kind, runtime_client, config))

    sent = runtime_client.converse.call_args.kwargs["inferenceConfig"]
    assert sent["topP"] == 0.8
    assert "temperature" not in sent
    assert config == original


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_explicit_model_and_sampling_values_are_sent_unchanged(kind, runtime_client):
    await send_request(make_agent(kind, runtime_client, full_config(kind), "custom-model"))

    sent = runtime_client.converse.call_args.kwargs
    assert sent["modelId"] == "custom-model"
    assert sent["inferenceConfig"] == {
        "temperature": 0.4,
        "topP": 0.8,
        "maxTokens": 500,
        "stopSequences": ["END"],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_the_retired_default_model_can_still_be_requested_explicitly(kind, runtime_client):
    await send_request(make_agent(kind, runtime_client, model_id=BEDROCK_MODEL_ID_CLAUDE_3_HAIKU))

    assert runtime_client.converse.call_args.kwargs["modelId"] == BEDROCK_MODEL_ID_CLAUDE_3_HAIKU


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_zero_top_p_is_preserved_without_a_default_temperature(kind, runtime_client):
    key = "top_p" if kind == "classifier" else "topP"
    await send_request(make_agent(kind, runtime_client, {key: 0.0}))

    sent = runtime_client.converse.call_args.kwargs["inferenceConfig"]
    assert sent["topP"] == 0.0
    assert "temperature" not in sent


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_none_omits_sampling_parameters(kind, runtime_client):
    key = "top_p" if kind == "classifier" else "topP"
    await send_request(make_agent(kind, runtime_client, {key: None, "temperature": None}))

    sent = runtime_client.converse.call_args.kwargs["inferenceConfig"]
    assert "topP" not in sent
    assert "temperature" not in sent


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config, expected",
    [
        ({"topP": 0.8, "stopSequences": ["END"]}, {"maxTokens": 1000, "topP": 0.8, "stopSequences": ["END"]}),
        ({"topP": 0.0, "top_p": 0.8}, {"maxTokens": 1000, "topP": 0.0, "stopSequences": []}),
        (
            {"topP": None, "top_p": 0.8, "stopSequences": [], "stop_sequences": ["END"]},
            {"maxTokens": 1000, "temperature": 0.0, "stopSequences": []},
        ),
    ],
)
async def test_classifier_accepts_converse_names_with_precedence_over_legacy_aliases(config, expected, runtime_client):
    original = deepcopy(config)
    await send_request(make_agent("classifier", runtime_client, config))

    assert runtime_client.converse.call_args.kwargs["inferenceConfig"] == expected
    assert config == original


@pytest.mark.asyncio
@pytest.mark.parametrize("config", [{"maxTokens": 3000}, {"maxTokens": 3000, "temperature": 0.5, "topP": 0.8}])
async def test_thinking_omits_sampling_parameters(runtime_client, config):
    original = deepcopy(config)
    agent = BedrockLLMAgent(
        BedrockLLMAgentOptions(
            name="Test",
            description="Test requests",
            client=runtime_client,
            inference_config=config,
            additional_model_request_fields={"thinking": {"type": "enabled", "budget_tokens": 2000}},
        )
    )
    await send_request(agent)

    sent = runtime_client.converse.call_args.kwargs
    assert "temperature" not in sent["inferenceConfig"]
    assert "topP" not in sent["inferenceConfig"]
    assert sent["inferenceConfig"]["maxTokens"] == 3000
    assert sent["additionalModelRequestFields"]["thinking"] == {"type": "enabled", "budget_tokens": 2000}
    assert config == original


@pytest.mark.asyncio
@pytest.mark.parametrize("foundation_model", [None, "custom-foundation-profile"])
async def test_inline_foundation_model_reaches_the_agent_request(runtime_client, foundation_model):
    agent = make_agent("inline", runtime_client)
    if foundation_model is not None:
        agent = BedrockInlineAgent(
            BedrockInlineAgentOptions(
                name="Test",
                description="Test requests",
                client=runtime_client,
                bedrock_agent_client=Mock(),
                foundation_model=foundation_model,
            )
        )
    runtime_client.converse.return_value = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "name": "inline_agent_creation",
                            "input": {
                                "action_group_names": [],
                                "knowledge_bases": [],
                                "description": "Answer the question",
                                "user_request": "hello",
                            },
                        }
                    }
                ],
            }
        }
    }
    runtime_client.converse.side_effect = None
    agent.bedrock_agent_client.invoke_inline_agent.return_value = {"completion": []}

    await send_request(agent)

    assert agent.bedrock_agent_client.invoke_inline_agent.call_args.kwargs["foundationModel"] == (
        foundation_model or MODEL_ID
    )
