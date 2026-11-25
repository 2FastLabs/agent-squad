from unittest.mock import Mock

import boto3
import pytest
from botocore.validate import validate_parameters

from product_search_agent import ProductSearchAgent, ProductSearchAgentOptions


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sampling, expected",
    [
        ({}, {"temperature": 0.0}),
        ({"temperature": None, "top_p": 0.8}, {"topP": 0.8}),
    ],
)
async def test_search_request_uses_valid_sampling(sampling, expected):
    client = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    shape = client.meta.service_model.operation_model("Converse").input_shape

    def converse(**params):
        validate_parameters(params, shape)
        return {"output": {"message": {"content": [{"text": "[]"}]}}}

    client.converse = Mock(side_effect=converse)
    try:
        agent = ProductSearchAgent(
            ProductSearchAgentOptions(
                name="Search",
                description="Search products",
                client=client,
                model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0",
                region="us-east-1",
                **sampling,
            )
        )
        await agent.process_request("find a product", "user", "session", [])

        sent = client.converse.call_args.kwargs
        assert sent["modelId"] == "global.anthropic.claude-haiku-4-5-20251001-v1:0"
        assert sent["inferenceConfig"] == {
            "maxTokens": 1000,
            "stopSequences": [],
            **expected,
        }
    finally:
        client.close()
