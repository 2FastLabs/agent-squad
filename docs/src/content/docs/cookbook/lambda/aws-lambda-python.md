---
title: AWS Lambda Python with Agent Squad
description: How to set up the Agent Squad System for AWS Lambda using Python
---

The Agent Squad framework can be used inside an AWS Lambda function like any other library. This guide outlines the process of setting up the Agent Squad System for use with AWS Lambda using Python.

## Prerequisites

- AWS account with appropriate permissions
- Python 3.12 or later installed
- Basic familiarity with AWS Lambda and Python

## Installation and Setup

1. **Create a New Project Directory**

   ```bash
   mkdir multi-agent-lambda && cd multi-agent-lambda
   ```

2. **Create and Activate a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the Agent Squad framework**

   ```bash
   pip install agent-squad boto3
   ```

4. **Create Requirements File**

   ```bash
   pip freeze > requirements.txt
   ```

## Lambda Function Structure

Create a new file named `lambda_function.py` in your project directory. Here's a high-level overview of what your Lambda function should include:

```python
import json
import asyncio
from typing import Dict, Any
from agent_squad.orchestrator import AgentSquad, AgentSquadConfig
from agent_squad.agents import BedrockLLMAgent, BedrockLLMAgentOptions, AgentResponse
from agent_squad.types import ConversationMessage

# Initialize orchestrator
orchestrator = AgentSquad(AgentSquadConfig(
    # Configuration options
))

# Add agents e.g Tech Agent
tech_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
    name="Tech Agent",
    streaming=False,
    description="Specializes in technology areas including software development, hardware, AI, \
            cybersecurity, blockchain, cloud computing, emerging tech innovations, and pricing/costs \
            related to technology products and services.",
    model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0",
))

orchestrator.add_agent(tech_agent)

def serialize_agent_response(response: Any) -> Dict[str, Any]:

    text_response = ''
    if isinstance(response, AgentResponse) and response.streaming is False:
        # Handle regular response
        if isinstance(response.output, str):
            text_response = response.output
        elif isinstance(response.output, ConversationMessage):
                text_response = response.output.content[0].get('text')

    """Convert AgentResponse into a JSON-serializable dictionary."""
    return {
        "metadata": {
            "agent_id": response.metadata.agent_id,
            "agent_name": response.metadata.agent_name,
            "user_input": response.metadata.user_input,
            "session_id": response.metadata.session_id,
        },
        "output": text_response,
        "streaming": response.streaming,
    }

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    try:
        user_input = event.get('query')
        user_id = event.get('userId')
        session_id = event.get('sessionId')
        response = asyncio.run(orchestrator.route_request(user_input, user_id, session_id))

        # Serialize the AgentResponse to a JSON-compatible format
        serialized_response = serialize_agent_response(response)

        return {
            "statusCode": 200,
            "body": json.dumps(serialized_response)
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "Internal Server Error"})
        }
```

Customize the orchestrator configuration and agent setup according to your specific requirements.

## Deployment

Use your preferred method to deploy the Lambda function (e.g., AWS CDK, Terraform, Serverless Framework, AWS SAM, or manual deployment through AWS Console).

## IAM Permissions

Ensure your Lambda function's execution role has permissions to:
- Invoke Amazon Bedrock models
- Write to CloudWatch Logs

