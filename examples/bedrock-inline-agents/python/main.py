import asyncio
import uuid
import sys
from agent_squad.agents import BedrockInlineAgent, BedrockInlineAgentOptions
import boto3

action_groups_list = [
    {
        'actionGroupName': 'CodeInterpreterAction',
        'parentActionGroupSignature': 'AMAZON.CodeInterpreter',
        'description':'Use this to write and execute python code to answer questions and other tasks.'
    },
    {
        "actionGroupExecutor": {
            "lambda": "arn:aws:lambda:region:0123456789012:function:my-function-name"
        },
        "actionGroupName": "MyActionGroupName",
        "apiSchema": {
            "s3": {
                "s3BucketName": "bucket-name",
                "s3ObjectKey": "openapi-schema.json"
            }
        },
        "description": "My action group for doing a specific task"
    }
]

knowledge_bases = [
    {
        "knowledgeBaseId": "knowledge-base-id-01",
        "description": 'This is my knowledge base for documents 01',
    },
    {
        "knowledgeBaseId": "knowledge-base-id-02",
        "description": 'This is my knowledge base for documents 02',
    },
    {
        "knowledgeBaseId": "knowledge-base-id-0",
        "description": 'This is my knowledge base for documents 03',
    }
]

bedrock_inline_agent = BedrockInlineAgent(BedrockInlineAgentOptions(
    name="Inline Agent Creator for Agents for Amazon Bedrock",
    region='us-east-1',
    model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0",
    description="Specalized in creating Agent to solve customer request dynamically. You are provided with a list of Action groups and Knowledge bases which can help you in answering customer request",
    action_groups_list=action_groups_list,
    bedrock_agent_client=boto3.client('bedrock-agent-runtime', region_name='us-east-1'),
    client=boto3.client('bedrock-runtime', region_name='us-west-2'),
    knowledge_bases=knowledge_bases,
    enableTrace=True
))

async def run_inline_agent(user_input, user_id, session_id):
    response = await bedrock_inline_agent.process_request(user_input, user_id, session_id, [], None)
    return response

if __name__ == "__main__":

    session_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    print("Welcome to the interactive Multi-Agent system. Type 'quit' to exit.")

    while True:
        # Get user input
        user_input = input("\nYou: ").strip()

        if user_input.lower() == 'quit':
            print("Exiting the program. Goodbye!")
            sys.exit()

        # Run the async function
        response = asyncio.run(run_inline_agent(user_input=user_input, user_id=user_id, session_id=session_id))
        print(response.content[0].get('text','No response'))