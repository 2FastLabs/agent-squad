
import uuid
from uuid import UUID
import asyncio
from typing import Optional, Any
import json
import sys

from tools import weather_tool

from agent_squad.orchestrator import AgentSquad, AgentSquadConfig
from agent_squad.agents import (BedrockLLMAgent,
                        BedrockLLMAgentOptions,
                        AgentResponse,
                        AgentStreamResponse,
                        AgentCallbacks)
from agent_squad.types import ConversationMessage, ParticipantRole
from agent_squad.utils import AgentToolCallbacks
from dotenv import load_dotenv

load_dotenv()
class LLMAgentCallbacks(AgentCallbacks):
    async def on_agent_start(
        self,
        agent_name,
        input: Any,
        messages: list[Any],
        run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> dict:

        return {"id":1234}

    async def on_agent_end(
        self,
        agent_name,
        response: Any,
        messages: list[Any],
        run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        pass

    async def on_llm_start(
        self,
        name: str,
        input: Any,
        run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        pass

    async def on_llm_end(
        self,
        name: str,
        output: Any,
        run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        pass


class CustomToolCallbacks(AgentToolCallbacks):
    async def on_tool_start(
        self,
        tool_name: str,
        input: Any,
        run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        print(tool_name)
        print(input)
        print(metadata)

    async def on_tool_end(
        self,
        tool_name: str,
        output: Any,
        run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        print(tool_name)
        print(output)
        print(metadata)


async def handle_request(_orchestrator: AgentSquad, _user_input:str, _user_id:str, _session_id:str):
    stream_response = True
    response:AgentResponse = await _orchestrator.route_request(_user_input, _user_id, _session_id, {}, stream_response)

    # Print metadata
    print("\nMetadata:")
    print(f"Selected Agent: {response.metadata.agent_name}")
    if stream_response and response.streaming:
        async for chunk in response.output:
            if isinstance(chunk, AgentStreamResponse):
                if response.streaming:
                    if (chunk.thinking):
                        print(f"\033[34m{chunk.thinking}\033[0m", end='', flush=True)
                    elif (chunk.text):
                        print(chunk.text, end='', flush=True)
    else:
        if isinstance(response.output, ConversationMessage):
            print(response.output.content[0]['text'])

            # Safely extract thinking content from response
            thinking_content = None
            for content_item in response.output.content:
                if isinstance(content_item, dict) and 'reasoningContent' in content_item:
                    thinking_content = content_item['reasoningContent']
                    break

            if thinking_content:
                print(f"\nThinking: {thinking_content}")
        elif isinstance(response.output, str):
            print(response.output)
        else:
            print(response.output)

def custom_input_payload_encoder(input_text: str,
                                 chat_history: list[Any],
                                 user_id: str,
                                 session_id: str,
                                 additional_params: Optional[dict[str, str]] = None) -> str:
    return json.dumps({
        'hello':'world'
    })

def custom_output_payload_decoder(response: dict[str, Any]) -> Any:
    decoded_response = json.loads(
        json.loads(
            response['Payload'].read().decode('utf-8')
        )['body'])['response']
    return ConversationMessage(
            role=ParticipantRole.ASSISTANT.value,
            content=[{'text': decoded_response}]
        )

if __name__ == "__main__":

    # Initialize the orchestrator with some options
    orchestrator = AgentSquad(options=AgentSquadConfig(
        LOG_AGENT_CHAT=True,
        LOG_CLASSIFIER_CHAT=True,
        LOG_CLASSIFIER_RAW_OUTPUT=True,
        LOG_CLASSIFIER_OUTPUT=True,
        LOG_EXECUTION_TIMES=True,
        MAX_RETRIES=3,
        USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED=True,
        MAX_MESSAGE_PAIRS_PER_AGENT=10,
    ))

    # Add some agents
    tech_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
        name="Tech Agent",
        streaming=True,
        description="Specializes in technology areas including software development, hardware, AI, \
            cybersecurity, blockchain, cloud computing, emerging tech innovations, and pricing/costs \
            related to technology products and services.",
        model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0",
        # callbacks=LLMAgentCallbacks()
    ))
    orchestrator.add_agent(tech_agent)


    # Add some agents
    tech_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
        name="Health Agent",
        streaming=True,
        inference_config={
            "maxTokens": 4096,
            "temperature":1.0
        },
        description="Specializes in health and well being.",
        model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        additional_model_request_fields={
            "thinking": {
                "type": "enabled",
                "budget_tokens": 4000
            }
        }
    ))
    orchestrator.add_agent(tech_agent)

    # Add a Anthropic weather agent with a tool in anthropic's tool format
    # weather_agent = AnthropicAgent(AnthropicAgentOptions(
    #     api_key=os.getenv('ANTHROPIC_API_KEY', None),
    #     name="Weather Agent",
    #     streaming=True,
    #     model_id="claude-3-7-sonnet-20250219",
    #     description="Specialized agent for giving weather condition from a city.",
    #     tool_config={
    #         'tool': [tool.to_claude_format() for tool in weather_tool.weather_tools.tools],
    #         'toolMaxRecursions': 5,
    #         'useToolHandler': weather_tool.anthropic_weather_tool_handler
    #     },
    #     inference_config={
    #         "maxTokens": 4096,
    #         "temperature":1.0,
    #         "topP":1.0
    #     }
    #     ,
    #     additional_model_request_fields = {
    #         "thinking": {
    #             "type": "enabled",
    #             "budget_tokens": 4000
    #         }
    #     },
    #     callbacks=LLMAgentCallbacks()
    # ))

    # Add an Anthropic weather agent with Tools class
    # weather_agent = AnthropicAgent(AnthropicAgentOptions(
    #     api_key='api-key',
    #     name="Weather Agent",
    #     streaming=True,
    #     description="Specialized agent for giving weather condition from a city.",
    #     tool_config={
    #         'tool': weather_tool.weather_tools,
    #         'toolMaxRecursions': 5,
    #     },
    #     callbacks=LLMAgentCallbacks()
    # ))

    # Add a Bedrock weather agent with Tools class
    # weather_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
    #     name="Weather Agent",
    #     streaming=False,
    #     description="Specialized agent for giving weather condition from a city.",
    #     tool_config={
    #         'tool': weather_tool.weather_tools,
    #         'toolMaxRecursions': 5,
    #     },
    #     callbacks=LLMAgentCallbacks(),
    # ))

    # Add a Bedrock weather agent with custom handler and bedrock's tool format
    weather_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
        name="Weather Agent",
        streaming=True,
        model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        description="Specialized agent for giving weather condition from a city.",
        tool_config={
            'tool': [tool.to_bedrock_format() for tool in weather_tool.weather_tools.tools],
            'toolMaxRecursions': 5,
            'useToolHandler': weather_tool.bedrock_weather_tool_handler
        },
        additional_model_request_fields={
            "thinking": {
                "type": "enabled",
                "budget_tokens": 4000
            }
        },
        inference_config={
            "maxTokens": 4096,
            "temperature":1.0
        },
    ))


    weather_agent.set_system_prompt(weather_tool.weather_tool_prompt)
    orchestrator.add_agent(weather_agent)

    USER_ID = "user123"
    SESSION_ID = str(uuid.uuid4())

    print("Welcome to the interactive Multi-Agent system. Type 'quit' to exit.")

    while True:
        # Get user input
        user_input = input("\nYou: ").strip()

        if user_input.lower() == 'quit':
            print("Exiting the program. Goodbye!")
            sys.exit()

        if user_input != '':
            # Run the async function
            asyncio.run(handle_request(orchestrator, user_input, USER_ID, SESSION_ID))
