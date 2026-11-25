
import uuid
import asyncio
from typing import Optional, List, Dict, Any
import json
import sys

from tools import weather_tool

from agent_squad.orchestrator import AgentSquad, AgentSquadConfig
from agent_squad.agents import (BedrockLLMAgent,
                        BedrockLLMAgentOptions,
                        AgentResponse,
                        AnthropicAgent, AnthropicAgentOptions,
                        AgentCallbacks)
from agent_squad.types import ConversationMessage, ParticipantRole
from agent_squad.classifiers import BedrockClassifier, BedrockClassifierOptions
from agent_squad.utils import AgentTools

class LLMAgentCallbacks(AgentCallbacks):
    def on_llm_new_token(self, token: str) -> None:
        # handle response streaming here
        print(token, end='', flush=True)


async def handle_request(_orchestrator: AgentSquad, _user_input:str, _user_id:str, _session_id:str):
    response:AgentResponse = await _orchestrator.route_request(_user_input, _user_id, _session_id)

    # Print metadata
    print("\nMetadata:")
    print(f"Selected Agent: {response.metadata.agent_name}")
    if isinstance(response, AgentResponse) and response.streaming is False:
        # Handle regular response
        if isinstance(response.output, str):
            print(response.output)
        elif isinstance(response.output, ConversationMessage):
                print(response.output.content[0].get('text'))

def custom_input_payload_encoder(input_text: str,
                                 chat_history: List[Any],
                                 user_id: str,
                                 session_id: str,
                                 additional_params: Optional[Dict[str, str]] = None) -> str:
    return json.dumps({
        'hello':'world'
    })

def custom_output_payload_decoder(response: Dict[str, Any]) -> Any:
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
        callbacks=LLMAgentCallbacks()
    ))
    orchestrator.add_agent(tech_agent)

    # Add a Anthropic weather agent with a tool in anthropic's tool format
    # weather_agent = AnthropicAgent(AnthropicAgentOptions(
    #     api_key='api-key',
    #     name="Weather Agent",
    #     streaming=False,
    #     description="Specialized agent for giving weather condition from a city.",
    #     tool_config={
    #         'tool': [tool.to_claude_format() for tool in weather_tool.weather_tools.tools],
    #         'toolMaxRecursions': 5,
    #         'useToolHandler': weather_tool.anthropic_weather_tool_handler
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
        streaming=False,
        description="Specialized agent for giving weather condition from a city.",
        tool_config={
            'tool': [tool.to_bedrock_format() for tool in weather_tool.weather_tools.tools],
            'toolMaxRecursions': 5,
            'useToolHandler': weather_tool.bedrock_weather_tool_handler
        }
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

        # Run the async function
        asyncio.run(handle_request(orchestrator, user_input, USER_ID, SESSION_ID))
