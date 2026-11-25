
import uuid
import asyncio
import sys
from mcp import stdio_client, StdioServerParameters
from agent_squad.orchestrator import AgentSquad, AgentSquadConfig
from agent_squad.agents import (BedrockLLMAgent,
                        BedrockLLMAgentOptions,
                        AgentResponse,
                        AgentStreamResponse,
                        AgentOptions,
                        StrandsAgent)
from agent_squad.types import ConversationMessage
from strands.models import BedrockModel
from strands import tool
from strands_tools import calculator
from strands.tools.mcp import MCPClient

import logging

# Configure the root strands logger
logging.getLogger("strands").setLevel(logging.ERROR)


# For macOS/Linux:
stdio_mcp_client = MCPClient(lambda: stdio_client(
    StdioServerParameters(
        command="uvx",
        args=["awslabs.aws-documentation-mcp-server@latest"]
    )
))
cost_analysis_mcp_client = MCPClient(lambda: stdio_client(
    StdioServerParameters(
        command="uvx",
        args=["awslabs.cost-analysis-mcp-server@latest"]
    )
))

@tool
def get_user_location() -> str:
    """Get the user's location
    """

    # Implement user location lookup logic here
    return "Seattle, USA"

@tool
def weather(location: str) -> str:
    """Get weather information for a location

    Args:
        location: City or location name
    """

    # Implement weather lookup logic here
    return f"Weather for {location}: Sunny, 72°F"


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
                    print(chunk.text, end='', flush=True)
    else:
        if isinstance(response.output, ConversationMessage):
            print(response.output.content[0]['text'])
        elif isinstance(response.output, str):
            print(response.output)
        else:
            print(response.output)

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
    health_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
        name="Health Agent",
        streaming=False,
        description="Specializes in health and well being.",
        model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0",
    ))
    orchestrator.add_agent(health_agent)

    weather_agent = StrandsAgent(
        options=AgentOptions(
            name="Weather Agent",
            description="Specialized agent for giving weather condition from a city.",
        ),
        model=BedrockModel(
            temperature=0.3,
            top_p=0.8,
            streaming=True
        ),
        callback_handler=None,
        tools=[get_user_location, weather],
    )
    orchestrator.add_agent(weather_agent)

    math_agent = StrandsAgent(
        options=AgentOptions(
            name="Calculator Agent",
            description="Specializes in performing calculations.",
        ),
        model=BedrockModel(
            temperature=0.3,
            top_p=0.8,
            streaming=True
        ),
        callback_handler=None,
        tools=[calculator],
    )

    orchestrator.add_agent(math_agent)

    # Create AWS Documentation Agent with MCP client
    aws_documentation_agent = StrandsAgent(
        options=AgentOptions(
            name="AWS Documentation Agent",
            description="Specializes in answering questions about AWS services and cost calculation",
        ),
        model=BedrockModel(
            temperature=0.3,
            top_p=0.8,
            streaming=True
        ),
        callback_handler=None,
        mcp_clients=[stdio_mcp_client, cost_analysis_mcp_client],
    )

    orchestrator.add_agent(aws_documentation_agent)

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
