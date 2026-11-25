from typing import AsyncIterator, Any, Optional
import sys
import asyncio
import uuid
from uuid import UUID
import os
from datetime import datetime, timezone
from agent_squad.utils import Logger
from agent_squad.orchestrator import AgentSquad, AgentSquadConfig
from agent_squad.agents import (
    BedrockLLMAgent, BedrockLLMAgentOptions,
    AgentResponse,
    LexBotAgent, LexBotAgentOptions,
    AmazonBedrockAgent, AmazonBedrockAgentOptions,
    SupervisorAgent, SupervisorAgentOptions,
    AgentStreamResponse,
    AgentCallbacks,
)
from agent_squad.classifiers import ClassifierResult
from agent_squad.types import ConversationMessage
from agent_squad.storage import DynamoDbChatStorage
from agent_squad.utils import AgentTools, AgentTool, AgentToolCallbacks

try:
    from agent_squad.agents import AnthropicAgent,  AnthropicAgentOptions
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False


from weather_tool import weather_tool_description, weather_tool_handler, weather_tool_prompt
from dotenv import load_dotenv

load_dotenv()

class LLMAgentCallbacks(AgentCallbacks):
    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        # handle response streaming here
        if 'thinking' in kwargs:
            print(f"\033[31m{token}\033[0m", end='', flush=True)

class SupervisorToolsCallbacks (AgentToolCallbacks):
    async def on_tool_start(
        self,
        tool_name,
        input: Any,
        run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        print(f"Tool {tool_name} started with input {input}")

    async def on_tool_end(
        self,
        tool_name,
        output: Any,
        run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        print(f"Tool {tool_name} ended with output {output}")

    async def on_tool_error(
        self,
        tool_name,
        error: Exception,
        run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        print(f"Tool {tool_name} error: {error}")

tech_agent = BedrockLLMAgent(
    options=BedrockLLMAgentOptions(
        name="TechAgent",
        description="You are a tech agent. You are responsible for answering questions about tech. You are only allowed to answer questions about tech. You are not allowed to answer questions about anything else.",
        model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0",
    )
)

sales_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
    name="SalesAgent",
    description="You are a sales agent. You are responsible for answering questions about sales. You are only allowed to answer questions about sales. You are not allowed to answer questions about anything else.",
    model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0",
))

claim_agent = AmazonBedrockAgent(AmazonBedrockAgentOptions(
    name="Claim Agent",
    description="Specializes in handling claims and disputes.",
    agent_id=os.getenv('CLAIM_AGENT_ID',None),
    agent_alias_id=os.getenv('CLAIM_AGENT_ALIAS_ID',None)
))

weather_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
        name="WeatherAgent",
        streaming=True,
        model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        description="Specialized agent for giving weather forecast condition from a city.",
        tool_config={
            'tool':weather_tool_description,
            'toolMaxRecursions': 5,
            'useToolHandler': weather_tool_handler
        },
        inference_config={
            "temperature":1.0,
            "maxTokens": 4096
        },
        additional_model_request_fields={
            "thinking": {
                "type": "enabled",
                "budget_tokens": 4000
            }
        },
        callbacks=LLMAgentCallbacks()
    ))
weather_agent.set_system_prompt(weather_tool_prompt)



health_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
    name="HealthAgent",
    description="You are a health agent. You are responsible for answering questions about health. You are only allowed to answer questions about health. You are not allowed to answer questions about anything else.",
    model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0",
))

travel_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
    name="TravelAgent",
    description="You are a travel assistant agent. You are responsible for answering questions about travel, activities, sight seesing about a city and surrounding",
    model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0",
))

airlines_agent = LexBotAgent(LexBotAgentOptions(name='AirlinesBot',
                                              description='Helps users book their flight. This bot works with US metric time and date.',
                                              locale_id='en_US',
                                              bot_id=os.getenv('AIRLINES_BOT_ID', None),
                                              bot_alias_id=os.getenv('AIRLINES_BOT_ALIAS_ID', None)))

if _ANTHROPIC_AVAILABLE:
    lead_agent = AnthropicAgent(AnthropicAgentOptions(
        api_key=os.getenv('ANTHROPIC_API_KEY', None),
        name="SupervisorAgent",
        model_id="claude-3-7-sonnet-20250219",
        description="You are a supervisor agent. You are responsible for managing the flow of the conversation. You are only allowed to manage the flow of the conversation. You are not allowed to answer questions about anything else.",
        streaming=True,
        inference_config={
            "maxTokens": 4096,
            "temperature":1.0,
            "topP":1.0
        }
        ,
        additional_model_request_fields = {
            "thinking": {
                "type": "enabled",
                "budget_tokens": 4000
            }
        }
    ))
else:
    lead_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
        name="SupervisorAgent",
        model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        streaming=True,
        inference_config={
            "temperature":1.0,
            "maxTokens": 4096
        },
        additional_model_request_fields={
            "thinking": {
                "type": "enabled",
                "budget_tokens": 4000
            }
        },
        description="You are a supervisor agent. You are responsible for managing the flow of the conversation. You are only allowed to manage the flow of the conversation. You are not allowed to answer questions about anything else.",
    ))

async def get_current_date():
        """
        Get the current date in US format.
        """
        Logger.info('Using Tool : get_current_date')
        return datetime.now(timezone.utc).strftime('%m/%d/%Y')  # from datetime import datetime, timezone


supervisor = SupervisorAgent(
    SupervisorAgentOptions(
        name="SupervisorAgent",
        description="My Supervisor agent description",
        lead_agent=lead_agent,
        team=[airlines_agent, travel_agent, tech_agent, sales_agent, health_agent, claim_agent, weather_agent],
        storage=DynamoDbChatStorage(
            table_name=os.getenv('DYNAMODB_CHAT_HISTORY_TABLE_NAME', None),
            region='us-east-1'
        ),
        trace=True,
        extra_tools=AgentTools(tools=[AgentTool(
            name="get_current_date",
            func=get_current_date,
        )], callbacks=SupervisorToolsCallbacks())
    ))

async def handle_request(_orchestrator: AgentSquad, _user_input:str, _user_id:str, _session_id:str):
    classifier_result=ClassifierResult(selected_agent=supervisor, confidence=1.0)

    response:AgentResponse = await _orchestrator.agent_process_request(_user_input, _user_id, _session_id, classifier_result, {}, True)

    # Print metadata
    print("\nMetadata:")
    print(f"Selected Agent: {response.metadata.agent_name}")
    if isinstance(response, AgentResponse) and response.streaming is False:
        # Handle regular response
        if isinstance(response.output, str):
            print(f"\033[34m{response.output}\033[0m")
        elif isinstance(response.output, ConversationMessage):
                print(f"\033[34m{response.output.content[0].get('text')}\033[0m")
    if response.streaming:
         if isinstance(response.output, AsyncIterator):
            async for chunk in response.output:
                if isinstance(chunk, AgentStreamResponse):
                    print(f"\033[34m{chunk.text}\033[0m", end='', flush=True)
                    if chunk.thinking:
                        print(f"\033[31m{chunk.thinking}\033[0m", end='', flush=True)


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
    ),
    storage=DynamoDbChatStorage(
        table_name=os.getenv('DYNAMODB_CHAT_HISTORY_TABLE_NAME', None),
        region='us-east-1')
    )

    USER_ID = str(uuid.uuid4())
    SESSION_ID = str(uuid.uuid4())

    print(f"""Welcome to the interactive Multi-Agent system.\n
I'm here to assist you with your questions.
Here is the list of available agents:
- TechAgent: Anything related to technology
- SalesAgent: Weather you want to sell a boat, a car or house, I can give you advice
- HealthAgent: You can ask me about your health, diet, exercise, etc.
- AirlinesBot: I can help you book a flight
- WeatherAgent: I can tell you the weather in a given city
- TravelAgent: I can help you plan your next trip.
- ClaimAgent: Anything regarding the current claim you have or general information about them.
""")

    while True:
        # Get user input
        user_input = input("\nYou: ").strip()

        if user_input.lower() == 'quit':
            print("Exiting the program. Goodbye!")
            sys.exit()

        # Run the async function
        if user_input is not None and user_input != '':
            asyncio.run(handle_request(orchestrator, user_input, USER_ID, SESSION_ID))