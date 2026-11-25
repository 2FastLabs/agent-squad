
import uuid
import asyncio
from typing import Optional, Any
import json
import sys
import os
from tools import weather_tool
from agent_squad.orchestrator import AgentSquad, AgentSquadConfig
from agent_squad.agents import (BedrockLLMAgent,
                        BedrockLLMAgentOptions,
                        AgentStreamResponse,
                        AgentCallbacks)
from agent_squad.types import ConversationMessage, ParticipantRole
from agent_squad.utils import AgentTools, AgentToolCallbacks, AgentTool
from agent_squad.classifiers import BedrockClassifier, BedrockClassifierOptions, ClassifierCallbacks, ClassifierResult
from langfuse.decorators import observe, langfuse_context
from langfuse import Langfuse
from uuid import UUID
from datetime import datetime, timezone
from dotenv import load_dotenv
import logging

load_dotenv()  # take environment variables

langfuse = Langfuse()

class BedrockClassifierCallbacks(ClassifierCallbacks):

    async def on_classifier_start(
        self,
        name,
        input: Any,
        run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            inputs = []
            inputs.append({'role':'system', 'content':kwargs.get('system')})
            inputs.extend([{'role':'user', 'content':input}])
            langfuse_context.update_current_observation(
                name=name,
                start_time=datetime.now(timezone.utc),
                input=inputs,
                model=kwargs.get('modelId'),
                model_parameters=kwargs.get('inferenceConfig'),
                tags=tags,
                metadata=metadata
            )
        except Exception as e:
            logging.error(e)
            pass

    async def on_classifier_stop(
        self,
        name,
        output: ClassifierResult,
        run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            langfuse_context.update_current_observation(
                output={'role':'assistant', 'content':{
                            'selected_agent' : output.selected_agent.name if output.selected_agent is not None else 'No agent selected',
                            'confidence' : output.confidence,
                        }
                },
                end_time=datetime.now(timezone.utc),
                name=name,
                tags=tags,
                metadata=metadata,
                usage={
                    'input':kwargs.get('usage',{}).get('inputTokens'),
                    "output": kwargs.get('usage', {}).get('outputTokens'),
                    "total": kwargs.get('usage', {}).get('totalTokens')
                },
            )
        except Exception as e:
            logging.error(e)
            pass


class LLMAgentCallbacks(AgentCallbacks):

    async def on_agent_start(
        self,
        agent_name,
        payload_input: Any,
        messages: list[Any],
        run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            langfuse_context.update_current_observation(
                input=payload_input,
                start_time=datetime.now(timezone.utc),
                name=agent_name,
                tags=tags,
                metadata=metadata
            )
        except Exception as e:
            logging.error(e)
            pass

    async def on_agent_end(
        self,
        agent_name,
        response: Any,
        messages:list[Any],
        run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            langfuse_context.update_current_observation(
                end_time=datetime.now(timezone.utc),
                name=agent_name,
                user_id=kwargs.get('user_id'),
                session_id=kwargs.get('session_id'),
                output=response,
                tags=tags,
                metadata=metadata
            )
        except Exception as e:
            logging.error(e)
            pass

    async def on_llm_start(
        self,
        name:str,
        payload_input: Any,
        run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        logging.debug('on_llm_start')


    @observe(as_type='generation', capture_input=False)
    async def on_llm_end(
        self,
        name:str,
        output: Any,
        run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            msgs = []
            msgs.append({'role':'system', 'content': kwargs.get('input').get('system')})
            msgs.extend(kwargs.get('input').get('messages'))
            langfuse_context.update_current_observation(
                name=name,
                input=msgs,
                output=output,
                model=kwargs.get('input').get('modelId'),
                model_parameters=kwargs.get('inferenceConfig'),
                usage={
                    'input':kwargs.get('usage',{}).get('inputTokens'),
                    "output": kwargs.get('usage', {}).get('outputTokens'),
                    "total": kwargs.get('usage', {}).get('totalTokens')
                },
                tags=tags,
                metadata=metadata
            )
        except Exception as e:
            logging.error(e)
            pass


class ToolsCallbacks(AgentToolCallbacks):

    @observe(as_type='span', name='on_tool_start', capture_input=False)
    async  def on_tool_start(
        self,
        tool_name,
        payload_input: Any,
        run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        langfuse_context.update_current_observation(
            name=tool_name,
            input=input
        )

    @observe(as_type='span', name='on_tool_end', capture_input=False)
    async def on_tool_end(
        self,
        tool_name,
        payload_input: Any,
        output: dict,
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        langfuse_context.update_current_observation(
            input=payload_input,
            name=tool_name,
            output=output
        )

@observe(as_type='generation', name='classify_request')
async def classify_request(_orchestrator: AgentSquad, _user_input:str, _user_id:str, _session_id:str) -> ClassifierResult:
    result:ClassifierResult = await _orchestrator.classify_request(_user_input, _user_id, _session_id)
    return result

@observe(as_type='generation', name='agent_process_request')
async def agent_process_request(_orchestrator: AgentSquad, user_input: str,
                               user_id: str,
                               session_id: str,
                               classifier_result: ClassifierResult,
                               additional_params: dict[str, str],
                               stream_response):
    response = await _orchestrator.agent_process_request(user_input, user_id, session_id, classifier_result, additional_params, stream_response)
    # Print metadata
    print("\nMetadata:")
    print(f"Selected Agent: {response.metadata.agent_name}")
    final_response = ''
    if stream_response and response.streaming:
        async for chunk in response.output:
            if isinstance(chunk, AgentStreamResponse):
                if response.streaming:
                    final_response += chunk.text
                    print(chunk.text, end='', flush=True)
    else:
        if isinstance(response.output, ConversationMessage):
            print(response.output.content[0]['text'])
            final_response = response.output.content[0]['text']
        elif isinstance(response.output, str):
            print(response.output)
            final_response = response.output
        else:
            print(response.output)
            final_response = response.output

    return final_response


@observe(as_type='generation', name='handle_request')
async def handle_request(_orchestrator: AgentSquad, _user_input:str, _user_id:str, _session_id:str) -> str:

    stream_response = True
    classification_result:ClassifierResult = await classify_request(_orchestrator, _user_input, _user_id, _session_id)
    if classification_result.selected_agent is None:
        return "No agent selected. Please try again."
    return await agent_process_request(_orchestrator, _user_input, _user_id, _session_id, classification_result,{}, stream_response)



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

weather_tools:AgentTools = AgentTools(tools=[AgentTool(name="Weather_Tool",
                            description="Get the current weather for a given location, based on its WGS84 coordinates.",
                            func=weather_tool.fetch_weather_data
                            )],
                            callbacks=ToolsCallbacks())

@observe(as_type="generation", name="python-demo")
def run_main():

    classifier = BedrockClassifier(BedrockClassifierOptions(
        model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0",
        callbacks=BedrockClassifierCallbacks()
    ))
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
    classifier=classifier)

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

    # Add Health agents
    health_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
        name="Health Agent",
        streaming=False,
        description="Specializes in health and well being.",
        model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0",
        callbacks=LLMAgentCallbacks(),
    ))
    orchestrator.add_agent(health_agent)

    # Add a Bedrock weather agent with custom handler and bedrock's tool format
    weather_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
        name="Weather Agent",
        streaming=False,
        description="Specialized agent for giving weather condition from a city.",
        tool_config={
            'tool': weather_tools,
            'toolMaxRecursions': 5,
        },
        callbacks=LLMAgentCallbacks()
    ))

    weather_agent.set_system_prompt(weather_tool.weather_tool_prompt)
    orchestrator.add_agent(weather_agent)

    USER_ID = "user123"
    SESSION_ID = str(uuid.uuid4())

    user_inputs = []
    final_responses = []

    print("Welcome to the interactive Agent-Squad system. Type 'quit' to exit.")

    while True:
        # Get user input
        user_input = input("\nYou: ").strip()

        if user_input.lower() == 'quit':
            print("Exiting the program. Goodbye!")
            sys.exit()

        # Run the async function
        user_inputs.append(user_input)
        langfuse_context.update_current_trace(
            input=user_inputs,
            user_id=USER_ID,
            session_id=SESSION_ID
        )
        response = asyncio.run(handle_request(orchestrator, user_input, USER_ID, SESSION_ID))
        final_responses.append(response)
        langfuse_context.update_current_trace(
            output=final_responses
        )

        langfuse.flush()

if __name__ == "__main__":

    run_main()
