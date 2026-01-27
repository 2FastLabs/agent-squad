from typing import Any, AsyncIterable
from dataclasses import dataclass, fields, asdict, replace
import time
from agent_squad.utils.logger import Logger
from agent_squad.types import (ConversationMessage,
                                            ParticipantRole,
                                            AgentSquadConfig,
                                            TimestampedMessage)
from agent_squad.classifiers import Classifier,ClassifierResult
from agent_squad.agents import (Agent,
                                             AgentStreamResponse,
                                             AgentResponse,
                                             AgentProcessingResult)
from agent_squad.storage import ChatStorage
from agent_squad.storage import InMemoryChatStorage
try:
    from agent_squad.classifiers import BedrockClassifier, BedrockClassifierOptions
    _BEDROCK_AVAILABLE = True
except ImportError:
    _BEDROCK_AVAILABLE = False

@dataclass
class AgentSquad:
    def __init__(self,
                 options: AgentSquadConfig | None = None,
                 storage: ChatStorage | None = None,
                 classifier: Classifier  | None = None,
                 logger: Logger | None = None,
                 default_agent: Agent | None = None):

        DEFAULT_CONFIG=AgentSquadConfig()

        if options is None:
            options = {}
        if isinstance(options, dict):
            # Filter out keys that are not part of AgentSquadConfig fields
            valid_keys = {f.name for f in fields(AgentSquadConfig)}
            options = {k: v for k, v in options.items() if k in valid_keys}
            options = AgentSquadConfig(**options)
        elif not isinstance(options, AgentSquadConfig):
            raise ValueError("options must be a dictionary or an AgentSquadConfig instance")


        self.config = replace(DEFAULT_CONFIG, **asdict(options))
        self.storage = storage


        self.logger = Logger(self.config, logger)
        self.agents: dict[str, Agent] = {}
        self.storage = storage or InMemoryChatStorage()

        if classifier:
            self.classifier = classifier
        elif _BEDROCK_AVAILABLE:
            self.classifier = BedrockClassifier(options=BedrockClassifierOptions())
        else:
            raise ValueError("No classifier provided and BedrockClassifier is not available. Please provide a classifier.")

        self.execution_times: dict[str, float] = {}
        self.default_agent: Agent = default_agent


    def add_agent(self, agent: Agent):
        if agent.id in self.agents:
            raise ValueError(f"An agent with ID '{agent.id}' already exists.")
        self.agents[agent.id] = agent
        self.classifier.set_agents(self.agents)

    def get_default_agent(self) -> Agent:
        return self.default_agent

    def set_default_agent(self, agent: Agent):
        self.default_agent = agent

    def get_all_agents(self) -> dict[str, dict[str, str]]:
        return {key: {
            "name": agent.name,
            "description": agent.description
        } for key, agent in self.agents.items()}

    async def dispatch_to_agent(self, params: dict[str, Any]
                                ) -> tuple[ConversationMessage | AsyncIterable[Any], Agent]:
        user_input = params['user_input']
        user_id = params['user_id']
        session_id = params['session_id']
        classifier_result:ClassifierResult = params['classifier_result']
        additional_params = params.get('additional_params', {})

        if not classifier_result.selected_agent:
            return ConversationMessage(
                role=ParticipantRole.ASSISTANT.value,
                content=[{'text': "I'm sorry, but I need more information to understand your request. Could you please be more specific?"}]
            ), None

        selected_agent = classifier_result.selected_agent
        agent_chat_history = await self.storage.fetch_chat(user_id, session_id, selected_agent.id)

        self.logger.print_chat_history(agent_chat_history, selected_agent.id)

        response = await self.measure_execution_time(
            f"Agent {selected_agent.name} | Processing request",
            lambda: selected_agent.process_request(user_input,
                                                   user_id,
                                                   session_id,
                                                   agent_chat_history,
                                                   additional_params)
        )

        return response, selected_agent

    async def classify_request(self,
                             user_input: str,
                             user_id: str,
                             session_id: str) -> ClassifierResult:
        """Classify user request with conversation history."""
        try:
            chat_history = await self.storage.fetch_all_chats(user_id, session_id) or []
            classifier_result = await self.measure_execution_time(
                "Classifying user intent",
                lambda: self.classifier.classify(user_input, chat_history)
            )

            if self.config.LOG_CLASSIFIER_OUTPUT:
                self.print_intent(user_input, classifier_result)

            if not classifier_result.selected_agent:
                if self.config.USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED and self.default_agent:
                    classifier_result = self.get_fallback_result()
                    self.logger.info("Using default agent as no agent was selected")

            return classifier_result

        except Exception as error:
            self.logger.error(f"Error during intent classification: {str(error)}")
            raise error

    async def agent_process_request(self,
                               user_input: str,
                               user_id: str,
                               session_id: str,
                               classifier_result: ClassifierResult,
                               additional_params: dict[str, str] | None = None,
                               stream_response: bool | None = False # wether to stream back the response from the agent
    ) -> AgentResponse:
        """Process agent response and handle chat storage."""
        try:
            if classifier_result.selected_agent:
                agent_response, dispatched_agent = await self.dispatch_to_agent({
                    "user_input": user_input,
                    "user_id": user_id,
                    "session_id": session_id,
                    "classifier_result": classifier_result,
                    "additional_params": additional_params
                })

                metadata = self.create_metadata(classifier_result,
                                            user_input,
                                            user_id,
                                            session_id,
                                            additional_params)

                await self.save_message(
                    ConversationMessage(
                        role=ParticipantRole.USER.value,
                        content=[{'text': user_input}]
                    ),
                    user_id,
                    session_id,
                    classifier_result.selected_agent
                )

                # Save intermediate tool conversation messages (toolUse/toolResult pairs)
                if dispatched_agent and hasattr(dispatched_agent, 'tool_conversation') and dispatched_agent.tool_conversation:
                    await self.save_messages(
                        dispatched_agent.tool_conversation,
                        user_id,
                        session_id,
                        classifier_result.selected_agent
                    )

                final_response = None
                if classifier_result.selected_agent.is_streaming_enabled():
                    if stream_response:
                        if isinstance(agent_response, AsyncIterable):
                            # Create an async generator function to handle the streaming
                            selected_agent_ref = classifier_result.selected_agent
                            dispatched_agent_ref = dispatched_agent
                            async def process_stream():
                                full_message = None
                                async for chunk in agent_response:
                                    if isinstance(chunk, AgentStreamResponse):
                                        if chunk.final_message:
                                            full_message = chunk.final_message
                                        yield chunk
                                    else:
                                        Logger.error("Invalid response type from agent. Expected AgentStreamResponse")
                                        pass

                                # Save intermediate tool messages collected during streaming
                                if dispatched_agent_ref and hasattr(dispatched_agent_ref, 'tool_conversation') and dispatched_agent_ref.tool_conversation:
                                    await self.save_messages(
                                        dispatched_agent_ref.tool_conversation,
                                        user_id,
                                        session_id,
                                        selected_agent_ref
                                    )

                                if full_message:
                                    await self.save_message(full_message,
                                                        user_id,
                                                        session_id,
                                                        selected_agent_ref)


                            final_response = process_stream()
                    else:
                        selected_agent_ref = classifier_result.selected_agent
                        dispatched_agent_ref = dispatched_agent
                        async def process_stream() -> ConversationMessage:
                            full_message = None
                            async for chunk in agent_response:
                                if isinstance(chunk, AgentStreamResponse):
                                    if chunk.final_message:
                                        full_message = chunk.final_message
                                else:
                                    Logger.error("Invalid response type from agent. Expected AgentStreamResponse")
                                    pass

                            # Save intermediate tool messages collected during streaming
                            if dispatched_agent_ref and hasattr(dispatched_agent_ref, 'tool_conversation') and dispatched_agent_ref.tool_conversation:
                                await self.save_messages(
                                    dispatched_agent_ref.tool_conversation,
                                    user_id,
                                    session_id,
                                    selected_agent_ref
                                )

                            if full_message:
                                await self.save_message(full_message,
                                                user_id,
                                                session_id,
                                                selected_agent_ref)
                            return full_message
                        final_response = await process_stream()


                else:  # Non-streaming response
                    final_response = agent_response
                    await self.save_message(final_response,
                                            user_id,
                                            session_id,
                                            classifier_result.selected_agent)

                return AgentResponse(
                    metadata=metadata,
                    output=final_response,
                    streaming=classifier_result.selected_agent.is_streaming_enabled()
                )
            else:
                # classified didn't find a proper agent
                error =  self.config.NO_SELECTED_AGENT_MESSAGE or "I'm sorry, but I need more information to understand your request. Could you please be more specific?"
                return AgentResponse(
                    metadata=self.create_metadata(None, user_input, user_id, session_id, additional_params),
                    output=ConversationMessage(
                        role=ParticipantRole.ASSISTANT.value,
                        content=[{'text': error}]
                    ),
                    streaming=False
                )

        except Exception as error:
            self.logger.error(f"Error during agent processing: {str(error)}")
            raise error

    async def route_request(self,
                            user_input: str,
                            user_id: str,
                            session_id: str,
                            additional_params: dict[str, str] | None = None,
                            stream_response: bool | None = False
    ) -> AgentResponse:
        """Route user request to appropriate agent."""
        self.execution_times.clear()

        try:
            classifier_result = await self.classify_request(user_input, user_id, session_id)

            if not classifier_result.selected_agent:
                return AgentResponse(
                    metadata=self.create_metadata(classifier_result, user_input, user_id, session_id, additional_params),
                    output=ConversationMessage(
                        role=ParticipantRole.ASSISTANT.value,
                        content=[{'text': self.config.NO_SELECTED_AGENT_MESSAGE}]
                    ),
                    streaming=False
                )

            return await self.agent_process_request(
                user_input,
                user_id,
                session_id,
                classifier_result,
                additional_params,
                stream_response
            )

        except Exception as error:
            error_message = self.config.GENERAL_ROUTING_ERROR_MSG_MESSAGE or str(error)
            return AgentResponse(
                metadata=self.create_metadata(None, user_input, user_id, session_id, additional_params),
                output=ConversationMessage(
                    role=ParticipantRole.ASSISTANT.value,
                    content=[{'text': error_message}]
                ),
                streaming=False
            )

        finally:
            self.logger.print_execution_times(self.execution_times)


    def print_intent(self, user_input: str, intent_classifier_result: ClassifierResult) -> None:
        """Print the classified intent."""
        self.logger.log_header('Classified Intent')
        self.logger.info(f"> Text: {user_input}")
        selected_agent_string = intent_classifier_result.selected_agent.name \
                                                if intent_classifier_result.selected_agent \
                                                    else 'No agent selected'
        self.logger.info(f"> Selected Agent: {selected_agent_string}")
        self.logger.info(f"> Confidence: {intent_classifier_result.confidence:.2f}")
        self.logger.info('')

    async def measure_execution_time(self, timer_name: str, fn):
        if not self.config.LOG_EXECUTION_TIMES:
            return await fn()

        start_time = time.time()
        self.execution_times[timer_name] = start_time

        try:
            result = await fn()
            end_time = time.time()
            duration = end_time - start_time
            self.execution_times[timer_name] = duration
            return result
        except Exception as error:
            end_time = time.time()
            duration = end_time - start_time
            self.execution_times[timer_name] = duration
            raise error

    def create_metadata(self,
                        intent_classifier_result: ClassifierResult | None,
                        user_input: str,
                        user_id: str,
                        session_id: str,
                        additional_params: dict[str, str]) -> AgentProcessingResult:
        base_metadata = AgentProcessingResult(
            user_input=user_input,
            agent_id="no_agent_selected",
            agent_name="No Agent",
            user_id=user_id,
            session_id=session_id,
            additional_params=additional_params
        )

        if not intent_classifier_result or not intent_classifier_result.selected_agent:
            if (base_metadata.additional_params):
                base_metadata.additional_params['error_type'] = 'classification_failed'
            else:
                base_metadata.additional_params = {'error_type': 'classification_failed'}
        else:
            base_metadata.agent_id = intent_classifier_result.selected_agent.id
            base_metadata.agent_name = intent_classifier_result.selected_agent.name

        return base_metadata

    def get_fallback_result(self) -> ClassifierResult:
        return ClassifierResult(selected_agent=self.get_default_agent(), confidence=0)

    async def save_message(self,
                           message: ConversationMessage,
                           user_id: str, session_id: str,
                           agent: Agent):
        if agent and agent.save_chat:
            return await self.storage.save_chat_message(user_id,
                                                        session_id,
                                                        agent.id,
                                                        message,
                                                        self.config.MAX_MESSAGE_PAIRS_PER_AGENT)
    async def save_messages(self,
                           messages: list[ConversationMessage] | list[TimestampedMessage],
                           user_id: str, session_id: str,
                           agent: Agent):
        if agent and agent.save_chat:
            for message in messages:
                # TODO: change this to self.storage.save_chat_messages() when SupervisorAgent is merged
                await self.storage.save_chat_message(user_id,
                                                        session_id,
                                                        agent.id,
                                                        message,
                                                        self.config.MAX_MESSAGE_PAIRS_PER_AGENT)
