from typing import Union, AsyncIterable, Optional, Any
from agent_squad.types import ConversationMessage, ParticipantRole
from agent_squad.utils.logger import Logger
from .agent import Agent, AgentOptions

class ChainAgentOptions(AgentOptions):
    def __init__(self, agents: list[Agent], default_output: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.agents = agents
        self.default_output = default_output

class ChainAgent(Agent):
    def __init__(self, options: ChainAgentOptions):
        super().__init__(options)
        self.agents = options.agents
        self.default_output = options.default_output or "No output generated from the chain."
        if len(self.agents) == 0:
            raise ValueError("ChainAgent requires at least one agent in the chain.")

    async def process_request(
        self,
        input_text: str,
        user_id: str,
        session_id: str,
        chat_history: list[ConversationMessage],
        additional_params: Optional[dict[str, str]] = None
    ) -> Union[ConversationMessage, AsyncIterable[Any]]:
        current_input = input_text
        final_response: Union[ConversationMessage, AsyncIterable[Any]]

        for i, agent in enumerate(self.agents):
            is_last_agent = i == len(self.agents) - 1
            try:
                response = await agent.process_request(
                    current_input,
                    user_id,
                    session_id,
                    chat_history,
                    additional_params
                )
                if self.is_conversation_message(response):
                    if response.content and 'text' in response.content[0]:
                        current_input = response.content[0]['text']
                        final_response = response
                    else:
                        Logger.warn(f"Agent {agent.name} returned no text content.")
                        return self.create_default_response()
                elif self.is_async_iterable(response):
                    if not is_last_agent:
                        Logger.warn(f"Intermediate agent {agent.name} returned a streaming response, which is not allowed.")
                        return self.create_default_response()
                    # It's the last agent and streaming is allowed
                    final_response = response
                else:
                    Logger.warn(f"Agent {agent.name} returned an invalid response type.")
                    return self.create_default_response()

                # If it's not the last agent, ensure we have a non-streaming response to pass to the next agent
                if not is_last_agent and not self.is_conversation_message(final_response):
                    Logger.error(f"Expected non-streaming response from intermediate agent {agent.name}")
                    return self.create_default_response()

            except Exception as error:
                Logger.error(f"Error processing request with agent {agent.name}:{str(error)}")
                raise Exception(f"Error processing request with agent {agent.name}:{str(error)}") from error

        return final_response

    @staticmethod
    def is_async_iterable(obj: Any) -> bool:
        return hasattr(obj, '__aiter__')

    @staticmethod
    def is_conversation_message(response: Any) -> bool:
        return (
            isinstance(response, ConversationMessage) and
            hasattr(response, 'role') and
            hasattr(response, 'content') and
            isinstance(response.content, list)
        )

    def create_default_response(self) -> ConversationMessage:
        return ConversationMessage(
            role=ParticipantRole.ASSISTANT.value,
            content=[{"text": self.default_output}]
        )