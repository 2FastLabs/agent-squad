from typing import Union, AsyncIterable, Optional, Any, TypeAlias
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from agent_squad.types import ConversationMessage
from agent_squad.utils import Logger
from uuid import UUID

# Type aliases for complex types
AgentParamsType: TypeAlias = dict[str, Any]
AgentOutputType: TypeAlias = Union[str, "AgentStreamResponse", Any]  # Forward reference


@dataclass
class AgentProcessingResult:
    """
    Contains metadata about the result of an agent's processing.

    Attributes:
        user_input: The original input from the user
        agent_id: Unique identifier for the agent
        agent_name: Display name of the agent
        user_id: Identifier for the user
        session_id: Identifier for the current session
        additional_params: Optional additional parameters for the agent
    """

    user_input: str
    agent_id: str
    agent_name: str
    user_id: str
    session_id: str
    additional_params: AgentParamsType = field(default_factory=dict)


@dataclass
class AgentStreamResponse:
    """
    Represents a streaming response from an agent.

    Attributes:
        text: The current text in the stream
        final_message: The complete message when streaming is complete
    """

    text: str = ""
    thinking: Optional[str] = ""
    final_message: Optional[ConversationMessage] = None
    final_thinking: Optional[str] = None


@dataclass
class AgentResponse:
    """
    Complete response from an agent, including metadata and output.

    Attributes:
        metadata: Processing metadata
        output: The actual output from the agent
        streaming: Whether this response is streaming
    """

    metadata: AgentProcessingResult
    output: AgentOutputType
    streaming: bool


class AgentCallbacks:

    async def on_agent_start(
        self,
        agent_name,
        payload_input: Any,
        messages: list[Any],
        run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> dict:
        """
        Callback method that runs when an agent starts processing.

        This method is called at the beginning of an agent's execution, providing information
        about the agent session and its context.

        Parameters:
            self: The instance of the callback handler class.
            agent_name: Name of the agent that is starting.
            payload_input: Dictionary containing the agent's input.
            messages: List of message dictionaries representing the conversation history.
            run_id: Unique identifier for this specific agent run.
            tags: Optional list of string tags associated with this agent run.
            metadata: Optional dictionary containing additional metadata about the run.
            **kwargs: Additional keyword arguments that might be passed to the callback.

        Returns:
            dict: The agent tracking information, this is made available to all other
                callbacks using the agent_tracking_info key in kwargs.
        """
        return {}

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
        """
        Callback method that runs when an agent completes its processing.

        This method is called at the end of an agent's execution, providing information
        about the completed agent session and its response.

        Parameters:
            self: The instance of the callback handler class.
            agent_name: Name of the agent that is completing.
            response: Dictionary containing the agent's response or output.
            run_id: Unique identifier for this specific agent run.
            tags: Optional list of string tags associated with this agent run.
            metadata: Optional dictionary containing additional metadata about the run.
            **kwargs: Additional keyword arguments that might be passed to the callback.

        Returns:
            Any: The return value is implementation-dependent.
        """
        pass

    async def on_llm_start(
        self,
        name: str,
        payload_input: Any,
        run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Callback method that runs when an llm starts processing.

        This method is called at the beginning of an llm's execution, providing information
        about the llm session and its context.

        Parameters:
            self: The instance of the callback handler class.
            name: Name of the agent whose LLM call is starting.
            payload_input: Dictionary containing the LLM's input.
            run_id: Unique identifier for this specific agent run.
            tags: Optional list of string tags associated with this agent run.
            metadata: Optional dictionary containing additional metadata about the run.
            **kwargs: Additional keyword arguments that might be passed to the callback.

        Returns:
            Any: The return value is implementation-dependent.
        """
        pass

    """
    Defines callbacks that can be triggered during agent processing.
    Provides default implementations that can be overridden by subclasses.
    """
    async def on_llm_new_token(self,
                               token: str,
                               **kwargs: Any) -> None:
        """
        Called when a new token is generated by the LLM.

        Args:
            self: The instance of the callback handler class.
            token: The new token generated (text)
            **kwargs: Additional keyword arguments that might be passed to the callback.
        """
        pass  # Default implementation does nothing


    async def on_llm_end(
        self,
        name: str,
        output: Any,
        run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Callback method that runs when an llm stops.

        This method is called at the end of an llm's execution, providing information
        about the llm session and its context.

        Parameters:
            self: The instance of the callback handler class.
            name: Name of the LLM that is starting.
            output: Dictionary containing the agent's input.
            run_id: Unique identifier for this specific agent run.
            tags: Optional list of string tags associated with this agent run.
            metadata: Optional dictionary containing additional metadata about the run.
            **kwargs: Additional keyword arguments that might be passed to the callback.

        Returns:
            Any: The return value is implementation-dependent.
    """
    pass


@dataclass
class AgentOptions:
    """
    Configuration options for an agent.

    Attributes:
        name: The display name of the agent
        description: A description of the agent's purpose and capabilities
        save_chat: Whether to save the chat history
        callbacks: Optional callbacks for agent events
        LOG_AGENT_DEBUG_TRACE: Whether to enable debug tracing for this agent
    """

    name: str
    description: str
    save_chat: bool = True
    callbacks: Optional[AgentCallbacks] = None
    # Optional: Flag to enable/disable agent debug trace logging
    # If true, the agent will log additional debug information
    LOG_AGENT_DEBUG_TRACE: Optional[bool] = False


class Agent(ABC):
    """
    Abstract base class for all agents in the system.

    Implements common functionality and defines the required interface
    for concrete agent implementations.
    """

    def __init__(self, options: AgentOptions):
        """
        Initialize a new agent with the given options.

        Args:
            options: Configuration options for this agent
        """
        self.name = options.name
        self.id = self.generate_key_from_name(options.name)
        self.description = options.description
        self.save_chat = options.save_chat
        self.callbacks = (
            options.callbacks if options.callbacks is not None else AgentCallbacks()
        )
        self.log_debug_trace = options.LOG_AGENT_DEBUG_TRACE

    def is_streaming_enabled(self) -> bool:
        """
        Whether this agent supports streaming responses.

        Returns:
            True if streaming is enabled, False otherwise
        """
        return False

    @staticmethod
    def generate_key_from_name(name: str) -> str:
        """
        Generate a standardized key from an agent name.

        Args:
            name: The display name to convert

        Returns:
            A lowercase, hyphenated key with special characters removed
        """
        # Remove special characters and replace spaces with hyphens
        key = re.sub(r"[^a-zA-Z0-9\s-]", "", name)
        key = re.sub(r"\s+", "-", key)
        return key.lower()

    @abstractmethod
    async def process_request(
        self,
        input_text: str,
        user_id: str,
        session_id: str,
        chat_history: list[ConversationMessage],
        additional_params: Optional[AgentParamsType] = None,
    ) -> Union[ConversationMessage, AsyncIterable[AgentOutputType]]:
        """
        Process a user request and generate a response.

        Args:
            input_text: The user's input text
            user_id: Identifier for the user
            session_id: Identifier for the current session
            chat_history: List of previous messages in the conversation
            additional_params: Optional additional parameters

        Returns:
            Either a complete message or an async iterable for streaming responses
        """
        pass

    def log_debug(self, class_name: str, message: str, data: Any = None) -> None:
        """
        Log a debug message if debug tracing is enabled.

        Args:
            class_name: Name of the class logging the message
            message: The message to log
            data: Optional data to include in the log
        """
        if self.log_debug_trace:
            prefix = f"> {class_name} \n> {self.name} \n>"
            if data:
                Logger.info(f"{prefix} {message} \n> {data}")
            else:
                Logger.info(f"{prefix} {message} \n>")
