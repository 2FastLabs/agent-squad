from abc import ABC, abstractmethod
from typing import Optional, Union
from agent_squad.types import ConversationMessage, TimestampedMessage

class ChatStorage(ABC):
    """Abstract base class representing the interface for an agent.
    """
    def is_same_role_as_last_message(self,
                               conversation: list[ConversationMessage],
                               new_message: ConversationMessage) -> bool:
        """
        Check if the new message is consecutive with the last message in the conversation.

        Args:
            conversation (list[ConversationMessage]): The existing conversation.
            new_message (ConversationMessage): The new message to check.

        Returns:
            bool: True if the new message is consecutive, False otherwise.
        """
        if not conversation:
            return False
        return conversation[-1].role == new_message.role

    def trim_conversation(self,
                          conversation: list[ConversationMessage],
                          max_history_size: Optional[int] = None) -> list[ConversationMessage]:
        """
        Trim the conversation to the specified maximum history size.

        Args:
            conversation (list[ConversationMessage]): The conversation to trim.
            max_history_size (Optional[int]): The maximum number of messages to keep.

        Returns:
            list[ConversationMessage]: The trimmed conversation.
        """
        if max_history_size is None:
            return conversation
        # Ensure max_history_size is even to maintain complete binoms
        if max_history_size % 2 == 0:
            adjusted_max_history_size = max_history_size
        else:
            adjusted_max_history_size = max_history_size - 1
        # A size of 0 (or 1, rounded down to 0) means keep no messages.
        # conversation[-0:] is conversation[:], which would keep everything.
        if adjusted_max_history_size <= 0:
            return []
        return conversation[-adjusted_max_history_size:]

    @abstractmethod
    async def save_chat_message(self,
                                user_id: str,
                                session_id: str,
                                agent_id: str,
                                new_message: Union[ConversationMessage, TimestampedMessage],
                                max_history_size: Optional[int] = None) -> bool:
        """
        Save a new chat message.

        Args:
            user_id (str): The user ID.
            session_id (str): The session ID.
            agent_id (str): The agent ID.
            new_message (ConversationMessage or TimestampedMessage): The new message to save.
            max_history_size (Optional[int]): The maximum history size.

        Returns:
            bool: True if the message was saved successfully, False otherwise.
        """

    @abstractmethod
    async def save_chat_messages(self,
                                user_id: str,
                                session_id: str,
                                agent_id: str,
                                new_messages: Union[list[ConversationMessage], list[TimestampedMessage]],
                                max_history_size: Optional[int] = None) -> bool:
        """
        Save multiple messages at once.

        Args:
            user_id (str): The user ID.
            session_id (str): The session ID.
            agent_id (str): The agent ID.
            new_messages (list[ConversationMessage or TimestampedMessage]): The list of messages to save.
            max_history_size (Optional[int]): The maximum history size.

        Returns:
            bool: True if the messages were saved successfully, False otherwise.
        """

    @abstractmethod
    async def fetch_chat(self,
                         user_id: str,
                         session_id: str,
                         agent_id: str,
                         max_history_size: Optional[int] = None) -> list[ConversationMessage]:
        """
        Fetch chat messages.

        Args:
            user_id (str): The user ID.
            session_id (str): The session ID.
            agent_id (str): The agent ID.
            max_history_size (Optional[int]): The maximum number of messages to fetch.

        Returns:
            list[ConversationMessage]: The fetched chat messages.
        """

    @abstractmethod
    async def fetch_all_chats(self,
                              user_id: str,
                              session_id: str) -> list[ConversationMessage]:
        """
        Fetch all chat messages for a user and session.

        Args:
            user_id (str): The user ID.
            session_id (str): The session ID.

        Returns:
            list[ConversationMessage]: All chat messages for the user and session.
        """
