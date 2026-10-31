from typing import Optional, Union
import time
from collections import defaultdict
from agent_squad.storage import ChatStorage
from agent_squad.types import ConversationMessage, TimestampedMessage
from agent_squad.utils import Logger

class InMemoryChatStorage(ChatStorage):
    """In-memory history preserving timestamps on TimestampedMessage inputs.

    Ordinary ConversationMessage inputs receive save-time timestamps. Merged
    history is sorted by the persisted timestamps.
    """

    def __init__(self):
        super().__init__()
        self.conversations = defaultdict(list)

    async def save_chat_message(
        self,
        user_id: str,
        session_id: str,
        agent_id: str,
        new_message: Union[ConversationMessage, TimestampedMessage],
        max_history_size: Optional[int] = None
    ) -> list[dict]:
        key = self._generate_key(user_id, session_id, agent_id)
        conversation = self.conversations[key]

        if self.is_same_role_as_last_message(conversation, new_message):
            Logger.debug(f"> Consecutive {new_message.role} \
                       message detected for agent {agent_id}. Not saving.")
            return self._remove_timestamps(conversation)

        if isinstance(new_message, ConversationMessage):
            timestamped_message = TimestampedMessage(
                role=new_message.role,
                content=new_message.content,
                timestamp=new_message.timestamp if isinstance(new_message, TimestampedMessage) else 0)

        conversation.append(timestamped_message)

        conversation = self.trim_conversation(conversation, max_history_size)
        self.conversations[key] = conversation
        return self._remove_timestamps(conversation)


    async def save_chat_messages(self,
                                user_id: str,
                                session_id: str,
                                agent_id: str,
                                new_messages: Union[list[ConversationMessage], list[TimestampedMessage]],
                                max_history_size: Optional[int] = None
    ) -> bool:
        key = self._generate_key(user_id, session_id, agent_id)
        conversation = self.conversations[key]
        #TODO: check messages are consecutive

        # if self.is_same_role_as_last_message(conversation, new_message):
        #     Logger.debug(f"> Consecutive {new_message.role} \
        #                message detected for agent {agent_id}. Not saving.")
        #     return self._remove_timestamps(conversation)

        if isinstance(new_messages[0], ConversationMessage):  # Check only first message
            new_messages = [TimestampedMessage(
                    role=new_message.role,
                    content=new_message.content,
                    timestamp=new_message.timestamp if isinstance(new_message, TimestampedMessage) else 0
                    )
                for new_message in new_messages]

        conversation.extend(new_messages)
        conversation = self.trim_conversation(conversation, max_history_size)
        self.conversations[key] = conversation
        return self._remove_timestamps(conversation)


    async def fetch_chat(
        self,
        user_id: str,
        session_id: str,
        agent_id: str,
        max_history_size: Optional[int] = None
    ) -> list[dict]:
        key = self._generate_key(user_id, session_id, agent_id)
        conversation = self.conversations[key]
        if max_history_size is not None:
            conversation = self.trim_conversation(conversation, max_history_size)
        return self._remove_timestamps(conversation)

    async def fetch_all_chats(
        self,
        user_id: str,
        session_id: str
    ) -> list[ConversationMessage]:
        all_messages = []
        for key, messages in self.conversations.items():
            stored_user_id, stored_session_id, agent_id = key.split('#')
            if stored_user_id == user_id and stored_session_id == session_id:
                for message in messages:
                    new_content = message.content if message.content else []

                    if len(new_content) > 0 and message.role == "assistant":
                        new_content = [{'text':f"[{agent_id}] {new_content[0]['text']}"}]
                    all_messages.append(TimestampedMessage(
                        role=message.role,
                        content=new_content,
                        timestamp=message.timestamp
                    ))

        # Sort messages by timestamp
        all_messages.sort(key=lambda x: x.timestamp)
        return self._remove_timestamps(all_messages)

    @staticmethod
    def _generate_key(user_id: str, session_id: str, agent_id: str) -> str:
        return f"{user_id}#{session_id}#{agent_id}"

    @staticmethod
    def _remove_timestamps(messages: list[dict]) -> list[ConversationMessage]:
        return [ConversationMessage(
            role=message.role,
            content=message.content
            ) for message in messages]
