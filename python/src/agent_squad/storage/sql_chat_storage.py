import time
import json
from typing import Optional, Union
from libsql_client import create_client
from agent_squad.storage import ChatStorage
from agent_squad.types import ConversationMessage, ParticipantRole, TimestampedMessage
from agent_squad.utils import Logger

class SqlChatStorage(ChatStorage):
    """SQL chat storage supporting local SQLite and remote Turso databases.

    Existing TimestampedMessage inputs keep their timestamps; ordinary messages
    receive generated timestamps. Merged history uses the persisted timestamps.
    """

    def __init__(
        self,
        url: str,
        auth_token: str | None = None
    ):
        """Initialize SQL storage.

        Args:
            url: Database URL (e.g., 'file:local.db' or 'libsql://your-db-url.com')
            auth_token: Authentication token for remote databases (optional)
        """
        super().__init__()
        self.client = create_client(
            url=url,
            auth_token=auth_token
        )

    async def initialize(self) -> None:
        """Initialize the database asynchronously. Must be called after creating the instance."""
        await self._initialize_database()

    async def _initialize_database(self) -> None:
        """Create necessary tables and indexes if they don't exist."""
        try:
            # Create conversations table
            await self.client.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    message_index INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    PRIMARY KEY (user_id, session_id, agent_id, message_index)
                )
            """)

            # Create index for faster queries
            await self.client.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_lookup
                ON conversations(user_id, session_id, agent_id)
            """)
        except Exception as error:
            Logger.error(f"Error initializing database: {str(error)}")
            raise error

    async def save_chat_message(
        self,
        user_id: str,
        session_id: str,
        agent_id: str,
        new_message: Union[ConversationMessage, TimestampedMessage],
        max_history_size: Optional[int] = None
    ) -> list[ConversationMessage]:
        """Save a new chat message."""
        try:
            # Fetch existing conversation
            existing_conversation = await self.fetch_chat(user_id, session_id, agent_id)

            if self.is_same_role_as_last_message(existing_conversation, new_message):
                Logger.debug(f"> Consecutive {new_message.role} message detected for agent {agent_id}. Not saving.")
                return existing_conversation

            # Convert to TimestampedMessage if needed
            if isinstance(new_message, ConversationMessage):
                new_message = TimestampedMessage(
                    role=new_message.role,
                    content=new_message.content,
                    timestamp=new_message.timestamp if isinstance(new_message, TimestampedMessage) else 0
                )

            # Get next message index
            result = await self.client.execute("""
                SELECT COALESCE(MAX(message_index) + 1, 0) as next_index
                FROM conversations
                WHERE user_id = ? AND session_id = ? AND agent_id = ?
            """, [user_id, session_id, agent_id])
            next_index = result[0]['next_index']
            content = json.dumps(new_message.content)

            # Insert new message
            await self.client.execute("""
                INSERT INTO conversations (
                    user_id, session_id, agent_id, message_index,
                    role, content, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [
                user_id, session_id, agent_id, next_index,
                new_message.role, content, new_message.timestamp or int(time.time() * 1000)
            ])

            # Clean up old messages if max_history_size is set
            if max_history_size is not None:
                await self.client.execute("""
                    DELETE FROM conversations
                    WHERE user_id = ?
                        AND session_id = ?
                        AND agent_id = ?
                        AND message_index <= (
                            SELECT MAX(message_index) - ?
                            FROM conversations
                            WHERE user_id = ?
                                AND session_id = ?
                                AND agent_id = ?
                        )
                """, [
                    user_id, session_id, agent_id,
                    max_history_size,
                    user_id, session_id, agent_id
                ])

            # Return updated conversation
            return await self.fetch_chat(user_id, session_id, agent_id)

        except Exception as error:
            Logger.error(f"Error saving message: {str(error)}")
            raise error

    def _validate_message_content(self, content: Optional[list[dict[str, str]]]) -> None:
        """Validate message content before serialization."""
        if content is None:
            raise ValueError("Message content cannot be None")
        if not isinstance(content, list):
            raise ValueError("Message content must be a list")
        if not all(isinstance(item, dict) for item in content):
            raise ValueError("Message content must be a list of dictionaries")

    async def save_chat_messages(
        self,
        user_id: str,
        session_id: str,
        agent_id: str,
        new_messages: Union[list[ConversationMessage], list[TimestampedMessage]],
        max_history_size: Optional[int] = None
    ) -> list[ConversationMessage]:
        """Save multiple chat messages in a single transaction."""
        try:
            if not new_messages:
                return await self.fetch_chat(user_id, session_id, agent_id)

            # Convert messages to TimestampedMessage if needed
            timestamped_messages = []
            base_timestamp = int(time.time() * 1000)

            for i, message in enumerate(new_messages):
                if isinstance(message, ConversationMessage):
                    timestamped_messages.append(TimestampedMessage(
                        role=message.role,
                        content=message.content,
                        timestamp=message.timestamp if isinstance(message, TimestampedMessage) else base_timestamp + i
                    ))
                else:
                    timestamped_messages.append(message)

            # Get next message index
            result = await self.client.execute("""
                SELECT COALESCE(MAX(message_index) + 1, 0) as next_index
                FROM conversations
                WHERE user_id = ? AND session_id = ? AND agent_id = ?
            """, [user_id, session_id, agent_id])

            next_index = result[0]['next_index']

            # Validate and prepare all messages first to catch any errors
            message_params = []
            for i, message in enumerate(timestamped_messages):
                self._validate_message_content(message.content)
                content = json.dumps(message.content)
                message_params.append([
                    user_id, session_id, agent_id, next_index + i,
                    message.role, content, message.timestamp or (base_timestamp + i)
                ])

            # Insert messages one by one
            for params in message_params:
                await self.client.execute("""
                    INSERT INTO conversations (
                        user_id, session_id, agent_id, message_index,
                        role, content, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, params)

            # Clean up old messages if max_history_size is set
            if max_history_size is not None:
                await self.client.execute("""
                    DELETE FROM conversations
                    WHERE user_id = ?
                        AND session_id = ?
                        AND agent_id = ?
                        AND message_index <= (
                            SELECT MAX(message_index) - ?
                            FROM conversations
                            WHERE user_id = ?
                                AND session_id = ?
                                AND agent_id = ?
                        )
                """, [
                    user_id, session_id, agent_id,
                    max_history_size,
                    user_id, session_id, agent_id
                ])

            # Return updated conversation
            return await self.fetch_chat(user_id, session_id, agent_id)

        except Exception as error:
            Logger.error(f"Error saving messages: {str(error)}")
            raise error

    async def fetch_chat(
        self,
        user_id: str,
        session_id: str,
        agent_id: str,
        max_history_size: int | None = None
    ) -> list[ConversationMessage]:
        """Fetch chat messages."""
        try:
            query = """
                SELECT role, content, timestamp
                FROM conversations
                WHERE user_id = ? AND session_id = ? AND agent_id = ?
                ORDER BY message_index {}
            """.format('DESC' if max_history_size else 'ASC')

            params = [user_id, session_id, agent_id]

            result = await self.client.execute(query, params)
            messages = list(result)  # Convert ResultSet to list

            if max_history_size:
                messages = messages[:max_history_size]
                messages.reverse()

            return [
                ConversationMessage(
                    role=msg['role'],
                    content=json.loads(msg['content'])
                ) for msg in messages
            ]
        except Exception as error:
            Logger.error(f"Error fetching chat: {str(error)}")
            raise error

    async def fetch_all_chats(
        self,
        user_id: str,
        session_id: str
    ) -> list[ConversationMessage]:
        """Fetch all chat messages for a user and session."""
        try:
            result = await self.client.execute("""
                SELECT role, content, timestamp, agent_id
                FROM conversations
                WHERE user_id = ? AND session_id = ?
                ORDER BY timestamp ASC
            """, [user_id, session_id])

            return [
                ConversationMessage(
                    role=msg['role'],
                    content=self._format_content(
                        msg['role'],
                        json.loads(msg['content']),
                        msg['agent_id']
                    )
                ) for msg in result
            ]
        except Exception as error:
            Logger.error(f"Error fetching all chats: {str(error)}")
            raise error

    def _format_content(
        self,
        role: str,
        content: list | str,
        agent_id: str
    ) -> list[dict[str, str]]:
        """Format message content with agent ID for assistant messages."""
        if role == ParticipantRole.ASSISTANT.value:
            text = content[0]['text'] if isinstance(content, list) else content
            return [{'text': f"[{agent_id}] {text}"}]
        return content if isinstance(content, list) else [{'text': content}]

    async def close(self) -> None:
        """Close the database connection."""
        try:
            await self.client.close()
        except Exception as error:
            Logger.error(f"Error closing database connection: {str(error)}")
            raise error