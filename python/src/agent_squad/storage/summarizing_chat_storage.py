"""
SummarizingChatStorage — a ChatStorage wrapper that keeps conversation history
compact by summarizing old messages whenever the in-memory buffer grows past a
threshold.

Raw messages are always written to the inner storage unchanged — they remain
available for analytics, audit, or replay via ``fetch_all_chats``. The
summarizer only affects what the agent sees through ``fetch_chat``.

How it works
~~~~~~~~~~~~
An in-memory buffer is maintained per (user, session, agent) slot.

* The buffer is **activated lazily** on the first ``fetch_chat`` call that
  finds history above the threshold.  Before that, all operations are pure
  delegations to the inner storage.

* Once the buffer is active, every ``save_chat_message`` / ``save_chat_messages``
  appends the new message to it and, if the buffer exceeds the threshold again,
  calls the summarizer **immediately** — so the next ``fetch_chat`` is always
  fast and never triggers an LLM call.

* ``fetch_all_chats`` is never intercepted: the raw full history is always
  available to the classifier and for analytics or audit purposes.

Usage::

    from agent_squad.storage import SummarizingChatStorage, InMemoryChatStorage
    from agent_squad.types import ConversationMessage, ParticipantRole

    async def my_summarizer(
        history: list[ConversationMessage],
        keep_last: int,
    ) -> list[ConversationMessage]:
        old = history[:-keep_last * 2]
        recent = history[-keep_last * 2:]
        summary_text = await call_llm_to_summarize(old)
        return [
            ConversationMessage(
                role=ParticipantRole.USER.value,
                content=[{"text": f"[Summary]: {summary_text}"}],
            )
        ] + recent

    storage = SummarizingChatStorage(
        storage=InMemoryChatStorage(),
        summarizer=my_summarizer,
        trigger_at=20,   # compress when buffer exceeds 20 pairs (40 messages)
        keep_last=2,     # keep the 2 most recent pairs verbatim
    )
"""

from typing import Callable, Awaitable, Optional, Union

from agent_squad.storage.chat_storage import ChatStorage
from agent_squad.types import ConversationMessage, TimestampedMessage


class SummarizingChatStorage(ChatStorage):
    """A ``ChatStorage`` wrapper that keeps agent context small via summarization.

    Raw messages are always saved to the inner storage untouched. The summarizer
    only affects what ``fetch_chat`` returns to the agent. ``fetch_all_chats``
    always returns the raw, full history.

    Args:
        storage: The inner ``ChatStorage`` to wrap.
        summarizer: Async callable ``(history, keep_last) -> compressed``.
            Receives the current buffer and the number of recent pairs to keep
            verbatim. Must return the compressed history.
        trigger_at: Number of message **pairs** above which the buffer is
            compressed. Default: 20.
        keep_last: Number of most-recent message pairs passed to the summarizer
            to keep verbatim. Default: 2.
    """

    def __init__(
        self,
        storage: ChatStorage,
        summarizer: Callable[[list[ConversationMessage], int], Awaitable[list[ConversationMessage]]],
        trigger_at: int = 20,
        keep_last: int = 2,
    ) -> None:
        super().__init__()
        self._storage = storage
        self._summarizer = summarizer
        self._trigger_at = trigger_at
        self._keep_last = keep_last
        # Per-(user, session, agent) in-memory buffer of the current compressed
        # history. A missing key means the buffer is not yet active for that slot
        # — saves are pure delegations until the first fetch crosses the threshold.
        self._buffers: dict[str, list[ConversationMessage]] = {}

    @staticmethod
    def _key(user_id: str, session_id: str, agent_id: str) -> str:
        return f"{user_id}#{session_id}#{agent_id}"

    @staticmethod
    def _to_conversation_message(msg: Union[ConversationMessage, TimestampedMessage]) -> ConversationMessage:
        if isinstance(msg, ConversationMessage):
            return msg
        return ConversationMessage(role=msg.role, content=msg.content)

    async def _compress_if_needed(self, key: str) -> None:
        buf = self._buffers.get(key)
        if buf is not None and len(buf) > self._trigger_at * 2:
            self._buffers[key] = await self._summarizer(buf, self._keep_last)

    async def save_chat_message(
        self,
        user_id: str,
        session_id: str,
        agent_id: str,
        new_message: Union[ConversationMessage, TimestampedMessage],
        max_history_size: Optional[int] = None,
    ) -> bool:
        key = self._key(user_id, session_id, agent_id)
        if key in self._buffers:
            self._buffers[key].append(self._to_conversation_message(new_message))
            await self._compress_if_needed(key)
        return await self._storage.save_chat_message(
            user_id, session_id, agent_id, new_message, max_history_size
        )

    async def save_chat_messages(
        self,
        user_id: str,
        session_id: str,
        agent_id: str,
        new_messages: Union[list[ConversationMessage], list[TimestampedMessage]],
        max_history_size: Optional[int] = None,
    ) -> bool:
        key = self._key(user_id, session_id, agent_id)
        if key in self._buffers:
            for msg in new_messages:
                self._buffers[key].append(self._to_conversation_message(msg))
            await self._compress_if_needed(key)
        return await self._storage.save_chat_messages(
            user_id, session_id, agent_id, new_messages, max_history_size
        )

    async def fetch_chat(
        self,
        user_id: str,
        session_id: str,
        agent_id: str,
        max_history_size: Optional[int] = None,
    ) -> list[ConversationMessage]:
        key = self._key(user_id, session_id, agent_id)

        # Buffer is active — return it directly (no storage read, no LLM call).
        if key in self._buffers:
            return self._buffers[key]

        # Cold start: load raw history from the inner store.
        history = await self._storage.fetch_chat(user_id, session_id, agent_id, max_history_size)

        if len(history) > self._trigger_at * 2:
            compressed = await self._summarizer(history, self._keep_last)
            self._buffers[key] = compressed
            return compressed

        return history

    async def fetch_all_chats(
        self,
        user_id: str,
        session_id: str,
    ) -> list[ConversationMessage]:
        # Never intercepted — raw history always available for analytics/audit.
        return await self._storage.fetch_all_chats(user_id, session_id)
