"""
SummarizingChatStorage — a ChatStorage wrapper that compresses long conversation
histories using a user-supplied async callable.

When the history for a given agent exceeds ``trigger_at`` message pairs, the
wrapper calls the summarizer, writes the compressed result back to the inner
store (so subsequent fetches are fast), and returns the compressed history.

``fetch_all_chats`` is never intercepted — the classifier always sees the full
cross-agent picture unmodified.

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
                content=[{"text": f"[Conversation summary]: {summary_text}"}],
            )
        ] + recent

    storage = SummarizingChatStorage(
        storage=InMemoryChatStorage(),
        summarizer=my_summarizer,
        trigger_at=20,   # summarize when history exceeds 20 pairs
        keep_last=2,     # keep the 2 most recent pairs verbatim
    )
"""

from typing import Callable, Awaitable, Optional, Union

from agent_squad.storage.chat_storage import ChatStorage
from agent_squad.types import ConversationMessage, TimestampedMessage


class SummarizingChatStorage(ChatStorage):
    """A ``ChatStorage`` wrapper that automatically compresses long histories.

    Wraps any ``ChatStorage`` implementation. On every ``fetch_chat`` call,
    if the returned history exceeds ``trigger_at`` message pairs the summarizer
    callable is invoked, the compressed result is written back to the inner
    store, and the compressed history is returned to the caller.

    All other methods (``save_chat_message``, ``save_chat_messages``,
    ``fetch_all_chats``) are pure delegations to the inner storage.

    Args:
        storage: The inner ``ChatStorage`` to wrap.
        summarizer: An async callable with the signature
            ``async (history: list[ConversationMessage], keep_last: int)
            -> list[ConversationMessage]``.
            Receives the full history and the number of recent pairs to
            preserve verbatim. Must return the compressed history.
        trigger_at: Number of message **pairs** (user + assistant = 1 pair)
            above which summarization is triggered. Default: 20.
        keep_last: Number of most-recent message pairs to keep verbatim and
            pass to the summarizer as context. Default: 2.
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
        # Internal cache: key → compressed history.
        # After summarization the compressed result is stored here so subsequent
        # fetch_chat calls return immediately without hitting the inner storage or
        # re-invoking the summarizer. A save_chat_message call invalidates the entry
        # so the next fetch re-evaluates from the inner store.
        self._cache: dict[str, list[ConversationMessage]] = {}

    @staticmethod
    def _cache_key(user_id: str, session_id: str, agent_id: str) -> str:
        return f"{user_id}#{session_id}#{agent_id}"

    async def save_chat_message(
        self,
        user_id: str,
        session_id: str,
        agent_id: str,
        new_message: Union[ConversationMessage, TimestampedMessage],
        max_history_size: Optional[int] = None,
    ) -> bool:
        # Invalidate cache so the next fetch_chat re-evaluates from the inner store.
        self._cache.pop(self._cache_key(user_id, session_id, agent_id), None)
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
        # Invalidate cache so the next fetch_chat re-evaluates from the inner store.
        self._cache.pop(self._cache_key(user_id, session_id, agent_id), None)
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
        key = self._cache_key(user_id, session_id, agent_id)

        # Return cached compressed history if available.
        if key in self._cache:
            return self._cache[key]

        # Fetch the full history — we need it untruncated to evaluate the
        # trigger threshold. max_history_size is accepted for interface
        # compatibility but size management is delegated to trigger_at/keep_last.
        history = await self._storage.fetch_chat(user_id, session_id, agent_id)

        if len(history) > self._trigger_at * 2:
            compressed = await self._summarizer(history, self._keep_last)
            # Cache the compressed result — subsequent fetches return this directly.
            self._cache[key] = compressed
            return compressed

        return history

    async def fetch_all_chats(
        self,
        user_id: str,
        session_id: str,
    ) -> list[ConversationMessage]:
        # Never intercepted — the classifier must see the full cross-agent history.
        return await self._storage.fetch_all_chats(user_id, session_id)
