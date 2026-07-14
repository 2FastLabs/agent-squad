import pytest
from unittest.mock import AsyncMock, MagicMock
from agent_squad.types import ConversationMessage, ParticipantRole
from agent_squad.storage import InMemoryChatStorage
from agent_squad.storage.summarizing_chat_storage import SummarizingChatStorage


def make_message(role: str, text: str) -> ConversationMessage:
    return ConversationMessage(role=role, content=[{"text": text}])


def make_history(num_pairs: int) -> list[ConversationMessage]:
    """Return a list of alternating user/assistant messages (num_pairs pairs)."""
    history = []
    for i in range(num_pairs):
        history.append(make_message(ParticipantRole.USER.value, f"User message {i + 1}"))
        history.append(make_message(ParticipantRole.ASSISTANT.value, f"Assistant message {i + 1}"))
    return history


async def identity_summarizer(history, keep_last):
    """Summarizer that returns the last keep_last pairs verbatim."""
    return history[-(keep_last * 2):]


@pytest.fixture
def inner_storage():
    return InMemoryChatStorage()


@pytest.mark.asyncio
async def test_below_trigger_returns_history_unchanged(inner_storage):
    """History below trigger passes through with no summarizer call."""
    summarizer = AsyncMock(side_effect=identity_summarizer)
    storage = SummarizingChatStorage(inner_storage, summarizer, trigger_at=5, keep_last=2)

    history = make_history(3)
    await inner_storage.save_chat_messages("u", "s", "a", history)

    result = await storage.fetch_chat("u", "s", "a")

    assert len(result) == 6
    summarizer.assert_not_called()


@pytest.mark.asyncio
async def test_at_trigger_boundary_does_not_summarize(inner_storage):
    """Exactly trigger_at*2 messages — condition is strictly >, no summarization."""
    summarizer = AsyncMock(side_effect=identity_summarizer)
    storage = SummarizingChatStorage(inner_storage, summarizer, trigger_at=5, keep_last=2)

    history = make_history(5)  # exactly 10 messages = trigger_at * 2
    await inner_storage.save_chat_messages("u", "s", "a", history)

    result = await storage.fetch_chat("u", "s", "a")

    assert len(result) == 10
    summarizer.assert_not_called()


@pytest.mark.asyncio
async def test_above_trigger_calls_summarizer(inner_storage):
    """History above trigger calls summarizer exactly once."""
    summarizer = AsyncMock(side_effect=identity_summarizer)
    storage = SummarizingChatStorage(inner_storage, summarizer, trigger_at=5, keep_last=2)

    history = make_history(6)  # 12 messages > 10
    await inner_storage.save_chat_messages("u", "s", "a", history)

    await storage.fetch_chat("u", "s", "a")

    summarizer.assert_called_once()


@pytest.mark.asyncio
async def test_summarizer_receives_full_history(inner_storage):
    """Summarizer receives the full history as first argument."""
    received_history = []

    async def capturing_summarizer(history, keep_last):
        received_history.extend(history)
        return history[-4:]

    storage = SummarizingChatStorage(inner_storage, capturing_summarizer, trigger_at=5, keep_last=2)
    history = make_history(6)
    await inner_storage.save_chat_messages("u", "s", "a", history)

    await storage.fetch_chat("u", "s", "a")

    assert len(received_history) == 12


@pytest.mark.asyncio
async def test_summarizer_receives_keep_last(inner_storage):
    """Summarizer receives keep_last as second argument."""
    received_keep_last = []

    async def capturing_summarizer(history, keep_last):
        received_keep_last.append(keep_last)
        return history[-4:]

    storage = SummarizingChatStorage(inner_storage, capturing_summarizer, trigger_at=5, keep_last=3)
    history = make_history(6)
    await inner_storage.save_chat_messages("u", "s", "a", history)

    await storage.fetch_chat("u", "s", "a")

    assert received_keep_last == [3]


@pytest.mark.asyncio
async def test_fetch_chat_returns_compressed_result(inner_storage):
    """fetch_chat returns the compressed result from the summarizer."""
    compressed = [make_message(ParticipantRole.USER.value, "Summary of previous conversation")]

    async def fixed_summarizer(history, keep_last):
        return compressed

    storage = SummarizingChatStorage(inner_storage, fixed_summarizer, trigger_at=5, keep_last=2)
    history = make_history(6)
    await inner_storage.save_chat_messages("u", "s", "a", history)

    result = await storage.fetch_chat("u", "s", "a")

    assert len(result) == 1
    assert result[0].content[0]["text"] == "Summary of previous conversation"


@pytest.mark.asyncio
async def test_save_back_caches_compressed_result(inner_storage):
    """After summarization, the compressed result is cached so subsequent fetches
    return it directly without re-calling the summarizer or inner storage."""
    call_count = 0

    async def counting_summarizer(history, keep_last):
        nonlocal call_count
        call_count += 1
        return [make_message(ParticipantRole.USER.value, "Compressed")]

    storage = SummarizingChatStorage(inner_storage, counting_summarizer, trigger_at=5, keep_last=2)
    history = make_history(6)
    await inner_storage.save_chat_messages("u", "s", "a", history)

    first = await storage.fetch_chat("u", "s", "a")
    second = await storage.fetch_chat("u", "s", "a")

    # Summarizer called only once — cache served the second fetch.
    assert call_count == 1
    assert len(first) == 1
    assert first[0].content[0]["text"] == "Compressed"
    assert second == first


@pytest.mark.asyncio
async def test_subsequent_fetch_uses_compressed_history(inner_storage):
    """After save-back, subsequent fetch returns the compressed history."""
    call_count = 0

    async def counting_summarizer(history, keep_last):
        nonlocal call_count
        call_count += 1
        return [make_message(ParticipantRole.USER.value, "Summary")]

    storage = SummarizingChatStorage(inner_storage, counting_summarizer, trigger_at=5, keep_last=2)
    history = make_history(6)
    await inner_storage.save_chat_messages("u", "s", "a", history)

    await storage.fetch_chat("u", "s", "a")
    result = await storage.fetch_chat("u", "s", "a")

    # Summarizer called only once — second fetch uses the saved-back compressed history
    assert call_count == 1
    assert len(result) == 1


@pytest.mark.asyncio
async def test_fetch_all_chats_never_intercepted(inner_storage):
    """fetch_all_chats is never intercepted regardless of history length."""
    summarizer = AsyncMock(side_effect=identity_summarizer)
    storage = SummarizingChatStorage(inner_storage, summarizer, trigger_at=5, keep_last=2)

    history = make_history(6)
    await inner_storage.save_chat_messages("u", "s", "a", history)

    result = await storage.fetch_all_chats("u", "s")

    summarizer.assert_not_called()
    assert len(result) == 12


@pytest.mark.asyncio
async def test_save_chat_message_delegates_to_inner(inner_storage):
    """save_chat_message reaches the inner storage unchanged."""
    summarizer = AsyncMock(side_effect=identity_summarizer)
    storage = SummarizingChatStorage(inner_storage, summarizer, trigger_at=5, keep_last=2)

    msg = make_message(ParticipantRole.USER.value, "Hello")
    await storage.save_chat_message("u", "s", "a", msg)

    saved = await inner_storage.fetch_chat("u", "s", "a")
    assert len(saved) == 1
    assert saved[0].content[0]["text"] == "Hello"


@pytest.mark.asyncio
async def test_summarizer_error_propagates(inner_storage):
    """If the summarizer raises, fetch_chat propagates the exception."""
    async def failing_summarizer(history, keep_last):
        raise ValueError("summarizer failed")

    storage = SummarizingChatStorage(inner_storage, failing_summarizer, trigger_at=5, keep_last=2)
    history = make_history(6)
    await inner_storage.save_chat_messages("u", "s", "a", history)

    with pytest.raises(ValueError, match="summarizer failed"):
        await storage.fetch_chat("u", "s", "a")
