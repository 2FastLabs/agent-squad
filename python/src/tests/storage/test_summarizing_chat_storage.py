import pytest
from unittest.mock import AsyncMock
from agent_squad.types import ConversationMessage, ParticipantRole
from agent_squad.storage import InMemoryChatStorage
from agent_squad.storage.summarizing_chat_storage import SummarizingChatStorage


def user_msg(text: str) -> ConversationMessage:
    return ConversationMessage(role=ParticipantRole.USER.value, content=[{"text": text}])


def assistant_msg(text: str) -> ConversationMessage:
    return ConversationMessage(role=ParticipantRole.ASSISTANT.value, content=[{"text": text}])


def make_history(n_pairs: int) -> list[ConversationMessage]:
    msgs = []
    for i in range(n_pairs):
        msgs.append(user_msg(f"User {i + 1}"))
        msgs.append(assistant_msg(f"Assistant {i + 1}"))
    return msgs


async def seed(storage: InMemoryChatStorage, msgs: list[ConversationMessage]) -> None:
    for msg in msgs:
        await storage.save_chat_message("u", "s", "a", msg)


# ---------------------------------------------------------------------------
# fetch_chat — lazy buffer activation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_below_trigger_returns_raw_history():
    inner = InMemoryChatStorage()
    await seed(inner, make_history(3))  # 6 msgs, trigger_at=5 → threshold=10
    summarizer = AsyncMock(side_effect=lambda h, k: h)
    storage = SummarizingChatStorage(inner, summarizer, trigger_at=5, keep_last=2)

    result = await storage.fetch_chat("u", "s", "a")

    assert len(result) == 6
    summarizer.assert_not_called()


@pytest.mark.asyncio
async def test_at_boundary_no_summarization():
    inner = InMemoryChatStorage()
    await seed(inner, make_history(5))  # exactly 10 = trigger_at * 2
    summarizer = AsyncMock(side_effect=lambda h, k: h)
    storage = SummarizingChatStorage(inner, summarizer, trigger_at=5, keep_last=2)

    result = await storage.fetch_chat("u", "s", "a")

    assert len(result) == 10
    summarizer.assert_not_called()


@pytest.mark.asyncio
async def test_above_trigger_calls_summarizer_on_fetch():
    inner = InMemoryChatStorage()
    await seed(inner, make_history(6))  # 12 > 10
    summarizer = AsyncMock(side_effect=lambda h, k: h[-k * 2:])
    storage = SummarizingChatStorage(inner, summarizer, trigger_at=5, keep_last=2)

    await storage.fetch_chat("u", "s", "a")

    summarizer.assert_called_once()
    assert summarizer.call_args[0][1] == 2


@pytest.mark.asyncio
async def test_fetch_returns_compressed_result():
    inner = InMemoryChatStorage()
    await seed(inner, make_history(6))
    compressed = [user_msg("Summary")]
    summarizer = AsyncMock(return_value=compressed)
    storage = SummarizingChatStorage(inner, summarizer, trigger_at=5, keep_last=2)

    result = await storage.fetch_chat("u", "s", "a")

    assert result == compressed


@pytest.mark.asyncio
async def test_subsequent_fetch_returns_buffer_without_calling_summarizer_again():
    inner = InMemoryChatStorage()
    await seed(inner, make_history(6))
    summarizer = AsyncMock(return_value=[user_msg("Summary")])
    storage = SummarizingChatStorage(inner, summarizer, trigger_at=5, keep_last=2)

    await storage.fetch_chat("u", "s", "a")
    result = await storage.fetch_chat("u", "s", "a")

    summarizer.assert_called_once()
    assert len(result) == 1
    assert result[0].content[0]["text"] == "Summary"


# ---------------------------------------------------------------------------
# save — pure delegation before buffer active
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_save_before_buffer_active_delegates_to_inner():
    inner = InMemoryChatStorage()
    summarizer = AsyncMock()
    storage = SummarizingChatStorage(inner, summarizer, trigger_at=5, keep_last=2)

    await storage.save_chat_message("u", "s", "a", user_msg("Hello"))

    saved = await inner.fetch_chat("u", "s", "a")
    assert len(saved) == 1
    summarizer.assert_not_called()


# ---------------------------------------------------------------------------
# save — buffer management after activation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_save_after_buffer_active_appends_to_buffer():
    inner = InMemoryChatStorage()
    await seed(inner, make_history(6))
    summarizer = AsyncMock(return_value=[user_msg("Summary")])
    storage = SummarizingChatStorage(inner, summarizer, trigger_at=5, keep_last=2)

    await storage.fetch_chat("u", "s", "a")  # activates buffer = [Summary]
    await storage.save_chat_message("u", "s", "a", user_msg("New"))

    result = await storage.fetch_chat("u", "s", "a")
    assert len(result) == 2  # [Summary, New]
    assert result[-1].content[0]["text"] == "New"


@pytest.mark.asyncio
async def test_save_triggers_compression_when_buffer_exceeds_threshold():
    inner = InMemoryChatStorage()
    await seed(inner, make_history(6))

    call_count = 0

    async def summarizer(history, keep_last):
        nonlocal call_count
        call_count += 1
        return [user_msg(f"Summary {call_count}")]

    storage = SummarizingChatStorage(inner, summarizer, trigger_at=5, keep_last=2)

    await storage.fetch_chat("u", "s", "a")
    assert call_count == 1

    for i in range(11):
        role = ParticipantRole.USER if i % 2 == 0 else ParticipantRole.ASSISTANT
        await storage.save_chat_message("u", "s", "a",
            ConversationMessage(role=role.value, content=[{"text": f"m{i}"}]))

    assert call_count == 2

    result = await storage.fetch_chat("u", "s", "a")
    # Buffer = [Summary 2] + any messages added after the second compression.
    assert result[0].content[0]["text"] == "Summary 2"


# ---------------------------------------------------------------------------
# fetch_all_chats — never intercepted
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_all_chats_returns_raw_history():
    inner = InMemoryChatStorage()
    await seed(inner, make_history(6))
    summarizer = AsyncMock(return_value=[user_msg("Summary")])
    storage = SummarizingChatStorage(inner, summarizer, trigger_at=5, keep_last=2)

    await storage.fetch_chat("u", "s", "a")
    result = await storage.fetch_all_chats("u", "s")

    assert len(result) == 12
    summarizer.assert_called_once()


# ---------------------------------------------------------------------------
# Base storage integrity
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_base_storage_always_receives_raw_messages():
    inner = InMemoryChatStorage()
    await seed(inner, make_history(6))
    summarizer = AsyncMock(return_value=[user_msg("Summary")])
    storage = SummarizingChatStorage(inner, summarizer, trigger_at=5, keep_last=2)

    await storage.fetch_chat("u", "s", "a")
    await storage.save_chat_message("u", "s", "a", user_msg("New message"))

    raw = await inner.fetch_chat("u", "s", "a")
    assert len(raw) == 13
    assert raw[-1].content[0]["text"] == "New message"


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_summarizer_error_propagates_from_fetch():
    inner = InMemoryChatStorage()
    await seed(inner, make_history(6))
    summarizer = AsyncMock(side_effect=RuntimeError("summarizer failed"))
    storage = SummarizingChatStorage(inner, summarizer, trigger_at=5, keep_last=2)

    with pytest.raises(RuntimeError, match="summarizer failed"):
        await storage.fetch_chat("u", "s", "a")


@pytest.mark.asyncio
async def test_summarizer_error_propagates_from_save():
    inner = InMemoryChatStorage()
    await seed(inner, make_history(6))

    call_count = 0

    async def summarizer(history, keep_last):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return [user_msg("Summary")]
        raise RuntimeError("summarizer failed on save")

    storage = SummarizingChatStorage(inner, summarizer, trigger_at=5, keep_last=2)
    await storage.fetch_chat("u", "s", "a")

    with pytest.raises(RuntimeError, match="summarizer failed on save"):
        for i in range(11):
            await storage.save_chat_message("u", "s", "a", user_msg(f"m{i}"))
