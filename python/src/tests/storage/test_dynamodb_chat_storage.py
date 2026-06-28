import pytest
from moto import mock_aws
import boto3
from decimal import Decimal
from agent_squad.types import ConversationMessage, ParticipantRole, TimestampedMessage
from agent_squad.storage import DynamoDbChatStorage

@pytest.fixture
def dynamodb_table():
    with mock_aws():
        dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
        table = dynamodb.create_table(
            TableName='test_table',
            KeySchema=[
                {'AttributeName': 'PK', 'KeyType': 'HASH'},
                {'AttributeName': 'SK', 'KeyType': 'RANGE'}
            ],
            AttributeDefinitions=[
                {'AttributeName': 'PK', 'AttributeType': 'S'},
                {'AttributeName': 'SK', 'AttributeType': 'S'}
            ],
            BillingMode='PAY_PER_REQUEST'
        )
        yield table

@pytest.fixture
def chat_storage(dynamodb_table):
    return DynamoDbChatStorage(table_name='test_table', region='us-east-1', ttl_key='TTL', ttl_duration=3600)

@pytest.fixture
def filtered_chat_storage(dynamodb_table):
    return DynamoDbChatStorage(
        table_name='test_table',
        region='us-east-1',
        content_filters=[('acme-corp', '#COMPANY#'), ('secret-token-xyz', '#TOKEN#')]
    )

@pytest.mark.asyncio
async def test_save_and_fetch_chat_message(chat_storage):
    user_id = 'user1'
    session_id = 'session1'
    agent_id = 'agent1'
    message = ConversationMessage(role=ParticipantRole.USER.value, content=[{'text': 'Hello'}])

    # Save message
    saved_messages = await chat_storage.save_chat_message(user_id, session_id, agent_id, message)
    assert len(saved_messages) == 1
    assert saved_messages[0].role == ParticipantRole.USER.value
    assert saved_messages[0].content == [{'text': 'Hello'}]

    # Fetch message
    fetched_messages = await chat_storage.fetch_chat(user_id, session_id, agent_id)
    assert len(fetched_messages) == 1
    assert fetched_messages[0].role == ParticipantRole.USER.value
    assert fetched_messages[0].content == [{'text': 'Hello'}]

@pytest.mark.asyncio
async def test_fetch_chat_with_timestamp(chat_storage):
    user_id = 'user1'
    session_id = 'session1'
    agent_id = 'agent1'
    message = ConversationMessage(role=ParticipantRole.USER.value, content=[{'text': 'Hello'}])

    await chat_storage.save_chat_message(user_id, session_id, agent_id, message)

    fetched_messages = await chat_storage.fetch_chat_with_timestamp(user_id, session_id, agent_id)
    assert len(fetched_messages) == 1
    assert isinstance(fetched_messages[0], TimestampedMessage)
    assert fetched_messages[0].role == ParticipantRole.USER.value
    assert fetched_messages[0].content == [{'text': 'Hello'}]
    assert isinstance(fetched_messages[0].timestamp, Decimal)

@pytest.mark.asyncio
async def test_fetch_all_chats(chat_storage):
    user_id = 'user1'
    session_id = 'session1'

    for i in range(5):
        message = ConversationMessage(role=ParticipantRole.USER.value, content=[{'text': f'Message {i}'}])
        await chat_storage.save_chat_message(user_id, session_id, 'agent_id', message)
        message = ConversationMessage(role=ParticipantRole.ASSISTANT.value, content=[{'text': f'Message {i}'}])
        await chat_storage.save_chat_message(user_id, session_id, 'agent_id', message)

    all_chats = await chat_storage.fetch_all_chats(user_id, session_id)
    assert len(all_chats) == 10
    for i in range(5):
        assert (all_chats[i*2].content[0]['text'] == f'Message {i}')
        assert (all_chats[(i*2)+1].content[0]['text'] == f'[agent_id] Message {i}')

@pytest.mark.asyncio
async def test_consecutive_message_handling(chat_storage):
    user_id = 'user1'
    session_id = 'session1'
    agent_id = 'agent1'
    message1 = ConversationMessage(role=ParticipantRole.USER.value, content=[{'text': 'Hello'}])
    message2 = ConversationMessage(role=ParticipantRole.USER.value, content=[{'text': 'World'}])

    await chat_storage.save_chat_message(user_id, session_id, agent_id, message1)
    await chat_storage.save_chat_message(user_id, session_id, agent_id, message2)

    fetched_messages = await chat_storage.fetch_chat(user_id, session_id, agent_id)
    assert len(fetched_messages) == 1
    assert fetched_messages[0].content == [{'text': 'Hello'}]

@pytest.mark.asyncio
async def test_trim_conversation(chat_storage):
    user_id = 'user1'
    session_id = 'session1'
    agent_id = 'agent1'

    for i in range(5):
        message = ConversationMessage(role=ParticipantRole.USER.value, content=[{'text': f'Message {i}'}])
        await chat_storage.save_chat_message(user_id, session_id, agent_id, message, max_history_size=3)
        message = ConversationMessage(role=ParticipantRole.ASSISTANT.value, content=[{'text': f'Message {i}'}])
        await chat_storage.save_chat_message(user_id, session_id, agent_id, message, max_history_size=3)

    fetched_messages = await chat_storage.fetch_chat(user_id, session_id, agent_id)
    assert len(fetched_messages) == 2
    assert fetched_messages[0].content == [{'text': 'Message 4'}]
    assert fetched_messages[0].role == ParticipantRole.USER.value
    assert fetched_messages[1].content == [{'text': 'Message 4'}]
    assert fetched_messages[1].role == ParticipantRole.ASSISTANT.value

@pytest.mark.asyncio
async def test_save_and_fetch_chat_messages(chat_storage):
    """
    Testing saving multiple ConversationMessage at once
    """
    messages = []
    for i in range(5):
        if i % 2 == 0:
            message = ConversationMessage(role=ParticipantRole.USER.value, content=[{'text': f'Message {i}'}])
        else:
            message = ConversationMessage(role=ParticipantRole.ASSISTANT.value, content=[{'text': f'Message {i}'}])
        messages.append(message)

    await chat_storage.save_chat_messages('user1', 'session1', 'agent1', messages)
    user_id = 'user1'
    session_id = 'session1'
    agent_id = 'agent1'
    fetched_messages = await chat_storage.fetch_chat(user_id, session_id, agent_id)
    assert len(fetched_messages) == 5
    assert fetched_messages[0].content == [{'text': 'Message 0'}]
    assert fetched_messages[0].role == ParticipantRole.USER.value
    assert fetched_messages[1].content == [{'text': 'Message 1'}]
    assert fetched_messages[1].role == ParticipantRole.ASSISTANT.value
    assert fetched_messages[2].content == [{'text': 'Message 2'}]
    assert fetched_messages[2].role == ParticipantRole.USER.value
    assert fetched_messages[3].content == [{'text': 'Message 3'}]
    assert fetched_messages[3].role == ParticipantRole.ASSISTANT.value
    assert fetched_messages[4].content == [{'text': 'Message 4'}]
    assert fetched_messages[4].role == ParticipantRole.USER.value


@pytest.mark.asyncio
async def test_content_filters_replace_on_save_and_restore_on_fetch(filtered_chat_storage):
    """Sensitive values should be stored as placeholders and restored on fetch."""
    user_id = 'user1'
    session_id = 'session1'
    agent_id = 'agent1'
    message = ConversationMessage(
        role=ParticipantRole.USER.value,
        content=[{'text': 'Hello from acme-corp'}]
    )

    await filtered_chat_storage.save_chat_message(user_id, session_id, agent_id, message)

    # Verify placeholder is stored in DynamoDB
    raw = filtered_chat_storage.table.get_item(
        Key={'PK': user_id, 'SK': filtered_chat_storage._generate_key(user_id, session_id, agent_id)}
    )
    stored_text = raw['Item']['conversation'][0]['content'][0]['text']
    assert stored_text == 'Hello from #COMPANY#'

    # Verify original value is returned by fetch_chat
    fetched = await filtered_chat_storage.fetch_chat(user_id, session_id, agent_id)
    assert fetched[0].content == [{'text': 'Hello from acme-corp'}]


@pytest.mark.asyncio
async def test_content_filters_fetch_chat_with_timestamp(filtered_chat_storage):
    """fetch_chat_with_timestamp must also restore original values."""
    user_id = 'user1'
    session_id = 'session1'
    agent_id = 'agent1'
    message = ConversationMessage(
        role=ParticipantRole.USER.value,
        content=[{'text': 'token: secret-token-xyz'}]
    )

    await filtered_chat_storage.save_chat_message(user_id, session_id, agent_id, message)

    fetched = await filtered_chat_storage.fetch_chat_with_timestamp(user_id, session_id, agent_id)
    assert fetched[0].content == [{'text': 'token: secret-token-xyz'}]
    assert isinstance(fetched[0].timestamp, Decimal)


@pytest.mark.asyncio
async def test_content_filters_multiple_filters_round_trip(filtered_chat_storage):
    """Multiple filters should all be applied and reversed correctly."""
    user_id = 'user1'
    session_id = 'session1'
    agent_id = 'agent1'

    msg1 = ConversationMessage(
        role=ParticipantRole.USER.value,
        content=[{'text': 'Company acme-corp uses secret-token-xyz'}]
    )
    msg2 = ConversationMessage(
        role=ParticipantRole.ASSISTANT.value,
        content=[{'text': 'Confirmed for acme-corp'}]
    )

    await filtered_chat_storage.save_chat_message(user_id, session_id, agent_id, msg1)
    await filtered_chat_storage.save_chat_message(user_id, session_id, agent_id, msg2)

    fetched = await filtered_chat_storage.fetch_chat(user_id, session_id, agent_id)
    assert fetched[0].content == [{'text': 'Company acme-corp uses secret-token-xyz'}]
    assert fetched[1].content == [{'text': 'Confirmed for acme-corp'}]


@pytest.mark.asyncio
async def test_content_filters_save_chat_messages(filtered_chat_storage):
    """save_chat_messages should also apply content filters."""
    messages = [
        ConversationMessage(role=ParticipantRole.USER.value, content=[{'text': 'acme-corp question'}]),
        ConversationMessage(role=ParticipantRole.ASSISTANT.value, content=[{'text': 'acme-corp answer'}]),
    ]

    await filtered_chat_storage.save_chat_messages('user1', 'session1', 'agent1', messages)
    fetched = await filtered_chat_storage.fetch_chat('user1', 'session1', 'agent1')

    assert fetched[0].content == [{'text': 'acme-corp question'}]
    assert fetched[1].content == [{'text': 'acme-corp answer'}]


@pytest.mark.asyncio
async def test_no_content_filters_behaviour_unchanged(chat_storage):
    """Storage without content_filters must behave identically to the original."""
    user_id = 'user1'
    session_id = 'session1'
    agent_id = 'agent1'
    message = ConversationMessage(role=ParticipantRole.USER.value, content=[{'text': 'plain text'}])

    saved = await chat_storage.save_chat_message(user_id, session_id, agent_id, message)
    assert saved[0].content == [{'text': 'plain text'}]

    fetched = await chat_storage.fetch_chat(user_id, session_id, agent_id)
    assert fetched[0].content == [{'text': 'plain text'}]


@pytest.mark.asyncio
async def test_save_and_fetch_chat_messages_timestamp(chat_storage):
    """
    Testing saving multiple ConversationMessage at once
    """
    messages = []
    for i in range(5):
        if i % 2 == 0:
            message = TimestampedMessage(role=ParticipantRole.USER.value, content=[{'text': f'Message {i}'}])
        else:
            message = TimestampedMessage(role=ParticipantRole.ASSISTANT.value, content=[{'text': f'Message {i}'}])
        messages.append(message)

    await chat_storage.save_chat_messages('user1', 'session1', 'agent1', messages)
    user_id = 'user1'
    session_id = 'session1'
    agent_id = 'agent1'
    fetched_messages = await chat_storage.fetch_chat(user_id, session_id, agent_id)
    assert len(fetched_messages) == 5
    assert fetched_messages[0].content == [{'text': 'Message 0'}]
    assert fetched_messages[0].role == ParticipantRole.USER.value
    assert fetched_messages[1].content == [{'text': 'Message 1'}]
    assert fetched_messages[1].role == ParticipantRole.ASSISTANT.value
    assert fetched_messages[2].content == [{'text': 'Message 2'}]
    assert fetched_messages[2].role == ParticipantRole.USER.value
    assert fetched_messages[3].content == [{'text': 'Message 3'}]
    assert fetched_messages[3].role == ParticipantRole.ASSISTANT.value
    assert fetched_messages[4].content == [{'text': 'Message 4'}]
    assert fetched_messages[4].role == ParticipantRole.USER.value