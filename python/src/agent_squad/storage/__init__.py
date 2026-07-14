"""
Storage implementations for chat history.
"""
from .chat_storage import ChatStorage
from .in_memory_chat_storage import InMemoryChatStorage
from .summarizing_chat_storage import SummarizingChatStorage

_AWS_AVAILABLE = False
_SQL_AVAILABLE = False

try:
    from .dynamodb_chat_storage import DynamoDbChatStorage
    _AWS_AVAILABLE = True
except ImportError:
    _AWS_AVAILABLE = False

try:
    from .sql_chat_storage import SqlChatStorage
    _SQL_AVAILABLE = True
except ImportError:
    _SQL_AVAILABLE = False

__all__ = [
    'ChatStorage',
    'InMemoryChatStorage',
    'SummarizingChatStorage',
]

if _AWS_AVAILABLE:
    __all__.extend([
        'DynamoDbChatStorage'
    ])

if _SQL_AVAILABLE:
    __all__.extend([
        'SqlChatStorage'
    ])