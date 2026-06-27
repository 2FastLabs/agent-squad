from enum import Enum
from typing import TypedDict, Optional, Any
from dataclasses import dataclass
import time

# Constants
BEDROCK_MODEL_ID_CLAUDE_3_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"
BEDROCK_MODEL_ID_CLAUDE_3_SONNET = "anthropic.claude-3-sonnet-20240229-v1:0"
BEDROCK_MODEL_ID_CLAUDE_3_5_SONNET = "anthropic.claude-3-5-sonnet-20240620-v1:0"
BEDROCK_MODEL_ID_CLAUDE_3_7_SONNET = "anthropic.claude-3-7-sonnet-20250219-v1:0"
BEDROCK_MODEL_ID_LLAMA_3_70B = "meta.llama3-70b-instruct-v1:0"
OPENAI_MODEL_ID_GPT_O_MINI = "gpt-4o-mini"
ANTHROPIC_MODEL_ID_CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20240620"

class AgentProviderType(Enum):
    BEDROCK = "BEDROCK"
    ANTHROPIC = "ANTHROPIC"
    OPENAI = "OPENAI"

class AgentTypes(Enum):
    DEFAULT = "Common Knowledge"
    CLASSIFIER = "classifier"

class ToolInput(TypedDict):
    userinput: str
    selected_agent: str
    confidence: str

class RequestMetadata(TypedDict):
    user_input: str
    agent_id: str
    agent_name: str
    user_id: str
    session_id: str
    additional_params :Optional[dict[str, str]]
    error_type: Optional[str]


class ParticipantRole(Enum):
    ASSISTANT = "assistant"
    USER = "user"
    TOOL = "tool"

class ConversationMessage:
    role: ParticipantRole
    content: list[Any]

    def __init__(self, role: ParticipantRole, content: Optional[list[Any]] = None):
        self.role = role
        self.content = content

class TimestampedMessage(ConversationMessage):
    def __init__(self,
                 role: ParticipantRole,
                 content: Optional[list[Any]] = None,
                 timestamp: int = 0):
        super().__init__(role, content)  # Call the parent constructor
        self.timestamp = timestamp or int(time.time() * 1000)      # Initialize the timestamp attribute (in ms)

TemplateVariables = dict[str, str | list[str]]

@dataclass
class AgentSquadConfig:
    LOG_AGENT_CHAT: bool = False    # pylint: disable=invalid-name
    LOG_CLASSIFIER_CHAT: bool = False   # pylint: disable=invalid-name
    LOG_CLASSIFIER_RAW_OUTPUT: bool = False # pylint: disable=invalid-name
    LOG_CLASSIFIER_OUTPUT: bool = False # pylint: disable=invalid-name
    LOG_EXECUTION_TIMES: bool = False   # pylint: disable=invalid-name
    MAX_RETRIES: int = 3    # pylint: disable=invalid-name
    USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED: bool = True   # pylint: disable=invalid-name
    CLASSIFICATION_ERROR_MESSAGE: str = None
    NO_SELECTED_AGENT_MESSAGE: str = "I'm sorry, I couldn't determine how to handle your request.\
    Could you please rephrase it?"  # pylint: disable=invalid-name
    GENERAL_ROUTING_ERROR_MSG_MESSAGE: str = None
    MAX_MESSAGE_PAIRS_PER_AGENT: int = 100  # pylint: disable=invalid-name
