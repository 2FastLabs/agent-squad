from typing import List, Dict, Optional, Any
from agent_squad.types import ConversationMessage, ParticipantRole, BEDROCK_MODEL_ID_CLAUDE_HAIKU_4_5
from agent_squad.utils import conversation_to_dict, Logger
from dataclasses import dataclass
from .agent import Agent, AgentOptions
import boto3

@dataclass
class BedrockTranslatorAgentOptions(AgentOptions):
    source_language: Optional[str] = None
    target_language: Optional[str] = None
    inference_config: Optional[Dict[str, Any]] = None
    model_id: Optional[str] = None
    region: Optional[str] = None
    client: Optional[Any] = None

class BedrockTranslatorAgent(Agent):
    def __init__(self, options: BedrockTranslatorAgentOptions):
        super().__init__(options)
        self.source_language = options.source_language
        self.target_language = options.target_language or 'English'
        self.model_id = options.model_id or BEDROCK_MODEL_ID_CLAUDE_HAIKU_4_5
        if options.client:
            self.client = options.client
        else:
            self.client = boto3.client('bedrock-runtime', region_name=options.region)

        # Default inference configuration. topP isn't set by default: Claude
        # Haiku 4.5 and newer reject temperature and topP in the same request.
        self.inference_config: Dict[str, Any] = options.inference_config or {
            'maxTokens': 1000,
            'temperature': 0.0,
            'stopSequences': []
        }

        # Define the translation tool
        self.tools = [{
            "toolSpec": {
                "name": "Translate",
                "description": "Translate text to target language",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "translation": {
                                "type": "string",
                                "description": "The translated text",
                            },
                        },
                        "required": ["translation"],
                    },
                },
            },
        }]

    async def process_request(self,
                              input_text: str,
                              user_id: str,
                              session_id: str,
                              chat_history: List[ConversationMessage],
                              additional_params: Optional[Dict[str, str]] = None) -> ConversationMessage:
        # Check if input is a number and return it as-is if true
        if input_text.isdigit():
            return ConversationMessage(
                role=ParticipantRole.ASSISTANT.value,
                content=[{"text": input_text}]
            )

        # Prepare user message
        user_message = ConversationMessage(
            role=ParticipantRole.USER.value,
            content=[{"text": f"<userinput>{input_text}</userinput>"}]
        )

        # Construct system prompt
        system_prompt = "You are a translator. Translate the text within the <userinput> tags"
        if self.source_language:
            system_prompt += f" from {self.source_language} to {self.target_language}"
        else:
            system_prompt += f" to {self.target_language}"
        system_prompt += ". Only provide the translation using the Translate tool."

        # Prepare the converse command for Bedrock
        converse_cmd = {
            "modelId": self.model_id,
            "messages": [conversation_to_dict(user_message)],
            "system": [{"text": system_prompt}],
            "toolConfig": {
                "tools": self.tools,
                "toolChoice": {
                    "tool": {
                        "name": "Translate",
                    },
                },
            },
            # A value set to None is left out, for models that reject sampling parameters.
            'inferenceConfig': {key: value for key, value in self.inference_config.items() if value is not None}
        }

        try:
            # Send request to Bedrock
            response = self.client.converse(**converse_cmd)

            if 'output' not in response:
                raise ValueError("No output received from Bedrock model")

            if response['output'].get('message', {}).get('content'):
                response_content_blocks = response['output']['message']['content']

                for content_block in response_content_blocks:
                    if "toolUse" in content_block:
                        tool_use = content_block["toolUse"]
                        if not tool_use:
                            raise ValueError("No tool use found in the response")

                        if not isinstance(tool_use.get('input'), dict) or 'translation' not in tool_use['input']:
                            raise ValueError("Tool input does not match expected structure")

                        translation = tool_use['input']['translation']
                        if not isinstance(translation, str):
                            raise ValueError("Translation is not a string")

                        # Return the translated text
                        return ConversationMessage(
                            role=ParticipantRole.ASSISTANT.value,
                            content=[{"text": translation}]
                        )

            raise ValueError("No valid tool use found in the response")
        except Exception as error:
            Logger.error(f"Error processing translation request:{str(error)}")
            raise error

    def set_source_language(self, language: Optional[str]):
        """Set the source language for translation"""
        self.source_language = language

    def set_target_language(self, language: str):
        """Set the target language for translation"""
        self.target_language = language