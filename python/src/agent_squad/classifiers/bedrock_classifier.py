import os
from typing import List, Optional, Dict, Any
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from agent_squad.utils.helpers import is_tool_input
from agent_squad.utils import Logger
from agent_squad.types import ConversationMessage, ParticipantRole, BEDROCK_MODEL_ID_CLAUDE_HAIKU_4_5
from agent_squad.classifiers import Classifier, ClassifierResult, ClassifierCallbacks
from agent_squad.shared import user_agent

class BedrockClassifierOptions:
    def __init__(
        self,
        model_id: Optional[str] = None,
        region: Optional[str] = None,
        inference_config: Optional[Dict] = None,
        client: Optional[Any] = None,
        callbacks: Optional[ClassifierCallbacks] = None
    ):
        self.model_id = model_id
        self.region = region
        self.inference_config = inference_config if inference_config is not None else {}
        self.client = client
        self.callbacks = callbacks or ClassifierCallbacks()


class BedrockClassifier(Classifier):
    def __init__(self, options: BedrockClassifierOptions):
        super().__init__()
        self.region = options.region or os.environ.get('AWS_REGION')
        if options.client:
            self.client = options.client
        else:
            self.client = boto3.client('bedrock-runtime', region_name=self.region)

        self.callbacks = options.callbacks
        user_agent.register_feature_to_client(self.client, feature="bedrock-classifier")

        self.model_id = options.model_id or BEDROCK_MODEL_ID_CLAUDE_HAIKU_4_5
        self.system_prompt: str
        # top_p isn't set by default: Claude Haiku 4.5 and newer reject
        # temperature and top_p in the same request.
        self.inference_config = {
            'maxTokens': options.inference_config.get('maxTokens', 1000),
            'temperature':  options.inference_config.get('temperature', 0.0),
            'topP': options.inference_config.get('top_p'),
            'stopSequences': options.inference_config.get('stop_sequences', [])
        }
        self.tools = [
            {
                "toolSpec": {
                    "name": "analyzePrompt",
                    "description": "Analyze the user input and provide structured output",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "userinput": {
                                    "type": "string",
                                    "description": "The original user input",
                                },
                                "selected_agent": {
                                    "type": "string",
                                    "description": "The name of the selected agent",
                                },
                                "confidence": {
                                    "type": "number",
                                    "description": "Confidence level between 0 and 1",
                                },
                            },
                            "required": ["userinput", "selected_agent", "confidence"],
                        },
                    },
                },
            },
        ]


    async def process_request(self,
                              input_text: str,
                              chat_history: List[ConversationMessage]) -> ClassifierResult:
        user_message = ConversationMessage(
            role=ParticipantRole.USER.value,
            content=[{"text": input_text}]
        )

        toolConfig = {
            "tools": self.tools,
        }

        # ToolChoice is only supported by Anthropic Claude 3 models and by Mistral AI Mistral Large.
        # https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolChoice.html
        if "anthropic" in self.model_id or 'mistral-large' in self.model_id:
            toolConfig['toolChoice'] = {
                "tool": {
                    "name": "analyzePrompt",
                },
            }

        # A value set to None is left out, for models that reject sampling
        # parameters (e.g. {"temperature": None, "top_p": None} for Claude Opus 5).
        inference_config = {key: value for key, value in self.inference_config.items() if value is not None}

        converse_cmd = {
            "modelId": self.model_id,
            # Converse rejects unknown keys, so don't send ConversationMessage.__dict__
            # (it carries a `citations` attribute).
            "messages": [{"role": user_message.role, "content": user_message.content}],
            "system": [{"text": self.system_prompt}],
            "toolConfig": toolConfig,
            "inferenceConfig": inference_config,
        }

        try:
            kwargs = {
                "modelId": self.model_id,
                "system": self.system_prompt,
                "inferenceConfig": inference_config,
            }
            await self.callbacks.on_classifier_start('on_classifier_start', input_text, **kwargs)
            response = self.client.converse(**converse_cmd)

            if not response.get('output'):
                raise ValueError("No output received from Bedrock model")

            if response['output'].get('message', {}).get('content'):
                response_content_blocks = response['output']['message']['content']

                for content_block in response_content_blocks:
                    if 'toolUse' in content_block:
                        tool_use = content_block['toolUse']
                        if not tool_use:
                            raise ValueError("No tool use found in the response")

                        if not is_tool_input(tool_use['input']):
                            raise ValueError(f"Tool input does not match expected structure: {str(tool_use)}")

                        intent_classifier_result: ClassifierResult = ClassifierResult(
                            selected_agent=self.get_agent_by_id(tool_use['input']['selected_agent']),
                            confidence=float(tool_use['input']['confidence'])
                        )
                        kwargs = {
                            "usage": response.get('usage'),
                        }
                        await self.callbacks.on_classifier_stop('on_classifier_stop', intent_classifier_result, **kwargs)
                        return intent_classifier_result

            raise ValueError("No valid tool use found in the response")

        except (BotoCoreError, ClientError) as error:
            Logger.error(f"Error processing request:{str(error)}")
            raise error
