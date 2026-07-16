from typing import List, Optional, Any
from anthropic import Anthropic
from anthropic.types import Message
from agent_squad.utils.helpers import is_tool_input
from agent_squad.utils.logger import Logger
from agent_squad.types import ConversationMessage
from agent_squad.classifiers import Classifier, ClassifierResult, ClassifierCallbacks

ANTHROPIC_MODEL_ID_CLAUDE_SONNET_4 = "claude-sonnet-4-20250514"

class AnthropicClassifierOptions:
    def __init__(self,
                 api_key: str,
                 model_id: Optional[str] = None,
                 inference_config: Optional[dict[str, Any]] = None,
                callbacks: Optional[ClassifierCallbacks] = None
                 ):
        self.api_key = api_key
        self.model_id = model_id
        self.inference_config = inference_config or {}
        self.callbacks = callbacks or ClassifierCallbacks()

class AnthropicClassifier(Classifier):
    def __init__(self, options: AnthropicClassifierOptions):
        super().__init__()

        if not options.api_key:
            raise ValueError("Anthropic API key is required")

        self.client = Anthropic(api_key=options.api_key)
        self.model_id = options.model_id or ANTHROPIC_MODEL_ID_CLAUDE_SONNET_4

        self.callbacks = options.callbacks

        default_max_tokens = 1000
        self.inference_config = {
            'max_tokens': options.inference_config.get('max_tokens', default_max_tokens),
            'temperature': options.inference_config.get('temperature', 0.0),
            'stop_sequences': options.inference_config.get('stop_sequences', []),
        }
        # Only pass top_p if explicitly set — newer Anthropic models reject both temperature and top_p
        if 'top_p' in options.inference_config:
            self.inference_config['top_p'] = options.inference_config['top_p']

        self.tools: List[dict] = [
            {
                'name': 'analyzePrompt',
                'description': 'Analyze the user input and provide structured output',
                'input_schema': {
                    'type': 'object',
                    'properties': {
                        'userinput': {
                            'type': 'string',
                            'description': 'The original user input',
                        },
                        'selected_agent': {
                            'type': 'string',
                            'description': 'The name of the selected agent',
                        },
                        'confidence': {
                            'type': 'number',
                            'description': 'Confidence level between 0 and 1',
                        },
                    },
                    'required': ['userinput', 'selected_agent', 'confidence'],
                },
            }
        ]

        self.system_prompt = "You are an AI assistant."  # Add your system prompt here


    async def process_request(self,
                              input_text: str,
                              chat_history: List[ConversationMessage]) -> ClassifierResult:
        user_message = {"role": "user", "content": input_text}

        try:

            kwargs = {
                "modelId": self.model_id,
                "system": self.system_prompt,
                "inferenceConfig": {
                    "maxTokens": self.inference_config['max_tokens'],
                    "temperature": self.inference_config['temperature'],
                    "stopSequences": self.inference_config['stop_sequences'],
                },
            }
            if 'top_p' in self.inference_config:
                kwargs['inferenceConfig']['topP'] = self.inference_config['top_p']

            await self.callbacks.on_classifier_start('on_classifier_start', input_text, **kwargs)

            call_params = {
                "model": self.model_id,
                "max_tokens": self.inference_config['max_tokens'],
                "messages": [user_message],
                "system": self.system_prompt,
                "temperature": self.inference_config['temperature'],
                "tools": self.tools,
            }
            if 'top_p' in self.inference_config:
                call_params['top_p'] = self.inference_config['top_p']

            response:Message = self.client.messages.create(**call_params)

            tool_use = next((c for c in response.content if c.type == "tool_use"), None)

            if not tool_use:
                raise ValueError("No tool use found in the response")

            if not is_tool_input(tool_use.input):
                raise ValueError("Tool input does not match expected structure")

            intent_classifier_result = ClassifierResult(
                selected_agent=self.get_agent_by_id(tool_use.input['selected_agent']),
                confidence=float(tool_use.input['confidence'])
            )

            kwargs = {
                "usage": {
                    'inputTokens':response.usage.input_tokens,
                    'outputTokens':response.usage.output_tokens,
                    'totalTokens':response.usage.input_tokens + response.usage.output_tokens
                },
            }

            await self.callbacks.on_classifier_stop('on_classifier_stop', intent_classifier_result, **kwargs)

            return intent_classifier_result

        except Exception as error:
            Logger.error(f"Error processing request:{str(error)}")
            raise error
