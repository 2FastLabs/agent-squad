import json
from typing import List, Optional, Dict, Any
from openai import OpenAI
from agent_squad.utils.helpers import is_tool_input
from agent_squad.utils.logger import Logger
from agent_squad.types import ConversationMessage
from agent_squad.classifiers import Classifier, ClassifierResult

OPENAI_MODEL_ID_GPT_O_MINI = "gpt-4o-mini"

class OpenAIClassifierOptions:
    def __init__(self,
                 api_key: str,
                 model_id: Optional[str] = None,
                 inference_config: Optional[Dict[str, Any]] = None,
                 client: Optional[Any] = None):
        self.api_key = api_key
        self.model_id = model_id
        self.inference_config = inference_config or {}
        self.client = client

class OpenAIClassifier(Classifier):
    def __init__(self, options: OpenAIClassifierOptions):
        super().__init__()

        if not options.api_key:
            raise ValueError("OpenAI API key is required")

        if options.client:
            self.client = options.client
        else:
            self.client = OpenAI(api_key=options.api_key)

        self.model_id = options.model_id or OPENAI_MODEL_ID_GPT_O_MINI

        default_max_tokens = 1000
        self.inference_config = {
            'max_tokens': options.inference_config.get('max_tokens', default_max_tokens),
            'temperature': options.inference_config.get('temperature', 0.0),
            'top_p': options.inference_config.get('top_p', 0.9),
            'stop': options.inference_config.get('stop_sequences', []),
        }

        self.tools = [
            {
                'type': 'function',
                'function': {
                    'name': 'analyzePrompt',
                    'description': 'Analyze the user input and provide structured output',
                    'parameters': {
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
                },
            }
        ]

        self.system_prompt = "You are an AI assistant."  # Add your system prompt here

    async def process_request(self,
                            input_text: str,
                            chat_history: List[ConversationMessage]) -> ClassifierResult:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input_text}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=self.inference_config['max_tokens'],
                temperature=self.inference_config['temperature'],
                top_p=self.inference_config['top_p'],
                tools=self.tools,
                tool_choice={"type": "function", "function": {"name": "analyzePrompt"}}
            )

            tool_call = response.choices[0].message.tool_calls[0]

            if not tool_call or tool_call.function.name != "analyzePrompt":
                raise ValueError("No valid tool call found in the response")

            tool_input = json.loads(tool_call.function.arguments)

            if not is_tool_input(tool_input):
                raise ValueError("Tool input does not match expected structure")

            intent_classifier_result = ClassifierResult(
                selected_agent=self.get_agent_by_id(tool_input['selected_agent']),
                confidence=float(tool_input['confidence'])
            )

            return intent_classifier_result

        except Exception as error:
            Logger.error(f"Error processing request: {str(error)}")
            raise error
