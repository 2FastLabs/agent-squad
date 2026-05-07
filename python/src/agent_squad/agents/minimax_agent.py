from typing import AsyncIterable, Optional, Any, AsyncGenerator
from dataclasses import dataclass
from openai import OpenAI
from agent_squad.agents import (
    Agent,
    AgentOptions,
    AgentStreamResponse
)
from agent_squad.types import (
    ConversationMessage,
    ParticipantRole,
    MINIMAX_MODEL_ID_M2_7,
    TemplateVariables
)
from agent_squad.utils import Logger
from agent_squad.retrievers import Retriever


MINIMAX_API_BASE_URL = "https://api.minimax.io/v1"


@dataclass
class MiniMaxAgentOptions(AgentOptions):
    api_key: str = None
    model: Optional[str] = None
    streaming: Optional[bool] = None
    inference_config: Optional[dict[str, Any]] = None
    custom_system_prompt: Optional[dict[str, Any]] = None
    retriever: Optional[Retriever] = None
    client: Optional[Any] = None
    base_url: Optional[str] = None


class MiniMaxAgent(Agent):
    def __init__(self, options: MiniMaxAgentOptions):
        super().__init__(options)
        if not options.api_key:
            raise ValueError("MiniMax API key is required")

        if options.client:
            self.client = options.client
        else:
            self.client = OpenAI(
                api_key=options.api_key,
                base_url=options.base_url or MINIMAX_API_BASE_URL
            )

        self.model = options.model or MINIMAX_MODEL_ID_M2_7
        self.streaming = options.streaming or False
        self.retriever: Optional[Retriever] = options.retriever

        # Default inference configuration
        default_inference_config = {
            'maxTokens': 1000,
            'temperature': None,
            'topP': None,
            'stopSequences': None
        }

        if options.inference_config:
            self.inference_config = {**default_inference_config, **options.inference_config}
        else:
            self.inference_config = default_inference_config

        # Clamp temperature for MiniMax API (must be > 0 when set)
        temp = self.inference_config.get('temperature')
        if temp is not None and temp <= 0:
            self.inference_config['temperature'] = 0.01

        # Initialize system prompt
        self.prompt_template = f"""You are a {self.name}.
        {self.description} Provide helpful and accurate information based on your expertise.
        You will engage in an open-ended conversation, providing helpful and accurate information based on your expertise.
        The conversation will proceed as follows:
        - The human may ask an initial question or provide a prompt on any topic.
        - You will provide a relevant and informative response.
        - The human may then follow up with additional questions or prompts related to your previous response,
          allowing for a multi-turn dialogue on that topic.
        - Or, the human may switch to a completely new and unrelated topic at any point.
        - You will seamlessly shift your focus to the new topic, providing thoughtful and coherent responses
          based on your broad knowledge base.
        Throughout the conversation, you should aim to:
        - Understand the context and intent behind each new question or prompt.
        - Provide substantive and well-reasoned responses that directly address the query.
        - Draw insights and connections from your extensive knowledge when appropriate.
        - Ask for clarification if any part of the question or prompt is ambiguous.
        - Maintain a consistent, respectful, and engaging tone tailored to the human's communication style.
        - Seamlessly transition between topics as the human introduces new subjects."""

        self.system_prompt = ""
        self.custom_variables: TemplateVariables = {}

        if options.custom_system_prompt:
            self.set_system_prompt(
                options.custom_system_prompt.get('template'),
                options.custom_system_prompt.get('variables')
            )

    def is_streaming_enabled(self) -> bool:
        return self.streaming is True

    async def process_request(
        self,
        input_text: str,
        user_id: str,
        session_id: str,
        chat_history: list[ConversationMessage],
        additional_params: Optional[dict[str, str]] = None
    ) -> ConversationMessage | AsyncIterable[Any]:
        try:
            self.update_system_prompt()

            system_prompt = self.system_prompt

            if self.retriever:
                response = await self.retriever.retrieve_and_combine_results(input_text)
                context_prompt = "\nHere is the context to use to answer the user's question:\n" + response
                system_prompt += context_prompt

            messages = [
                {"role": "system", "content": system_prompt},
                *[{
                    "role": msg.role.lower(),
                    "content": msg.content[0].get('text', '') if msg.content else ''
                } for msg in chat_history],
                {"role": "user", "content": input_text}
            ]

            request_options = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.inference_config.get('maxTokens'),
                "temperature": self.inference_config.get('temperature'),
                "top_p": self.inference_config.get('topP'),
                "stop": self.inference_config.get('stopSequences'),
                "stream": self.streaming
            }
            if self.streaming:
                return self.handle_streaming_response(request_options)
            else:
                return await self.handle_single_response(request_options)

        except Exception as error:
            Logger.error(f"Error in MiniMax API call: {str(error)}")
            raise error

    async def handle_single_response(self, request_options: dict[str, Any]) -> ConversationMessage:
        try:
            request_options['stream'] = False
            chat_completion = self.client.chat.completions.create(**request_options)

            if not chat_completion.choices:
                raise ValueError('No choices returned from MiniMax API')

            assistant_message = chat_completion.choices[0].message.content

            if not isinstance(assistant_message, str):
                raise ValueError('Unexpected response format from MiniMax API')

            return ConversationMessage(
                role=ParticipantRole.ASSISTANT.value,
                content=[{"text": assistant_message}]
            )

        except Exception as error:
            Logger.error(f'Error in MiniMax API call: {str(error)}')
            raise error

    async def handle_streaming_response(self, request_options: dict[str, Any]) -> AsyncGenerator[AgentStreamResponse, None]:
        try:
            stream = self.client.chat.completions.create(**request_options)
            accumulated_message = []

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    chunk_content = chunk.choices[0].delta.content
                    accumulated_message.append(chunk_content)
                    await self.callbacks.on_llm_new_token(chunk_content)
                    yield AgentStreamResponse(text=chunk_content)

            # Store the complete message in the instance for later access if needed
            yield AgentStreamResponse(final_message=ConversationMessage(
                role=ParticipantRole.ASSISTANT.value,
                content=[{"text": ''.join(accumulated_message)}]
            ))

        except Exception as error:
            Logger.error(f"Error getting stream from MiniMax model: {str(error)}")
            raise error

    def set_system_prompt(self,
                         template: Optional[str] = None,
                         variables: Optional[TemplateVariables] = None) -> None:
        if template:
            self.prompt_template = template
        if variables:
            self.custom_variables = variables
        self.update_system_prompt()

    def update_system_prompt(self) -> None:
        all_variables: TemplateVariables = {**self.custom_variables}
        self.system_prompt = self.replace_placeholders(self.prompt_template, all_variables)

    @staticmethod
    def replace_placeholders(template: str, variables: TemplateVariables) -> str:
        import re
        def replace(match):
            key = match.group(1)
            if key in variables:
                value = variables[key]
                return '\n'.join(value) if isinstance(value, list) else str(value)
            return match.group(0)

        return re.sub(r'{{(\w+)}}', replace, template)
