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
    OPENAI_MODEL_ID_GPT_O_MINI,
    TemplateVariables,
    AgentProviderType
)
from agent_squad.utils import Logger, AgentTools, AgentTool
from agent_squad.retrievers import Retriever



@dataclass
class OpenAIAgentOptions(AgentOptions):
    api_key: str = None
    model: Optional[str] = None
    streaming: Optional[bool] = None
    inference_config: Optional[dict[str, Any]] = None
    custom_system_prompt: Optional[dict[str, Any]] = None
    retriever: Optional[Retriever] = None
    client: Optional[Any] = None
    tool_config: Optional[dict[str, Any] | AgentTools] = None


class OpenAIAgent(Agent):
    def __init__(self, options: OpenAIAgentOptions):
        super().__init__(options)
        if not options.api_key:
            raise ValueError("OpenAI API key is required")

        if options.client:
            self.client = options.client
        else:
            self.client = OpenAI(api_key=options.api_key)


        self.model = options.model or OPENAI_MODEL_ID_GPT_O_MINI
        self.streaming = options.streaming or False
        self.retriever: Optional[Retriever] = options.retriever
        self.tool_config: Optional[dict[str, Any]] = options.tool_config

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
        self.default_max_recursions: int = 5

        if options.custom_system_prompt:
            self.set_system_prompt(
                options.custom_system_prompt.get('template'),
                options.custom_system_prompt.get('variables')
            )



    def is_streaming_enabled(self) -> bool:
        return self.streaming is True

    async def _prepare_system_prompt(self, input_text: str) -> str:
        """Prepare the system prompt with optional retrieval context."""

        self.update_system_prompt()
        system_prompt = self.system_prompt

        if self.retriever:
            response = await self.retriever.retrieve_and_combine_results(input_text)
            system_prompt += f"\nHere is the context to use to answer the user's question:\n{response}"

        return system_prompt

    def _prepare_conversation(
        self,
        input_text: str,
        chat_history: list[ConversationMessage],
        system_prompt: str
    ) -> list[Any]:
        """Prepare the conversation history with the new user message."""

        messages = [
            {"role": "system", "content": system_prompt},
            *[{
                "role": msg.role.lower(),
                "content": msg.content[0].get('text', '') if msg.content else ''
            } for msg in chat_history],
            {"role": "user", "content": input_text}
        ]

        return messages

    def _prepare_tool_config(self) -> dict:
        """Prepare tool configuration based on the tool type."""

        if isinstance(self.tool_config["tool"], AgentTools):
            return self.tool_config["tool"].to_openai_format()

        if isinstance(self.tool_config["tool"], list):
            return [
                    tool.to_openai_format() if isinstance(tool, AgentTool) else tool
                    for tool in self.tool_config['tool']
                ]

        raise RuntimeError("Invalid tool config")

    def _build_input(
            self,
            messages: list[Any],
            ) -> dict:
        """Build the conversation command with all necessary configurations."""
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
            request_options["stream_options"] = {
                "include_usage": True,
            }

        if self.tool_config:
            request_options["tools"] = self._prepare_tool_config()

        return request_options

    def _get_max_recursions(self) -> int:
        """Get the maximum number of recursions based on tool configuration."""
        if not self.tool_config:
            return 1
        return self.tool_config.get('toolMaxRecursions', self.default_max_recursions)

    async def _handle_streaming(
        self,
        payload_input: dict,
        messages: list[Any],
        max_recursions: int,
        agent_tracking_info: dict[str, Any] | None = None
    ) -> AsyncIterable[Any]:
        """Handle streaming response processing with tool recursion."""
        continue_with_tools = True
        final_response = None

        async def stream_generator():
            nonlocal continue_with_tools, final_response, max_recursions

            while continue_with_tools and max_recursions > 0:
                response = self.handle_streaming_response(payload_input)

                async for chunk in response:
                    if chunk.final_message:
                        final_response = chunk.final_message # do not yield the full message as it need to be converted in Conversation Message
                    else:
                        yield chunk

                if any('function' == content.get('type') for content in final_response.content):
                    payload_input['messages'].append({"role": "assistant", "content": None, "tool_calls": final_response.content})
                    tool_response = await self._process_tool_block(final_response, messages, agent_tracking_info)
                    payload_input['messages'].extend(tool_response.content)
                else:
                    continue_with_tools = False
                    # yield last message
                    kwargs = {
                        "agent_name": self.name,
                        "response": final_response,
                        "messages": messages,
                        "agent_tracking_info": agent_tracking_info
                    }
                    await self.callbacks.on_agent_end(**kwargs)

                    yield AgentStreamResponse(final_message=ConversationMessage(role=ParticipantRole.ASSISTANT.value, content=[{"text": final_response.content[0]['text']}]))

                max_recursions -= 1

        return stream_generator()

    async def _process_with_strategy(
        self,
        streaming: bool,
        request_options: dict,
        messages: list[Any],
        agent_tracking_info: dict[str, Any] | None = None
    ) -> ConversationMessage | AsyncIterable[Any]:
        """Process the request using the specified strategy."""

        max_recursions = self._get_max_recursions()

        if streaming:
            return await self._handle_streaming(request_options, messages, max_recursions, agent_tracking_info)
        response = await self._handle_single_response_loop(request_options, messages, max_recursions, agent_tracking_info)

        kwargs = {
            "agent_name": self.name,
            "response": response,
            "messages": messages,
            "agent_tracking_info": agent_tracking_info
        }
        await self.callbacks.on_agent_end(**kwargs)
        return response

    async def _process_tool_block(self, llm_response: Any, conversation: list[Any], agent_tracking_info: dict[str, Any] | None = None) -> (Any):
        if 'useToolHandler' in self.tool_config:
            # tool process logic is handled elsewhere
            tool_response = await self.tool_config['useToolHandler'](llm_response, conversation)
        else:
            # tool process logic is handled in AgentTools class
            if isinstance(self.tool_config['tool'], AgentTools):
                additional_params = {
                    "agent_name": self.name,
                    "agent_tracking_info": agent_tracking_info
                }
                tool_response = await self.tool_config['tool'].tool_handler(AgentProviderType.OPENAI.value, llm_response, conversation, additional_params)
            else:
                raise ValueError("You must use AgentTools class when not providing a custom tool handler")
        return tool_response

    async def _handle_single_response_loop(
        self,
        payload_input: Any,
        messages: list[Any],
        max_recursions: int,
        agent_tracking_info: dict[str, Any] | None = None
    ) -> ConversationMessage:
        """Handle single response processing with tool recursion."""

        continue_with_tools = True
        llm_response = None
        text_response = ''

        while continue_with_tools and max_recursions > 0:
            llm_response:ConversationMessage = await self.handle_single_response(payload_input)

            if any('function' == content.get('type') for content in llm_response.content):
                payload_input['messages'].append({"role": "assistant", "content": None, "tool_calls": llm_response.content})
                tool_response = await self._process_tool_block(llm_response, messages, agent_tracking_info)
                payload_input['messages'].extend(tool_response.content)

            else:
                continue_with_tools = False
                if llm_response.content:
                    text_response = llm_response.content[0]['text']
                else:
                    text_response = 'No final response generated'

            max_recursions -= 1

        return ConversationMessage(role=ParticipantRole.ASSISTANT.value, content=[{"text": text_response}])

    async def process_request(
        self,
        input_text: str,
        user_id: str,
        session_id: str,
        chat_history: list[ConversationMessage],
        additional_params: Optional[dict[str, str]] = None
    ) -> ConversationMessage | AsyncIterable[Any]:

        kwargs = {
            'agent_name': self.name,
            'payload_input': input_text,
            'messages': [*chat_history],
            'additional_params': additional_params,
            'user_id': user_id,
            'session_id': session_id
        }
        agent_tracking_info = await self.callbacks.on_agent_start(**kwargs)


        system_prompt = await self._prepare_system_prompt(input_text)
        messages = self._prepare_conversation(input_text, chat_history, system_prompt)
        request_options = self._build_input(messages)

        return await self._process_with_strategy(self.streaming, request_options, messages, agent_tracking_info)


    async def handle_single_response(self, request_options: dict[str, Any]) -> ConversationMessage:
        try:
            await self.callbacks.on_llm_start(self.name, payload_input=request_options.get('messages')[-1], **request_options)

            request_options['stream'] = False
            chat_completion = self.client.chat.completions.create(**request_options)

            if not chat_completion.choices:
                raise ValueError('No choices returned from OpenAI API')

            assistant_message = chat_completion.choices[0].message.content

            content = [{"text": assistant_message}]
            if not isinstance(assistant_message, str):
                if isinstance(chat_completion.choices[0].message.tool_calls, list):
                    content = [tc.to_dict() for tc in chat_completion.choices[0].message.tool_calls]

                else:
                    raise ValueError('Unexpected response format from OpenAI API')

            kwargs = {
                'usage':{
                    'inputTokens':chat_completion.usage.prompt_tokens,
                    'outputTokens':chat_completion.usage.completion_tokens,
                    'totalTokens':chat_completion.usage.total_tokens,
                },
                'input': {
                    'modelId': request_options.get('model'),
                    'messages': request_options.get('messages'),
                    'system': request_options.get('messages')[0]['content'],
                },
                'inferenceConfig':{
                    "temperature": request_options.get('temperature'),
                    "top_p": request_options.get('top_p'),
                    "stop_sequences": request_options.get('stop'),
                    "max_tokens": request_options.get('max_tokens')
                }
            }
            await self.callbacks.on_llm_end(self.name, output=content, **kwargs)

            return ConversationMessage(
                role=ParticipantRole.ASSISTANT.value,
                content=content
            )

        except Exception as error:
            Logger.error(f'Error in OpenAI API call: {str(error)}')
            raise error

    async def handle_streaming_response(self, request_options: dict[str, Any]) -> AsyncGenerator[AgentStreamResponse, None]:
        try:
            await self.callbacks.on_llm_start(self.name, payload_input=request_options.get('messages')[-1], **request_options)

            # non-streaming call of tools for simplicity 
            if request_options.get('tools'):
                request_options['stream'] = False
                chat_completion = self.client.chat.completions.create(**request_options)
                chunk = chat_completion

                if not chat_completion.choices:
                    raise ValueError('No choices returned from OpenAI API')

                assistant_message = chat_completion.choices[0].message.content

                content = [{"text": assistant_message}]
                if not isinstance(assistant_message, str):
                    if isinstance(chat_completion.choices[0].message.tool_calls, list):
                        content = [tc.to_dict() for tc in chat_completion.choices[0].message.tool_calls]

                    else:
                        raise ValueError('Unexpected response format from OpenAI API')

                await self.callbacks.on_llm_new_token(assistant_message)
                yield AgentStreamResponse(text=assistant_message)

            else:
                stream = self.client.chat.completions.create(**request_options)
                accumulated_message = []

                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        chunk_content = chunk.choices[0].delta.content
                        accumulated_message.append(chunk_content)
                        await self.callbacks.on_llm_new_token(chunk_content)
                        yield AgentStreamResponse(text=chunk_content)

                # Store the complete message in the instance for later access if needed
                final_content = ''.join(accumulated_message)
                content = [{"text": final_content}]

            yield AgentStreamResponse(final_message=ConversationMessage(
                role=ParticipantRole.ASSISTANT.value,
                content=content
            ))

            kwargs = {
                'usage':{
                    'inputTokens':chunk.usage.prompt_tokens,
                    'outputTokens':chunk.usage.completion_tokens,
                    'totalTokens':chunk.usage.total_tokens,
                },
                'input': {
                    'modelId': request_options.get('model'),
                    'messages': request_options.get('messages'),
                    'system': request_options.get('messages')[0]['content'],
                },
                'inferenceConfig':{
                    "temperature": request_options.get('temperature'),
                    "top_p": request_options.get('top_p'),
                    "stop_sequences": request_options.get('stop'),
                    "max_tokens": request_options.get('max_tokens')
                }
            }
            await self.callbacks.on_llm_end(self.name, output=content, **kwargs)

        except Exception as error:
            Logger.error(f"Error getting stream from OpenAI model: {str(error)}")
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
