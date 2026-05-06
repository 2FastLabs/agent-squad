from typing import AsyncIterable, Optional, Any, AsyncGenerator
from dataclasses import dataclass
import re
from anthropic import AsyncAnthropic, Anthropic
from anthropic.types import Message
from agent_squad.agents import Agent, AgentOptions, AgentStreamResponse
from agent_squad.types import ConversationMessage, ParticipantRole, TemplateVariables, AgentProviderType
from agent_squad.utils import Logger, AgentTools, AgentTool
from agent_squad.retrievers import Retriever


@dataclass
class AnthropicAgentOptions(AgentOptions):
    """
    Configuration options for the Anthropic agent.

    Attributes:
        api_key: Anthropic API key.
        client: Optional pre-configured Anthropic client instance.
        model_id: The Anthropic model ID to use.
        streaming: Whether to enable streaming responses.
        inference_config: Configuration for the model inference.
        retriever: Optional retriever for context augmentation.
        tool_config: Configuration for tools.
        custom_system_prompt: Custom system prompt configuration.
        additional_model_request_fields: Additional fields to include in the model request.
            Use this for model-specific parameters like "thinking".
    """
    api_key: Optional[str] = None
    client: Optional[Any] = None
    model_id: str = "claude-sonnet-4-20250514"
    streaming: Optional[bool] = False
    inference_config: Optional[dict[str, Any]] = None
    retriever: Optional[Retriever] = None
    tool_config: Optional[dict[str, Any] | AgentTools] = None
    custom_system_prompt: Optional[dict[str, Any]] = None
    additional_model_request_fields: Optional[dict[str, Any]] = None


class AnthropicAgent(Agent):
    def __init__(self, options: AnthropicAgentOptions):
        super().__init__(options)

        if not options.api_key and not options.client:
            raise ValueError("Anthropic API key or Anthropic client is required")

        self.streaming = options.streaming

        if options.client:
            if self.streaming:
                if not isinstance(options.client, AsyncAnthropic):
                    raise ValueError("If streaming is enabled, the provided client must be an AsyncAnthropic client")
            elif not isinstance(options.client, Anthropic):
                raise ValueError("If streaming is disabled, the provided client must be an Anthropic client")
            self.client = options.client
        elif self.streaming:
            self.client = AsyncAnthropic(api_key=options.api_key)
        else:
            self.client = Anthropic(api_key=options.api_key)

        self.system_prompt = ""
        self.custom_variables = {}

        self.default_max_recursions: int = 5

        self.model_id = options.model_id

        default_inference_config = {"maxTokens": 1000, "temperature": 0.1, "stopSequences": []}

        if options.inference_config:
            self.inference_config = {**default_inference_config, **options.inference_config}
        else:
            self.inference_config = default_inference_config

        # Initialize additional_model_request_fields
        self.additional_model_request_fields: Optional[dict[str, Any]] = options.additional_model_request_fields or {}

        self.retriever = options.retriever
        self.tool_config: Optional[dict[str, Any]] = options.tool_config

        self.prompt_template: str = f"""You are a {self.name}.
        {self.description}
        Provide helpful and accurate information based on your expertise.
        You will engage in an open-ended conversation,
        providing helpful and accurate information based on your expertise.
        The conversation will proceed as follows:
        - The human may ask an initial question or provide a prompt on any topic.
        - You will provide a relevant and informative response.
        - The human may then follow up with additional questions or prompts related to your previous
        response, allowing for a multi-turn dialogue on that topic.
        - Or, the human may switch to a completely new and unrelated topic at any point.
        - You will seamlessly shift your focus to the new topic, providing thoughtful and
        coherent responses based on your broad knowledge base.
        Throughout the conversation, you should aim to:
        - Understand the context and intent behind each new question or prompt.
        - Provide substantive and well-reasoned responses that directly address the query.
        - Draw insights and connections from your extensive knowledge when appropriate.
        - Ask for clarification if any part of the question or prompt is ambiguous.
        - Maintain a consistent, respectful, and engaging tone tailored
        to the human's communication style.
        - Seamlessly transition between topics as the human introduces new subjects."""

        if options.custom_system_prompt:
            self.set_system_prompt(
                options.custom_system_prompt.get("template"), options.custom_system_prompt.get("variables")
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

    def _prepare_conversation(self, input_text: str, chat_history: list[ConversationMessage]) -> list[Any]:
        """Prepare the conversation history with the new user message."""

        messages = [
            {
                "role": "user" if msg.role == ParticipantRole.USER.value else "assistant",
                "content": msg.content[0]["text"] if msg.content else "",
            }
            for msg in chat_history
        ]
        messages.append({"role": "user", "content": input_text})

        return messages

    def _prepare_tool_config(self) -> dict:
        """Prepare tool configuration based on the tool type."""

        if isinstance(self.tool_config["tool"], AgentTools):
            return self.tool_config["tool"].to_claude_format()

        if isinstance(self.tool_config["tool"], list):
            return [
                tool.to_claude_format() if isinstance(tool, AgentTool) else tool for tool in self.tool_config["tool"]
            ]

        raise RuntimeError("Invalid tool config")

    def _build_input(self, messages: list[Any], system_prompt: str) -> dict:
        """
        Build the conversation command with all necessary configurations.

        This method constructs the input dictionary for the Anthropic API call, including:
        - Core parameters (model, tokens, temperature, etc.)
        - Additional model request fields from options.additional_model_request_fields
        - Tool configuration if provided

        Returns:
            dict: The complete input configuration for the API call
        """
        json_input = {
            "model": self.model_id,
            "max_tokens": self.inference_config.get("maxTokens"),
            "messages": messages,
            "system": system_prompt,
            "temperature": self.inference_config.get("temperature"),
            "stop_sequences": self.inference_config.get("stopSequences"),
        }

        # Only pass top_p if explicitly set — newer Anthropic models reject both temperature and top_p
        top_p = self.inference_config.get("topP")
        if top_p is not None:
            json_input["top_p"] = top_p

        # Add any additional model request fields
        if self.additional_model_request_fields:
            for key, value in self.additional_model_request_fields.items():
                json_input[key] = value

        if self.tool_config:
            json_input["tools"] = self._prepare_tool_config()

        return json_input

    def _get_max_recursions(self) -> int:
        """Get the maximum number of recursions based on tool configuration."""
        if not self.tool_config:
            return 1
        return self.tool_config.get("toolMaxRecursions", self.default_max_recursions)

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
        accumulated_thinking = ""

        async def stream_generator():
            nonlocal continue_with_tools, final_response, max_recursions, accumulated_thinking

            while continue_with_tools and max_recursions > 0:
                response = self.handle_streaming_response(payload_input)

                async for chunk in response:
                    if chunk.final_message:
                        final_response = chunk.final_message
                        # Capture final thinking if available
                        if chunk.final_thinking:
                            accumulated_thinking = chunk.final_thinking
                    else:
                        # Accumulate thinking if present in non-final chunks
                        if chunk.thinking:
                            accumulated_thinking += chunk.thinking
                        yield chunk

                if final_response and any(hasattr(content, 'type') and content.type == "tool_use" for content in final_response.content):
                    payload_input["messages"].append({"role": "assistant", "content": final_response.content})
                    tool_response = await self._process_tool_block(final_response, messages, agent_tracking_info)
                    payload_input["messages"].append(tool_response)
                    # Collect tool_response for storage persistence
                    if isinstance(tool_response, dict) and tool_response.get("role") == "user":
                        self._pending_tool_responses.append(
                            ConversationMessage(
                                role=ParticipantRole.USER.value,
                                content=tool_response.get("content", [])
                            )
                        )

                else:
                    continue_with_tools = False
                    # yield last message
                    kwargs = {
                        "agent_name": self.name,
                        "response": final_response,
                        "messages": messages,
                        "agent_tracking_info": agent_tracking_info,
                    }
                    await self.callbacks.on_agent_end(**kwargs)

                    # Create content list with text from final_response
                    content_list = []

                    # Add text content, filter out empty items
                    for content in final_response.content:
                        if hasattr(content, 'text') and content.text:
                            content_list.append({"text": content.text})

                    # Add thinking to the content if it exists
                    if accumulated_thinking:
                        content_list.append({"thinking": accumulated_thinking})

                    yield AgentStreamResponse(
                        final_message=ConversationMessage(
                            role=ParticipantRole.ASSISTANT.value,
                            content=content_list
                        ),
                        final_thinking=accumulated_thinking
                    )

                max_recursions -= 1

        return stream_generator()

    async def _process_with_strategy(
        self,
        streaming: bool,
        payload_input: dict,
        messages: list[Any],
        agent_tracking_info: dict[str, Any] | None = None
    ) -> ConversationMessage | AsyncIterable[Any]:
        """Process the request using the specified strategy."""

        max_recursions = self._get_max_recursions()

        if streaming:
            return await self._handle_streaming(payload_input, messages, max_recursions, agent_tracking_info)
        response = await self._handle_single_response_loop(payload_input, messages, max_recursions, agent_tracking_info)

        kwargs = {
            "agent_name": self.name,
            "response": response,
            "messages": messages,
            "agent_tracking_info": agent_tracking_info,
        }
        await self.callbacks.on_agent_end(**kwargs)
        return response

    async def _process_tool_block(
        self, llm_response: Any, conversation: list[Any], agent_tracking_info: dict[str, Any] | None = None
    ) -> Any:
        if "useToolHandler" in self.tool_config:
            # tool process logic is handled elsewhere
            tool_response = await self.tool_config["useToolHandler"](llm_response, conversation)
        else:
            # tool process logic is handled in AgentTools class
            if isinstance(self.tool_config["tool"], AgentTools):
                additional_params = {"agent_name": self.name, "agent_tracking_info": agent_tracking_info}
                tool_response = await self.tool_config["tool"].tool_handler(
                    AgentProviderType.ANTHROPIC.value, llm_response, conversation, additional_params
                )
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
        llm_content = None

        while continue_with_tools and max_recursions > 0:
            llm_response: Message = await self.handle_single_response(payload_input)
            if any(hasattr(content, 'type') and content.type == "tool_use" for content in llm_response.content):
                payload_input["messages"].append({"role": "assistant", "content": llm_response.content})
                tool_response = await self._process_tool_block(llm_response, messages, agent_tracking_info)
                payload_input["messages"].append(tool_response)
                # Collect tool_response for storage persistence
                if isinstance(tool_response, dict) and tool_response.get("role") == "user":
                    self._pending_tool_responses.append(
                        ConversationMessage(
                            role=ParticipantRole.USER.value,
                            content=tool_response.get("content", [])
                        )
                    )
            else:
                continue_with_tools = False
                llm_content = llm_response.content or [{"text": "No final response generated"}]

            max_recursions -= 1

        return ConversationMessage(role=ParticipantRole.ASSISTANT.value, content=llm_content)

    async def process_request(
        self,
        input_text: str,
        user_id: str,
        session_id: str,
        chat_history: list[ConversationMessage],
        additional_params: Optional[dict[str, str]] = None,
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

        messages = self._prepare_conversation(input_text, chat_history)
        system_prompt = await self._prepare_system_prompt(input_text)
        json_input = self._build_input(messages, system_prompt)

        return await self._process_with_strategy(self.streaming, json_input, messages, agent_tracking_info)

    async def handle_single_response(self, input_data: dict) -> Any:
        try:
            await self.callbacks.on_llm_start(self.name, payload_input=input_data.get('messages')[-1], **input_data)
            response:Message = self.client.messages.create(**input_data)

            kwargs = {
                "usage": {
                    "inputTokens": response.usage.input_tokens,
                    "outputTokens": response.usage.output_tokens,
                    "totalTokens": response.usage.input_tokens + response.usage.output_tokens,
                },
                "input": {
                    "modelId": response.model,
                    "messages": input_data.get("messages"),
                    "system": input_data.get("system"),
                },
                "inferenceConfig": {
                    "temperature": input_data.get("temperature"),
                    "top_p": input_data.get("top_p"),
                    "stop_sequences": input_data.get("stop_sequences"),
                },
            }
            await self.callbacks.on_llm_end(self.name, output=response.content, **kwargs)

            return response
        except Exception as error:
            Logger.error(f"Error invoking Anthropic: {error}")
            raise error

    async def handle_streaming_response(self, payload_input) -> AsyncGenerator[AgentStreamResponse, None]:
        message = {}
        content = []
        accumulated = {}
        accumulated_thinking = ""
        message["content"] = content

        try:

            await self.callbacks.on_llm_start(self.name, payload_input=payload_input.get('messages')[-1], **payload_input)
            async with self.client.messages.stream(**payload_input) as stream:
                async for event in stream:
                    if event.type == "thinking":
                        await self.callbacks.on_llm_new_token(token="", thinking=event.thinking)
                        accumulated_thinking += event.thinking
                        yield AgentStreamResponse(thinking=event.thinking)
                    elif event.type == "text":
                        await self.callbacks.on_llm_new_token(event.text)
                        yield AgentStreamResponse(text=event.text)
                    elif event.type == "content_block_stop":
                        pass

                # Get the accumulated final message after consuming the stream
                accumulated: Message = await stream.get_final_message()

            # We need to yield the whole content to keep the tool use block
            # This should be a single yield with the final message
            yield AgentStreamResponse(
                text="",  # Empty text for the final chunk
                final_message=accumulated,
                final_thinking=accumulated_thinking
            )

            kwargs = {
                "usage": {
                    "inputTokens": accumulated.usage.input_tokens,
                    "outputTokens": accumulated.usage.output_tokens,
                    "totalTokens": accumulated.usage.input_tokens + accumulated.usage.output_tokens,
                },
                "input": {
                    "modelId": accumulated.model,
                    "messages": payload_input.get("messages"),
                    "system": payload_input.get("system"),
                },
                "inferenceConfig": {
                    "temperature": payload_input.get("temperature"),
                    "top_p": payload_input.get("top_p"),
                    "stop_sequences": payload_input.get("stop_sequences"),
                    "max_tokens": payload_input.get("max_tokens"),
                },
                "final_thinking": accumulated_thinking,

            }
            await self.callbacks.on_llm_end(self.name, output=accumulated, **kwargs)

        except Exception as error:
            Logger.error(f"Error getting stream from Anthropic model: {str(error)}")
            raise error

    def set_system_prompt(self, template: Optional[str] = None, variables: Optional[TemplateVariables] = None) -> None:
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
        def replace(match):
            key = match.group(1)
            if key in variables:
                value = variables[key]
                return "\n".join(value) if isinstance(value, list) else str(value)
            return match.group(0)

        return re.sub(r"{{(\w+)}}", replace, template)
