from typing import Any, Optional, AsyncGenerator, AsyncIterable
from dataclasses import dataclass
import re
import json
import boto3
from agent_squad.agents import Agent, AgentOptions, AgentStreamResponse
from agent_squad.types import (
    ConversationMessage,
    ParticipantRole,
    BEDROCK_MODEL_ID_CLAUDE_3_HAIKU,
    TemplateVariables,
    AgentProviderType,
)
from agent_squad.utils import (
    conversation_to_dict,
    Logger,
    AgentTools,
    AgentTool,
)
from agent_squad.retrievers import Retriever
from agent_squad.shared import user_agent


@dataclass
class BedrockLLMAgentOptions(AgentOptions):
    model_id: Optional[str] = None
    region: Optional[str] = None
    streaming: Optional[bool] = None
    inference_config: Optional[dict[str, Any]] = None
    guardrail_config: Optional[dict[str, str]] = None
    retriever: Optional[Retriever] = None
    tool_config: dict[str, Any] | AgentTools | None = None
    custom_system_prompt: Optional[dict[str, Any]] = None
    client: Optional[Any] = None
    additional_model_request_fields: Optional[dict[str, Any]] = None


class BedrockLLMAgent(Agent):
    def __init__(self, options: BedrockLLMAgentOptions):
        super().__init__(options)
        if options.client:
            self.client = options.client
        else:
            if options.region:
                self.client = boto3.client("bedrock-runtime", region_name=options.region)
            else:
                self.client = boto3.client("bedrock-runtime")

        user_agent.register_feature_to_client(self.client, feature="bedrock-llm-agent")

        self.model_id: str = options.model_id or BEDROCK_MODEL_ID_CLAUDE_3_HAIKU
        self.streaming: bool = options.streaming
        self.inference_config: dict[str, Any]

        default_inference_config = {
            "maxTokens": 1000,
            "temperature": 0.0,
            "topP": 0.9,
            "stopSequences": [],
        }

        if options.inference_config:
            self.inference_config = {
                **default_inference_config,
                **options.inference_config,
            }
        else:
            self.inference_config = default_inference_config

        self.additional_model_request_fields: Optional[dict[str, Any]] = options.additional_model_request_fields or {}
        # if thinking is enabled, unset top_p
        if self.additional_model_request_fields.get("thinking", {}).get("type") == "enabled":
            Logger.warn("Removing topP for thinking mode")
            del self.inference_config["topP"]

        self.guardrail_config: Optional[dict[str, str]] = options.guardrail_config or {}

        self.retriever: Optional[Retriever] = options.retriever
        self.tool_config: Optional[dict[str, Any]] = options.tool_config

        self.prompt_template: str = f"""You are a {self.name}.
        {self.description}
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

        self.system_prompt: str = ""
        self.custom_variables: TemplateVariables = {}
        self.default_max_recursions: int = 20

        if options.custom_system_prompt:
            self.set_system_prompt(
                options.custom_system_prompt.get("template"),
                options.custom_system_prompt.get("variables"),
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
        self, input_text: str, chat_history: list[ConversationMessage]
    ) -> list[ConversationMessage]:
        """Prepare the conversation history with the new user message."""

        user_message = ConversationMessage(role=ParticipantRole.USER.value, content=[{"text": input_text}])
        return [*chat_history, user_message]

    def _build_conversation_command(self, conversation: list[ConversationMessage], system_prompt: str) -> dict:
        """Build the conversation command with all necessary configurations."""

        inference_config = {
            "maxTokens": self.inference_config.get("maxTokens"),
            "temperature": self.inference_config.get("temperature"),
            "stopSequences": self.inference_config.get("stopSequences"),
        }

        # Only add topP if it exists in the inference_config
        if "topP" in self.inference_config:
            inference_config["topP"] = self.inference_config["topP"]

        command = {
            "modelId": self.model_id,
            "messages": conversation_to_dict(conversation),
            "system": [{"text": system_prompt}],
            "inferenceConfig": inference_config,
        }

        if self.guardrail_config:
            command["guardrailConfig"] = self.guardrail_config

        if self.additional_model_request_fields:
            command["additionalModelRequestFields"] = self.additional_model_request_fields

        if self.tool_config:
            command["toolConfig"] = self._prepare_tool_config()

        return command

    def _prepare_tool_config(self) -> dict:
        """Prepare tool configuration based on the tool type."""

        if isinstance(self.tool_config["tool"], AgentTools):
            return {"tools": self.tool_config["tool"].to_bedrock_format()}

        if isinstance(self.tool_config["tool"], list):
            return {
                "tools": [
                    tool.to_bedrock_format() if isinstance(tool, AgentTool) else tool
                    for tool in self.tool_config["tool"]
                ]
            }

        raise RuntimeError("Invalid tool config")

    def _get_max_recursions(self) -> int:
        """Get the maximum number of recursions based on tool configuration."""
        if not self.tool_config:
            return 1
        return self.tool_config.get("toolMaxRecursions", self.default_max_recursions)

    async def _handle_single_response_loop(
        self,
        command: dict,
        conversation: list[ConversationMessage],
        max_recursions: int,
        agent_tracking_info: dict,
    ) -> ConversationMessage:
        """Handle single response processing with tool recursion."""

        continue_with_tools = True
        llm_response = None
        accumulated_thinking = None

        while continue_with_tools and max_recursions > 0:
            llm_response = await self.handle_single_response(command, agent_tracking_info)

            # Extract thinking content if present in the response
            for content_item in llm_response.content:
                if isinstance(content_item, dict) and "reasoningContent" in content_item:
                    accumulated_thinking = content_item["reasoningContent"]
                    break

            conversation.append(llm_response)

            if any("toolUse" in content for content in llm_response.content):
                tool_response = await self._process_tool_block(llm_response, conversation, agent_tracking_info)
                conversation.append(tool_response)
                command["messages"] = conversation_to_dict(conversation)
            else:
                continue_with_tools = False

            max_recursions -= 1

        # Add final_thinking to agent tracking info for callbacks
        if accumulated_thinking:
            if not agent_tracking_info:
                agent_tracking_info = {}
            agent_tracking_info["final_thinking"] = accumulated_thinking

        return llm_response

    async def _handle_streaming(
        self,
        command: dict,
        conversation: list[ConversationMessage],
        max_recursions: int,
        agent_tracking_info: dict,
    ) -> AsyncIterable[Any]:
        """Handle streaming response processing with tool recursion."""
        continue_with_tools = True
        final_response = None
        accumulated_thinking = ""  # Track thinking across chunks

        async def stream_generator():
            nonlocal continue_with_tools, final_response, max_recursions, accumulated_thinking

            while continue_with_tools and max_recursions > 0:
                response = self.handle_streaming_response(command, agent_tracking_info=agent_tracking_info)

                async for chunk in response:
                    if isinstance(chunk, AgentStreamResponse):
                        yield chunk

                        if chunk.final_message:
                            final_response = chunk.final_message
                            # Capture final thinking if available
                            if chunk.final_thinking:
                                accumulated_thinking = chunk.final_thinking

                conversation.append(final_response)

                if any("toolUse" in content for content in final_response.content):
                    tool_response = await self._process_tool_block(final_response, conversation, agent_tracking_info)

                    conversation.append(tool_response)
                    command["messages"] = conversation_to_dict(conversation)
                else:
                    continue_with_tools = False

                max_recursions -= 1

            kwargs = {
                "agent_name": self.name,
                "response": final_response,
                "messages": conversation,
                "agent_tracking_info": agent_tracking_info,
                "final_thinking": accumulated_thinking if accumulated_thinking else None,
            }
            await self.callbacks.on_agent_end(**kwargs)

        return stream_generator()

    async def _process_with_strategy(
        self,
        streaming: bool,
        command: dict,
        conversation: list[ConversationMessage],
        agent_tracking_info: dict,
    ) -> ConversationMessage | AsyncIterable[Any]:
        """Process the request using the specified strategy."""

        max_recursions = self._get_max_recursions()

        if streaming:
            return await self._handle_streaming(command, conversation, max_recursions, agent_tracking_info)
        response = await self._handle_single_response_loop(command, conversation, max_recursions, agent_tracking_info)

        kwargs = {
            "agent_name": self.name,
            "response": response,
            "messages": conversation,
            "agent_tracking_info": agent_tracking_info,
        }

        await self.callbacks.on_agent_end(**kwargs)
        return response

    async def process_request(
        self,
        input_text: str,
        user_id: str,
        session_id: str,
        chat_history: list[ConversationMessage],
        additional_params: Optional[dict[str, str]] = None,
    ) -> ConversationMessage | AsyncIterable[Any]:
        """
        Process a conversation request either in streaming or single response mode.
        """
        kwargs = {
            "agent_name": self.name,
            "payload_input": input_text,
            "messages": [*chat_history],
            "additional_params": additional_params,
            "user_id": user_id,
            "session_id": session_id,
        }
        agent_tracking_info = await self.callbacks.on_agent_start(**kwargs)

        conversation = self._prepare_conversation(input_text, chat_history)
        system_prompt = await self._prepare_system_prompt(input_text)

        command = self._build_conversation_command(conversation, system_prompt)

        return await self._process_with_strategy(self.streaming, command, conversation, agent_tracking_info)

    async def _process_tool_block(
        self,
        llm_response: ConversationMessage,
        conversation: list[ConversationMessage],
        agent_tracking_info: dict[str, Any] | None = None,
    ) -> ConversationMessage:
        if "useToolHandler" in self.tool_config:
            # tool process logic is handled elsewhere
            tool_response = await self.tool_config["useToolHandler"](llm_response, conversation)
        else:
            additional_params = {
                "agent_name": self.name,
                "agent_tracking_info": agent_tracking_info,
            }
            # tool process logic is handled in AgentTools class
            if isinstance(self.tool_config["tool"], AgentTools):
                tool_response = await self.tool_config["tool"].tool_handler(
                    AgentProviderType.BEDROCK.value,
                    llm_response,
                    conversation,
                    additional_params,
                )
            else:
                raise ValueError("You must use AgentTools class when not providing a custom tool handler")
        return tool_response

    async def handle_single_response(
        self, converse_input: dict[str, Any], agent_tracking_info: dict
    ) -> ConversationMessage:
        try:
            kwargs = {
                "name": self.name,
                "payload_input": converse_input.get("messages")[-1],
                "converse_input": converse_input,
                "agent_tracking_info": agent_tracking_info,
            }
            await self.callbacks.on_llm_start(**kwargs)

            response = self.client.converse(**converse_input)
            if "output" not in response:
                raise ValueError("No output received from Bedrock model")

            # Extract thinking content if available
            thinking_content = None
            if "reasoningContent" in response["output"]["message"]["content"][0]:
                if "reasoningText" in response["output"]["message"]["content"][0]["reasoningContent"]:
                    thinking_content = response["output"]["message"]["content"][0]["reasoningContent"]

            # Get content from response and filter for text items
            response_content = response["output"]["message"]["content"]
            content = []

            # Go through response content and save text items
            for item in response_content:
                if isinstance(item, dict) and "text" in item:
                    content.append(item)

            toolInUse = False
            # Go through response content and save toolUse items
            for item in response_content:
                if isinstance(item, dict) and "toolUse" in item:
                    content.append(item)
                    toolInUse = True

            # when a tool is used, the next iteration should have the reasoningContent at the first location
            if toolInUse:
                if thinking_content:
                    content.insert(0,{"reasoningContent": thinking_content})
            elif thinking_content:
                content.append({"reasoningContent": thinking_content})

            kwargs = {
                "name": self.name,
                "output": response.get("output", {}).get("message"),
                "usage": response.get("usage"),
                "system": converse_input.get("system")[0].get("text"),
                "input": converse_input,
                "inferenceConfig": converse_input.get("inferenceConfig"),
                "agent_tracking_info": agent_tracking_info,
                "final_thinking": thinking_content,  # Add thinking to callback
            }
            await self.callbacks.on_llm_end(**kwargs)

            return ConversationMessage(
                role=response["output"]["message"]["role"],
                content=content,
            )
        except Exception as error:
            Logger.error(f"Error invoking Bedrock model:{str(error)}")
            raise error

    async def handle_streaming_response(
        self,
        converse_input: dict[str, Any],
        agent_tracking_info: dict,
    ) -> AsyncGenerator[AgentStreamResponse, None]:
        """
        Handle streaming response from Bedrock model.
        Yields StreamChunk objects containing text chunks, thinking content, or the final message.

        When thinking is enabled through additional_model_request_fields, this method will:
        1. Process "reasoningContent" events as thinking content
        2. Accumulate thinking content throughout the streaming process
        3. Include the final thinking content in the final message
        4. Pass thinking tokens to callbacks with the thinking=True flag

        Args:
            converse_input: Input for the conversation
            agent_tracking_info: Tracking information for callbacks

        Yields:
            AgentStreamResponse: Contains text chunks, thinking content, or the final message with thinking
        """
        try:
            kwargs = {
                "name": self.name,
                "payload_input": converse_input.get("messages")[-1],
                "messages": converse_input,
                "agent_tracking_info": agent_tracking_info,
            }
            await self.callbacks.on_llm_start(**kwargs)
            response = self.client.converse_stream(**converse_input)

            metadata = {}
            message = {}
            content = []
            message["content"] = content
            text = ""
            thinking_signature = {}
            thinking = ""
            accumulated_thinking = ""  # Add this for complete thinking content
            tool_use = {}

            for chunk in response["stream"]:
                if "messageStart" in chunk:
                    message["role"] = chunk["messageStart"]["role"]
                elif "contentBlockStart" in chunk:
                    tool = chunk["contentBlockStart"]["start"]["toolUse"]
                    tool_use["toolUseId"] = tool["toolUseId"]
                    tool_use["name"] = tool["name"]
                elif "contentBlockDelta" in chunk:
                    delta = chunk["contentBlockDelta"]["delta"]
                    if "toolUse" in delta:
                        if "input" not in tool_use:
                            tool_use["input"] = ""
                        tool_use["input"] += delta["toolUse"]["input"]
                    elif "text" in delta:
                        text += delta["text"]
                        token_kwargs = {
                            "token": delta["text"],
                            "agent_tracking_info": agent_tracking_info,
                        }
                        await self.callbacks.on_llm_new_token(**token_kwargs)
                        # yield the text chunk
                        yield AgentStreamResponse(text=delta["text"])
                    elif "reasoningContent" in delta:
                        if "text" in delta["reasoningContent"]:
                            thinking_text = delta["reasoningContent"]["text"]
                            accumulated_thinking += thinking_text
                            token_kwargs = {
                                "token": thinking_text,
                                "agent_tracking_info": agent_tracking_info,
                                "thinking": True,
                            }
                            await self.callbacks.on_llm_new_token(**token_kwargs)
                            # yield with thinking field instead of text
                            yield AgentStreamResponse(thinking=thinking_text)
                        elif "signature" in delta["reasoningContent"]:
                            thinking_signature = delta["reasoningContent"]["signature"]
                elif "contentBlockStop" in chunk:
                    if "input" in tool_use and tool_use.get("input"):
                        tool_use["input"] = json.loads(tool_use["input"])
                        content.append({"toolUse": tool_use})
                        tool_use = {}
                    else:
                        if text:
                            content.append({"text": text})
                            text = ""
                elif "metadata" in chunk:
                    metadata = chunk.get("metadata")

            # Get content from response and filter for text items
            response_content = message["content"]
            _content = []

            # Go through response content and save text items
            for item in response_content:
                if isinstance(item, dict) and "text" in item:
                    _content.append(item)

            toolInUse = False
            # Go through response content and save toolUse items
            for item in response_content:
                if isinstance(item, dict) and "toolUse" in item:
                    _content.append(item)
                    toolInUse = True

            # when a tool is used, the next iteration should have the reasoningContent at the first index
            if toolInUse:
                if accumulated_thinking:
                    _content.insert(0,{"reasoningContent": {"reasoningText": {"text": accumulated_thinking, "signature":thinking_signature}}})
            elif accumulated_thinking:
                _content.append({"reasoningContent": {"reasoningText": {"text": accumulated_thinking, "signature":thinking_signature}}})


            final_message = ConversationMessage(role=ParticipantRole.ASSISTANT.value, content=_content)

            kwargs = {
                "name": self.name,
                "output": _content,
                "usage": metadata.get("usage"),
                "system": converse_input.get("system")[0].get("text"),
                "input": converse_input,
                "agent_tracking_info": agent_tracking_info,
                "final_thinking": accumulated_thinking if accumulated_thinking else None,
            }
            await self.callbacks.on_llm_end(**kwargs)

            # yield the final message with thinking
            yield AgentStreamResponse(
                final_message=final_message,
                final_thinking=accumulated_thinking if accumulated_thinking else None
            )
        except Exception as error:
            Logger.error(f"Error getting stream from Bedrock model: {str(error)}")
            raise error

    def set_system_prompt(
        self,
        template: Optional[str] = None,
        variables: Optional[TemplateVariables] = None,
    ) -> None:
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
