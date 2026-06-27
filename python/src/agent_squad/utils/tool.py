from typing import Any, Optional, Callable, get_type_hints
import inspect
from functools import wraps
import re
from dataclasses import dataclass
from agent_squad.types import (
    AgentProviderType,
    ConversationMessage,
    ParticipantRole,
)
from uuid import UUID
import json


@dataclass
class PropertyDefinition:
    type: str
    description: str
    enum: Optional[list] = None


@dataclass
class AgentToolResult:
    tool_use_id: str
    content: Any

    def to_anthropic_format(self) -> dict:
        return {
            "type": "tool_result",
            "tool_use_id": self.tool_use_id,
            "content": self.content,
        }

    def to_bedrock_format(self) -> dict:
        return {
            "toolResult": {
                "toolUseId": self.tool_use_id,
                "content": [{"text": self.content}],
            }
        }

    def to_openai_format(self) -> dict:
        return {
            "role": ParticipantRole.TOOL.value,
            "tool_call_id": self.tool_use_id,
            "content": self.content,
        }


class AgentToolCallbacks:
    async def on_tool_start(
        self,
        tool_name,
        payload_input: Any,
        run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        pass

    async def on_tool_end(
        self,
        tool_name,
        payload_input: Any,
        output: Any,
        run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        pass

    async def on_tool_error(
        self,
        tool_name,
        payload_input: Any,
        error: Exception,
        run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        pass


class AgentTool:
    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        properties: Optional[dict[str, dict[str, Any]]] = None,
        required: Optional[list[str]] = None,
        func: Optional[Callable] = None,
        enum_values: Optional[dict[str, list]] = None,
    ):

        self.name = name
        # Extract docstring if description not provided
        if description is None:
            docstring = inspect.getdoc(func)
            if docstring:
                # Get the first paragraph of the docstring (before any parameter descriptions)
                self.func_description = docstring.split("\n\n")[0].strip()
            else:
                self.func_description = f"Function to {name}"
        else:
            self.func_description = description
        self.enum_values = enum_values or {}

        if not func:
            raise ValueError("Function must be provided")

        # Extract properties from the function if not passed
        self.properties = properties or self._extract_properties(func)
        self.required = required or list(self.properties.keys())
        self.func = self._wrap_function(func)

        # Add enum values to properties if they exist
        for prop_name, enum_vals in self.enum_values.items():
            if prop_name in self.properties:
                self.properties[prop_name]["enum"] = enum_vals

    def _extract_properties(self, func: Callable) -> dict[str, dict[str, Any]]:
        """Extract properties from the function's signature and type hints"""
        # Get function's type hints and signature
        type_hints = get_type_hints(func)
        sig = inspect.signature(func)

        # Parse docstring for parameter descriptions
        docstring = inspect.getdoc(func) or ""
        param_descriptions = {}

        # Extract parameter descriptions using regex
        param_matches = re.finditer(r":param\s+(\w+)\s*:\s*([^:\n]+)", docstring)
        for match in param_matches:
            param_name = match.group(1)
            description = match.group(2).strip()
            param_descriptions[param_name] = description

        properties = {}
        for param_name, _param in sig.parameters.items():
            # Skip 'self' parameter for class methods
            if param_name == "self":
                continue

            param_type = type_hints.get(param_name, Any)

            # Convert Python types to JSON schema types
            type_mapping = {
                int: "integer",
                float: "number",
                str: "string",
                bool: "boolean",
                list: "array",
                dict: "object",
            }

            json_type = type_mapping.get(param_type, "string")

            # Use docstring description if available, else create a default one
            description = param_descriptions.get(
                param_name, f"The {param_name} parameter"
            )

            properties[param_name] = {"type": json_type, "description": description}

        return properties

    def _wrap_function(self, func: Callable) -> Callable:
        """Wrap the function to preserve its metadata and handle async/sync functions"""

        @wraps(func)
        async def wrapper(**kwargs):
            result = func(**kwargs)
            if inspect.iscoroutine(result):
                return await result
            return result

        return wrapper

    def to_claude_format(self) -> dict[str, Any]:
        """Convert generic tool definition to Claude format"""
        return {
            "name": self.name,
            "description": self.func_description,
            "input_schema": {
                "type": "object",
                "properties": self.properties,
                "required": self.required,
            },
        }

    def to_bedrock_format(self) -> dict[str, Any]:
        """Convert generic tool definition to Bedrock format"""
        return {
            "toolSpec": {
                "name": self.name,
                "description": self.func_description,
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": self.properties,
                        "required": self.required,
                    }
                },
            }
        }

    def to_openai_format(self) -> dict[str, Any]:
        """Convert generic tool definition to OpenAI format"""
        return {
            "type": "function",
            "function": {
                "name": self.name.lower().replace("_tool", ""),
                "description": self.func_description,
                "parameters": {
                    "type": "object",
                    "properties": self.properties,
                    "required": self.required,
                    "additionalProperties": False,
                },
            },
        }


class AgentTools:
    def __init__(
        self, tools: list[AgentTool], callbacks: Optional[AgentToolCallbacks] = None
    ):
        self.tools: list[AgentTool] = tools
        self.callbacks = callbacks or AgentToolCallbacks()

    async def tool_handler(
        self,
        provider_type,
        response: Any,
        _conversation: list[dict[str, Any]],
        agent_info: Optional[dict[str, Any]] = None,
    ) -> Any:
        if not response.content:
            raise ValueError("No content blocks in response")

        tool_results = []
        content_blocks = response.content

        for block in content_blocks:
            # Determine if it's a tool use block based on platform
            tool_use_block = self._get_tool_use_block(provider_type, block)
            if not tool_use_block:
                continue

            tool_name = (
                tool_use_block.get("name")
                if provider_type == AgentProviderType.BEDROCK.value
                else tool_use_block.get("name")
                if provider_type == AgentProviderType.OPENAI.value
                else tool_use_block.name
            )

            tool_id = (
                tool_use_block.get("toolUseId")
                if provider_type == AgentProviderType.BEDROCK.value
                else block.get('id')
                if provider_type == AgentProviderType.OPENAI.value
                else tool_use_block.id
            )

            # Get input based on platform
            input_data = (
                tool_use_block.get("input", {})
                if provider_type == AgentProviderType.BEDROCK.value
                else json.loads(tool_use_block.get("arguments", "{}"))
                if provider_type == AgentProviderType.OPENAI.value
                else tool_use_block.input
            )

            # Process the tool use
            await self.callbacks.on_tool_start(
                tool_name, input_data, metadata={"agent_info": agent_info}
            )
            result = await self._process_tool(tool_name, input_data)
            await self.callbacks.on_tool_end(
                tool_name, input_data, result, metadata={"agent_info": agent_info}
            )

            # Create tool result
            tool_result = AgentToolResult(tool_id, result)

            # Format according to platform
            formatted_result = (
                tool_result.to_bedrock_format()
                if provider_type == AgentProviderType.BEDROCK.value
                else tool_result.to_openai_format()
                if provider_type == AgentProviderType.OPENAI.value
                else tool_result.to_anthropic_format()
            )

            tool_results.append(formatted_result)

        # Create and return appropriate message format
        if provider_type == AgentProviderType.BEDROCK.value:
            return ConversationMessage(
                role=ParticipantRole.USER.value, content=tool_results
            )
        elif provider_type == AgentProviderType.OPENAI.value:
            return ConversationMessage(
                role=ParticipantRole.TOOL.value, content=tool_results
            )
        else:
            return {"role": ParticipantRole.USER.value, "content": tool_results}

    def _get_tool_use_block(
        self, provider_type: AgentProviderType, block: dict
    ) -> dict | None:
        """Extract tool use block based on platform format."""
        if provider_type == AgentProviderType.BEDROCK.value and "toolUse" in block:
            return block["toolUse"]
        elif (
            provider_type == AgentProviderType.ANTHROPIC.value
            and block.get('type') == "tool_use"
        ):
            return block
        elif provider_type == AgentProviderType.OPENAI.value and block.get('type') == "function":
            return block["function"]
        return None

    async def _process_tool(self, tool_name, input_data):
        try:
            tool = next(tool for tool in self.tools if tool.name == tool_name)
            return await tool.func(**input_data)
        except StopIteration:
            return f"Tool '{tool_name}' not found"

    def to_claude_format(self) -> list[dict[str, Any]]:
        """Convert all tools to Claude format"""
        return [tool.to_claude_format() for tool in self.tools]

    def to_bedrock_format(self) -> list[dict[str, Any]]:
        """Convert all tools to Bedrock format"""
        return [tool.to_bedrock_format() for tool in self.tools]

    def to_openai_format(self) -> list[dict[str, Any]]:
        """Convert all tools to Claude format"""
        return [tool.to_openai_format() for tool in self.tools]
