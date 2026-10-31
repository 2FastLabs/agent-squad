from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
import json
import os
import boto3
from agent_squad.utils import conversation_to_dict, Logger
from agent_squad.agents import Agent, AgentOptions
from agent_squad.types import (ConversationMessage,
                       ParticipantRole,
                       BEDROCK_MODEL_ID_CLAUDE_HAIKU_4_5,
                       TemplateVariables)
import re

# BedrockInlineAgentOptions Dataclass
@dataclass
class BedrockInlineAgentOptions(AgentOptions):
    model_id: Optional[str] = None
    region: Optional[str] = None
    inference_config: Optional[Dict[str, Any]] = None
    client: Optional[Any] = None
    bedrock_agent_client: Optional[Any] = None
    foundation_model: Optional[str] = None
    action_groups_list: List[Dict[str, Any]] = field(default_factory=list)
    knowledge_bases: Optional[List[Dict[str, Any]]] = None
    custom_system_prompt: Optional[Dict[str, Any]] = None
    enableTrace: Optional[bool] = False


# BedrockInlineAgent Class
class BedrockInlineAgent(Agent):

    TOOL_NAME = 'inline_agent_creation'
    TOOL_INPUT_SCHEMA = {
        "json": {
            "type": "object",
            "properties": {
                "action_group_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A string array of action group names needed to solve the customer request"
                },
                "knowledge_bases": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A string array of knowledge base names needed to solve the customer request"
                },
                "description": {
                    "type": "string",
                    "description": "Description to instruct the agent how to solve the user request using available action groups and knowledge bases."
                },
                "user_request": {
                    "type": "string",
                    "description": "The initial user request"
                }
            },
            "required": ["action_group_names", "description", "user_request", "knowledge_bases"],
        }
    }

    def __init__(self, options: BedrockInlineAgentOptions):
        super().__init__(options)

        # Initialize Bedrock client
        if options.client:
            self.client = options.client
        else:
            if options.region:
                self.client = boto3.client(
                    'bedrock-runtime',
                    region_name=options.region or os.environ.get('AWS_REGION')
                )
            else:
                self.client = boto3.client('bedrock-runtime')

        self.model_id: str = options.model_id or BEDROCK_MODEL_ID_CLAUDE_HAIKU_4_5

        # Initialize bedrock agent client
        if options.bedrock_agent_client:
            self.bedrock_agent_client = options.bedrock_agent_client
        else:
            if options.region:
                self.bedrock_agent_client = boto3.client(
                    'bedrock-agent-runtime',
                    region_name=options.region or os.environ.get('AWS_REGION')
                )
            else:
                self.bedrock_agent_client = boto3.client('bedrock-agent-runtime')

        # Set model ID
        self.model_id = options.model_id or BEDROCK_MODEL_ID_CLAUDE_HAIKU_4_5

        self.foundation_model = options.foundation_model or BEDROCK_MODEL_ID_CLAUDE_HAIKU_4_5

        # Set inference configuration. topP isn't set by default: Claude Haiku
        # 4.5 and newer reject temperature and topP in the same request.
        default_inference_config = {
            'maxTokens': 1000,
            'temperature': 0.0,
            'stopSequences': []
        }
        self.inference_config = {**default_inference_config, **(options.inference_config or {})}

        # Store action groups and knowledge bases
        self.action_groups_list = options.action_groups_list
        self.knowledge_bases = options.knowledge_bases or []

        # Define inline agent tool configuration
        self.inline_agent_tool = [{
            "toolSpec": {
                "name": BedrockInlineAgent.TOOL_NAME,
                "description": "Create an inline agent with a list of action groups and knowledge bases",
                "inputSchema": self.TOOL_INPUT_SCHEMA,
            }
        }]

        # Define the tool handler
        self.use_tool_handler = self.inline_agent_tool_handler

        # Configure tool usage
        self.tool_config = {
            'tool': self.inline_agent_tool,
            'toolMaxRecursions': 1,
            'useToolHandler': self.use_tool_handler,
        }

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
- Seamlessly transition between topics as the human introduces new subjects.
"""
        self.prompt_template += "\n\nHere are the action groups that you can use to solve the customer request:\n"
        self.prompt_template += "<action_groups>\n"
        for action_group in self.action_groups_list:
            self.prompt_template += f"Action Group Name: {action_group.get('actionGroupName')}\n"
            self.prompt_template += f"Action Group Description: {action_group.get('description','')}\n"
        self.prompt_template += "</action_groups>\n"

        self.prompt_template += "\n\nHere are the knwoledge bases that you can use to solve the customer request:\n"
        self.prompt_template += "<knowledge_bases>\n"
        for kb in self.knowledge_bases:
            self.prompt_template += f"Knowledge Base ID: {kb['knowledgeBaseId']}\n"
            self.prompt_template += f"Knowledge Base Description: {kb.get('description', '')}\n"
        self.prompt_template += "</knowledge_bases>\n"

        self.system_prompt: str = ""
        self.custom_variables: TemplateVariables = {}
        self.default_max_recursions: int = 20

        if options.custom_system_prompt:
            self.set_system_prompt(
                options.custom_system_prompt.get('template'),
                options.custom_system_prompt.get('variables')
            )

        self.enableTrace = options.enableTrace


    async def inline_agent_tool_handler(self, session_id, response, conversation):
        """Handler for processing tool use."""
        response_content_blocks = response.content

        if not response_content_blocks:
            raise ValueError("No content blocks in response")

        for content_block in response_content_blocks:
            if "toolUse" in content_block:
                tool_use_block = content_block["toolUse"]
                tool_use_name = tool_use_block.get("name")
                if tool_use_name == "inline_agent_creation":

                    action_group_names = tool_use_block["input"].get('action_group_names', [])
                    kb_names = tool_use_block["input"].get('knowledge_bases','')

                    description = tool_use_block["input"].get('description', '')
                    user_request = tool_use_block["input"].get('user_request', '')

                    self.log_debug("BedrockInlineAgent", 'Tool Handler Parameters', {
                        'user_request':user_request,
                        'action_group_names':action_group_names,
                        'kb_names':kb_names,
                        'description':description,
                        'session_id':session_id
                    })


                    # Fetch relevant action groups
                    action_groups = [
                        item for item in self.action_groups_list
                        if item.get('actionGroupName') in action_group_names
                    ]
                    for entry in action_groups:
                        # remove description for AMAZON.CodeInterpreter
                        if 'parentActionGroupSignature' in entry and \
                        entry['parentActionGroupSignature'] == 'AMAZON.CodeInterpreter':
                            entry.pop('description', None)

                    kbs = []
                    if kb_names and self.knowledge_bases:
                        kbs = [item for item in self.knowledge_bases
                              if item.get('knowledgeBaseId') in kb_names]

                    self.log_debug("BedrockInlineAgent", 'Action Group & Knowledge Base', {
                        'action_groups':action_groups,
                        'kbs':kbs
                    })

                    self.log_debug("BedrockInlineAgent", 'Invoking Inline Agent', {
                        'foundationModel': self.foundation_model,
                        'enableTrace': self.enableTrace,
                        'sessionId':session_id
                    })

                    inline_response = self.bedrock_agent_client.invoke_inline_agent(
                        actionGroups=action_groups,
                        knowledgeBases=kbs,
                        enableTrace=self.enableTrace,
                        endSession=False,
                        foundationModel=self.foundation_model,
                        inputText=user_request,
                        instruction=description,
                        sessionId=session_id
                    )

                    eventstream = inline_response.get('completion')
                    tool_results = []
                    for event in eventstream:
                        Logger.info(event) if self.enableTrace else None
                        if 'chunk' in event:
                            chunk = event['chunk']
                            if 'bytes' in chunk:
                                tool_results.append(chunk['bytes'].decode('utf-8'))

                    # Return the tool results as a new message
                    return ConversationMessage(
                        role=ParticipantRole.ASSISTANT.value,
                        content=[{'text': ''.join(tool_results)}]
                    )

        raise ValueError("Tool use block not handled")

    async def process_request(
        self,
        input_text: str,
        user_id: str,
        session_id: str,
        chat_history: List[ConversationMessage],
        additional_params: Optional[Dict[str, str]] = None
    ) -> ConversationMessage:
        try:
            # Create the user message
            user_message = ConversationMessage(
                role=ParticipantRole.USER.value,
                content=[{'text': input_text}]
            )

            # Combine chat history with current message
            conversation = [*chat_history, user_message]

            self.update_system_prompt()

            self.log_debug("BedrockInlineAgent", 'System Prompt', self.system_prompt)

            system_prompt = self.system_prompt

            converse_cmd = {
            'modelId': self.model_id,
            'messages': conversation_to_dict(conversation),
            'system': [{'text': system_prompt}],
            # A value set to None is left out, for models that reject sampling parameters.
            'inferenceConfig': {
                key: value for key, value in {
                    'maxTokens': self.inference_config.get('maxTokens'),
                    'temperature': self.inference_config.get('temperature'),
                    'topP': self.inference_config.get('topP'),
                    'stopSequences': self.inference_config.get('stopSequences'),
                }.items() if value is not None
            },
            'toolConfig': {
                    'tools': self.inline_agent_tool,
                    "toolChoice": {
                        "tool": {
                            "name": BedrockInlineAgent.TOOL_NAME,
                        },
                    },
                }
            }
            # Call Bedrock's converse API
            response = self.client.converse(**converse_cmd)

            if 'output' not in response:
                raise ValueError("No output received from Bedrock model")

            bedrock_response = ConversationMessage(
                role=response['output']['message']['role'],
                content=response['output']['message']['content']
            )

            # Check if tool use is required
            for content in bedrock_response.content:
                if isinstance(content, dict) and 'toolUse' in content:
                    return await self.use_tool_handler(session_id, bedrock_response, conversation)

            # Return Bedrock's initial response if no tool is used
            return bedrock_response

        except Exception as error:
            Logger.error(f"Error processing request with Bedrock: {str(error)}")
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
        def replace(match):
            key = match.group(1)
            if key in variables:
                value = variables[key]
                return '\n'.join(value) if isinstance(value, list) else str(value)
            return match.group(0)

        return re.sub(r'{{(\w+)}}', replace, template)
