import ast
import asyncio
import operator
import uuid

import boto3
import streamlit as st
from agent_squad.orchestrator import AgentSquad, AgentSquadConfig
from agent_squad.agents import (
    AgentResponse,
    AgentStreamResponse,
    BedrockLLMAgent,
    BedrockLLMAgentOptions,
)
from agent_squad.classifiers import BedrockClassifier, BedrockClassifierOptions
from agent_squad.types import ConversationMessage
from agent_squad.utils import AgentTool, AgentTools
from weather_tool import get_weather


def test_aws_connection():
    """Test the AWS connection and return a status message."""
    try:
        boto3.client('sts').get_caller_identity()
        return True
    except Exception:
        return False


# Set up the Streamlit app
st.set_page_config(page_title="Agent Squad Chat Demo", page_icon="💬")
st.title("Agent Squad Chat Demo 💬")
st.caption("""
Experience the power of multi-agent orchestration with this interactive chat demo.
Ask questions about technology, health, weather, or math - the system automatically
routes your query to the most appropriate specialized agent.

To learn more about the agents used in this demo, visit
[this link](https://github.com/awslabs/agent-squad/tree/main/examples/python/chat-demo).
""")

# Check AWS connection
if not test_aws_connection():
    st.error("AWS connection failed. Please check your AWS credentials and region configuration.")
    st.warning("Visit the AWS documentation for guidance on setting up your credentials and region.")
    st.stop()

# Define weather tool
weather_tool = AgentTool(
    name='get_weather',
    description='Get current weather conditions for a city',
    properties={
        'city': {
            'type': 'string',
            'description': 'The city name to get weather for'
        }
    },
    func=get_weather,
    required=['city']
)

# Define math tool
def calculate(expression: str) -> str:
    """
    Safely evaluate a mathematical expression using AST.

    :param expression: The mathematical expression to evaluate
    """
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def safe_eval(node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant type: {type(node.value)}")
        elif isinstance(node, ast.BinOp):
            left = safe_eval(node.left)
            right = safe_eval(node.right)
            op = operators.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = safe_eval(node.operand)
            op = operators.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            return op(operand)
        elif isinstance(node, ast.Expression):
            return safe_eval(node.body)
        else:
            raise ValueError(f"Unsupported expression type: {type(node).__name__}")

    try:
        tree = ast.parse(expression, mode='eval')
        result = safe_eval(tree)
        return str(result)
    except (SyntaxError, ValueError) as e:
        return f"Error: {str(e)}"
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


math_tool = AgentTool(
    name='calculate',
    description='Evaluate a mathematical expression',
    properties={
        'expression': {
            'type': 'string',
            'description': 'The mathematical expression to evaluate (e.g., "2 + 2", "10 * 5")'
        }
    },
    func=calculate,
    required=['expression']
)


@st.cache_resource
def create_orchestrator():
    """Create and configure the AgentSquad orchestrator with specialized agents."""

    # Initialize the orchestrator with Bedrock classifier
    orchestrator = AgentSquad(options=AgentSquadConfig(
        LOG_AGENT_CHAT=True,
        LOG_CLASSIFIER_CHAT=True,
        LOG_CLASSIFIER_RAW_OUTPUT=True,
        LOG_CLASSIFIER_OUTPUT=True,
        LOG_EXECUTION_TIMES=True,
        MAX_RETRIES=3,
        MAX_MESSAGE_PAIRS_PER_AGENT=10,
    ))

    # Set up the classifier
    orchestrator.set_classifier(BedrockClassifier(BedrockClassifierOptions(
        model_id='anthropic.claude-3-haiku-20240307-v1:0'
    )))

    # Tech Agent - for technology and software questions
    tech_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
        name="Tech Agent",
        description="Specializes in technology topics including software development, "
                    "programming, AI, cloud computing, cybersecurity, and technical concepts.",
        streaming=True,
        model_id='anthropic.claude-3-haiku-20240307-v1:0',
    ))
    tech_agent.set_system_prompt("""You are a knowledgeable tech expert. Provide clear,
accurate information about technology topics. Use code examples when helpful.
Format responses with markdown for better readability.""")

    # Health Agent - for health and wellness questions
    health_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
        name="Health Agent",
        description="Focuses on health topics including general wellness, nutrition, "
                    "fitness, mental health, and medical terminology. Does not provide medical diagnoses.",
        streaming=True,
        model_id='anthropic.claude-3-haiku-20240307-v1:0',
    ))
    health_agent.set_system_prompt("""You are a helpful health information assistant.
Provide general health and wellness information. Always recommend consulting healthcare
professionals for medical advice. Format responses clearly with markdown.""")

    # Weather Agent - for weather queries with tool
    weather_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
        name="Weather Agent",
        description="Provides current weather information for cities around the world.",
        streaming=True,
        model_id='anthropic.claude-3-haiku-20240307-v1:0',
        tool_config={
            'tool': AgentTools(tools=[weather_tool]),
            'toolMaxRecursions': 5,
        },
    ))
    weather_agent.set_system_prompt("""You are a weather assistant. Use the get_weather
tool to fetch current weather data and present it in a friendly, readable format.""")

    # Math Agent - for mathematical calculations
    math_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
        name="Math Agent",
        description="Handles mathematical questions, calculations, and explains math concepts.",
        streaming=True,
        model_id='anthropic.claude-3-haiku-20240307-v1:0',
        tool_config={
            'tool': AgentTools(tools=[math_tool]),
            'toolMaxRecursions': 5,
        },
    ))
    math_agent.set_system_prompt("""You are a math assistant. Help with calculations using
the calculate tool and explain mathematical concepts clearly. Show your work when solving problems.""")

    # Add all agents to the orchestrator
    orchestrator.add_agent(tech_agent)
    orchestrator.add_agent(health_agent)
    orchestrator.add_agent(weather_agent)
    orchestrator.add_agent(math_agent)

    return orchestrator


async def process_message(orchestrator: AgentSquad, user_input: str, user_id: str, session_id: str):
    """Process a user message through the orchestrator."""
    response: AgentResponse = await orchestrator.route_request(
        user_input,
        user_id,
        session_id
    )
    return response


def get_response_text(response: AgentResponse) -> str:
    """Extract text content from agent response."""
    if isinstance(response.output, str):
        return response.output
    elif isinstance(response.output, ConversationMessage):
        content = response.output.content
        if isinstance(content, list) and len(content) > 0:
            return content[0].get('text', str(content))
        return str(content)
    return str(response.output)


async def stream_response(response: AgentResponse):
    """Stream response chunks for streaming responses."""
    if response.streaming:
        async for chunk in response.output:
            if isinstance(chunk, AgentStreamResponse):
                yield chunk.text
            elif isinstance(chunk, str):
                yield chunk
    else:
        yield get_response_text(response)


# Initialize orchestrator
orchestrator = create_orchestrator()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Sidebar with agent information
with st.sidebar:
    st.header("Available Agents")
    st.markdown("""
    **Tech Agent** - Technology, programming, AI, cloud computing

    **Health Agent** - Wellness, nutrition, fitness, mental health

    **Weather Agent** - Current weather conditions for any city

    **Math Agent** - Calculations and mathematical concepts
    """)

    st.divider()

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.divider()
    st.caption(f"Session: {st.session_state.session_id[:8]}...")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "agent" in message:
            st.caption(f"Handled by: {message['agent']}")

# Chat input
if prompt := st.chat_input("Ask me anything about tech, health, weather, or math..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Process the message
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(
                process_message(
                    orchestrator,
                    prompt,
                    st.session_state.user_id,
                    st.session_state.session_id
                )
            )

            agent_name = response.metadata.agent_name

            # Handle streaming vs non-streaming response
            if response.streaming:
                # Collect streamed content
                async def collect_stream():
                    collected = ""
                    async for chunk in stream_response(response):
                        collected += chunk
                    return collected

                full_response = loop.run_until_complete(collect_stream())
                st.markdown(full_response)
            else:
                full_response = get_response_text(response)
                st.markdown(full_response)

            st.caption(f"Handled by: {agent_name}")

    # Add assistant response to chat history
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "agent": agent_name
    })
