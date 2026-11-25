import uuid
import asyncio
import streamlit as st
import boto3
from search_web import search_web
from agent_squad.orchestrator import AgentSquad, AgentSquadConfig
from agent_squad.agents import (
    AgentResponse,
    BedrockLLMAgent,
    BedrockLLMAgentOptions,
    SupervisorAgent, SupervisorAgentOptions
)
from agent_squad.types import ConversationMessage
from agent_squad.classifiers import ClassifierResult
from agent_squad.utils import AgentTools, AgentTool

# Function to test AWS connection
def test_aws_connection():
    """Test the AWS connection and return a status message."""
    try:
        # Attempt to create an S3 client as a test
        boto3.client('sts').get_caller_identity()
        return True
    except Exception as e:
        print(f"Incomplete AWS credentials. Please check your AWS configuration.")

    return False

# Set up the Streamlit app
st.title("AI Movie Production Demo 🎬")
st.caption("""
Bring your movie ideas to life with AI Movie Production by collaborating with AI agents powered by Anthropic's Claude for script writing and casting.

To learn more about the agents used in this demo visit [this link](https://github.com/awslabs/agent-squad/tree/main/examples/python/movie-production).

           """)

st.caption("")

# Check AWS connection
if not test_aws_connection():
    st.error("AWS connection failed. Please check your AWS credentials and region configuration.")
    st.warning("Visit the AWS documentation for guidance on setting up your credentials and region.")
    st.stop()

# Define the tools
search_web_tool = AgentTool(name='search_web',
                       description='Search Web for information',
                       properties={
                           'query': {
                               'type': 'string',
                               'description': 'The search query'
                           }
                       },
                       func=search_web,
                       required=['query'])

# Define the agents
script_writer_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
    model_id='us.anthropic.claude-haiku-4-5-20251001-v1:0',
    name="ScriptWriterAgent",
    description="""\
You are an expert screenplay writer. Given a movie idea and genre,
develop a compelling script outline with character descriptions and key plot points.

Your tasks consist of:
1. Write a script outline with 3-5 main characters and key plot points.
2. Outline the three-act structure and suggest 2-3 twists.
3. Ensure the script aligns with the specified genre and target audience.
"""
))

casting_director_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
    model_id='global.anthropic.claude-haiku-4-5-20251001-v1:0',
    name="CastingDirectorAgent",
    description="""\
You are a talented casting director. Given a script outline and character descriptions,\
suggest suitable actors for the main roles, considering their past performances and current availability.

Your tasks consist of:
1. Suggest 1-2 actors for each main role.
2. Check actors' current status using the search_web tool.
3. Provide a brief explanation for each casting suggestion.
4. Consider diversity and representation in your casting choices.
5. Provide a final response with all the actors you suggest for the main roles.
""",
    tool_config={
        'tool': AgentTools(tools=[search_web_tool]),
        'toolMaxRecursions': 20,
    },
    save_chat=False
))

movie_producer_supervisor = BedrockLLMAgent(BedrockLLMAgentOptions(
    model_id='us.anthropic.claude-haiku-4-5-20251001-v1:0',
    name='MovieProducerAgent',
    description="""\
Experienced movie producer overseeing script and casting.

Your tasks consist of:
1. Ask ScriptWriter Agent for a script outline based on the movie idea.
2. Pass the outline to CastingDirectorAgent for casting suggestions.
3. Summarize the script outline and casting suggestions.
4. Provide a concise movie concept overview.
5. Make sure to respond with a markdown format without mentioning it.
"""
))

supervisor = SupervisorAgent(SupervisorAgentOptions(
    name="SupervisorAgent",
    description="My Supervisor agent description",
    lead_agent=movie_producer_supervisor,
    team=[script_writer_agent, casting_director_agent],
    trace=True
))

# Define async function for handling requests
async def handle_request(_orchestrator: AgentSquad, _user_input: str, _user_id: str, _session_id: str):
    classifier_result = ClassifierResult(selected_agent=supervisor, confidence=1.0)

    response: AgentResponse = await _orchestrator.agent_process_request(_user_input, _user_id, _session_id, classifier_result)

    # Print metadata
    print("\nMetadata:")
    print(f"Selected Agent: {response.metadata.agent_name}")
    if isinstance(response, AgentResponse) and response.streaming is False:
        # Handle regular response
        if isinstance(response.output, str):
            return response.output
        elif isinstance(response.output, ConversationMessage):
            return response.output.content[0].get('text')

# Initialize the orchestrator
orchestrator = AgentSquad(options=AgentSquadConfig(
    LOG_AGENT_CHAT=True,
    LOG_CLASSIFIER_CHAT=True,
    LOG_CLASSIFIER_RAW_OUTPUT=True,
    LOG_CLASSIFIER_OUTPUT=True,
    LOG_EXECUTION_TIMES=True,
    MAX_RETRIES=3,
    USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED=True,
    MAX_MESSAGE_PAIRS_PER_AGENT=10,
))

USER_ID = str(uuid.uuid4())
SESSION_ID = str(uuid.uuid4())

# Input fields for the movie concept
movie_idea = st.text_area("Describe your movie idea in a few sentences:")
genre = st.selectbox("Select the movie genre:", ["Action", "Comedy", "Drama", "Sci-Fi", "Horror", "Romance", "Thriller"])
target_audience = st.selectbox("Select the target audience:", ["General", "Children", "Teenagers", "Adults", "Mature"])
estimated_runtime = st.slider("Estimated runtime (in minutes):", 30, 180, 120)

# Process the movie concept
if st.button("Develop Movie Concept"):
    with st.spinner("Developing movie concept..."):
        input_text = (
            f"Movie idea: {movie_idea}, Genre: {genre}, "
            f"Target audience: {target_audience}, Estimated runtime: {estimated_runtime} minutes"
        )
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(handle_request(orchestrator, input_text, USER_ID, SESSION_ID))
        st.write(response)
