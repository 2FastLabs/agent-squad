# %% Imports:
import asyncio

from multi_agent_orchestrator.agents import (
    AgentResponse,
    BedrockLLMAgent,
    BedrockLLMAgentOptions,
    ChainAgent,
    ChainAgentOptions,
)
from multi_agent_orchestrator.classifiers import (
    BedrockClassifier,
    BedrockClassifierOptions,
)
from multi_agent_orchestrator.orchestrator import (
    MultiAgentOrchestrator,
    OrchestratorConfig,
)
from parallel_agent_prompts import ANALYZER_PROMPT, JUDGE_PROMPT

from parallel_agent import ParallelAgent, ParallelAgentOptions

# %% Define a base query with placeholder user/session IDs to trigger the sentiment workflow:
QUERY = "Perform sentiment analysis on the provided social media posts from business sellers on an Amazon-style e-commerce website."
USER_ID = "user_123"
SESSION_ID = "session_456"

# %% Example input posts to use for sentiment analysis workflow:
INPUT_POSTS = {
    "original_post": "I'm considering switching to eco-friendly packaging for my products, but I'm worried about the increased costs. Has anyone made this transition? What was your experience like?",
    "reply_1": "We made the switch last year and it was definitely worth it! Yes, there's an initial cost increase, but we've seen a boost in customer loyalty and positive reviews. It's also opened up a new eco-conscious market for us.",
    "reply_2": "I tried it and had to switch back. The costs were too high and my customers weren't willing to pay more for the products. It might work for some niches, but it wasn't sustainable for my business.",
    "reply_3": "It's a tough decision, but I think it's the right move in the long run. We gradually transitioned over 6 months, which helped spread out the costs. Our customers appreciated the change and it's now a key part of our brand identity.",
    "reply_4": "I'm in the same boat as you. I want to make the switch but I'm hesitant. Has anyone found any good wholesale suppliers for eco-friendly packaging? The ones I've found so far are prohibitively expensive.",
    "reply_5": "We made the switch two years ago and haven't looked back. It was challenging at first, but we found creative ways to offset the costs, like optimizing our shipping processes. Plus, it's given us great marketing material!",
    "reply_6": "I think it really depends on your target market. Our customers are primarily young, urban professionals who value sustainability. For them, the eco-friendly packaging is a huge selling point. Know your audience before making the leap.",
    "reply_7": "Have you considered a hybrid approach? We use eco-friendly packaging for our premium line and standard packaging for our basic line. This allows customers to choose and has worked well for us.",
    "reply_8": "I'm against it. In my experience, most customers don't care about eco-friendly packaging - they care about getting their product quickly and cheaply. I think it's just a trend that will pass. Focus on your core business instead.",
    "reply_9": "We transitioned last quarter and it's been a mixed bag. On one hand, we've attracted new environmentally-conscious customers. On the other, we've had to increase our prices slightly. Overall, I think it was the right decision for us.",
    "reply_10": "Before you make the switch, do a thorough cost-benefit analysis. We found that while our packaging costs went up, we were able to reduce our overall waste, which ended up saving us money in the long run. It's not just about the immediate costs.",
}

# %% Create dictionary with model specs:
MODELS = {
    "analyzer_1": {
        "model_id": "cohere.command-r-plus-v1:0",
        "inference_config": {
            "maxTokens": 4096,
            "temperature": 0,
            "topK": 1,
            "topP": 1,
        },
    },
    "analyzer_2": {
        "model_id": "ai21.jamba-1-5-large-v1:0",
        "inference_config": {
            "maxTokens": 4096,
            "temperature": 0,
            "topK": 1,
            "topP": 1,
        },
    },
    "analyzer_3": {
        "model_id": "mistral.mistral-large-2402-v1:0",
        "inference_config": {
            "maxTokens": 4096,
            "temperature": 0,
            "topK": 1,
            "topP": 1,
        },
    },
    "judge": {
        "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "system_prompt": JUDGE_PROMPT,
        "inference_config": {
            "maxTokens": 4096,
            "temperature": 0,
            "topK": 1,
            "topP": 1,
        },
    },
}


# %% Set up request handler:
async def handle_request(
    orchestrator: MultiAgentOrchestrator,
    user_input: str,
    user_id: str,
    session_id: str,
):
    response: AgentResponse = await orchestrator.route_request(
        user_input,
        user_id,
        session_id,
    )

    metadata_output = f"\nSelected Agent: {response.metadata.agent_name}\n"
    print(metadata_output)

    response_output = f"\nAgent Response: {response.output.content[0]['text']}\n"
    print(response_output)

    return response


# %% Define main function:
async def main():
    # Set up individual analyzer agents that will be included in the parallel agent:
    print("Initializing analyzer agents ...")
    analyzer_agents = [
        BedrockLLMAgent(
            BedrockLLMAgentOptions(
                name=agent_name,
                streaming=False,
                description="Expert AI assistant designed to analyze open-text data and extract sentiment insights from social media posts",
                model_id=agent_config["model_id"],
                inference_config=agent_config["inference_config"],
            )
        )
        for agent_name, agent_config in MODELS.items()
        if agent_name != "judge"
    ]

    # Update system prompts for all analyzer agents:
    print("Updating system prompts for analyzer agents ...")
    for agent in analyzer_agents:
        agent.set_system_prompt(
            ANALYZER_PROMPT,
            {"INPUT_POSTS": INPUT_POSTS},
        )

    # Define the parallel agent using the analyzer agents:
    print("Spinning up parallel agent ...")
    parallel_agent = ParallelAgent(
        ParallelAgentOptions(
            name="AdvancedParallelAgent",
            description="A complex parallel agent that runs member agent operations concurrently",
            agents=analyzer_agents,
            save_chat=True,
        )
    )

    # Define the final judge agent:
    print("Defining judge agent ...")
    judge_agent = BedrockLLMAgent(
        BedrockLLMAgentOptions(
            name="judge",
            streaming=False,
            description="Expert AI assistant that specializes in verifying, improving, and finalizing sentiment analyses performed by other AI assistants",
            model_id=MODELS["judge"]["model_id"],
            inference_config=MODELS["judge"]["inference_config"],
        )
    )

    # Update system prompt for judge agent:
    print("Updating system prompt for judge agent ...")
    judge_agent.set_system_prompt(
        JUDGE_PROMPT,
        {"INPUT_POSTS": INPUT_POSTS},
    )

    # Establish the final chain agent to construct the final flow, such that:
    #   1. The parallel agent is run first, where each of the 3 individual analyzer agents concurrently perform their own sentiment analysis
    #   2. The final judge agent uses the outputs from the parallel agent (and thus, all analyzer agents) to render final results
    print("Establishing final chain agent ...")
    chain_agent = ChainAgent(
        ChainAgentOptions(
            name="Final Chain Agent",
            description="Advanced chain agent that links all final agents",
            agents=[parallel_agent, judge_agent],
            save_chat=True,
        )
    )

    # Set up classifier with orchestrator & add final chain agent:
    print("Setting up classifier ...")
    classifier = BedrockClassifier(
        BedrockClassifierOptions(
            model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        )
    )

    print("Creating MultiAgentOrchestrator with final chain agent ...")
    orchestrator = MultiAgentOrchestrator(
        options=OrchestratorConfig(
            LOG_AGENT_CHAT=True,
            LOG_CLASSIFIER_CHAT=True,
            LOG_CLASSIFIER_RAW_OUTPUT=True,
            LOG_CLASSIFIER_OUTPUT=True,
            LOG_EXECUTION_TIMES=True,
            MAX_RETRIES=3,
            USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED=True,
            MAX_MESSAGE_PAIRS_PER_AGENT=10,
        ),
        classifier=classifier,
    )

    orchestrator.add_agent(chain_agent)

    # Run the orchestrator using the base query:
    print(f"Running orchestrator using base query: {QUERY}...")
    await handle_request(
        orchestrator=orchestrator,
        user_input=QUERY,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )


# %% Main program call:
if __name__ == "__main__":
    asyncio.run(main())
