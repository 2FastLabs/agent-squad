import uuid
import asyncio
import argparse
from queue import Queue
from threading import Thread

from agent_squad.orchestrator import AgentSquad, AgentResponse, AgentSquadConfig
from agent_squad.types import ConversationMessage
from agent_squad.classifiers import BedrockClassifier, BedrockClassifierOptions
from agent_squad.storage import DynamoDbChatStorage
from agent_squad.agents import (
    BedrockLLMAgent,
    AgentResponse,
    AgentCallbacks,
    BedrockLLMAgentOptions,
)

from typing import Dict, List, Any

from product_search_agent import ProductSearchAgent, ProductSearchAgentOptions
from prompts import RETURNS_PROMPT, GREETING_AGENT_PROMPT


class MyCustomHandler(AgentCallbacks):
    def __init__(self, queue) -> None:
        super().__init__()
        self._queue = queue
        self._stop_signal = None

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        self._queue.put(token)

    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        print("generation started")

    async def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        print("\n\ngeneration concluded")
        self._queue.put(self._stop_signal)

def setup_orchestrator(streamer_queue):

    classifier = BedrockClassifier(BedrockClassifierOptions(
         model_id='global.anthropic.claude-haiku-4-5-20251001-v1:0',
    ))


    orchestrator = AgentSquad(options=AgentSquadConfig(
        LOG_AGENT_CHAT=True,
        LOG_CLASSIFIER_CHAT=True,
        LOG_CLASSIFIER_RAW_OUTPUT=True,
        LOG_CLASSIFIER_OUTPUT=True,
        LOG_EXECUTION_TIMES=True,
        MAX_RETRIES=3,
        USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED=False,
        NO_SELECTED_AGENT_MESSAGE = """
I'm not quite sure how to help with that. Could you please:

- Provide more details, or
- Rephrase your question?

If you're unsure where to start, try saying **"hello"** to see:

- A list of available agents
- Their specific roles and capabilities

This will help you understand the kinds of questions and topics our system can assist you with.
""",
        MAX_MESSAGE_PAIRS_PER_AGENT=10
    ),
    classifier = classifier
     )

    product_search_agent = ProductSearchAgent(ProductSearchAgentOptions(
        name="Product Search Agent",
        description="Specializes in e-commerce product searches and listings. Handles queries about finding specific products, product rankings, specifications, price comparisons within an online shopping context. Use this agent for shopping-related queries and product discovery in a retail environment.",
        model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0",
        save_chat=True,
    ))

    my_handler = MyCustomHandler(streamer_queue)
    returns_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
        name="Returns and Terms Assistant",
        streaming=True,
        description="Specializes in explaining return policies, refund processes, and terms & conditions. Provides clear guidance on customer rights, warranty claims, and special cases while maintaining up-to-date knowledge of consumer protection regulations and e-commerce best practices.",
        model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0",
        #TODO SET a retriever to fetch data from a knowledge base
        callbacks=my_handler
    ))

    returns_agent.set_system_prompt(RETURNS_PROMPT)

    orchestrator.add_agent(product_search_agent)
    orchestrator.add_agent(returns_agent)

    agents = orchestrator.get_all_agents()

    greeting_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
        name="Greeting agent",
        streaming=True,
        description="Says hello and lists the available agents",
        model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0",
        save_chat=False,
        callbacks=my_handler
    ))

    agent_list = "\n".join([f"{i}-{info['name']}: {info['description']}" for i, (_, info) in enumerate(agents.items(), 1)])

    greeting_prompt = GREETING_AGENT_PROMPT(agent_list)
    greeting_agent.set_system_prompt(greeting_prompt)

    orchestrator.add_agent(greeting_agent)
    return orchestrator

async def start_generation(query, user_id, session_id, streamer_queue):
    try:
        # Create a new orchestrator for this query
        orchestrator = setup_orchestrator(streamer_queue)

        response = await orchestrator.route_request(query, user_id, session_id)
        if isinstance(response, AgentResponse) and response.streaming is False:
            if isinstance(response.output, str):
                streamer_queue.put(response.output)
            elif isinstance(response.output, ConversationMessage):
                streamer_queue.put(response.output.content[0].get('text'))
    except Exception as e:
        print(f"Error in start_generation: {e}")
    finally:
        streamer_queue.put(None)  # Signal the end of the response

async def response_generator(query, user_id, session_id):
    streamer_queue = Queue()

    # Start the generation process in a separate thread
    Thread(target=lambda: asyncio.run(start_generation(query, user_id, session_id, streamer_queue))).start()

    #print("Waiting for the response...")
    while True:
        try:
            value = await asyncio.get_event_loop().run_in_executor(None, streamer_queue.get)
            if value is None:
                break
            yield value
            streamer_queue.task_done()
        except Exception as e:
            print(f"Error in response_generator: {e}")
            break

async def run_chatbot():
    user_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())

    while True:
        query = input("\nEnter your query (or 'quit' to exit): ").strip()
        if query.lower() == 'quit':
            break
        try:
            async for token in response_generator(query, user_id, session_id):
                print(token, end='', flush=True)
            print()  # New line after response
        except Exception as error:
            print("Error:", error)

if __name__ == "__main__":
    asyncio.run(run_chatbot())