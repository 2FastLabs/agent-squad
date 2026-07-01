"""A small grounded shopping chatbot.

Two models, not one: a *gatherer* calls tools and sees the raw catalog data but never speaks to the
user; an isolated *presenter* writes the reply from ONLY the curated tool output, so it cannot invent
a price, rating, or stock status. A chit-chat turn that calls no tools is answered by the gatherer
directly, skipping the presenter.

Run:
    export ANTHROPIC_API_KEY=sk-...
    pip install -r requirements.txt
    python main.py
"""

import asyncio
import os

from agent_squad.agents import (
    AnthropicAgent,
    AnthropicAgentOptions,
    GroundedAgent,
    GroundedAgentOptions,
    PresenterPrompt,
)
from agent_squad.types import ConversationMessage, ParticipantRole
from agent_squad.utils import AgentTool, AgentTools

# --- A tiny in-memory product catalog (stands in for a real backend) ---------------------------
CATALOG = {
    "p1": {"id": "p1", "name": "Aurora Wireless Headphones", "price": 89.0, "rating": 4.6, "stock": 12},
    "p2": {"id": "p2", "name": "Nimbus Bluetooth Speaker", "price": 59.0, "rating": 4.3, "stock": 0},
    "p3": {"id": "p3", "name": "Zephyr Earbuds", "price": 39.0, "rating": 4.1, "stock": 40},
}


def search_products(query: str, max_price: float = 1000.0) -> list[dict]:
    """Search the catalog by name substring, cheapest first.

    :param query: text to match against product names
    :param max_price: only return products at or below this price
    """
    q = query.lower()
    hits = [p for p in CATALOG.values() if q in p["name"].lower() and p["price"] <= max_price]
    return sorted(hits, key=lambda p: p["price"])


def get_product(product_id: str) -> dict:
    """Look up a single product by id.

    :param product_id: the product id, e.g. "p1"
    """
    return CATALOG.get(product_id, {"error": f"no product with id {product_id}"})


tools = AgentTools([
    AgentTool(name="search_products", func=search_products, required=["query"]),
    AgentTool(name="get_product", func=get_product, required=["product_id"]),
])

# --- The two models ----------------------------------------------------------------------------
api_key = os.environ["ANTHROPIC_API_KEY"]

# The gatherer owns the tools and its own system prompt. It gathers facts and never speaks.
gatherer = AnthropicAgent(AnthropicAgentOptions(
    name="Shop brain",
    description="Gathers product facts by calling tools.",
    api_key=api_key,
    model_id="claude-3-5-sonnet-20240620",
    tool_config={"tool": tools, "toolMaxRecursions": 5},
    custom_system_prompt={"template": """
        You are the data brain of a shopping assistant.
        GATHER the facts needed to answer the user — never write the final reply.
        - Call whatever tools you need; you may chain several.
        - Use the chat history to resolve follow-ups ("cheaper ones?", "is it in stock?").
        - Never invent values. If a tool returns nothing, say so.
        - Do NOT address the user or format anything — the presenter does that.
    """},
))

# The presenter has NO tools. It speaks only from the curated feed, so it can't invent values.
# A cheaper/smaller model is a fine fit here.
presenter = AnthropicAgent(AnthropicAgentOptions(
    name="Shop voice",
    description="Presents gathered product facts to the user.",
    api_key=api_key,
    model_id="claude-3-5-haiku-20241022",
))

presenter_prompt = PresenterPrompt(
    default="Use ONLY the data provided; never invent a value. Be concise and friendly.",
    per_tool={
        "search_products": (
            "You are presenting product search results. Use ONLY the data block — never invent a "
            "price, rating, name, or stock status. Lead with the best match (name + price), then "
            "one standout detail. Two short sentences."
        ),
        "get_product": (
            "You are presenting one product. State its name, price, rating, and whether it is in "
            "stock, in one sentence. Use only the data provided."
        ),
    },
)

shop = GroundedAgent(GroundedAgentOptions(
    name="Shop",
    description="A grounded shopping assistant.",
    gatherer=gatherer,
    presenter=presenter,
    tools=tools,
    presenter_prompt=presenter_prompt,
))


# --- A minimal console chat loop ---------------------------------------------------------------
async def main() -> None:
    user_id, session_id = "demo-user", "demo-session"
    history: list[ConversationMessage] = []
    print("Shop assistant ready. Try: 'headphones under €100?' or 'is the speaker in stock?'")
    print("(empty line or Ctrl-C to quit)\n")

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            break

        response = await shop.process_request(user_input, user_id, session_id, history)
        reply = response.content[0]["text"]
        print(f"shop> {reply}\n")

        # Persist both sides so follow-ups ("cheaper ones?") have context.
        history.append(ConversationMessage(role=ParticipantRole.USER.value, content=[{"text": user_input}]))
        history.append(ConversationMessage(role=ParticipantRole.ASSISTANT.value, content=[{"text": reply}]))


if __name__ == "__main__":
    asyncio.run(main())
