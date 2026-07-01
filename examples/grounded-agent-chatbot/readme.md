# Grounded Agent — small shopping chatbot (Python)

A minimal console chatbot built on `GroundedAgent`, the framework's two-LLM anti-hallucination
pattern:

- a **gatherer** calls tools (a tiny in-memory product catalog) and sees the raw data, but never
  speaks to the user;
- an isolated **presenter** writes the reply from **only** the curated tool output, so it can't
  invent a price, rating, or stock status;
- a chit-chat turn that calls no tools is answered by the gatherer directly, skipping the presenter.

## Run

```bash
export ANTHROPIC_API_KEY=sk-...
pip install -r requirements.txt
python main.py
```

## Try

```
you> wireless headphones under €100?
shop> The best match is Aurora Wireless Headphones at €89 — highly rated at 4.6★.

you> is the speaker in stock?
shop> The Nimbus Bluetooth Speaker (€59) is currently out of stock.

you> hi there
shop> Hi! Ask me about our headphones, speakers, or earbuds.   # no tool call → presenter skipped
```

The presenter only ever sees the curated tool feed plus your question, so every price and stock
figure it reports is grounded in what the catalog actually returned.

See the [Agent Squad documentation](https://2fastlabs.github.io/agent-squad/) — the **Grounded
Agent** page under Built-in Agents — for the full API.
