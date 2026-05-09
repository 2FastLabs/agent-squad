# Claude Multi-Agent Quorum

Three Claude agents deliberate in parallel and vote on whether to enforce a governance action. Enforcement only happens when 2/3 agents agree.

## Pattern

```
Signal → [Safety Agent] ─┐
       → [Audit Agent]  ─┼→ Vote tally → ENFORCE / DISMISS / WARN
       → [Cost Agent]   ─┘
         (parallel, ThreadPoolExecutor)
```

Each agent evaluates the signal from a different angle:
- **Safety** — blast radius and exfiltration risk
- **Audit** — CloudTrail evidence and policy violations
- **Cost** — financial exposure (threshold-based)

Agent failures count as `ABSTAIN` — one failure can never block a clear majority.

## Run

```bash
pip install anthropic pydantic
python claude_quorum_agent.py
```

`DEMO_MODE=true` by default — runs without an API key. Set `ANTHROPIC_API_KEY` and `DEMO_MODE=false` for live Claude agents.

## Why this pattern matters

Single-agent decisions are risky for high-stakes, irreversible actions. A quorum:
- Prevents false positives — no single agent acts unilaterally
- Provides explainability — three independent reasoning chains
- Degrades safely — one failure cannot block a clear majority

## Source

Extracted from [ai-sentinel-ecosystem](https://github.com/TanishkaMarrott/ai-sentinel-ecosystem) — validated across 30 scenarios with 98.4% detection accuracy and zero false-positive quarantines.