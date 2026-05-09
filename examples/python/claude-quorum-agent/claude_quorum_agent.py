"""
Claude Multi-Agent Quorum — AWS Cost Governance Example

Three Claude agents deliberate in parallel, each evaluating a suspicious AWS
event from a different specialist angle. Enforcement only happens when at least
2/3 agents agree (majority vote).

Pattern:
- Safety agent  — blast radius and exfiltration risk
- Audit agent   — CloudTrail evidence verification
- Cost agent    — financial exposure estimation
- Orchestrator  — runs all three concurrently, tallies votes

DEMO_MODE=true by default — runs without API calls for zero-cost testing.
Set ANTHROPIC_API_KEY and DEMO_MODE=false to run with live Claude agents.
"""

import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from pydantic import BaseModel
import anthropic

DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-7")


# --- Schemas ---

class Vote(str, Enum):
    APPROVE = "APPROVE"   # enforce (quarantine / restrict)
    REJECT = "REJECT"     # dismiss, no action
    ABSTAIN = "ABSTAIN"   # agent failed or uncertain


class AgentVerdict(BaseModel):
    agent: str
    vote: Vote
    confidence: float   # 0.0–1.0
    reasoning: str


class QuorumResult(BaseModel):
    signal: dict
    verdicts: list[AgentVerdict]
    decision: str   # ENFORCE | DISMISS | WARN
    approve_count: int
    reject_count: int


# --- Demo verdicts (no API calls) ---

DEMO_VERDICTS = {
    "safety": AgentVerdict(
        agent="safety",
        vote=Vote.APPROVE,
        confidence=0.92,
        reasoning="NAT Gateway creation enables data exfiltration. Blast radius: HIGH.",
    ),
    "audit": AgentVerdict(
        agent="audit",
        vote=Vote.APPROVE,
        confidence=0.88,
        reasoning="CloudTrail shows ec2:CreateNatGateway at 02:14 UTC. No change request found.",
    ),
    "cost": AgentVerdict(
        agent="cost",
        vote=Vote.REJECT,
        confidence=0.60,
        reasoning="Estimated $45/month — below $50 threshold. Cost alone does not justify enforcement.",
    ),
}


# --- Agent runners ---

def run_agent(role: str, signal: dict, client: anthropic.Anthropic) -> AgentVerdict:
    if DEMO_MODE:
        return DEMO_VERDICTS[role]

    prompts = {
        "safety": (
            "You are a cloud security safety agent. Evaluate this AWS event signal for "
            "blast radius and data exfiltration risk. Reply with JSON: "
            '{"vote":"APPROVE|REJECT|ABSTAIN","confidence":0.0-1.0,"reasoning":"..."}'
        ),
        "audit": (
            "You are a cloud audit agent. Evaluate this AWS event for CloudTrail evidence "
            "and policy violations. Reply with JSON: "
            '{"vote":"APPROVE|REJECT|ABSTAIN","confidence":0.0-1.0,"reasoning":"..."}'
        ),
        "cost": (
            "You are a cloud cost agent. Evaluate this AWS event for financial exposure. "
            "APPROVE if estimated monthly cost exceeds $50. Reply with JSON: "
            '{"vote":"APPROVE|REJECT|ABSTAIN","confidence":0.0-1.0,"reasoning":"..."}'
        ),
    }

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=256,
            messages=[{
                "role": "user",
                "content": f"{prompts[role]}\n\nSignal: {json.dumps(signal)}",
            }],
        )
        raw = json.loads(response.content[0].text)
        return AgentVerdict(agent=role, **raw)
    except Exception as e:
        return AgentVerdict(agent=role, vote=Vote.ABSTAIN, confidence=0.0, reasoning=str(e))


# --- Quorum orchestrator ---

def run_quorum(signal: dict) -> QuorumResult:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", "demo"))
    roles = ["safety", "audit", "cost"]

    verdicts = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(run_agent, role, signal, client): role for role in roles}
        for future in as_completed(futures):
            verdicts.append(future.result())

    approve = sum(1 for v in verdicts if v.vote == Vote.APPROVE)
    reject = sum(1 for v in verdicts if v.vote == Vote.REJECT)

    if approve >= 2:
        decision = "ENFORCE"
    elif reject >= 2:
        decision = "DISMISS"
    else:
        decision = "WARN"

    return QuorumResult(
        signal=signal,
        verdicts=verdicts,
        decision=decision,
        approve_count=approve,
        reject_count=reject,
    )


# --- Demo ---

if __name__ == "__main__":
    signal = {
        "account_id": "123456789012",
        "event": "ec2:CreateNatGateway",
        "region": "us-east-1",
        "timestamp": "2026-05-09T02:14:00Z",
        "user": "lab-user-001",
    }

    print(f"Signal: {signal['event']} in {signal['account_id']}\n")
    result = run_quorum(signal)

    for v in result.verdicts:
        print(f"  [{v.vote}] {v.agent} (confidence: {v.confidence:.0%}): {v.reasoning}")

    print(f"\nDecision: {result.decision} ({result.approve_count} APPROVE / {result.reject_count} REJECT)")

    if result.decision == "ENFORCE":
        print("→ Action: Apply SCP quarantine to account")
    elif result.decision == "DISMISS":
        print("→ Action: No action taken")
    else:
        print("→ Action: Route to human review queue")