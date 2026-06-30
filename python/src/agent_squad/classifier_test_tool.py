"""classifier_test_tool.py — Evaluate classifier routing accuracy and performance.

Usage::

    python -m agent_squad.classifier_test_tool --config tests.json

The JSON file describes agents and test cases.  Example::

    {
        "classifier": {
            "type": "bedrock",
            "model_id": "us.anthropic.claude-3-5-haiku-20241022-v1:0"
        },
        "agents": [
            {"name": "flight_agent",  "description": "Books and manages flights"},
            {"name": "weather_agent", "description": "Answers weather questions"},
            {"name": "billing_agent", "description": "Handles billing and payments"}
        ],
        "tests": [
            {"input": "I need to book a flight to Paris", "expected": "flight_agent"},
            {"input": "What is the weather in New York?",  "expected": "weather_agent"},
            {"input": "I was charged twice", "expected": "billing_agent",
             "min_confidence": 0.8}
        ]
    }

Fields
------
classifier.type       : "bedrock" | "anthropic" | "openai"  (default: "bedrock")
classifier.model_id   : model identifier passed to the chosen classifier (optional)
agents[].name         : unique agent key
agents[].description  : description used for routing
tests[].input         : user message (string)
tests[].expected      : agent name the classifier should select
tests[].min_confidence: minimum acceptable confidence 0–1 (optional, default 0)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from agent_squad.classifiers import Classifier, ClassifierCallbacks, ClassifierResult
from agent_squad.agents import Agent, AgentOptions
from agent_squad.types import ConversationMessage


# ---------------------------------------------------------------------------
# Minimal stub agent used only for registration — no real LLM calls.
# ---------------------------------------------------------------------------
class _StubAgent(Agent):
    async def process_request(self, input_text, user_id, session_id, chat_history, additional_params=None):
        return ConversationMessage(role="assistant", content=[{"text": ""}])


# ---------------------------------------------------------------------------
# Callback that captures token usage from on_classifier_stop kwargs.
# ---------------------------------------------------------------------------
@dataclass
class _MetricsCallback(ClassifierCallbacks):
    input_tokens: int = 0
    output_tokens: int = 0

    async def on_classifier_stop(self, name, output, **kwargs):
        usage = kwargs.get("usage") or {}
        self.input_tokens += usage.get("inputTokens", 0)
        self.output_tokens += usage.get("outputTokens", 0)


# ---------------------------------------------------------------------------
# Per-test result
# ---------------------------------------------------------------------------
@dataclass
class ClassifierTestResult:
    index: int
    input_text: str
    expected: str
    routed_to: str
    confidence: float
    min_confidence: float
    latency_ms: float
    input_tokens: int
    output_tokens: int
    passed: bool
    failure_reason: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_classifier(cfg: dict) -> Classifier:
    classifier_type = cfg.get("type", "bedrock").lower()
    model_id = cfg.get("model_id")

    if classifier_type == "bedrock":
        try:
            from agent_squad.classifiers import BedrockClassifier, BedrockClassifierOptions
        except ImportError as e:
            sys.exit(f"bedrock classifier requires the 'aws' extra: {e}")
        opts = BedrockClassifierOptions()
        if model_id:
            opts.model_id = model_id
        return BedrockClassifier(opts)

    if classifier_type == "anthropic":
        try:
            from agent_squad.classifiers import AnthropicClassifier, AnthropicClassifierOptions
        except ImportError as e:
            sys.exit(f"anthropic classifier requires the 'anthropic' extra: {e}")
        opts = AnthropicClassifierOptions()
        if model_id:
            opts.model_id = model_id
        return AnthropicClassifier(opts)

    if classifier_type == "openai":
        try:
            from agent_squad.classifiers import OpenAIClassifier, OpenAIClassifierOptions
        except ImportError as e:
            sys.exit(f"openai classifier requires the 'openai' extra: {e}")
        opts = OpenAIClassifierOptions()
        if model_id:
            opts.model_id = model_id
        return OpenAIClassifier(opts)

    sys.exit(f"Unknown classifier type '{classifier_type}'. Choose: bedrock, anthropic, openai")


def _print_result(r: ClassifierTestResult) -> None:
    status = "✅ PASSED" if r.passed else "❌ FAILED"
    print(f"\nTest {r.index}: {status}")
    print(f"  Input      : {r.input_text}")
    print(f"  Expected   : {r.expected}")
    print(f"  Routed to  : {r.routed_to}")
    print(f"  Confidence : {r.confidence:.2f}" +
          (f"  (min: {r.min_confidence:.2f})" if r.min_confidence > 0 else ""))
    print(f"  Latency    : {r.latency_ms:.0f} ms")
    print(f"  Tokens     : {r.input_tokens} in / {r.output_tokens} out")
    if r.failure_reason:
        print(f"  Reason     : {r.failure_reason}")


def _print_summary(results: list[ClassifierTestResult]) -> None:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    avg_latency = sum(r.latency_ms for r in results) / total if total else 0.0
    avg_conf = sum(r.confidence for r in results) / total if total else 0.0
    total_in = sum(r.input_tokens for r in results)
    total_out = sum(r.output_tokens for r in results)

    high = sum(1 for r in results if r.confidence >= 0.8)
    med  = sum(1 for r in results if 0.5 <= r.confidence < 0.8)
    low  = sum(1 for r in results if r.confidence < 0.5)

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    pct = f"{passed/total*100:.1f}%" if total else "n/a"
    print(f"  Tests passed   : {passed}/{total}  ({pct})")
    print(f"  Avg latency    : {avg_latency:.0f} ms")
    print(f"  Avg confidence : {avg_conf:.2f}")
    print(f"  Total tokens   : {total_in} in / {total_out} out")
    print(f"\nConfidence distribution:")
    print(f"  High (≥0.8)    : {high}")
    print(f"  Medium (0.5–0.8): {med}")
    print(f"  Low (<0.5)     : {low}")

    if passed < total:
        print("\nFailed tests:")
        for r in results:
            if not r.passed:
                print(f"  Test {r.index}: expected={r.expected}  got={r.routed_to}  ({r.failure_reason})")


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------
async def run(config_path: Path) -> list[ClassifierTestResult]:
    raw = json.loads(config_path.read_text())

    classifier = _build_classifier(raw.get("classifier", {}))

    # Register stub agents so the classifier can route between them.
    agents: dict[str, Agent] = {}
    for a in raw.get("agents", []):
        stub = _StubAgent(AgentOptions(name=a["name"], description=a["description"]))
        agents[a["name"]] = stub
        classifier.set_agents({a["name"]: stub})

    print(f"Classifier : {type(classifier).__name__}")
    print(f"Agents     : {', '.join(agents)}")
    print(f"Test cases : {len(raw.get('tests', []))}")
    print("=" * 50)

    results: list[ClassifierTestResult] = []

    for i, tc in enumerate(raw.get("tests", []), start=1):
        input_text: str = tc["input"]
        expected: str = tc["expected"]
        min_conf: float = float(tc.get("min_confidence", 0.0))

        cb = _MetricsCallback()
        classifier.callbacks = cb

        t0 = time.perf_counter()
        try:
            result: ClassifierResult = await classifier.classify(input_text, [])
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            tr = ClassifierTestResult(
                index=i,
                input_text=input_text,
                expected=expected,
                routed_to="ERROR",
                confidence=0.0,
                min_confidence=min_conf,
                latency_ms=latency_ms,
                input_tokens=cb.input_tokens,
                output_tokens=cb.output_tokens,
                passed=False,
                failure_reason=str(exc),
            )
            results.append(tr)
            _print_result(tr)
            continue

        latency_ms = (time.perf_counter() - t0) * 1000
        routed_to = result.selected_agent.name if result.selected_agent else "none"
        confidence = float(result.confidence)

        correct = routed_to == expected
        meets_conf = confidence >= min_conf
        passed = correct and meets_conf

        failure_reason = ""
        if not correct:
            failure_reason = f"misrouted to '{routed_to}'"
        elif not meets_conf:
            failure_reason = f"confidence {confidence:.2f} below minimum {min_conf:.2f}"

        tr = ClassifierTestResult(
            index=i,
            input_text=input_text,
            expected=expected,
            routed_to=routed_to,
            confidence=confidence,
            min_confidence=min_conf,
            latency_ms=latency_ms,
            input_tokens=cb.input_tokens,
            output_tokens=cb.output_tokens,
            passed=passed,
            failure_reason=failure_reason,
        )
        results.append(tr)
        _print_result(tr)

    _print_summary(results)
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test Agent Squad classifier routing accuracy and performance."
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="FILE",
        help="Path to the JSON test configuration file.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"Config file not found: {config_path}")

    results = asyncio.run(run(config_path))
    failed = sum(1 for r in results if not r.passed)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
