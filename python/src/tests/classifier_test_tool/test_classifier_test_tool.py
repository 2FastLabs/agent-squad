"""Unit tests for classifier_test_tool — all AWS calls are mocked."""
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from agent_squad.classifier_test_tool import run, ClassifierTestResult
from agent_squad.classifiers import ClassifierResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(name="flight_agent"):
    agent = MagicMock()
    agent.name = name
    return agent


def _classifier_result(agent_name: str, confidence: float = 0.95):
    return ClassifierResult(
        selected_agent=_make_agent(agent_name),
        confidence=confidence,
    )


def _write_config(tmp_path: Path, cfg: dict) -> Path:
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    return p


BASE_CFG = {
    "classifier": {"type": "bedrock"},
    "agents": [
        {"name": "flight_agent", "description": "Books flights"},
        {"name": "weather_agent", "description": "Weather info"},
    ],
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_correct_routing_passes(tmp_path):
    cfg = {**BASE_CFG, "tests": [{"input": "book a flight", "expected": "flight_agent"}]}
    config_path = _write_config(tmp_path, cfg)

    with patch("agent_squad.classifier_test_tool._build_classifier") as mock_build:
        classifier = MagicMock()
        classifier.set_agents = MagicMock()
        classifier.callbacks = None
        classifier.classify = AsyncMock(return_value=_classifier_result("flight_agent", 0.95))
        mock_build.return_value = classifier

        results = await run(config_path)

    assert len(results) == 1
    assert results[0].passed is True
    assert results[0].routed_to == "flight_agent"
    assert results[0].confidence == 0.95


@pytest.mark.asyncio
async def test_wrong_routing_fails(tmp_path):
    cfg = {**BASE_CFG, "tests": [{"input": "what's the weather?", "expected": "weather_agent"}]}
    config_path = _write_config(tmp_path, cfg)

    with patch("agent_squad.classifier_test_tool._build_classifier") as mock_build:
        classifier = MagicMock()
        classifier.set_agents = MagicMock()
        classifier.callbacks = None
        classifier.classify = AsyncMock(return_value=_classifier_result("flight_agent", 0.7))
        mock_build.return_value = classifier

        results = await run(config_path)

    assert results[0].passed is False
    assert "misrouted" in results[0].failure_reason


@pytest.mark.asyncio
async def test_min_confidence_below_threshold_fails(tmp_path):
    cfg = {**BASE_CFG, "tests": [
        {"input": "book a flight", "expected": "flight_agent", "min_confidence": 0.9}
    ]}
    config_path = _write_config(tmp_path, cfg)

    with patch("agent_squad.classifier_test_tool._build_classifier") as mock_build:
        classifier = MagicMock()
        classifier.set_agents = MagicMock()
        classifier.callbacks = None
        classifier.classify = AsyncMock(return_value=_classifier_result("flight_agent", 0.7))
        mock_build.return_value = classifier

        results = await run(config_path)

    assert results[0].passed is False
    assert "confidence" in results[0].failure_reason


@pytest.mark.asyncio
async def test_min_confidence_met_passes(tmp_path):
    cfg = {**BASE_CFG, "tests": [
        {"input": "book a flight", "expected": "flight_agent", "min_confidence": 0.8}
    ]}
    config_path = _write_config(tmp_path, cfg)

    with patch("agent_squad.classifier_test_tool._build_classifier") as mock_build:
        classifier = MagicMock()
        classifier.set_agents = MagicMock()
        classifier.callbacks = None
        classifier.classify = AsyncMock(return_value=_classifier_result("flight_agent", 0.95))
        mock_build.return_value = classifier

        results = await run(config_path)

    assert results[0].passed is True


@pytest.mark.asyncio
async def test_classifier_exception_recorded_as_failure(tmp_path):
    cfg = {**BASE_CFG, "tests": [{"input": "book a flight", "expected": "flight_agent"}]}
    config_path = _write_config(tmp_path, cfg)

    with patch("agent_squad.classifier_test_tool._build_classifier") as mock_build:
        classifier = MagicMock()
        classifier.set_agents = MagicMock()
        classifier.callbacks = None
        classifier.classify = AsyncMock(side_effect=RuntimeError("connection refused"))
        mock_build.return_value = classifier

        results = await run(config_path)

    assert results[0].passed is False
    assert results[0].routed_to == "ERROR"
    assert "connection refused" in results[0].failure_reason


@pytest.mark.asyncio
async def test_multiple_tests_mixed_results(tmp_path):
    cfg = {**BASE_CFG, "tests": [
        {"input": "flight to Paris", "expected": "flight_agent"},
        {"input": "will it rain?", "expected": "weather_agent"},
    ]}
    config_path = _write_config(tmp_path, cfg)

    responses = [
        _classifier_result("flight_agent", 0.95),
        _classifier_result("flight_agent", 0.60),  # wrong
    ]

    with patch("agent_squad.classifier_test_tool._build_classifier") as mock_build:
        classifier = MagicMock()
        classifier.set_agents = MagicMock()
        classifier.callbacks = None
        classifier.classify = AsyncMock(side_effect=responses)
        mock_build.return_value = classifier

        results = await run(config_path)

    assert len(results) == 2
    assert results[0].passed is True
    assert results[1].passed is False


@pytest.mark.asyncio
async def test_result_includes_latency_and_index(tmp_path):
    cfg = {**BASE_CFG, "tests": [{"input": "a flight", "expected": "flight_agent"}]}
    config_path = _write_config(tmp_path, cfg)

    with patch("agent_squad.classifier_test_tool._build_classifier") as mock_build:
        classifier = MagicMock()
        classifier.set_agents = MagicMock()
        classifier.callbacks = None
        classifier.classify = AsyncMock(return_value=_classifier_result("flight_agent"))
        mock_build.return_value = classifier

        results = await run(config_path)

    assert results[0].index == 1
    assert results[0].latency_ms >= 0


@pytest.mark.asyncio
async def test_empty_tests_list(tmp_path):
    cfg = {**BASE_CFG, "tests": []}
    config_path = _write_config(tmp_path, cfg)

    with patch("agent_squad.classifier_test_tool._build_classifier") as mock_build:
        classifier = MagicMock()
        classifier.set_agents = MagicMock()
        classifier.callbacks = None
        mock_build.return_value = classifier

        results = await run(config_path)

    assert results == []
