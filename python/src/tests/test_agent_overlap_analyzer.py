import pytest
from agent_squad.agent_overlap_analyzer import (
    AgentOverlapAnalyzer,
    AnalysisResult,
    OverlapResult,
    UniquenessScore,
)

TRAVEL_AGENTS = {
    "flight_agent": {
        "name": "Flight Agent",
        "description": "Helps users search, book, and manage airline flights and tickets",
    },
    "hotel_agent": {
        "name": "Hotel Agent",
        "description": "Assists with hotel reservations, room selection, and accommodation bookings",
    },
    "weather_agent": {
        "name": "Weather Agent",
        "description": "Provides weather forecasts and climate information for travel destinations",
    },
}

SIMILAR_AGENTS = {
    "agent_a": {
        "name": "Agent A",
        "description": "Helps users book flights and airline tickets for travel",
    },
    "agent_b": {
        "name": "Agent B",
        "description": "Assists users to book flights and airline reservations for travel",
    },
}


def test_analyze_overlap_returns_analysis_result():
    analyzer = AgentOverlapAnalyzer(TRAVEL_AGENTS)
    result = analyzer.analyze_overlap()
    assert isinstance(result, AnalysisResult)


def test_pairwise_overlap_has_correct_number_of_pairs():
    analyzer = AgentOverlapAnalyzer(TRAVEL_AGENTS)
    result = analyzer.analyze_overlap()
    # 3 agents → 3 pairs C(3,2)
    assert len(result.pairwise_overlap) == 3


def test_pairwise_keys_are_formatted_with_double_underscore():
    analyzer = AgentOverlapAnalyzer(TRAVEL_AGENTS)
    result = analyzer.analyze_overlap()
    for key in result.pairwise_overlap:
        assert "__" in key
        parts = key.split("__")
        assert len(parts) == 2
        assert parts[0] in TRAVEL_AGENTS
        assert parts[1] in TRAVEL_AGENTS


def test_overlap_percentage_is_valid_string():
    analyzer = AgentOverlapAnalyzer(TRAVEL_AGENTS)
    result = analyzer.analyze_overlap()
    for overlap in result.pairwise_overlap.values():
        assert isinstance(overlap, OverlapResult)
        assert overlap.overlap_percentage.endswith("%")
        value = float(overlap.overlap_percentage.rstrip("%"))
        assert 0.0 <= value <= 100.0


def test_potential_conflict_levels_are_valid():
    analyzer = AgentOverlapAnalyzer(TRAVEL_AGENTS)
    result = analyzer.analyze_overlap()
    valid_levels = {"High", "Medium", "Low"}
    for overlap in result.pairwise_overlap.values():
        assert overlap.potential_conflict in valid_levels


def test_uniqueness_scores_count_matches_agent_count():
    analyzer = AgentOverlapAnalyzer(TRAVEL_AGENTS)
    result = analyzer.analyze_overlap()
    assert len(result.uniqueness_scores) == len(TRAVEL_AGENTS)


def test_uniqueness_score_is_valid_string():
    analyzer = AgentOverlapAnalyzer(TRAVEL_AGENTS)
    result = analyzer.analyze_overlap()
    for score in result.uniqueness_scores:
        assert isinstance(score, UniquenessScore)
        assert score.uniqueness_score.endswith("%")
        value = float(score.uniqueness_score.rstrip("%"))
        assert 0.0 <= value <= 100.0


def test_high_conflict_for_very_similar_agents():
    analyzer = AgentOverlapAnalyzer(SIMILAR_AGENTS)
    result = analyzer.analyze_overlap()
    overlap = list(result.pairwise_overlap.values())[0]
    assert overlap.potential_conflict == "High"


def test_returns_none_for_single_agent(capsys):
    agents = {"solo": {"name": "Solo", "description": "A single agent"}}
    analyzer = AgentOverlapAnalyzer(agents)
    result = analyzer.analyze_overlap()
    assert result is None
    output = capsys.readouterr().out
    assert "at least two agents" in output
    assert "solo" in output


def test_returns_none_for_empty_agents(capsys):
    analyzer = AgentOverlapAnalyzer({})
    result = analyzer.analyze_overlap()
    assert result is None
    output = capsys.readouterr().out
    assert "at least two agents" in output


def test_identical_descriptions_produce_high_overlap():
    agents = {
        "a1": {"name": "A1", "description": "handles customer billing and payment processing"},
        "a2": {"name": "A2", "description": "handles customer billing and payment processing"},
    }
    analyzer = AgentOverlapAnalyzer(agents)
    result = analyzer.analyze_overlap()
    overlap = list(result.pairwise_overlap.values())[0]
    assert overlap.potential_conflict == "High"
    assert float(overlap.overlap_percentage.rstrip("%")) > 90.0


def test_completely_different_descriptions_produce_low_overlap():
    agents = {
        "billing": {
            "name": "Billing",
            "description": "processes invoices payments transactions refunds financial accounting ledger",
        },
        "weather": {
            "name": "Weather",
            "description": "forecasts temperature precipitation humidity wind climate meteorology",
        },
    }
    analyzer = AgentOverlapAnalyzer(agents)
    result = analyzer.analyze_overlap()
    overlap = list(result.pairwise_overlap.values())[0]
    assert overlap.potential_conflict == "Low"
