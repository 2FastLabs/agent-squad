from __future__ import annotations
import math
import re
from dataclasses import dataclass
from typing import Optional

_STOPWORDS = frozenset([
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'this', 'that', 'these',
    'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
    'who', 'when', 'where', 'how', 'why', 'not', 'from', 'as', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down',
    'out', 'off', 'over', 'under', 'then', 'once', 'so',
])


@dataclass
class OverlapResult:
    overlap_percentage: str
    potential_conflict: str  # "High" | "Medium" | "Low"


@dataclass
class UniquenessScore:
    agent: str
    uniqueness_score: str


@dataclass
class AnalysisResult:
    pairwise_overlap: dict[str, OverlapResult]
    uniqueness_scores: list[UniquenessScore]


class AgentOverlapAnalyzer:
    """Analyses description overlap between agents using TF-IDF cosine similarity.

    Mirrors the TypeScript ``AgentOverlapAnalyzer``
    (``typescript/src/agentOverlapAnalyzer.ts``).

    Args:
        agents: Mapping of agent key to ``{"name": ..., "description": ...}``.
    """

    def __init__(self, agents: dict[str, dict[str, str]]) -> None:
        self._agents = agents

    def analyze_overlap(self) -> Optional[AnalysisResult]:
        """Run the overlap analysis and print results to stdout.

        Returns:
            :class:`AnalysisResult` when two or more agents are present,
            ``None`` otherwise.
        """
        agent_names = list(self._agents.keys())
        agent_descriptions = [self._agents[k]['description'] for k in agent_names]

        if len(agent_names) < 2:
            print("Agent Overlap Analysis requires at least two agents.")
            print(f"Current number of agents: {len(agent_names)}")
            if len(agent_names) == 1:
                print("\nSingle Agent Information:")
                print(f"Agent Name: {agent_names[0]}")
                print(f"Description: {agent_descriptions[0]}")
            return None

        tokenized = [self._tokenize(d) for d in agent_descriptions]
        tfidf_vectors = self._build_tfidf(tokenized)

        pairwise_overlap: dict[str, OverlapResult] = {}
        for i in range(len(agent_names)):
            for j in range(i + 1, len(agent_names)):
                similarity = self._cosine_similarity(tfidf_vectors[i], tfidf_vectors[j])
                key = f"{agent_names[i]}__{agent_names[j]}"
                pairwise_overlap[key] = OverlapResult(
                    overlap_percentage=f"{similarity * 100:.2f}%",
                    potential_conflict="High" if similarity > 0.3 else "Medium" if similarity > 0.1 else "Low",
                )

        uniqueness_scores: list[UniquenessScore] = []
        for i, name in enumerate(agent_names):
            similarities: list[float] = []
            for j in range(len(agent_names)):
                if i == j:
                    continue
                lo, hi = min(i, j), max(i, j)
                key = f"{agent_names[lo]}__{agent_names[hi]}"
                result = pairwise_overlap.get(key)
                if result:
                    similarities.append(float(result.overlap_percentage.rstrip('%')) / 100)
            avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
            uniqueness_scores.append(UniquenessScore(
                agent=name,
                uniqueness_score=f"{(1 - avg_sim) * 100:.2f}%",
            ))

        print("Pairwise Overlap Results:")
        print("_________________________\n")
        for key, result in pairwise_overlap.items():
            agent1, agent2 = key.split("__")
            print(f"{agent1} - {agent2}:")
            print(f"- Overlap Percentage - {result.overlap_percentage}")
            print(f"- Potential Conflict - {result.potential_conflict}\n")

        print("\nUniqueness Scores:")
        print("_________________\n")
        for score in uniqueness_scores:
            print(f"Agent: {score.agent}, Uniqueness Score: {score.uniqueness_score}")

        return AnalysisResult(
            pairwise_overlap=pairwise_overlap,
            uniqueness_scores=uniqueness_scores,
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        tokens = re.split(r'\W+', text.lower())
        return [t for t in tokens if t and t not in _STOPWORDS]

    @staticmethod
    def _build_tfidf(documents: list[list[str]]) -> list[dict[str, float]]:
        n = len(documents)

        tf_vectors: list[dict[str, float]] = []
        for doc in documents:
            counts: dict[str, int] = {}
            for word in doc:
                counts[word] = counts.get(word, 0) + 1
            total = len(doc) or 1
            tf_vectors.append({w: c / total for w, c in counts.items()})

        df: dict[str, int] = {}
        for doc in documents:
            for word in set(doc):
                df[word] = df.get(word, 0) + 1
        idf = {w: math.log((n + 1) / (cnt + 1)) + 1 for w, cnt in df.items()}

        return [{w: score * idf.get(w, 1.0) for w, score in tf.items()} for tf in tf_vectors]

    @staticmethod
    def _cosine_similarity(vec1: dict[str, float], vec2: dict[str, float]) -> float:
        terms = set(vec1) | set(vec2)
        dot = sum(vec1.get(t, 0.0) * vec2.get(t, 0.0) for t in terms)
        mag1 = math.sqrt(sum(v * v for v in vec1.values()))
        mag2 = math.sqrt(sum(v * v for v in vec2.values()))
        return dot / (mag1 * mag2) if mag1 and mag2 else 0.0
