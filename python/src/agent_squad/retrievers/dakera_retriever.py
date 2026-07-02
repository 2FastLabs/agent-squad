import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent_squad.retrievers import Retriever


@dataclass
class DakeraRetrieverOptions:
    """Options for the Dakera retriever.

    Attributes:
        namespace: The Dakera namespace to query.
        api_key: Dakera API key (a ``dk-...`` token). Falls back to the
            ``DAKERA_API_KEY`` environment variable when not set.
        url: Base URL of the Dakera server. Falls back to the ``DAKERA_URL``
            environment variable and finally to ``http://localhost:3000``.
        top_k: Maximum number of results to return.
        filter: Optional Dakera metadata filter applied to the query.
    """

    namespace: str
    api_key: Optional[str] = None
    url: str = ""
    top_k: int = 10
    filter: Optional[Dict] = None


class DakeraRetriever(Retriever):
    """Retriever backed by a self-hosted `Dakera <https://dakera.ai>`_ memory server.

    Uses Dakera's text-query API (server-side embedding) to fetch the most
    relevant documents for a query, which agents can use as retrieval-augmented
    context.
    """

    def __init__(self, options: DakeraRetrieverOptions):
        super().__init__(options)
        self.options = options

        if not self.options.namespace:
            raise ValueError("namespace is required in options")

        api_key = self.options.api_key or os.getenv("DAKERA_API_KEY")
        if not api_key:
            raise ValueError("api_key is required (set it in options or the DAKERA_API_KEY env var)")
        url = self.options.url or os.getenv("DAKERA_URL") or "http://localhost:3000"

        try:
            from dakera import AsyncDakeraClient
        except ImportError as exc:
            raise ImportError(
                "The 'dakera' package is required to use DakeraRetriever. "
                "Install it with: pip install 'agent-squad[dakera]'"
            ) from exc

        self.client = AsyncDakeraClient(base_url=url, api_key=api_key)

    async def retrieve(
        self,
        text: str,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict] = None,
    ) -> List[Any]:
        """Retrieve the documents most relevant to ``text`` from Dakera.

        Returns the list of ``TextSearchResult`` objects (``.id``, ``.score``,
        ``.text``, ``.metadata``) from the Dakera text query.
        """
        if not text:
            raise ValueError("Input text is required for retrieve")

        response = await self.client.query_text(
            self.options.namespace,
            text=text,
            top_k=top_k or self.options.top_k,
            filter=metadata_filter if metadata_filter is not None else self.options.filter,
        )
        return response.results

    async def retrieve_and_combine_results(
        self,
        text: str,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict] = None,
    ) -> str:
        """Retrieve results for ``text`` and combine their text into one string."""
        results = await self.retrieve(text, top_k, metadata_filter)
        return self.combine_retrieval_results(results)

    async def retrieve_and_generate(self, text: str) -> Any:
        """Not supported: Dakera is a retrieval-only backend (no generation)."""
        raise NotImplementedError(
            "DakeraRetriever does not support retrieve_and_generate; use retrieve or retrieve_and_combine_results."
        )

    @staticmethod
    def combine_retrieval_results(results: List[Any]) -> str:
        """Join the ``text`` field of each result into a single newline-separated string."""
        return "\n".join(result.text for result in results if result is not None and getattr(result, "text", None))
