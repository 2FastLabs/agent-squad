import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_squad.retrievers import DakeraRetriever, DakeraRetrieverOptions


@pytest.fixture
def dakera_client(monkeypatch):
    """Inject a fake ``dakera`` module so DakeraRetriever's lazy import picks up a mock."""
    client = MagicMock()
    client.query_text = AsyncMock()
    module = MagicMock()
    module.AsyncDakeraClient.return_value = client
    monkeypatch.setitem(sys.modules, "dakera", module)
    return module, client


def _retriever(**overrides):
    opts = DakeraRetrieverOptions(namespace="docs", api_key="dk-fake", **overrides)
    return DakeraRetriever(opts)


def test_init_requires_namespace(dakera_client):
    with pytest.raises(ValueError, match="namespace is required"):
        DakeraRetriever(DakeraRetrieverOptions(namespace="", api_key="dk-fake"))


def test_init_requires_api_key(monkeypatch):
    monkeypatch.delenv("DAKERA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="api_key is required"):
        DakeraRetriever(DakeraRetrieverOptions(namespace="docs"))


def test_init_uses_env_vars(dakera_client, monkeypatch):
    module, _ = dakera_client
    monkeypatch.setenv("DAKERA_API_KEY", "dk-env")
    monkeypatch.setenv("DAKERA_URL", "http://env-host:9999")
    DakeraRetriever(DakeraRetrieverOptions(namespace="docs"))
    module.AsyncDakeraClient.assert_called_once_with(
        base_url="http://env-host:9999", api_key="dk-env"
    )


def test_init_defaults_to_localhost(dakera_client, monkeypatch):
    module, _ = dakera_client
    monkeypatch.delenv("DAKERA_URL", raising=False)
    _retriever()
    module.AsyncDakeraClient.assert_called_once_with(
        base_url="http://localhost:3000", api_key="dk-fake"
    )


@pytest.mark.asyncio
async def test_retrieve(dakera_client):
    _, client = dakera_client
    client.query_text.return_value = SimpleNamespace(
        results=[SimpleNamespace(id="a", score=0.9, text="alpha")]
    )
    retriever = _retriever(top_k=5)
    results = await retriever.retrieve("hello")

    client.query_text.assert_awaited_once()
    call = client.query_text.await_args
    assert call.args[0] == "docs"
    assert call.kwargs["text"] == "hello"
    assert call.kwargs["top_k"] == 5
    assert results[0].text == "alpha"


@pytest.mark.asyncio
async def test_retrieve_uses_option_top_k_and_filter(dakera_client):
    _, client = dakera_client
    client.query_text.return_value = SimpleNamespace(results=[])
    retriever = _retriever(top_k=7, filter={"lang": {"$eq": "en"}})
    await retriever.retrieve("hello")
    call = client.query_text.await_args
    assert call.kwargs["top_k"] == 7
    assert call.kwargs["filter"] == {"lang": {"$eq": "en"}}


@pytest.mark.asyncio
async def test_retrieve_empty_text_raises(dakera_client):
    retriever = _retriever()
    with pytest.raises(ValueError, match="Input text is required"):
        await retriever.retrieve("")


@pytest.mark.asyncio
async def test_retrieve_and_combine_results(dakera_client):
    _, client = dakera_client
    client.query_text.return_value = SimpleNamespace(
        results=[
            SimpleNamespace(id="a", score=0.9, text="alpha"),
            SimpleNamespace(id="b", score=0.8, text="beta"),
            SimpleNamespace(id="c", score=0.7, text=None),
        ]
    )
    retriever = _retriever()
    combined = await retriever.retrieve_and_combine_results("q")
    # None-text results are skipped.
    assert combined == "alpha\nbeta"


@pytest.mark.asyncio
async def test_retrieve_and_generate_not_supported(dakera_client):
    retriever = _retriever()
    with pytest.raises(NotImplementedError):
        await retriever.retrieve_and_generate("q")
