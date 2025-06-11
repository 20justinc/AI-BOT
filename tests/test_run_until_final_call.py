import types
import sys
import asyncio
from pathlib import Path
import pytest
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# --- stub required external modules ---
utils_mod = types.ModuleType('utils')
utils_mod.list_abbreviations = lambda: []
sys.modules['utils'] = utils_mod

openai_mod = types.ModuleType('openai')
class AsyncOpenAI: pass
from typing import Generic, TypeVar
T = TypeVar('T')
class AsyncStream(Generic[T]): pass
openai_mod.AsyncOpenAI = AsyncOpenAI
openai_mod.AsyncStream = AsyncStream
sys.modules['openai'] = openai_mod

openai_chat_mod = types.ModuleType('openai.types.chat')
class ChatCompletion: pass
class ChatCompletionChunk: pass
class ChatCompletionContentPartParam: pass
class ChatCompletionMessageParam: pass
openai_chat_mod.ChatCompletion = ChatCompletion
openai_chat_mod.ChatCompletionChunk = ChatCompletionChunk
openai_chat_mod.ChatCompletionContentPartParam = ChatCompletionContentPartParam
openai_chat_mod.ChatCompletionMessageParam = ChatCompletionMessageParam
sys.modules['openai.types.chat'] = openai_chat_mod

azure_aio_mod = types.ModuleType('azure.search.documents.aio')
class SearchClient: pass
azure_aio_mod.SearchClient = SearchClient
sys.modules['azure.search.documents.aio'] = azure_aio_mod

azure_models_mod = types.ModuleType('azure.search.documents.models')
class VectorQuery: pass
class RawVectorQuery: pass
class CaptionResult: pass
class QueryType:
    SEMANTIC = 1
azure_models_mod.VectorQuery = VectorQuery
azure_models_mod.RawVectorQuery = RawVectorQuery
azure_models_mod.CaptionResult = CaptionResult
azure_models_mod.QueryType = QueryType
sys.modules['azure.search.documents.models'] = azure_models_mod

msg_mod = types.ModuleType('integration_azure.messagebuilder')
class MessageBuilder:
    def __init__(self, system_prompt, model_id):
        self.messages = [{"role": "system", "content": system_prompt}]
    def insert_message(self, role, content, index=None):
        self.messages.append({"role": role, "content": content})
    def count_tokens_for_message(self, message):
        return len(str(message.get("content", "")))
msg_mod.MessageBuilder = MessageBuilder
sys.modules['integration_azure.messagebuilder'] = msg_mod

auth_mod = types.ModuleType('integration_azure.authentication')
class AuthenticationHelper:
    def build_security_filters(self, overrides, auth_claims):
        return None
auth_mod.AuthenticationHelper = AuthenticationHelper
sys.modules['integration_azure.authentication'] = auth_mod

modelhelper_mod = types.ModuleType('integration_azure.modelhelper')
modelhelper_mod.get_token_limit = lambda model: 8000
sys.modules['integration_azure.modelhelper'] = modelhelper_mod

# --- now import the module under test ---
from rag_response_generator.azure_rag_responder import RagResponder, Document

# helper to construct responder with mocked openai client
class DummyOpenAIClient:
    def __init__(self, create_mock):
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=create_mock))
        self.embeddings = types.SimpleNamespace(create=AsyncMock())

class DummyAuth(AuthenticationHelper):
    pass

def test_streaming_returns_rag_stream(monkeypatch):
    async def run():
        create_mock = AsyncMock()
        openai_client = DummyOpenAIClient(create_mock)

        responder = RagResponder(
        search_client=SearchClient(),
        auth_helper=DummyAuth(),
        openai_client=openai_client,
        chatgpt_model="gpt",
        chatgpt_deployment=None,
        embedding_deployment=None,
        embedding_model="embed",
        sourcepage_field="sourcepage",
        content_field="content",
        query_language="en",
        query_speller="en",
        )

        monkeypatch.setattr(responder, "build_filter", lambda o, a: None)
        monkeypatch.setattr(responder, "compute_text_embedding", AsyncMock(return_value=None))
        monkeypatch.setattr(responder, "search", AsyncMock(return_value=[
            Document(id="1", content="c", embedding=None, image_embedding=None, category=None, sourcepage="sp", sourcefile=None, oids=None, groups=None, captions=[])
        ]))
        monkeypatch.setattr(responder, "get_search_query", lambda c, q: "query")
        monkeypatch.setattr(responder, "get_messages_from_history", lambda **kw: [{"role": "user", "content": "x"}])

        async def rag_stream_mock(*args, **kwargs):
            return []
        rag_stream_mock = AsyncMock(side_effect=rag_stream_mock)
        monkeypatch.setattr(responder, "_rag_stream", rag_stream_mock)

        extra, coro = await responder.run_until_final_call(
            messages=[{"role": "user", "content": "hello"}],
            overrides={},
            auth_claims={},
            should_stream=True,
        )

        # only the search query generation call executed so far
        assert create_mock.call_count == 1
        rag_stream_mock.assert_not_called()

        result = await coro
        if asyncio.iscoroutine(result):
            await result
        rag_stream_mock.assert_called_once()

    asyncio.run(run())

def test_non_streaming_calls_chat_completion(monkeypatch):
    async def run():
        create_mock = AsyncMock()
        openai_client = DummyOpenAIClient(create_mock)
        responder = RagResponder(
            search_client=SearchClient(),
            auth_helper=DummyAuth(),
            openai_client=openai_client,
            chatgpt_model="gpt",
            chatgpt_deployment=None,
            embedding_deployment=None,
            embedding_model="embed",
            sourcepage_field="sourcepage",
            content_field="content",
            query_language="en",
            query_speller="en",
        )

        monkeypatch.setattr(responder, "build_filter", lambda o, a: None)
        monkeypatch.setattr(responder, "compute_text_embedding", AsyncMock(return_value=None))
        monkeypatch.setattr(responder, "search", AsyncMock(return_value=[
            Document(id="1", content="c", embedding=None, image_embedding=None, category=None, sourcepage="sp", sourcefile=None, oids=None, groups=None, captions=[])
        ]))
        monkeypatch.setattr(responder, "get_search_query", lambda c, q: "query")
        monkeypatch.setattr(responder, "get_messages_from_history", lambda **kw: [{"role": "user", "content": "x"}])

        async def rag_stream_mock(*args, **kwargs):
            return []
        rag_stream_mock = AsyncMock(side_effect=rag_stream_mock)
        monkeypatch.setattr(responder, "_rag_stream", rag_stream_mock)

        extra, coro = await responder.run_until_final_call(
            messages=[{"role": "user", "content": "hello"}],
            overrides={},
            auth_claims={},
            should_stream=False,
        )

        # chat completion called twice: once for search query, once for final answer
        assert create_mock.call_count == 2
        rag_stream_mock.assert_not_called()

        result = await coro
        if asyncio.iscoroutine(result):
            await result
        # awaiting should not trigger extra calls
        assert create_mock.call_count == 2

    asyncio.run(run())
