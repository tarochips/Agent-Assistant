from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.agent.graph import build_agent_graph
from app.core.config import Settings
from app.llm.client import LLMStreamChunk, LLMToolCall
from app.retrieval.models import SearchResult


class FakeRetriever:
    def __init__(self) -> None:
        self.results = [
            SearchResult(
                chunk_id="doc-1:0",
                document_id="doc-1",
                filename="guide.md",
                chunk_index=0,
                text="LangGraph 使用 StateGraph 组织节点和边。",
                score=0.8,
            )
        ]
        self.search_calls: list[str] = []
        self.deleted: list[str] = []

    async def search(self, query: str, *, top_k: int, min_score: float):
        self.search_calls.append(query)
        return [result for result in self.results if result.score >= min_score][:top_k]

    async def remove_document(self, document_id: str) -> bool:
        self.deleted.append(document_id)
        return True


class FakeDocumentRepository:
    async def list_documents(self):
        return [{"doc_id": "doc-1", "filename": "guide.md", "chunk_count": 1}]


class ScriptedLLM:
    def __init__(self, replies: list[tuple[str, list[LLMToolCall]]]) -> None:
        self.replies = list(replies)
        self.prompts: list[list[dict]] = []

    async def stream_with_tools(self, messages, _tools) -> AsyncIterator[LLMStreamChunk]:
        self.prompts.append(messages)
        text, tool_calls = self.replies.pop(0)
        if text:
            yield LLMStreamChunk(delta=text)
        yield LLMStreamChunk(complete=True, tool_calls=tool_calls)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        data_dir=tmp_path / "data",
        static_dir=tmp_path,
        retrieval_top_k=3,
        retrieval_min_score=0.0,
    )


def _tool_call(name: str, arguments: str) -> LLMToolCall:
    return LLMToolCall(id="call-1", name=name, arguments=arguments)


def _graph(tmp_path: Path, llm: ScriptedLLM, retriever: FakeRetriever):
    return build_agent_graph(
        retriever=retriever,
        document_repository=FakeDocumentRepository(),
        llm_client=llm,
        settings=_settings(tmp_path),
        checkpointer=InMemorySaver(),
    )


@pytest.mark.asyncio
async def test_graph_retrieves_context_and_returns_citations(tmp_path: Path) -> None:
    retriever = FakeRetriever()
    llm = ScriptedLLM([("StateGraph 是图的构造入口。[S1]", [])])
    graph = _graph(tmp_path, llm, retriever)

    result = await graph.ainvoke(
        {
            "messages": [{"role": "user", "content": "LangGraph 如何组织流程？"}],
            "question": "LangGraph 如何组织流程？",
        },
        config={"configurable": {"thread_id": "session-1"}},
    )

    assert retriever.search_calls == ["LangGraph 如何组织流程？"]
    assert result["citations"][0]["filename"] == "guide.md"
    assert result["messages"][-1]["content"].startswith("StateGraph")
    assert "[S1]" in llm.prompts[0][0]["content"]


@pytest.mark.asyncio
async def test_agent_branches_to_tool_then_loops_back_to_model(tmp_path: Path) -> None:
    retriever = FakeRetriever()
    llm = ScriptedLLM(
        [
            ("", [_tool_call("search_knowledge", '{"query":"checkpoint","top_k":2}')]),
            ("检索结果说明了 checkpoint。[S1]", []),
        ]
    )
    graph = _graph(tmp_path, llm, retriever)

    result = await graph.ainvoke(
        {
            "messages": [{"role": "user", "content": "checkpoint 是什么？"}],
            "question": "checkpoint 是什么？",
        },
        config={"configurable": {"thread_id": "session-2"}},
    )

    assert len(llm.prompts) == 2
    assert retriever.search_calls == ["checkpoint 是什么？", "checkpoint"]
    assert any(message["role"] == "tool" for message in result["messages"])
    assert result["messages"][-1]["content"] == "检索结果说明了 checkpoint。[S1]"


@pytest.mark.asyncio
async def test_delete_tool_pauses_and_resumes_after_approval(tmp_path: Path) -> None:
    retriever = FakeRetriever()
    llm = ScriptedLLM(
        [
            ("", [_tool_call("delete_document", '{"document_id":"doc-1"}')]),
            ("文档已删除。", []),
        ]
    )
    graph = _graph(tmp_path, llm, retriever)
    config = {"configurable": {"thread_id": "session-3"}}
    graph_input = {
        "messages": [{"role": "user", "content": "删除 doc-1 文档"}],
        "question": "删除 doc-1 文档",
    }

    paused = await graph.ainvoke(graph_input, config=config)

    assert paused["__interrupt__"][0].value["action"] == "delete_document"
    assert retriever.deleted == []

    resumed = await graph.ainvoke(Command(resume={"approved": True}), config=config)

    assert retriever.deleted == ["doc-1"]
    assert resumed["messages"][-1]["content"] == "文档已删除。"
