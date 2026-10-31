import asyncio
import json

import pytest

from app.agent.research import build_research_graph
from app.llm.client import LLMStreamChunk, LLMToolCall
from app.retrieval.models import SearchResult


class ResearchLLM:
    def __init__(self, reviews=None, invalid_report=False):
        self.reviews = list(
            reviews
            or [
                {
                    "decision": "research",
                    "critique": "需要补充",
                    "queries": ["extra", "overflow"],
                    "evidence_ids": ["doc:0"],
                },
                {
                    "decision": "accept",
                    "critique": "证据充分",
                    "queries": [],
                    "evidence_ids": ["doc:0"],
                },
            ]
        )
        self.invalid_report = invalid_report
        self.draft_count = 0
        self.worker_inputs = []

    async def stream_with_tools(self, messages, tools):
        assert [t["function"]["name"] for t in tools] == ["search_knowledge"]
        query = messages[1]["content"]
        self.worker_inputs.append((query, sum(m["role"] == "tool" for m in messages)))
        if not any(m["role"] == "tool" for m in messages):
            yield LLMStreamChunk(
                complete=True,
                tool_calls=[
                    LLMToolCall("call-" + query, "search_knowledge", json.dumps({"query": query}))
                ],
            )
        else:
            yield LLMStreamChunk(delta="研究发现")
            yield LLMStreamChunk(complete=True)

    async def stream(self, messages):
        system = messages[0]["content"]
        if "规划员" in system:
            yield json.dumps(["first", "second", "ignored"])
        elif "审查 Agent" in system:
            yield json.dumps(self.reviews.pop(0))
        elif "撰写 Agent" in system:
            self.draft_count += 1
            assert "[S1]" in messages[1]["content"]
            yield "研究报告 [S99]" if self.invalid_report else f"研究报告 {self.draft_count} [S1]"
        else:
            yield "基于资料的分析"


class ResearchRetriever:
    def __init__(self, empty=False):
        self.queries = []
        self.active = 0
        self.max_active = 0
        self.empty = empty
        self.parallel_started = asyncio.Event()

    async def search(self, query, *, top_k, min_score):
        self.queries.append(query)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        if self.active >= 2:
            self.parallel_started.set()
        if query in {"first", "second"}:
            await asyncio.wait_for(self.parallel_started.wait(), timeout=3)
        self.active -= 1
        if self.empty:
            return []
        return [SearchResult("doc:0", "doc", "doc.md", 0, "资料内容", 0.8)]


def verdict(decision, critique="需要修改", queries=None, evidence_ids=None):
    return {
        "decision": decision,
        "critique": critique,
        "queries": queries or [],
        "evidence_ids": evidence_ids or [],
    }


@pytest.mark.asyncio
async def test_agents_research_parallel_review_supplement_and_recheck(service_bundle):
    settings = service_bundle["settings"].model_copy(
        update={"research_max_tasks": 2, "research_max_supplemental": 1}
    )
    retriever = ResearchRetriever()
    llm = ResearchLLM()
    graph = build_research_graph(retriever=retriever, llm_client=llm, settings=settings)
    result = await graph.ainvoke({"question": "test"})
    assert retriever.queries == ["first", "second", "extra"]
    assert retriever.max_active == 2
    assert all((query, 0) in llm.worker_inputs for query in retriever.queries)
    assert all(f["searches"] == [f["query"]] for f in result["findings"])
    assert result["review_rounds"] == 2
    assert len(result["citations"]) == 1
    assert result["report"] == "研究报告 2 [S1]"


@pytest.mark.asyncio
async def test_reviewer_requests_revision_and_report_is_rechecked(service_bundle):
    llm = ResearchLLM(reviews=[verdict("revise"), verdict("accept", "完成")])
    graph = build_research_graph(
        retriever=ResearchRetriever(), llm_client=llm, settings=service_bundle["settings"]
    )
    result = await graph.ainvoke({"question": "test"})
    assert llm.draft_count == 2
    assert result["review_rounds"] == 2
    assert result["report"] == "研究报告 2 [S1]"


@pytest.mark.asyncio
async def test_review_budget_returns_explicit_unapproved_warning(service_bundle):
    llm = ResearchLLM(reviews=[verdict("revise"), verdict("revise", "仍有缺陷")])
    graph = build_research_graph(
        retriever=ResearchRetriever(), llm_client=llm, settings=service_bundle["settings"]
    )
    result = await graph.ainvoke({"question": "test"})
    assert result["review_rounds"] == 2
    assert "审查未通过" in result["report"]
    assert "仍有缺陷" in result["report"]


@pytest.mark.asyncio
async def test_research_without_evidence_is_explicit(service_bundle):
    llm = ResearchLLM(reviews=[verdict("accept", "资料不足已注明")])
    graph = build_research_graph(
        retriever=ResearchRetriever(empty=True), llm_client=llm, settings=service_bundle["settings"]
    )
    result = await graph.ainvoke({"question": "test"})
    assert not result["citations"]
    assert "暂无足够相关资料" in result["report"]


@pytest.mark.asyncio
async def test_invalid_citation_cannot_be_approved(service_bundle):
    llm = ResearchLLM(reviews=[verdict("accept"), verdict("accept")], invalid_report=True)
    graph = build_research_graph(
        retriever=ResearchRetriever(), llm_client=llm, settings=service_bundle["settings"]
    )
    with pytest.raises(ValueError, match="invalid source"):
        await graph.ainvoke({"question": "test"})
    assert llm.draft_count == 2


@pytest.mark.asyncio
async def test_invalid_reviewer_output_fails_closed(service_bundle):
    llm = ResearchLLM(reviews=[{"decision": "accept", "critique": "ok", "evidence_ids": ["fake"]}])
    graph = build_research_graph(
        retriever=ResearchRetriever(), llm_client=llm, settings=service_bundle["settings"]
    )
    with pytest.raises(ValueError, match="不存在"):
        await graph.ainvoke({"question": "test"})


def test_research_stream_persists_report_and_sources(client, service_bundle):
    from app.api.dependencies import get_chat_service
    from app.services.chat_service import ChatService

    old = service_bundle["chat_service"]
    llm = ResearchLLM(reviews=[verdict("accept", "通过")])
    graph = build_research_graph(
        retriever=ResearchRetriever(), llm_client=llm, settings=service_bundle["settings"]
    )
    service = ChatService(
        service_bundle["session_repository"], old._graph, service_bundle["settings"], graph
    )
    client.app.dependency_overrides[get_chat_service] = lambda: service
    session_id = client.post("/api/sessions").json()["id"]
    response = client.post(
        "/api/chat/stream", json={"message": "test", "session_id": session_id, "mode": "research"}
    )
    assert "event: completed" in response.text
    assert "研究报告 1 [S1]" in response.text
    saved = client.get(f"/api/sessions/{session_id}").json()
    assert saved["messages"][-1]["sources"][0]["chunk_id"] == "doc:0"


@pytest.mark.asyncio
async def test_research_worker_tools_are_limited_and_rounds_bounded(service_bundle):
    from app.agent.research_worker import build_research_worker

    class ToolLoopLLM:
        def __init__(self):
            self.calls = 0

        async def stream_with_tools(self, messages, tools):
            self.calls += 1
            name = "delete_document" if self.calls == 1 else "search_knowledge"
            args = {"document_id": "doc"} if self.calls == 1 else {"query": "unique"}
            yield LLMStreamChunk(
                complete=True, tool_calls=[LLMToolCall(str(self.calls), name, json.dumps(args))]
            )

        async def stream(self, messages):
            yield "有依据的总结"

    settings = service_bundle["settings"].model_copy(update={"research_worker_max_tool_rounds": 2})
    llm = ToolLoopLLM()
    retriever = ResearchRetriever()
    worker = build_research_worker(retriever=retriever, llm_client=llm, settings=settings)
    result = await worker.ainvoke(
        {
            "question": "q",
            "messages": [{"role": "user", "content": "q"}],
            "evidence": [],
            "searches": [],
            "rounds": 0,
        }
    )
    assert llm.calls == 3
    assert result["rounds"] == 2
    assert retriever.queries == ["unique"]
    assert "未知工具" in next(m["content"] for m in result["messages"] if m["role"] == "tool")
    assert result["analysis"] == "有依据的总结"


@pytest.mark.asyncio
async def test_research_worker_invalid_arguments_do_not_execute(service_bundle):
    from app.agent.research_worker import build_research_worker

    class InvalidToolLLM:
        async def stream_with_tools(self, messages, tools):
            yield LLMStreamChunk(
                complete=True,
                tool_calls=[LLMToolCall("bad", "search_knowledge", '{"query":"ok", "top_k":999}')],
            )

    retriever = ResearchRetriever()
    settings = service_bundle["settings"].model_copy(update={"research_worker_max_tool_rounds": 1})
    graph = build_research_worker(
        retriever=retriever, llm_client=InvalidToolLLM(), settings=settings
    )
    result = await graph.ainvoke(
        {
            "question": "q",
            "messages": [{"role": "user", "content": "q"}],
            "evidence": [],
            "searches": [],
            "rounds": 0,
        }
    )
    assert not retriever.queries
    assert any(m["role"] == "tool" and "参数无效" in m["content"] for m in result["messages"])
    assert "未获取" in result["analysis"]


@pytest.mark.asyncio
async def test_failed_worker_cancels_parallel_workers(service_bundle):
    class FailingRetriever:
        def __init__(self):
            self.started = asyncio.Event()
            self.cancelled = False

        async def search(self, query, *, top_k, min_score):
            if query == "first":
                await self.started.wait()
                raise RuntimeError("retrieval failed")
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    retriever = FailingRetriever()
    settings = service_bundle["settings"].model_copy(update={"research_max_tasks": 2})
    graph = build_research_graph(retriever=retriever, llm_client=ResearchLLM(), settings=settings)
    with pytest.raises(RuntimeError, match="retrieval failed"):
        await graph.ainvoke({"question": "test"})
    assert retriever.cancelled
