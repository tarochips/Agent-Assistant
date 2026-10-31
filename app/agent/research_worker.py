"""A read-only research agent with isolated state and bounded tool use."""

import asyncio
import json
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from app.agent.schemas import SearchKnowledgeArguments
from app.core.config import Settings
from app.services._async_utils import resolve


class WorkerState(TypedDict, total=False):
    question: str
    messages: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    searches: list[str]
    rounds: int
    analysis: str


SEARCH_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "检索知识库中与研究问题相关的资料。",
            "parameters": SearchKnowledgeArguments.model_json_schema(),
        },
    }
]


def build_research_worker(*, retriever: Any, llm_client: Any, settings: Settings) -> Any:
    async def decide(state: WorkerState) -> dict[str, Any]:
        async def collect() -> tuple[str, list[Any]]:
            parts: list[str] = []
            calls: list[Any] = []
            async for item in llm_client.stream_with_tools(state["messages"], SEARCH_TOOL):
                if item.delta:
                    parts.append(item.delta)
                if item.complete:
                    calls = item.tool_calls
            return "".join(parts), calls

        text, calls = await asyncio.wait_for(collect(), settings.research_timeout_seconds)
        message: dict[str, Any] = {"role": "assistant", "content": text}
        if calls:
            message["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.arguments},
                }
                for c in calls
            ]
        return {"messages": [*state["messages"], message]}

    def route(state: WorkerState) -> Literal["tools", "finish"]:
        if (
            state["messages"][-1].get("tool_calls")
            and state.get("rounds", 0) < settings.research_worker_max_tool_rounds
        ):
            return "tools"
        return "finish"

    async def tools(state: WorkerState) -> dict[str, Any]:
        calls = state["messages"][-1]["tool_calls"]
        evidence = list(state.get("evidence", []))
        known = {item["chunk_id"] for item in evidence}
        searches = list(state.get("searches", []))
        replies = []
        for index, call in enumerate(calls):
            fn = call.get("function", {})
            content = "每轮只允许一个知识库检索工具调用。"
            if index == 0 and len(calls) == 1:
                if fn.get("name") != "search_knowledge":
                    content = "未知工具；只允许 search_knowledge。"
                else:
                    try:
                        args = SearchKnowledgeArguments.model_validate_json(
                            fn.get("arguments", "{}")
                        )
                        if not args.query.strip():
                            raise ValueError("查询不能为空")
                    except (ValidationError, ValueError):
                        content = "检索参数无效。"
                    else:
                        query = args.query.strip()
                        if query.casefold() in {q.casefold() for q in searches}:
                            content = "该查询已经执行，请选择其他查询。"
                        else:
                            searches.append(query)

                            async def search(query: str = query, top_k: int = args.top_k) -> Any:
                                return await resolve(
                                    retriever.search(
                                        query,
                                        top_k=min(top_k, settings.retrieval_top_k),
                                        min_score=settings.retrieval_min_score,
                                    )
                                )

                            results = await asyncio.wait_for(
                                search(), settings.research_timeout_seconds
                            )
                            new_results = []
                            for result in results:
                                item = {
                                    "chunk_id": result.chunk_id,
                                    "document_id": result.document_id,
                                    "filename": result.filename,
                                    "chunk_index": result.chunk_index,
                                    "text": result.text,
                                    "score": result.score,
                                    "citation": result.to_citation().model_dump(mode="json"),
                                }
                                new_results.append(item)
                                if result.chunk_id not in known:
                                    evidence.append(item)
                                    known.add(result.chunk_id)
                            content = (
                                json.dumps(new_results, ensure_ascii=False)
                                if new_results
                                else "未找到资料。"
                            )
            replies.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": content})
        return {
            "messages": [*state["messages"], *replies],
            "evidence": evidence,
            "searches": searches,
            "rounds": state.get("rounds", 0) + 1,
        }

    async def finish(state: WorkerState) -> dict[str, Any]:
        if not state.get("evidence"):
            return {"analysis": "未获取到相关证据。"}
        last = state["messages"][-1]
        if last.get("role") == "assistant" and not last.get("tool_calls") and last.get("content"):
            return {"analysis": last["content"]}

        async def collect() -> str:
            return "".join(
                [
                    part
                    async for part in llm_client.stream(
                        [
                            {
                                "role": "system",
                                "content": "你是研究员。仅根据资料概括发现、证据编号和不足。",
                            },
                            {
                                "role": "user",
                                "content": json.dumps(
                                    {"question": state["question"], "evidence": state["evidence"]},
                                    ensure_ascii=False,
                                ),
                            },
                        ]
                    )
                ]
            )

        return {"analysis": await asyncio.wait_for(collect(), settings.research_timeout_seconds)}

    graph = StateGraph(WorkerState)
    graph.add_node("decide", decide)
    graph.add_node("tools", tools)
    graph.add_node("finish", finish)
    graph.add_edge(START, "decide")
    graph.add_conditional_edges("decide", route, {"tools": "tools", "finish": "finish"})
    graph.add_edge("tools", "decide")
    graph.add_edge("finish", END)
    return graph.compile()
