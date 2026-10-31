import json

import pytest

from app.services.context_manager import ContextManager, ContextSummaryError


class SummaryLLM:
    def __init__(self, replies=None):
        self.replies = list(replies or ["第一次摘要", "更新后的摘要"])
        self.inputs = []

    async def stream(self, messages):
        self.inputs.append(json.loads(messages[1]["content"]))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        yield reply


def add_exchanges(repository, session_id, count):
    for index in range(count):
        repository.append_exchange(session_id, f"旧问题{index}", f"旧回答{index}", [])


@pytest.mark.asyncio
async def test_summary_incrementally_covers_old_messages_and_keeps_originals(service_bundle):
    repo = service_bundle["session_repository"]
    session = repo.create()
    add_exchanges(repo, session["id"], 3)
    settings = service_bundle["settings"].model_copy(
        update={"history_max_messages": 2, "context_summary_batch_messages": 2}
    )
    llm = SummaryLLM()
    manager = ContextManager(repo, llm, settings)
    result = await manager.prepare(repo.get(session["id"]))
    assert result.summary == "更新后的摘要"
    assert [item["content"] for item in result.recent] == ["旧问题2", "旧回答2"]
    assert len(repo.get(session["id"])["messages"]) == 6
    assert repo.get(session["id"])["summary_message_count"] == 4
    assert llm.inputs[0]["existing_summary"] == ""
    assert llm.inputs[1]["existing_summary"] == "第一次摘要"
    assert [m["content"] for m in llm.inputs[1]["new_history"]] == ["旧问题1", "旧回答1"]

    again = await manager.prepare(repo.get(session["id"]))
    assert again == result
    assert len(llm.inputs) == 2


@pytest.mark.asyncio
async def test_summary_failure_leaves_originals_and_unadvanced_cursor(service_bundle):
    repo = service_bundle["session_repository"]
    session = repo.create()
    add_exchanges(repo, session["id"], 2)
    settings = service_bundle["settings"].model_copy(update={"history_max_messages": 2})
    llm = SummaryLLM([RuntimeError("unavailable")])
    with pytest.raises(ContextSummaryError):
        await ContextManager(repo, llm, settings).prepare(repo.get(session["id"]))
    saved = repo.get(session["id"])
    assert len(saved["messages"]) == 4
    assert saved.get("summary_message_count", 0) == 0
    assert not saved.get("context_summary")


@pytest.mark.asyncio
async def test_summary_compare_and_swap_rejects_stale_write(service_bundle):
    repo = service_bundle["session_repository"]
    session = repo.create()
    add_exchanges(repo, session["id"], 2)
    stale = repo.get(session["id"])
    assert repo.update_summary(session["id"], expected_count=0, new_count=2, summary="先写入")
    settings = service_bundle["settings"].model_copy(update={"history_max_messages": 2})
    with pytest.raises(ContextSummaryError, match="concurrently"):
        await ContextManager(repo, SummaryLLM(), settings).prepare(stale)
    assert repo.get(session["id"])["context_summary"] == "先写入"


@pytest.mark.asyncio
async def test_recent_budget_keeps_conversation_turns_together(service_bundle):
    repo = service_bundle["session_repository"]
    session = repo.create()
    add_exchanges(repo, session["id"], 2)
    settings = service_bundle["settings"].model_copy(update={"history_max_messages": 1})
    manager = ContextManager(repo, SummaryLLM(["摘要"]), settings)
    result = await manager.prepare(repo.get(session["id"]))
    assert result.recent == []
    assert repo.get(session["id"])["summary_message_count"] == 4


def test_chat_includes_summary_recent_messages_and_retrieved_evidence(client, service_bundle):
    repo = service_bundle["session_repository"]
    settings = service_bundle["settings"]
    settings.history_max_messages = 2
    session = repo.create()
    add_exchanges(repo, session["id"], 2)
    client.post(
        "/api/documents/upload",
        files={"file": ("guide.md", "知识库内容与项目相关", "text/markdown")},
    )
    response = client.post(
        "/api/chat/stream", json={"message": "知识库内容", "session_id": session["id"]}
    )
    assert "event: completed" in response.text
    prompt = service_bundle["fake_llm"].chat_prompts[-1]
    assert "历史对话摘要" in prompt[0]["content"]
    assert "guide.md" in prompt[0]["content"]
    assert [m["content"] for m in prompt[1:]] == ["旧问题1", "旧回答1", "知识库内容"]
    saved = repo.get(session["id"])
    assert saved["summary_message_count"] == 2
    assert len(saved["messages"]) == 6


def test_summary_error_is_reported_without_persisting_new_exchange(client, service_bundle):
    repo = service_bundle["session_repository"]
    service_bundle["settings"].history_max_messages = 0
    service_bundle["chat_service"]._context._llm = SummaryLLM([RuntimeError("down")])
    session = repo.create()
    add_exchanges(repo, session["id"], 1)
    response = client.post(
        "/api/chat/stream", json={"message": "test", "session_id": session["id"]}
    )
    assert "CONTEXT_SUMMARY_ERROR" in response.text
    assert len(repo.get(session["id"])["messages"]) == 2


@pytest.mark.asyncio
async def test_failed_later_batch_retains_only_successful_summary_cursor(service_bundle):
    repo = service_bundle["session_repository"]
    session = repo.create()
    add_exchanges(repo, session["id"], 3)
    settings = service_bundle["settings"].model_copy(
        update={"history_max_messages": 2, "context_summary_batch_messages": 2}
    )
    with pytest.raises(ContextSummaryError):
        await ContextManager(
            repo, SummaryLLM(["已摘要的前两条", RuntimeError("down")]), settings
        ).prepare(repo.get(session["id"]))
    saved = repo.get(session["id"])
    assert saved["summary_message_count"] == 2
    assert saved["context_summary"] == "已摘要的前两条"
    assert len(saved["messages"]) == 6


def test_summary_is_preserved_when_tool_approval_resumes(client, service_bundle):
    from app.llm.client import LLMToolCall

    repo = service_bundle["session_repository"]
    service_bundle["settings"].history_max_messages = 2
    session = repo.create()
    add_exchanges(repo, session["id"], 2)
    doc_id = client.post(
        "/api/documents/upload", files={"file": ("remove.md", "可删除测试", "text/markdown")}
    ).json()["doc_id"]
    service_bundle["fake_llm"].replies = [
        ("", [LLMToolCall("delete", "delete_document", json.dumps({"document_id": doc_id}))]),
        ("文档已删除。", []),
    ]
    paused = client.post(
        "/api/chat/stream", json={"message": f"删除 {doc_id}", "session_id": session["id"]}
    )
    assert "event: approval_required" in paused.text
    resumed = client.post(f"/api/chat/{session['id']}/resume", json={"approved": True})
    assert "event: completed" in resumed.text
    assert "历史对话摘要" in service_bundle["fake_llm"].chat_prompts[-1][0]["content"]
    assert len(service_bundle["fake_llm"].summary_prompts) == 1


def test_interactive_research_receives_summary_and_recent_history(client, service_bundle):
    from app.api.dependencies import get_chat_service
    from app.services.chat_service import ChatService

    class CapturingResearchGraph:
        async def ainvoke(self, state):
            self.state = state
            return {"report": "研究结果", "citations": []}

    repo = service_bundle["session_repository"]
    service_bundle["settings"].history_max_messages = 2
    session = repo.create()
    add_exchanges(repo, session["id"], 2)
    research = CapturingResearchGraph()
    original = service_bundle["chat_service"]
    service = ChatService(
        repo, original._graph, service_bundle["settings"], research, service_bundle["fake_llm"]
    )
    client.app.dependency_overrides[get_chat_service] = lambda: service
    response = client.post(
        "/api/chat/stream",
        json={"message": "研究它", "session_id": session["id"], "mode": "research"},
    )
    assert "event: completed" in response.text
    assert research.state["conversation_context"]["summary"] == "历史对话摘要"
    assert [m["content"] for m in research.state["conversation_context"]["recent"]] == [
        "旧问题1",
        "旧回答1",
    ]


@pytest.mark.asyncio
async def test_recent_character_budget_moves_complete_turn_into_summary(service_bundle):
    repo = service_bundle["session_repository"]
    session = repo.create()
    repo.append_exchange(session["id"], "旧问题", "旧回答", [])
    repo.append_exchange(session["id"], "近期问题", "很长的回答", [])
    settings = service_bundle["settings"].model_copy(
        update={"history_max_messages": 20, "context_recent_max_chars": 9}
    )
    result = await ContextManager(repo, SummaryLLM(["摘要"]), settings).prepare(
        repo.get(session["id"])
    )
    assert [message["content"] for message in result.recent] == ["近期问题", "很长的回答"]
    assert repo.get(session["id"])["summary_message_count"] == 2


@pytest.mark.asyncio
async def test_oversized_single_message_fails_without_advancing_summary(service_bundle):
    repo = service_bundle["session_repository"]
    session = repo.create()
    repo.append_exchange(session["id"], "x" * 200, "answer", [])
    settings = service_bundle["settings"].model_copy(
        update={"history_max_messages": 0, "context_summary_batch_chars": 100}
    )
    with pytest.raises(ContextSummaryError, match="batch limit"):
        await ContextManager(repo, SummaryLLM(), settings).prepare(repo.get(session["id"]))
    assert repo.get(session["id"]).get("summary_message_count", 0) == 0
