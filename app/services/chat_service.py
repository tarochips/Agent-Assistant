import logging
from collections.abc import AsyncIterator
from typing import Any

from langgraph.types import Command

from app.core.config import Settings
from app.core.exceptions import ConflictError, NotFoundError
from app.retrieval.vector import StaleEmbeddingError
from app.schemas.chat import ChatRequest
from app.schemas.events import (
    ApprovalRequiredEvent,
    Citation,
    CompletedEvent,
    ErrorEvent,
    RetrievalEvent,
    TokenEvent,
)
from app.services._async_utils import resolve
from app.streaming.sse import encode_sse

logger = logging.getLogger("rag_app.chat")


class ChatService:
    def __init__(
        self,
        session_repository: Any,
        graph: Any,
        settings: Settings,
    ) -> None:
        self._sessions = session_repository
        self._graph = graph
        self._settings = settings

    async def stream(self, request: ChatRequest) -> AsyncIterator[str]:
        try:
            session = await self._resolve_session(request.session_id)
            session_id = str(session["id"])
            config = self._config(session_id)
            snapshot = await self._graph.aget_state(config)
            if _pending_interrupts(snapshot):
                yield self._error(
                    "APPROVAL_PENDING",
                    "此会话有待确认的工具操作，请先完成确认或拒绝。",
                )
                return

            history = self._history_messages(session.get("messages", []))
            input_state = {
                "messages": [*history, {"role": "user", "content": request.message}],
                "question": request.message,
            }
        except Exception:
            logger.exception("chat_setup_failed")
            yield self._error("CHAT_SETUP_ERROR", "会话初始化失败，请检查会话和数据库状态。")
            return

        async for event in self._run_graph(input_state, session_id):
            yield event

    async def ensure_resumable(self, session_id: str) -> None:
        session = await resolve(self._sessions.get(session_id))
        if session is None:
            raise NotFoundError("session not found", code="SESSION_NOT_FOUND")
        snapshot = await self._graph.aget_state(self._config(session_id))
        if not _pending_interrupts(snapshot):
            raise ConflictError(
                "该会话当前没有等待确认的 LangGraph 中断。",
                code="NO_PENDING_APPROVAL",
            )

    async def pending_approval(self, session_id: str) -> ApprovalRequiredEvent | None:
        session = await resolve(self._sessions.get(session_id))
        if session is None:
            raise NotFoundError("session not found", code="SESSION_NOT_FOUND")
        snapshot = await self._graph.aget_state(self._config(session_id))
        pending = _pending_interrupts(snapshot)
        if not pending:
            return None
        payload = pending[0]
        return ApprovalRequiredEvent(
            session_id=session_id,
            action=str(payload.get("action", "tool")),
            document_id=str(payload.get("document_id", "")),
            message=str(payload.get("message", "需要确认工具操作。")),
        )

    async def resume(self, session_id: str, approved: bool) -> AsyncIterator[str]:
        async for event in self._run_graph(
            Command(resume={"approved": approved}),
            session_id,
        ):
            yield event

    async def _run_graph(self, graph_input: Any, session_id: str) -> AsyncIterator[str]:
        config = self._config(session_id)
        emitted_approval = False
        try:
            async for part in self._graph.astream(
                graph_input,
                config=config,
                stream_mode=["custom", "updates"],
                version="v2",
            ):
                if part["type"] == "custom":
                    data = part["data"]
                    if data.get("type") == "token" and data.get("delta"):
                        yield encode_sse("token", TokenEvent(delta=data["delta"]))
                elif part["type"] == "updates":
                    for node_update in part["data"].values():
                        if not isinstance(node_update, dict):
                            continue
                        if "citations" in node_update and "context" in node_update:
                            citations = [
                                Citation.model_validate(item) for item in node_update["citations"]
                            ]
                            yield encode_sse(
                                "retrieval",
                                RetrievalEvent(session_id=session_id, sources=citations),
                            )
                        if "__interrupt__" in node_update:
                            emitted_approval = True
                            for value in _interrupt_values(node_update["__interrupt__"]):
                                yield encode_sse(
                                    "approval_required",
                                    ApprovalRequiredEvent(
                                        session_id=session_id,
                                        action=str(value.get("action", "tool")),
                                        document_id=str(value.get("document_id", "")),
                                        message=str(value.get("message", "需要确认工具操作。")),
                                    ),
                                )

            snapshot = await self._graph.aget_state(config)
            interruptions = _pending_interrupts(snapshot)
            if interruptions:
                if not emitted_approval:
                    for value in interruptions:
                        yield encode_sse(
                            "approval_required",
                            ApprovalRequiredEvent(
                                session_id=session_id,
                                action=str(value.get("action", "tool")),
                                document_id=str(value.get("document_id", "")),
                                message=str(value.get("message", "需要确认工具操作。")),
                            ),
                        )
                return

            state = snapshot.values or {}
            answer = _last_assistant_answer(state.get("messages", []))
            question = str(state.get("question", ""))
            citations = [Citation.model_validate(item) for item in state.get("citations", [])]
            await resolve(
                self._sessions.append_exchange(
                    session_id,
                    question,
                    answer,
                    [citation.model_dump(mode="json") for citation in citations],
                )
            )
            yield encode_sse(
                "completed",
                CompletedEvent(session_id=session_id, citations=citations),
            )
        except StaleEmbeddingError:
            logger.exception("stale_embeddings session_id=%s", session_id)
            yield self._error(
                "STALE_EMBEDDINGS",
                "数据库中仍有旧模型向量，请先运行 python -m app.retrieval.rebuild_embeddings。",
            )
        except Exception:
            logger.exception("agent_graph_failed session_id=%s", session_id)
            yield self._error(
                "AGENT_GRAPH_ERROR",
                "Agent 执行失败，请检查模型、工具和数据库配置后重试。",
            )

    async def _resolve_session(self, session_id: str | None) -> dict[str, Any]:
        if session_id:
            session = await resolve(self._sessions.get(session_id))
            if session is not None:
                return session
        return await resolve(self._sessions.create())

    def _history_messages(self, history: list[dict[str, Any]]) -> list[dict[str, str]]:
        filtered = [
            {"role": message["role"], "content": message["content"]}
            for message in history
            if message.get("role") in {"user", "assistant"}
            and isinstance(message.get("content"), str)
        ]
        max_history = self._settings.history_max_messages
        return filtered[-max_history:] if max_history else []

    @staticmethod
    def _config(session_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": session_id}}

    @staticmethod
    def _error(code: str, message: str) -> str:
        return encode_sse("error", ErrorEvent(code=code, message=message))


def _interrupt_values(raw: Any) -> list[dict[str, Any]]:
    values = raw if isinstance(raw, (tuple, list)) else [raw]
    return [value.value for value in values if isinstance(getattr(value, "value", None), dict)]


def _pending_interrupts(snapshot: Any) -> list[dict[str, Any]]:
    pending: list[dict[str, Any]] = []
    for task in getattr(snapshot, "tasks", ()):
        for value in getattr(task, "interrupts", ()):
            payload = getattr(value, "value", None)
            if isinstance(payload, dict):
                pending.append(payload)
    return pending


def _last_assistant_answer(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant" and not message.get("tool_calls"):
            content = message.get("content")
            return content if isinstance(content, str) else ""
    return ""
