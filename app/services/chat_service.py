import logging
from collections.abc import AsyncIterator

from app.core.config import Settings
from app.llm.client import LLMClient
from app.repositories.sessions import JsonSessionRepository
from app.retrieval.models import SearchResult
from app.retrieval.tfidf import TfidfRetriever
from app.schemas.chat import ChatRequest
from app.schemas.events import CompletedEvent, ErrorEvent, RetrievalEvent, TokenEvent
from app.streaming.sse import encode_sse

logger = logging.getLogger("rag_app.chat")


class ChatService:
    def __init__(
        self,
        session_repository: JsonSessionRepository,
        retriever: TfidfRetriever,
        llm_client: LLMClient,
        settings: Settings,
    ) -> None:
        self._sessions = session_repository
        self._retriever = retriever
        self._llm = llm_client
        self._settings = settings

    async def stream(self, request: ChatRequest) -> AsyncIterator[str]:
        session = self._resolve_session(request.session_id)
        session_id = str(session["id"])
        history = session.get("messages", [])[-self._settings.history_max_messages :]

        results = self._retriever.search(
            request.message,
            top_k=self._settings.retrieval_top_k,
            min_score=self._settings.retrieval_min_score,
        )
        citations = [result.to_citation() for result in results]
        yield encode_sse(
            "retrieval",
            RetrievalEvent(session_id=session_id, sources=citations),
        )

        messages = self._build_messages(history, request.message, results)
        answer_parts: list[str] = []
        try:
            async for token in self._llm.stream(messages):
                if not token:
                    continue
                answer_parts.append(token)
                yield encode_sse("token", TokenEvent(delta=token))
        except Exception:
            logger.exception("llm_stream_failed session_id=%s", session_id)
            yield encode_sse(
                "error",
                ErrorEvent(
                    code="LLM_STREAM_ERROR",
                    message="模型调用失败，请检查模型配置后重试",
                ),
            )
            return

        full_answer = "".join(answer_parts)
        citation_dicts = [citation.model_dump(mode="json") for citation in citations]
        self._sessions.append_exchange(
            session_id,
            request.message,
            full_answer,
            citation_dicts,
        )
        yield encode_sse(
            "completed",
            CompletedEvent(session_id=session_id, citations=citations),
        )

    def _resolve_session(self, session_id: str | None) -> dict:
        if session_id:
            session = self._sessions.get(session_id)
            if session is not None:
                return session
        return self._sessions.create()

    @staticmethod
    def _build_messages(
        history: list[dict], query: str, results: list[SearchResult]
    ) -> list[dict[str, str]]:
        if results:
            sections = []
            for index, result in enumerate(results, start=1):
                sections.append(
                    f"[S{index}] 来源：{result.filename}，分块：{result.chunk_index + 1}\n"
                    f"{result.text}"
                )
            context = "\n\n---\n\n".join(sections)
            system_prompt = (
                "你是一个知识库助手。请仅根据参考资料回答问题，不要补充资料之外的事实。"
                "重要结论后请使用 [S1]、[S2] 这样的编号标注来源。"
                "如果参考资料无法回答，请明确说“知识库中暂无相关内容”。\n\n"
                f"【参考资料】\n{context}"
            )
        else:
            system_prompt = (
                "你是一个知识库助手。本次检索没有找到达到相关性阈值的资料。"
                "请只回答“知识库中暂无相关内容”，并建议用户补充资料或换一种问法。"
            )

        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for message in history:
            role = message.get("role")
            content = message.get("content")
            if role in {"user", "assistant"} and isinstance(content, str):
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": query})
        return messages
