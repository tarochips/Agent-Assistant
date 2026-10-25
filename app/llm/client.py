from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from app.core.config import Settings


class LLMClient(Protocol):
    def stream_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator["LLMStreamChunk"]: ...


@dataclass(frozen=True, slots=True)
class LLMToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class LLMStreamChunk:
    delta: str = ""
    complete: bool = False
    tool_calls: list[LLMToolCall] = field(default_factory=list)


class DeepSeekClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        if not self._settings.deepseek_api_key:
            raise RuntimeError("DEEPSEEK_API_KEY 未配置")
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self._settings.deepseek_api_key,
                base_url=self._settings.deepseek_base_url,
            )
        return self._client

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        response = await self._get_client().chat.completions.create(
            model=self._settings.deepseek_model,
            messages=cast(list[ChatCompletionMessageParam], messages),
            stream=True,
            temperature=self._settings.llm_temperature,
            max_tokens=self._settings.llm_max_tokens,
        )
        async for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    async def stream_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[LLMStreamChunk]:
        response = await self._get_client().chat.completions.create(
            model=self._settings.deepseek_model,
            messages=cast(list[ChatCompletionMessageParam], messages),
            tools=cast(Any, tools),
            tool_choice="auto",
            stream=True,
            extra_body={"thinking": {"type": "disabled"}},
            temperature=self._settings.llm_temperature,
            max_tokens=self._settings.llm_max_tokens,
        )
        pending_calls: dict[int, dict[str, str]] = {}
        async for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                yield LLMStreamChunk(delta=delta.content)
            for partial in delta.tool_calls or []:
                call = pending_calls.setdefault(
                    partial.index,
                    {"id": "", "name": "", "arguments": ""},
                )
                if partial.id:
                    call["id"] = partial.id
                if partial.function and partial.function.name:
                    call["name"] += partial.function.name
                if partial.function and partial.function.arguments:
                    call["arguments"] += partial.function.arguments

        yield LLMStreamChunk(
            complete=True,
            tool_calls=[
                LLMToolCall(
                    id=call["id"],
                    name=call["name"],
                    arguments=call["arguments"],
                )
                for _index, call in sorted(pending_calls.items())
            ],
        )
