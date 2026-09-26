from collections.abc import AsyncIterator
from typing import Protocol

from openai import AsyncOpenAI

from app.core.config import Settings


class LLMClient(Protocol):
    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]: ...


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
            messages=messages,
            stream=True,
            temperature=self._settings.llm_temperature,
            max_tokens=self._settings.llm_max_tokens,
        )
        async for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
