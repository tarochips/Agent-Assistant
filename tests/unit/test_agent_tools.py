from pathlib import Path

import pytest

from app.agent.tools import execute_tool, openai_tool_schemas
from app.core.config import Settings


def test_openai_tool_schemas_come_from_pydantic_models() -> None:
    schemas = openai_tool_schemas()
    by_name = {schema["function"]["name"]: schema["function"] for schema in schemas}

    assert set(by_name) == {"search_knowledge", "list_documents", "delete_document"}
    search_schema = by_name["search_knowledge"]["parameters"]
    assert search_schema["properties"]["query"]["type"] == "string"
    assert search_schema["properties"]["top_k"]["maximum"] == 10


@pytest.mark.asyncio
async def test_invalid_tool_arguments_are_rejected_before_execution(tmp_path: Path) -> None:
    class ShouldNotBeCalled:
        async def search(self, *_args, **_kwargs):
            raise AssertionError("invalid arguments must not reach the retriever")

    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        data_dir=tmp_path / "data",
        static_dir=tmp_path,
    )

    result = await execute_tool(
        "search_knowledge",
        '{"query":"test","top_k":100}',
        retriever=ShouldNotBeCalled(),
        document_repository=object(),
        settings=settings,
    )

    assert result.content.startswith("工具参数无效")
