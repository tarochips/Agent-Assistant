from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """JSON-serializable state checkpointed for one conversation thread."""

    messages: list[dict[str, Any]]
    question: str
    context: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    tool_rounds: int
