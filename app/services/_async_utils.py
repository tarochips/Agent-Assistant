import inspect
from typing import Any


async def resolve(value: Any) -> Any:
    """Await async repository results while keeping the JSON test adapter usable."""
    if inspect.isawaitable(value):
        return await value
    return value
