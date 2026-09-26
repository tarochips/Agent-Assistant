import json
from typing import Any

from pydantic import BaseModel


def encode_sse(event: str, payload: BaseModel | dict[str, Any]) -> str:
    if isinstance(payload, BaseModel):
        data = payload.model_dump(mode="json")
    else:
        data = payload
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {encoded}\n\n"
