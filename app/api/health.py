from typing import Annotated

from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
def live() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
def ready(settings: Annotated[Settings, Depends(get_settings)]) -> dict:
    return {
        "status": "ready",
        "dependencies": {
            "storage": "ok" if settings.data_dir.exists() else "missing",
            "llm": "configured" if settings.deepseek_api_key else "missing_api_key",
        },
    }
