from typing import Annotated

from fastapi import APIRouter, Depends, Response
from fastapi.responses import JSONResponse

from app.core.config import Settings, get_settings

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
def live() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def ready(settings: Annotated[Settings, Depends(get_settings)]) -> Response:
    try:
        from app.core.database import check_database

        await check_database()
        database_status = "ok"
    except Exception:
        database_status = "unavailable"

    is_ready = database_status == "ok"
    return JSONResponse(
        status_code=200 if is_ready else 503,
        content={
            "status": "ready" if is_ready else "not_ready",
            "dependencies": {
                "storage": database_status,
                "vectorizer": "local",
                "llm": "configured" if settings.deepseek_api_key else "missing_api_key",
            },
        },
    )
