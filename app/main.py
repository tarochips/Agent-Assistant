import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.router import api_router
from app.core.config import get_settings
from app.core.exceptions import AppError
from app.core.logging import configure_logging, request_logging_middleware


@asynccontextmanager
async def lifespan(_application: FastAPI):
    yield
    database_module = sys.modules.get("app.core.database")
    if database_module is not None:
        await database_module.dispose_engine()


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging()

    application = FastAPI(
        title=settings.app_name,
        version="2.0.0",
        description="A PostgreSQL-backed RAG knowledge base with local vector retrieval.",
        lifespan=lifespan,
    )

    application.middleware("http")(request_logging_middleware)

    @application.exception_handler(AppError)
    async def handle_app_error(_request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    application.include_router(api_router)
    application.mount(
        "/",
        StaticFiles(directory=settings.static_dir, html=True),
        name="static",
    )
    return application


app = create_app()
