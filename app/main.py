import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.engine import make_url

from app.api.dependencies import (
    get_chat_service,
    get_document_repository,
    get_llm_client,
    get_retriever,
)
from app.api.router import api_router
from app.core.config import get_settings
from app.core.exceptions import AppError
from app.core.logging import configure_logging, request_logging_middleware


@asynccontextmanager
async def lifespan(application: FastAPI):
    if get_chat_service in application.dependency_overrides:
        yield
    else:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        from app.agent.graph import build_agent_graph

        settings = get_settings()
        checkpoint_url = make_url(settings.database_url).set(drivername="postgresql")
        async with AsyncPostgresSaver.from_conn_string(
            checkpoint_url.render_as_string(hide_password=False)
        ) as checkpointer:
            await checkpointer.setup()
            application.state.agent_graph = build_agent_graph(
                retriever=get_retriever(),
                document_repository=get_document_repository(),
                llm_client=get_llm_client(),
                settings=settings,
                checkpointer=checkpointer,
            )
            yield
    database_module = sys.modules.get("app.core.database")
    if database_module is not None:
        await database_module.dispose_engine()


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging()

    application = FastAPI(
        title=settings.app_name,
        version="4.0.0",
        description="A PostgreSQL-backed RAG knowledge base with a stateful LangGraph agent.",
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
