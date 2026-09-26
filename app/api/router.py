from fastapi import APIRouter

from app.api import chat, documents, health, sessions

api_router = APIRouter(prefix="/api")
api_router.include_router(health.router)
api_router.include_router(sessions.router)
api_router.include_router(documents.router)
api_router.include_router(chat.router)
