from fastapi import APIRouter
from sqlalchemy.exc import SQLAlchemyError

from src.db.native_search import search_text_gin

router = APIRouter(prefix="/api/music", tags=["music"])


@router.get("/search")
def search_music(query: str = "amor", limit: int = 10):
    try:
        results = search_text_gin(query=query, limit=limit, modality="music")
        return {
            "query": query,
            "modality": "music_text",
            "engine": "postgres_gin_full_text",
            "results": results,
        }
    except (SQLAlchemyError, ValueError) as exc:
        return {
            "query": query,
            "modality": "music_text",
            "engine": "postgres_gin_full_text",
            "database": "error",
            "detail": exc.__class__.__name__,
            "results": [],
        }
