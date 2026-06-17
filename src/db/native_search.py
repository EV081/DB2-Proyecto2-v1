from __future__ import annotations

from time import perf_counter
from typing import Any, Iterable

from sqlalchemy import text

from src.db.database import get_session


def _vector_literal(values: Iterable[float]) -> str:
    return "[" + ",".join(f"{float(v):.8f}" for v in values) + "]"


def _row_to_dict(row: Any) -> dict[str, Any]:
    return dict(row._mapping)


# ============================================================================
# Musica - Lyrics (texto)
# ============================================================================
def _search_songs_lyrics_fts(query: str, limit: int, index_kind: str) -> list[dict]:
    sql = """
        SELECT
            s.id,
            s.title,
            s.artist,
            ts_rank(s.lyrics_tsv, plainto_tsquery('english', :query)) AS score,
            :engine AS engine
        FROM songs s
        WHERE s.lyrics_tsv @@ plainto_tsquery('english', :query)
        ORDER BY score DESC, s.id ASC
        LIMIT :limit
    """
    engine = f"postgres_{index_kind}_full_text"
    with get_session() as session:
        rows = session.execute(text(sql), {"query": query, "limit": limit, "engine": engine}).all()
    return [_row_to_dict(r) for r in rows]


def search_songs_lyrics_gin(query: str, limit: int = 10) -> list[dict]:
    return _search_songs_lyrics_fts(query, limit, "gin")


def search_songs_lyrics_gist(query: str, limit: int = 10) -> list[dict]:
    return _search_songs_lyrics_fts(query, limit, "gist")


def search_songs_lyrics_pgvector(query_emb: list[float], limit: int = 10) -> list[dict]:
    vec = _vector_literal(query_emb)
    sql = """
        SELECT
            s.id,
            s.title,
            s.artist,
            1.0 - (s.lyrics_emb <=> CAST(:vec AS vector)) AS score,
            'pgvector_hnsw_cosine' AS engine
        FROM songs s
        WHERE s.lyrics_emb IS NOT NULL
        ORDER BY s.lyrics_emb <=> CAST(:vec AS vector)
        LIMIT :limit
    """
    with get_session() as session:
        rows = session.execute(text(sql), {"vec": vec, "limit": limit}).all()
    return [_row_to_dict(r) for r in rows]


def search_songs_audio_pgvector(query_emb: list[float], limit: int = 10) -> list[dict]:
    vec = _vector_literal(query_emb)
    sql = """
        SELECT
            s.id,
            s.title,
            s.artist,
            1.0 - (s.audio_emb <=> CAST(:vec AS vector)) AS score,
            'pgvector_hnsw_cosine' AS engine
        FROM songs s
        WHERE s.audio_emb IS NOT NULL
        ORDER BY s.audio_emb <=> CAST(:vec AS vector)
        LIMIT :limit
    """
    with get_session() as session:
        rows = session.execute(text(sql), {"vec": vec, "limit": limit}).all()
    return [_row_to_dict(r) for r in rows]


# ============================================================================
# Fashion - Descripcion (texto)
# ============================================================================
def _search_products_desc_fts(query: str, limit: int, index_kind: str) -> list[dict]:
    sql = """
        SELECT
            p.id,
            p.name,
            p.category,
            ts_rank(p.description_tsv, plainto_tsquery('english', :query)) AS score,
            :engine AS engine
        FROM products p
        WHERE p.description_tsv @@ plainto_tsquery('english', :query)
        ORDER BY score DESC, p.id ASC
        LIMIT :limit
    """
    engine = f"postgres_{index_kind}_full_text"
    with get_session() as session:
        rows = session.execute(text(sql), {"query": query, "limit": limit, "engine": engine}).all()
    return [_row_to_dict(r) for r in rows]


def search_products_desc_gin(query: str, limit: int = 10) -> list[dict]:
    return _search_products_desc_fts(query, limit, "gin")


def search_products_desc_gist(query: str, limit: int = 10) -> list[dict]:
    return _search_products_desc_fts(query, limit, "gist")


def search_products_desc_pgvector(query_emb: list[float], limit: int = 10) -> list[dict]:
    vec = _vector_literal(query_emb)
    sql = """
        SELECT
            p.id,
            p.name,
            p.category,
            1.0 - (p.desc_emb <=> CAST(:vec AS vector)) AS score,
            'pgvector_hnsw_cosine' AS engine
        FROM products p
        WHERE p.desc_emb IS NOT NULL
        ORDER BY p.desc_emb <=> CAST(:vec AS vector)
        LIMIT :limit
    """
    with get_session() as session:
        rows = session.execute(text(sql), {"vec": vec, "limit": limit}).all()
    return [_row_to_dict(r) for r in rows]


def search_products_image_pgvector(query_emb: list[float], limit: int = 10) -> list[dict]:
    vec = _vector_literal(query_emb)
    sql = """
        SELECT
            p.id,
            p.name,
            p.category,
            1.0 - (p.image_emb <=> CAST(:vec AS vector)) AS score,
            'pgvector_hnsw_cosine' AS engine
        FROM products p
        WHERE p.image_emb IS NOT NULL
        ORDER BY p.image_emb <=> CAST(:vec AS vector)
        LIMIT :limit
    """
    with get_session() as session:
        rows = session.execute(text(sql), {"vec": vec, "limit": limit}).all()
    return [_row_to_dict(r) for r in rows]


__all__ = [
    "search_songs_lyrics_gin",
    "search_songs_lyrics_gist",
    "search_songs_lyrics_pgvector",
    "search_songs_audio_pgvector",
    "search_products_desc_gin",
    "search_products_desc_gist",
    "search_products_desc_pgvector",
    "search_products_image_pgvector",
]