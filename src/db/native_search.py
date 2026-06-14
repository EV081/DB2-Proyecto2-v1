import json
import sys
from time import perf_counter
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src.db.database import get_session


def _vector_literal(values: list[float]) -> str:
    if len(values) != 8:
        raise ValueError("query_vector must have exactly 8 dimensions")
    return "[" + ",".join(str(float(value)) for value in values) + "]"


def _row_to_dict(row: Any) -> dict[str, Any]:
    return dict(row._mapping)


def _log_search(query: str, modality: str, engine: str, latency_ms: float) -> None:
    try:
        with get_session() as session:
            session.execute(
                text(
                    """
                    INSERT INTO search_logs (query, modality, engine, latency_ms)
                    VALUES (:query, :modality, :engine, :latency_ms)
                    """
                ),
                {
                    "query": query,
                    "modality": modality,
                    "engine": engine,
                    "latency_ms": latency_ms,
                },
            )
            session.commit()
    except SQLAlchemyError:
        return


def insert_item(title: str, modality: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    with get_session() as session:
        row = session.execute(
            text(
                """
                INSERT INTO items (title, modality, metadata)
                VALUES (:title, :modality, CAST(:metadata AS jsonb))
                RETURNING id, title, modality, metadata, created_at
                """
            ),
            {
                "title": title,
                "modality": modality,
                "metadata": json.dumps(metadata or {}),
            },
        ).one()
        session.commit()
        return _row_to_dict(row)


def insert_chunk(
    item_id: int,
    chunk_index: int,
    content: str,
    histogram: dict[str, Any] | None = None,
    embedding: list[float] | None = None,
) -> dict[str, Any]:
    with get_session() as session:
        row = session.execute(
            text(
                """
                INSERT INTO chunks (item_id, chunk_index, content, histogram, embedding)
                VALUES (
                    :item_id,
                    :chunk_index,
                    :content,
                    CAST(:histogram AS jsonb),
                    CAST(:embedding AS vector)
                )
                RETURNING id, item_id, chunk_index, content, histogram, created_at
                """
            ),
            {
                "item_id": item_id,
                "chunk_index": chunk_index,
                "content": content,
                "histogram": json.dumps(histogram or {}),
                "embedding": _vector_literal(embedding or [0.0] * 8),
            },
        ).one()
        session.commit()
        return _row_to_dict(row)


def _search_text(
    query: str,
    engine: str,
    limit: int,
    modality: str | None = None,
) -> list[dict[str, Any]]:
    started_at = perf_counter()
    with get_session() as session:
        rows = session.execute(
            text(
                """
                SELECT
                    c.id,
                    c.item_id,
                    i.title,
                    c.content,
                    ts_rank(
                        to_tsvector('spanish', coalesce(c.content, '')),
                        plainto_tsquery('spanish', :query)
                    ) AS score,
                    :engine AS engine
                FROM chunks c
                JOIN items i ON i.id = c.item_id
                WHERE to_tsvector('spanish', coalesce(c.content, ''))
                      @@ plainto_tsquery('spanish', :query)
                      AND (:modality IS NULL OR i.modality = :modality)
                ORDER BY score DESC, c.id ASC
                LIMIT :limit
                """
            ),
            {"query": query, "engine": engine, "limit": limit, "modality": modality},
        ).all()

    latency_ms = (perf_counter() - started_at) * 1000
    _log_search(query=query, modality=modality or "text", engine=engine, latency_ms=latency_ms)
    return [_row_to_dict(row) for row in rows]


def search_text_gin(
    query: str,
    limit: int = 10,
    modality: str | None = None,
) -> list[dict[str, Any]]:
    """Search chunks with PostgreSQL full-text search backed by the GIN tsvector index."""
    return _search_text(
        query=query,
        engine="postgres_gin_full_text",
        limit=limit,
        modality=modality,
    )


def search_text_gist(
    query: str,
    limit: int = 10,
    modality: str | None = None,
) -> list[dict[str, Any]]:
    """Search chunks with PostgreSQL full-text search backed by the GiST tsvector index."""
    return _search_text(
        query=query,
        engine="postgres_gist_full_text",
        limit=limit,
        modality=modality,
    )


def search_vector_pgvector(query_vector: list[float], limit: int = 10) -> list[dict[str, Any]]:
    started_at = perf_counter()
    vector_literal = _vector_literal(query_vector)
    with get_session() as session:
        rows = session.execute(
            text(
                """
                SELECT
                    c.id,
                    c.item_id,
                    i.title,
                    c.content,
                    c.embedding <=> CAST(:query_vector AS vector) AS distance,
                    'pgvector_hnsw_cosine' AS engine
                FROM chunks c
                JOIN items i ON i.id = c.item_id
                WHERE c.embedding IS NOT NULL
                ORDER BY c.embedding <=> CAST(:query_vector AS vector)
                LIMIT :limit
                """
            ),
            {"query_vector": vector_literal, "limit": limit},
        ).all()

    latency_ms = (perf_counter() - started_at) * 1000
    _log_search(
        query=vector_literal,
        modality="vector",
        engine="pgvector_hnsw_cosine",
        latency_ms=latency_ms,
    )
    return [_row_to_dict(row) for row in rows]


def search_metadata_gin(filters: dict[str, Any], limit: int = 10) -> list[dict[str, Any]]:
    started_at = perf_counter()
    with get_session() as session:
        rows = session.execute(
            text(
                """
                SELECT
                    i.id,
                    i.id AS item_id,
                    i.title,
                    i.metadata,
                    1.0 AS score,
                    'postgres_gin_jsonb' AS engine
                FROM items i
                WHERE i.metadata @> CAST(:filters AS jsonb)
                ORDER BY i.id ASC
                LIMIT :limit
                """
            ),
            {"filters": json.dumps(filters), "limit": limit},
        ).all()

    latency_ms = (perf_counter() - started_at) * 1000
    _log_search(
        query=json.dumps(filters),
        modality="metadata",
        engine="postgres_gin_jsonb",
        latency_ms=latency_ms,
    )
    return [_row_to_dict(row) for row in rows]


def _get_or_create_item(title: str, modality: str, metadata: dict[str, Any]) -> int:
    with get_session() as session:
        existing = session.execute(
            text("SELECT id FROM items WHERE title = :title AND modality = :modality LIMIT 1"),
            {"title": title, "modality": modality},
        ).scalar()
        if existing:
            return int(existing)

    return int(insert_item(title=title, modality=modality, metadata=metadata)["id"])


def seed_demo_data() -> dict[str, Any]:
    """Insert tiny Hito 2 demo rows for local PostgreSQL tests; no real datasets are loaded."""
    music_id = _get_or_create_item(
        title="Cancion demo amor",
        modality="music",
        metadata={"type": "song", "artist": "demo", "source": "hito2"},
    )
    store_id = _get_or_create_item(
        title="Camisa azul demo",
        modality="store",
        metadata={"type": "product", "category": "ropa", "color": "azul", "source": "hito2"},
    )

    created_chunks = []
    with get_session() as session:
        music_chunk_exists = session.execute(
            text("SELECT id FROM chunks WHERE item_id = :item_id AND chunk_index = 0 LIMIT 1"),
            {"item_id": music_id},
        ).scalar()
        store_chunk_exists = session.execute(
            text("SELECT id FROM chunks WHERE item_id = :item_id AND chunk_index = 0 LIMIT 1"),
            {"item_id": store_id},
        ).scalar()

    if not music_chunk_exists:
        created_chunks.append(
            insert_chunk(
                item_id=music_id,
                chunk_index=0,
                content="Cancion romantica demo sobre amor y recuerdos",
                histogram={"amor": 3, "demo": 1},
                embedding=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            )
        )
    if not store_chunk_exists:
        created_chunks.append(
            insert_chunk(
                item_id=store_id,
                chunk_index=0,
                content="Camisa azul de algodon para tienda demo",
                histogram={"camisa": 2, "azul": 2},
                embedding=[0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            )
        )

    return {"items": [music_id, store_id], "created_chunks": len(created_chunks)}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "seed-demo":
        print(json.dumps(seed_demo_data(), indent=2))
    else:
        print("Usage: python -m src.db.native_search seed-demo")
