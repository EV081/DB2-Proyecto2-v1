from datetime import datetime
from typing import Any

from sqlalchemy import Column, DateTime, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, Relationship, SQLModel

try:
    from pgvector.sqlalchemy import Vector
except ImportError:  # Allows imports before dependencies are installed.
    Vector = None


def vector_column(dimensions: int) -> Column:
    if Vector is None:
        return Column(Text, nullable=True)
    return Column(Vector(dimensions), nullable=True)


class Item(SQLModel, table=True):
    __tablename__ = "items"

    id: int | None = Field(default=None, primary_key=True)
    title: str
    modality: str = Field(index=True)
    metadata_: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column("metadata", JSONB, nullable=False, server_default="{}"),
    )
    created_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), server_default=func.now(), nullable=False),
    )

    chunks: list["Chunk"] = Relationship(back_populates="item")


class Chunk(SQLModel, table=True):
    __tablename__ = "chunks"

    id: int | None = Field(default=None, primary_key=True)
    item_id: int = Field(foreign_key="items.id", index=True)
    chunk_index: int
    content: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    histogram: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, server_default="{}"),
    )
    embedding: Any | None = Field(default=None, sa_column=vector_column(8))
    created_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), server_default=func.now(), nullable=False),
    )

    item: Item | None = Relationship(back_populates="chunks")


class SearchLog(SQLModel, table=True):
    __tablename__ = "search_logs"

    id: int | None = Field(default=None, primary_key=True)
    query: str
    modality: str = Field(index=True)
    engine: str
    latency_ms: float | None = None
    created_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), server_default=func.now(), nullable=False),
    )
