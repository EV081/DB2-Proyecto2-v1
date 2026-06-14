CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS items (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    modality TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    embedding vector(8),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    item_id INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT,
    histogram JSONB DEFAULT '{}'::jsonb,
    embedding vector(8),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS search_logs (
    id SERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    modality TEXT NOT NULL,
    engine TEXT NOT NULL DEFAULT 'unknown',
    latency_ms DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE items
    ALTER COLUMN metadata SET DEFAULT '{}'::jsonb;

ALTER TABLE chunks
    ALTER COLUMN histogram SET DEFAULT '{}'::jsonb;

ALTER TABLE search_logs
    ADD COLUMN IF NOT EXISTS engine TEXT NOT NULL DEFAULT 'unknown',
    ADD COLUMN IF NOT EXISTS latency_ms DOUBLE PRECISION;

CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
ON chunks
USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_chunks_content_gin
ON chunks
USING gin (to_tsvector('spanish', coalesce(content, '')));

CREATE INDEX IF NOT EXISTS idx_chunks_content_gist
ON chunks
USING gist (to_tsvector('spanish', coalesce(content, '')));

CREATE INDEX IF NOT EXISTS idx_items_metadata_gin
ON items
USING gin (metadata);

INSERT INTO items (title, modality, metadata, embedding)
VALUES
    ('Cancion demo amor', 'music', '{"type": "song", "artist": "demo", "source": "hito2"}', '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]'),
    ('Camisa azul demo', 'store', '{"type": "product", "category": "ropa", "color": "azul", "source": "hito2"}', '[0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]')
ON CONFLICT DO NOTHING;

INSERT INTO chunks (item_id, chunk_index, content, histogram, embedding)
SELECT id, 0, 'Cancion romantica demo sobre amor y recuerdos', '{"amor": 3, "demo": 1}', '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]'
FROM items
WHERE title = 'Cancion demo amor'
ON CONFLICT DO NOTHING;

INSERT INTO chunks (item_id, chunk_index, content, histogram, embedding)
SELECT id, 0, 'Camisa azul de algodon para tienda demo', '{"camisa": 2, "azul": 2}', '[0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]'
FROM items
WHERE title = 'Camisa azul demo'
ON CONFLICT DO NOTHING;
