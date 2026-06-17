-- Schema para Hito 3: Apps 2 (Musica) + 4 (Fashion).
-- Drop+recreate. Para re-aplicar despues de cambios:
--   docker compose down -v && docker compose up -d

CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS search_logs CASCADE;
DROP TABLE IF EXISTS songs CASCADE;
DROP TABLE IF EXISTS products CASCADE;
DROP TABLE IF EXISTS codebooks CASCADE;
DROP TABLE IF EXISTS items CASCADE;
DROP TABLE IF EXISTS chunks CASCADE;

-- ============================================================================
-- App 2: Busqueda Musical Inteligente
-- ============================================================================
CREATE TABLE songs (
    id           SERIAL PRIMARY KEY,
    title        TEXT NOT NULL,
    artist       TEXT,
    genre        TEXT,
    lyrics_path  TEXT,
    audio_path   TEXT,
    lyrics_text  TEXT,
    lyrics_tsv   tsvector GENERATED ALWAYS AS
                 (to_tsvector('english', coalesce(lyrics_text, ''))) STORED,
    lyrics_hist  JSONB NOT NULL DEFAULT '{}'::jsonb,
    audio_hist   JSONB NOT NULL DEFAULT '{}'::jsonb,
    -- Dim por defecto: text=1000, audio=200. El ETL ajusta con ALTER si el
    -- codebook real difiere.
    lyrics_emb   vector(1000),
    audio_emb    vector(200),
    metadata     JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_songs_lyrics_tsv_gin
    ON songs USING gin (lyrics_tsv);
CREATE INDEX idx_songs_lyrics_tsv_gist
    ON songs USING gist (lyrics_tsv);
CREATE INDEX idx_songs_metadata_gin
    ON songs USING gin (metadata);
CREATE INDEX idx_songs_lyrics_emb_hnsw
    ON songs USING hnsw (lyrics_emb vector_cosine_ops);
CREATE INDEX idx_songs_audio_emb_hnsw
    ON songs USING hnsw (audio_emb vector_cosine_ops);

-- ============================================================================
-- App 4: Recomendacion Multimodal (Fashion)
-- ============================================================================
CREATE TABLE products (
    id              SERIAL PRIMARY KEY,
    name            TEXT NOT NULL,
    category        TEXT,
    subcategory     TEXT,
    image_path      TEXT,
    description     TEXT,
    description_tsv tsvector GENERATED ALWAYS AS
                    (to_tsvector('english', coalesce(description, ''))) STORED,
    desc_hist       JSONB NOT NULL DEFAULT '{}'::jsonb,
    image_hist      JSONB NOT NULL DEFAULT '{}'::jsonb,
    desc_emb        vector(1000),
    image_emb       vector(128),
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_products_desc_tsv_gin
    ON products USING gin (description_tsv);
CREATE INDEX idx_products_desc_tsv_gist
    ON products USING gist (description_tsv);
CREATE INDEX idx_products_metadata_gin
    ON products USING gin (metadata);
CREATE INDEX idx_products_desc_emb_hnsw
    ON products USING hnsw (desc_emb vector_cosine_ops);
CREATE INDEX idx_products_image_emb_hnsw
    ON products USING hnsw (image_emb vector_cosine_ops);

-- ============================================================================
-- Codebooks: persistencia del vocabulario por app/modalidad
-- ============================================================================
CREATE TABLE codebooks (
    id             SERIAL PRIMARY KEY,
    app            TEXT NOT NULL,        -- 'music' | 'fashion'
    modality       TEXT NOT NULL,        -- 'text' | 'audio' | 'image'
    codebook_size  INTEGER NOT NULL,
    bag_of_words   JSONB,                -- texto: ["love", "heart", ...]
    centroids_path TEXT,                 -- audio/imagen: path a binario .npy
    index_dir      TEXT,                 -- path al InvertedIndex (SPIMI)
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (app, modality)
);

-- ============================================================================
-- Logs de busqueda para benchmark Fase 4
-- ============================================================================
CREATE TABLE search_logs (
    id           SERIAL PRIMARY KEY,
    app          TEXT NOT NULL,           -- 'music' | 'fashion'
    modality     TEXT NOT NULL,           -- 'text' | 'audio' | 'image'
    engine       TEXT NOT NULL,           -- 'spimi' | 'pgvector' | 'gin' | 'gist'
    query        TEXT,
    latency_ms   DOUBLE PRECISION,
    n_results    INTEGER,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_search_logs_grouping
    ON search_logs (app, modality, engine);
