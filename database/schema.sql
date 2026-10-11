-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Stores chunked trial text with embeddings and metadata.
-- One row = one chunk from one trial field (e.g. eligibility_criteria § Inclusion Criteria).
CREATE TABLE IF NOT EXISTS trial_chunks (
    id            SERIAL PRIMARY KEY,
    nct_id        TEXT        NOT NULL,
    title         TEXT        NOT NULL,
    phase         TEXT,
    condition     TEXT,
    status        TEXT,
    sponsor       TEXT,
    source_field  TEXT        NOT NULL,  -- 'brief_summary' | 'eligibility_criteria' | 'detailed_description'
    section       TEXT,                  -- section heading within the field, if present
    chunk_text    TEXT        NOT NULL,
    embedding     vector(1536),          -- text-embedding-3-small output dimension
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Unique key for idempotent upserts: one chunk per (trial, field, section)
    UNIQUE (nct_id, source_field, section)
);

-- Cosine similarity index (ivfflat; tune lists after bulk load)
CREATE INDEX IF NOT EXISTS trial_chunks_embedding_idx
    ON trial_chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Metadata filter indexes used by the retrieval layer
-- Note: condition/phase/status use ILIKE %value% — btree cannot accelerate
-- wildcard prefix searches, so only nct_id gets a btree index.
-- Trigram indexes (pg_trgm) would help ILIKE but add extension complexity;
-- acceptable for Phase 1 given the ~26K trial dataset size.
CREATE INDEX IF NOT EXISTS trial_chunks_nct_id_idx ON trial_chunks (nct_id);
CREATE INDEX IF NOT EXISTS trial_chunks_status_idx ON trial_chunks (status);
