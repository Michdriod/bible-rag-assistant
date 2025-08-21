-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable pg_trgm for better text search (optional but recommended)
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ============================================================================
-- INDEXES FOR BIBLE RAG OPTIMIZATION
-- ============================================================================

-- 1. Exact reference lookups (most important for Bible study)
CREATE INDEX IF NOT EXISTS idx_bible_verses_reference ON bible_verses(reference);

-- 2. Book, chapter, verse lookup (for context retrieval)
CREATE INDEX IF NOT EXISTS idx_bible_verses_book_chapter_verse ON bible_verses(book, chapter, verse);

-- 3. Individual component indexes for filtering
CREATE INDEX IF NOT EXISTS idx_bible_verses_book ON bible_verses(book);
CREATE INDEX IF NOT EXISTS idx_bible_verses_chapter ON bible_verses(chapter);

-- 4. Composite index for book + chapter (for chapter retrieval)
CREATE INDEX IF NOT EXISTS idx_bible_verses_book_chapter ON bible_verses(book, chapter);

-- 5. Full-text search index for keyword fallback
CREATE INDEX IF NOT EXISTS idx_bible_verses_text_gin ON bible_verses USING GIN(to_tsvector('english', text));

-- 6. Vector similarity index for semantic search
-- NOTE: This should be created AFTER data ingestion for better performance
-- The lists parameter should be roughly sqrt(total_rows)
-- For ~31,000 verses: sqrt(31000) ≈ 176, rounded to 200 for safety
CREATE INDEX IF NOT EXISTS idx_bible_verses_embedding 
ON bible_verses USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 200);

-- Alternative vector index using HNSW (if available in your PostgreSQL version)
-- HNSW is generally faster for queries but slower to build
-- Uncomment the line below and comment the ivfflat index above if you prefer HNSW:
-- CREATE INDEX IF NOT EXISTS idx_bible_verses_embedding_hnsw 
-- ON bible_verses USING hnsw (embedding vector_cosine_ops);

-- ============================================================================
-- PERFORMANCE OPTIMIZATION QUERIES
-- ============================================================================

-- Update table statistics for better query planning
ANALYZE bible_verses;

-- ============================================================================
-- SAMPLE QUERIES FOR TESTING YOUR BIBLE RAG SYSTEM
-- ============================================================================

-- 1. Exact reference lookup (should use idx_bible_verses_reference)
SELECT * FROM bible_verses WHERE reference = 'John 3:16';

-- 2. Book and chapter lookup with ordering (should use idx_bible_verses_book_chapter)
SELECT reference, text 
FROM bible_verses 
WHERE book = 'John' AND chapter = 3 
ORDER BY verse;

-- 3. Context retrieval: Get verses around John 3:16
SELECT reference, text, verse
FROM bible_verses 
WHERE book = 'John' AND chapter = 3 AND verse BETWEEN 15 AND 17
ORDER BY verse;

-- 4. Semantic search using cosine similarity (replace with actual query embedding)
-- This is a template - your application will replace the embedding values
SELECT 
    reference,
    text,
    book,
    chapter,
    verse,
    1 - (embedding <=> '[0.1, 0.2, 0.3, ...]'::vector) AS similarity_score
FROM bible_verses 
WHERE 1 - (embedding <=> '[0.1, 0.2, 0.3, ...]'::vector) >= 0.3
ORDER BY embedding <=> '[0.1, 0.2, 0.3, ...]'::vector 
LIMIT 5;

-- 5. Full-text search fallback (should use idx_bible_verses_text_gin)
SELECT 
    reference, 
    text,
    ts_rank(to_tsvector('english', text), plainto_tsquery('english', 'love world')) as rank
FROM bible_verses 
WHERE to_tsvector('english', text) @@ plainto_tsquery('english', 'love world')
ORDER BY ts_rank(to_tsvector('english', text), plainto_tsquery('english', 'love world')) DESC
LIMIT 5;

-- 6. Combined semantic + keyword search (for hybrid RAG)
-- This would be used when semantic search returns few results
WITH semantic_results AS (
    SELECT 
        reference, text, book, chapter, verse,
        1 - (embedding <=> '[0.1, 0.2, 0.3, ...]'::vector) AS similarity_score,
        'semantic' as search_type
    FROM bible_verses 
    WHERE 1 - (embedding <=> '[0.1, 0.2, 0.3, ...]'::vector) >= 0.2
    ORDER BY embedding <=> '[0.1, 0.2, 0.3, ...]'::vector 
    LIMIT 3
),
keyword_results AS (
    SELECT 
        reference, text, book, chapter, verse,
        ts_rank(to_tsvector('english', text), plainto_tsquery('english', 'love world')) as similarity_score,
        'keyword' as search_type
    FROM bible_verses 
    WHERE to_tsvector('english', text) @@ plainto_tsquery('english', 'love world')
    AND reference NOT IN (SELECT reference FROM semantic_results)
    ORDER BY ts_rank(to_tsvector('english', text), plainto_tsquery('english', 'love world')) DESC
    LIMIT 2
)
SELECT * FROM semantic_results
UNION ALL
SELECT * FROM keyword_results
ORDER BY similarity_score DESC;

-- 7. Topic exploration: Find verses in specific books
SELECT reference, text 
FROM bible_verses 
WHERE book IN ('Romans', 'Ephesians', 'Galatians')
AND to_tsvector('english', text) @@ plainto_tsquery('english', 'faith grace')
ORDER BY book, chapter, verse;

-- 8. Range reference lookup (for queries like "Genesis 1:1-3")
SELECT reference, text, verse
FROM bible_verses 
WHERE book = 'Genesis' AND chapter = 1 AND verse BETWEEN 1 AND 3
ORDER BY verse;

-- ============================================================================
-- MAINTENANCE QUERIES
-- ============================================================================

-- Check index usage and table statistics
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan as index_scans,
    idx_tup_read as tuples_read,
    idx_tup_fetch as tuples_fetched
FROM pg_stat_user_indexes 
WHERE tablename = 'bible_verses'
ORDER BY idx_scan DESC;

-- Check table size and row count
SELECT 
    pg_size_pretty(pg_total_relation_size('bible_verses')) as table_size,
    COUNT(*) as total_verses,
    COUNT(DISTINCT book) as unique_books,
    COUNT(DISTINCT chapter) as unique_chapters
FROM bible_verses;

-- Verify vector index is being used for similarity searches
EXPLAIN (ANALYZE, BUFFERS) 
SELECT reference, text, 1 - (embedding <=> '[0.1, 0.2, 0.3, ...]'::vector) AS similarity
FROM bible_verses 
ORDER BY embedding <=> '[0.1, 0.2, 0.3, ...]'::vector 
LIMIT 5;





















-- -- Enable pgvector extension
-- CREATE EXTENSION IF NOT EXISTS vector;

-- -- Create index for exact reference lookups
-- CREATE INDEX IF NOT EXISTS idx_bible_verses_reference ON bible_verses(reference);
-- CREATE INDEX IF NOT EXISTS idx_bible_verses_book_chapter_verse ON bible_verses(book, chapter, verse);

-- -- Create vector similarity index for semantic search
-- CREATE INDEX IF NOT EXISTS idx_bible_verses_embedding ON bible_verses USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- -- Sample queries for testing

-- -- Exact reference lookup
-- SELECT * FROM bible_verses WHERE reference = 'John 3:16';

-- -- Semantic search using cosine similarity
-- SELECT 
--     reference,
--     text,
--     1 - (embedding <=> '[0.1, 0.2, ...]'::vector) AS similarity
-- FROM bible_verses 
-- ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector 
-- LIMIT 3;

-- -- Full-text search fallback
-- SELECT * FROM bible_verses 
-- WHERE to_tsvector('english', text) @@ plainto_tsquery('english', 'love world');

-- -- Book and chapter lookup
-- SELECT * FROM bible_verses 
-- WHERE book = 'John' AND chapter = 3 
-- ORDER BY verse;