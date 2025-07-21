-- ========================================
-- SIMPLE QUERIES FOR metric_rag APPLICATION
-- No filtering - just raw data for your app
-- ========================================

-- ===========================================
-- RAW DATA QUERIES (SELECT * - Complete Data)
-- ===========================================

-- 1. ALL RUNS FOR metric_rag APPLICATION (Raw Data)
SELECT * 
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE CONTAINS(RECORD_ATTRIBUTES::string, 'metric_rag')
   OR CONTAINS(RESOURCE_ATTRIBUTES::string, 'metric_rag')
ORDER BY timestamp DESC;

-- 2. MOST RECENT RUN (Raw Data) - Find the latest run properly
SELECT * 
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE RECORD_ATTRIBUTES['ai.observability.run.name'] = (
    SELECT RECORD_ATTRIBUTES['ai.observability.run.name']
    FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
    WHERE RECORD_ATTRIBUTES['ai.observability.run.name'] IS NOT NULL
    ORDER BY timestamp DESC
    LIMIT 1
)
ORDER BY timestamp DESC;

-- 3. CURRENT DAY RUNS (Raw Data)
SELECT * 
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE DATE(timestamp) = CURRENT_DATE()
ORDER BY timestamp DESC;

-- ===========================================
-- SEPARATED COLUMNS QUERIES (Extracted Attributes)
-- ===========================================

-- 4. ALL RUNS FOR metric_rag APPLICATION (Separated Columns)
SELECT 
    timestamp,
    start_timestamp,
    record_type,
    
    -- Trace Information
    TRACE['trace_id'] as trace_id,
    TRACE['span_id'] as span_id,
    
    -- Record Information
    RECORD['name'] as span_name,
    RECORD['kind'] as span_kind,
    
    -- Custom Attributes (Separated Columns)
    RECORD_ATTRIBUTES['ai.observability.run.name'] as run_name,
    RECORD_ATTRIBUTES['custom.username'] as username,
    RECORD_ATTRIBUTES['custom.query.toxicity.score'] as query_toxicity_score,
    RECORD_ATTRIBUTES['custom.query.is.toxic'] as query_is_toxic,
    RECORD_ATTRIBUTES['custom.response.toxicity.score'] as response_toxicity_score,
    RECORD_ATTRIBUTES['custom.response.is.toxic'] as response_is_toxic,
    RECORD_ATTRIBUTES['custom.hallucination.score'] as hallucination_score,
    RECORD_ATTRIBUTES['custom.has.hallucination'] as has_hallucination,
    RECORD_ATTRIBUTES['custom.response.length'] as response_length,
    
    -- System Information
    RESOURCE_ATTRIBUTES['snow.executable.name'] as executable_name,
    RESOURCE_ATTRIBUTES['snow.database.name'] as database_name,
    RESOURCE_ATTRIBUTES['snow.schema.name'] as schema_name,
    RESOURCE_ATTRIBUTES['db.user'] as db_user,
    
    -- Raw JSON for reference
    RECORD_ATTRIBUTES,
    RESOURCE_ATTRIBUTES
    
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE CONTAINS(RECORD_ATTRIBUTES::string, 'metric_rag')
   OR CONTAINS(RESOURCE_ATTRIBUTES::string, 'metric_rag')
ORDER BY timestamp DESC;

-- 5. MOST RECENT RUN (Separated Columns) - Find the latest run properly
SELECT 
    timestamp,
    start_timestamp,
    record_type,
    
    -- Trace Information
    TRACE['trace_id'] as trace_id,
    TRACE['span_id'] as span_id,
    
    -- Record Information
    RECORD['name'] as span_name,
    RECORD['kind'] as span_kind,
    
    -- Custom Attributes (Separated Columns)
    RECORD_ATTRIBUTES['ai.observability.run.name'] as run_name,
    RECORD_ATTRIBUTES['custom.username'] as username,
    RECORD_ATTRIBUTES['custom.query.toxicity.score'] as query_toxicity_score,
    RECORD_ATTRIBUTES['custom.query.is.toxic'] as query_is_toxic,
    RECORD_ATTRIBUTES['custom.response.toxicity.score'] as response_toxicity_score,
    RECORD_ATTRIBUTES['custom.response.is.toxic'] as response_is_toxic,
    RECORD_ATTRIBUTES['custom.hallucination.score'] as hallucination_score,
    RECORD_ATTRIBUTES['custom.has.hallucination'] as has_hallucination,
    RECORD_ATTRIBUTES['custom.response.length'] as response_length,
    
    -- System Information
    RESOURCE_ATTRIBUTES['snow.executable.name'] as executable_name,
    RESOURCE_ATTRIBUTES['snow.database.name'] as database_name,
    RESOURCE_ATTRIBUTES['snow.schema.name'] as schema_name,
    RESOURCE_ATTRIBUTES['db.user'] as db_user,
    
    -- Raw JSON for reference
    RECORD_ATTRIBUTES,
    RESOURCE_ATTRIBUTES
    
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE RECORD_ATTRIBUTES['ai.observability.run.name'] = (
    SELECT RECORD_ATTRIBUTES['ai.observability.run.name']
    FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
    WHERE RECORD_ATTRIBUTES['ai.observability.run.name'] IS NOT NULL
    ORDER BY timestamp DESC
    LIMIT 1
)
ORDER BY timestamp DESC;

-- 6. CURRENT DAY RUNS (Separated Columns)
SELECT 
    timestamp,
    start_timestamp,
    record_type,
    
    -- Trace Information
    TRACE['trace_id'] as trace_id,
    TRACE['span_id'] as span_id,
    
    -- Record Information
    RECORD['name'] as span_name,
    RECORD['kind'] as span_kind,
    
    -- Custom Attributes (Separated Columns)
    RECORD_ATTRIBUTES['ai.observability.run.name'] as run_name,
    RECORD_ATTRIBUTES['custom.username'] as username,
    RECORD_ATTRIBUTES['custom.query.toxicity.score'] as query_toxicity_score,
    RECORD_ATTRIBUTES['custom.query.is.toxic'] as query_is_toxic,
    RECORD_ATTRIBUTES['custom.response.toxicity.score'] as response_toxicity_score,
    RECORD_ATTRIBUTES['custom.response.is.toxic'] as response_is_toxic,
    RECORD_ATTRIBUTES['custom.hallucination.score'] as hallucination_score,
    RECORD_ATTRIBUTES['custom.has.hallucination'] as has_hallucination,
    RECORD_ATTRIBUTES['custom.response.length'] as response_length,
    
    -- System Information
    RESOURCE_ATTRIBUTES['snow.executable.name'] as executable_name,
    RESOURCE_ATTRIBUTES['snow.database.name'] as database_name,
    RESOURCE_ATTRIBUTES['snow.schema.name'] as schema_name,
    RESOURCE_ATTRIBUTES['db.user'] as db_user,
    
    -- Raw JSON for reference
    RECORD_ATTRIBUTES,
    RESOURCE_ATTRIBUTES
    
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE DATE(timestamp) = CURRENT_DATE()
ORDER BY timestamp DESC;

-- ===========================================
-- BONUS: See what runs are available
-- ===========================================

-- 7. DISCOVER AVAILABLE RUNS (to see what run names exist)
SELECT 
    RECORD_ATTRIBUTES['ai.observability.run.name'] as run_name,
    COUNT(*) as trace_count,
    MIN(timestamp) as first_trace,
    MAX(timestamp) as last_trace
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE RECORD_ATTRIBUTES['ai.observability.run.name'] IS NOT NULL
GROUP BY RECORD_ATTRIBUTES['ai.observability.run.name']
ORDER BY last_trace DESC;
