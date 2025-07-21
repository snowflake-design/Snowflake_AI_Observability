-- ========================================
-- SIMPLE QUERIES FOR metric_rag APPLICATION
-- Using direct attribute access only (no CONTAINS)
-- ========================================

-- ===========================================
-- RAW DATA QUERIES (SELECT * - Complete Data)
-- ===========================================

-- 1. ALL RUNS FOR metric_rag APPLICATION (Raw Data)
SELECT * 
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE RECORD_ATTRIBUTES['ai.observability.app.name'] = 'metric_rag'
   OR RESOURCE_ATTRIBUTES['ai.observability.app.name'] = 'metric_rag'
ORDER BY timestamp DESC;

-- 2. MOST RECENT RUN FOR metric_rag APPLICATION (Raw Data)
SELECT * 
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE RECORD_ATTRIBUTES['ai.observability.run.name'] = (
    SELECT RECORD_ATTRIBUTES['ai.observability.run.name']
    FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
    WHERE RECORD_ATTRIBUTES['ai.observability.run.name'] IS NOT NULL
      AND (
        RECORD_ATTRIBUTES['ai.observability.app.name'] = 'metric_rag'
        OR RESOURCE_ATTRIBUTES['ai.observability.app.name'] = 'metric_rag'
      )
    ORDER BY timestamp DESC
    LIMIT 1
)
ORDER BY timestamp DESC;

-- 3. CURRENT DAY RUNS FOR metric_rag APPLICATION (Raw Data)
SELECT * 
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE DATE(timestamp) = CURRENT_DATE()
  AND (
    RECORD_ATTRIBUTES['ai.observability.app.name'] = 'metric_rag'
    OR RESOURCE_ATTRIBUTES['ai.observability.app.name'] = 'metric_rag'
  )
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
    RECORD_ATTRIBUTES['ai.observability.app.name'] as app_name,
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
WHERE RECORD_ATTRIBUTES['ai.observability.app.name'] = 'metric_rag'
   OR RESOURCE_ATTRIBUTES['ai.observability.app.name'] = 'metric_rag'
ORDER BY timestamp DESC;

-- 5. MOST RECENT RUN FOR metric_rag APPLICATION (Separated Columns)
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
    RECORD_ATTRIBUTES['ai.observability.app.name'] as app_name,
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
      AND (
        RECORD_ATTRIBUTES['ai.observability.app.name'] = 'metric_rag'
        OR RESOURCE_ATTRIBUTES['ai.observability.app.name'] = 'metric_rag'
      )
    ORDER BY timestamp DESC
    LIMIT 1
)
ORDER BY timestamp DESC;

-- 6. CURRENT DAY RUNS FOR metric_rag APPLICATION (Separated Columns)
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
    RECORD_ATTRIBUTES['ai.observability.app.name'] as app_name,
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
  AND (
    RECORD_ATTRIBUTES['ai.observability.app.name'] = 'metric_rag'
    OR RESOURCE_ATTRIBUTES['ai.observability.app.name'] = 'metric_rag'
  )
ORDER BY timestamp DESC;

-- ===========================================
-- DISCOVERY QUERIES (To understand what data exists)
-- ===========================================

-- 7. DISCOVER ALL APPLICATIONS (to see what app names exist)
SELECT 
    RECORD_ATTRIBUTES['ai.observability.app.name'] as app_name,
    COUNT(*) as trace_count,
    MIN(timestamp) as first_trace,
    MAX(timestamp) as last_trace
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE RECORD_ATTRIBUTES['ai.observability.app.name'] IS NOT NULL
GROUP BY RECORD_ATTRIBUTES['ai.observability.app.name']
ORDER BY last_trace DESC;

-- 8. DISCOVER RUNS FOR metric_rag APPLICATION (to see what runs exist for your app)
SELECT 
    RECORD_ATTRIBUTES['ai.observability.app.name'] as app_name,
    RECORD_ATTRIBUTES['ai.observability.run.name'] as run_name,
    COUNT(*) as trace_count,
    MIN(timestamp) as first_trace,
    MAX(timestamp) as last_trace
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE RECORD_ATTRIBUTES['ai.observability.app.name'] = 'metric_rag'
GROUP BY RECORD_ATTRIBUTES['ai.observability.app.name'], RECORD_ATTRIBUTES['ai.observability.run.name']
ORDER BY last_trace DESC;

-- 9. SIMPLE DEBUG - See recent events to understand structure
SELECT 
    timestamp,
    record_type,
    RECORD_ATTRIBUTES,
    RESOURCE_ATTRIBUTES
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
ORDER BY timestamp DESC
LIMIT 10;

-- 10. ALTERNATIVE - If app name is not stored properly, look for custom attributes
SELECT *
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE RECORD_ATTRIBUTES['custom.username'] IS NOT NULL
   OR RECORD_ATTRIBUTES['custom.query.toxicity.score'] IS NOT NULL
   OR RECORD_ATTRIBUTES['custom.hallucination.score'] IS NOT NULL
ORDER BY timestamp DESC;
