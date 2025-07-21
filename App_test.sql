-- ========================================
-- CORRECTED QUERIES BASED ON ACTUAL DATA STRUCTURE
-- Using proper attribute paths from RECORD_ATTRIBUTES
-- ========================================

-- ===========================================
-- RAW DATA QUERIES (SELECT * - Complete Data)
-- ===========================================

-- 1. ALL RUNS FOR YOUR APPLICATION "METRICS_RAG" (Raw Data)
SELECT * 
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE RECORD_ATTRIBUTES['snow.ai.observability.object.name'] = 'METRICS_RAG'
ORDER BY timestamp DESC;

-- 2. MOST RECENT RUN FOR YOUR APPLICATION (Raw Data)
SELECT * 
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE RECORD_ATTRIBUTES['snow.ai.observability.run.id'] = (
    SELECT RECORD_ATTRIBUTES['snow.ai.observability.run.id']
    FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
    WHERE RECORD_ATTRIBUTES['snow.ai.observability.object.name'] = 'METRICS_RAG'
      AND RECORD_ATTRIBUTES['snow.ai.observability.run.id'] IS NOT NULL
    ORDER BY timestamp DESC
    LIMIT 1
)
ORDER BY timestamp DESC;

-- 3. SPECIFIC RUN "test_run_v26" (Raw Data)
SELECT * 
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE RECORD_ATTRIBUTES['snow.ai.observability.run.name'] = 'test_run_v26'
ORDER BY timestamp DESC;

-- 4. CURRENT DAY RUNS FOR YOUR APPLICATION (Raw Data)
SELECT * 
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE DATE(timestamp) = CURRENT_DATE()
  AND RECORD_ATTRIBUTES['snow.ai.observability.object.name'] = 'METRICS_RAG'
ORDER BY timestamp DESC;

-- ===========================================
-- SEPARATED COLUMNS QUERIES (Extracted Attributes)
-- ===========================================

-- 5. ALL RUNS FOR YOUR APPLICATION (Separated Columns)
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
    
    -- AI Observability Attributes (From RECORD_ATTRIBUTES)
    RECORD_ATTRIBUTES['snow.ai.observability.object.name'] as app_name,
    RECORD_ATTRIBUTES['snow.ai.observability.run.id'] as run_id,
    RECORD_ATTRIBUTES['snow.ai.observability.run.name'] as run_name,
    RECORD_ATTRIBUTES['snow.ai.observability.object.type'] as object_type,
    RECORD_ATTRIBUTES['snow.ai.observability.object.version.name'] as version_name,
    RECORD_ATTRIBUTES['snow.ai.observability.database.name'] as database_name,
    RECORD_ATTRIBUTES['snow.ai.observability.schema.name'] as schema_name,
    RECORD_ATTRIBUTES['ai.observability.span_type'] as span_type,
    
    -- Function Call Information
    RECORD_ATTRIBUTES['ai.observability.call.function'] as function_name,
    RECORD_ATTRIBUTES['ai.observability.call.kwargs.query'] as input_query,
    RECORD_ATTRIBUTES['ai.observability.call.return'] as function_output,
    
    -- Record Root Information (for main application flow)
    RECORD_ATTRIBUTES['ai.observability.record_root.input'] as record_input,
    RECORD_ATTRIBUTES['ai.observability.record_root.output'] as record_output,
    RECORD_ATTRIBUTES['ai.observability.record_root.ground_truth_output'] as ground_truth,
    
    -- Custom Safety Attributes
    RECORD_ATTRIBUTES['custom.username'] as username,
    RECORD_ATTRIBUTES['custom.query_toxicity_score'] as query_toxicity_score,
    RECORD_ATTRIBUTES['custom.response_toxicity_score'] as response_toxicity_score,
    RECORD_ATTRIBUTES['custom.hallucination_score'] as hallucination_score,
    RECORD_ATTRIBUTES['custom.is_high_quality_response'] as is_high_quality_response,
    RECORD_ATTRIBUTES['custom.response_length'] as response_length,
    RECORD_ATTRIBUTES['custom.context_count'] as context_count,
    
    -- Identifiers
    RECORD_ATTRIBUTES['ai.observability.app_id'] as app_id,
    RECORD_ATTRIBUTES['ai.observability.input_id'] as input_id,
    RECORD_ATTRIBUTES['ai.observability.record_id'] as record_id,
    
    -- Raw JSON for reference
    RECORD_ATTRIBUTES,
    RESOURCE_ATTRIBUTES
    
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE RECORD_ATTRIBUTES['snow.ai.observability.object.name'] = 'METRICS_RAG'
ORDER BY timestamp DESC;

-- 6. MOST RECENT RUN FOR YOUR APPLICATION (Separated Columns)
SELECT 
    timestamp,
    start_timestamp,
    record_type,
    
    -- Trace Information
    TRACE['trace_id'] as trace_id,
    TRACE['span_id'] as span_id,
    
    -- AI Observability Attributes
    RECORD_ATTRIBUTES['snow.ai.observability.object.name'] as app_name,
    RECORD_ATTRIBUTES['snow.ai.observability.run.id'] as run_id,
    RECORD_ATTRIBUTES['snow.ai.observability.run.name'] as run_name,
    RECORD_ATTRIBUTES['ai.observability.span_type'] as span_type,
    
    -- Function Information
    RECORD_ATTRIBUTES['ai.observability.call.function'] as function_name,
    RECORD_ATTRIBUTES['ai.observability.call.kwargs.query'] as input_query,
    
    -- Custom Safety Attributes
    RECORD_ATTRIBUTES['custom.username'] as username,
    RECORD_ATTRIBUTES['custom.query_toxicity_score'] as query_toxicity_score,
    RECORD_ATTRIBUTES['custom.response_toxicity_score'] as response_toxicity_score,
    RECORD_ATTRIBUTES['custom.hallucination_score'] as hallucination_score,
    RECORD_ATTRIBUTES['custom.is_high_quality_response'] as is_high_quality_response,
    RECORD_ATTRIBUTES['custom.response_length'] as response_length,
    RECORD_ATTRIBUTES['custom.context_count'] as context_count,
    
    -- System Information (from RESOURCE_ATTRIBUTES if needed)
    RESOURCE_ATTRIBUTES['db.user'] as db_user,
    RESOURCE_ATTRIBUTES['snow.warehouse.name'] as warehouse_name,
    
    -- Raw JSON for reference
    RECORD_ATTRIBUTES,
    RESOURCE_ATTRIBUTES
    
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE RECORD_ATTRIBUTES['snow.ai.observability.run.name'] = 'test_run_v26'
ORDER BY timestamp DESC;

-- 7. SAFETY METRICS ANALYSIS
SELECT 
    timestamp,
    RECORD_ATTRIBUTES['custom.username'] as username,
    RECORD_ATTRIBUTES['ai.observability.call.kwargs.query'] as query,
    RECORD_ATTRIBUTES['custom.query_toxicity_score']::FLOAT as query_toxicity,
    RECORD_ATTRIBUTES['custom.response_toxicity_score']::FLOAT as response_toxicity,
    RECORD_ATTRIBUTES['custom.hallucination_score']::FLOAT as hallucination_score,
    RECORD_ATTRIBUTES['custom.is_high_quality_response']::BOOLEAN as is_high_quality,
    RECORD_ATTRIBUTES['custom.response_length']::INT as response_length,
    RECORD_ATTRIBUTES['custom.context_count']::INT as context_count,
    
    -- Calculate safety flags
    CASE 
        WHEN RECORD_ATTRIBUTES['custom.query_toxicity_score']::FLOAT > 0.5 THEN 'TOXIC_QUERY'
        WHEN RECORD_ATTRIBUTES['custom.response_toxicity_score']::FLOAT > 0.5 THEN 'TOXIC_RESPONSE'
        WHEN RECORD_ATTRIBUTES['custom.hallucination_score']::FLOAT > 0.8 THEN 'HIGH_HALLUCINATION'
        ELSE 'SAFE'
    END as safety_status
    
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE RECORD_ATTRIBUTES['snow.ai.observability.object.name'] = 'METRICS_RAG'
  AND RECORD_ATTRIBUTES['custom.username'] IS NOT NULL
ORDER BY timestamp DESC;

-- ===========================================
-- DISCOVERY QUERIES
-- ===========================================

-- 8. DISCOVER ALL APPLICATIONS (to see what app names exist)
SELECT 
    RECORD_ATTRIBUTES['snow.ai.observability.object.name'] as app_name,
    RECORD_ATTRIBUTES['snow.ai.observability.object.type'] as object_type,
    COUNT(*) as trace_count,
    MIN(timestamp) as first_trace,
    MAX(timestamp) as last_trace
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE RECORD_ATTRIBUTES['snow.ai.observability.object.name'] IS NOT NULL
GROUP BY 
    RECORD_ATTRIBUTES['snow.ai.observability.object.name'],
    RECORD_ATTRIBUTES['snow.ai.observability.object.type']
ORDER BY last_trace DESC;

-- 9. DISCOVER RUNS FOR YOUR APPLICATION (to see what runs exist)
SELECT 
    RECORD_ATTRIBUTES['snow.ai.observability.object.name'] as app_name,
    RECORD_ATTRIBUTES['snow.ai.observability.run.id'] as run_id,
    RECORD_ATTRIBUTES['snow.ai.observability.run.name'] as run_name,
    RECORD_ATTRIBUTES['snow.ai.observability.object.version.name'] as version,
    COUNT(*) as trace_count,
    MIN(timestamp) as first_trace,
    MAX(timestamp) as last_trace
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE RECORD_ATTRIBUTES['snow.ai.observability.object.name'] = 'METRICS_RAG'
GROUP BY 
    RECORD_ATTRIBUTES['snow.ai.observability.object.name'], 
    RECORD_ATTRIBUTES['snow.ai.observability.run.id'],
    RECORD_ATTRIBUTES['snow.ai.observability.run.name'],
    RECORD_ATTRIBUTES['snow.ai.observability.object.version.name']
ORDER BY last_trace DESC;

-- 10. SPAN TYPE ANALYSIS
SELECT 
    RECORD_ATTRIBUTES['ai.observability.span_type'] as span_type,
    RECORD_ATTRIBUTES['snow.ai.observability.run.name'] as run_name,
    COUNT(*) as span_count,
    AVG(RECORD_ATTRIBUTES['custom.response_length']::INT) as avg_response_length,
    AVG(RECORD_ATTRIBUTES['custom.hallucination_score']::FLOAT) as avg_hallucination_score
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE RECORD_ATTRIBUTES['snow.ai.observability.object.name'] = 'METRICS_RAG'
  AND RECORD_ATTRIBUTES['ai.observability.span_type'] IS NOT NULL
GROUP BY 
    RECORD_ATTRIBUTES['ai.observability.span_type'],
    RECORD_ATTRIBUTES['snow.ai.observability.run.name']
ORDER BY span_count DESC;

-- 11. USER ACTIVITY ANALYSIS
SELECT 
    RECORD_ATTRIBUTES['custom.username'] as username,
    COUNT(*) as query_count,
    AVG(RECORD_ATTRIBUTES['custom.response_length']::INT) as avg_response_length,
    AVG(RECORD_ATTRIBUTES['custom.hallucination_score']::FLOAT) as avg_hallucination_score,
    MAX(timestamp) as last_activity
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE RECORD_ATTRIBUTES['snow.ai.observability.object.name'] = 'METRICS_RAG'
  AND RECORD_ATTRIBUTES['custom.username'] IS NOT NULL
GROUP BY RECORD_ATTRIBUTES['custom.username']
ORDER BY query_count DESC;
