-- ========================================
-- COMPREHENSIVE AI OBSERVABILITY TRACES ANALYSIS
-- App: metric_rag | Run: test_run_v27 | User: data_scientist_user
-- ========================================

-- ===========================================
-- 1. CURRENT RUN TRACES (test_run_v27)
-- ===========================================

-- Get all traces for your specific run
SELECT 
    timestamp,
    start_timestamp,
    record_type,
    TRACE['trace_id'] as trace_id,
    TRACE['span_id'] as span_id,
    RECORD['name'] as span_name,
    RECORD_ATTRIBUTES['ai_observability_run_name'] as run_name,
    RECORD_ATTRIBUTES['custom_username'] as username,
    RECORD_ATTRIBUTES['custom_query_toxicity_score'] as query_toxicity,
    RECORD_ATTRIBUTES['custom_response_toxicity_score'] as response_toxicity,
    RECORD_ATTRIBUTES['custom_hallucination_score'] as hallucination,
    RECORD_ATTRIBUTES['custom_response_length'] as response_length,
    RESOURCE_ATTRIBUTES['snow.executable.name'] as executable_name,
    RECORD_ATTRIBUTES,
    RESOURCE_ATTRIBUTES
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE RECORD_ATTRIBUTES['ai_observability_run_name'] = 'test_run_v27'
ORDER BY timestamp DESC;

-- ===========================================
-- 2. CURRENT DAY TRACES (All your app activity today)
-- ===========================================

-- Get all traces for your application today
SELECT 
    timestamp,
    start_timestamp,
    record_type,
    TRACE['trace_id'] as trace_id,
    TRACE['span_id'] as span_id,
    RECORD['name'] as span_name,
    RECORD_ATTRIBUTES['ai_observability_run_name'] as run_name,
    RECORD_ATTRIBUTES['custom_username'] as username,
    RECORD_ATTRIBUTES['custom_query_toxicity_score'] as query_toxicity,
    RECORD_ATTRIBUTES['custom_response_toxicity_score'] as response_toxicity,
    RECORD_ATTRIBUTES['custom_hallucination_score'] as hallucination,
    RESOURCE_ATTRIBUTES['snow.executable.name'] as executable_name
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE DATE(timestamp) = CURRENT_DATE()
  AND (
    RECORD_ATTRIBUTES['custom_username'] = 'data_scientist_user'
    OR CONTAINS(RESOURCE_ATTRIBUTES::string, 'metric_rag')
    OR CONTAINS(RECORD_ATTRIBUTES::string, 'data_scientist_user')
  )
ORDER BY timestamp DESC;

-- ===========================================
-- 3. LIFETIME TRACES (All historical traces for your app)
-- ===========================================

-- Get all traces ever recorded for your application
SELECT 
    timestamp,
    start_timestamp,
    record_type,
    TRACE['trace_id'] as trace_id,
    TRACE['span_id'] as span_id,
    RECORD['name'] as span_name,
    RECORD_ATTRIBUTES['ai_observability_run_name'] as run_name,
    RECORD_ATTRIBUTES['custom_username'] as username,
    RECORD_ATTRIBUTES['custom_query_toxicity_score'] as query_toxicity,
    RECORD_ATTRIBUTES['custom_response_toxicity_score'] as response_toxicity,
    RECORD_ATTRIBUTES['custom_hallucination_score'] as hallucination,
    RESOURCE_ATTRIBUTES['snow.executable.name'] as executable_name,
    DATE(timestamp) as event_date
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE RECORD_ATTRIBUTES['custom_username'] = 'data_scientist_user'
   OR CONTAINS(RESOURCE_ATTRIBUTES::string, 'metric_rag')
   OR CONTAINS(RECORD_ATTRIBUTES::string, 'data_scientist_user')
ORDER BY timestamp DESC;

-- ===========================================
-- 4. APPLICATION TRACES SUMMARY BY DAY
-- ===========================================

-- Daily summary of your application traces
SELECT 
    DATE(timestamp) as trace_date,
    RECORD_ATTRIBUTES['custom_username'] as username,
    COUNT(*) as total_traces,
    COUNT(DISTINCT TRACE['trace_id']) as unique_traces,
    COUNT(DISTINCT RECORD_ATTRIBUTES['ai_observability_run_name']) as unique_runs,
    AVG(RECORD_ATTRIBUTES['custom_query_toxicity_score']::float) as avg_query_toxicity,
    AVG(RECORD_ATTRIBUTES['custom_response_toxicity_score']::float) as avg_response_toxicity,
    AVG(RECORD_ATTRIBUTES['custom_hallucination_score']::float) as avg_hallucination,
    MIN(timestamp) as first_trace,
    MAX(timestamp) as last_trace
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE RECORD_ATTRIBUTES['custom_username'] = 'data_scientist_user'
   OR CONTAINS(RESOURCE_ATTRIBUTES::string, 'metric_rag')
GROUP BY DATE(timestamp), RECORD_ATTRIBUTES['custom_username']
ORDER BY trace_date DESC;

-- ===========================================
-- 5. TRACES BY RUN NAME (All runs for your app)
-- ===========================================

-- Summary by run name
SELECT 
    RECORD_ATTRIBUTES['ai_observability_run_name'] as run_name,
    RECORD_ATTRIBUTES['custom_username'] as username,
    COUNT(*) as total_traces,
    COUNT(DISTINCT TRACE['trace_id']) as unique_traces,
    AVG(RECORD_ATTRIBUTES['custom_query_toxicity_score']::float) as avg_query_toxicity,
    AVG(RECORD_ATTRIBUTES['custom_response_toxicity_score']::float) as avg_response_toxicity,
    AVG(RECORD_ATTRIBUTES['custom_hallucination_score']::float) as avg_hallucination,
    MIN(timestamp) as run_start,
    MAX(timestamp) as run_end,
    DATEDIFF('second', MIN(timestamp), MAX(timestamp)) as run_duration_seconds
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE RECORD_ATTRIBUTES['custom_username'] = 'data_scientist_user'
   OR CONTAINS(RESOURCE_ATTRIBUTES::string, 'metric_rag')
GROUP BY RECORD_ATTRIBUTES['ai_observability_run_name'], RECORD_ATTRIBUTES['custom_username']
ORDER BY run_start DESC;

-- ===========================================
-- 6. RECENT TRACES (Last 24 hours)
-- ===========================================

-- Get traces from last 24 hours
SELECT 
    timestamp,
    start_timestamp,
    record_type,
    TRACE['trace_id'] as trace_id,
    TRACE['span_id'] as span_id,
    RECORD['name'] as span_name,
    RECORD_ATTRIBUTES['ai_observability_run_name'] as run_name,
    RECORD_ATTRIBUTES['custom_username'] as username,
    RECORD_ATTRIBUTES['custom_query_toxicity_score'] as query_toxicity,
    RECORD_ATTRIBUTES['custom_response_toxicity_score'] as response_toxicity,
    RECORD_ATTRIBUTES['custom_hallucination_score'] as hallucination,
    RESOURCE_ATTRIBUTES['snow.executable.name'] as executable_name
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '24 HOURS'
  AND (
    RECORD_ATTRIBUTES['custom_username'] = 'data_scientist_user'
    OR CONTAINS(RESOURCE_ATTRIBUTES::string, 'metric_rag')
    OR CONTAINS(RECORD_ATTRIBUTES::string, 'data_scientist_user')
  )
ORDER BY timestamp DESC;

-- ===========================================
-- 7. TRACE PERFORMANCE ANALYSIS
-- ===========================================

-- Performance analysis of your traces
SELECT 
    RECORD_ATTRIBUTES['ai_observability_run_name'] as run_name,
    record_type,
    RECORD['name'] as span_name,
    COUNT(*) as span_count,
    AVG(DATEDIFF('millisecond', start_timestamp, timestamp)) as avg_duration_ms,
    MIN(DATEDIFF('millisecond', start_timestamp, timestamp)) as min_duration_ms,
    MAX(DATEDIFF('millisecond', start_timestamp, timestamp)) as max_duration_ms,
    AVG(RECORD_ATTRIBUTES['custom_response_length']::int) as avg_response_length
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE RECORD_ATTRIBUTES['custom_username'] = 'data_scientist_user'
  AND start_timestamp IS NOT NULL
  AND timestamp IS NOT NULL
GROUP BY RECORD_ATTRIBUTES['ai_observability_run_name'], record_type, RECORD['name']
ORDER BY avg_duration_ms DESC;

-- ===========================================
-- 8. SAFETY METRICS ANALYSIS (Toxicity & Hallucination)
-- ===========================================

-- Safety analysis across all your traces
SELECT 
    RECORD_ATTRIBUTES['ai_observability_run_name'] as run_name,
    DATE(timestamp) as trace_date,
    COUNT(*) as total_requests,
    
    -- Toxicity Analysis
    AVG(RECORD_ATTRIBUTES['custom_query_toxicity_score']::float) as avg_query_toxicity,
    AVG(RECORD_ATTRIBUTES['custom_response_toxicity_score']::float) as avg_response_toxicity,
    SUM(CASE WHEN RECORD_ATTRIBUTES['custom_query_is_toxic']::boolean = true THEN 1 ELSE 0 END) as toxic_queries,
    SUM(CASE WHEN RECORD_ATTRIBUTES['custom_response_is_toxic']::boolean = true THEN 1 ELSE 0 END) as toxic_responses,
    
    -- Hallucination Analysis  
    AVG(RECORD_ATTRIBUTES['custom_hallucination_score']::float) as avg_hallucination,
    SUM(CASE WHEN RECORD_ATTRIBUTES['custom_has_hallucination']::boolean = true THEN 1 ELSE 0 END) as hallucinated_responses,
    
    -- Safety Score (0-100, higher is safer)
    ROUND(100 - (
        (AVG(RECORD_ATTRIBUTES['custom_query_toxicity_score']::float) * 50) +
        (AVG(RECORD_ATTRIBUTES['custom_response_toxicity_score']::float) * 50) +
        (AVG(RECORD_ATTRIBUTES['custom_hallucination_score']::float) * 30)
    ), 2) as safety_score
    
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE RECORD_ATTRIBUTES['custom_username'] = 'data_scientist_user'
  AND RECORD_ATTRIBUTES['custom_query_toxicity_score'] IS NOT NULL
GROUP BY RECORD_ATTRIBUTES['ai_observability_run_name'], DATE(timestamp)
ORDER BY trace_date DESC, safety_score ASC;

-- ===========================================
-- 9. COMPLETE TRACE DETAILS (Deep Dive)
-- ===========================================

-- Complete trace information with all details
SELECT 
    timestamp,
    start_timestamp,
    record_type,
    TRACE['trace_id'] as trace_id,
    TRACE['span_id'] as span_id,
    RECORD['name'] as span_name,
    RECORD['kind'] as span_kind,
    
    -- Run Information
    RECORD_ATTRIBUTES['ai_observability_run_name'] as run_name,
    RECORD_ATTRIBUTES['custom_username'] as username,
    
    -- Safety Metrics
    RECORD_ATTRIBUTES['custom_query_toxicity_score'] as query_toxicity,
    RECORD_ATTRIBUTES['custom_query_is_toxic'] as query_is_toxic,
    RECORD_ATTRIBUTES['custom_response_toxicity_score'] as response_toxicity,
    RECORD_ATTRIBUTES['custom_response_is_toxic'] as response_is_toxic,
    RECORD_ATTRIBUTES['custom_hallucination_score'] as hallucination,
    RECORD_ATTRIBUTES['custom_has_hallucination'] as has_hallucination,
    
    -- Performance Metrics
    RECORD_ATTRIBUTES['custom_response_length'] as response_length,
    DATEDIFF('millisecond', start_timestamp, timestamp) as duration_ms,
    
    -- System Information
    RESOURCE_ATTRIBUTES['snow.executable.name'] as executable_name,
    RESOURCE_ATTRIBUTES['snow.database.name'] as database_name,
    RESOURCE_ATTRIBUTES['snow.schema.name'] as schema_name,
    RESOURCE_ATTRIBUTES['db.user'] as db_user,
    
    -- Raw Data
    RECORD_ATTRIBUTES as all_record_attributes,
    RESOURCE_ATTRIBUTES as all_resource_attributes
    
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE RECORD_ATTRIBUTES['custom_username'] = 'data_scientist_user'
   OR RECORD_ATTRIBUTES['ai_observability_run_name'] = 'test_run_v27'
ORDER BY timestamp DESC;

-- ===========================================
-- 10. QUICK APP HEALTH CHECK
-- ===========================================

-- Quick health check for your application
SELECT 
    'Current Run (test_run_v27)' as scope,
    COUNT(*) as total_traces,
    COUNT(DISTINCT TRACE['trace_id']) as unique_traces,
    AVG(RECORD_ATTRIBUTES['custom_query_toxicity_score']::float) as avg_toxicity,
    AVG(RECORD_ATTRIBUTES['custom_hallucination_score']::float) as avg_hallucination,
    MAX(timestamp) as last_activity
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE RECORD_ATTRIBUTES['ai_observability_run_name'] = 'test_run_v27'

UNION ALL

SELECT 
    'Today' as scope,
    COUNT(*) as total_traces,
    COUNT(DISTINCT TRACE['trace_id']) as unique_traces,
    AVG(RECORD_ATTRIBUTES['custom_query_toxicity_score']::float) as avg_toxicity,
    AVG(RECORD_ATTRIBUTES['custom_hallucination_score']::float) as avg_hallucination,
    MAX(timestamp) as last_activity
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE DATE(timestamp) = CURRENT_DATE()
  AND RECORD_ATTRIBUTES['custom_username'] = 'data_scientist_user'

UNION ALL

SELECT 
    'Lifetime' as scope,
    COUNT(*) as total_traces,
    COUNT(DISTINCT TRACE['trace_id']) as unique_traces,
    AVG(RECORD_ATTRIBUTES['custom_query_toxicity_score']::float) as avg_toxicity,
    AVG(RECORD_ATTRIBUTES['custom_hallucination_score']::float) as avg_hallucination,
    MAX(timestamp) as last_activity
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE RECORD_ATTRIBUTES['custom_username'] = 'data_scientist_user'

ORDER BY 
    CASE scope 
        WHEN 'Current Run (test_run_v27)' THEN 1
        WHEN 'Today' THEN 2  
        WHEN 'Lifetime' THEN 3
    END;
