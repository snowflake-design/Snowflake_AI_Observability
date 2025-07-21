-- ========================================
-- Snowflake AI Observability SQL Queries
-- App: metric_rag | Run: test_run_v27
-- ========================================

-- Query 1: Get specific run data with all custom attributes
SELECT 
    timestamp,
    record_type,
    record_attributes:"ai_observability_run_name"::string as run_name,
    record_attributes:"custom_username"::string as username,
    record_attributes:"custom_query_toxicity_score"::float as query_toxicity_score,
    record_attributes:"custom_query_is_toxic"::boolean as query_is_toxic,
    record_attributes:"custom_response_toxicity_score"::float as response_toxicity_score,
    record_attributes:"custom_response_is_toxic"::boolean as response_is_toxic,
    record_attributes:"custom_hallucination_score"::float as hallucination_score,
    record_attributes:"custom_has_hallucination"::boolean as has_hallucination,
    record_attributes:"custom_response_length"::int as response_length,
    trace:"span_id"::string as span_id,
    trace:"trace_id"::string as trace_id
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE record_attributes:"ai_observability_run_name"::string = 'test_run_v27'
  AND record_attributes:"custom_username" IS NOT NULL
ORDER BY timestamp DESC;

-- ========================================

-- Query 2: Summary statistics for your run
SELECT 
    record_attributes:"ai_observability_run_name"::string as run_name,
    record_attributes:"custom_username"::string as username,
    COUNT(*) as total_records,
    AVG(record_attributes:"custom_query_toxicity_score"::float) as avg_query_toxicity,
    AVG(record_attributes:"custom_response_toxicity_score"::float) as avg_response_toxicity,
    AVG(record_attributes:"custom_hallucination_score"::float) as avg_hallucination_score,
    AVG(record_attributes:"custom_response_length"::int) as avg_response_length,
    SUM(CASE WHEN record_attributes:"custom_query_is_toxic"::boolean = true THEN 1 ELSE 0 END) as toxic_queries_count,
    SUM(CASE WHEN record_attributes:"custom_response_is_toxic"::boolean = true THEN 1 ELSE 0 END) as toxic_responses_count,
    SUM(CASE WHEN record_attributes:"custom_has_hallucination"::boolean = true THEN 1 ELSE 0 END) as hallucination_count
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE record_attributes:"ai_observability_run_name"::string = 'test_run_v27'
  AND record_attributes:"custom_username" IS NOT NULL
GROUP BY record_attributes:"ai_observability_run_name"::string, record_attributes:"custom_username"::string;

-- ========================================

-- Query 3: Get evaluation metrics for your run
SELECT 
    timestamp,
    record_attributes:"ai_observability_run_name"::string as run_name,
    record_attributes:"ai_observability_eval_metric_name"::string as metric_name,
    record_attributes:"ai_observability_eval_score"::float as metric_score,
    record_attributes:"ai_observability_eval_explanation"::string as metric_explanation
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE record_attributes:"ai_observability_run_name"::string = 'test_run_v27'
  AND record_attributes:"ai_observability_eval_metric_name" IS NOT NULL
ORDER BY timestamp DESC;

-- ========================================

-- Query 4: Safety Analysis - High Risk Queries/Responses
SELECT 
    timestamp,
    record_attributes:"custom_username"::string as username,
    record_attributes:"custom_query_toxicity_score"::float as query_toxicity,
    record_attributes:"custom_response_toxicity_score"::float as response_toxicity,
    record_attributes:"custom_hallucination_score"::float as hallucination_score,
    CASE 
        WHEN record_attributes:"custom_query_toxicity_score"::float > 0.5 THEN 'HIGH_TOXIC_QUERY'
        WHEN record_attributes:"custom_response_toxicity_score"::float > 0.5 THEN 'HIGH_TOXIC_RESPONSE'  
        WHEN record_attributes:"custom_hallucination_score"::float > 0.7 THEN 'HIGH_HALLUCINATION'
        ELSE 'SAFE'
    END as risk_level
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE record_attributes:"ai_observability_run_name"::string = 'test_run_v27'
  AND record_attributes:"custom_username" IS NOT NULL
ORDER BY 
    record_attributes:"custom_query_toxicity_score"::float DESC,
    record_attributes:"custom_response_toxicity_score"::float DESC,
    record_attributes:"custom_hallucination_score"::float DESC;

-- ========================================

-- Query 5: Time-series analysis of your run
SELECT 
    DATE_TRUNC('minute', timestamp) as minute_bucket,
    COUNT(*) as requests_per_minute,
    AVG(record_attributes:"custom_query_toxicity_score"::float) as avg_query_toxicity,
    AVG(record_attributes:"custom_response_toxicity_score"::float) as avg_response_toxicity,
    AVG(record_attributes:"custom_hallucination_score"::float) as avg_hallucination,
    AVG(record_attributes:"custom_response_length"::int) as avg_response_length
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE record_attributes:"ai_observability_run_name"::string = 'test_run_v27'
  AND record_attributes:"custom_username" IS NOT NULL
GROUP BY DATE_TRUNC('minute', timestamp)
ORDER BY minute_bucket DESC;

-- ========================================

-- Query 6: Complete raw data export for analysis
SELECT 
    timestamp,
    record_type,
    trace,
    resource_attributes,
    record_attributes,
    record
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE record_attributes:"ai_observability_run_name"::string = 'test_run_v27'
ORDER BY timestamp DESC;

-- ========================================

-- Query 7: Search for all runs by username
SELECT 
    record_attributes:"ai_observability_run_name"::string as run_name,
    record_attributes:"custom_username"::string as username,
    MIN(timestamp) as run_start,
    MAX(timestamp) as run_end,
    COUNT(*) as total_requests,
    AVG(record_attributes:"custom_hallucination_score"::float) as avg_hallucination
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE record_attributes:"custom_username"::string = 'data_scientist_user'
GROUP BY 
    record_attributes:"ai_observability_run_name"::string,
    record_attributes:"custom_username"::string
ORDER BY run_start DESC;

-- ========================================

-- Query 8: App performance comparison (if you have multiple apps)
SELECT 
    resource_attributes:"ai_observability_app_name"::string as app_name,
    record_attributes:"ai_observability_run_name"::string as run_name,
    COUNT(*) as total_requests,
    AVG(record_attributes:"custom_query_toxicity_score"::float) as avg_query_toxicity,
    AVG(record_attributes:"custom_response_toxicity_score"::float) as avg_response_toxicity,
    AVG(record_attributes:"custom_hallucination_score"::float) as avg_hallucination_score,
    AVG(record_attributes:"custom_response_length"::int) as avg_response_length
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE record_attributes:"custom_username" IS NOT NULL
GROUP BY 
    resource_attributes:"ai_observability_app_name"::string,
    record_attributes:"ai_observability_run_name"::string
ORDER BY avg_hallucination_score DESC;
