# Enhanced RAG with PII Detection - SQL Queries

## Raw Data Queries (Complete Records)

### 1. All Runs for Enhanced RAG Application
```sql
-- All data for enhanced_metric_rag application
SELECT * 
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE RECORD_ATTRIBUTES['snow.ai.observability.object.name'] = 'enhanced_metric_rag'
ORDER BY timestamp DESC;
```

### 2. Specific Enhanced Run Data
```sql
-- All data for enhanced_test_run_v30
SELECT * 
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE RECORD_ATTRIBUTES['snow.ai.observability.run.name'] = 'enhanced_test_run_v30'
ORDER BY timestamp DESC;
```

### 3. Most Recent Enhanced Run
```sql
-- Most recent run for enhanced_metric_rag
SELECT * 
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE RECORD_ATTRIBUTES['snow.ai.observability.run.id'] = (
    SELECT RECORD_ATTRIBUTES['snow.ai.observability.run.id']
    FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
    WHERE RECORD_ATTRIBUTES['snow.ai.observability.object.name'] = 'enhanced_metric_rag'
      AND RECORD_ATTRIBUTES['snow.ai.observability.run.id'] IS NOT NULL
    ORDER BY timestamp DESC
    LIMIT 1
)
ORDER BY timestamp DESC;
```

### 4. Current Day Enhanced Runs
```sql
-- Today's runs for enhanced_metric_rag
SELECT * 
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE DATE(timestamp) = CURRENT_DATE()
  AND RECORD_ATTRIBUTES['snow.ai.observability.object.name'] = 'enhanced_metric_rag'
ORDER BY timestamp DESC;
```

## Structured Queries (Extracted Attributes)

### 5. Enhanced RAG Application Overview
```sql
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
    
    -- AI Observability Attributes
    RECORD_ATTRIBUTES['snow.ai.observability.object.name'] as app_name,
    RECORD_ATTRIBUTES['snow.ai.observability.run.id'] as run_id,
    RECORD_ATTRIBUTES['snow.ai.observability.run.name'] as run_name,
    RECORD_ATTRIBUTES['snow.ai.observability.object.type'] as object_type,
    RECORD_ATTRIBUTES['snow.ai.observability.object.version.name'] as version_name,
    RECORD_ATTRIBUTES['snow.ai.observability.database.name'] as database_name,
    RECORD_ATTRIBUTES['snow.ai.observability.schema.name'] as schema_name,
    RECORD_ATTRIBUTES['ai.observability.span.type'] as span_type,
    
    -- Function Call Information
    RECORD_ATTRIBUTES['ai.observability.call.function'] as function_name,
    RECORD_ATTRIBUTES['ai.observability.call.kwargs.query'] as input_query,
    RECORD_ATTRIBUTES['ai.observability.call.return'] as function_output,
    
    -- Record Root Information
    RECORD_ATTRIBUTES['ai.observability.record.root.input'] as record_input,
    RECORD_ATTRIBUTES['ai.observability.record.root.output'] as record_output,
    RECORD_ATTRIBUTES['ai.observability.record.root.ground.truth.output'] as ground_truth,
    
    -- Enhanced Custom Attributes (PII Detection)
    RECORD_ATTRIBUTES['custom.username'] as username,
    RECORD_ATTRIBUTES['custom.query.has.pii'] as query_has_pii,
    RECORD_ATTRIBUTES['custom.original.query'] as original_query,
    RECORD_ATTRIBUTES['custom.masked.query'] as masked_query,
    
    -- Toxicity Detection (Response Only)
    RECORD_ATTRIBUTES['custom.response.toxicity.score'] as response_toxicity_score,
    RECORD_ATTRIBUTES['custom.response.is.toxic'] as response_is_toxic,
    
    -- Hallucination Detection
    RECORD_ATTRIBUTES['custom.hallucination.score'] as hallucination_score,
    RECORD_ATTRIBUTES['custom.has.hallucination'] as has_hallucination,
    
    -- Identifiers
    RECORD_ATTRIBUTES['ai.observability.app.id'] as app_id,
    RECORD_ATTRIBUTES['ai.observability.input.id'] as input_id,
    RECORD_ATTRIBUTES['ai.observability.record.id'] as record_id,
    
    -- Raw JSON for reference
    RECORD_ATTRIBUTES,
    RESOURCE_ATTRIBUTES
    
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE RECORD_ATTRIBUTES['snow.ai.observability.object.name'] = 'enhanced_metric_rag'
ORDER BY timestamp DESC;
```

### 6. Enhanced PII Detection Analysis
```sql
SELECT 
    timestamp,
    TRACE['trace_id'] as trace_id,
    
    -- User and Query Information
    RECORD_ATTRIBUTES['custom.username'] as username,
    RECORD_ATTRIBUTES['custom.original.query'] as original_query,
    RECORD_ATTRIBUTES['custom.masked.query'] as masked_query,
    
    -- PII Detection Results
    RECORD_ATTRIBUTES['custom.query.has.pii']::BOOLEAN as query_has_pii,
    
    -- Safety Analysis
    RECORD_ATTRIBUTES['custom.response.toxicity.score']::FLOAT as response_toxicity_score,
    RECORD_ATTRIBUTES['custom.response.is.toxic']::BOOLEAN as response_is_toxic,
    RECORD_ATTRIBUTES['custom.hallucination.score']::FLOAT as hallucination_score,
    RECORD_ATTRIBUTES['custom.has.hallucination']::BOOLEAN as has_hallucination,
    
    -- Span Information
    RECORD_ATTRIBUTES['ai.observability.span.type'] as span_type,
    RECORD_ATTRIBUTES['ai.observability.call.function'] as function_name,
    
    -- Calculate PII Risk Level
    CASE 
        WHEN RECORD_ATTRIBUTES['custom.query.has.pii']::BOOLEAN = true THEN 'PII_DETECTED'
        ELSE 'NO_PII'
    END as pii_status,
    
    -- Calculate Safety Status
    CASE 
        WHEN RECORD_ATTRIBUTES['custom.response.is.toxic']::BOOLEAN = true THEN 'TOXIC_RESPONSE'
        WHEN RECORD_ATTRIBUTES['custom.has.hallucination']::BOOLEAN = true THEN 'HALLUCINATION_DETECTED'
        ELSE 'SAFE'
    END as safety_status
    
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE RECORD_ATTRIBUTES['snow.ai.observability.object.name'] = 'enhanced_metric_rag'
  AND RECORD_ATTRIBUTES['snow.ai.observability.run.name'] = 'enhanced_test_run_v30'
  AND RECORD_ATTRIBUTES['custom.username'] IS NOT NULL
ORDER BY timestamp DESC;
```

### 7. Enhanced Safety Metrics Summary
```sql
SELECT 
    -- Run Information
    RECORD_ATTRIBUTES['snow.ai.observability.run.name'] as run_name,
    
    -- Overall Statistics
    COUNT(*) as total_queries,
    COUNT(DISTINCT RECORD_ATTRIBUTES['custom.username']) as unique_users,
    
    -- PII Statistics
    SUM(CASE WHEN RECORD_ATTRIBUTES['custom.query.has.pii']::BOOLEAN = true THEN 1 ELSE 0 END) as queries_with_pii,
    ROUND((SUM(CASE WHEN RECORD_ATTRIBUTES['custom.query.has.pii']::BOOLEAN = true THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 2) as pii_percentage,
    
    -- Toxicity Statistics (Response Only)
    AVG(RECORD_ATTRIBUTES['custom.response.toxicity.score']::FLOAT) as avg_response_toxicity_score,
    SUM(CASE WHEN RECORD_ATTRIBUTES['custom.response.is.toxic']::BOOLEAN = true THEN 1 ELSE 0 END) as toxic_responses,
    ROUND((SUM(CASE WHEN RECORD_ATTRIBUTES['custom.response.is.toxic']::BOOLEAN = true THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 2) as toxicity_percentage,
    
    -- Hallucination Statistics
    AVG(RECORD_ATTRIBUTES['custom.hallucination.score']::FLOAT) as avg_hallucination_score,
    SUM(CASE WHEN RECORD_ATTRIBUTES['custom.has.hallucination']::BOOLEAN = true THEN 1 ELSE 0 END) as hallucinated_responses,
    ROUND((SUM(CASE WHEN RECORD_ATTRIBUTES['custom.has.hallucination']::BOOLEAN = true THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 2) as hallucination_percentage,
    
    -- Time Range
    MIN(timestamp) as first_query,
    MAX(timestamp) as last_query
    
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE RECORD_ATTRIBUTES['snow.ai.observability.object.name'] = 'enhanced_metric_rag'
  AND RECORD_ATTRIBUTES['snow.ai.observability.run.name'] = 'enhanced_test_run_v30'
  AND RECORD_ATTRIBUTES['custom.username'] IS NOT NULL
GROUP BY RECORD_ATTRIBUTES['snow.ai.observability.run.name'];
```

### 8. Dynamic Username Activity Analysis
```sql
SELECT 
    RECORD_ATTRIBUTES['custom.username'] as username,
    COUNT(*) as query_count,
    
    -- PII Analysis per User
    SUM(CASE WHEN RECORD_ATTRIBUTES['custom.query.has.pii']::BOOLEAN = true THEN 1 ELSE 0 END) as queries_with_pii,
    
    -- Safety Metrics per User
    AVG(RECORD_ATTRIBUTES['custom.response.toxicity.score']::FLOAT) as avg_toxicity_score,
    AVG(RECORD_ATTRIBUTES['custom.hallucination.score']::FLOAT) as avg_hallucination_score,
    
    -- User Risk Profile
    CASE 
        WHEN SUM(CASE WHEN RECORD_ATTRIBUTES['custom.query.has.pii']::BOOLEAN = true THEN 1 ELSE 0 END) > 0 THEN 'HIGH_RISK'
        WHEN AVG(RECORD_ATTRIBUTES['custom.response.toxicity.score']::FLOAT) > 0.3 THEN 'MEDIUM_RISK'
        ELSE 'LOW_RISK'
    END as user_risk_profile,
    
    -- Activity Pattern
    MAX(timestamp) as last_activity,
    MIN(timestamp) as first_activity
    
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE RECORD_ATTRIBUTES['snow.ai.observability.object.name'] = 'enhanced_metric_rag'
  AND RECORD_ATTRIBUTES['snow.ai.observability.run.name'] = 'enhanced_test_run_v30'
  AND RECORD_ATTRIBUTES['custom.username'] IS NOT NULL
GROUP BY RECORD_ATTRIBUTES['custom.username']
ORDER BY query_count DESC;
```

### 9. PII Detection Details
```sql
SELECT 
    timestamp,
    RECORD_ATTRIBUTES['custom.username'] as username,
    RECORD_ATTRIBUTES['custom.original.query'] as original_query,
    RECORD_ATTRIBUTES['custom.masked.query'] as masked_query,
    
    -- Show the difference between original and masked
    CASE 
        WHEN RECORD_ATTRIBUTES['custom.original.query'] != RECORD_ATTRIBUTES['custom.masked.query'] 
        THEN 'PII_MASKED'
        ELSE 'NO_MASKING_NEEDED'
    END as masking_applied,
    
    -- Additional context
    RECORD_ATTRIBUTES['ai.observability.span.type'] as span_type,
    RECORD_ATTRIBUTES['ai.observability.call.function'] as function_name
    
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE RECORD_ATTRIBUTES['snow.ai.observability.object.name'] = 'enhanced_metric_rag'
  AND RECORD_ATTRIBUTES['snow.ai.observability.run.name'] = 'enhanced_test_run_v30'
  AND RECORD_ATTRIBUTES['custom.query.has.pii']::BOOLEAN = true
ORDER BY timestamp DESC;
```

### 10. Evaluation Metrics Results
```sql
SELECT 
    timestamp,
    RECORD_ATTRIBUTES['metric.name'] as metric_name,
    RECORD_ATTRIBUTES['score']::FLOAT as metric_score,
    RECORD_ATTRIBUTES['explanation'] as metric_explanation,
    RECORD_ATTRIBUTES['snow.ai.observability.run.name'] as run_name,
    
    -- Categorize metric performance
    CASE 
        WHEN RECORD_ATTRIBUTES['score']::FLOAT >= 0.8 THEN 'EXCELLENT'
        WHEN RECORD_ATTRIBUTES['score']::FLOAT >= 0.6 THEN 'GOOD'
        WHEN RECORD_ATTRIBUTES['score']::FLOAT >= 0.4 THEN 'FAIR'
        ELSE 'NEEDS_IMPROVEMENT'
    END as performance_category
    
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE RECORD_ATTRIBUTES['snow.ai.observability.object.name'] = 'enhanced_metric_rag'
  AND RECORD_ATTRIBUTES['snow.ai.observability.run.name'] = 'enhanced_test_run_v30'
  AND RECORD_ATTRIBUTES['metric.name'] IS NOT NULL
ORDER BY timestamp DESC;
```

## Discovery Queries

### 11. Discover All Enhanced Applications
```sql
SELECT 
    RECORD_ATTRIBUTES['snow.ai.observability.object.name'] as app_name,
    RECORD_ATTRIBUTES['snow.ai.observability.object.type'] as object_type,
    RECORD_ATTRIBUTES['snow.ai.observability.object.version.name'] as version,
    COUNT(*) as trace_count,
    MIN(timestamp) as first_trace,
    MAX(timestamp) as last_trace
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE RECORD_ATTRIBUTES['snow.ai.observability.object.name'] LIKE '%metric_rag%'
GROUP BY 
    RECORD_ATTRIBUTES['snow.ai.observability.object.name'],
    RECORD_ATTRIBUTES['snow.ai.observability.object.type'],
    RECORD_ATTRIBUTES['snow.ai.observability.object.version.name']
ORDER BY last_trace DESC;
```

### 12. Discover Enhanced Runs
```sql
SELECT 
    RECORD_ATTRIBUTES['snow.ai.observability.object.name'] as app_name,
    RECORD_ATTRIBUTES['snow.ai.observability.run.id'] as run_id,
    RECORD_ATTRIBUTES['snow.ai.observability.run.name'] as run_name,
    RECORD_ATTRIBUTES['snow.ai.observability.object.version.name'] as version,
    COUNT(*) as trace_count,
    MIN(timestamp) as first_trace,
    MAX(timestamp) as last_trace
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE RECORD_ATTRIBUTES['snow.ai.observability.object.name'] = 'enhanced_metric_rag'
GROUP BY 
    RECORD_ATTRIBUTES['snow.ai.observability.object.name'], 
    RECORD_ATTRIBUTES['snow.ai.observability.run.id'],
    RECORD_ATTRIBUTES['snow.ai.observability.run.name'],
    RECORD_ATTRIBUTES['snow.ai.observability.object.version.name']
ORDER BY last_trace DESC;
```

### 13. Enhanced Span Type Analysis
```sql
SELECT 
    RECORD_ATTRIBUTES['ai.observability.span.type'] as span_type,
    RECORD_ATTRIBUTES['snow.ai.observability.run.name'] as run_name,
    COUNT(*) as span_count,
    
    -- PII Analysis by Span Type
    SUM(CASE WHEN RECORD_ATTRIBUTES['custom.query.has.pii']::BOOLEAN = true THEN 1 ELSE 0 END) as spans_with_pii,
    
    -- Safety Analysis by Span Type
    AVG(RECORD_ATTRIBUTES['custom.response.toxicity.score']::FLOAT) as avg_toxicity_score,
    AVG(RECORD_ATTRIBUTES['custom.hallucination.score']::FLOAT) as avg_hallucination_score,
    
    -- Performance by Span Type
    COUNT(DISTINCT RECORD_ATTRIBUTES['custom.username']) as unique_users_per_span
    
FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS 
WHERE RECORD_ATTRIBUTES['snow.ai.observability.object.name'] = 'enhanced_metric_rag'
  AND RECORD_ATTRIBUTES['ai.observability.span.type'] IS NOT NULL
GROUP BY 
    RECORD_ATTRIBUTES['ai.observability.span.type'],
    RECORD_ATTRIBUTES['snow.ai.observability.run.name']
ORDER BY span_count DESC;
```

## Usage Notes

1. **Application Name**: `enhanced_metric_rag`
2. **Run Name**: `enhanced_test_run_v30`
3. **Version**: `v2.0`
4. **Custom Attributes**: Use dot notation (e.g., `custom.query.has.pii`)
5. **Enhanced Features**: PII detection, toxicity on responses only, hallucination detection
6. **Dynamic Usernames**: 8 different usernames from predefined list

## Key Enhanced Attributes

- `custom.username`: Dynamic username from list
- `custom.query.has.pii`: Boolean PII detection
- `custom.original.query`: Original user query
- `custom.masked.query`: PII-masked version
- `custom.response.toxicity.score`: Toxicity score (responses only)
- `custom.response.is.toxic`: Boolean toxicity flag
- `custom.hallucination.score`: Hallucination detection score
- `custom.has.hallucination`: Boolean hallucination flag
