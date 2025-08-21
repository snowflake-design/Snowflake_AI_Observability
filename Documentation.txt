# Snowflake AI Observability Documentation

## Overview

Snowflake AI Observability helps you evaluate and trace your AI applications. You can measure performance, run evaluations, and debug your applications. It uses TruLens library for tracking and evaluating applications.

## Types of Implementation

Based on the images provided:

**Type 1**: RAG application running inside Snowflake environment using Snowflake Notebooks
**Type 2**: On-premise RAG application that logs observability data to Snowflake using NPC (Network Policy Control)

## Prerequisites

### Required Privileges
```sql
GRANT DATABASE ROLE SNOWFLAKE.CORTEX_USER TO ROLE <your_role>;
GRANT CREATE EXTERNAL AGENT ON SCHEMA <your_schema> TO ROLE <your_role>;
GRANT CREATE TASK ON SCHEMA <your_schema> TO ROLE <your_role>;
GRANT EXECUTE TASK ON ACCOUNT TO ROLE <your_role>;
GRANT APPLICATION ROLE SNOWFLAKE.AI_OBSERVABILITY_EVENTS_LOOKUP TO ROLE <your_role>;
```

### Python Packages Required
```python
# Install these packages from Snowflake conda channel (for Type 1) or pip (for Type 2)
snowflake-ml-python
snowflake.core
trulens-core
trulens-providers-cortex
trulens-connectors-snowflake
```

---

## Type 1: Snowflake Environment Implementation

### Step 1: Setup Environment

Download the notebook from GitHub:
https://github.com/Snowflake-Labs/sfguide-getting-started-with-ai-observability/blob/main/getting-started-with-ai-observability.ipynb

Import the notebook file in Snowsight to create a new Snowflake notebook.

### Step 2: Install Packages in Snowflake Notebook

```python
# Install from Snowflake conda channel in your notebook
snowflake-ml-python
snowflake.core
trulens-core
trulens-providers-cortex
trulens-connectors-snowflake
```

### Step 3: Setup Snowflake Objects

```python
from snowflake.snowpark.context import get_active_session
session = get_active_session()
```

```sql
CREATE DATABASE IF NOT EXISTS cortex_search_tutorial_db;
CREATE OR REPLACE WAREHOUSE cortex_search_tutorial_wh WITH
    WAREHOUSE_SIZE='X-SMALL'
    AUTO_SUSPEND = 120
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED=TRUE;
USE WAREHOUSE cortex_search_tutorial_wh;
```

### Step 4: Data Preparation

#### Create Stage
```sql
CREATE OR REPLACE STAGE cortex_search_tutorial_db.public.fomc
    DIRECTORY = (ENABLE = TRUE)
    ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');
```

#### Upload Documents
Upload your PDF files to the stage through Snowsight:
1. Navigate to cortex_search_tutorial_db.public.fomc
2. Upload PDF files
3. Verify upload: `ls @cortex_search_tutorial_db.public.fomc`

#### Parse Documents
```sql
CREATE OR REPLACE TABLE CORTEX_SEARCH_TUTORIAL_DB.PUBLIC.PARSED_FOMC_CONTENT AS SELECT
    relative_path,
    TO_VARCHAR(
        SNOWFLAKE.CORTEX.PARSE_DOCUMENT(
            @cortex_search_tutorial_db.public.fomc,
            relative_path,
            {'mode': 'LAYOUT'}
        ):content
    ) AS parsed_text
FROM directory(@cortex_search_tutorial_db.public.fomc)
WHERE relative_path LIKE '%.pdf';
```

#### Chunk Text
```sql
CREATE OR REPLACE TABLE CORTEX_SEARCH_TUTORIAL_DB.PUBLIC.CHUNKED_FOMC_CONTENT (
    file_name VARCHAR,
    CHUNK VARCHAR
);

INSERT INTO CORTEX_SEARCH_TUTORIAL_DB.PUBLIC.CHUNKED_FOMC_CONTENT (file_name, CHUNK)
SELECT
    relative_path,
    c.value AS CHUNK
FROM
    CORTEX_SEARCH_TUTORIAL_DB.PUBLIC.PARSED_FOMC_CONTENT,
    LATERAL FLATTEN(input => SNOWFLAKE.CORTEX.SPLIT_TEXT_RECURSIVE_CHARACTER(
        parsed_text,
        'markdown',
        1800,
        250
    )) c;
```

#### Create Search Service
```sql
CREATE OR REPLACE CORTEX SEARCH SERVICE CORTEX_SEARCH_TUTORIAL_DB.PUBLIC.FOMC_SEARCH_SERVICE
    ON chunk
    WAREHOUSE = cortex_search_tutorial_wh
    TARGET_LAG = '1 hour'
    EMBEDDING_MODEL = 'snowflake-arctic-embed-l-v2.0'
AS (
    SELECT
        file_name,
        chunk
    FROM CORTEX_SEARCH_TUTORIAL_DB.PUBLIC.CHUNKED_FOMC_CONTENT
);
```

### Step 5: Create Retriever Class

```python
from snowflake.snowpark.context import get_active_session
session = get_active_session()
import os
from snowflake.core import Root
from typing import List
from snowflake.snowpark.session import Session

class CortexSearchRetriever:
    def __init__(self, snowpark_session: Session, limit_to_retrieve: int = 4):
        self._snowpark_session = snowpark_session
        self._limit_to_retrieve = limit_to_retrieve

    def retrieve(self, query: str) -> List[str]:
        root = Root(session)
        search_service = (root
            .databases["CORTEX_SEARCH_TUTORIAL_DB"]
            .schemas["PUBLIC"]
            .cortex_search_services["FOMC_SEARCH_SERVICE"]
        )
        resp = search_service.search(
            query=query,
            columns=["chunk"],
            limit=self._limit_to_retrieve
        )
        if resp.results:
            return [curr["chunk"] for curr in resp.results]
        else:
            return []

retriever = CortexSearchRetriever(snowpark_session=session, limit_to_retrieve=3)
retrieved_context = retriever.retrieve(query="how was inflation expected to evolve in 2024?")
```

### Step 6: Enable TruLens Tracing

```python
import os
os.environ["TRULENS_OTEL_TRACING"] = "1"
```

```sql
create or replace database observability_db;
use database observability_db;
create or replace schema observability_schema;
use schema observability_schema;
```

### Step 7: Create Instrumented RAG System

```python
from snowflake.cortex import complete
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes

class RAG:
    def __init__(self):
        self.retriever = CortexSearchRetriever(snowpark_session=session, limit_to_retrieve=4)

    @instrument(
        span_type=SpanAttributes.SpanType.RETRIEVAL,
        attributes={
            SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
        }
    )
    def retrieve_context(self, query: str) -> list:
        """Retrieve relevant text from vector store."""
        return self.retriever.retrieve(query)

    @instrument(span_type=SpanAttributes.SpanType.GENERATION)
    def generate_completion(self, query: str, context_str: list) -> str:
        """Generate answer from context."""
        prompt = f"""
You are an expert assistant extracting information from context provided.
Answer the question in long-form, fully and completely, based on the context. Do not hallucinate.
If you don´t have the information just say so. If you do have the information you need, just tell me the answer.

Context: {context_str}

Question:
{query}

Answer:
"""
        response = ""
        stream = complete("mistral-large2", prompt, stream=True)
        for update in stream:
            response += update
            print(update, end='')
        return response

    @instrument(
        span_type=SpanAttributes.SpanType.RECORD_ROOT,
        attributes={
            SpanAttributes.RECORD_ROOT.INPUT: "query",
            SpanAttributes.RECORD_ROOT.OUTPUT: "return",
        })
    def query(self, query: str) -> str:
        context_str = self.retrieve_context(query)
        return self.generate_completion(query, context_str)

rag = RAG()
```

### Step 8: Test RAG System

```python
response = rag.query("how was inflation expected to evolve in 2024?")
```

### Step 9: Register App with TruLens

```python
from trulens.apps.app import TruApp
from trulens.connectors.snowflake import SnowflakeConnector

tru_snowflake_connector = SnowflakeConnector(snowpark_session=session)

app_name = "fed_reserve_rag"
app_version = "cortex_search"

tru_rag = TruApp(
    rag,
    app_name=app_name,
    app_version=app_version,
    connector=tru_snowflake_connector
)
```

### Step 10: Prepare Test Dataset

Download the dataset: https://github.com/Snowflake-Labs/sfguide-getting-started-with-ai-observability/blob/main/fomc_dataset.csv

Upload fomc_dataset.csv to Snowflake:
1. Select Data → Add Data → Load data into a Table
2. Choose OBSERVABILITY_DB.OBSERVABILITY_SCHEMA
3. Create table FOMC_DATA
4. Map columns to QUERY and GROUND_TRUTH_RESPONSE

### Step 11: Configure and Run Evaluation

```python
from trulens.core.run import Run
from trulens.core.run import RunConfig

run_name = "experiment_1_run"
run_config = RunConfig(
    run_name=run_name,
    dataset_name="FOMC_DATA",
    description="Questions about the Federal Open Market Committee meetings",
    label="fomc_rag_eval",
    source_type="TABLE",
    dataset_spec={
        "input": "QUERY",
        "ground_truth_output": "GROUND_TRUTH_RESPONSE",
    },
)

run: Run = tru_rag.add_run(run_config=run_config)
```

### Step 12: Start Evaluation

```python
run.start()
```

### Step 13: Compute Metrics

```python
run.compute_metrics([
    "answer_relevance",
    "context_relevance",
    "groundedness",
])
```

---

## Type 2: On-Premise Implementation

### Step 1: Environment Setup

```python
import os
os.environ["TRULENS_OTEL_TRACING"] = "1"

# Cannot run in Snowflake Notebook - must be external Python environment
# Install packages using pip:
# pip install snowflake-ml-python>=2.1.2
# pip install snowflake.core
# pip install trulens-core
# pip install trulens-providers-cortex  
# pip install trulens-connectors-snowflake
```

### Step 2: Create Snowflake Connection

```python
from snowflake.snowpark import Session

connection_parameters = {
    "account": "<account>",
    "user": "<user>",
    "password": "<password>",
    "database": "<database>",
    "schema": "<schema>",
    "warehouse": "<warehouse>",
    "role": "<role>",
}

session = Session.builder.configs(connection_parameters).create()
```

### Step 3: Instrument Your Application

```python
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes

# Your application code with instrumentation
class YourRAGApp:
    @instrument(
        span_type=SpanAttributes.SpanType.RETRIEVAL,
        attributes={
            SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
        }
    )
    def retrieve_context(self, query: str) -> list:
        # Your retrieval logic
        return retrieved_contexts

    @instrument(span_type=SpanAttributes.SpanType.GENERATION)
    def generate_completion(self, query: str, context_str: list) -> str:
        # Your generation logic
        return response

    @instrument(
        span_type=SpanAttributes.SpanType.RECORD_ROOT,
        attributes={
            SpanAttributes.RECORD_ROOT.INPUT: "query",
            SpanAttributes.RECORD_ROOT.OUTPUT: "return",
        }
    )
    def query(self, query: str) -> str:
        context_str = self.retrieve_context(query)
        return self.generate_completion(query, context_str)
```

### Step 4: Register with Snowflake

```python
from trulens.apps.app import TruApp
from trulens.connectors.snowflake import SnowflakeConnector

tru_snowflake_connector = SnowflakeConnector(snowpark_session=session)

app = YourRAGApp()

tru_app = TruApp(
    app,
    app_name="your_app_name",
    app_version="v1.0",
    connector=tru_snowflake_connector
)
```

### Step 5: Create and Run Evaluation

```python
from trulens.core.run import RunConfig
import pandas as pd

# Using DataFrame
run_config = RunConfig(
    run_name="your_run_name",
    description="description",
    label="label",
    source_type="DATAFRAME",
    dataset_name="your_dataset_name",
    dataset_spec={
        "RECORD_ROOT.INPUT": "input_column",
        "RECORD_ROOT.GROUND_TRUTH_OUTPUT": "expected_output_column",
    },
    llm_judge_name="mistral-large2"  # Optional
)

run = tru_app.add_run(run_config=run_config)
run.start(input_df=your_dataframe)

# Using Snowflake Table
run_config = RunConfig(
    run_name="your_run_name",
    description="description", 
    label="label",
    source_type="TABLE",
    dataset_name="YOUR_TABLE_NAME",
    dataset_spec={
        "RECORD_ROOT.INPUT": "input_column",
        "RECORD_ROOT.GROUND_TRUTH_OUTPUT": "expected_output_column",
    }
)

run = tru_app.add_run(run_config=run_config)
run.start()
```

---

## Available Metrics and Span Attributes

### Span Types

#### RECORD_ROOT (Main Application)
```python
SpanAttributes.SpanType.RECORD_ROOT
# Attributes:
SpanAttributes.RECORD_ROOT.INPUT
SpanAttributes.RECORD_ROOT.OUTPUT
SpanAttributes.RECORD_ROOT.GROUND_TRUTH_OUTPUT
```

#### RETRIEVAL (Search/Retrieval)
```python
SpanAttributes.SpanType.RETRIEVAL
# Attributes:
SpanAttributes.RETRIEVAL.QUERY_TEXT
SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS
```

#### GENERATION (LLM Generation)
```python
SpanAttributes.SpanType.GENERATION
# Attributes:
SpanAttributes.GENERATION.INPUT_MESSAGES
SpanAttributes.GENERATION.OUTPUT_TEXT
SpanAttributes.GENERATION.MODEL_NAME
```

### Available Metrics

```python
run.compute_metrics([
    "answer_relevance",      # Does answer address the question?
    "context_relevance",     # Is retrieved context relevant?
    "groundedness",          # Is answer supported by context?
    "coherence",            # Is response logically consistent?
    "correctness",          # Is answer factually correct? (needs ground truth)
    "helpfulness",          # How helpful is the response?
    "controversiality",     # Does response avoid controversial topics?
    "maliciousness"         # Is response free from harmful content?
])
```

### Dataset Specification Attributes

```python
dataset_spec = {
    "RECORD_ROOT.INPUT": "your_input_column",
    "RECORD_ROOT.OUTPUT": "your_output_column", 
    "RECORD_ROOT.GROUND_TRUTH_OUTPUT": "your_expected_output_column",
    "RETRIEVAL.QUERY_TEXT": "your_query_column",
    "RETRIEVAL.RETRIEVED_CONTEXTS": "your_contexts_column"
}
```

---

## Run Management

### Check Run Status
```python
run.get_status()
# Statuses: INVOCATION_IN_PROGRESS, INVOCATION_COMPLETED, INVOCATION_PARTIALLY_COMPLETED
```

### List Runs
```python
tru_app.list_runs()
```

### Get Specific Run
```python
run = tru_app.get_run(run_name="your_run_name")
```

### Run Operations
```python
run.describe()  # View run details
run.cancel()    # Cancel running run
run.delete()    # Delete run metadata (keeps traces)
```

---

## Viewing Results

### Snowsight Navigation
1. Navigate to Snowsight
2. Go to AI & ML → Evaluations
3. Select your application
4. Select your run
5. View results:
   - Aggregated metrics
   - Individual record results
   - Detailed traces
   - LLM judge explanations

### What You Can See
- Run overview with summary statistics
- Individual record performance
- Detailed execution traces
- Metric scores with explanations
- Performance data (latency, tokens)
- Side-by-side run comparisons

---

## Key URLs Referenced

- Type 1 Tutorial: https://quickstarts.snowflake.com/guide/getting_started_with_ai_observability/
- Type 2 Documentation: https://docs.snowflake.com/en/user-guide/snowflake-cortex/ai-observability/evaluate-ai-applications
- AI Observability Reference: https://docs.snowflake.com/en/user-guide/snowflake-cortex/ai-observability/reference
- TruLens Documentation: https://www.trulens.org/

---

## Important Notes

1. **Type 1** runs entirely within Snowflake using Snowflake Notebooks
2. **Type 2** runs externally but logs observability data to Snowflake
3. Version 2.1.2+ of packages required for Type 2
4. Cannot run Type 2 in Snowflake Notebook environment
5. TRULENS_OTEL_TRACING=1 environment variable required for Type 2
6. All observability data stored in SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS table
7. Metrics computed using "LLM-as-a-judge" approach with scores 0.0-1.0
