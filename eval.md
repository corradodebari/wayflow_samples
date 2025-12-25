# Automated NL2SQL testing
Since SQLcl Wayflow agent translations (as proposed in the previous post) are not always reliable, someone asked me how to compare different prompts —or, more generally, more advanced approaches — to Natural Language–to–SQL translation.

Here’s a proposal to:
	•	Generate a dataset of (Question → SQL statement) pairs from an existing database schema
	•	Evaluate dataset accuracy (and compare prompts/models consistently)
  
<div align="center">
<img align="center" src="https://raw.githubusercontent.com/corradodebari/wayflow_samples/main/images/coverTesting.png" width="600px">
</div>

## Overview
This example leverages the [DataFlow project](https://github.com/OpenDCAI/DataFlow), an open-source framework (Apache-2.0) for data preparation and training. DataFlow turns noisy sources (PDFs, plain text, low-quality Q&A) into high-quality data to improve domain LLM performance—either through targeted training (pre-training, SFT, RL) or via RAG with cleaned and validated knowledge bases. It has been validated across healthcare, finance, and legal domains.

In particular, I used the **Text2SQL pipeline**, which translates natural-language questions into SQL queries and can enrich outputs with explanations, chain-of-thought reasoning, and schema context.

For this work, I did not run the full pipeline end-to-end. Instead, I reused it specifically for test dataset generation, to:
- inspect the database schema
- generate SQL statements grounded in the schema
- generate corresponding reference natural-language questions

This provides an independent and bias-free way to build (Question → SQL) pairs. The agent under test does not “author” its own benchmark: questions are derived from schema-consistent SQL generated directly from the actual database, and only then converted into natural language.

After generating a .jsonl file containing the question set, the testing phase starts: each question is submitted to the agent (modified to return SQL only) and the generated SQL is collected.

For scoring, I compare the result tables produced by the reference SQL and the predicted SQL, rather than attempting a syntactic/semantic comparison of the queries themselves (which is significantly more complex). This execution-based evaluation is widely used in Text2SQL benchmarks such as **Spider** [1.0](https://yale-lily.github.io/spider) and [2.0](https://spider2-sql.github.io/). With a static schema, it is generally reliable. I also introduced a few parameters to tune the matching criteria.

Finally, accuracy is computed as the percentage of correct translations over the total number of questions.

## Test Dataset Generation
It will be executed by: **<a href="https://github.com/corradodebari/wayflow_samples/blob/main/text2sql_pipeline_gen.py">text2sql_pipeline_gen.py</a>**, with the support of the connector developed for the Oracle DBMS **<a href="https://github.com/corradodebari/wayflow_samples/blob/main/utils
/database_connectors/oracledb_connector.py">oracledb_connector.py</a>**, based on the MySQL implementation.

Setting the `user`/schema are you interested to test your agent, for example the SH used in the original agent, and providing the credentials, in addition to llm references for chat/embeddings required:
```sql
    DB_CFG = {
            'user': 'SH',
            'password': 'Welcome_12345', 
            'host': 'localhost',
            'port': 1521,
            'service_name': 'FREEPDB1'
        }

    LLM_CFG =  {
            'chat_model': 'gpt-4o-mini',
            'chat_url' : 'https://api.openai.com/v1/chat/completions',
            'embedding_model' : 'text-embedding-3-small',
            'embedding_url': 'https://api.openai.com/v1/embeddings'
        }

    GEN_CONF = {
            'nq':10
        }
```
The modified `text2sql_pipeline_gen.py` runs only the first three stages of the pipeline:
1. **SQLGenerator**: automatically generates SQL query statements from the database schema, producing the raw SQL candidates used by downstream steps.
2. **SQLExecutionFilter**: validates SQL correctness by executing each statement and filtering out queries that fail to run successfully.
3. **Text2SQLQuestionGenerator**: generates natural-language questions for each SQL statement, building Text-to-SQL question/answer pairs. It can produce multiple candidate questions per query.

The script outputs 'cache/dataflow_cache_step_step3.jsonl' will later be extended with a set of gold fields and saved as 'cache/dataflow_cache_step_step3_updated.jsonl'.

## Testing
It will be executed by: **<a href="https://github.com/corradodebari/wayflow_samples/blob/main/howto_mcp_sqlcl_test.py">howto_mcp_sqlcl_test.py</a>**
The original agent file has been modified to generate SQL only, setting in the prompt:

```python
### Output Format
- Return ONLY the final raw SQL generated
- Do NOT include explanations, comments, or debug output

You are not allowed to skip schema validation.
You are not allowed to return raw SQL without execution
```
The questions collection will be get by:
- `cache/dataflow_cache_step_step3_updated.jsonl`

and the related SQL translated will be added in the file:
- `cache/dataflow_cache_step_step3_q_a.jsonl`  

## Scoring

It will be executed by: **<a href="https://github.com/corradodebari/wayflow_samples/blob/main/evaluate_sql.py">evaluate_sql.py</a>**

At the moment, a match (positive/negative) is determined by comparing query results, not SQL syntax.
Currently, scoring is computed as a simple accuracy ratio:

`accuracy = positive_matches / total_questions`

A more sophisticated approach would be to annotate each reference query with must-have fields (i.e., columns that must appear in the translated SQL). That would make the scoring more tolerant and better aligned with intent, but it also introduces manual labeling effort.

Because the translated SQL (B) may return additional columns compared to the reference (A)—and because analytical functions may produce equivalent results with different projections—the implementation favors a fast, execution-based matching strategy:
1.	Prune extra columns from B using a row-order–invariant column fingerprint.
2.	After pruning, A and B_reduced have the same shape.
3.	Match columns via permutation, prioritizing “most discriminative” columns first (e.g., by variance), to avoid exploring all possible column permutations.

This approach supports:
- row permutation (when order is irrelevant),
- column permutation (when projections are equivalent up to ordering),
- extra columns in B (with strong pruning).

Internally, it uses fingerprints (treated as multisets) plus backtracking, and only performs column permutations where necessary.

### Tunable matching parameters

To control how strict the comparison is between reference results (A) and translated results (B), the following parameters can be adjusted:
- **require_same_columns**: whether the result sets must have exactly the same columns.
	- **True**: B must have the same number of columns as A (no pruning/subset selection).
	- **False**: B may include extra columns; the matcher tries to align A within B (even if column names differ).
- **require_same_columns_names**: if True, column names must match exactly (in addition to values).
- **consider_duplicates**: if False, duplicate rows are collapsed (i.e., comparisons treat results as sets rather than multisets).
- **float_factor**: controls tolerance for floating-point comparisons, useful for aggregates like AVG() that may introduce small numeric differences. Floats are normalized via `int(round(v * float_factor))` (e.g., 100 ≈ 2 decimal places).
- **ignore_order**:
	- **False**: only column permutation is allowed; row order must match.
	- **True**: both column and row permutations are allowed.
	
	If the reference SQL contains an ORDER BY, ignore_order is automatically forced to False.

The output will be in `cache/dataflow_cache_step_step3_q_a_test.jsonl` 

### Examples

1. Queries to compare:

```sql
    -- Reference
    SELECT PROD_ID AS ID, UNIT_PRICE FROM SH.COSTS WHERE TIME_ID > TO_DATE('2020-01-01', 'YYYY-MM-DD') order by ID
    -- Translated
    SELECT PROD_ID, UNIT_COST, UNIT_PRICE FROM SH.COSTS WHERE TIME_ID > TO_DATE('2020-01-01', 'YYYY-MM-DD') order by PROD_ID
```
- With :
    - ignore_order = False
    - require_same_columns = True
    - require_same_columns_names = False

    NEGATIVE: Why? 2 fields in reference, 3 in translated

- With :
    - ignore_order = False
    - require_same_columns = False
    - require_same_columns_names = False

    POSITIVE: Why? The 2 fields in reference query are in translated, even names ID and PROD_ID are not the same. The order is guaranteed with ORDER BY clause on both.

2. Queries to compare:

```sql
    -- Reference
    SELECT PROD_ID AS ID, UNIT_PRICE FROM SH.COSTS WHERE TIME_ID > TO_DATE('2020-01-01', 'YYYY-MM-DD') order by ID
    -- Translated
    SELECT PROD_ID, UNIT_COST, UNIT_PRICE FROM SH.COSTS WHERE TIME_ID > TO_DATE('2020-01-01', 'YYYY-MM-DD')
```

- With:
    - ignore_order = True
    - require_same_columns = False
    - require_same_columns_names = False

    POSITIVE: Why? Even order by in the reference, the ignore_order allow permutation on rows to find the match on all rows.

## Setup & Running
### Dataflow/WayFlow env

```shell
python3.11 -m venv .venv2 --copies
pip3.11 install --upgrade pip wheel setuptools uv
source .venv2/bin/activate
pip install -e <GitHub/wayflow/wayflowcore absolute path> oci oracledb open-dataflow
```

### Run
```shell
# Copy: utils/database_connectors/oracledb_connector.py
source .venv2/bin/activate
export DF_API_KEY=$OPENAI_API_KEY
.venv2/bin/python text2sql_pipeline_gen.py
.venv2/bin/python howto_mcp_sqlcl_test.py 
.venv2/bin/python evaluate_sql.py
```

## Conclusion
With this implementation I've provided a way to test massively the agents running a NL2SQL translation, in order to tune the prompt and the orchestration to adapt the agent behavior and accuracy to the schema will be used as a target of the requests.

---

## Disclaimer
*The views expressed in this paper are my own and do not necessarily reflect the views of Oracle.*
