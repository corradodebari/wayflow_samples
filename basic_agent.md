# A Quoting Sales Agent in WayFlow with Oracle DB 26ai as vector store

<div align="center">
<img align="center" src="https://raw.githubusercontent.com/corradodebari/wayflow_samples/main/images/cover.png" width="600px">
</div>

## 1. High-level overview
In this demo we will explore how create a WayFlow based agent that leverages existing tabular data for similarity search in the Oracle AI DB 26ai. 

This script builds a **quoting assistant** that:

- Uses **Oracle Database** (SH sample schema) as a product catalog.
- Uses an **Ollama embedding model** to store text embeddings of product descriptions in an Oracle `VECTOR` type column.
- Exposes two database queries as **WayFlow tools**: one for **similarity** search, the other for an **exact** search.
- Wraps those tools in a **WayFlow Agent** that prepares a quote.
- Orchestrates the conversation with a small **Flow** (`InputMessageStep → AgentExecutionStep → OutputMessageStep`).
- Demonstrates **Agent Spec export/import** with `AgentSpecExporter` and `AgentSpecLoader`.  

<br><br>
<div align="center">
<img src="https://raw.githubusercontent.com/corradodebari/wayflow_samples/main/images/basic_agent.png" width="800px">
</div>

**Takeaways:**
- **WayFlow 25.4.1:** full documentation: **[here](https://oracle.github.io/wayflow/25.4.1/core/api/agentspec.html)**
- **Oracle AI Vector Search** User's Guide : **[here](https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/)**
- **Python Script** in the GitHub repo: **<a href="https://github.com/corradodebari/wayflow_samples/blob/main/quoting_sales_agent.py">quoting_sales_agent.py</a>**
- **DrawIO Project**:  **<a href="https://github.com/corradodebari/wayflow_samples/blob/main/Project.drawio">prj</a>**


## 2. Setup 
The following steps to prepare and run the demo:
- A Python 3.11 virtualenv with wayflowcore==25.4.1 and oracledb.
- An Ollama server running the nomic-embed-text model for embeddings.
- An OPENAI_API_KEY for the LLM.
- A local Oracle DB (DB Free / “DB26ai”) with the SH sample schema installed.

They are not WayFlow-specific but define the infrastructure that the WayFlow components (LLM, embedding model, and tools) will sit on.

```bash
# -----------
# Setup
# -----------
# python env:
# python3.11 -m venv .venv --copies
# source .venv/bin/activate
# pip install "wayflowcore==25.4.1" oracledb oci
#
# ollama model:
# ollama pull nomic-embed-text
# 
# Open AI Apikey
# export OPENAI_API_KEY=".............."
#
# Install DB26ai: 
# podman run -d --name db26ai -p 1521:1521 -e ORACLE_PWD=[PASSWORD] container-registry.oracle.com/database/free:latest
#
# Install Oracle DB Sample Schema from https://github.com/oracle/db-sample-schemas/releases/latest 
# wget https://github.com/oracle-samples/db-sample-schemas/archive/refs/tags/v23.3.tar.gz
# in /db-sample-schemas-23.3/sales_history
# sql sys/[PASSWORD]@//localhost:1521/FREEPDB1 as sysdba
# passord:[SCHEMA_PASSWORD]
# @sh_install.sql
#
# Run:
# source .venv/bin/activate
# python3.11 quoting_sales_agent.py
```

## 3. Imports and logging
Let's import the library we need: take in consideration that no LangChain or proprietary implementations are needed to interact with LLM providers. Wayflow offers a complete set of API to engage the LLMs. 

```python
import array
import os

import oracledb

from wayflowcore.embeddingmodels.ollamamodel import OllamaEmbeddingModel
from wayflowcore.models import LlmModelFactory, OllamaModel
from wayflowcore.models.llmgenerationconfig import LlmGenerationConfig
from wayflowcore.models.openaimodel import OpenAIModel
from wayflowcore.tools.toolhelpers import DescriptionMode, tool
from wayflowcore.agent import Agent, CallerInputMode
from wayflowcore.property import StringProperty
from wayflowcore.steps.agentexecutionstep import AgentExecutionStep
from wayflowcore.controlconnection import ControlFlowEdge
from wayflowcore.dataconnection import DataFlowEdge
from wayflowcore.flow import Flow
from wayflowcore.steps import InputMessageStep, OutputMessageStep
from wayflowcore.agentspec import AgentSpecExporter, AgentSpecLoader

import logging
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(message)s"
)
# Disable httpx and httpcore noisy logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
```

- oracledb is the Oracle Database Python driver; it provides the DB API connection and cursor used in tools and embedding preparation.
- WayFlow imports:
    - OllamaEmbeddingModel implements EmbeddingModel for an Ollama server. Its embed() method takes a list of strings and returns a list of float vectors.  ￼
    - LlmModelFactory, OllamaModel, OpenAIModel, and LlmGenerationConfig are part of the LLM abstraction layer; they let you configure different LLM backends while keeping a common interface.
	- tool and DescriptionMode come from the tool decorator API. The tool() helper turns a Python callable into a ServerTool with a schema inferred from its signature and docstring.  ￼
	- Agent and CallerInputMode form the agent runtime. An Agent is a component that can conduct multi-turn conversations and call tools.  ￼
	- StringProperty is a typed property used for input/output descriptors in Agents and Flows.
	- AgentExecutionStep, Flow, ControlFlowEdge, DataFlowEdge, InputMessageStep, OutputMessageStep are the Flow/Step primitives: Flow composes steps and edges into a conversational assistant.  ￼
	- AgentSpecExporter, AgentSpecLoader are helper classes that convert between WayFlow components and Agent Spec configurations (JSON/YAML).  ￼
- Logging is configured at INFO level, with HTTP-layer logs toned down to WARNING to avoid noisy output when calling OpenAI/Ollama.



## 4. Configuration block 
This block ties the script to specific external services but keeps them pluggable through WayFlow’s abstractions.

```python
# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DB_CFG = {
    "dsn": "localhost:1521/FREEPDB1",
    "user": "SH",
    "password": "********", #put [SCHEMA_PASSWORD]
}

PRODUCTS_TABLE = "PRODUCTS"
EMBEDDING_COLUMN = "EMBEDDINGS"
EMBEDDING_SOURCE_COLUMN = "PROD_DESC"

LLM_CONFIG = LlmGenerationConfig(temperature=0.2)

# Choose your LLM here; easy to swap if needed
# llm = OllamaModel(model_id="llama3.1", generation_config=None)
llm = OpenAIModel(
    model_id="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
    generation_config=LLM_CONFIG,
)

embedding_model = OllamaEmbeddingModel(
    base_url="http://localhost:11434",
    model_id="nomic-embed-text",
)
```

- `OPENAI_API_KEY` is read from the environment to support environment-driven configuration.  ￼
- `DB_CFG` sets the connection details for the SH sample schema: modify according your password for SH schema.
- `PRODUCTS_TABLE`, `EMBEDDING_COLUMN`, and `EMBEDDING_SOURCE_COLUMN` define how the Oracle schema is used:
    - `PRODUCTS` is an SH table with columns like PROD_ID, PROD_NAME, PROD_DESC, PROD_LIST_PRICE.
    - `EMBEDDINGS` will be a VECTOR column containing sentence embeddings of PROD_DESC.
- `LLM_CONFIG = LlmGenerationConfig(temperature=0.2)` sets the generation configuration (low temperature for more deterministic quotes).
- `llm`: The code chooses `OpenAIModel` as the LLM implementation, but the commented-out OllamaModel shows that you can easily swap to a self-hosted LLM while keeping the agent/flow logic identical.
- `embedding_model`: configured as an OllamaEmbeddingModel with base URL http://localhost:11434 and model `nomic-embed-text`.


## 5. Database helper functions

### 5.1 get_connection
Provides a single place for creating Oracle connections, using the `DB_CFG` above.
Autocommit is disabled to give the script transactional control when adding/dropping columns and updating embeddings.

```python
# ---------------------------------------------------------------------------
# DB HELPERS
# ---------------------------------------------------------------------------

def get_connection() -> oracledb.Connection:
    """Return a database connection with autocommit disabled."""
    conn = oracledb.connect(
        dsn=DB_CFG["dsn"],
        user=DB_CFG["user"],
        password=DB_CFG["password"],
    )
    conn.autocommit = False
    return conn
```

### 5.2 column_exists

It checks if the additional column it will be created to store the vector embeddings based on one or more fieds exists or not.

```python
def column_exists(cursor, table_name: str, column_name: str) -> bool:
    """
    Check if a column exists in ALL_TAB_COLUMNS.
    Oracle stores column names uppercase by default.
    """
    cursor.execute(
        """
        SELECT COUNT(*)
        FROM ALL_TAB_COLUMNS
        WHERE TABLE_NAME = :table_name
          AND COLUMN_NAME = :column_name
        """,
        table_name=table_name.upper(),
        column_name=column_name.upper(),
    )
    (count,) = cursor.fetchone()
    return count > 0
```

### 5.3 drop_column_if_exists
Uses `column_exists` to detect whether EMBEDDINGS should be dropped. This avoids errors on repeated runs and ensures a clean rebuild of embeddings on each demo run.

```python
def drop_column_if_exists(connection: oracledb.Connection, table_name: str, column_name: str) -> None:
    """Drop a column if it exists, otherwise do nothing."""
    with connection.cursor() as cursor:
        if not column_exists(cursor, table_name, column_name):
            logger.info(f"Column {column_name} does NOT exist in {table_name}. Nothing to drop.")
            return

        logger.info(f"Column {column_name} exists — dropping it...")
        alter_sql = f"ALTER TABLE {table_name} DROP COLUMN {column_name}"
        cursor.execute(alter_sql)
        connection.commit()
        logger.info(f"Column {column_name} successfully dropped from {table_name}.")
```

## 6. Embedding utilities 

### 6.1 embed_text_to_vector
The `embedding_model.embed([text])` returns a list of embeddings, one per input string. Here we take just the first (and only) embedding. The result is converted to an array('f', ...) of 32-bit floats, which is suitable for Oracle’s VECTOR type. This function bridges WayFlow’s embedding abstraction and the database-specific storage representation.

```python
def embed_text_to_vector(text: str) -> array.array:
    """Helper to get a float32 Oracle-ready vector from a single text string."""
    embeddings = embedding_model.embed([text])[0]
    return array.array("f", embeddings)
```

### 6.2 add_embeddings
This function encapsulates the offline indexing step:
1.	Ensure the EMBEDDINGS column exists (creating it as VECTOR(*,*) if needed).
2.	Fetch all rows and their description column.
3.	For each row, call `embed_text_to_vector` and update the embedding column.
While this logic is not WayFlow-specific, it’s a crucial part of the “connect assistants to your data” story in the WayFlow docs (embedding models + DB).  ￼
Embeddings are pre-computed so that the tool can perform fast similarity search via VECTOR_DISTANCE at query time.

**NOTICE:** for a production deployment, it should be used indexes of type `IVF` or `HNSW` on this vector column.

```python
def add_embeddings(table: str, fields: list[str]) -> None:
    """
    Add an embedding column to a table (if not present)
    and populate it for each row using the specified text column.
    """
    column_name = fields[0]

    with get_connection() as conn:
        with conn.cursor() as cursor:
            # 1) Ensure embedding column exists
            if column_exists(cursor, table, EMBEDDING_COLUMN):
                logger.info(f"Column {EMBEDDING_COLUMN} already exists in table {table}.")
            else:
                logger.info(f"Column {EMBEDDING_COLUMN} does NOT exist — adding it...")
                alter_sql = f"""
                    ALTER TABLE {table}
                    ADD {EMBEDDING_COLUMN} VECTOR(*,*)
                """
                cursor.execute(alter_sql)
                conn.commit()
                logger.info(f"Column {EMBEDDING_COLUMN} successfully added to {table}.")

            # 2) Select rows to embed (here: all rows)
            cursor.execute(
                f"""
                SELECT prod_id, {column_name}
                FROM {table}
                """
            )
            rows = cursor.fetchall()

            for prod_id, content in rows:
                if not content:
                    continue

                vector_data = embed_text_to_vector(content)
                cursor.execute(
                    f"""
                    UPDATE {table}
                    SET {EMBEDDING_COLUMN} = :emb
                    WHERE prod_id = :id
                    """,
                    emb=vector_data,
                    id=prod_id,
                )

            conn.commit()
            logger.info("All embeddings committed.")
```

## 7. Tool definitions

WayFlow tools are functions decorated with `@tool`. The tool() helper turns them into ServerTool objects with auto-inferred input/output schemas and descriptions.

### 7.1 get_product_by_description
This tool performs a **Similarity Search** on the Oracle DB. It will help the LLM Agent to find the actual products described in a free form text, providing as result the PROD_ID to continue in the quoting process.

- `@tool(description_mode=DescriptionMode.ONLY_DOCSTRING)`:
    - According to the docs, "only_docstring" means parameter descriptions are left empty and the function docstring is treated as the main description of the tool.  ￼
	- This is why the docstring is concise and user-oriented: “Looks for a product from a description and return the product_id.”
- The function itself:
	- Converts query into an embedding, q_vector.
	- Runs a similarity search against PRODUCTS.EMBEDDINGS using VECTOR_DISTANCE(..., COSINE) and selects the top-k (here k = 4).
	- Collects the results into a list of dicts.
	- Returns a human-readable string containing the best product’s PROD_ID and PROD_NAME.
- Note that the return type is a single str, which becomes the tool’s output descriptor via type inference. This aligns with the tool helper’s behavior: return types must be annotated so the tool schema can be inferred.

```python
# ---------------------------------------------------------------------------
# TOOLS
# ---------------------------------------------------------------------------

@tool(description_mode=DescriptionMode.ONLY_DOCSTRING)
def get_product_by_description(query: str) -> str:
    """Looks for a product from a description and return the product_id."""
    logger.info("TOOL:get_product_by_description called")

    q_vector = embed_text_to_vector(query)
    k = 4

    similarity_query = f"""
        SELECT
            d.prod_id,
            d.prod_name,
            d.prod_desc,
            (1 - VECTOR_DISTANCE(d.{EMBEDDING_COLUMN}, :q_embedding, COSINE)) AS cosine_similarity
        FROM {PRODUCTS_TABLE} d
        ORDER BY cosine_similarity DESC
        FETCH FIRST {k} ROWS ONLY
    """

    rows = []
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(similarity_query, q_embedding=q_vector)
            cols = [d[0] for d in cur.description]
            for rec in cur:
                rows.append(dict(zip(cols, rec)))

    if not rows:
        logger.info("TOOL:get_product_by_description found no matching products")
        return "No matching product found."

    best = rows[0]
    logger.info(f"TOOL:get_product_by_description\n {best}")
    return f"PROD_ID: {best['PROD_ID']} - PRODUCT_NAME: {best['PROD_NAME']}"
```

### 7.2 get_item_prices
This tool performs an **Exact Search** on the Oracle DB.

Created with @tool, again using `DescriptionMode.ONLY_DOCSTRING`. It takes a prod_id string, looks up PROD_LIST_PRICE from PRODUCTS, and returns the price as a string.
The design of these two tools matches WayFlow’s “Build Assistants with Tools” recommendation:
- Clear function names (get_product_by_description, get_item_prices).
- Concise docstrings for descriptions.
- Proper type annotations for auto-schema inference.  ￼

These tools form the toolbox of the agent defined in the next section.
```python
@tool(description_mode=DescriptionMode.ONLY_DOCSTRING)
def get_item_prices(prod_id: str) -> str:
    """Get the list price of a product from its prod_id."""
    logger.info("TOOL:get_item_prices called")

    sql = f"""
        SELECT /* get_item_prices */
               d.prod_list_price
        FROM {PRODUCTS_TABLE} d
        WHERE d.prod_id = :prod_id
        FETCH FIRST 1 ROWS ONLY
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {"prod_id": prod_id})
            row = cur.fetchone()
            price = row[0] if row else None

    logger.info(f"TOOL:get_item_prices\n {price}")
    return "" if price is None else str(price)
```

## 8. Agent configuration 

### 8.1 Output descriptor
- `StringProperty` describes a named, typed value in WayFlow. Properties are used as input/output descriptors for agents, steps, and flows.  ￼
- Here, `name="quote"` defines a single output field that will hold the final quote text.
- The default value is "", which is relevant for both the Agent and AgentExecutionStep output resolution behaviour.

```python
# ---------------------------------------------------------------------------
# AGENT SETUP
# ---------------------------------------------------------------------------

output = StringProperty(
    name="quote",
    description="quote to submit to the customer",
    default_value="",
)
```

### 8.2 sales_instructions prompt
This is the custom instruction for the agent. The custom_instruction is an additional system-level instructions appended to the prompt.  ￼

The prompt:
- Clearly states the role (“sales agent”) and task (prepare a quote).
- Enumerates **step-by-step reasoning** and explicit usage of the tools.
- Specifies a **strict output format** with an **example**, which helps the structured output step later.
- This aligns with the AgentExecutionStep guideline: custom instructions should focus on the task rather than generic “be helpful” phrasing.

```python
sales_instructions = """
You are a sales agent. Your task is to prepare a quote for a list of products the customer has requested. 

Follow these steps:
1. Find in the request the list of product items.
2. For each product you find, extract the requested quantity.
3. Use the tool get_product_by_description to find the prod_id for each product item.
4. Use the tool get_item_prices to find the unit price given the prod_id.
5. For each product, calculate the total price as: unit price × quantity.
6. Prepare the quote strictly in this format:

List of products:

Prod_ID: product ID
Prod_Name: product name
Prod_Desc: product description
Prod_List_Price: list price 
Quantity: number of items ordered
Tot_Price: total price for this product
------------------------------------------
Total_Order: sum of Tot_Price

Example:

Products
------------------------------------------
Prod_ID: 126
Prod_Name: Spiked Shoes
Prod_Desc: Spiked Cricket Shoes
Prod_List_Price: 28.99  
Quantity: 5
Tot_Price: 149.95

Prod_ID: 43
Prod_Name: Team shirt
Prod_Desc: Australian Cricket Team
Prod_List_Price: 44.99 
Quantity: 2
Tot_Price: 89.98
------------------------------------------
Total_Order: 239.93
"""
```

### 8.3 sales_agent
A WayFlow Agent is an LLM + set of tools + prompt configuration that can conduct a multi-step reasoning process.  ￼

Key arguments:
- `llm=llm`: the previously configured OpenAI LLM.
- `tools=[get_item_prices, get_product_by_description]`: the two Python functions we decorated with @tool. WayFlow automatically converts them into ServerTool instances when the agent is created.
- `custom_instruction=sales_instructions`: the sales-specific behaviour and formatting rules described above.
- `output_descriptors=[output]`:
    - when output_descriptors are specified, the agent is prompted to generate structured outputs corresponding to these descriptors.  ￼
    - Here that means the agent must produce a `quote` field.

```python
sales_agent = Agent(
    llm=llm,
    tools=[get_item_prices, get_product_by_description],
    custom_instruction=sales_instructions,
    output_descriptors=[output],
)
```

### 8.4 agent_step (AgentExecutionStep)
AgentExecutionStep is a Flow step that executes an Agent. If output_descriptors are given, it will ask the Agent to produce those outputs and only exit when they are available.  ￼
- `caller_input_mode=CallerInputMode.NEVER` means:
    - The Agent is not allowed to ask the user additional questions during this step.
	- The step won’t yield; it will run the Agent until the outputs are produced or a max iteration count is reached (not set).
- By passing `output_descriptors=[output]` both to the Agent and to this AgentExecutionStep, the flow enforces that:
	- The Agent must compute a quote field.
	- That field will be exposed as the step’s output named "quote".

```python
agent_step = AgentExecutionStep(
    name="agent_step",
    agent=writing_agent,
    caller_input_mode=CallerInputMode.NEVER,
    output_descriptors=[output],
)
```

## 9. Flow construction 

### 9.1 Quote template and message steps
`quote_template` is a simple Jinja template that takes a quote variable and surrounds it with a greeting and closing.

InputMessageStep and OutputMessageStep are standard steps provided by WayFlow for interacting with the conversation:
- `InputMessageStep` lets the user provide a message.
- `OutputMessageStep` posts a message to the conversation using a template.
Together with `AgentExecutionStep`, they form a 3-step flow.

```python
# ---------------------------------------------------------------------------
# FLOW SETUP
# ---------------------------------------------------------------------------

quote_template = """Dear customer,

Here is the quote requested for the list of products you asked for:

{{quote}}

Best
"""

user_step = InputMessageStep(name="user_step", message_template="")
send_quote_step = OutputMessageStep(
    name="send_quote_step",
    message_template=quote_template,
)
```

### 9.2 Flow with control and data edges
A Flow represents a graph of steps with control and data dependencies:
- `begin_step=user_step`: the flow starts at the user input step.
- control_flow_edges describe step ordering:
	1.	user_step → agent_step
	2.	agent_step → send_quote_step
	3.	send_quote_step → None (flow completion)
- `DataFlowEdge(agent_step, "quote", send_quote_step, "quote")` describes a data dependency:
	- The "quote" output produced by agent_step is wired into the "quote" input of send_quote_step.
	- In practice, this means the {{quote}} placeholder in quote_template will be filled with the Agent’s output.

This design follows the Flow API: control edges define what runs next, data edges define what values move between steps.  ￼


```python
flow = Flow(
    begin_step=user_step,
    control_flow_edges=[
        ControlFlowEdge(source_step=user_step, destination_step=agent_step),
        ControlFlowEdge(source_step=agent_step, destination_step=send_quote_step),
        ControlFlowEdge(source_step=send_quote_step, destination_step=None),
    ],
    data_flow_edges=[
        DataFlowEdge(agent_step, "quote", send_quote_step, "quote"),
    ],
)
```

## 10. Main execution and Agent Spec export

### 10.1 run_demo() function
1.	Rebuilding embeddings
    - Drops the EMBEDDINGS column if it exists, then calls add_embeddings to rebuild all embeddings.
    - This ensures the demo always runs with a freshly indexed product catalog.
2.	Running the flow
    - `conversation = flow.start_conversation()`:
	    - The Flow docs describe start_conversation as returning a Conversation object that encapsulates state and messages.  ￼
	- `conversation.execute()`:
	    - Executes the flow until it either yields or finishes. In this case:
	        - First execute() will drive the flow through the initial InputMessageStep (which may not actually add text yet).
	        - After append_user_message(...), the second execute() will let AgentExecutionStep run the agent with the user’s request and tools, and then send_quote_step will post the final quoting message.
	- assistant_reply = conversation.get_last_message() grabs the final OutputMessageStep message; the log prints the full quote.
3.	Agent Spec export / import
    - `AgentSpecExporter.to_json`converts the Flow into an Agent Spec JSON configuration.  ￼
	- This is useful for:
	    - Persisting the flow configuration.
	    - Moving the assistant between environments (e.g., from development to another runtime).
	    - Integrating with tools that understand Agent Spec.
    ```python
    serialized_flow = AgentSpecExporter().to_json(flow)
    ```
	- `AgentSpecLoader.load_json` takes the serialized config and reconstructs the WayFlow component (in this case, the Flow).  ￼
	- The tool_registry is required whenever the serialized config references tools:
	    - Keys are the tool names as they appear in the Agent Spec.
	    - Values are the actual callables used to re-instantiate the tools.
	- This round-trip (Flow → JSON → Flow) demonstrates that the configuration is portable and reproducible, which is a core design goal of WayFlow’s Agent Spec adapters.

    ```python
    tool_registry = {
    "get_product_by_description": get_product_by_description,
    "get_item_prices": get_item_prices,
    }
    _ = AgentSpecLoader(tool_registry=tool_registry).load_json(serialized_flow)
    ```

Full main execution code:

```python
# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------

def run_demo() -> None:
    # Rebuild embeddings column and populate it
    with get_connection() as conn:
        drop_column_if_exists(conn, PRODUCTS_TABLE, EMBEDDING_COLUMN)

    add_embeddings(PRODUCTS_TABLE, [EMBEDDING_SOURCE_COLUMN])

    # Execute flow / conversation
    conversation = flow.start_conversation()
    conversation.execute()

    conversation.append_user_message(
        "I want to buy two New Zealand Cricket t-shirt, and 3 Helmets"
    )
    conversation.execute()

    assistant_reply = conversation.get_last_message()
    logger.info(f"---\nAssistant >>> {assistant_reply.content}\n---")

    # Export / import flow spec
    serialized_flow = AgentSpecExporter().to_json(flow)

    tool_registry = {
        "get_product_by_description": get_product_by_description,
        "get_item_prices": get_item_prices,
    }
    _ = AgentSpecLoader(tool_registry=tool_registry).load_json(serialized_flow)


if __name__ == "__main__":
    run_demo()
```

## Execution samples:
This is an example of execution with the provided user message:
`I want to buy two New Zealand Cricket t-shirt, and 3 Helmets`

```bash
(.venv) cdebari@cdebari-mac mydemo % python3.11 quoting_sales_agent.py
wayflowcore.flow - INFO - No StartStep was given as part of the Flow, one will be added automatically.
__main__ - INFO - Column EMBEDDINGS exists — dropping it...
__main__ - INFO - Column EMBEDDINGS successfully dropped from PRODUCTS.
__main__ - INFO - Column EMBEDDINGS does NOT exist — adding it...
__main__ - INFO - Column EMBEDDINGS successfully added to PRODUCTS.
__main__ - INFO - All embeddings committed.
wayflowcore.executors._flowexecutor - INFO - user_step is yielding
__main__ - INFO - TOOL:get_product_by_description called
__main__ - INFO - TOOL:get_product_by_description
 {'PROD_ID': 42, 'PROD_NAME': 'Team shirt', 'PROD_DESC': 'New Zealand Cricket Team', 'COSINE_SIMILARITY': 0.7841360065081899}
__main__ - INFO - TOOL:get_product_by_description called
__main__ - INFO - TOOL:get_product_by_description
 {'PROD_ID': 116, 'PROD_NAME': 'Catchers Helmet', 'PROD_DESC': 'Catchers Helmet', 'COSINE_SIMILARITY': 0.790164319814014}
__main__ - INFO - TOOL:get_item_prices called
__main__ - INFO - TOOL:get_item_prices
 44.99
__main__ - INFO - TOOL:get_item_prices called
__main__ - INFO - TOOL:get_item_prices
 11.99
__main__ - INFO - ---
Assistant >>> Dear customer,

Here is the quote requested for the list of products you asked for:

Products
------------------------------------------
Prod_ID: 42
Prod_Name: Team shirt
Prod_Desc: New Zealand Cricket t-shirt
Prod_List_Price: 44.99  
Quantity: 2
Tot_Price: 89.98

Prod_ID: 116
Prod_Name: Catchers Helmet
Prod_Desc: Helmets
Prod_List_Price: 11.99 
Quantity: 3
Tot_Price: 35.97
------------------------------------------
Total_Order: 125.95

Best
---
```

## Closing Remarks

This is just a first simple example to start exploring the possibilities of WayFlow in conjunction with the Oracle DB 26ai vector store, in a hybrid setup: embeddings created with local models and an agent powered by OpenAI. Feel free to experiment with other combinations.
So, to all the multi-agent development experts out there, please forgive me: in the next installments, we’ll make the sales agent increasingly sophisticated. Promise!

---

## Disclaimer
*The views expressed in this paper are my own and do not necessarily reflect the views of Oracle.*