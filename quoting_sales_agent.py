# -----------
# Setup
# -----------
# python env:
# python3.11 -m venv .venv --copies
# source .venv/bin/activate
# pip install "wayflowcore==25.4.1" oracleb
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
    return cursor.fetchone()[0] > 0


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


def embed_text_to_vector(text: str) -> array.array:
    """Helper to get a float32 Oracle-ready vector from a single text string."""
    embeddings = embedding_model.embed([text])[0]
    return array.array("f", embeddings)


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


# ---------------------------------------------------------------------------
# AGENT SETUP
# ---------------------------------------------------------------------------

output = StringProperty(
    name="quote",
    description="quote to submit to the customer",
    default_value="",
)

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

writing_agent = Agent(
    llm=llm,
    tools=[get_item_prices, get_product_by_description],
    custom_instruction=sales_instructions,
    output_descriptors=[output],
)

agent_step = AgentExecutionStep(
    name="agent_step",
    agent=writing_agent,
    caller_input_mode=CallerInputMode.NEVER,
    output_descriptors=[output],
)


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