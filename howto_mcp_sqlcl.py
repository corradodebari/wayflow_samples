# ----------------------------------------------------------------------
# WayFlow Code Example - How to connect Oracle SQLCL as a Tool
# ----------------------------------------------------------------------

# How to use:
# Create a new Python virtual environment and install the latest WayFlow version.
# ```bash
# python3.11 -m venv .venv2 --copies
# source .venv2/bin/activate
# pip install -e <GitHub/wayflow/wayflowcore absolute path> oci
# ```
#
# ollama model:
# ollama pull nomic-embed-text 
# ollama pull gpt-oss:20b
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
# You can now run the script
# 1. As a Python file:
# ```bash
# .venv2/bin/python howto_mcp_sqlcl.py
# ```

from wayflowcore.executors.executionstatus import (
    FinishedStatus,
    UserMessageRequestStatus,
)
from wayflowcore.agent import Agent
from wayflowcore.mcp import (
    MCPTool,
    MCPToolBox,
    SSETransport,
    enable_mcp_without_auth,
    StdioTransport,
)
from wayflowcore.flow import Flow
from wayflowcore.steps import ToolExecutionStep
import os
import oci
from wayflowcore.models.openaimodel import OpenAIModel
from wayflowcore.models.llmgenerationconfig import LlmGenerationConfig
from wayflowcore.models import LlmModelFactory, OllamaModel

from wayflowcore.models.ocigenaimodel import OCIGenAIModel
from wayflowcore.models.ociclientconfig import OCIClientConfigWithApiKey



import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
# Disable httpx and httpcore noisy logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


mcp_command = "/Users/cdebari/sqlcl25/sqlcl/bin/sql"  # change to your own command

#model_id = {"OPENAI": "gpt-5-mini", "OLLAMA": "gpt-oss:20b","OCIGENAI":"openai.gpt-oss-120b" }
model_id = {"OPENAI": "gpt-5-mini", "OLLAMA": "gpt-oss:20b","OCIGENAI":"xai.grok-code-fast-1" }

model_temp = {"OPENAI": 1, "OLLAMA": 0.2, "OCIGENAI": 0.1}

connection = "DB23ai_sh"
llm_type = "OCIGENAI"


# --- High-level instructions for the agent (same logic as before) ---

query_instructions = f"""
You are an expert Oracle Text-to-SQL assistant.

Your task is to:
1. Translate the user's natural language request into a valid **Oracle SQL query**.
2. Ensure all referenced tables and columns exist before executing the final query.

### Schema Validation Rules
Before writing the final query, you MUST inspect the schema using:
- user_tables
- user_tab_columns

Use these views to:
- Verify that each referenced table exists
- Verify that each referenced column exists in its respective table
- Use column comments when available to disambiguate meaning

If a required table or column does NOT exist, you must:
- Either infer the most likely correct alternative
- Or ask the user for clarification (do NOT guess silently)

### Tool Execution Order (MANDATORY)
1. First, connect using the `connect` MCP tool with:
   - connection_name: {connection}
   - mcp_client: "wayflow"
   - model: "{model_id[llm_type]}"

2. Then execute ALL validation and final queries using the `run-sql` MCP tool with:
   - sql: <generated SQL>
   - mcp_client: "wayflow"
   - model: "{model_id[llm_type]}"

### SQL Generation Rules
- Use Oracle-compatible SQL only
- Do NOT reference tables or columns unless verified via schema queries
- Prefer explicit joins over implicit joins
- Avoid SELECT *
- Use meaningful column aliases
- Optimize for correctness over performance unless the user explicitly asks otherwise

### Output Format
- Return ONLY the final query results
- The output must be in **CSV format**
- Do NOT include explanations, comments, or debug output—CSV only

You are not allowed to skip schema validation.
You are not allowed to return raw SQL without execution.
"""

# --- User query examples ---
USER_QUERY = [
    """
for the Customer with ID 6 provide me the total of sales 
""",
    """
for the customer Emmerson living in Middelburg, provide me the total of sales 
""",
]

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_CONFIG = LlmGenerationConfig(temperature=model_temp[llm_type])

OCIGENAI_ENDPOINT = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"

client_config = OCIClientConfigWithApiKey(
    service_endpoint=OCIGENAI_ENDPOINT,
    auth_profile="DEFAULT",
    _auth_file_location="~/.oci/config"
)

if llm_type == "OPENAI":
    llm = OpenAIModel(
        model_id=model_id[llm_type],
        api_key=OPENAI_API_KEY,
        generation_config=LLM_CONFIG,
    )
elif llm_type == "OLLAMA":
    llm = OllamaModel(model_id=model_id[llm_type], generation_config=LLM_CONFIG)
else:
    llm = OCIGenAIModel(
        model_id=model_id[llm_type],
        client_config=client_config,
        compartment_id="ocid1.compartment.oc1..aaaaaaaaonknae....") #Set your compartment_id OCID

env = {"MODEL": model_id[llm_type], "MCP_CLIENT": "wayflow"}

enable_mcp_without_auth()
mcp_client = StdioTransport(command=mcp_command, args=["-mcp"], env=env)

mcp_toolbox = MCPToolBox(
    client_transport=mcp_client
)  # Expose all tools available @MCP Server

# Use the tool: run-sql to execute the query requested by user and translated in Oracle SQL.
assistant = Agent(
    llm=llm,
    tools=[mcp_toolbox],
    custom_instruction=query_instructions,
    max_iterations=30,
    can_finish_conversation=True
)

conversation = assistant.start_conversation()
conversation.append_user_message(USER_QUERY[1])
print("\nUser   >>>", USER_QUERY[1])

while True:
    status = conversation.execute()
    print("\n--- CONVERSATION STATE ---")
    for msg in conversation.get_messages():
        print(f"[{msg.role}] {msg.content}")

    reply = conversation.get_last_message()
    if reply is not None:
        print("\nAssistant >>>", reply.content)

    if isinstance(status, FinishedStatus):
        break
    if isinstance(status, UserMessageRequestStatus):
        # Non-interactive: stop here, or send an empty follow-up
        break
