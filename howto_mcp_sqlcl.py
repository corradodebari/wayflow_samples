# --------------------------------------------------------------------
# WayFlow Code Example - How to connect MCP tools SQLCL to Assistants
# --------------------------------------------------------------------

# How to use:
# Create a new Python virtual environment and install the latest WayFlow version.
# ```bash
# python3.11 -m venv .venv2 --copies
# source .venv2/bin/activate
# pip install -e <GitHub/wayflow/wayflowcore absolute path>
# ```
#
# ollama model:
# ollama pull nomic-embed-text 
# ollama pull gpt-oss:20b
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
from wayflowcore.models.openaimodel import OpenAIModel
from wayflowcore.models.llmgenerationconfig import LlmGenerationConfig
from wayflowcore.models import LlmModelFactory, OllamaModel


import logging

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
# Disable httpx and httpcore noisy logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


mcp_command = "/Users/cdebari/sqlcl25/sqlcl/bin/sql"  # change to your own command

model_id = {"OPENAI": "gpt-5-mini", "OLLAMA": "gpt-oss:20b"}
model_temp = {"OPENAI": 1, "OLLAMA": 0.2}

connection = "DB23ai_sh"
llm_type = "OLLAMA"


# --- High-level instructions for the agent (same logic as before) ---

query_instructions = f"""
You are a Database user assistant.
Translate the user request in an Oracle SQL query.
For the translation use the tool `run-sql` go get info about tables and views in the schema available, including comments, with query on tables:
- user_tab_columns
- user_tables
Check if each field exist in the table before submit the query.

- Use the `connect` MCP tool first to connect to the saved connection requested by the user
  (parameter: connection_name: {connection} , plus mcp_client="wayflow", model="{model_id[llm_type]}").
- Then use the `run-sql` MCP tool to execute the SQL query (parameter: sql, plus same mcp_client/model).
Return the query results to the user as a csv format.
"""

# --- Example user query ---
USER_QUERY = [
    """
for the Customer with ID 6 provide me the total of sales done
""",
    """
for the customer Emmerson living in Middelburg, provide me the total of sales done
""",
]

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_CONFIG = LlmGenerationConfig(temperature=model_temp[llm_type])

if llm_type == "OPENAI":
    llm = OpenAIModel(
        model_id=model_id[llm_type],
        api_key=OPENAI_API_KEY,
        generation_config=LLM_CONFIG,
    )
else:
    llm = OllamaModel(model_id=model_id[llm_type], generation_config=LLM_CONFIG)

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
    max_iterations=20,
    can_finish_conversation=True,
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
