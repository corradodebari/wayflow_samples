# A NL2SQL Agent for Oracle DB in WayFlow with the SQLcl MCP server

<div align="center">
<img align="center" src="https://raw.githubusercontent.com/corradodebari/wayflow_samples/main/images/cover2.png" width="600px">
</div>

## 1. High-level overview
In this demo, we introduce the use of the MCP tool call through the **SQLcl MCP Server**. This approach allows you to decouple tool logic from the database entities you want to query in **Oracle DB**. Instead of defining specific tool functions for each query, you simply describe what you are looking for in the prompt, and the LLM automatically translates your request into a SQL query using the tools exposed by the SQLcl MCP Server.

Unlike the previous WayFlow example—where a dedicated tool function had to be created for every single query—this approach enables you to run multiple different queries using the same tool, simply by changing the prompt.

However, while this method provides greater flexibility, its reliability strongly depends on:
- The LLM model being used
- The model’s ability to correctly interpret natural language and translate it into SQL
- The quality, clarity, and self-descriptiveness of the database schema

For these reasons, this approach is particularly well suited for prototyping, where it helps avoid creating a large number of specialized tools just to retrieve entities from the database. In a later phase, once the use cases are stabilized, you can replace the dynamic SQL generation with fixed queries and scale down to a smaller, more efficient LLM, reducing dependency on SQL-to-text translation accuracy.

The sample code provided allows you to test different models—both public and private—using **OpenAI**, **Ollama**, and **OCI GenAI**. Feel free to experiment and compare their behaviors.

In the next episode of this WayFlow walkthrough with Oracle DB, we will bring everything together into an enhanced version of the Quoting Sales Agent.

**Takeaways:**
- **WayFlow 25.4.1:** full documentation: **[here](https://oracle.github.io/wayflow/25.4.1/core/api/agentspec.html)**
- **SQLcl MCP Server:** **[User's Guide](https://docs.oracle.com/en/database/oracle/sql-developer-command-line/25.2/sqcug/sqlcl-mcp-server.html)**
- **Python Script** in the GitHub repo: **<a href="https://github.com/corradodebari/wayflow_samples/blob/main/howto_mcp_sqlcl">howto_mcp_sqlcl.py</a>**


## 2. Pre-requisites to Prepare and Run the Demo

To prepare and run the demo, ensure you have the following:
- A Python 3.11 virtual environment with the `oci` library installed
- The latest version of WayFlow, downloaded from the main branch of the repository:[here](https://github.com/oracle/wayflow)
- An Ollama server running with a model such as `gpt-oss:20b` or higher
- A properly configured OCI environment, with the config file set up
- Setup instructions are available [here](https://docs.oracle.com/en-us/iaas/Content/API/SDKDocs/cliinstall.htm)
- The code must be updated with:
- The path to your config file
- The required parameters obtained from the OCI Console
- An `OPENAI_API_KEY` for using OpenAI-based LLMs
- A local Oracle Database (DB Free / “DB26ai”) with the SH sample schema installed

```bash
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
```

## 3. Imports
To exploit an MCP server, the `wayflowcore.mcp` provides connection for `Stdio`, `SSE` and `StreamableHTTP` MCP protocols. For SQLcl we are going to use the first one, Stdio.

```python
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
```

## 4. Dictionaries and configuration block 
This block helps to set basic SQLcl params, and switch quickly among a public and private LLM providers.

```python
mcp_command = "/Users/cdebari/sqlcl25/sqlcl/bin/sql"  # change to your own command
connection = "DB23ai_sh"

#model_id = {"OPENAI": "gpt-5-mini", "OLLAMA": "gpt-oss:20b","OCIGENAI":"openai.gpt-oss-120b" }
model_id = {"OPENAI": "gpt-5-mini", "OLLAMA": "gpt-oss:20b","OCIGENAI":"xai.grok-code-fast-1" }

model_temp = {"OPENAI": 1, "OLLAMA": 0.2, "OCIGENAI": 0.1}

llm_type = "OCIGENAI"
```

- `mcp_command`: since Stdio requires the execution of a local command, set here the absolute path to your SQLcl command
- `connection`: set which connection you have already set with `CONNMGR`, and that will be used by the tools. In the example, it's set to the SH schema sample, as in the previous post.
- `model_id`: dictionary with LLMs model names for each provider.
- `model_temp`: dictionary with temperatures required. For `gpt-5-mini` is allowed only 1.
- `llm_type`: use this variable to change the LLM model on which you want to rely on.

## 5. Agent prompt for NL2SQL via SQLcl tool
This is one of the most critical parts of the implementation, as it defines how the user’s request is translated into an Oracle SQL query to retrieve the information needed to answer the question. The SQLcl tool is restricted to executing SQL SELECT statements only, so the correctness of the translation depends entirely on how well this prompt is defined.

To generate accurate SQL, the tool must be aware of the database schema, which is therefore included directly in the prompt. The version shown here is a proposed baseline: feel free to refine it if you notice inaccuracies when handling more complex queries. You can also modify the `### Output Format` section to structure the results differently and make them easier to propagate to downstream agents.

This part will be revisited and refined further when we integrate it into the multi-agent version of the solution.

```python 
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
```

## 6. LLM connection definition block

Here we define the three alternatives for the LLMs that can be used. We focus on OCI, not mentioned in the previous post.

This part define the `OCIGENAI_ENDPOINT`: change it for a different OCI region. Here it is also defined how to finde the authorization profile with fingerprint, key_file, tenancy, etc., in the example "~/.oci/config". You can have different profiles: in the example I'm using the DEFAULT one.

```python
OCIGENAI_ENDPOINT = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"

client_config = OCIClientConfigWithApiKey(
    service_endpoint=OCIGENAI_ENDPOINT,
    auth_profile="DEFAULT",
    _auth_file_location="~/.oci/config"
)
```

You need to get the Compartment OCID to complete the set of credentials needed to access the OCI GenAI endpoint:

```python
    llm = OCIGenAIModel(
        model_id=model_id[llm_type],
        client_config=client_config,
        compartment_id="ocid1.compartment.oc1..aaaaaaa.......") 
```

## 7. MCP Tool & Agent definition

Here is defined the way to call the SQLcl via Stdio protocol:

```python
env = {"MODEL": model_id[llm_type], "MCP_CLIENT": "wayflow"}

enable_mcp_without_auth()
mcp_client = StdioTransport(command=mcp_command, args=["-mcp"], env=env)

mcp_toolbox = MCPToolBox(
    client_transport=mcp_client
)  # Expose all tools available @MCP Server
```
- `env`: include in the env at running phase info about Model and MCP client info, that tools requires. 
- `enable_mcp_without_auth()`: in this case no authentication is required
- `mcp_client = StdioTransport(command=mcp_command, args=["-mcp"], env=env)`: to the transport we pass the -mcp that let the server to start
- `MCPToolBox`: provides to the LLM the access to all tools included, i.e.:
    - list-connections
    - connect
    - disconnect
    - run-sqlcl
    - run-sql
    It could be registered with `MCPTool`, one-by-one, only `connect` and `run-sql` we need in this case. We already know which connection to be used and only SELECT command we will need.

Finally, the Agent definition, where we set the mcp_toolbox as MCP server to call:

```python
assistant = Agent(
    llm=llm,
    tools=[mcp_toolbox],
    custom_instruction=query_instructions,
    max_iterations=30,
    can_finish_conversation=True
)
```

## Execution sample:
We can run through one of the example messages in USER_QUERY[] to test the SQLcl tool, with:
```python
conversation = assistant.start_conversation()
conversation.append_user_message(USER_QUERY[1])
```
For example, asking *"for the customer Emmerson living in Middelburg, provide me the total of sales"* we need to JOIN the table `CUSTOMERS` with `SALES` by `CUST_ID`, doing a Sum() of `QUANTITY_SOLD` x `AMOUNT_SOLD` for the `CUST_ID` found.

This a log trace example:

```bash
(.venv2) cdebari@cdebari-mac wayflow_samples % .venv2/bin/python howto_mcp_sqlcl.py
INFO:wayflowcore.models.ocigenaimodel:OCI model with model_id=xai.grok-code-fast-1 was resolved to use serving_mode=ServingMode.ON_DEMAND

User   >>> 
for the customer Emmerson living in Middelburg, provide me the total of sales 

INFO:wayflowcore.mcp._session_persistence:BlockingPortal loop started
---------- MCP SERVER STARTUP ----------
MCP Server started successfully on Tue Dec 09 18:42:03 CET 2025
Press Ctrl+C to stop the server
----------------------------------------
dic 09, 2025 6:42:03 PM io.modelcontextprotocol.server.McpAsyncServer$AsyncServerImpl lambda$asyncInitializeRequestHandler$5
INFO: Client initialize request - Protocol: 2025-06-18, Capabilities: ClientCapabilities[experimental=null, roots=null, sampling=null], Info: Implementation[name=mcp, version=0.1.0]
dic 09, 2025 6:42:03 PM io.modelcontextprotocol.server.McpAsyncServer$AsyncServerImpl lambda$asyncInitializeRequestHandler$5
WARNING: Client requested unsupported protocol version: 2025-06-18, so the server will sugggest the 2024-11-05 version instead

--- CONVERSATION STATE ---
[user] 
for the customer Emmerson living in Middelburg, provide me the total of sales 

[assistant] 
[assistant] ### DATABASE CONNECTION ESTABLISHED ###
Successfully connected to: **DB23ai_sh**
### DATABASE ENVIRONMENT CONTEXT ###
**1. Oracle Database Version:** 
23.0.0.0.0**2. Access Mode:** 
The Oracle database is currently in 'null' mode**3. NLS Parameters:** 
The current Oracle database uses the following NLS configuration:
Database character set:{"results":[{"columns":[{"name":"PARAMETER","type":"VARCHAR2"},{"name":"VALUE","type":"VARCHAR2"}],"items":
[
{"parameter":"NLS_CHARACTERSET","value":"AL32UTF8"}
]}]}

 Full NLS parameters: 
{"results":[{"columns":[{"name":"PARAMETER","type":"VARCHAR2"},{"name":"VALUE","type":"VARCHAR2"}],"items":
[
{"parameter":"NLS_RDBMS_VERSION","value":"23.0.0.0.0"}
,{"parameter":"NLS_NCHAR_CONV_EXCP","value":"FALSE"}
,{"parameter":"NLS_LENGTH_SEMANTICS","value":"BYTE"}
,{"parameter":"NLS_COMP","value":"BINARY"}
,{"parameter":"NLS_DUAL_CURRENCY","value":"$"}
,{"parameter":"NLS_TIMESTAMP_TZ_FORMAT","value":"DD-MON-RR HH.MI.SSXFF AM TZR"}
,{"parameter":"NLS_TIME_TZ_FORMAT","value":"HH.MI.SSXFF AM TZR"}
,{"parameter":"NLS_TIMESTAMP_FORMAT","value":"DD-MON-RR HH.MI.SSXFF AM"}
,{"parameter":"NLS_TIME_FORMAT","value":"HH.MI.SSXFF AM"}
,{"parameter":"NLS_SORT","value":"BINARY"}
,{"parameter":"NLS_DATE_LANGUAGE","value":"AMERICAN"}
,{"parameter":"NLS_DATE_FORMAT","value":"DD-MON-RR"}
,{"parameter":"NLS_CALENDAR","value":"GREGORIAN"}
,{"parameter":"NLS_NUMERIC_CHARACTERS","value":".,"}
,{"parameter":"NLS_NCHAR_CHARACTERSET","value":"AL16UTF16"}
,{"parameter":"NLS_CHARACTERSET","value":"AL32UTF8"}
,{"parameter":"NLS_ISO_CURRENCY","value":"AMERICA"}
,{"parameter":"NLS_CURRENCY","value":"$"}
,{"parameter":"NLS_TERRITORY","value":"AMERICA"}
,{"parameter":"NLS_LANGUAGE","value":"AMERICAN"}
]}]}
20 righe selezionate. 


 This is important for correctly interpreting data formats, encodings, and locale-specific behavior.
Here’s the Oracle database context you need: version, open mode, and NLS parameters. Use this information to run the following queries accurately and ensure they align with the database’s current environment and settings.
[assistant] 
[assistant] "TABLE_NAME"
"COUNTRIES"
"CUSTOMERS"
"PROMOTIONS"
"PRODUCTS"
"TIMES"
"CHANNELS"
"SALES"
"COSTS"
"SUPPLEMENTARY_DEMOGRAPHICS"
"CAL_MONTH_SALES_MV"
"FWEEK_PSCAT_SALES_MV"
"DR$SUP_TEXT_IDX$I"
"DR$SUP_TEXT_IDX$K"
"DR$SUP_TEXT_IDX$N"
"DR$SUP_TEXT_IDX$U"
"DR$SUP_TEXT_IDX$Q"
"DR$SUP_TEXT_IDX$C"
"DR$SUP_TEXT_IDX$B"
"DBTOOLS$MCP_LOG"

19 righe selezionate. 


[assistant] 
[assistant] Errore con inizio alla riga : 2 nel comando -
SELECT column_name, comments FROM user_tab_columns WHERE table_name = 'CUSTOMERS'
Errore alla riga del comando : 2 colonna : 21
Report errori -
Errore SQL: ORA-00904: "COMMENTS": identificativo non valido

https://docs.oracle.com/error-help/db/ora-00904/00904. 00000 -  "%s: invalid identifier"
*Cause:    The identifier or column name entered was invalid.
*Action:   Ensure the following

More Details :
https://docs.oracle.com/error-help/db/ora-00904/

[assistant] 
[assistant] "COLUMN_NAME"
"CUST_ID"
"CUST_FIRST_NAME"
"CUST_LAST_NAME"
"CUST_GENDER"
"CUST_YEAR_OF_BIRTH"
"CUST_MARITAL_STATUS"
"CUST_STREET_ADDRESS"
"CUST_POSTAL_CODE"
"CUST_CITY"
"CUST_CITY_ID"
"CUST_STATE_PROVINCE"
"CUST_STATE_PROVINCE_ID"
"COUNTRY_ID"
"CUST_MAIN_PHONE_NUMBER"
"CUST_INCOME_LEVEL"
"CUST_CREDIT_LIMIT"
"CUST_EMAIL"
"CUST_TOTAL"
"CUST_TOTAL_ID"
"CUST_SRC_ID"
"CUST_EFF_FROM"
"CUST_EFF_TO"
"CUST_VALID"

23 righe selezionate. 


[assistant] 
[assistant] "COLUMN_NAME"
"PROD_ID"
"CUST_ID"
"TIME_ID"
"CHANNEL_ID"
"PROMO_ID"
"QUANTITY_SOLD"
"AMOUNT_SOLD"

7 righe selezionate. 


[assistant] 
[assistant] "TOTAL_SALES"
14041,43


[assistant] TOTAL_SALES
14041,43

Assistant >>> TOTAL_SALES
14041,43
```
As you can see, a wrong query is executed, but finally the agent retrieves the write value. The list of tables is also get to start a reasoning on which table define the user query.


## Closing Remarks

Rather than providing a single example that combines all three ways of using the database as a source through tools — as shown in the cover image — I chose, at this stage, to focus exclusively on the MCP and SQLcl interaction, using OCI GenAI as the LLM provider.

In the next step of this WayFlow walkthrough, we will introduce a multi-agent demo that exploits the Oracle Database in its converged role for data storage and access. 

---

## Disclaimer
*The views expressed in this paper are my own and do not necessarily reflect the views of Oracle.*