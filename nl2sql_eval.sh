source .venv2/bin/activate
export DF_API_KEY=$OPENAI_API_KEY
.venv2/bin/python text2sql_pipeline_gen.py
.venv2/bin/python howto_mcp_sqlcl_test.py 
.venv2/bin/python evaluate_sql.py
