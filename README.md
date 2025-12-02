# Wayflow Samples
Collection of examples on Wayflow agentic framework

# A Quoting Sales Agent with WayFlow and Oracle AI DB Similarity Search

This document explains the quoting_sales_agent.py script block by block, with particular focus on how it uses WayFlow’s concepts: LLM models, embedding models, tools, agents, and flows.
It integrates Oracle DB 26ai as vector store to support similarity search for products details finding. 
Refer to WayFlow 25.4.1 documentation for further information.

1. Configuration, Imports, and Environment

"""
Quoting Sales Agent with WayFlow Core using OpenAI compatible llm and ollama embeddings.

This script implements a simple assistant that:
- Reads product catalog data from an Oracle database
- Creates vector embeddings of product descriptions using an Ollama embedding model
- Exposes two tools:
  * get_product_by_description: semantic product search
  * get_item_prices: price lookup by product ID
- Wraps these tools in a WayFlow Agent
- Orchestrates everything in a simple Flow with an input message, the agent,
  and an output message.
"""

This top-level docstring describes the overall architecture:
	•	Oracle DB holds the product catalog.
	•	Ollama provides an embedding model for semantic search.
	•	WayFlow is used to:
	•	Define tools for semantic search and pricing.
	•	Wrap them in an Agent that can call tools.
	•	Orchestrate a simple Flow for the conversation.

WayFlow models flows as collections of steps and transitions between them. A [Flow](https://oracle.github.io/wayflow/25.4.1/core/api/flows.html) represents a conversational assistant with steps and edges that define the control and data flow.
  ￼
