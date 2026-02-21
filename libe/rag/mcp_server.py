#!/usr/bin/env python3
"""
MCP Server for LibEnsemble Documentation RAG
Provides tools for querying libEnsemble documentation using RAG.
"""

import sys
import asyncio
import json
from pathlib import Path
from typing import Optional
from mcp.server import Server
from mcp.types import Tool, TextContent
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Initialize embedding model (local, no API needed)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Don't set default LLM - we only use raw mode which doesn't need it
Settings.llm = None

# Load the RAG index (look in the same directory as this script)
rag_index_dir = Path(__file__).parent / "rag_index"
try:
    ctx = StorageContext.from_defaults(persist_dir=str(rag_index_dir))
    index = load_index_from_storage(ctx)
    print("RAG index loaded successfully", file=sys.stderr)
except Exception as e:
    print(f"Error loading RAG index: {e}", file=sys.stderr)
    print(f"Expected location: {rag_index_dir}", file=sys.stderr)
    print("Run build_index.py first to create the index", file=sys.stderr)
    sys.exit(1)


def query_docs_full(question: str, top_k: int = 5) -> str:
    """Query full libEnsemble documentation.
    
    Note: This mode requires an LLM (OpenAI by default) for synthesis.
    If no API key is available, use query_docs_raw instead.
    """
    try:
        query_engine = index.as_query_engine(
            similarity_top_k=top_k,
            response_mode="compact"
        )
        response = query_engine.query(question)
        return str(response)
    except Exception as e:
        error_msg = str(e)
        if "OpenAI" in error_msg or "API key" in error_msg:
            # Fallback to raw mode if no LLM available
            print("Warning: No LLM available, falling back to raw mode", file=sys.stderr)
            return query_docs_raw(question, top_k) + "\n\n[Note: Synthesized response unavailable - returned raw chunks instead. Set OPENAI_API_KEY for synthesis.]"
        raise


def query_docs_raw(question: str, top_k: int = 5) -> str:
    """Query docs and return raw source chunks without LLM synthesis."""
    query_engine = index.as_query_engine(
        similarity_top_k=top_k,
        response_mode="no_text"
    )
    response = query_engine.query(question)
    
    # Format source nodes
    results = []
    for i, node in enumerate(response.source_nodes, 1):
        results.append(f"--- Source {i} (score: {node.score:.3f}) ---")
        results.append(node.text)
        if hasattr(node, 'metadata') and node.metadata:
            results.append(f"Metadata: {node.metadata}")
        results.append("")
    
    return "\n".join(results)


def query_generator_docs(question: str, top_k: int = 3) -> str:
    """Query documentation focused on generators and their options.
    
    Note: This mode requires an LLM for synthesis.
    If no API key is available, use query_docs_raw instead.
    """
    # Enhance query to focus on generators
    enhanced_question = f"Regarding libEnsemble generators and their configuration: {question}"
    
    try:
        query_engine = index.as_query_engine(
            similarity_top_k=top_k,
            response_mode="compact"
        )
        response = query_engine.query(enhanced_question)
        return str(response)
    except Exception as e:
        error_msg = str(e)
        if "OpenAI" in error_msg or "API key" in error_msg:
            # Fallback to raw mode if no LLM available
            print("Warning: No LLM available for synthesis, returning raw chunks", file=sys.stderr)
            return query_docs_raw(enhanced_question, top_k) + "\n\n[Note: Synthesized response unavailable - returned raw chunks instead. Set OPENAI_API_KEY for synthesis.]"
        raise


def query_sim_specs_docs(question: str, top_k: int = 3) -> str:
    """Query documentation focused on simulation specifications.
    
    Note: This mode requires an LLM for synthesis.
    If no API key is available, use query_docs_raw instead.
    """
    enhanced_question = f"Regarding libEnsemble simulation specifications (sim_specs): {question}"
    
    try:
        query_engine = index.as_query_engine(
            similarity_top_k=top_k,
            response_mode="compact"
        )
        response = query_engine.query(enhanced_question)
        return str(response)
    except Exception as e:
        error_msg = str(e)
        if "OpenAI" in error_msg or "API key" in error_msg:
            # Fallback to raw mode if no LLM available
            print("Warning: No LLM available for synthesis, returning raw chunks", file=sys.stderr)
            return query_docs_raw(enhanced_question, top_k) + "\n\n[Note: Synthesized response unavailable - returned raw chunks instead. Set OPENAI_API_KEY for synthesis.]"
        raise


# Create MCP server
app = Server("libe-docs-rag")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available RAG query tools."""
    return [
        Tool(
            name="query_libe_docs_raw",
            description="Query libEnsemble documentation and return raw text chunks. No LLM needed - returns exact documentation text.",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask about libEnsemble documentation"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of relevant chunks to retrieve (default: 5)",
                        "default": 5
                    }
                },
                "required": ["question"]
            }
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    question = arguments.get("question", "")
    top_k = arguments.get("top_k", 5)
    
    try:
        if name == "query_libe_docs_raw":
            result = query_docs_raw(question, top_k)
        else:
            result = f"Unknown tool: {name}"
        
        return [TextContent(type="text", text=result)]
    except Exception as e:
        error_msg = f"Error querying documentation: {str(e)}"
        print(error_msg, file=sys.stderr)
        return [TextContent(type="text", text=error_msg)]


async def main():
    """Run the MCP server."""
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
