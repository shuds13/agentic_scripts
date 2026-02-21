# LibEnsemble Documentation RAG

RAG (Retrieval Augmented Generation) system for querying libEnsemble documentation.

## Features

- **Local embeddings**: Uses HuggingFace sentence-transformers (no API keys needed for embeddings)
- **MCP Server**: Exposes RAG as tools for AI agents
- **Multiple query modes**:
  - Full documentation query with LLM synthesis (requires OpenAI API key)
  - Raw chunk retrieval (exact documentation text, **no API key needed**)
  - Generator-focused queries (for script generation, requires OpenAI API key)
  - Sim_specs-focused queries (requires OpenAI API key)

**Note**: The embedding/retrieval is always local (no API needed), but the "synthesis" modes (`query_libe_docs`, `query_generator_docs`, `query_sim_specs_docs`) use LlamaIndex's query engine which requires an LLM (OpenAI by default). If no API key is available, these modes will automatically fall back to raw chunk mode. Use `query_libe_docs_raw` directly if you want to avoid needing an API key.

## Setup

### 1. Install Dependencies

```bash
pip install llama-index llama-index-embeddings-huggingface mcp sentence-transformers
```

### 2. Build the Index

First, you need the libEnsemble documentation. Download it:

```bash
# Option 1: Clone the docs
wget -r -np -nH --cut-dirs=3 -R "index.html*" \
  https://libensemble.readthedocs.io/en/latest/

# Option 2: Use existing local docs
# Just point build_index.py to your docs directory
```

Then build the index:

```bash
python build_index.py
```

This creates a `rag_index/` directory with the vector embeddings.

### 3. Test It

```bash
# Query from command line
python rag_query.py "What generators are available?"

# Test the MCP server
python test_mcp_server.py
```

## Usage

### As MCP Server (Recommended)

The MCP server exposes 4 tools:

1. **query_libe_docs**: Full documentation query with synthesized answer
2. **query_libe_docs_raw**: Returns raw documentation chunks (no LLM)
3. **query_generator_docs**: Focused on generator documentation
4. **query_sim_specs_docs**: Focused on simulation specs

#### With LangChain Agent

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

# Connect to RAG MCP server
rag_server_params = StdioServerParameters(
    command="python",
    args=["rag/mcp_server.py"]
)

async with stdio_client(rag_server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        
        # Get RAG tools
        rag_tools = await session.list_tools()
        
        # Create agent with RAG tools
        agent = create_agent(llm, rag_langchain_tools)
        
        # Agent can now query docs
        result = await agent.ainvoke({
            "messages": [("user", "What options does APOSMM generator have?")]
        })
```

#### Environment Variable

Set the RAG MCP server location:

```bash
export RAG_MCP_SERVER=/path/to/script-creator/rag/mcp_server.py
```

### Direct Python Usage

```python
from rag_query import rag

# Simple query
answer = rag("How do I set bounds for optimization?")
print(answer)
```

## Use Cases

### Script Generation
Use `query_generator_docs` to get accurate information about generator options:

```
Q: "What are the main parameters for APOSMM generator?"
A: [Retrieves APOSMM documentation with configuration options]
```

### Debugging
Use `query_libe_docs` to understand error messages:

```
Q: "What does 'sim_specs must have in and out fields' mean?"
A: [Explains sim_specs structure and requirements]
```

### Interactive Script Review
Agent can reference docs while reviewing scripts:

```
Agent: "I see you're using persistent_aposmm. Let me check the docs for best practices..."
[Queries RAG]
Agent: "According to the documentation, you should set these options..."
```

## Architecture

```
rag/
├── mcp_server.py          # MCP server exposing RAG tools
├── build_index.py         # Build embeddings from docs
├── rag_query.py           # Simple CLI query interface
├── test_mcp_server.py     # Test suite
├── rag_index/             # Vector store (generated)
│   ├── docstore.json
│   ├── index_store.json
│   └── default__vector_store.json
└── README.md
```

## Future Enhancements

- [ ] Multiple specialized indexes (generators, allocation, etc.)
- [ ] Web API endpoint for website embedding
- [ ] Metadata filtering for context-specific retrieval
- [ ] Periodic index updates from latest docs
- [ ] Support for code examples extraction

## Integration with Agent Workflow

The RAG server is designed to work alongside the script generator MCP:

```python
# Multi-MCP agent setup
agent = create_agent(llm, [
    *script_generator_tools,  # From GENERATOR_MCP_SERVER
    *rag_tools,               # From RAG_MCP_SERVER
    *other_tools,             # Other Python MCP servers
])
```

Agent can now:
1. Query docs to understand requirements
2. Generate scripts with accurate parameters
3. Reference docs when fixing errors
4. Provide explanations based on official documentation
