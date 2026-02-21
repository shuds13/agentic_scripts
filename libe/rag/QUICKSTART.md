# Quick Start Guide

Get the RAG system running in 5 minutes.

## 1. Install Dependencies

```bash
cd rag/
pip install -r requirements.txt
```

## 2. Build the Index

If you already have docs downloaded:
```bash
python build_index.py libensemble.readthedocs.io/en/latest
```

If not, download first:
```bash
wget -r -np -nH --cut-dirs=3 -R 'index.html*' \
  https://libensemble.readthedocs.io/en/latest/
python build_index.py
```

This creates `rag_index/` with embeddings (~1-2 minutes on first run).

## 3. Test It

### Command Line Query
```bash
# Raw mode (no API key needed)
python rag_query.py --raw "What generators are available in libEnsemble?"

# Synthesis modes (require OPENAI_API_KEY)
python rag_query.py "What generators are available in libEnsemble?"
python rag_query.py --generator "What are APOSMM options?"
```

**Note**: The `--raw` mode returns exact documentation chunks without LLM synthesis and doesn't need an API key. Other modes use LlamaIndex's synthesis which requires OpenAI by default.

### Test MCP Server
```bash
python test_mcp_server.py
```

Should see:
```
✓ Direct RAG queries working
✓ MCP server integration working
✓ Direct MCP tool call works!
⚠ Skipping agent test (set OPENAI_API_KEY to test agent integration)
```

The core RAG functionality works without an API key. Agent integration requires `OPENAI_API_KEY` for the LLM that orchestrates tool calls.

## 4. Use with Your Agent

### Set Environment Variable
```bash
export RAG_MCP_SERVER=/path/to/script-creator/rag/mcp_server.py
```

### Run Example Agent (Uses Both MCP Servers)
```bash
python example_agent_with_rag.py
```

This demonstrates an agent that:
1. Queries documentation for generator info
2. Uses that info to generate accurate scripts

### Integrate into Your Own Agent

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Connect to RAG MCP
rag_params = StdioServerParameters(
    command="python",
    args=["/path/to/rag/mcp_server.py"]
)

async with stdio_client(rag_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        
        # Get RAG tools and add to your agent
        rag_tools = await session.list_tools()
        # ... create LangChain tools and add to agent
```

## Available Tools

The MCP server exposes 4 tools:

| Tool | Description | Use When |
|------|-------------|----------|
| `query_libe_docs` | Full docs with synthesis | General questions |
| `query_libe_docs_raw` | Raw chunks, no LLM | Want exact text |
| `query_generator_docs` | Generator-focused | Script generation |
| `query_sim_specs_docs` | Sim_specs-focused | Understanding sim_specs |

## Troubleshooting

**"RAG index not found"**
→ Run `python build_index.py` first

**"No documents found"**
→ Check docs directory exists and has content

**"Import error: mcp"**
→ `pip install mcp`

**"Import error: sentence_transformers"**
→ `pip install sentence-transformers`

## Next Steps

- [ ] Try the interactive agent example
- [ ] Integrate RAG into your script generation workflow
- [ ] Query docs during error fixing phase
- [ ] Experiment with different query modes

## Files Created

```
rag/
├── mcp_server.py              # MCP server (main interface)
├── build_index.py             # Build/rebuild index
├── rag_query.py               # CLI query tool
├── test_mcp_server.py         # Test suite
├── example_agent_with_rag.py  # Example multi-MCP agent
├── requirements.txt           # Dependencies
├── README.md                  # Full documentation
├── QUICKSTART.md             # This file
└── rag_index/                # Vector store (generated)
```
