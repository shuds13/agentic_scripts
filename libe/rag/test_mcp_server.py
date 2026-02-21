#!/usr/bin/env python3
"""
Test script for the RAG MCP server.
Tests both direct function calls and MCP integration with LangChain.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Test direct RAG functions
print("=" * 60)
print("Testing Direct RAG Functions")
print("=" * 60)

from mcp_server import query_docs_full, query_generator_docs, query_docs_raw

# Test 1: General documentation query
print("\n1. Testing general docs query...")
question = "What is libEnsemble?"
result = query_docs_full(question, top_k=3)
print(f"Q: {question}")
print(f"A: {result[:300]}...")

# Test 2: Generator-specific query
print("\n2. Testing generator docs query...")
question = "What are the main generator options for APOSMM?"
result = query_generator_docs(question, top_k=3)
print(f"Q: {question}")
print(f"A: {result[:300]}...")

# Test 3: Raw chunks query
print("\n3. Testing raw chunks query...")
question = "How do I set bounds for optimization?"
result = query_docs_raw(question, top_k=2)
print(f"Q: {question}")
print(f"A: {result[:400]}...")

print("\n" + "=" * 60)
print("Testing MCP Server Integration")
print("=" * 60)

async def test_mcp_integration():
    """Test MCP server with LangChain agent."""
    try:
        from langchain_openai import ChatOpenAI
        from langchain.agents import create_agent
        from langchain_core.tools import StructuredTool
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        import os
        
        # Get path to MCP server
        mcp_server_path = Path(__file__).parent / "mcp_server.py"
        server_params = StdioServerParameters(
            command="python",
            args=[str(mcp_server_path)]
        )
        
        print(f"\nConnecting to MCP server: {mcp_server_path}")
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # List available tools
                tools = await session.list_tools()
                print(f"\nAvailable tools: {[t.name for t in tools.tools]}")
                
                # Test raw query (doesn't need LLM)
                print("\n4a. Testing MCP tool call (query_libe_docs_raw - no LLM needed)...")
                question = "What generator functions are available in libEnsemble?"
                print(f"Q: {question}")
                
                result = await session.call_tool("query_libe_docs_raw", {"question": question, "top_k": 2})
                response_text = result.content[0].text if result.content else "No response"
                print(f"A: {response_text[:300]}...")
                print("✓ Direct MCP tool call works!")
                
                # Test with agent (requires OpenAI key)
                if not os.environ.get("OPENAI_API_KEY"):
                    print("\n4b. Skipping agent test (set OPENAI_API_KEY to test agent integration)")
                else:
                    print("\n4b. Testing agent with RAG tool...")
                    
                    # Use raw query mode (doesn't need LLM for synthesis)
                    async def call_rag_tool_raw(**kwargs):
                        result = await session.call_tool("query_libe_docs_raw", kwargs)
                        return result.content[0].text if result.content else "No response"
                    
                    lc_tool = StructuredTool(
                        name="query_libe_docs_raw",
                        description="Query libEnsemble documentation and return raw chunks",
                        args_schema=tools.tools[1].inputSchema,  # query_libe_docs_raw
                        coroutine=call_rag_tool_raw
                    )
                    
                    # Create agent
                    llm = ChatOpenAI(
                        model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
                        temperature=0,
                        base_url=os.environ.get("OPENAI_BASE_URL"),
                    )
                    agent = create_agent(llm, [lc_tool])
                    
                    result = await agent.ainvoke({
                        "messages": [("user", question)]
                    })
                    
                    print(f"A: {result['messages'][-1].content[:400]}...")
                    print("✓ Agent integration works!")
                
        print("\n✓ MCP server integration test passed!")
        
    except ImportError as e:
        print(f"\n⚠ Skipping MCP integration test (missing dependencies): {e}")
    except Exception as e:
        print(f"\n✗ MCP integration test failed: {e}")
        import traceback
        traceback.print_exc()

# Run async test
asyncio.run(test_mcp_integration())

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)
