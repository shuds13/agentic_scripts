#!/usr/bin/env python3
"""
Example agent that uses both script generator MCP and RAG MCP servers.
Demonstrates how the agent can query docs while generating scripts.
"""

import os
import sys
import asyncio
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import StructuredTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    # Find MCP servers
    script_creator_root = Path(__file__).parent.parent
    generator_mcp = os.environ.get('GENERATOR_MCP_SERVER') or str(script_creator_root / "mcp_server.mjs")
    rag_mcp = os.environ.get('RAG_MCP_SERVER') or str(Path(__file__).parent / "mcp_server.py")
    
    print(f"Generator MCP: {generator_mcp}")
    print(f"RAG MCP: {rag_mcp}")
    
    # Setup server parameters
    generator_params = StdioServerParameters(command="node", args=[generator_mcp])
    rag_params = StdioServerParameters(command="python", args=[rag_mcp])
    
    # Connect to both MCP servers
    async with stdio_client(generator_params) as (gen_read, gen_write), \
               stdio_client(rag_params) as (rag_read, rag_write):
        
        # Initialize sessions
        async with ClientSession(gen_read, gen_write) as gen_session, \
                   ClientSession(rag_read, rag_write) as rag_session:
            
            await gen_session.initialize()
            await rag_session.initialize()
            
            print("\n✓ Connected to both MCP servers")
            
            # Get tools from both servers
            gen_tools = await gen_session.list_tools()
            rag_tools = await rag_session.list_tools()
            
            print(f"\nGenerator tools: {[t.name for t in gen_tools.tools]}")
            print(f"RAG tools: {[t.name for t in rag_tools.tools]}")
            
            # Create LangChain tools
            async def call_generator(**kwargs):
                # Block custom_set_objective - AI always gets it wrong
                if 'custom_set_objective' in kwargs:
                    del kwargs['custom_set_objective']
                if 'set_objective_code' in kwargs:
                    del kwargs['set_objective_code']
                result = await gen_session.call_tool("CreateLibEnsembleScripts", kwargs)
                return result.content[0].text if result.content else "Scripts created"
            
            async def call_rag_general(**kwargs):
                result = await rag_session.call_tool("query_libe_docs", kwargs)
                return result.content[0].text if result.content else "No response"
            
            async def call_rag_generator(**kwargs):
                result = await rag_session.call_tool("query_generator_docs", kwargs)
                return result.content[0].text if result.content else "No response"
            
            async def call_rag_raw(**kwargs):
                result = await rag_session.call_tool("query_libe_docs_raw", kwargs)
                return result.content[0].text if result.content else "No response"
            
            lc_tools = [
                StructuredTool(
                    name="CreateLibEnsembleScripts",
                    description=gen_tools.tools[0].description,
                    args_schema=gen_tools.tools[0].inputSchema,
                    coroutine=call_generator
                ),
                StructuredTool(
                    name="query_libe_docs",
                    description=rag_tools.tools[0].description,
                    args_schema=rag_tools.tools[0].inputSchema,
                    coroutine=call_rag_general
                ),
                StructuredTool(
                    name="query_generator_docs",
                    description=rag_tools.tools[2].description,
                    args_schema=rag_tools.tools[2].inputSchema,
                    coroutine=call_rag_generator
                ),
                StructuredTool(
                    name="query_libe_docs_raw",
                    description=rag_tools.tools[1].description,
                    args_schema=rag_tools.tools[1].inputSchema,
                    coroutine=call_rag_raw
                ),
            ]
            
            # Create agent
            llm = ChatOpenAI(
                model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
                temperature=0,
                base_url=os.environ.get("OPENAI_BASE_URL"),
            )
            agent = create_agent(llm, lc_tools)
            
            print("\n✓ Agent created with both generator and RAG tools")
            
            # Example: Agent queries docs, then generates scripts
            print("\n" + "=" * 60)
            print("Example: Documentation-Aware Script Generation")
            print("=" * 60)
            
            prompt = """I need to create APOSMM optimization scripts.
First, look up what the main APOSMM generator options are.
Then create scripts with these specifications:
- 4 workers, 100 simulations
- Executable: /tmp/sim.x
- Input file: /tmp/input.txt
- Template variables: X0, X1
- Bounds: X0 [0, 1], X1 [-1, 2]
- Output file: output.txt"""
            
            print(f"\nPrompt:\n{prompt}\n")
            print("Agent working...\n")
            
            result = await agent.ainvoke({
                "messages": [("user", prompt)]
            })
            
            # Show agent's reasoning and final output
            print("\n" + "=" * 60)
            print("Agent Messages:")
            print("=" * 60)
            for msg in result["messages"]:
                if hasattr(msg, 'type'):
                    if msg.type == "ai":
                        print(f"\n[Agent]: {msg.content[:500]}")
                    elif msg.type == "tool":
                        print(f"\n[Tool Response]: {msg.content[:300]}...")
            
            print("\n" + "=" * 60)
            print("Final Response:")
            print("=" * 60)
            print(result["messages"][-1].content)
            
            print("\n✓ Example completed successfully!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
