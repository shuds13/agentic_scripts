#!/usr/bin/env python3
"""
Interactive LangChain agent for libEnsemble script generation and execution.

The agent has tools and the chat IS the agent loop:
- Each user message is a real user message in the conversation
- The agent responds with tool calls and text
- No fake 'ask_user' workaround

Without --interactive, it runs autonomously in a single invocation.

Requirements: pip install langchain langchain-openai mcp openai
For options: python libe_agent_interactive.py -h
"""

import os
import sys
import asyncio
import re
import subprocess
import argparse
import shutil
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import StructuredTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


DEFAULT_MODEL = "gpt-4o-mini"
MODEL = os.environ.get("LLM_MODEL", DEFAULT_MODEL)
SHOW_PROMPTS = False

# Marker so the web UI knows the script is waiting for input
INPUT_MARKER = "[INPUT_REQUESTED]"

DEFAULT_PROMPT = """Create six_hump_camel APOSMM scripts:
- Executable: /home/shudson/test_mcp/script-creator/six_hump_camel/six_hump_camel.x
- Input: /home/shudson/test_mcp/script-creator/six_hump_camel/input.txt
- Template vars: X0, X1
- 4 workers, 100 sims.
- The output file for each simulation is output.txt
- The bounds should be 0,1 and -1,2 for X0 and X1 respectively"""

SYSTEM_PROMPT = """You are a libEnsemble script assistant. You have tools to generate, read, write, run, and list scripts.

IMPORTANT RULES:
- Only use CreateLibEnsembleScripts ONCE to generate initial scripts. NEVER call it again.
- For ANY modifications the user requests, use read_file to see the current file, then write_file to save the edited version.
- If the user asks to see something, use read_file and show them the content.
- Don't run scripts unless the user explicitly asks you to run them.
- When reviewing scripts, highlight key configuration: generator bounds/parameters and the objective function.
- After running, if scripts fail, explain the error and offer to fix using write_file."""

ARCHIVE_ITEMS = [
    "ensemble", "ensemble.log", "libE_stats.txt",
    "*.npy", "*.pickle",
]

# Global state
mcp_session = None
WORK_DIR = None
ARCHIVE_COUNTER = 1
CURRENT_ARCHIVE = None


# ── Archiving ────────────────────────────────────────────────

def start_new_archive(action):
    global ARCHIVE_COUNTER, CURRENT_ARCHIVE
    CURRENT_ARCHIVE = f"{ARCHIVE_COUNTER}_{action}"
    (WORK_DIR / "versions" / CURRENT_ARCHIVE).mkdir(parents=True, exist_ok=True)
    ARCHIVE_COUNTER += 1


def archive_current_scripts():
    if not CURRENT_ARCHIVE:
        return
    dest = WORK_DIR / "versions" / CURRENT_ARCHIVE
    for f in WORK_DIR.glob("*.py"):
        shutil.copy(f, dest / f.name)


def archive_run_output(error_msg=""):
    if not CURRENT_ARCHIVE:
        return
    output_dir = WORK_DIR / "versions" / CURRENT_ARCHIVE / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    if error_msg:
        (output_dir / "error.txt").write_text(error_msg)
    for item in ARCHIVE_ITEMS:
        item_path = WORK_DIR / item
        if item_path.exists() and item_path.is_dir():
            shutil.copytree(str(item_path), str(output_dir / item), dirs_exist_ok=True)
            shutil.rmtree(str(item_path))
        else:
            for fp in WORK_DIR.glob(item):
                if fp.is_file():
                    shutil.copy(str(fp), str(output_dir / fp.name))
                    fp.unlink()


# ── Tool schemas ─────────────────────────────────────────────

class RunScriptInput(BaseModel):
    script_name: str = Field(description="Name of the Python script to run")

class ReadFileInput(BaseModel):
    filepath: str = Field(description="Path to file relative to work directory")

class WriteFileInput(BaseModel):
    filepath: str = Field(description="Path to file relative to work directory")
    content: str = Field(description="Full content to write")

class ListFilesInput(BaseModel):
    pass


# ── Tool implementations ────────────────────────────────────

async def run_script_tool(script_name: str) -> str:
    script_path = WORK_DIR / script_name
    if not script_path.exists():
        return f"ERROR: Script '{script_name}' not found"

    print(f"\nRunning {script_name}...", flush=True)
    try:
        result = subprocess.run(
            ["python", script_name], cwd=WORK_DIR,
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            print("✓ Script ran successfully", flush=True)
            return f"SUCCESS\nOutput:\n{result.stdout[:500]}"
        else:
            error_msg = f"Return code {result.returncode}\nStderr: {result.stderr}\nStdout: {result.stdout}"
            print(f"✗ Failed (code {result.returncode})", flush=True)
            archive_run_output(error_msg)
            return f"FAILED (code {result.returncode})\nStderr:\n{result.stderr}\nStdout:\n{result.stdout[:500]}"
    except subprocess.TimeoutExpired:
        return "ERROR: Script timed out (300s)"
    except Exception as e:
        return f"ERROR: {e}"


async def read_file_tool(filepath: str) -> str:
    file_path = WORK_DIR / filepath
    if not file_path.exists():
        return f"ERROR: File '{filepath}' not found"
    return file_path.read_text()


async def write_file_tool(filepath: str, content: str) -> str:
    try:
        (WORK_DIR / filepath).write_text(content)
        start_new_archive("fix")
        archive_current_scripts()
        print(f"- Saved: {WORK_DIR / filepath}", flush=True)
        return f"SUCCESS: Wrote {filepath}"
    except Exception as e:
        return f"ERROR: {e}"


async def list_files_tool() -> str:
    py_files = list(WORK_DIR.glob("*.py"))
    if not py_files:
        return "No Python files found"
    return "Files:\n" + "\n".join(f"- {f.name}" for f in py_files)


async def generate_scripts_mcp(**kwargs):
    """Call MCP tool to generate scripts, auto-save to work dir"""
    if 'custom_set_objective' in kwargs:
        del kwargs['custom_set_objective']
    if 'set_objective_code' in kwargs:
        del kwargs['set_objective_code']

    result = await mcp_session.call_tool("CreateLibEnsembleScripts", kwargs)
    scripts_text = result.content[0].text if result.content else ""

    if scripts_text and "===" in scripts_text:
        WORK_DIR.mkdir(exist_ok=True)
        pattern = r"=== (.+?) ===\n(.*?)(?=\n===|$)"
        for filename, content in re.findall(pattern, scripts_text, re.DOTALL):
            (WORK_DIR / filename.strip()).write_text(content.strip() + "\n")
            print(f"- Saved: {WORK_DIR / filename.strip()}", flush=True)
        start_new_archive("generated")
        archive_current_scripts()

    return scripts_text


# ── MCP server discovery ────────────────────────────────────

def find_mcp_server(user_path=None):
    locations = []
    if user_path:
        locations.append(Path(user_path))
    env_path = os.environ.get('GENERATOR_MCP_SERVER')
    if env_path:
        locations.append(Path(env_path))
    locations.extend([
        Path(__file__).parent.parent / "mcp_server.mjs",
        Path.cwd() / "mcp_server.mjs"
    ])
    for loc in locations:
        if loc.exists():
            return loc
    print("Error: Cannot find mcp_server.mjs")
    sys.exit(1)


# ── Main ─────────────────────────────────────────────────────

async def main():
    global mcp_session, WORK_DIR, SHOW_PROMPTS

    parser = argparse.ArgumentParser(
        description="Interactive agent for libEnsemble scripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python libe_agent_interactive.py --interactive
  python libe_agent_interactive.py --interactive --scripts my_scripts/
  python libe_agent_interactive.py --prompt "Create APOSMM scripts..."
        """
    )
    parser.add_argument("--interactive", action="store_true", help="Enable interactive chat mode")
    parser.add_argument("--scripts", help="Use existing scripts from directory")
    parser.add_argument("--prompt", help="Prompt for script generation")
    parser.add_argument("--prompt-file", help="Read prompt from file")
    parser.add_argument("--show-prompts", action="store_true")
    parser.add_argument("--mcp-server", help="Path to mcp_server.mjs")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=15)
    args = parser.parse_args()

    SHOW_PROMPTS = args.show_prompts
    interactive = args.interactive
    WORK_DIR = Path("generated_scripts")
    WORK_DIR.mkdir(exist_ok=True)

    # Connect to MCP server
    mcp_server = find_mcp_server(args.mcp_server)
    print(f"Generator MCP: {mcp_server}")
    server_params = StdioServerParameters(command="node", args=[str(mcp_server)])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_session = session
            print("✓ Connected to MCP server")

            # Get MCP tool schema
            mcp_tools = await session.list_tools()
            mcp_tool = mcp_tools.tools[0]

            # Build tools
            tools = [
                StructuredTool(
                    name=mcp_tool.name, description=mcp_tool.description,
                    args_schema=mcp_tool.inputSchema, coroutine=generate_scripts_mcp
                ),
                StructuredTool(name="run_script", description="Run a Python script. Returns SUCCESS or FAILED with error details.", args_schema=RunScriptInput, coroutine=run_script_tool),
                StructuredTool(name="read_file", description="Read a file to inspect its contents.", args_schema=ReadFileInput, coroutine=read_file_tool),
                StructuredTool(name="write_file", description="Write/overwrite a file to fix scripts.", args_schema=WriteFileInput, coroutine=write_file_tool),
                StructuredTool(name="list_files", description="List Python files in working directory.", args_schema=ListFilesInput, coroutine=list_files_tool),
            ]

            llm = ChatOpenAI(model=MODEL, temperature=0, base_url=os.environ.get("OPENAI_BASE_URL"))
            agent = create_agent(llm, tools)
            print("✓ Agent initialized\n")

            # Build initial message
            messages = [("system", SYSTEM_PROMPT)]

            if args.scripts:
                # Load existing scripts
                scripts_dir = Path(args.scripts)
                for f in sorted(scripts_dir.glob("*.py")):
                    shutil.copy(f, WORK_DIR)
                    print(f"Copied: {f.name}")
                start_new_archive("copied_scripts")
                archive_current_scripts()

                run_scripts = list(WORK_DIR.glob("run_*.py"))
                run_name = run_scripts[0].name if run_scripts else "run_libe.py"
                initial_msg = f"I have libEnsemble scripts. The main script is '{run_name}'. Please review them and highlight the key configuration."

            elif args.prompt:
                initial_msg = args.prompt
            elif args.prompt_file:
                initial_msg = Path(args.prompt_file).read_text()
            elif interactive:
                print("Describe the scripts you want to generate (or press Enter for default demo):", flush=True)
                print(INPUT_MARKER, flush=True)
                user_input = input().strip()
                initial_msg = user_input if user_input else DEFAULT_PROMPT
                if not user_input:
                    print("Using default demo prompt")
            else:
                initial_msg = DEFAULT_PROMPT

            if not interactive:
                # Autonomous mode: single invocation, agent does everything
                goal = f"""{initial_msg}

After generating/loading scripts: review them, run them, fix errors and retry (max 3 attempts). Report the result."""
                messages.append(("user", goal))

                if SHOW_PROMPTS:
                    print(f"Goal: {goal}\n")
                print("Starting agent...\n")

                result = await agent.ainvoke({"messages": messages})
                print(f"\n{'='*60}")
                print("✓ Agent completed")
                print(f"{'='*60}")
                print(result["messages"][-1].content)

            else:
                # Interactive mode: chat loop with automatic refine cycle
                goal = f"""User request: {initial_msg}

Instructions:
1. Use CreateLibEnsembleScripts to generate the initial scripts.
2. Read each generated script using read_file.
3. Check the scripts match the user's request (bounds, sims, paths, parameters, etc).
4. If anything doesn't match, fix it using write_file. Common issues: wrong bounds, wrong sim count, missing paths.
5. Present a concise summary of the scripts and what you fixed (if anything).
6. Then wait for the user's feedback."""
                messages.append(("user", goal))
                print("Starting agent...\n")

                while True:
                    try:
                        # Agent turn
                        result = await agent.ainvoke({"messages": messages})
                        messages = result["messages"]

                        # Print agent's response
                        response = messages[-1].content
                        if response:
                            print(f"\n{response}", flush=True)
                    except Exception as e:
                        print(f"\n⚠️ Agent error: {e}", flush=True)

                    # Wait for user input
                    print(INPUT_MARKER, flush=True)
                    user_input = input().strip()

                    if not user_input or user_input.lower() in ('quit', 'exit', 'done'):
                        print("\n✓ Session ended")
                        break

                    # Add as a proper HumanMessage to match LangGraph's message format
                    from langchain_core.messages import HumanMessage, SystemMessage
                    # Remind the model to respond to the user, not continue previous task
                    messages.append(SystemMessage(content="STOP. Read the user's next message carefully and respond to exactly what they ask. Do not continue previous tasks."))
                    messages.append(HumanMessage(content=user_input))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
