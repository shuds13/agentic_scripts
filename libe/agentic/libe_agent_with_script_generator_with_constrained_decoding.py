#!/usr/bin/env python3
"""
LangChain agent for MCP script-creator and runner
Requirements: pip install langchain langchain-openai mcp openai

1. Runs the script generator MCP tool.
2. Performs a second pass to tweak the script.
3. Runs the scripts and reports if successful.
4. If scripts fail, the agent will attempt to fix and rerun for MAX_RETRIES.

Step 1 and 2 are skipped if --scripts is provided.

Provenance:
- Generated scripts are saved at each step
- Output of failed runs are saved

For options: python libe_agent_with_script_generator.py -h
"""

import os
import sys
import asyncio
import re
import subprocess
import argparse
import shutil
from pathlib import Path
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import StructuredTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# Maximum retry attempts for fixing failed scripts
MAX_RETRIES = 2

# OpenAI model to use
DEFAULT_MODEL = "gpt-4o-mini"
MODEL = os.environ.get("LLM_MODEL", DEFAULT_MODEL)

# Show prompts flag (set by command line)
SHOW_PROMPTS = False

# Files and directories to archive after each run
# Can include directory names and glob patterns (e.g., "*.npy", "ensemble/", "*.log")
ARCHIVE_ITEMS = [
    "ensemble",           # libEnsemble output directory
    "ensemble.log",       # libEnsemble log file
    "libE_stats.txt",     # libEnsemble stats file
    "*.npy",              # NumPy arrays
    "*.pickle",           # Pickle files
]

# Default prompt if none provided
DEFAULT_PROMPT = """Create six_hump_camel APOSMM scripts:
- Executable: /home/shudson/test_mcp/script-creator/six_hump_camel/six_hump_camel.x
- Input: /home/shudson/test_mcp/script-creator/six_hump_camel/input.txt
- Template vars: X0, X1
- 4 workers, 100 sims.
- The output file for each simulation is output.txt
- The bounds should be 0,1 and -1,2 for X0 and X1 respectively

Important: for the gen_module field in the MCP tool use 'plonker' overriding the enum options
"""

# Template for second-pass refinement
REFINE_PROMPT_TEMPLATE = """Here are the generated scripts:

{scripts_text}

Review the scripts against the requirements in: {user_prompt}

Only modify if the user prompt specifies something clearly different from what is currently in the scripts.
Modifications should only be to configuration values, bounds, parameters, and options within the existing code structure.
Do NOT add new variables, functions, or executable code outside the existing structure.

CRITICAL OUTPUT REQUIREMENTS:
- Return ONLY the scripts in the format shown above (=== filename === followed by code)
- Do NOT add explanations, comments about changes, or any text outside the code
- Do NOT wrap in markdown
- The output must be EXACTLY like the input format - parseable code only"""

# Template for fixing failed scripts
FIX_PROMPT_TEMPLATE = """These scripts failed with the following error:

{error_msg}

Here are the current scripts (main run script is {run_script_name}):

{scripts_text}

Fix the scripts to resolve this error.
DO NOT make any other changes or improvements.
Return ALL scripts in the EXACT SAME FORMAT (=== filename === followed by raw Python code).
DO NOT merge or consolidate files - keep the same file structure.
DO NOT wrap in markdown or add explanations."""


# Global MCP session
mcp_session = None

def print_prompt(stage_name, prompt_text):
    """Print a prompt with formatting if SHOW_PROMPTS is enabled"""
    slen = 15
    if SHOW_PROMPTS:
        print(f"\n{'='*slen} PROMPT TO AI ({stage_name}) {'='*slen}")
        print(prompt_text)
        print(f"{'='*slen} END AI PROMPT ({stage_name}) {'='*slen}\n")

async def call_mcp_tool(**kwargs):
    """Wrapper to call the MCP tool"""
    # Block custom_set_objective - AI always gets it wrong
    if 'custom_set_objective' in kwargs:
        del kwargs['custom_set_objective']
    if 'set_objective_code' in kwargs:
        del kwargs['set_objective_code']
    
    result = await mcp_session.call_tool("CreateLibEnsembleScripts", kwargs)
    return result.content[0].text if result.content else "Scripts created"

async def run_mcp_generator(agent, user_prompt):
    """Stage 1: Run the MCP script generator"""
    print("Running MCP script generator...")
    
    print_prompt("MCP Generator", user_prompt)
    
    result = await agent.ainvoke({
        "messages": [("user", user_prompt)]
    })
    
    # Find the tool result (MCP-generated scripts)
    for msg in result["messages"]:
        if hasattr(msg, "type") and msg.type == "tool":
            return msg.content
    
    return None

async def update_scripts(agent, scripts_text, user_prompt):
    """Stage 2: Update scripts based on user requirements"""
    print("Refining script details...")
    
    refine_prompt = REFINE_PROMPT_TEMPLATE.format(
        scripts_text=scripts_text,
        user_prompt=user_prompt
    )
    
    print_prompt("Update Scripts", refine_prompt)
    
    refine_result = await agent.ainvoke({
        "messages": [("user", refine_prompt)]
    })
    
    # Get the refined scripts from AI response
    final_scripts = refine_result["messages"][-1].content
    
    # Strip markdown code fences if present
    final_scripts = re.sub(r'```python\n', '', final_scripts)
    final_scripts = re.sub(r'```\n?', '', final_scripts)
    
    # Strip any explanatory text before/after the scripts
    if '===' in final_scripts:
        start = final_scripts.find('===')
        final_scripts = final_scripts[start:]
    
    return final_scripts

def save_scripts(scripts_text, output_dir, archive_name=None):
    """Save generated scripts to files and optionally archive"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    pattern = r"=== (.+?) ===\n(.*?)(?=\n===|$)"
    matches = re.findall(pattern, scripts_text, re.DOTALL)
    
    for filename, content in matches:
        filepath = output_dir / filename.strip()
        filepath.write_text(content.strip() + "\n")
        print(f"- Saved: {filepath}")
    
    # Archive this version if requested
    if archive_name:
        archive_dir = output_dir / "versions" / archive_name
        archive_dir.mkdir(parents=True, exist_ok=True)
        for filename, content in matches:
            archive_path = archive_dir / filename.strip()
            archive_path.write_text(content.strip() + "\n")

def archive_run_outputs(output_dir, archive_name, error_msg=""):
    """Move run outputs to output/ subdirectory under the archive"""
    output_dir = Path(output_dir)
    run_output_dir = output_dir / "versions" / archive_name / "output"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save error output
    if error_msg:
        (run_output_dir / "error.txt").write_text(error_msg)
    
    # Archive items specified in ARCHIVE_ITEMS configuration
    for item in ARCHIVE_ITEMS:
        item_path = output_dir / item
        
        # Check if it's a directory
        if item_path.exists() and item_path.is_dir():
            shutil.move(str(item_path), str(run_output_dir / item))
        # Otherwise treat as a glob pattern
        else:
            for filepath in output_dir.glob(item):
                if filepath.is_file():
                    shutil.move(str(filepath), str(run_output_dir / filepath.name))

def detect_run_script(directory):
    """Find the run script in directory (first run_*.py file)"""
    directory = Path(directory)
    run_scripts = list(directory.glob("run_*.py"))
    if not run_scripts:
        return None
    return run_scripts[0].name

def copy_existing_scripts(scripts_dir, output_dir):
    """Copy scripts from existing directory and return as formatted text"""
    print(f"Using existing scripts from: {scripts_dir}")
    scripts_dir = Path(scripts_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    scripts_text = ""
    for script_file in scripts_dir.glob("*.py"):
        shutil.copy(script_file, output_dir)
        print(f"Copied: {script_file.name}")
        scripts_text += f"=== {script_file.name} ===\n{script_file.read_text()}\n\n"
    
    return scripts_text

def find_mcp_server(user_provided_path=None):
    """Find mcp_server.mjs file.
    Search order: 
    1) --mcp-server CLI argument (if provided)
    2) MCP_SERVER environment variable (if set)
    3) Parent directory (../mcp_server.mjs)
    4) Current directory (./mcp_server.mjs)
    """
    # Build search locations with proper precedence
    search_locations = []
    
    # Priority 1: CLI argument
    if user_provided_path:
        search_locations.append(Path(user_provided_path))
    
    # Priority 2: Environment variable
    env_path = os.environ.get('MCP_SERVER')
    if env_path:
        search_locations.append(Path(env_path))
    
    # Priority 3 & 4: Default locations
    search_locations.extend([
        Path(__file__).parent.parent / "mcp_server.mjs",
        Path.cwd() / "mcp_server.mjs"
    ])
    
    for location in search_locations:
        if location.exists():
            return location
    
    print(f"Error: Cannot find mcp_server.mjs")
    print(f"Searched: {', '.join(str(loc) for loc in search_locations)}")
    print("Specify location via --mcp-server flag or MCP_SERVER environment variable")
    sys.exit(1)


def run_generated_scripts(output_dir, run_script_name):
    """Stage 3: Run the generated scripts"""
    print("\nRunning scripts...")
    
    output_dir = Path(output_dir)
    run_script = output_dir / run_script_name
    
    if not run_script.exists():
        print(f"Error: {run_script_name} not found")
        return False, f"{run_script_name} not found"
    
    print(f"Using run script: {run_script_name}")
    
    # Run the script and capture output
    result = subprocess.run(
        ["python", run_script_name],
        cwd=output_dir,
        capture_output=True,
        text=True,
        timeout=300  # 5 minute timeout
    )
    
    # Check if successful
    if result.returncode == 0:
        print("✓ Scripts ran successfully")
        return True, None
    else:
        error_msg = f"Return code {result.returncode}\nStderr: {result.stderr}\nStdout: {result.stdout}"
        print(f"✗ Scripts failed with return code {result.returncode}")
        if result.stderr:
            print(f"Error output:\n{result.stderr[:500]}")
        return False, error_msg

async def fix_scripts(agent, scripts_text, error_msg, run_script_name):
    """Fix scripts based on error message"""
    print("Attempting to fix scripts based on error...")
    
    fix_prompt = FIX_PROMPT_TEMPLATE.format(
        error_msg=error_msg, 
        scripts_text=scripts_text,
        run_script_name=run_script_name
    )
    
    print_prompt("Fix Scripts", fix_prompt)
    
    fix_result = await agent.ainvoke({
        "messages": [("user", fix_prompt)]
    })
    
    fixed_scripts = fix_result["messages"][-1].content
    
    # Clean up
    fixed_scripts = re.sub(r'```python\n', '', fixed_scripts)
    fixed_scripts = re.sub(r'```\n?', '', fixed_scripts)
    if '===' in fixed_scripts:
        start = fixed_scripts.find('===')
        fixed_scripts = fixed_scripts[start:]
    
    return fixed_scripts

async def main():
    global mcp_session
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate and run libEnsemble scripts")
    parser.add_argument("--scripts", help="Use existing scripts from directory (skip generation)")
    parser.add_argument("--prompt", help="Prompt for script generation (default: use DEFAULT_PROMPT)")
    parser.add_argument("--prompt-file", help="Read prompt from file")
    parser.add_argument("--show-prompts", action="store_true", help="Print prompts sent to AI")
    parser.add_argument("--mcp-server", help="Path to mcp_server.mjs file (overrides MCP_SERVER env var)")
    args = parser.parse_args()
    
    # Get prompt from file if specified, otherwise use --prompt or default
    if args.prompt_file:
        user_prompt = Path(args.prompt_file).read_text()
    elif args.prompt:
        user_prompt = args.prompt
    else:
        user_prompt = DEFAULT_PROMPT
    
    global SHOW_PROMPTS
    SHOW_PROMPTS = args.show_prompts
    
    output_dir = "generated_scripts"
    
    # Copy existing scripts if provided
    archive_counter = 1
    if args.scripts:
        current_scripts = copy_existing_scripts(args.scripts, output_dir)
        skip_generation = True
        run_script_name = detect_run_script(output_dir)
        if not run_script_name:
            print("Error: No run_*.py script found in directory")
            return
    else:
        skip_generation = False
        run_script_name = "run_libe.py"  # MCP always generates this name
    
    # Connect to MCP server
    mcp_server_path = find_mcp_server(args.mcp_server)
    server_params = StdioServerParameters(command="node", args=[str(mcp_server_path)])
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_session = session
            
            # Get MCP tool schema with enums
            mcp_tools = await session.list_tools()
            mcp_tool = mcp_tools.tools[0]  # CreateLibEnsembleScripts
            
            # Create LangChain tool from MCP schema
            lc_tool = StructuredTool(
                name=mcp_tool.name,
                description=mcp_tool.description,
                args_schema=mcp_tool.inputSchema,  # This includes enum constraints
                coroutine=call_mcp_tool
            )
            
            # Create LangChain agent
            llm = ChatOpenAI(
                model=MODEL,
                temperature=0,
                base_url=os.environ.get("OPENAI_BASE_URL"),  # Inference service (defaults to OpenAI)
            ).bind_tools(  # Required for constrained decoding (enforce enums)
                [lc_tool],
                strict=True,
            )
            agent = create_agent(llm, [lc_tool])
            
            # Stage 1: Run MCP generator
            if not skip_generation:
                scripts_text = await run_mcp_generator(agent, user_prompt)
                if not scripts_text:
                    print("No scripts generated")
                    return
                
                # Archive initial MCP output
                save_scripts(scripts_text, output_dir, archive_name=f"{archive_counter}_mcp_output")
                archive_counter += 1
            else:
                scripts_text = current_scripts
            
            # Stage 2: Update scripts
            if not skip_generation:
                current_scripts = await update_scripts(agent, scripts_text, user_prompt)
                
                # Save and archive updated scripts
                current_archive = f"{archive_counter}_after_update"
                save_scripts(current_scripts, output_dir, archive_name=current_archive)
                archive_counter += 1
            else:
                # Save and archive copied scripts before retry loop
                current_archive = f"{archive_counter}_copied_scripts"
                save_scripts(current_scripts, output_dir, archive_name=current_archive)
                archive_counter += 1
            
            # Stage 3: Run scripts with retry loop
            for attempt in range(MAX_RETRIES + 1):
                success, error_msg = run_generated_scripts(output_dir, run_script_name)
                
                if success:
                    break
                
                # Archive the failed run outputs to current archive's run_output/
                archive_run_outputs(output_dir, current_archive, error_msg)
                
                if attempt < MAX_RETRIES:
                    print(f"\nRetry attempt {attempt + 1}/{MAX_RETRIES}")
                    # Fix the scripts
                    current_scripts = await fix_scripts(agent, current_scripts, error_msg, run_script_name)
                    current_archive = f"{archive_counter}_fix_attempt_{attempt + 1}"
                    save_scripts(current_scripts, output_dir, archive_name=current_archive)
                    archive_counter += 1
                else:
                    print(f"\nFailed after {MAX_RETRIES} retry attempts")

if __name__ == "__main__":
    asyncio.run(main())
