#!/usr/bin/env python3
"""
Autonomous LangChain agent for running and fixing libEnsemble scripts
Requirements: pip install langchain langchain-openai

This version gives the agent tools and lets it decide the workflow autonomously.
Instead of hardcoded stages, the agent decides:
- When to run scripts
- When to read files
- When to fix errors
- When to retry

For options: python libe_agent_basic_auto.py -h
"""

import os
import sys
import asyncio
import subprocess
import argparse
import shutil
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import StructuredTool


# OpenAI model to use
DEFAULT_MODEL = "gpt-4o-mini"
MODEL = os.environ.get("LLM_MODEL", DEFAULT_MODEL)

# Working directory for scripts
WORK_DIR = None

# Archive counter for version tracking
ARCHIVE_COUNTER = 1

# Current archive name (scripts and their output go together)
CURRENT_ARCHIVE = None

# Files and directories to archive after each run
ARCHIVE_ITEMS = [
    "ensemble",           # libEnsemble output directory
    "ensemble.log",       # libEnsemble log file
    "libE_stats.txt",     # libEnsemble stats file
    "*.npy",              # NumPy arrays
    "*.pickle",           # Pickle files
]


# Tool schemas
class RunScriptInput(BaseModel):
    script_name: str = Field(description="Name of the Python script to run (e.g., 'run_libe.py')")

class ReadFileInput(BaseModel):
    filepath: str = Field(description="Path to the file to read (relative to work directory)")

class WriteFileInput(BaseModel):
    filepath: str = Field(description="Path to the file to write (relative to work directory)")
    content: str = Field(description="Content to write to the file")

class ListFilesInput(BaseModel):
    pass  # No input needed


# Archiving functions
def start_new_archive(action: str):
    """Start a new archive version (scripts + their output go together)"""
    global ARCHIVE_COUNTER, CURRENT_ARCHIVE
    CURRENT_ARCHIVE = f"{ARCHIVE_COUNTER}_{action}"
    archive_dir = WORK_DIR / "versions" / CURRENT_ARCHIVE
    archive_dir.mkdir(parents=True, exist_ok=True)
    ARCHIVE_COUNTER += 1
    print(f"[Archive] Started new version: {CURRENT_ARCHIVE}")


def archive_current_scripts():
    """Archive all current scripts to the current archive"""
    if CURRENT_ARCHIVE is None:
        return
    
    archive_dir = WORK_DIR / "versions" / CURRENT_ARCHIVE
    
    # Copy all Python scripts to the archive
    for script_file in WORK_DIR.glob("*.py"):
        shutil.copy(script_file, archive_dir / script_file.name)
    
    print(f"[Archive] Saved scripts to: {CURRENT_ARCHIVE}/")


def archive_run_output(error_msg: str):
    """Archive run output to the current archive (same dir as the scripts)"""
    if CURRENT_ARCHIVE is None:
        return
    
    archive_dir = WORK_DIR / "versions" / CURRENT_ARCHIVE
    output_dir = archive_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save error output
    (output_dir / "error.txt").write_text(error_msg)
    
    # Archive configured items
    for item in ARCHIVE_ITEMS:
        item_path = WORK_DIR / item
        
        if item_path.exists() and item_path.is_dir():
            shutil.copytree(str(item_path), str(output_dir / item), dirs_exist_ok=True)
            shutil.rmtree(str(item_path))
        else:
            for filepath in WORK_DIR.glob(item):
                if filepath.is_file():
                    shutil.copy(str(filepath), str(output_dir / filepath.name))
                    filepath.unlink()
    
    print(f"[Archive] Saved run output to: {CURRENT_ARCHIVE}/output/")


# Tool implementations
async def run_script_tool(script_name: str) -> str:
    """Run a Python script and return output or error"""
    script_path = WORK_DIR / script_name
    timeout_seconds = 300
    
    if not script_path.exists():
        msg = f"ERROR: Script '{script_name}' not found in {WORK_DIR}"
        print(f"\n{msg}\n")
        return msg
    
    print("\nRunning scripts...")
    
    try:
        result = subprocess.run(
            ["python", script_name],
            cwd=WORK_DIR,
            capture_output=True,
            text=True,
            timeout=timeout_seconds
        )
        
        if result.returncode == 0:
            print("✓ Script ran successfully")
            msg = f"SUCCESS: Script ran successfully.\nOutput:\n{result.stdout[:500]}"
        else:
            error_msg = f"Return code {result.returncode}\nStderr: {result.stderr}\nStdout: {result.stdout}"
            print(f"✗ Scripts failed with return code {result.returncode}")
            if result.stderr:
                error_summary = result.stderr.strip().split('\n')[-1]
                print(f"Error summary: {error_summary}\n")
            # Archive the failed run output (goes with the current scripts)
            archive_run_output(error_msg)
            msg = f"FAILED: Script failed with return code {result.returncode}\n\nStderr:\n{result.stderr}\n\nStdout:\n{result.stdout[:500]}"
        
        return msg
        
    except subprocess.TimeoutExpired:
        msg = f"ERROR: Script timed out after {timeout_seconds} seconds"
        print(f"\n{msg}\n")
        return msg
    except Exception as e:
        msg = f"ERROR: {str(e)}"
        print(msg)
        return msg


async def read_file_tool(filepath: str) -> str:
    """Read a file and return its contents"""
    file_path = WORK_DIR / filepath
    
    if not file_path.exists():
        return f"ERROR: File '{filepath}' not found"
    
    try:
        content = file_path.read_text()
        return content
    except Exception as e:
        return f"ERROR reading file: {str(e)}"


async def write_file_tool(filepath: str, content: str) -> str:
    """Write content to a file"""
    file_path = WORK_DIR / filepath
    
    try:
        file_path.write_text(content)
        # Start a new archive for this fixed version
        start_new_archive("script_fix")
        archive_current_scripts()
        return f"SUCCESS: Wrote {len(content)} characters to {filepath}"
    except Exception as e:
        msg = f"ERROR writing file: {str(e)}"
        print(msg)
        return msg


async def list_files_tool() -> str:
    """List all Python files in the working directory"""
    try:
        py_files = list(WORK_DIR.glob("*.py"))
        if not py_files:
            return "No Python files found"
        return "Python files:\n" + "\n".join([f"- {f.name}" for f in py_files])
    except Exception as e:
        return f"ERROR listing files: {str(e)}"


def setup_work_directory(scripts_dir: str) -> Path:
    """Copy scripts to working directory and archive initial version"""
    global WORK_DIR
    scripts_dir = Path(scripts_dir)
    work_dir = Path("generated_scripts")
    work_dir.mkdir(exist_ok=True)
    WORK_DIR = work_dir
    
    # Copy all Python files to work dir
    for script_file in scripts_dir.glob("*.py"):
        shutil.copy(script_file, work_dir)
        print(f"Copied: {script_file.name}")
    
    # Start archive for initial scripts
    start_new_archive("copied_scripts")
    archive_current_scripts()
    
    return work_dir


async def main():
    global WORK_DIR
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Autonomous agent to run and fix libEnsemble scripts")
    parser.add_argument("--scripts", required=True, help="Directory containing scripts to run")
    parser.add_argument("--max-iterations", type=int, default=10, 
                       help="Maximum agent iterations (default: 10)")
    args = parser.parse_args()
    
    # Setup working directory
    WORK_DIR = setup_work_directory(args.scripts)
    print(f"\nWorking directory: {WORK_DIR}\n")
    
    # Detect run script
    run_scripts = list(WORK_DIR.glob("run_*.py"))
    if not run_scripts:
        print("Error: No run_*.py script found")
        return
    run_script_name = run_scripts[0].name
    
    # Create tools
    run_tool = StructuredTool(
        name="run_script",
        description="Run a Python script. Returns SUCCESS if it works, FAILED with error details if it fails.",
        args_schema=RunScriptInput,
        coroutine=run_script_tool
    )
    
    read_tool = StructuredTool(
        name="read_file",
        description="Read a file and return its contents. Use this to inspect scripts before fixing them.",
        args_schema=ReadFileInput,
        coroutine=read_file_tool
    )
    
    write_tool = StructuredTool(
        name="write_file",
        description="Write content to a file. Use this to fix scripts that have errors.",
        args_schema=WriteFileInput,
        coroutine=write_file_tool
    )
    
    list_tool = StructuredTool(
        name="list_files",
        description="List all Python files in the working directory.",
        args_schema=ListFilesInput,
        coroutine=list_files_tool
    )
    
    # Create agent
    llm = ChatOpenAI(
        model=MODEL,
        temperature=0,
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    agent = create_agent(llm, [run_tool, read_tool, write_tool, list_tool])
    
    # Give agent the goal
    goal = f"""Your goal is to successfully run the script '{run_script_name}'.

Instructions:
1. First, run the script using the run_script tool
2. If it succeeds, report success and stop
3. If it fails, read the failing script to understand the error
4. Fix the script by writing the corrected version
5. Run it again to verify the fix works
6. Repeat until successful or you determine it's unfixable, or you have tried three times.

The script is in the current working directory. Use the tools available to accomplish this goal.

Start by running '{run_script_name}'."""
    
    print("="*60)
    print("AGENT GOAL:")
    print(goal)
    print("="*60)
    print("\nStarting autonomous agent...")
    
    try:
        result = await agent.ainvoke({
            "messages": [("user", goal)]
        })
        
        print("\n" + "="*60)
        print("AGENT COMPLETED")
        print("="*60)
        print("\nFinal response:")
        print(result["messages"][-1].content)
        
    except Exception as e:
        print(f"\nAgent error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
