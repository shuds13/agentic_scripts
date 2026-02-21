#!/usr/bin/env python3
"""
LangChain agent for running libEnsemble scripts with error recovery
Requirements: pip install langchain langchain-openai

1. Runs existing libEnsemble scripts.
2. If scripts fail, the agent will attempt to fix and rerun for MAX_RETRIES.

Provenance:
- Scripts are saved at each step
- Output of failed runs are saved

For options: python libe_agent_basic.py -h
"""

import os
import sys
import asyncio
import re
import subprocess
import argparse
import shutil
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent


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



def print_prompt(stage_name, prompt_text):
    """Print a prompt with formatting if SHOW_PROMPTS is enabled"""
    slen = 15
    if SHOW_PROMPTS:
        print(f"\n{'='*slen} PROMPT TO AI ({stage_name}) {'='*slen}")
        print(prompt_text)
        print(f"{'='*slen} END AI PROMPT ({stage_name}) {'='*slen}\n")

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

def run_scripts(output_dir, run_script_name):
    """Run the scripts"""
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
    global SHOW_PROMPTS
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run and fix libEnsemble scripts")
    parser.add_argument("--scripts", required=True, help="Directory containing scripts to run")
    parser.add_argument("--show-prompts", action="store_true", help="Print prompts sent to AI")
    args = parser.parse_args()
    
    SHOW_PROMPTS = args.show_prompts
    output_dir = "generated_scripts"
    
    # Copy existing scripts
    current_scripts = copy_existing_scripts(args.scripts, output_dir)
    run_script_name = detect_run_script(output_dir)
    if not run_script_name:
        print("Error: No run_*.py script found in directory")
        return
    
    llm = ChatOpenAI(
        model=MODEL,
        temperature=0,
        base_url=os.environ.get("OPENAI_BASE_URL"),  # Inference service (defaults to OpenAI)
    )
    agent = create_agent(llm, [])
    
    # Save and archive copied scripts before retry loop
    archive_counter = 1
    current_archive = f"{archive_counter}_copied_scripts"
    save_scripts(current_scripts, output_dir, archive_name=current_archive)
    archive_counter += 1
    
    # Run scripts with retry loop
    for attempt in range(MAX_RETRIES + 1):
        success, error_msg = run_scripts(output_dir, run_script_name)
        
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
