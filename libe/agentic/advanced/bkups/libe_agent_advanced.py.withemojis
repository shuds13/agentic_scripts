#!/usr/bin/env python3
"""
Advanced LangChain agent for running, fixing, and tuning libEnsemble APOSMM scripts

This agent can:
1. Run scripts and fix errors (like the basic agent)
2. Detect when APOSMM fails to find enough minima
3. Analyze optimization results (.npy and .pickle files)
4. Adjust APOSMM configuration to improve results
5. Re-run with previous points as H0

For options: python libe_agent_advanced.py -h
"""

import os
import sys
import asyncio
import subprocess
import argparse
import shutil
import pickle
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

import numpy as np


# OpenAI model to use
DEFAULT_MODEL = "gpt-4o-mini"
MODEL = os.environ.get("LLM_MODEL", DEFAULT_MODEL)

# Working directory for scripts
WORK_DIR = None

# Log file for AI interactions
LOG_FILE = None

# Archive counter for version tracking
ARCHIVE_COUNTER = 1

# Current archive name (scripts and their output go together)
CURRENT_ARCHIVE = None

# Track optimization attempts - HARD LIMIT
OPTIMIZATION_ATTEMPTS = 0
MAX_OPTIMIZATION_ATTEMPTS = 3

# Files and directories to archive after each run
ARCHIVE_ITEMS = [
    "ensemble",           # libEnsemble output directory
    "ensemble.log",       # libEnsemble log file
    "libE_stats.txt",     # libEnsemble stats file
    "*.npy",              # NumPy arrays
    "*.pickle",           # Pickle files
]

# APOSMM options schema (embedded for agent reference)
APOSMM_OPTIONS_SCHEMA = """
APOSMM Generator Options (gen_specs["user"] parameters):

REQUIRED:
- lb: array[number] - Lower bounds of the search domain
- ub: array[number] - Upper bounds of the search domain  
- localopt_method: string - Local optimizer name (see below)
- initial_sample_size: integer - Points to evaluate before local optimization

OPTIONAL (you can adjust these):
- max_active_runs: integer (default 1) - Max simultaneous local optimization runs
- sample_points: array[array[number]] - User-specified initial sample points
- dist_to_bound_multiple: number - Fraction for initial localopt step size
- mu: number - Minimum distance from boundaries for localopt starts
- nu: number - Minimum distance from existing minima for localopt starts
- rk_const: number - Multiplier for r_k value when launching new runs

LOCAL OPTIMIZERS (derivative-free, for scalar f):
- scipy_Nelder-Mead: Nelder-Mead via SciPy (robust, good default)
- scipy_COBYLA: Constrained optimization via SciPy
- LN_BOBYQA: NLopt bound-constrained quadratic model (good for smooth problems)
- LN_SBPLX: NLopt subplex method (good for noisy objectives)
- LN_NELDERMEAD: NLopt Nelder-Mead (robust but slower)
- LN_COBYLA: NLopt constrained optimization

TOLERANCE OPTIONS (DO NOT LOOSEN - this is cheating):
- For SciPy: xatol, fatol (absolute tolerances)
- For NLopt: xtol_abs, ftol_abs, xtol_rel, ftol_rel
"""


def log_message(msg: str):
    """Log a message to both console and log file"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted)
    if LOG_FILE:
        with open(LOG_FILE, "a") as f:
            f.write(formatted + "\n")


def log_ai_interaction(role: str, content: str):
    """Log AI interactions to the log file"""
    if LOG_FILE:
        timestamp = datetime.now().strftime("%H:%M:%S")
        with open(LOG_FILE, "a") as f:
            f.write(f"\n[{timestamp}] === {role.upper()} ===\n")
            f.write(content[:2000] + ("..." if len(content) > 2000 else "") + "\n")


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

class ReadOptResultsInput(BaseModel):
    pass  # Will find the latest .npy file automatically

class ReadPersisInfoInput(BaseModel):
    pass  # Will find the latest .pickle file automatically

class GetApossmmOptionsInput(BaseModel):
    pass  # No input needed


# Archiving functions
def start_new_archive(action: str):
    """Start a new archive version (scripts + their output go together)"""
    global ARCHIVE_COUNTER, CURRENT_ARCHIVE
    CURRENT_ARCHIVE = f"{ARCHIVE_COUNTER}_{action}"
    archive_dir = WORK_DIR / "versions" / CURRENT_ARCHIVE
    archive_dir.mkdir(parents=True, exist_ok=True)
    ARCHIVE_COUNTER += 1
    log_message(f"[Archive] Started new version: {CURRENT_ARCHIVE}")


def archive_current_scripts():
    """Archive all current scripts to the current archive"""
    if CURRENT_ARCHIVE is None:
        return
    
    archive_dir = WORK_DIR / "versions" / CURRENT_ARCHIVE
    
    # Copy all Python scripts to the archive
    for script_file in WORK_DIR.glob("*.py"):
        shutil.copy(script_file, archive_dir / script_file.name)
    
    log_message(f"[Archive] Saved scripts to: {CURRENT_ARCHIVE}/")


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
    
    log_message(f"[Archive] Saved run output to: {CURRENT_ARCHIVE}/output/")


def find_latest_file(pattern: str) -> Optional[Path]:
    """Find the most recently modified file matching pattern"""
    files = list(WORK_DIR.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda f: f.stat().st_mtime)


# Tool implementations
async def run_script_tool(script_name: str) -> str:
    """Run a Python script and return output or error"""
    global OPTIMIZATION_ATTEMPTS
    
    # HARD CHECK: Refuse to run if we've exceeded attempts
    if OPTIMIZATION_ATTEMPTS >= MAX_OPTIMIZATION_ATTEMPTS:
        msg = f"STOPPED: Maximum optimization attempts ({MAX_OPTIMIZATION_ATTEMPTS}) reached. Cannot run again."
        log_message(f"⛔ {msg}")
        return msg
    
    script_path = WORK_DIR / script_name
    timeout_seconds = 300
    
    if not script_path.exists():
        msg = f"ERROR: Script '{script_name}' not found in {WORK_DIR}"
        log_message(f"❌ {msg}")
        return msg
    
    log_message(f"🚀 Running script: {script_name}...")
    
    try:
        result = subprocess.run(
            ["python", script_name],
            cwd=WORK_DIR,
            capture_output=True,
            text=True,
            timeout=timeout_seconds
        )
        
        combined_output = f"Stdout:\n{result.stdout}\n\nStderr:\n{result.stderr}"
        
        if result.returncode == 0:
            # Check for APOSMM success/failure
            if "APOSMM_SUCCESS" in result.stdout:
                log_message("✅ SUCCESS: APOSMM found enough minima!")
                msg = f"SUCCESS: Script ran and APOSMM found enough minima!\nOutput:\n{result.stdout}"
            elif "APOSMM_NOT_SUCCEEDED" in result.stdout:
                OPTIMIZATION_ATTEMPTS += 1
                remaining = MAX_OPTIMIZATION_ATTEMPTS - OPTIMIZATION_ATTEMPTS
                log_message(f"⚠️  APOSMM did not find enough minima (attempt {OPTIMIZATION_ATTEMPTS}/{MAX_OPTIMIZATION_ATTEMPTS})")
                
                msg = f"OPTIMIZATION_NEEDED: Script ran but APOSMM did not find enough minima.\n"
                msg += f"Attempt {OPTIMIZATION_ATTEMPTS}/{MAX_OPTIMIZATION_ATTEMPTS} - Remaining attempts: {remaining}\n"
                msg += f"Output:\n{result.stdout}\n\n"
                
                if remaining > 0:
                    msg += "NEXT STEPS:\n"
                    msg += "1. Use read_optimization_results to see where best points are\n"
                    msg += "2. Use get_aposmm_options to understand tuning options\n"
                    msg += "3. MOST EFFECTIVE: Narrow the bounds (lb, ub) to focus on promising regions\n"
                    msg += "4. Modify the script and run again\n"
                    msg += "\nIMPORTANT: Base your changes on the ANALYSIS of results, not random guessing."
                else:
                    msg += "\n⛔ NO ATTEMPTS REMAINING. You must stop now."
            else:
                log_message("✓ Script completed (no APOSMM status detected)")
                msg = f"SUCCESS: Script ran successfully.\nOutput:\n{result.stdout[:1000]}"
        else:
            error_msg = f"Return code {result.returncode}\nStderr: {result.stderr}\nStdout: {result.stdout}"
            log_message(f"❌ Script failed with return code {result.returncode}")
            if result.stderr:
                error_summary = result.stderr.strip().split('\n')[-1]
                log_message(f"   Error: {error_summary}")
            # Archive the failed run output (goes with the current scripts)
            archive_run_output(error_msg)
            msg = f"FAILED: Script failed with return code {result.returncode}\n\nStderr:\n{result.stderr}\n\nStdout:\n{result.stdout[:500]}"
        
        return msg
        
    except subprocess.TimeoutExpired:
        msg = f"ERROR: Script timed out after {timeout_seconds} seconds"
        log_message(f"⏱️  {msg}")
        return msg
    except Exception as e:
        msg = f"ERROR: {str(e)}"
        log_message(f"❌ {msg}")
        return msg


async def read_file_tool(filepath: str) -> str:
    """Read a file and return its contents"""
    log_message(f"📖 Reading file: {filepath}")
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
    log_message(f"✏️  Writing file: {filepath}")
    file_path = WORK_DIR / filepath
    
    try:
        file_path.write_text(content)
        # Start a new archive for this fixed version
        start_new_archive("script_fix")
        archive_current_scripts()
        return f"SUCCESS: Wrote {len(content)} characters to {filepath}"
    except Exception as e:
        msg = f"ERROR writing file: {str(e)}"
        log_message(f"❌ {msg}")
        return msg


async def list_files_tool() -> str:
    """List all Python files in the working directory"""
    log_message("📁 Listing files...")
    try:
        py_files = list(WORK_DIR.glob("*.py"))
        npy_files = list(WORK_DIR.glob("*.npy"))
        pickle_files = list(WORK_DIR.glob("*.pickle"))
        
        result = "Files in working directory:\n\n"
        
        if py_files:
            result += "Python scripts:\n" + "\n".join([f"- {f.name}" for f in py_files]) + "\n\n"
        
        if npy_files:
            result += "NumPy result files (.npy):\n" + "\n".join([f"- {f.name}" for f in npy_files]) + "\n\n"
        
        if pickle_files:
            result += "Pickle files (.pickle):\n" + "\n".join([f"- {f.name}" for f in pickle_files]) + "\n\n"
        
        if not py_files and not npy_files and not pickle_files:
            result = "No relevant files found"
        
        return result
    except Exception as e:
        return f"ERROR listing files: {str(e)}"


async def read_optimization_results_tool() -> str:
    """Read and summarize the latest .npy optimization results file"""
    log_message("📊 Analyzing optimization results...")
    try:
        npy_file = find_latest_file("*.npy")
        if npy_file is None:
            return "ERROR: No .npy result files found. Run the script first."
        
        H = np.load(npy_file)
        
        result = f"=== Optimization Results from {npy_file.name} ===\n\n"
        result += f"Total points evaluated: {len(H)}\n"
        result += f"Available fields: {list(H.dtype.names)}\n\n"
        
        # Basic statistics on f values
        if 'f' in H.dtype.names:
            f_vals = H['f']
            valid_f = f_vals[~np.isnan(f_vals)]
            result += f"Objective function (f) statistics:\n"
            result += f"  Min f: {np.min(valid_f):.6f}\n"
            result += f"  Max f: {np.max(valid_f):.6f}\n"
            result += f"  Mean f: {np.mean(valid_f):.6f}\n"
            result += f"  Std f: {np.std(valid_f):.6f}\n\n"
        
        # Local minima found
        if 'local_min' in H.dtype.names:
            local_min_mask = H['local_min']
            num_local_min = np.sum(local_min_mask)
            result += f"Local minima found: {num_local_min}\n"
            
            if num_local_min > 0 and 'x' in H.dtype.names:
                min_points = H[local_min_mask]
                result += "Local minima locations and values:\n"
                for i, pt in enumerate(min_points[:10]):  # Show up to 10
                    result += f"  {i+1}. x={pt['x']}, f={pt['f']:.6f}\n"
                if num_local_min > 10:
                    result += f"  ... and {num_local_min - 10} more\n"
            result += "\n"
        
        # Points that were part of local optimization runs
        if 'local_pt' in H.dtype.names:
            num_local_pt = np.sum(H['local_pt'])
            result += f"Points in local optimization runs: {num_local_pt}\n\n"
        
        # Show x range (actual sampled domain) vs current bounds
        if 'x' in H.dtype.names:
            x_vals = H['x']
            result += f"Sampled x ranges (where points actually are):\n"
            for dim in range(x_vals.shape[1]):
                result += f"  Dim {dim}: [{np.min(x_vals[:, dim]):.4f}, {np.max(x_vals[:, dim]):.4f}]\n"
            result += "\n"
            
            # Suggest tighter bounds based on best points
            if 'f' in H.dtype.names:
                valid_indices = ~np.isnan(H['f'])
                valid_H = H[valid_indices]
                sorted_indices = np.argsort(valid_H['f'])[:max(10, len(valid_H)//10)]  # Top 10%
                best_x = valid_H['x'][sorted_indices]
                
                result += "SUGGESTED BOUNDS (based on top 10% best points):\n"
                for dim in range(best_x.shape[1]):
                    margin = (np.max(best_x[:, dim]) - np.min(best_x[:, dim])) * 0.2
                    suggested_lb = np.min(best_x[:, dim]) - margin
                    suggested_ub = np.max(best_x[:, dim]) + margin
                    result += f"  Dim {dim}: lb={suggested_lb:.4f}, ub={suggested_ub:.4f}\n"
                result += "\n"
        
        # Best points (top 5 lowest f)
        if 'f' in H.dtype.names and 'x' in H.dtype.names:
            valid_indices = ~np.isnan(H['f'])
            valid_H = H[valid_indices]
            sorted_indices = np.argsort(valid_H['f'])[:5]
            result += "Top 5 best points (lowest f):\n"
            for i, idx in enumerate(sorted_indices):
                pt = valid_H[idx]
                result += f"  {i+1}. x={pt['x']}, f={pt['f']:.6f}\n"
        
        log_message(f"   Found {num_local_min if 'local_min' in H.dtype.names else '?'} local minima in {len(H)} points")
        return result
        
    except Exception as e:
        return f"ERROR reading optimization results: {str(e)}"


async def read_persis_info_tool() -> str:
    """Read and summarize the latest .pickle persis_info file"""
    log_message("📋 Reading persistent generator info...")
    try:
        pickle_file = find_latest_file("*.pickle")
        if pickle_file is None:
            return "ERROR: No .pickle files found. Run the script first."
        
        with open(pickle_file, 'rb') as f:
            persis_info = pickle.load(f)
        
        result = f"=== Persistent Info from {pickle_file.name} ===\n\n"
        
        # persis_info is typically a dict with worker IDs as keys
        result += f"Number of entries: {len(persis_info)}\n"
        result += f"Keys: {list(persis_info.keys())}\n\n"
        
        # Look for generator's persis_info (usually worker 1 or has 'run_order' key)
        gen_info = None
        for key, val in persis_info.items():
            if isinstance(val, dict):
                if 'run_order' in val or 'old_runs' in val or 'total_runs' in val:
                    gen_info = val
                    result += f"Generator persistent info (worker {key}):\n"
                    break
        
        if gen_info:
            # Show relevant optimization run info
            if 'total_runs' in gen_info:
                result += f"  Total optimization runs started: {gen_info['total_runs']}\n"
            
            if 'run_order' in gen_info:
                run_order = gen_info['run_order']
                result += f"  Completed optimization runs: {len(run_order)}\n"
                result += f"  Run order (sim_ids per run):\n"
                for run_id, sim_ids in list(run_order.items())[:5]:
                    result += f"    Run {run_id}: {len(sim_ids)} evaluations\n"
                if len(run_order) > 5:
                    result += f"    ... and {len(run_order) - 5} more runs\n"
            
            if 'old_runs' in gen_info:
                result += f"  Old/completed runs: {len(gen_info['old_runs'])}\n"
            
            # Any other useful keys
            for key in ['num_active_runs', 'next_to_give']:
                if key in gen_info:
                    result += f"  {key}: {gen_info[key]}\n"
        else:
            result += "Could not find generator-specific info. Raw structure:\n"
            for key, val in persis_info.items():
                if isinstance(val, dict):
                    result += f"  Worker {key}: {list(val.keys())[:5]}...\n"
                else:
                    result += f"  Worker {key}: {type(val)}\n"
        
        return result
        
    except Exception as e:
        return f"ERROR reading persis_info: {str(e)}"


async def get_aposmm_options_tool() -> str:
    """Return APOSMM configuration options and guidance"""
    log_message("📚 Retrieving APOSMM options...")
    guidance = """
=== APOSMM Configuration Options ===

""" + APOSMM_OPTIONS_SCHEMA + """

=== TUNING STRATEGIES (in order of effectiveness) ===

1. **NARROW THE BOUNDS** (MOST EFFECTIVE):
   Look at read_optimization_results output for "SUGGESTED BOUNDS" and 
   update lb/ub to focus on the promising region. This is usually the 
   single most impactful change.

2. INCREASE SAMPLING: Raise initial_sample_size to explore more.

3. MORE PARALLEL RUNS: Increase max_active_runs.

4. ADJUST SPACING: Increase nu to spread out local opt starting points.

=== USING H0 (Previous Results) ===

To reuse points from a previous run, add H0 to the Ensemble:

    import numpy as np
    
    # Load previous results
    H0 = np.load("previous_run_history_....npy")
    
    # In Ensemble constructor, add H0 parameter:
    ensemble = Ensemble(
        libE_specs=libE_specs,
        gen_specs=gen_specs,
        sim_specs=sim_specs,
        alloc_specs=alloc_specs,
        exit_criteria=exit_criteria,
        H0=H0,  # <-- Add this line
    )

=== FORBIDDEN (CHEATING) ===

DO NOT loosen tolerances (xatol, fatol, xtol_abs, ftol_abs).
DO NOT randomly try different optimizers without reason.
"""
    return guidance


def setup_work_directory(scripts_dir: str) -> Path:
    """Copy scripts to working directory and archive initial version"""
    global WORK_DIR, LOG_FILE
    scripts_dir = Path(scripts_dir)
    work_dir = Path("generated_scripts")
    work_dir.mkdir(exist_ok=True)
    WORK_DIR = work_dir
    
    # Setup log file
    LOG_FILE = work_dir / "agent_log.txt"
    with open(LOG_FILE, "w") as f:
        f.write(f"=== Agent Log Started: {datetime.now()} ===\n\n")
    
    # Copy all Python files to work dir
    for script_file in scripts_dir.glob("*.py"):
        shutil.copy(script_file, work_dir)
        log_message(f"Copied: {script_file.name}")
    
    # Start archive for initial scripts
    start_new_archive("copied_scripts")
    archive_current_scripts()
    
    return work_dir


class LoggingCallback:
    """Callback to log agent interactions"""
    
    def __init__(self):
        self.step = 0
    
    def on_tool_start(self, tool_name: str, tool_input: dict):
        self.step += 1
        log_message(f"🔧 Step {self.step}: Using tool '{tool_name}'")
        log_ai_interaction("TOOL_CALL", f"{tool_name}: {json.dumps(tool_input, default=str)[:500]}")
    
    def on_tool_end(self, output: str):
        log_ai_interaction("TOOL_RESULT", output[:1000])
    
    def on_agent_action(self, action: str):
        log_ai_interaction("AGENT_THINKING", action)


async def main():
    global WORK_DIR
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Advanced agent to run, fix, and tune libEnsemble APOSMM scripts")
    parser.add_argument("--scripts", required=True, help="Directory containing scripts to run")
    parser.add_argument("--max-iterations", type=int, default=15, 
                       help="Maximum agent iterations (default: 15)")
    args = parser.parse_args()
    
    # Setup working directory
    WORK_DIR = setup_work_directory(args.scripts)
    log_message(f"Working directory: {WORK_DIR}")
    log_message(f"Log file: {LOG_FILE}")
    
    # Detect run script
    run_scripts = list(WORK_DIR.glob("run_*.py"))
    if not run_scripts:
        log_message("Error: No run_*.py script found")
        return
    run_script_name = run_scripts[0].name
    
    # Create tools
    run_tool = StructuredTool(
        name="run_script",
        description="""Run a Python script. Returns:
- SUCCESS if script ran and achieved its goal (APOSMM found enough minima)
- OPTIMIZATION_NEEDED if script ran but APOSMM didn't find enough minima
- FAILED if script had errors
- STOPPED if maximum attempts reached (you must stop)""",
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
        description="Write content to a file. Use this to fix scripts or adjust APOSMM configuration.",
        args_schema=WriteFileInput,
        coroutine=write_file_tool
    )
    
    list_tool = StructuredTool(
        name="list_files",
        description="List all Python, .npy, and .pickle files in the working directory.",
        args_schema=ListFilesInput,
        coroutine=list_files_tool
    )
    
    read_opt_results_tool = StructuredTool(
        name="read_optimization_results",
        description="""Read the latest .npy optimization results and get SUGGESTED BOUNDS.
Shows: points evaluated, f statistics, local minima found, sampled ranges, and 
SUGGESTED BOUNDS based on where the best points are located.
ALWAYS use this after OPTIMIZATION_NEEDED before making changes.""",
        args_schema=ReadOptResultsInput,
        coroutine=read_optimization_results_tool
    )
    
    read_persis_tool = StructuredTool(
        name="read_persis_info",
        description="""Read the .pickle file with optimization run details.
Shows which runs completed and how many evaluations each took.""",
        args_schema=ReadPersisInfoInput,
        coroutine=read_persis_info_tool
    )
    
    aposmm_options_tool = StructuredTool(
        name="get_aposmm_options",
        description="""Get APOSMM configuration options, tuning strategies, and H0 usage.
IMPORTANT: Do NOT loosen tolerances - this is cheating.""",
        args_schema=GetApossmmOptionsInput,
        coroutine=get_aposmm_options_tool
    )
    
    tools = [run_tool, read_tool, write_tool, list_tool, 
             read_opt_results_tool, read_persis_tool, aposmm_options_tool]
    
    # Create agent
    llm = ChatOpenAI(
        model=MODEL,
        temperature=0,
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    agent = create_agent(llm, tools)
    
    # Give agent the goal
    goal = f"""Your goal is to successfully run the APOSMM optimization script '{run_script_name}' 
and ensure it finds enough local minima.

WORKFLOW:
1. Run the script
2. If SUCCESS: Report and stop
3. If FAILED (errors): Read the script, fix errors, run again
4. If OPTIMIZATION_NEEDED (not enough minima):
   a. FIRST: Use read_optimization_results to analyze where good points are
   b. Look at the SUGGESTED BOUNDS in the output
   c. Modify the script to narrow lb/ub to focus on the promising region
   d. Optionally add H0 to reuse previous points (see get_aposmm_options)
   e. Run again
5. If STOPPED: You've used all attempts. Report final status and stop.

CRITICAL RULES:
- Maximum {MAX_OPTIMIZATION_ATTEMPTS} optimization attempts total
- ALWAYS analyze results BEFORE making changes (don't guess randomly)
- The MOST EFFECTIVE change is narrowing bounds based on where best points are
- Do NOT loosen tolerances (cheating)
- Do NOT randomly try different optimizers

Start by running '{run_script_name}'."""

    print("="*60)
    print("AGENT GOAL:")
    print(goal)
    print("="*60)
    log_message("Starting autonomous agent...")
    log_ai_interaction("GOAL", goal)
    
    try:
        result = await agent.ainvoke({
            "messages": [("user", goal)]
        })
        
        # Log final messages
        for msg in result.get("messages", []):
            if hasattr(msg, 'content'):
                role = type(msg).__name__
                log_ai_interaction(role, str(msg.content)[:2000])
        
        print("\n" + "="*60)
        print("AGENT COMPLETED")
        print("="*60)
        print("\nFinal response:")
        print(result["messages"][-1].content)
        
        log_message("Agent completed")
        log_message(f"Full log saved to: {LOG_FILE}")
        
    except Exception as e:
        log_message(f"Agent error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
