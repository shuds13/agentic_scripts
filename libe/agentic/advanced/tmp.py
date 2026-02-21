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


def log_ai_interaction(role: str, content: str, tool_calls=None):
    """Log AI interactions to the log file - full content, no truncation"""
    if LOG_FILE:
        timestamp = datetime.now().strftime("%H:%M:%S")
        with open(LOG_FILE, "a") as f:
            f.write(f"\n[{timestamp}] === {role.upper()} ===\n")
            if content:
                f.write(content + "\n")
            if tool_calls:
                f.write("Tool calls:\n")
                for tc in tool_calls:
                    f.write(f"  - {tc.get('name', 'unknown')}: {json.dumps(tc.get('args', {}), default=str)}\n")


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


class AddH0ToScriptInput(BaseModel):
    script_name: str = Field(description="Name of the run script to modify (e.g., 'run_example.py')")


class ModifyBoundsInput(BaseModel):
    script_name: str = Field(description="Name of the run script to modify (e.g., 'run_example.py')")
    lb: list = Field(description="New lower bounds as a list (e.g., [-1.0, 0.5])")
    ub: list = Field(description="New upper bounds as a list (e.g., [0.0, 1.5])")


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


def archive_run_output(run_info: str, is_failure: bool = False):
    """Archive run output to the current archive (same dir as the scripts).
    
    Files are COPIED (not moved) so they remain available for H0 in subsequent runs.
    """
    if CURRENT_ARCHIVE is None:
        return
    
    archive_dir = WORK_DIR / "versions" / CURRENT_ARCHIVE
    output_dir = archive_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save run info
    info_file = "error.txt" if is_failure else "run_info.txt"
    (output_dir / info_file).write_text(run_info)
    
    # Archive configured items (COPY, don't move - files needed for H0)
    for item in ARCHIVE_ITEMS:
        item_path = WORK_DIR / item
        
        if item_path.exists() and item_path.is_dir():
            shutil.copytree(str(item_path), str(output_dir / item), dirs_exist_ok=True)
        else:
            for filepath in WORK_DIR.glob(item):
                if filepath.is_file():
                    shutil.copy(str(filepath), str(output_dir / filepath.name))
    
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
        log_message(msg)
        return msg
    
    script_path = WORK_DIR / script_name
    timeout_seconds = 300
    
    if not script_path.exists():
        msg = f"ERROR: Script '{script_name}' not found in {WORK_DIR}"
        log_message(msg)
        return msg
    
    log_message(f"Running script: {script_name}...")
    
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
                log_message("SUCCESS: APOSMM found enough minima!")
                archive_run_output(f"SUCCESS\n{combined_output}", is_failure=False)
                msg = f"SUCCESS: Script ran and APOSMM found enough minima!\nOutput:\n{result.stdout}"
            elif "APOSMM_NOT_SUCCEEDED" in result.stdout:
                OPTIMIZATION_ATTEMPTS += 1
                remaining = MAX_OPTIMIZATION_ATTEMPTS - OPTIMIZATION_ATTEMPTS
                log_message(f"APOSMM did not find enough minima (attempt {OPTIMIZATION_ATTEMPTS}/{MAX_OPTIMIZATION_ATTEMPTS})")
                archive_run_output(f"OPTIMIZATION_NEEDED (attempt {OPTIMIZATION_ATTEMPTS})\n{combined_output}", is_failure=False)
                
                msg = f"OPTIMIZATION_NEEDED: Script ran but APOSMM did not find enough minima.\n"
                msg += f"Attempt {OPTIMIZATION_ATTEMPTS}/{MAX_OPTIMIZATION_ATTEMPTS} - Remaining attempts: {remaining}\n"
                msg += f"Output:\n{result.stdout}\n\n"
                
                if remaining > 0:
                    msg += "NEXT STEPS:\n"
                    msg += "1. Use read_optimization_results to analyze the data\n"
                    msg += "2. Use get_aposmm_options to see available tuning options\n"
                    msg += "3. Decide what changes might help (bounds, optimizer, H0, etc.)\n"
                    msg += "4. Modify the script and run again\n"
                    msg += "\nIMPORTANT: Base your changes on ANALYSIS of results, not random guessing."
                else:
                    msg += "\nNO ATTEMPTS REMAINING. You must stop now."
            else:
                log_message("Script completed (no APOSMM status detected)")
                archive_run_output(f"COMPLETED\n{combined_output}", is_failure=False)
                msg = f"SUCCESS: Script ran successfully.\nOutput:\n{result.stdout[:1000]}"
        else:
            error_msg = f"Return code {result.returncode}\nStderr: {result.stderr}\nStdout: {result.stdout}"
            log_message(f"Script failed with return code {result.returncode}")
            if result.stderr:
                error_summary = result.stderr.strip().split('\n')[-1]
                log_message(f"   Error: {error_summary}")
            # Archive the failed run output
            archive_run_output(error_msg, is_failure=True)
            msg = f"FAILED: Script failed with return code {result.returncode}\n\nStderr:\n{result.stderr}\n\nStdout:\n{result.stdout[:500]}"
        
        return msg
        
    except subprocess.TimeoutExpired:
        msg = f"ERROR: Script timed out after {timeout_seconds} seconds"
        log_message(msg)
        return msg
    except Exception as e:
        msg = f"ERROR: {str(e)}"
        log_message(msg)
        return msg


async def read_file_tool(filepath: str) -> str:
    """Read a file and return its contents"""
    log_message(f"Reading file: {filepath}")
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
    log_message(f"Writing file: {filepath}")
    file_path = WORK_DIR / filepath
    
    try:
        file_path.write_text(content)
        # Start a new archive for this fixed version
        start_new_archive("script_fix")
        archive_current_scripts()
        log_message(f"Wrote {len(content)} characters to {filepath}")
        return f"SUCCESS: Wrote {len(content)} characters to {filepath}"
    except Exception as e:
        msg = f"ERROR writing file: {str(e)}"
        log_message(msg)
        return msg


async def list_files_tool() -> str:
    """List all Python files in the working directory"""
    log_message("Listing files...")
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
    log_message("Analyzing optimization results...")
    try:
        npy_file = find_latest_file("*.npy")
        if npy_file is None:
            return "ERROR: No .npy result files found. Run the script first."
        
        H = np.load(npy_file)
        
        result = f"=== Optimization Results from {npy_file.name} ===\n"
        result += f"(To use as H0, add: H0 = np.load(\"{npy_file.name}\") before Ensemble creation)\n\n"
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
    log_message("Reading persistent generator info...")
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
    log_message("Retrieving APOSMM options...")
    guidance = """
=== APOSMM Configuration Options ===

""" + APOSMM_OPTIONS_SCHEMA + """

=== TUNING STRATEGIES ===

Analyze the optimization results and consider what changes might help:

1. **BOUNDS (lb, ub)**: If best points cluster in a region, consider focusing there.
   
2. **OPTIMIZER (localopt_method)**: Different optimizers suit different problems.
   Options: scipy_Nelder-Mead, LN_BOBYQA, LD_MMA, etc.

3. **SPACING (nu)**: Controls how far apart local opt starting points must be.

4. **USE H0**: Pass previous results to the next run to build on prior work.

NOTE: max_active_runs should not exceed num_workers.

=== USING H0 WITH APOSMM (Class-based API) ===

To reuse points from a previous run, you must add TWO things:

STEP 1: Add this code block AFTER gen_specs is defined but BEFORE the Ensemble is created:

    # Load previous results - use the .npy filename from read_optimization_results
    H0 = np.load("run_example_history_length=302_evals=300_workers=4.npy")
    
    # Recompute x_on_cube for the new bounds
    lb = gen_specs.user["lb"]
    ub = gen_specs.user["ub"]
    H0["x_on_cube"] = (H0["x"] - lb) / (ub - lb)

STEP 2: Add H0=H0 to the Ensemble constructor:

    ensemble = Ensemble(
        libE_specs=libE_specs,
        gen_specs=gen_specs,
        sim_specs=sim_specs,
        alloc_specs=alloc_specs,
        exit_criteria=exit_criteria,
        executor=exctr,
        H0=H0,  # <-- Must add this
    )

COMMON MISTAKE: Adding H0=H0 without the np.load() code above. 
You MUST do BOTH steps or you'll get "Initial points in H have never been given"

=== FORBIDDEN (CHEATING) ===

DO NOT loosen tolerances (xatol, fatol, xtol_abs, ftol_abs).
DO NOT randomly try different optimizers without reason.

=== COMMON MISTAKES ===

1. Adding H0=H0 to Ensemble without the np.load() line - causes "Initial points never given" error
2. Forgetting to recompute x_on_cube when bounds change
3. Writing incomplete scripts (summaries instead of full code)
"""
    return guidance


async def add_h0_to_script_tool(script_name: str) -> str:
    """Add H0 (previous results) loading to a script"""
    import re
    
    script_path = WORK_DIR / script_name
    if not script_path.exists():
        return f"ERROR: Script '{script_name}' not found"
    
    # Find the latest .npy file
    npy_file = find_latest_file("*.npy")
    if npy_file is None:
        return "ERROR: No .npy file found to use as H0. Run the script first."
    
    log_message(f"Adding H0 from {npy_file.name} to {script_name}")
    
    content = script_path.read_text()
    changes_made = []
    
    # Check if our proper H0 code already exists (uses glob to find latest .npy)
    has_proper_h0 = "_npy_files = sorted(glob.glob" in content and "_completed = _H0_all" in content
    
    if has_proper_h0:
        # Our proper code already exists, nothing to do
        changes_made.append("H0 code already present (uses glob for latest file)")
    else:
        # Need to add H0 loading code
        # Find the Ensemble creation line
        ensemble_match = re.search(r'(\n\s*)(ensemble\s*=\s*Ensemble\()', content)
        if not ensemble_match:
            return "ERROR: Could not find Ensemble creation in script"
        
        # H0 loading code - find latest .npy and filter to completed points within current bounds
        h0_code = '''
    # Load previous results as H0 (find latest .npy file)
    import glob
    _npy_files = sorted(glob.glob("run_example_history_*.npy"), key=lambda f: os.path.getmtime(f))
    if _npy_files:
        _H0_all = np.load(_npy_files[-1])
        _lb = gen_specs.user["lb"]
        _ub = gen_specs.user["ub"]
        
        # Filter to COMPLETED points within current bounds that have been given back to gen
        _completed = _H0_all["sim_ended"] & _H0_all["gen_informed"]
        _in_bounds = np.all((_H0_all["x"] >= _lb) & (_H0_all["x"] <= _ub), axis=1)
        _valid = _completed & _in_bounds
        H0 = _H0_all[_valid].copy()
        
        if len(H0) > 0:
            # Recompute x_on_cube for new bounds
            H0["x_on_cube"] = (H0["x"] - _lb) / (_ub - _lb)
            # Renumber sim_ids to be sequential (required by libEnsemble)
            H0["sim_id"] = np.arange(len(H0))
            print("Using", len(H0), "of", len(_H0_all), "completed points from", _npy_files[-1])
        else:
            H0 = None
            print("No completed points within current bounds, starting fresh")
    else:
        H0 = None
        print("No previous results found, starting fresh")
'''
        # Insert before Ensemble
        content = content.replace(
            ensemble_match.group(0),
            h0_code + ensemble_match.group(0)
        )
        changes_made.append(f"Added H0 loading from {npy_file.name}")
    
    # Add H0=H0 to Ensemble if not present
    if "H0=H0" not in content and "H0 = H0" not in content:
        # Find executor=exctr pattern and add H0 after it
        content = content.replace(
            "executor=exctr\n    )",
            "executor=exctr,\n        H0=H0\n    )"
        )
        changes_made.append("Added H0=H0 to Ensemble constructor")
    
    # When using H0 with persistent_aposmm_alloc, we need to:
    # 1. Add sim_ended, local_pt, local_min to persis_in (APOSMM needs these for H0)
    # 2. Set inputs to match persis_in
    # 3. Set initial_sample_size=0 (H0 IS the sample)
    persis_in_match = re.search(r'persis_in=\[([^\]]+)\]', content)
    if persis_in_match:
        persis_in_fields = persis_in_match.group(1)
        fields_to_add = []
        
        # Add required fields for H0
        for field in ["sim_ended", "local_pt", "local_min"]:
            if f'"{field}"' not in persis_in_fields and f"'{field}'" not in persis_in_fields:
                fields_to_add.append(f'"{field}"')
        
        if fields_to_add:
            new_persis_in = f'persis_in=[{persis_in_fields}, {", ".join(fields_to_add)}]'
            content = re.sub(r'persis_in=\[[^\]]+\]', new_persis_in, content)
            changes_made.append(f"Added {', '.join(fields_to_add)} to persis_in (required for H0)")
            persis_in_fields = f'{persis_in_fields}, {", ".join(fields_to_add)}'
        
        # Set inputs to match persis_in
        if "inputs=[]" in content:
            content = content.replace("inputs=[]", f"inputs=[{persis_in_fields}]")
            changes_made.append("Set inputs to match persis_in (required for H0)")
    
    # Set initial_sample_size=0 when using H0 (H0 provides the sample)
    content = re.sub(
        r'"initial_sample_size":\s*\d+',
        '"initial_sample_size": 0',
        content
    )
    changes_made.append("Set initial_sample_size=0 (H0 provides sample)")
    
    # Write the modified script
    script_path.write_text(content)
    
    # Archive
    start_new_archive("add_h0")
    archive_current_scripts()
    
    result = f"SUCCESS: Added H0 to {script_name}\n"
    result += f"Loaded {npy_file.name} with x_on_cube recalculation\n"
    for change in changes_made:
        result += f"  - {change}\n"
    
    log_message(f"Added H0: {', '.join(changes_made)}")
    return result


async def modify_bounds_tool(script_name: str, lb: list, ub: list) -> str:
    """Modify the bounds (lb, ub) in a script"""
    import re
    
    script_path = WORK_DIR / script_name
    if not script_path.exists():
        return f"ERROR: Script '{script_name}' not found"
    
    log_message(f"Modifying bounds in {script_name}: lb={lb}, ub={ub}")
    
    content = script_path.read_text()
    
    # Update lb
    lb_str = f"np.array({lb})"
    content = re.sub(
        r'"lb":\s*np\.array\(\[[^\]]+\]\)',
        f'"lb": {lb_str}',
        content
    )
    
    # Update ub
    ub_str = f"np.array({ub})"
    content = re.sub(
        r'"ub":\s*np\.array\(\[[^\]]+\]\)',
        f'"ub": {ub_str}',
        content
    )
    
    # Write the modified script
    script_path.write_text(content)
    
    # Archive
    start_new_archive("bounds_update")
    archive_current_scripts()
    
    log_message(f"Updated bounds: lb={lb}, ub={ub}")
    return f"SUCCESS: Updated bounds in {script_name}\n  lb = {lb}\n  ub = {ub}"


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
        log_message(f"Step {self.step}: Using tool '{tool_name}'")
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
        description="""Write content to a file.

CRITICAL REQUIREMENTS:
- You MUST write the COMPLETE file content, not a summary or example
- Do NOT write placeholders like "# Your existing code continues here"  
- Do NOT write just the changed parts - include EVERY line

WORKFLOW:
1. Use read_file to get the FULL current script
2. Copy ALL of that content
3. Change ONLY the specific values needed (e.g., lb, ub arrays)
4. Write back the COMPLETE script with all original code intact

If you write an incomplete file, the script will fail to run.""",
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
        description="""Read the latest .npy optimization results.
Shows: points evaluated, f statistics, local minima found, sampled x ranges, and top 5 best points.
Use this to understand the optimization landscape before making changes.""",
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
    
    add_h0_tool = StructuredTool(
        name="add_h0_to_script",
        description="""Add H0 (previous optimization results) to a script. 
ALWAYS use this tool to add H0 - do NOT write H0 code yourself.
This handles all the required setup: loading, filtering, x_on_cube, sim_id renumbering, and spec updates.""",
        args_schema=AddH0ToScriptInput,
        coroutine=add_h0_to_script_tool
    )
    
    modify_bounds_tool_obj = StructuredTool(
        name="modify_bounds",
        description="""Modify the bounds (lb, ub) in a script.
Use this to change the search bounds based on your analysis of optimization results.
This safely updates only the bounds without rewriting the entire script.""",
        args_schema=ModifyBoundsInput,
        coroutine=modify_bounds_tool
    )
    
    tools = [run_tool, read_tool, write_tool, list_tool, 
             read_opt_results_tool, read_persis_tool, aposmm_options_tool, add_h0_tool, modify_bounds_tool_obj]
    
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
   a. Use read_optimization_results to analyze the data
   b. Use get_aposmm_options to see what you can adjust
   c. Use add_h0_to_script to add previous results to the next run
   d. Use modify_bounds to change lb/ub if needed (based on your analysis)
   e. Run again
5. If STOPPED: You've used all attempts. Report final status and stop.

CRITICAL RULES:
- Maximum {MAX_OPTIMIZATION_ATTEMPTS} optimization attempts total
- ALWAYS analyze results BEFORE making changes
- Base decisions on data, not random guessing
- For H0: ALWAYS use the add_h0_to_script tool - never write H0 code yourself
- Do NOT loosen tolerances (that's cheating)

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
        
        # Log all messages with full content
        for msg in result.get("messages", []):
            role = type(msg).__name__
            content = str(msg.content) if hasattr(msg, 'content') else ""
            tool_calls = None
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                tool_calls = [{"name": tc.get("name"), "args": tc.get("args")} for tc in msg.tool_calls]
            log_ai_interaction(role, content, tool_calls)
        
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
