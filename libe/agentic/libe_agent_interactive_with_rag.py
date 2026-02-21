#!/usr/bin/env python3
"""
Interactive LangChain agent for libEnsemble script generation with targeted RAG lookups.

Features:
1. Generate scripts with MCP tool
2. Interactive review with targeted RAG suggestions for each section
3. User can approve or request changes
4. Runs scripts with retry on errors (using RAG for fixes)

Requirements: pip install langchain langchain-openai mcp openai

For options: python libe_agent_interactive_with_rag.py -h
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
from langchain_core.tools import StructuredTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# Maximum retry attempts for fixing failed scripts
MAX_RETRIES = 2

# OpenAI model to use
DEFAULT_MODEL = "gpt-4o-mini"
MODEL = os.environ.get("LLM_MODEL", DEFAULT_MODEL)

# Show prompts flag
SHOW_PROMPTS = False

# Interactive mode flag
INTERACTIVE_MODE = False

# Debug RAG/LLM interactions flag
DEBUG_DUMP = True

# Files and directories to archive after each run
ARCHIVE_ITEMS = [
    "ensemble", "ensemble.log", "libE_stats.txt",
    "*.npy", "*.pickle",
]

# Default prompt if none provided
DEFAULT_PROMPT = """Create six_hump_camel APOSMM scripts:
- Executable: /home/shudson/test_mcp/script-creator/six_hump_camel/six_hump_camel.x
- Input: /home/shudson/test_mcp/script-creator/six_hump_camel/input.txt
- Template vars: X0, X1
- 4 workers, 100 sims.
- The output file for each simulation is output.txt
- The bounds should be 0,1 and -1,2 for X0 and X1 respectively"""

# Templates for reviewing specific sections
EXTRACT_GENERATOR_TEMPLATE = """Look at these scripts and extract the generator configuration:

{scripts_text}

Show me:
- Generator name (e.g., APOSMM, uniform_sampling)
- Generator options from gen_specs['user'] (including bounds lb/ub, if present)

Be concise. Show only the configuration, not explanations."""

EXTRACT_OBJECTIVE_TEMPLATE = """Look at these scripts and find where the objective value is set:

{scripts_text}

Priority order:
1. If there's a function called set_objective_value() or set_objective() → show ONLY that function
2. Otherwise, show where the objective is computed (e.g., in the simulation function)

Show only the most relevant function code, be concise."""

# RAG query templates for targeted documentation lookup
RAG_GENERATOR_QUERY_TEMPLATE = """Here is the generator configuration from generated libEnsemble scripts:

{generator_config}

Based on this configuration, what should I check or consider? Are the parameter values appropriate? What are common issues with these settings?"""

RAG_OBJECTIVE_QUERY_TEMPLATE = """Here is the objective function from generated libEnsemble scripts:

{objective_code}

Based on this code, what should I check? Are there better patterns for reading output files and setting objective values? What are common issues?"""

RAG_ERROR_QUERY_TEMPLATE = """LibEnsemble scripts failed with this error:

{error_preview}

What could cause this error and how should it be fixed?"""

# Template for refinement with user feedback
REFINE_WITH_FEEDBACK_TEMPLATE = """Here are the generated scripts:

{scripts_text}

User feedback: {feedback}

Update the scripts based on this feedback.
Return ONLY the scripts in the format (=== filename === followed by code).
Do NOT add explanations or wrap in markdown."""

# Template for fixing failed scripts with RAG help
FIX_WITH_RAG_TEMPLATE = """These scripts failed with the following error:

{error_msg}

Here are the current scripts (main run script is {run_script_name}):

{scripts_text}

Documentation context:
{rag_context}

Fix the scripts to resolve this error.
DO NOT make any other changes or improvements.
Return ALL scripts in the EXACT SAME FORMAT (=== filename === followed by raw Python code).
DO NOT merge or consolidate files - keep the same file structure.
DO NOT wrap in markdown or add explanations."""


# Global MCP sessions
generator_session = None
rag_session = None


def print_prompt(stage_name, prompt_text):
    """Print a prompt with formatting if SHOW_PROMPTS is enabled"""
    if SHOW_PROMPTS:
        slen = 15
        print(f"\n{'='*slen} PROMPT TO AI ({stage_name}) {'='*slen}")
        print(prompt_text)
        print(f"{'='*slen} END AI PROMPT ({stage_name}) {'='*slen}\n")


def print_section(title, content, width=70):
    """Print a formatted section"""
    print(f"\n{'='*width}")
    print(f"  {title}")
    print('='*width)
    print(content)
    print('='*width)


def clean_html(text):
    """Remove HTML tags from text (in case RAG returns some)"""
    import re
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Clean up extra whitespace
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()




def get_user_input(prompt_text):
    """Get user input in interactive mode"""
    if not INTERACTIVE_MODE:
        return "y"
    
    response = input(f"\n{prompt_text} (y/edit/stop): ").strip().lower()
    return response


def dump_debug_files(rag_query=None, rag_response=None, llm_prompt=None, stage=""):
    """Dump RAG query/response and LLM prompt to debug files"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if rag_query and rag_response:
        rag_dump_file = Path("generated_scripts") / f"debug_rag_{stage}_{timestamp}.txt"
        rag_dump_file.parent.mkdir(exist_ok=True)
        rag_dump_file.write_text(f"""RAG QUERY:
{'='*70}
{rag_query}

RAG RESPONSE:
{'='*70}
{rag_response}
""")
        print(f"  (RAG query/response dumped to {rag_dump_file})")
    
    if llm_prompt:
        llm_dump_file = Path("generated_scripts") / f"debug_llm_{stage}_{timestamp}.txt"
        llm_dump_file.parent.mkdir(exist_ok=True)
        llm_dump_file.write_text(f"""LLM PROMPT ({stage}):
{'='*70}
{llm_prompt}
""")
        print(f"  (LLM prompt dumped to {llm_dump_file})")


async def call_generator_tool(**kwargs):
    """Wrapper to call the generator MCP tool"""
    # Block custom_set_objective - AI always gets it wrong
    if 'custom_set_objective' in kwargs:
        del kwargs['custom_set_objective']
    if 'set_objective_code' in kwargs:
        del kwargs['set_objective_code']
    
    result = await generator_session.call_tool("CreateLibEnsembleScripts", kwargs)
    return result.content[0].text if result.content else "Scripts created"


async def call_rag_tool(question, top_k=3):
    """Wrapper to call RAG tool (returns raw chunks)"""
    try:
        result = await rag_session.call_tool("query_libe_docs_raw", {
            "question": question,
            "top_k": top_k
        })
        return result.content[0].text if result.content else "No docs found"
    except Exception as e:
        print(f"\n✗ RAG query failed: {e}", file=sys.stderr)
        return f"[RAG unavailable: {e}]"


async def run_mcp_generator(agent, user_prompt):
    """Stage 1: Run the MCP script generator"""
    print("\n" + "="*70)
    print("  STAGE 1: Generating Scripts")
    print("="*70)
    
    print_prompt("MCP Generator", user_prompt)
    
    result = await agent.ainvoke({
        "messages": [("user", user_prompt)]
    })
    
    # Find the tool result (MCP-generated scripts)
    for msg in result["messages"]:
        if hasattr(msg, "type") and msg.type == "tool":
            return msg.content
    
    return None


async def interactive_review_with_rag(agent, scripts_text):
    """Stage 2: Interactive review with targeted RAG suggestions"""
    if not INTERACTIVE_MODE:
        print("\n[Skipping interactive review - use --interactive to enable]")
        return scripts_text
    
    print("\n" + "="*70)
    print("  STAGE 2: Interactive Script Review (with RAG suggestions)")
    print("="*70)
    
    current_scripts = scripts_text
    
    # ==========================================
    # Review 1: Generator configuration
    # ==========================================
    print("\n--- Generator Configuration ---")
    gen_prompt = EXTRACT_GENERATOR_TEMPLATE.format(scripts_text=current_scripts)
    print_prompt("Extract Generator", gen_prompt)
    
    gen_result = await agent.ainvoke({
        "messages": [("user", gen_prompt)]
    })
    gen_info = gen_result["messages"][-1].content
    print(gen_info)
    
    # Query RAG with the actual generator configuration
    print(f"\n[Querying documentation with this generator configuration...]")
    
    rag_query = RAG_GENERATOR_QUERY_TEMPLATE.format(generator_config=gen_info)
    rag_suggestions = await call_rag_tool(rag_query, top_k=3)
    
    # Show RAG suggestions (condensed and cleaned)
    if rag_suggestions and len(rag_suggestions) > 200:
        print("\n📚 Documentation Suggestions:")
        sources = rag_suggestions.split("--- Source")
        for i, src in enumerate(sources[1:3], 1):  # Skip first empty split, show 2 sources
            if src.strip():
                # Clean HTML and show preview
                clean_src = clean_html(src)
                preview = clean_src[:500].strip()
                if preview:
                    print(f"\n  Source {i}:")
                    print(f"  {preview}...")
                    print()
    
    response = get_user_input("Does this look correct?")
    
    if response == 'stop':
        print("✗ Stopped by user")
        return None
    elif response == 'edit':
        feedback = input("What would you like to change? ")
        refine_prompt = REFINE_WITH_FEEDBACK_TEMPLATE.format(
            scripts_text=current_scripts,
            feedback=feedback
        )
        print_prompt("Refine", refine_prompt)
        
        refine_result = await agent.ainvoke({
            "messages": [("user", refine_prompt)]
        })
        current_scripts = refine_result["messages"][-1].content
        
        # Clean up
        current_scripts = re.sub(r'```python\n', '', current_scripts)
        current_scripts = re.sub(r'```\n?', '', current_scripts)
        if '===' in current_scripts:
            start = current_scripts.find('===')
            current_scripts = current_scripts[start:]
        
        print("✓ Updated")
    else:
        print("✓ Approved")
    
    # ==========================================
    # Review 2: Set objective function
    # ==========================================
    print("\n--- Set Objective Function ---")
    obj_prompt = EXTRACT_OBJECTIVE_TEMPLATE.format(scripts_text=current_scripts)
    print_prompt("Extract Objective", obj_prompt)
    
    obj_result = await agent.ainvoke({
        "messages": [("user", obj_prompt)]
    })
    obj_info = obj_result["messages"][-1].content
    print(obj_info)
    
    # Query RAG with the actual objective function code
    print("\n[Querying documentation with this objective function...]")
    rag_query = RAG_OBJECTIVE_QUERY_TEMPLATE.format(objective_code=obj_info)
    rag_suggestions = await call_rag_tool(rag_query, top_k=3)
    
    # Show RAG suggestions (condensed and cleaned)
    if rag_suggestions and len(rag_suggestions) > 200:
        print("\n📚 Documentation Suggestions:")
        sources = rag_suggestions.split("--- Source")
        for i, src in enumerate(sources[1:3], 1):  # Skip first empty split, show 2 sources
            if src.strip():
                # Clean HTML and show preview
                clean_src = clean_html(src)
                preview = clean_src[:500].strip()
                if preview:
                    print(f"\n  Source {i}:")
                    print(f"  {preview}...")
                    print()
    
    response = get_user_input("Does this look correct?")
    
    if response == 'stop':
        print("✗ Stopped by user")
        return None
    elif response == 'edit':
        feedback = input("What would you like to change? ")
        refine_prompt = REFINE_WITH_FEEDBACK_TEMPLATE.format(
            scripts_text=current_scripts,
            feedback=feedback
        )
        print_prompt("Refine", refine_prompt)
        
        refine_result = await agent.ainvoke({
            "messages": [("user", refine_prompt)]
        })
        current_scripts = refine_result["messages"][-1].content
        
        # Clean up
        current_scripts = re.sub(r'```python\n', '', current_scripts)
        current_scripts = re.sub(r'```\n?', '', current_scripts)
        if '===' in current_scripts:
            start = current_scripts.find('===')
            current_scripts = current_scripts[start:]
        
        print("✓ Updated")
    else:
        print("✓ Approved")
    
    print("\n✓ All sections approved")
    return current_scripts


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
    """Move run outputs to archive"""
    output_dir = Path(output_dir)
    run_output_dir = output_dir / "versions" / archive_name / "output"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    if error_msg:
        (run_output_dir / "error.txt").write_text(error_msg)
    
    for item in ARCHIVE_ITEMS:
        item_path = output_dir / item
        if item_path.exists() and item_path.is_dir():
            shutil.move(str(item_path), str(run_output_dir / item))
        else:
            for filepath in output_dir.glob(item):
                if filepath.is_file():
                    shutil.move(str(filepath), str(run_output_dir / filepath.name))


def detect_run_script(directory):
    """Find the run script in directory"""
    directory = Path(directory)
    run_scripts = list(directory.glob("run_*.py"))
    return run_scripts[0].name if run_scripts else None


def run_generated_scripts(output_dir, run_script_name):
    """Stage 3: Run the generated scripts"""
    print("\n" + "="*70)
    print("  STAGE 3: Running Scripts")
    print("="*70)
    
    output_dir = Path(output_dir)
    run_script = output_dir / run_script_name
    
    if not run_script.exists():
        return False, f"{run_script_name} not found"
    
    result = subprocess.run(
        ["python", run_script_name],
        cwd=output_dir,
        capture_output=True,
        text=True,
        timeout=300
    )
    
    if result.returncode == 0:
        print("✓ Scripts ran successfully")
        return True, None
    else:
        error_msg = f"Return code {result.returncode}\nStderr: {result.stderr}\nStdout: {result.stdout}"
        print(f"✗ Scripts failed with return code {result.returncode}")
        if result.stderr:
            print(f"Error output:\n{result.stderr[:500]}")
        return False, error_msg


async def fix_scripts_with_rag(agent, scripts_text, error_msg, run_script_name):
    """Fix scripts using RAG documentation"""
    print("\n" + "="*70)
    print("  STAGE 4: Fixing Scripts (with documentation lookup)")
    print("="*70)
    
    # Query RAG about the error
    print("[Querying documentation about this error...]")
    error_preview = error_msg[:300]
    rag_query = RAG_ERROR_QUERY_TEMPLATE.format(error_preview=error_preview)
    rag_context = await call_rag_tool(rag_query, top_k=3)
    
    # Clean HTML from RAG context
    rag_context = clean_html(rag_context)
    
    fix_prompt = FIX_WITH_RAG_TEMPLATE.format(
        error_msg=error_msg[:1000],
        scripts_text=scripts_text,
        run_script_name=run_script_name,
        rag_context=rag_context[:1000]  # Limit RAG context
    )
    
    # Dump debug files if enabled
    if DEBUG_DUMP:
        dump_debug_files(
            rag_query=rag_query,
            rag_response=rag_context,
            llm_prompt=fix_prompt,
            stage="error_fix"
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


def find_mcp_server(server_type, user_provided_path=None):
    """Find MCP server file"""
    if server_type == "generator":
        env_var = "GENERATOR_MCP_SERVER"
        defaults = [
            Path(__file__).parent.parent / "mcp_server.mjs",
            Path.cwd() / "mcp_server.mjs"
        ]
    else:  # rag
        env_var = "RAG_MCP_SERVER"
        defaults = [
            Path(__file__).parent.parent / "rag" / "mcp_server.py",
            Path.cwd() / "rag" / "mcp_server.py"
        ]
    
    search_locations = []
    if user_provided_path:
        search_locations.append(Path(user_provided_path))
    
    env_path = os.environ.get(env_var)
    if env_path:
        search_locations.append(Path(env_path))
    
    search_locations.extend(defaults)
    
    for location in search_locations:
        if location.exists():
            return location
    
    print(f"Error: Cannot find {server_type} MCP server")
    print(f"Searched: {', '.join(str(loc) for loc in search_locations)}")
    print(f"Set {env_var} environment variable or use --{server_type}-mcp-server flag")
    sys.exit(1)


async def main():
    global generator_session, rag_session, SHOW_PROMPTS, INTERACTIVE_MODE, DEBUG_DUMP
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate and run libEnsemble scripts with interactive review and targeted RAG lookups",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode with RAG suggestions
  python libe_agent_interactive_with_rag.py --interactive

  # Review existing scripts interactively with RAG
  python libe_agent_interactive_with_rag.py --interactive --scripts my_scripts/

  # Non-interactive (still uses RAG for error fixes)
  python libe_agent_interactive_with_rag.py --prompt "Create APOSMM scripts..."
        """
    )
    parser.add_argument("--interactive", action="store_true", 
                       help="Enable interactive review mode with RAG suggestions")
    parser.add_argument("--scripts", help="Use existing scripts from directory")
    parser.add_argument("--prompt", help="Prompt for script generation")
    parser.add_argument("--prompt-file", help="Read prompt from file")
    parser.add_argument("--show-prompts", action="store_true", help="Print prompts sent to AI")
    parser.add_argument("--debug-dump", action="store_true", 
                       help="Dump RAG queries/responses and LLM prompts to debug files")
    parser.add_argument("--generator-mcp-server", help="Path to generator mcp_server.mjs")
    parser.add_argument("--rag-mcp-server", help="Path to RAG mcp_server.py")
    parser.add_argument("--generate-only", action="store_true", 
                       help="Only generate/review scripts, don't run them")
    args = parser.parse_args()
    
    SHOW_PROMPTS = args.show_prompts
    INTERACTIVE_MODE = args.interactive
    DEBUG_DUMP = args.debug_dump or DEBUG_DUMP  # Flag overrides, or keep manual setting
    
    # Get prompt (only used for generation)
    if args.scripts:
        user_prompt = None
    elif args.prompt_file:
        user_prompt = Path(args.prompt_file).read_text()
    elif args.prompt:
        user_prompt = args.prompt
    else:
        user_prompt = DEFAULT_PROMPT
    
    output_dir = "generated_scripts"
    archive_counter = 1
    
    # Find MCP servers
    generator_mcp = find_mcp_server("generator", args.generator_mcp_server)
    rag_mcp = find_mcp_server("rag", args.rag_mcp_server)
    
    print(f"Generator MCP: {generator_mcp}")
    print(f"RAG MCP: {rag_mcp}")
    
    # Connect to both MCP servers
    generator_params = StdioServerParameters(command="node", args=[str(generator_mcp)])
    rag_params = StdioServerParameters(command="python", args=[str(rag_mcp)])
    
    async with stdio_client(generator_params) as (gen_read, gen_write), \
               stdio_client(rag_params) as (rag_read, rag_write):
        
        async with ClientSession(gen_read, gen_write) as gen_session, \
                   ClientSession(rag_read, rag_write) as r_session:
            
            await gen_session.initialize()
            await r_session.initialize()
            
            generator_session = gen_session
            rag_session = r_session
            
            print("✓ Connected to both MCP servers")
            
            # Get tools from generator server
            gen_tools = await gen_session.list_tools()
            
            # Create LangChain tool (only generator - RAG called directly)
            lc_tool = StructuredTool(
                name="CreateLibEnsembleScripts",
                description=gen_tools.tools[0].description,
                args_schema=gen_tools.tools[0].inputSchema,
                coroutine=call_generator_tool
            )
            
            # Create agent
            try:
                llm = ChatOpenAI(
                    model=MODEL,
                    temperature=0,
                    base_url=os.environ.get("OPENAI_BASE_URL"),
                )
                agent = create_agent(llm, [lc_tool])
            except Exception as e:
                print(f"\n✗ Error creating agent: {e}")
                print("\nPossible causes:")
                print("  - OPENAI_API_KEY not set or invalid")
                print("  - Network connection issue")
                print(f"\nCurrent OPENAI_API_KEY: {'SET' if os.environ.get('OPENAI_API_KEY') else 'NOT SET'}")
                sys.exit(1)
            
            print("✓ Agent initialized")
            
            # Determine if we're using existing scripts
            if args.scripts:
                print(f"\nLoading existing scripts from: {args.scripts}")
                scripts_dir = Path(args.scripts)
                current_scripts = ""
                for script_file in sorted(scripts_dir.glob("*.py")):
                    current_scripts += f"=== {script_file.name} ===\n{script_file.read_text()}\n\n"
                run_script_name = detect_run_script(args.scripts)
                if not run_script_name:
                    print("Error: No run_*.py script found")
                    return
                
                # Interactive review with RAG
                current_scripts = await interactive_review_with_rag(agent, current_scripts)
                if current_scripts is None:
                    return  # User stopped
                save_scripts(current_scripts, output_dir, f"{archive_counter}_reviewed")
                archive_counter += 1
            else:
                # Generate new scripts
                if not user_prompt:
                    print("Error: Need --prompt or --prompt-file for generation")
                    return
                    
                run_script_name = "run_libe.py"
                scripts_text = await run_mcp_generator(agent, user_prompt)
                
                if not scripts_text or "===" not in scripts_text:
                    print("✗ No scripts generated")
                    return
                
                save_scripts(scripts_text, output_dir, f"{archive_counter}_generated")
                archive_counter += 1
                
                # Interactive review with RAG
                current_scripts = await interactive_review_with_rag(agent, scripts_text)
                if current_scripts is None:
                    return  # User stopped
                save_scripts(current_scripts, output_dir, f"{archive_counter}_reviewed")
                archive_counter += 1
            
            # Stop here if generate-only
            if args.generate_only:
                print("\n✓ Generation/review complete (--generate-only mode)")
                return
            
            # Run scripts with retry loop
            for attempt in range(MAX_RETRIES + 1):
                current_archive = f"{archive_counter-1}_attempt_{attempt}" if attempt > 0 else f"{archive_counter-1}_reviewed"
                success, error_msg = run_generated_scripts(output_dir, run_script_name)
                
                if success:
                    print(f"\n{'='*70}")
                    print("  ✓ SUCCESS - Scripts completed successfully!")
                    print('='*70)
                    break
                
                archive_run_outputs(output_dir, current_archive, error_msg)
                
                if attempt < MAX_RETRIES:
                    print(f"\n{'='*70}")
                    print(f"  Retry attempt {attempt + 1}/{MAX_RETRIES}")
                    print('='*70)
                    
                    # Fix with RAG
                    current_scripts = await fix_scripts_with_rag(
                        agent, current_scripts, error_msg, run_script_name
                    )
                    archive_counter += 1
                    save_scripts(current_scripts, output_dir, f"{archive_counter}_fix_{attempt+1}")
                else:
                    print(f"\n{'='*70}")
                    print(f"  ✗ FAILED after {MAX_RETRIES} retry attempts")
                    print('='*70)


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
