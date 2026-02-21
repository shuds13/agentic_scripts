#!/usr/bin/env python3
"""
Simple command-line interface for querying libEnsemble documentation.

Usage:
    python rag_query.py "What generators are available?"
    python rag_query.py --raw "How do I set bounds?"
    python rag_query.py --generator "What options does APOSMM have?"
"""

import sys
import argparse
from pathlib import Path
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Force local embedding model so loading never touches OpenAI
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load index (look in same directory as this script)
index_dir = Path(__file__).parent / "rag_index"
if not index_dir.exists():
    print(f"ERROR: RAG index not found at {index_dir}")
    print("Run: python build_index.py")
    sys.exit(1)

try:
    ctx = StorageContext.from_defaults(persist_dir=str(index_dir))
    index = load_index_from_storage(ctx)
except Exception as e:
    print(f"ERROR loading index: {e}")
    print(f"Index location: {index_dir}")
    print("Try rebuilding: python build_index.py")
    sys.exit(1)


def rag(q, mode="full", top_k=5):
    """Query the RAG index.
    
    Args:
        q: Question to ask
        mode: "full" (synthesized answer), "raw" (source chunks), "generator" (generator-focused)
        top_k: Number of chunks to retrieve
    """
    if mode == "raw":
        # Return raw source chunks
        query_engine = index.as_query_engine(
            similarity_top_k=top_k,
            response_mode="no_text"
        )
        res = query_engine.query(q)
        
        output = []
        for i, node in enumerate(res.source_nodes, 1):
            output.append(f"\n{'='*60}")
            output.append(f"Source {i} (score: {node.score:.3f})")
            output.append('='*60)
            output.append(node.text)
        return "\n".join(output)
    
    elif mode == "generator":
        # Focus on generator documentation
        enhanced_q = f"Regarding libEnsemble generators and their configuration: {q}"
        query_engine = index.as_query_engine(
            similarity_top_k=top_k,
            response_mode="compact"
        )
        res = query_engine.query(enhanced_q)
        return str(res)
    
    else:  # mode == "full"
        # Full query with LLM synthesis
        query_engine = index.as_query_engine(
            similarity_top_k=top_k,
            response_mode="compact"
        )
        res = query_engine.query(q)
        return str(res)


def main():
    parser = argparse.ArgumentParser(
        description="Query libEnsemble documentation using RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rag_query.py "What generators are available?"
  python rag_query.py --raw "How do I set bounds?"
  python rag_query.py --generator "What options does APOSMM have?"
  python rag_query.py --top-k 3 "What is sim_specs?"
        """
    )
    parser.add_argument("question", nargs="+", help="Question to ask")
    parser.add_argument("--raw", action="store_true", 
                       help="Return raw source chunks without synthesis")
    parser.add_argument("--generator", action="store_true",
                       help="Focus query on generator documentation")
    parser.add_argument("--top-k", type=int, default=5,
                       help="Number of chunks to retrieve (default: 5)")
    
    args = parser.parse_args()
    
    question = " ".join(args.question)
    
    # Determine mode
    if args.raw:
        mode = "raw"
    elif args.generator:
        mode = "generator"
    else:
        mode = "full"
    
    print(f"\nQuestion: {question}")
    print("=" * 60)
    
    result = rag(question, mode=mode, top_k=args.top_k)
    print(result)
    print()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python rag_query.py 'your question here'")
        print("Use --help for more options")
        sys.exit(1)
    
    main()
