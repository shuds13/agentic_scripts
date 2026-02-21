#!/usr/bin/env python3
"""
Build RAG index from libEnsemble documentation.

Usage:
    python build_index.py [docs_directory]

If docs_directory is not provided, tries:
1. libensemble.readthedocs.io/en/latest
2. ../libensemble/docs
3. Downloads docs if none found

Requirements:
    pip install beautifulsoup4 lxml
"""

import sys
import re
from pathlib import Path
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from bs4 import BeautifulSoup

# Configure embedding model (local, no API needed)
print("Loading embedding model...")
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Find documentation directory
docs_dir = None
if len(sys.argv) > 1:
    docs_dir = Path(sys.argv[1])
else:
    # Try common locations
    candidates = [
        Path("libensemble.readthedocs.io/en/latest"),
        Path("../libensemble/docs"),
        Path("../../libensemble/docs"),
    ]
    for candidate in candidates:
        if candidate.exists():
            docs_dir = candidate
            break

if not docs_dir or not docs_dir.exists():
    print("ERROR: Documentation directory not found!")
    print("\nUsage: python build_index.py [docs_directory]")
    print("\nOr download docs first:")
    print("  wget -r -np -nH --cut-dirs=3 -R 'index.html*' \\")
    print("    https://libensemble.readthedocs.io/en/latest/")
    sys.exit(1)

print(f"Reading documentation from: {docs_dir}")

# Load documents
try:
    docs = SimpleDirectoryReader(str(docs_dir), recursive=True).load_data()
    print(f"Loaded {len(docs)} document chunks")
except Exception as e:
    print(f"ERROR loading documents: {e}")
    sys.exit(1)

if len(docs) == 0:
    print("ERROR: No documents found in directory!")
    sys.exit(1)

# Clean HTML tags from documents
print("Cleaning HTML tags from documents...")
cleaned_docs = []
for doc in docs:
    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(doc.text, 'html.parser')
    
    # Extract text (this removes all HTML tags)
    clean_text = soup.get_text()
    
    # Clean up extra whitespace
    clean_text = re.sub(r'\n\s*\n+', '\n\n', clean_text)  # Multiple newlines -> double newline
    clean_text = re.sub(r' +', ' ', clean_text)  # Multiple spaces -> single space
    clean_text = clean_text.strip()
    
    # Create new document with cleaned text
    cleaned_doc = Document(
        text=clean_text,
        metadata=doc.metadata
    )
    cleaned_docs.append(cleaned_doc)

print(f"Cleaned {len(cleaned_docs)} documents")

# Build index
print("Building vector index (this may take a few minutes)...")
index = VectorStoreIndex.from_documents(cleaned_docs, show_progress=True)

# Save index (in same directory as this script)
output_dir = Path(__file__).parent / "rag_index"
print(f"Saving index to {output_dir}/")
index.storage_context.persist(str(output_dir))

print(f"\n✓ Successfully built index with {len(docs)} documents")
print(f"✓ Index saved to {output_dir}/")
print("\nYou can now query the documentation:")
print(f"  python rag_query.py 'What generators are available?'")
print(f"  python mcp_server.py")
