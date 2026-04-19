"""
Entry point — runs both RAG systems and compares them.

Usage:
    python main.py --build        # First run: build indexes
    python main.py --query        # Single query mode
    python main.py --batch        # Run all test queries
"""

import argparse
import os
import sys
import time

PDF_PATH = "./data/progit.pdf"

# ── Sample test queries about Pro Git ──
TEST_QUERIES = [
    "What is the difference between git merge and git rebase?",
    "How does git stash work?",
    "What are git branches and how do you create one?",
    "Explain what HEAD means in git.",
    "How do you resolve a merge conflict in git?",
]


def build_indexes():
    from src.vectordb_rag import VectorDBRAG
    from src.page_index_rag import PageIndexRAG

    print("\n" + "="*60)
    print("  STEP 1: Building VectorDB RAG Index")
    print("="*60)
    vdb = VectorDBRAG(PDF_PATH)
    # chunk_size=1200 → ~600 chunks total, well under 1000/day limit
    vdb.build_index(chunk_size=1200, chunk_overlap=150)

    print("\n" + "="*60)
    print("  STEP 2: Building Page Index RAG Index")
    print("="*60)
    pi = PageIndexRAG(PDF_PATH)
    pi.build_index()

    print("\n✅ Both indexes built successfully!")

def run_single_query(query: str):
    """Run a single query through both RAGs and compare."""
    from src.vectordb_rag import VectorDBRAG
    from src.page_index_rag import PageIndexRAG
    from src.evaluator import RAGEvaluator

    vdb = VectorDBRAG(PDF_PATH)
    pi  = PageIndexRAG(PDF_PATH)

    print(f"\n[Query] {query}\n")

    print("[VectorDB RAG] Generating answer...")
    vdb_result = vdb.generate(query, top_k=5)

    time.sleep(2)  # rate limit

    print("[Page Index RAG] Generating answer...")
    pi_result = pi.generate(query, top_k=5)

    print("\n--- VectorDB RAG Answer ---")
    print(vdb_result["answer"])

    print("\n--- Page Index RAG Answer ---")
    print(pi_result["answer"])

    print("\n[Evaluator] Running comparison metrics...")
    evaluator = RAGEvaluator()
    evaluator.compare(vdb_result, pi_result)


def run_batch():
    """Run all TEST_QUERIES and produce aggregate comparison."""
    from src.vectordb_rag import VectorDBRAG
    from src.page_index_rag import PageIndexRAG
    from src.evaluator import RAGEvaluator

    vdb = VectorDBRAG(PDF_PATH)
    pi  = PageIndexRAG(PDF_PATH)
    evaluator = RAGEvaluator()

    evaluator.batch_compare(vdb, pi, TEST_QUERIES, top_k=5)


if __name__ == "__main__":
    if not os.path.exists(PDF_PATH):
        print(f"[ERROR] PDF not found at '{PDF_PATH}'")
        print("Please place progit.pdf inside the ./data/ folder.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="RAG Comparison: VectorDB vs Page Index")
    parser.add_argument("--build", action="store_true", help="Build indexes (first-time setup)")
    parser.add_argument("--query", type=str, default=None, help="Run a single query")
    parser.add_argument("--batch", action="store_true", help="Run all test queries")
    args = parser.parse_args()

    if args.build:
        build_indexes()
    elif args.query:
        run_single_query(args.query)
    elif args.batch:
        run_batch()
    else:
        # Default: interactive mode
        print("\n RAG Comparison App (Pro Git)")
        print("="*40)
        print("1) Build indexes first:  python main.py --build")
        print("2) Single query:         python main.py --query 'What is git rebase?'")
        print("3) Batch evaluation:     python main.py --batch")