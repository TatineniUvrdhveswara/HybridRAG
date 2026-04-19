import time
import json
import os
from typing import List, Dict, Any

import nltk
import numpy as np
from rouge_score import rouge_scorer
from tabulate import tabulate

from src.utils import get_gemini_model, configure_gemini

# Download NLTK data once
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

class RAGEvaluator:
    """
    Compares VectorDB RAG vs Page Index RAG using multiple metrics:

    1.  Latency           — total response time (seconds)
    2.  Retrieval Score   — avg similarity/BM25 score of top-k chunks
    3.  Context Length    — total characters in retrieved context
    4.  Answer Length     — word count of final answer
    5.  ROUGE-L           — n-gram overlap between answers
    6.  LLM Faithfulness  — Gemini judges if answer is grounded in context
    7.  LLM Relevance     — Gemini judges if answer is relevant to question
    """

    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    # ──────────────────────────────────────────────
    # Core evaluation helpers
    # ──────────────────────────────────────────────

    def _llm_score(self, prompt: str) -> float:
        try:
            from src.utils import generate_content
            text = generate_content(prompt).strip()
            import re
            matches = re.findall(r"\b(0\.\d+|1\.0|0|1)\b", text)
            return float(matches[0]) if matches else 0.5
        except Exception as e:
            print(f"  [WARN] LLM scoring failed: {e}")
            return -1.0

    def faithfulness_score(self, context: str, answer: str) -> float:
        """
        LLM-as-judge: Is the answer fully grounded in the context?
        Returns 0.0 (hallucination) → 1.0 (fully grounded).
        """
        prompt = f"""Rate how faithfully the answer is grounded in the provided context.
Score strictly between 0.0 (completely ungrounded/hallucinated) and 1.0 (fully grounded).
Respond with ONLY a decimal number like 0.8

Context:
{context[:2000]}

Answer:
{answer}

Faithfulness Score:"""
        return self._llm_score(prompt)

    def relevance_score(self, question: str, answer: str) -> float:
        """
        LLM-as-judge: Does the answer directly address the question?
        Returns 0.0 (irrelevant) → 1.0 (highly relevant).
        """
        prompt = f"""Rate how well the answer addresses the question.
Score strictly between 0.0 (completely irrelevant) and 1.0 (perfectly answers it).
Respond with ONLY a decimal number like 0.9

Question: {question}
Answer: {answer}

Relevance Score:"""
        return self._llm_score(prompt)

    def rouge_l_score(self, answer_a: str, answer_b: str) -> float:
        """ROUGE-L overlap between two answers (similarity, not quality)."""
        scores = self.scorer.score(answer_a, answer_b)
        return round(scores["rougeL"].fmeasure, 4)

    def retrieval_diversity(self, retrieved_chunks: List[Dict]) -> float:
        """
        Measures how spread out the retrieved chunks are across pages.
        Higher = more diverse page coverage.
        """
        pages = [c["page_num"] for c in retrieved_chunks]
        unique_pages = len(set(pages))
        return round(unique_pages / len(pages), 4) if pages else 0.0

    # ──────────────────────────────────────────────
    # Main comparison
    # ──────────────────────────────────────────────

    def evaluate_single(
        self,
        result: Dict[str, Any],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Evaluate one RAG result and return all metrics."""
        context = "\n".join(c["text"] for c in result["retrieved_chunks"])

        if verbose:
            print(f"  → Scoring faithfulness for [{result['method']}]...")
        faith = self.faithfulness_score(context, result["answer"])
        time.sleep(1)  # rate limit courtesy

        if verbose:
            print(f"  → Scoring relevance for [{result['method']}]...")
        relev = self.relevance_score(result["query"], result["answer"])
        time.sleep(1)

        metrics = {
            "method": result["method"],
            "query": result["query"],
            "answer_preview": result["answer"][:200] + "...",
            # Speed
            "latency_total_sec": result["total_time_sec"],
            "latency_retrieval_sec": result["retrieval_time_sec"],
            "latency_generation_sec": result["generation_time_sec"],
            # Retrieval quality
            "avg_retrieval_score": result["avg_retrieval_score"],
            "num_chunks_retrieved": result["num_chunks_retrieved"],
            "retrieval_diversity": self.retrieval_diversity(result["retrieved_chunks"]),
            # Answer quality
            "context_length_chars": result["context_length"],
            "answer_word_count": len(result["answer"].split()),
            "faithfulness_score": faith,
            "relevance_score": relev,
        }
        return metrics

    def compare(
        self,
        vectordb_result: Dict[str, Any],
        pageindex_result: Dict[str, Any],
        save_path: str = "./results/comparison.json"
    ):
        """Run full comparison between two RAG results and print a table."""
        print("\n" + "="*60)
        print("  EVALUATING VectorDB RAG...")
        print("="*60)
        vdb_metrics = self.evaluate_single(vectordb_result)

        print("\n" + "="*60)
        print("  EVALUATING Page Index RAG...")
        print("="*60)
        pi_metrics = self.evaluate_single(pageindex_result)

        # ROUGE-L between the two answers
        rouge = self.rouge_l_score(vectordb_result["answer"], pageindex_result["answer"])

        # ── Print comparison table ──
        rows = []
        metric_keys = [
            ("latency_total_sec",       "Total Latency (s)"),
            ("latency_retrieval_sec",   "Retrieval Latency (s)"),
            ("latency_generation_sec",  "Generation Latency (s)"),
            ("avg_retrieval_score",     "Avg Retrieval Score"),
            ("retrieval_diversity",     "Retrieval Diversity"),
            ("context_length_chars",    "Context Length (chars)"),
            ("answer_word_count",       "Answer Word Count"),
            ("faithfulness_score",      "Faithfulness (LLM judge)"),
            ("relevance_score",         "Relevance (LLM judge)"),
        ]
        for key, label in metric_keys:
            vdb_val = vdb_metrics.get(key, "N/A")
            pi_val  = pi_metrics.get(key, "N/A")

            # Winner highlight
            try:
                winner = "VectorDB ✓" if float(vdb_val) > float(pi_val) else "PageIdx ✓"
                if key in ("latency_total_sec", "latency_retrieval_sec",
                           "latency_generation_sec", "context_length_chars"):
                    winner = "VectorDB ✓" if float(vdb_val) < float(pi_val) else "PageIdx ✓"
            except Exception:
                winner = "—"
            rows.append([label, vdb_val, pi_val, winner])

        print("\n" + "="*70)
        print("  RAG COMPARISON RESULTS")
        print("="*70)
        print(f"  Query: {vectordb_result['query']}")
        print(f"  ROUGE-L (answer similarity): {rouge}")
        print()
        print(tabulate(
            rows,
            headers=["Metric", "VectorDB RAG", "Page Index RAG", "Better"],
            tablefmt="rounded_outline",
            floatfmt=".4f"
        ))
        print("="*70 + "\n")

        # Save results
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        combined = {
            "query": vectordb_result["query"],
            "rouge_l_between_answers": rouge,
            "vectordb_rag": {**vdb_metrics, "full_answer": vectordb_result["answer"]},
            "page_index_rag": {**pi_metrics, "full_answer": pageindex_result["answer"]},
        }
        with open(save_path, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"[Evaluator] Results saved to '{save_path}'")
        return combined

    def batch_compare(
        self,
        vectordb_rag,
        page_index_rag,
        queries: List[str],
        top_k: int = 5
    ) -> List[Dict]:
        """Run comparison across multiple queries and produce a summary."""
        all_results = []
        for i, query in enumerate(queries):
            print(f"\n[Batch {i+1}/{len(queries)}] Query: {query}")

            vdb_result = vectordb_rag.generate(query, top_k)
            time.sleep(2)  # rate limit
            pi_result  = page_index_rag.generate(query, top_k)
            time.sleep(2)

            result = self.compare(
                vdb_result,
                pi_result,
                save_path=f"./results/query_{i+1}.json"
            )
            all_results.append(result)
            time.sleep(3)   # be gentle with free tier

        self._print_summary(all_results)
        return all_results

    def _print_summary(self, all_results: List[Dict]):
        """Print aggregate stats across all queries."""
        def avg(key, method_key):
            vals = [
                r[method_key][key]
                for r in all_results
                if isinstance(r[method_key].get(key), (int, float))
                and r[method_key][key] >= 0
            ]
            return round(np.mean(vals), 4) if vals else "N/A"

        print("\n" + "="*70)
        print("  AGGREGATE SUMMARY")
        print("="*70)
        summary = [
            ["Avg Total Latency (s)",       avg("latency_total_sec", "vectordb_rag"),       avg("latency_total_sec", "page_index_rag")],
            ["Avg Faithfulness",            avg("faithfulness_score", "vectordb_rag"),       avg("faithfulness_score", "page_index_rag")],
            ["Avg Relevance",               avg("relevance_score", "vectordb_rag"),          avg("relevance_score", "page_index_rag")],
            ["Avg Retrieval Diversity",     avg("retrieval_diversity", "vectordb_rag"),      avg("retrieval_diversity", "page_index_rag")],
        ]
        print(tabulate(summary, headers=["Metric", "VectorDB RAG", "Page Index RAG"],
                       tablefmt="rounded_outline"))
        print("="*70)