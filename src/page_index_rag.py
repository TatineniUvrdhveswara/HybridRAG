import time
import pickle
import os
from typing import List, Dict, Any
from src.utils import generate_content
from rank_bm25 import BM25Okapi
from src.pdf_processor import PDFProcessor, Page
from src.utils import get_gemini_model

INDEX_CACHE = "./chroma_db/bm25_page_index.pkl"

class PageIndexRAG:
    """
    RAG using BM25 keyword-based Page Index.
    Each page is a retrieval unit — no chunking or embedding needed.
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.model_name = "gemini-1.5-flash"
        self.pages: List[Page] = []
        self.bm25 = None
        self.tokenized_corpus = []

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + lowercase tokenizer."""
        return text.lower().split()

    def build_index(self):
        """Extract pages → build BM25 index → save to disk."""
        processor = PDFProcessor(self.pdf_path)
        self.pages = processor.extract_pages()

        print(f"[PageIndex] Building BM25 index over {len(self.pages)} pages...")
        self.tokenized_corpus = [self._tokenize(p.text) for p in self.pages]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # Cache to disk for fast reload
        os.makedirs(os.path.dirname(INDEX_CACHE), exist_ok=True)
        with open(INDEX_CACHE, "wb") as f:
            pickle.dump({
                "pages": self.pages,
                "tokenized_corpus": self.tokenized_corpus
            }, f)
        print(f"[PageIndex] BM25 index built and cached at '{INDEX_CACHE}'")

    def load_index(self):
        """Load cached BM25 index."""
        if not os.path.exists(INDEX_CACHE):
            raise FileNotFoundError(
                f"Cache not found at '{INDEX_CACHE}'. Run build_index() first."
            )
        with open(INDEX_CACHE, "rb") as f:
            data = pickle.load(f)
        self.pages = data["pages"]
        self.tokenized_corpus = data["tokenized_corpus"]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"[PageIndex] Loaded BM25 index — {len(self.pages)} pages")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """BM25 keyword search over pages."""
        if self.bm25 is None:
            self.load_index()

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k page indices sorted by BM25 score
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        retrieved = []
        for idx in top_indices:
            page = self.pages[idx]
            retrieved.append({
                "text": page.text,
                "page_num": page.page_num,
                "score": round(float(scores[idx]), 4),
                "source": "page_index"
            })
        return retrieved

    def generate(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Full RAG pipeline: retrieve pages → augment → generate answer."""
        start_time = time.perf_counter()

        # Step 1: Retrieve
        retrieved_pages = self.retrieve(query, top_k)
        retrieval_time = time.perf_counter() - start_time

        # Step 2: Build context
        context = "\n\n---\n\n".join(
            [f"[Page {p['page_num']}] {p['text'][:800]}" for p in retrieved_pages]
            # Limit each page to 800 chars to avoid token overflow
        )

        # Step 3: Prompt
        prompt = f"""You are a helpful assistant answering questions about the Pro Git book.

Use ONLY the following retrieved page content to answer the question.
If the answer is not in the context, say "I don't have enough context to answer."

Context:
{context}

Question: {query}

Answer:"""

        # Step 4: Generate
        gen_start = time.perf_counter()
        answer_text = generate_content(prompt)
        generation_time = time.perf_counter() - gen_start

        total_time = time.perf_counter() - start_time

        return {
            "query": query,
            "answer": answer_text,
            "retrieved_chunks": retrieved_pages,
            "retrieval_time_sec": round(retrieval_time, 3),
            "generation_time_sec": round(generation_time, 3),
            "total_time_sec": round(total_time, 3),
            "method": "Page Index RAG",
            "num_chunks_retrieved": len(retrieved_pages),
            "context_length": len(context),
            "avg_retrieval_score": round(
                sum(p["score"] for p in retrieved_pages) / len(retrieved_pages), 4
            ) if retrieved_pages else 0
        }