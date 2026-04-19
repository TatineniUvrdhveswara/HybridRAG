import time
from typing import List, Dict, Any
from src.vectordb_rag import VectorDBRAG
from src.page_index_rag import PageIndexRAG
from src.utils import generate_content

class HybridRAG:
    """
    Combines VectorDB RAG (semantic) and Page Index RAG (keyword) using
    Reciprocal Rank Fusion (RRF) to produce a hybrid retrieved context.
    """
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.vdb = VectorDBRAG(pdf_path)
        self.pi = PageIndexRAG(pdf_path)
        
        # We assume VectorDBRAG shares model inference with default
        self.model_name = self.vdb.model_name

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves top_k chunks from both VectorDB and PageIndex,
        then merges and reranks them using Reciprocal Rank Fusion.
        """
        vdb_results = self.vdb.retrieve(query, top_k=top_k)
        pi_results = self.pi.retrieve(query, top_k=top_k)
        
        # RRF constant 
        # rrf_score = 1 / (rank + k)
        K = 60
        
        fused_scores = {}
        chunk_map = {}
        
        # Rank VectorDB results
        for rank, chunk in enumerate(vdb_results):
            # Identifying the chunk by its exact text or page_num/source combo 
            # VectorDB returns short chunks. PageIndex returns whole pages.
            # To merge them fairly, we will group by page number for the context.
            # However, VectorDB gives specific short text. Since we will feed this to the LLM,
            # we will retain the chunk's text and uniquely identify it by its snippet.
            key = chunk["text"].strip()
            if key not in fused_scores:
                fused_scores[key] = 0
            fused_scores[key] += 1.0 / (rank + 1 + K)
            chunk_map[key] = chunk
            
        # Rank PageIndex results
        for rank, chunk in enumerate(pi_results):
            # Since PageIndex is the whole page, we want to isolate it.
            # We'll just slice the first 800 chars like the original implementation
            short_text = chunk["text"][:800].strip()
            if short_text not in fused_scores:
                fused_scores[short_text] = 0
            fused_scores[short_text] += 1.0 / (rank + 1 + K)
            
            # Create a localized version of the page chunk
            chunk_map[short_text] = {
                "text": short_text,
                "page_num": chunk["page_num"],
                "score": chunk["score"], # We'll overwrite with RRF but keep original for debug
                "source": "hybrid"
            }
            
        # Sort by fused score
        sorted_chunks = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        
        retrieved = []
        for text_key, rrf_score in sorted_chunks[:top_k]:
            c = chunk_map[text_key]
            c["score"] = round(rrf_score, 4) # Overwrite with RRF score
            retrieved.append(c)
            
        return retrieved

    def generate(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Full RAG pipeline for Hybrid."""
        start_time = time.perf_counter()

        # Step 1: Retrieve
        retrieved_chunks = self.retrieve(query, top_k)
        retrieval_time = time.perf_counter() - start_time

        # Step 2: Build context
        context = "\n\n---\n\n".join(
            [f"[Page {c['page_num']}] {c['text']}" for c in retrieved_chunks]
        )

        # Step 3: Prompt
        prompt = f"""You are a helpful assistant answering questions about the Pro Git book.

Use ONLY the following retrieved context to answer the question.
If the answer is not in the context, say "I don't have enough context to answer."

Context:
{context}

Question: {query}

Answer:"""

        # Step 4: Generate
        gen_start = time.perf_counter()
        answer_text = generate_content(prompt, self.model_name)
        generation_time = time.perf_counter() - gen_start

        total_time = time.perf_counter() - start_time

        return {
            "query": query,
            "answer": answer_text,
            "retrieved_chunks": retrieved_chunks,
            "retrieval_time_sec": round(retrieval_time, 3),
            "generation_time_sec": round(generation_time, 3),
            "total_time_sec": round(total_time, 3),
            "method": "Hybrid RAG",
            "num_chunks_retrieved": len(retrieved_chunks),
            "context_length": len(context),
            "avg_retrieval_score": round(
                sum(c["score"] for c in retrieved_chunks) / len(retrieved_chunks), 4
            ) if retrieved_chunks else 0
        }
