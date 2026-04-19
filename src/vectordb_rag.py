import chromadb
import time
from tqdm import tqdm
from typing import List, Dict, Any
from src.utils import generate_content
from src.pdf_processor import PDFProcessor, Chunk
from src.utils import get_embedding, get_query_embedding, get_gemini_model

COLLECTION_NAME = "progit_chunks"

class VectorDBRAG:
    """
    RAG using ChromaDB vector store + Google text-embedding-004 embeddings.
    Retrieval is based on semantic similarity (cosine distance).
    """

    def __init__(self, pdf_path: str, persist_dir: str = "./chroma_db"):
        self.pdf_path = pdf_path
        self.persist_dir = persist_dir
        self.model_name = get_gemini_model()

        # Persistent ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = None

    def build_index(self, chunk_size: int = 1200, chunk_overlap: int = 150):
        """
        Build index with resume support.
        If interrupted, re-running skips already-embedded chunks.
        chunk_size=1200 keeps total chunks under 1000 (free tier daily limit).
        """
        import re

        # Try to load existing collection (resume mode)
        try:
            self.collection = self.client.get_collection(COLLECTION_NAME)
            existing_ids = set(self.collection.get()["ids"])
            print(f"[VectorDB] Resuming — {len(existing_ids)} chunks already embedded")
        except Exception:
            # Fresh start
            try:
                self.client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass
            self.collection = self.client.create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            existing_ids = set()
            print(f"[VectorDB] Fresh index created")

        # Extract chunks
        processor = PDFProcessor(self.pdf_path)
        chunks: List[Chunk] = processor.extract_chunks(chunk_size, chunk_overlap)

        # Filter out already-embedded chunks
        remaining = [c for c in chunks if c.chunk_id not in existing_ids]
        print(f"[VectorDB] {len(chunks)} total chunks, {len(remaining)} remaining to embed")

        if not remaining:
            print("[VectorDB] Index already complete!")
            return

        batch_size = 5  # smaller batch = fewer wasted requests on rate limit hit
        failed_count = 0

        for i in tqdm(range(0, len(remaining), batch_size)):
            batch = remaining[i: i + batch_size]

            embeddings, ids, documents, metadatas = [], [], [], []
            for chunk in batch:
                success = False
                for attempt in range(5):  # up to 5 retries
                    try:
                        emb = get_embedding(chunk.text)
                        embeddings.append(emb)
                        ids.append(chunk.chunk_id)
                        documents.append(chunk.text)
                        metadatas.append({
                            "page_num": chunk.page_num,
                            "start_char": chunk.start_char,
                            "end_char": chunk.end_char
                        })
                        success = True
                        time.sleep(0.15)  # ~6-7 req/sec, safe for free tier
                        break
                    except Exception as e:
                        err_msg = str(e)
                        if "429" in err_msg:
                            # Parse suggested retry delay from error message
                            match = re.search(r"retryDelay.*?(\d+)s", err_msg)
                            wait = int(match.group(1)) + 2 if match else 65
                            print(f"\n  [Rate Limit] Daily quota hit. Waiting {wait}s...")
                            time.sleep(wait)
                        else:
                            print(f"\n  [WARN] Attempt {attempt+1} failed: {e}")
                            time.sleep(2 ** attempt)

                if not success:
                    failed_count += 1
                    print(f"\n  [SKIP] {chunk.chunk_id} failed all retries")

            # Save batch immediately — so progress is preserved even if interrupted
            if embeddings:
                self.collection.add(
                    embeddings=embeddings,
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )

        total = self.collection.count()
        print(f"\n[VectorDB] Index built — {total} vectors stored, {failed_count} skipped")
    def load_index(self):
        """Load existing index (skip re-embedding)."""
        self.collection = self.client.get_collection(COLLECTION_NAME)
        print(f"[VectorDB] Loaded index — {self.collection.count()} vectors")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Semantic retrieval: embed query → cosine similarity search."""
        if self.collection is None:
            self.load_index()

        query_emb = get_query_embedding(query)
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            retrieved.append({
                "text": doc,
                "page_num": meta["page_num"],
                "score": round(1 - dist, 4),   # convert distance to similarity
                "source": "vectordb"
            })
        return retrieved

    def generate(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Full RAG pipeline: retrieve → augment prompt → generate answer."""
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
            "method": "VectorDB RAG",
            "num_chunks_retrieved": len(retrieved_chunks),
            "context_length": len(context),
            "avg_retrieval_score": round(
                sum(c["score"] for c in retrieved_chunks) / len(retrieved_chunks), 4
            ) if retrieved_chunks else 0
        }