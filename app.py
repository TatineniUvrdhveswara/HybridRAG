import os
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.vectordb_rag import VectorDBRAG
from src.page_index_rag import PageIndexRAG
from src.hybrid_rag import HybridRAG
from src.evaluator import RAGEvaluator

app = FastAPI(title="Pro Git RAG Comparison API")

# Allow Web UI to fetch
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PDF_PATH = "./data/progit.pdf"

# Initialize singletons at startup to save memory/speed
vdb = None
pi = None
hybrid = None
evaluator = None

@app.on_event("startup")
def load_models():
    global vdb, pi, hybrid, evaluator
    print("Loading indexing systems...")
    vdb = VectorDBRAG(PDF_PATH)
    pi = PageIndexRAG(PDF_PATH)
    hybrid = HybridRAG(PDF_PATH)
    evaluator = RAGEvaluator()
    print("Initialization complete.")

class QueryRequest(BaseModel):
    query: str

@app.post("/api/compare")
def compare_query(req: QueryRequest):
    query = req.query
    print(f"\n[API] Received query: {query}")
    
    # 1. Run inference sequentially to respect rate limits
    print("  -> Running VectorDB RAG...")
    vdb_result = vdb.generate(query, top_k=5)
    time.sleep(2)
    
    print("  -> Running Page Index RAG...")
    pi_result = pi.generate(query, top_k=5)
    time.sleep(2)
    
    print("  -> Running Hybrid RAG...")
    hybrid_result = hybrid.generate(query, top_k=5)
    
    # 2. Evaluate all 3
    print("  -> Evaluating responses...")
    vdb_metrics = evaluator.evaluate_single(vdb_result, verbose=False)
    time.sleep(1)
    pi_metrics = evaluator.evaluate_single(pi_result, verbose=False)
    time.sleep(1)
    hybrid_metrics = evaluator.evaluate_single(hybrid_result, verbose=False)
    
    # Calculate ROUGE between combinations (optional, mostly for insight)
    vdb_hybrid_rouge = evaluator.rouge_l_score(vdb_result["answer"], hybrid_result["answer"])
    pi_hybrid_rouge = evaluator.rouge_l_score(pi_result["answer"], hybrid_result["answer"])
    
    # 3. Format response
    return {
        "query": query,
        "results": {
            "vectordb": {
                **vdb_metrics,
                "full_answer": vdb_result["answer"],
                "retrieved_chunks": vdb_result["retrieved_chunks"]
            },
            "page_index": {
                **pi_metrics,
                "full_answer": pi_result["answer"],
                "retrieved_chunks": pi_result["retrieved_chunks"]
            },
            "hybrid": {
                **hybrid_metrics,
                "full_answer": hybrid_result["answer"],
                "retrieved_chunks": hybrid_result["retrieved_chunks"]
            }
        },
        "comparisons": {
            "rouge_vdb_hybrid": vdb_hybrid_rouge,
            "rouge_pi_hybrid": pi_hybrid_rouge
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
