# RAG Comparison: VectorDB vs Page Index

Compares two RAG strategies on the Pro Git book using Gemini (free tier).

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key
echo "GOOGLE_API_KEY=your_key_here" > .env

# 4. Place your PDF
cp /path/to/progit.pdf data/

# 5. Build both indexes (takes ~10 mins for full book on free tier)
python main.py --build

# 6. Ask a single question
python main.py --query "What is git rebase?"

# 7. Run all 5 test queries with full metrics report
python main.py --batch
```

## Architecture

| | VectorDB RAG | Page Index RAG |
|---|---|---|
| Chunking | 500-char sliding window | Whole pages |
| Index | ChromaDB (cosine) | BM25 (keyword) |
| Retrieval | Semantic similarity | Keyword frequency |
| Embedding | Google text-embedding-004 | None (BM25 is TF-IDF) |
| Speed | Slower (embedding overhead) | Faster |
| Strength | Semantic / conceptual queries | Exact keyword queries |

## Metrics Explained

- **Faithfulness** — LLM-as-judge: is the answer grounded in retrieved context?
- **Relevance** — LLM-as-judge: does the answer actually answer the question?
- **ROUGE-L** — n-gram overlap between the two systems' answers
- **Retrieval Diversity** — how many unique pages were retrieved (spread)
- **Latency** — wall-clock time for retrieval + generation