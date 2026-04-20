# Hybrid RAG Comparison Dashboard

A full-stack application designed to compare three different Retrieval-Augmented Generation (RAG) strategies on the Pro Git book using Gemini, with an automatic fallback to Groq.

## 🚀 Architecture

This application compares:
1. **VectorDB RAG**: Semantic retrieval using ChromaDB and `text-embedding-004`. Good at conceptual understanding based on 500-char chunks.
2. **Page Index RAG**: Keyword-based retrieval using BM25 across entire document pages. Good at exact keyword matching.
3. **Hybrid RAG 🌟**: Combines both vector similarity and keyword frequency using **Reciprocal Rank Fusion (RRF)** to deliver the best of both worlds.

| Feature         | VectorDB RAG                  | Page Index RAG           | Hybrid RAG (RRF)                |
|-----------------|--------------------------------|--------------------------|----------------------------------|
| Chunking        | 500-char sliding window        | Whole pages              | Both                             |
| Index           | ChromaDB (cosine)              | BM25 (keyword)           | Fused Array                      |
| Generative AI   | Gemini 2.5 Flash / Groq (Llama)| Gemini 2.5 Flash / Groq  | Gemini 2.5 Flash / Groq          |

## 📊 Metrics Explained

- **Faithfulness**: LLM-as-judge (0.0 to 1.0) grading if the generated answer is strictly grounded in the retrieved context.
- **Relevance**: LLM-as-judge (0.0 to 1.0) grading if the answer directly addresses the question asked.
- **ROUGE-L**: Measures n-gram overlap between answers from the different pipelines to find structural similarities.
- **Retrieval Diversity**: Measures the spread of retrieved chunks across unique document pages.
- **Latency**: Measures wall-clock performance split into retrieval and generation stages.

## 💻 Local Setup

1. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate       # Windows: venv\Scripts\activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   cd frontend && npm install && cd ..
   ```

3. **Add API Keys**
   Record your keys in a `.env` file at the root:
   ```bash
   GOOGLE_API_KEY=your_gemini_key_here
   GROQ_API_KEY=your_groq_key_here
   ```

4. **Prepare Data**
   Place `progit.pdf` inside the `./data/` folder and trigger an index build (takes ~10 mins):
   ```bash
   python main.py --build
   ```

5. **Run the Application**
   Open two terminals:
   * **Terminal 1 (Backend)**: `python app.py` starts FastAPI on port 8000.
   * **Terminal 2 (Frontend)**: `cd frontend && npm run dev` starts React on port 5173.

---

## 🌍 Complete Deployment Guide 

To easily host this for free on the public internet, deploy the backend to **Render** and the frontend to **Vercel**. 

*(Note: We have removed `chroma_db/` from `.gitignore` so your local built indexes upload to GitHub, bypassing timeout limits).*

### 1. Push to GitHub
Commit your latest code globally:
```bash
git add .
git commit -m "Ready for Deployment"
git push -u origin main
```

### 2. Deploy Backend to Render
1. Go to [Render](https://render.com/) and click **New +** -> **Web Service**.
2. Choose **Build and deploy from a Git repository**. Give Render access to this GitHub repository.
3. Configure the exact following settings on Render:
   * **Name**: `hybrid-rag-backend`
   * **Branch**: `main`
   * **Runtime**: `Python 3`
   * **Build Command**: `pip install -r requirements.txt`
   * **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
4. Expand **Advanced** and click **Add Environment Variable** to add these three variables:
   * **Key**: `PYTHON_VERSION` -> **Value**: `3.11.9`  *(This forces Render to use Python 3.11 instead of its default)*
   * **Key**: `GOOGLE_API_KEY` -> **Value**: `your_gemini_key_here`
   * **Key**: `GROQ_API_KEY`   -> **Value**: `your_groq_key_here`
6. Deployment will take ~5 minutes. Wait until you see "Live" and copy the URL (e.g., `https://hybrid-rag-xyz.onrender.com`).

### 3. Deploy Frontend to Vercel
1. Go to [Vercel](https://vercel.com/) and click **Add New** -> **Project**.
2. Import this same GitHub repository.
3. Click **Edit** next to Root Directory and pick the `frontend` folder. Accept to return back.
4. Set Framework preset to **Vite**.
5. Expand the **Environment Variables** tab and add:
   * **Name**: `VITE_API_URL`
   * **Value**: PASTE_RENDER_URL_HERE (No trailing slash! E.g. `https://hybrid-rag.onrender.com`)
6. Click **Deploy**. The frontend will be live in ~60 seconds!