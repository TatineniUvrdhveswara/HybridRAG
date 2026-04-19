import { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './index.css';

function App() {
  const [query, setQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [data, setData] = useState(null);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query) return;

    setIsLoading(true);
    setData(null);

    try {
      const response = await fetch('http://localhost:8000/api/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });
      const result = await response.json();
      setData(result);
    } catch (error) {
      console.error("Error fetching data:", error);
      alert("Failed to fetch answers. Make sure the backend is running!");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container">
      <header className="header">
        <h1 className="title">
          <span className="gradient-text">Hybrid</span> RAG Comparison
        </h1>
        <p className="subtitle">
          Discover why combining <strong>Vector Search</strong> and <strong>Page Keywords</strong> is optimal for real-world document retrieval.
        </p>

        <form onSubmit={handleSearch} className="search-form">
          <input
            type="text"
            className="search-input"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a question about the Pro Git book... (e.g. 'What is git rebase?')"
          />
          <button type="submit" className="search-button" disabled={isLoading}>
            {isLoading ? 'Searching...' : 'Compare Paths'}
          </button>
        </form>
      </header>

      {isLoading && (
        <div className="loading-state">
          <div className="spinner"></div>
          <p>Querying 3 individual RAG systems... This takes about 10-15 seconds to respect rate limits.</p>
        </div>
      )}

      {data && (
        <main className="dashboard">
          <div className="results-grid">
            <ResultColumn title="VectorDB RAG" data={data.results.vectordb} icon="v" />
            <ResultColumn title="Page Index RAG" data={data.results.page_index} icon="P" />
            <ResultColumn title="Hybrid RAG 🌟" data={data.results.hybrid} isHybrid={true} icon="H" />
          </div>

          <div className="metrics-dashboard panel">
            <h2>🏆 Aggregate Metrics Dashboard</h2>
            
            <div className="charts-wrapper">
              <MetricsCharts data={data.results} />
            </div>

            <div className="metrics-grid">
              {/* Header Row */}
              <div className="metric-header-row">
                <div className="metric-label header-cell">Metric</div>
                <div className="metric-cell header-cell">VectorDB</div>
                <div className="metric-cell header-cell">Page Index</div>
                <div className="metric-cell header-cell">Hybrid</div>
              </div>

              {/* Row: Faithfulness */}
              <MetricRow 
                label="Faithfulness (Relevance to Context)" 
                vdb={data.results.vectordb.faithfulness_score} 
                pi={data.results.page_index.faithfulness_score} 
                hybrid={data.results.hybrid.faithfulness_score} 
                higherIsBetter={true}
              />
              
              {/* Row: Relevance */}
              <MetricRow 
                label="Answer Quality (Relevance to Query)" 
                vdb={data.results.vectordb.relevance_score} 
                pi={data.results.page_index.relevance_score} 
                hybrid={data.results.hybrid.relevance_score} 
                higherIsBetter={true}
              />

              {/* Row: Retrieval Diversity */}
              <MetricRow 
                label="Retrieval Diversity (Unique Pages)" 
                vdb={data.results.vectordb.retrieval_diversity} 
                pi={data.results.page_index.retrieval_diversity} 
                hybrid={data.results.hybrid.retrieval_diversity} 
                higherIsBetter={true}
              />

              {/* Row: Context Length */}
              <MetricRow 
                label="Context Length (Chars)" 
                vdb={data.results.vectordb.context_length_chars} 
                pi={data.results.page_index.context_length_chars} 
                hybrid={data.results.hybrid.context_length_chars} 
                higherIsBetter={true} // Usually more context is better if it's faithful
              />
            </div>
            <div className="hybrid-conclusion">
              <p>🤖 Notice how the <strong>Hybrid RAG</strong> often scores highest in <em>Retrieval Diversity</em> and <em>Faithfulness</em> by combining the exact matches of Page Index with the semantic nuances of VectorDB. ROUGE overlap shows it takes the best bits of both!</p>
            </div>
          </div>
        </main>
      )}
    </div>
  );
}

function MetricRow({ label, vdb, pi, hybrid, higherIsBetter }) {
  // Determine winner
  let maxVal = Math.max(vdb, pi, hybrid);
  let minVal = Math.min(vdb, pi, hybrid);
  const bestVal = higherIsBetter ? maxVal : minVal;

  return (
    <div className="metric-row">
      <div className="metric-label">{label}</div>
      <div className={`metric-cell ${vdb === bestVal && vdb !== 0 ? 'winner' : ''}`}>{vdb.toFixed(2)}</div>
      <div className={`metric-cell ${pi === bestVal && pi !== 0 ? 'winner' : ''}`}>{pi.toFixed(2)}</div>
      <div className={`metric-cell ${hybrid === bestVal && hybrid !== 0 ? 'winner' : ''}`}>{hybrid.toFixed(2)}</div>
    </div>
  );
}

function ResultColumn({ title, data, isHybrid, icon }) {
  return (
    <div className={`result-column panel ${isHybrid ? 'hybrid-highlight' : ''}`}>
      <div className="col-header">
        <span className="icon">{icon}</span>
        <h3>{title}</h3>
      </div>
      
      <div className="answer-section">
        <h4>Generated Answer</h4>
        <p className="answer-text">{data.full_answer}</p>
        <div className="mini-latency">Took {data.latency_total_sec.toFixed(2)}s total ({data.latency_retrieval_sec.toFixed(2)}s retrieval)</div>
      </div>

      <div className="context-section">
        <details>
          <summary>View Retrieved Context ({data.retrieved_chunks.length} chunks)</summary>
          <div className="chunks-list">
            {data.retrieved_chunks.map((chk, idx) => (
              <div key={idx} className="chunk-card">
                <span className="chunk-meta">Page {chk.page_num} | Score: {chk.score} | Source: {chk.source.toUpperCase()}</span>
                <p>{chk.text}</p>
              </div>
            ))}
          </div>
        </details>
      </div>
    </div>
  );
}

function MetricsCharts({ data }) {
  const chartData = [
    {
      name: 'Faithfulness',
      VectorDB: data.vectordb.faithfulness_score,
      PageIndex: data.page_index.faithfulness_score,
      Hybrid: data.hybrid.faithfulness_score,
    },
    {
      name: 'Relevance',
      VectorDB: data.vectordb.relevance_score,
      PageIndex: data.page_index.relevance_score,
      Hybrid: data.hybrid.relevance_score,
    },
    {
      name: 'Diversity',
      VectorDB: data.vectordb.retrieval_diversity,
      PageIndex: data.page_index.retrieval_diversity,
      Hybrid: data.hybrid.retrieval_diversity,
    },
  ];

  return (
    <div className="chart-container">
      <ResponsiveContainer width="100%" height={350}>
        <BarChart
          data={chartData}
          margin={{ top: 20, right: 30, left: 0, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
          <XAxis dataKey="name" stroke="#a1a1aa" tickLine={false} axisLine={false} />
          <YAxis stroke="#a1a1aa" tickLine={false} axisLine={false} />
          <Tooltip 
            cursor={{fill: 'rgba(255,255,255,0.05)'}} 
            contentStyle={{ backgroundColor: 'rgba(25, 28, 36, 0.9)', borderColor: 'rgba(255,255,255,0.1)', color: '#fff', borderRadius: '8px' }} 
          />
          <Legend wrapperStyle={{ paddingTop: '20px' }} />
          <Bar dataKey="VectorDB" fill="#3a86ff" radius={[4, 4, 0, 0]} />
          <Bar dataKey="PageIndex" fill="#06d6a0" radius={[4, 4, 0, 0]} />
          <Bar dataKey="Hybrid" fill="#9d4edd" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export default App;
