import { useState, useEffect } from 'react';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
} from 'recharts';
import './index.css';

const COLORS = ['#3a86ff', '#06d6a0', '#9d4edd'];

function App() {
  const [query, setQuery] = useState('');
  const [route, setRoute] = useState(window.location.hash || '#compare');
  const [isLoading, setIsLoading] = useState(false);
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const handleHashChange = () => setRoute(window.location.hash || '#compare');
    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query) return;

    setIsLoading(true);
    setError(null);

    const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';

    try {
      const response = await fetch(`${apiUrl}/api/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}`);
      }

      const result = await response.json();
      setData(result);
      window.location.hash = '#compare';
    } catch (err) {
      console.error('Error fetching data:', err);
      setError('Unable to fetch results. Verify the backend is running and try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const navClass = (target) => (route === target ? 'tab-link active' : 'tab-link');
  const results = data?.results;

  return (
    <div className="container">
      <header className="header">
        <h1 className="title">
          <span className="gradient-text">Hybrid</span> RAG Comparison
        </h1>
        <p className="subtitle">
          Search once, compare three RAG systems, and review metrics on a separate dashboard page.
        </p>

        <div className="nav-tabs">
          <a href="#compare" className={navClass('#compare')}>
            Compare Responses
          </a>
          <a href="#dashboard" className={navClass('#dashboard')}>
            Metrics Dashboard
          </a>
        </div>

        <form onSubmit={handleSearch} className="search-form">
          <input
            type="text"
            className="search-input"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a question about the Pro Git book..."
          />
          <button type="submit" className="search-button" disabled={isLoading}>
            {isLoading ? 'Searching...' : 'Compare Paths'}
          </button>
        </form>
      </header>

      {error && <div className="error-banner">{error}</div>}

      {isLoading && (
        <div className="loading-state">
          <div className="spinner"></div>
          <p>Querying 3 individual RAG systems... This takes about 10-15 seconds.</p>
        </div>
      )}

      {route === '#compare' && (
        <main className="compare-page">
          {results ? (
            <div className="results-grid">
              <ResultColumn title="VectorDB RAG" data={results.vectordb} icon="V" />
              <ResultColumn title="Page Index RAG" data={results.page_index} icon="P" />
              <ResultColumn title="Hybrid RAG 🌟" data={results.hybrid} isHybrid={true} icon="H" />
            </div>
          ) : (
            <div className="panel empty-panel">
              <h2>Run a query first</h2>
              <p>Enter a question above and submit. Then use the "Metrics Dashboard" tab to review scoring and visualizations separately.</p>
            </div>
          )}
        </main>
      )}

      {route === '#dashboard' && (
        <main className="dashboard-page">
          <div className="panel dashboard-panel">
            <div className="dashboard-header">
              <div>
                <h2>📊 Dashboard Overview</h2>
                <p>Visual comparisons and metrics for the latest query. Use the Compare tab to view only the generated answers.</p>
              </div>
              {results && (
                <button className="dashboard-action" onClick={() => (window.location.hash = '#compare')}>
                  View Responses
                </button>
              )}
            </div>

            {!results ? (
              <div className="empty-panel dashboard-empty">
                <h3>No metrics yet</h3>
                <p>Submit a query on the Compare page first to generate results and enable dashboard visualizations.</p>
              </div>
            ) : (
              <>
                <div className="stats-row">
                  <StatCard label="Query" value={`"${data.query}"`} />
                  <StatCard label="Best System" value={bestSystem(results)} />
                  <StatCard label="Hybrid ROUGE" value={data.comparisons.rouge_vdb_hybrid.toFixed(2)} />
                  <StatCard label="Chunks Fetched" value={totalChunks(results)} />
                </div>

                <div className="visualization-panels">
                  <div className="panel visual-panel">
                    <h3>Performance Comparison</h3>
                    <MetricsBarChart data={results} />
                  </div>

                  <div className="panel visual-panel">
                    <h3>Quality Radar</h3>
                    <MetricsRadar data={results} />
                  </div>

                  <div className="panel visual-panel line-panel">
                    <h3>Metric Trends</h3>
                    <MetricsLineChart data={results} />
                  </div>

                  <div className="pie-column">
                    <div className="panel small-card">
                      <h4>Retrieved Chunks</h4>
                      <RetrievalPie data={results} />
                    </div>
                    <div className="panel small-card">
                      <h4>Latency Breakdown</h4>
                      <LatencyPie data={results} />
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>
        </main>
      )}
    </div>
  );
}

function StatCard({ label, value }) {
  return (
    <div className="stat-card">
      <span className="stat-label">{label}</span>
      <span className="stat-value">{value}</span>
    </div>
  );
}

function bestSystem(results) {
  const hybridScore = results.hybrid.faithfulness_score + results.hybrid.relevance_score;
  const vdbScore = results.vectordb.faithfulness_score + results.vectordb.relevance_score;
  const piScore = results.page_index.faithfulness_score + results.page_index.relevance_score;
  const winner = Math.max(hybridScore, vdbScore, piScore);

  if (winner === hybridScore) return 'Hybrid RAG';
  if (winner === vdbScore) return 'VectorDB RAG';
  return 'Page Index RAG';
}

function totalChunks(results) {
  return results.vectordb.retrieved_chunks.length + results.page_index.retrieved_chunks.length + results.hybrid.retrieved_chunks.length;
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

function MetricsBarChart({ data }) {
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
      <ResponsiveContainer width="100%" height={320}>
        <BarChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.16)" vertical={false} />
          <XAxis dataKey="name" stroke="#ffffff" tickLine={false} axisLine={false} />
          <YAxis stroke="#ffffff" tickLine={false} axisLine={false} />
          <Tooltip
            cursor={{ fill: 'rgba(255,255,255,0.08)' }}
            contentStyle={{ backgroundColor: 'rgba(10, 13, 25, 0.98)', borderColor: 'rgba(255,255,255,0.18)', borderRadius: '10px' }}
          />
          <Legend wrapperStyle={{ paddingTop: '18px', color: '#f0f0f5' }} />
          <Bar dataKey="VectorDB" fill="#5d5cff" radius={[6, 6, 0, 0]} />
          <Bar dataKey="PageIndex" fill="#00f4c4" radius={[6, 6, 0, 0]} />
          <Bar dataKey="Hybrid" fill="#ff4d94" radius={[6, 6, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function MetricsLineChart({ data }) {
  const lineData = [
    {
      metric: 'Faithfulness',
      VectorDB: data.vectordb.faithfulness_score,
      PageIndex: data.page_index.faithfulness_score,
      Hybrid: data.hybrid.faithfulness_score,
    },
    {
      metric: 'Relevance',
      VectorDB: data.vectordb.relevance_score,
      PageIndex: data.page_index.relevance_score,
      Hybrid: data.hybrid.relevance_score,
    },
    {
      metric: 'Diversity',
      VectorDB: data.vectordb.retrieval_diversity,
      PageIndex: data.page_index.retrieval_diversity,
      Hybrid: data.hybrid.retrieval_diversity,
    },
  ];

  return (
    <div className="line-chart">
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={lineData} margin={{ top: 10, right: 8, left: -10, bottom: 0 }}>
          <CartesianGrid strokeDasharray="4 4" stroke="rgba(255,255,255,0.12)" />
          <XAxis dataKey="metric" stroke="#ffffff" axisLine={false} tickLine={false} />
          <YAxis stroke="#ffffff" axisLine={false} tickLine={false} />
          <Tooltip
            cursor={{ stroke: 'rgba(255,255,255,0.18)', strokeWidth: 1 }}
            contentStyle={{ backgroundColor: 'rgba(10, 13, 25, 0.96)', borderColor: 'rgba(255,255,255,0.14)', borderRadius: '10px' }}
          />
          <Legend verticalAlign="top" wrapperStyle={{ color: '#f0f0f5', paddingBottom: 10 }} />
          <Line type="monotone" dataKey="VectorDB" stroke="#5d5cff" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 6 }} />
          <Line type="monotone" dataKey="PageIndex" stroke="#00f4c4" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 6 }} />
          <Line type="monotone" dataKey="Hybrid" stroke="#ff4d94" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 6 }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function MetricsRadar({ data }) {
  const radarData = [
    { metric: 'Faithfulness', VectorDB: data.vectordb.faithfulness_score, PageIndex: data.page_index.faithfulness_score, Hybrid: data.hybrid.faithfulness_score },
    { metric: 'Relevance', VectorDB: data.vectordb.relevance_score, PageIndex: data.page_index.relevance_score, Hybrid: data.hybrid.relevance_score },
    { metric: 'Diversity', VectorDB: data.vectordb.retrieval_diversity, PageIndex: data.page_index.retrieval_diversity, Hybrid: data.hybrid.retrieval_diversity },
  ];

  return (
    <ResponsiveContainer width="100%" height={320}>
      <RadarChart data={radarData} outerRadius={115}>
        <PolarGrid stroke="rgba(255,255,255,0.12)" />
        <PolarAngleAxis dataKey="metric" stroke="#a1a1aa" />
        <PolarRadiusAxis angle={90} domain={[0, 'dataMax']} tick={false} />
        <Radar name="VectorDB" dataKey="VectorDB" stroke="#3a86ff" fill="#3a86ff" fillOpacity={0.35} />
        <Radar name="PageIndex" dataKey="PageIndex" stroke="#06d6a0" fill="#06d6a0" fillOpacity={0.35} />
        <Radar name="Hybrid" dataKey="Hybrid" stroke="#9d4edd" fill="#9d4edd" fillOpacity={0.35} />
        <Legend wrapperStyle={{ color: '#f0f0f5', top: 10 }} />
      </RadarChart>
    </ResponsiveContainer>
  );
}

function RetrievalPie({ data }) {
  const chartData = [
    { name: 'VectorDB', value: data.vectordb.retrieved_chunks.length },
    { name: 'PageIndex', value: data.page_index.retrieved_chunks.length },
    { name: 'Hybrid', value: data.hybrid.retrieved_chunks.length },
  ];

  return (
    <ResponsiveContainer width="100%" height={220}>
      <PieChart>
        <Pie data={chartData} dataKey="value" nameKey="name" innerRadius={45} outerRadius={80} paddingAngle={4}>
          {chartData.map((entry, index) => (
            <Cell key={entry.name} fill={COLORS[index % COLORS.length]} />
          ))}
        </Pie>
        <Tooltip
          contentStyle={{ backgroundColor: 'rgba(15,17,23,0.95)', borderColor: 'rgba(255,255,255,0.12)', borderRadius: '10px' }}
        />
      </PieChart>
    </ResponsiveContainer>
  );
}

function LatencyPie({ data }) {
  const chartData = [
    { name: 'VectorDB', value: data.vectordb.latency_total_sec },
    { name: 'PageIndex', value: data.page_index.latency_total_sec },
    { name: 'Hybrid', value: data.hybrid.latency_total_sec },
  ];

  return (
    <ResponsiveContainer width="100%" height={220}>
      <PieChart>
        <Pie data={chartData} dataKey="value" nameKey="name" innerRadius={45} outerRadius={80} paddingAngle={4}>
          {chartData.map((entry, index) => (
            <Cell key={entry.name} fill={COLORS[index % COLORS.length]} />
          ))}
        </Pie>
        <Tooltip
          contentStyle={{ backgroundColor: 'rgba(15,17,23,0.95)', borderColor: 'rgba(255,255,255,0.12)', borderRadius: '10px' }}
        />
      </PieChart>
    </ResponsiveContainer>
  );
}

export default App;
