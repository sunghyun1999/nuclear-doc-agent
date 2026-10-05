import { useState, useRef, useEffect } from 'react'
import OntologyGraph from './OntologyGraph'

const API = import.meta.env.DEV ? 'http://localhost:8000' : ''

interface Source {
  document: string
  page: number
  relevance: number
  clauses?: string
}

interface Confidence {
  level: 'high' | 'medium' | 'low'
  score: number
}

interface Message {
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  confidence?: Confidence
}

interface DocInfo {
  source: string
  chunks: number
  strategy: string
}

interface EvalCaseResult {
  question: string
  retrieval_recall_no_rerank: number
  retrieval_recall_with_rerank: number
  rerank_helped: boolean
  answer_score: number
  retrieved_pages: number[]
  expected_pages: number[]
}

interface EvalResult {
  total_cases: number
  avg_retrieval_recall: number
  avg_answer_score: number
  rerank_improvement_rate: string
  results: EvalCaseResult[]
}

interface ChunkingComparison {
  comparison: Record<string, { total_chunks: number; avg_retrieval_recall: number }>
  total_eval_cases: number
  recommendation: string
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [docs, setDocs] = useState<DocInfo[]>([])
  const [totalChunks, setTotalChunks] = useState(0)
  const [strategy, setStrategy] = useState('fixed_size')
  const [useRerank, setUseRerank] = useState(true)
  const [tab, setTab] = useState<'chat' | 'eval' | 'ontology'>('chat')
  const [evalResult, setEvalResult] = useState<EvalResult | null>(null)
  const [chunkingResult, setChunkingResult] = useState<ChunkingComparison | null>(null)
  const [evaluating, setEvaluating] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => { fetchDocs() }, [])
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function fetchDocs() {
    try {
      const res = await fetch(`${API}/documents`)
      const data = await res.json()
      setDocs(data.documents)
      setTotalChunks(data.total_chunks)
    } catch { /* ignore */ }
  }

  async function handleUpload(file: File) {
    setUploading(true)
    const formData = new FormData()
    formData.append('file', file)
    try {
      const res = await fetch(`${API}/documents/upload?strategy=${strategy}`, {
        method: 'POST', body: formData,
      })
      const data = await res.json()
      if (res.ok) {
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: `"${data.filename}" uploaded.\n${data.total_pages} pages, ${data.total_chunks} chunks (${data.strategy}).`,
        }])
        fetchDocs()
      } else { alert(data.detail || 'Upload failed') }
    } catch { alert('Upload failed — is the backend running?') }
    finally { setUploading(false) }
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    const question = input.trim()
    if (!question || loading) return
    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: question }])
    setLoading(true)
    try {
      const res = await fetch(`${API}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, top_k: 5, use_rerank: useRerank }),
      })
      const data = await res.json()
      if (res.ok) {
        setMessages(prev => [...prev, {
          role: 'assistant', content: data.answer, sources: data.sources, confidence: data.confidence,
        }])
      } else {
        setMessages(prev => [...prev, {
          role: 'assistant', content: `Error: ${data.detail}`,
        }])
      }
    } catch {
      setMessages(prev => [...prev, {
        role: 'assistant', content: 'Error: Could not connect to backend.',
      }])
    } finally { setLoading(false) }
  }

  async function pollResult(endpoint: string, setter: (d: any) => void) {
    while (true) {
      await new Promise(r => setTimeout(r, 3000))
      try {
        const res = await fetch(`${API}${endpoint}`)
        const data = await res.json()
        if (data.status === 'done') { setter(data.result); return }
        if (data.status === 'error') { alert(data.result?.error || 'Error'); return }
      } catch { return }
    }
  }

  async function runEval() {
    setEvaluating(true)
    setEvalResult(null)
    try {
      await fetch(`${API}/evaluate`, { method: 'POST' })
      await pollResult('/evaluate/result', setEvalResult)
    } catch { alert('Evaluation failed') }
    finally { setEvaluating(false) }
  }

  async function runChunkingComparison() {
    setEvaluating(true)
    setChunkingResult(null)
    try {
      await fetch(`${API}/evaluate/chunking-comparison`, { method: 'POST' })
      await pollResult('/evaluate/chunking-result', setChunkingResult)
    } catch { alert('Comparison failed') }
    finally { setEvaluating(false) }
  }

  return (
    <div style={s.container}>
      <header style={s.header}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <div>
            <h1 style={s.title}>Nuclear Doc Agent</h1>
            <p style={s.subtitle}>AI-Assisted Regulatory Review for SMR Safety Analysis</p>
          </div>
          <div style={{ marginLeft: 'auto', display: 'flex', gap: 4 }}>
            <button
              style={{ ...s.tabBtn, ...(tab === 'chat' ? s.tabActive : {}) }}
              onClick={() => setTab('chat')}
            >Chat</button>
            <button
              style={{ ...s.tabBtn, ...(tab === 'eval' ? s.tabActive : {}) }}
              onClick={() => setTab('eval')}
            >Evaluation</button>
            <button
              style={{ ...s.tabBtn, ...(tab === 'ontology' ? s.tabActive : {}) }}
              onClick={() => setTab('ontology')}
            >Ontology</button>
          </div>
        </div>
      </header>

      <div style={s.main}>
        {/* Sidebar */}
        <aside style={s.sidebar}>
          <h3 style={s.sidebarTitle}>Documents</h3>
          <div style={s.uploadSection}>
            <select value={strategy} onChange={e => setStrategy(e.target.value)} style={s.select}>
              <option value="fixed_size">Fixed Size Chunking</option>
              <option value="paragraph">Paragraph Chunking</option>
            </select>
            <button
              style={{ ...s.uploadBtn, opacity: uploading ? 0.6 : 1 }}
              onClick={() => fileInputRef.current?.click()}
              disabled={uploading}
            >{uploading ? 'Uploading...' : 'Upload PDF'}</button>
            <input ref={fileInputRef} type="file" accept=".pdf" hidden
              onChange={e => { const f = e.target.files?.[0]; if (f) handleUpload(f); e.target.value = '' }} />
          </div>

          <label style={s.toggle}>
            <input type="checkbox" checked={useRerank} onChange={e => setUseRerank(e.target.checked)} />
            <span style={{ marginLeft: 6, fontSize: 13 }}>Re-ranking</span>
          </label>

          <div style={s.docList}>
            {docs.length === 0 ? (
              <p style={s.emptyText}>No documents yet.</p>
            ) : docs.map((doc, i) => (
              <div key={i} style={s.docItem}>
                <span style={s.docName}>{doc.source}</span>
                <span style={s.docMeta}>{doc.chunks} chunks &middot; {doc.strategy}</span>
              </div>
            ))}
          </div>
          <p style={s.totalChunks}>Total indexed: {totalChunks} chunks</p>
        </aside>

        {/* Main content */}
        <div style={s.chatArea}>
          {tab === 'ontology' ? (
            <OntologyGraph />
          ) : tab === 'chat' ? (
            <>
              <div style={s.messages}>
                {messages.length === 0 && (
                  <div style={s.emptyChat}>
                    <p style={{ fontSize: 16, color: '#aaa' }}>Upload IAEA SSR-2/1, NRC 10 CFR, or other nuclear regulatory PDFs</p>
                    <p style={{ fontSize: 13, color: '#666', marginTop: 4 }}>AI-assisted document review with full source traceability</p>
                    <p style={s.examples}>Try: &quot;What are the defence-in-depth requirements for reactor coolant systems?&quot;</p>
                  </div>
                )}
                {messages.map((msg, i) => (
                  <div key={i} style={{ ...s.message, ...(msg.role === 'user' ? s.userMsg : s.assistantMsg) }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <div style={s.msgRole}>{msg.role === 'user' ? 'You' : 'Agent'}</div>
                      {msg.confidence && (
                        <span style={{
                          fontSize: 10, fontWeight: 700, padding: '2px 8px', borderRadius: 4,
                          background: msg.confidence.level === 'high' ? '#1b5e20' : msg.confidence.level === 'medium' ? '#e65100' : '#b71c1c',
                          color: '#fff', textTransform: 'uppercase', letterSpacing: 0.5,
                        }}>
                          {msg.confidence.level} confidence ({(msg.confidence.score * 100).toFixed(0)}%)
                        </span>
                      )}
                    </div>
                    <div style={s.msgContent}>{msg.content}</div>
                    {msg.sources && msg.sources.length > 0 && (
                      <div style={s.sources}>
                        <strong>Sources:</strong>
                        {msg.sources.map((src, j) => (
                          <span key={j} style={s.sourceTag}>
                            {src.document} p.{src.page}{src.clauses ? ` [${src.clauses}]` : ''} ({(src.relevance * 100).toFixed(1)}%)
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
                {loading && (
                  <div style={{ ...s.message, ...s.assistantMsg }}>
                    <div style={s.msgRole}>Agent</div>
                    <div style={s.msgContent}>Thinking...</div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>
              <form onSubmit={handleSubmit} style={s.inputArea}>
                <input value={input} onChange={e => setInput(e.target.value)}
                  placeholder="Ask about nuclear regulations..." style={s.input} disabled={loading} />
                <button type="submit" style={{ ...s.sendBtn, opacity: loading || !input.trim() ? 0.5 : 1 }}
                  disabled={loading || !input.trim()}>Send</button>
              </form>
            </>
          ) : (
            /* Evaluation Tab */
            <div style={s.evalContainer}>
              <div style={s.evalHeader}>
                <h2 style={s.evalTitle}>RAG Pipeline Evaluation</h2>
                <div style={{ display: 'flex', gap: 8 }}>
                  <button style={s.evalBtn} onClick={runEval} disabled={evaluating}>
                    {evaluating ? 'Running...' : 'Run Evaluation'}
                  </button>
                  <button style={{ ...s.evalBtn, background: '#7c4dff' }}
                    onClick={runChunkingComparison} disabled={evaluating}>
                    {evaluating ? 'Running...' : 'Compare Chunking'}
                  </button>
                </div>
              </div>

              {evalResult && (
                <div style={s.evalResults}>
                  <div style={s.metricsRow}>
                    <div style={s.metricCard}>
                      <div style={s.metricValue}>{(evalResult.avg_retrieval_recall * 100).toFixed(1)}%</div>
                      <div style={s.metricLabel}>Avg Retrieval Recall</div>
                    </div>
                    <div style={s.metricCard}>
                      <div style={s.metricValue}>{(evalResult.avg_answer_score * 100).toFixed(1)}%</div>
                      <div style={s.metricLabel}>Avg Answer Quality</div>
                    </div>
                    <div style={s.metricCard}>
                      <div style={s.metricValue}>{evalResult.rerank_improvement_rate}</div>
                      <div style={s.metricLabel}>Re-rank Improved</div>
                    </div>
                    <div style={s.metricCard}>
                      <div style={s.metricValue}>{evalResult.total_cases}</div>
                      <div style={s.metricLabel}>Test Cases</div>
                    </div>
                  </div>

                  <h3 style={{ color: '#aaa', fontSize: 14, margin: '20px 0 10px' }}>Per-Question Results</h3>
                  <table style={s.table}>
                    <thead>
                      <tr>
                        <th style={s.th}>Question</th>
                        <th style={s.th}>Recall (no rerank)</th>
                        <th style={s.th}>Recall (reranked)</th>
                        <th style={s.th}>Answer Score</th>
                        <th style={s.th}>Rerank Helped</th>
                      </tr>
                    </thead>
                    <tbody>
                      {evalResult.results.map((r, i) => (
                        <tr key={i}>
                          <td style={s.td}>{r.question.slice(0, 60)}...</td>
                          <td style={s.td}>{(r.retrieval_recall_no_rerank * 100).toFixed(0)}%</td>
                          <td style={s.td}>{(r.retrieval_recall_with_rerank * 100).toFixed(0)}%</td>
                          <td style={s.td}>{(r.answer_score * 100).toFixed(0)}%</td>
                          <td style={s.td}>{r.rerank_helped ? 'Yes' : 'No'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {chunkingResult && (
                <div style={{ ...s.evalResults, marginTop: 20 }}>
                  <h3 style={{ color: '#4fc3f7', fontSize: 16, margin: '0 0 12px' }}>
                    Chunking Strategy Comparison
                  </h3>
                  <div style={s.metricsRow}>
                    {Object.entries(chunkingResult.comparison).map(([strategy, data]) => (
                      <div key={strategy} style={{
                        ...s.metricCard,
                        border: chunkingResult.recommendation === strategy
                          ? '2px solid #4fc3f7' : '1px solid #2a2d35',
                      }}>
                        <div style={s.metricValue}>
                          {(data.avg_retrieval_recall * 100).toFixed(1)}%
                        </div>
                        <div style={s.metricLabel}>
                          {strategy === 'fixed_size' ? 'Fixed Size' : 'Paragraph'}
                        </div>
                        <div style={{ fontSize: 11, color: '#666', marginTop: 4 }}>
                          {data.total_chunks} chunks
                        </div>
                        {chunkingResult.recommendation === strategy && (
                          <div style={{ fontSize: 11, color: '#4fc3f7', marginTop: 4 }}>
                            Recommended
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {!evalResult && !chunkingResult && !evaluating && (
                <div style={s.emptyChat}>
                  <p>Run evaluation to measure RAG pipeline quality.</p>
                  <p style={s.examples}>10 golden Q&amp;A pairs &middot; Retrieval recall &middot; Answer quality &middot; Re-ranking impact</p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

const s: Record<string, React.CSSProperties> = {
  container: { fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif', height: '100vh', display: 'flex', flexDirection: 'column', background: '#0f1117', color: '#e0e0e0', margin: 0 },
  header: { padding: '16px 24px', borderBottom: '1px solid #2a2d35', background: '#161922' },
  title: { margin: 0, fontSize: '20px', color: '#4fc3f7', fontWeight: 700 },
  subtitle: { margin: '4px 0 0', fontSize: '13px', color: '#888' },
  tabBtn: { padding: '6px 16px', background: 'transparent', border: '1px solid #2a2d35', borderRadius: '6px', color: '#888', cursor: 'pointer', fontSize: '13px' },
  tabActive: { background: '#1a3a5c', color: '#4fc3f7', borderColor: '#4fc3f7' },
  main: { flex: 1, display: 'flex', overflow: 'hidden' },
  sidebar: { width: '280px', borderRight: '1px solid #2a2d35', padding: '16px', display: 'flex', flexDirection: 'column', background: '#131620' },
  sidebarTitle: { margin: '0 0 12px', fontSize: '14px', color: '#aaa', textTransform: 'uppercase', letterSpacing: '1px' },
  uploadSection: { display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '12px' },
  select: { padding: '8px', background: '#1e2130', border: '1px solid #2a2d35', borderRadius: '6px', color: '#e0e0e0', fontSize: '13px' },
  uploadBtn: { padding: '10px', background: '#4fc3f7', color: '#0f1117', border: 'none', borderRadius: '6px', cursor: 'pointer', fontWeight: 600, fontSize: '13px' },
  toggle: { display: 'flex', alignItems: 'center', padding: '8px 0', marginBottom: '12px', cursor: 'pointer', color: '#ccc' },
  docList: { flex: 1, overflowY: 'auto' },
  docItem: { padding: '10px', background: '#1e2130', borderRadius: '6px', marginBottom: '8px', display: 'flex', flexDirection: 'column', gap: '4px' },
  docName: { fontSize: '13px', fontWeight: 600, wordBreak: 'break-all' },
  docMeta: { fontSize: '11px', color: '#888' },
  emptyText: { fontSize: '13px', color: '#555', textAlign: 'center', marginTop: '20px' },
  totalChunks: { fontSize: '12px', color: '#555', textAlign: 'center', marginTop: '8px' },
  chatArea: { flex: 1, display: 'flex', flexDirection: 'column' },
  messages: { flex: 1, overflowY: 'auto', padding: '20px' },
  emptyChat: { textAlign: 'center', marginTop: '100px', color: '#555' },
  examples: { fontSize: '13px', color: '#4fc3f7', marginTop: '8px' },
  message: { maxWidth: '80%', padding: '12px 16px', borderRadius: '12px', marginBottom: '12px', fontSize: '14px', lineHeight: '1.6', whiteSpace: 'pre-wrap' },
  userMsg: { background: '#1a3a5c', marginLeft: 'auto', borderBottomRightRadius: '4px' },
  assistantMsg: { background: '#1e2130', marginRight: 'auto', borderBottomLeftRadius: '4px' },
  msgRole: { fontSize: '11px', color: '#4fc3f7', fontWeight: 700, marginBottom: '4px', textTransform: 'uppercase' },
  msgContent: { wordBreak: 'break-word' },
  sources: { marginTop: '10px', paddingTop: '8px', borderTop: '1px solid #2a2d35', fontSize: '12px', display: 'flex', flexWrap: 'wrap', gap: '6px', alignItems: 'center' },
  sourceTag: { background: '#0d2137', border: '1px solid #1a3a5c', padding: '2px 8px', borderRadius: '4px', fontSize: '11px', color: '#4fc3f7' },
  inputArea: { padding: '16px 20px', borderTop: '1px solid #2a2d35', display: 'flex', gap: '10px' },
  input: { flex: 1, padding: '12px 16px', background: '#1e2130', border: '1px solid #2a2d35', borderRadius: '8px', color: '#e0e0e0', fontSize: '14px', outline: 'none' },
  sendBtn: { padding: '12px 24px', background: '#4fc3f7', color: '#0f1117', border: 'none', borderRadius: '8px', cursor: 'pointer', fontWeight: 600, fontSize: '14px' },
  // Eval tab
  evalContainer: { flex: 1, overflowY: 'auto', padding: '20px' },
  evalHeader: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' },
  evalTitle: { margin: 0, fontSize: '18px', color: '#e0e0e0' },
  evalBtn: { padding: '10px 20px', background: '#4fc3f7', color: '#0f1117', border: 'none', borderRadius: '6px', cursor: 'pointer', fontWeight: 600, fontSize: '13px' },
  evalResults: { background: '#161922', borderRadius: '8px', padding: '20px', border: '1px solid #2a2d35' },
  metricsRow: { display: 'flex', gap: '12px', flexWrap: 'wrap' },
  metricCard: { flex: 1, minWidth: '120px', background: '#1e2130', borderRadius: '8px', padding: '16px', textAlign: 'center', border: '1px solid #2a2d35' },
  metricValue: { fontSize: '24px', fontWeight: 700, color: '#4fc3f7' },
  metricLabel: { fontSize: '12px', color: '#888', marginTop: '4px' },
  table: { width: '100%', borderCollapse: 'collapse', marginTop: '8px' },
  th: { textAlign: 'left', padding: '8px', borderBottom: '1px solid #2a2d35', fontSize: '12px', color: '#888' },
  td: { padding: '8px', borderBottom: '1px solid #1e2130', fontSize: '13px', color: '#ccc' },
}
