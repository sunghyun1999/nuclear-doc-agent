import { useState, useEffect, useRef, useCallback } from 'react'

const API = import.meta.env.DEV ? 'http://localhost:8000' : ''

const CAT_COLORS: Record<string, string> = {
  'Management': '#ff6b6b',
  'Safety Fundamentals': '#ffa726',
  'Design Basis': '#ffee58',
  'Design Principles': '#66bb6a',
  'Reactor & Coolant': '#42a5f5',
  'Containment': '#ab47bc',
  'I&C Systems': '#26c6da',
  'Support Systems': '#8d6e63',
  'Waste & Radiation': '#ef5350',
}

interface Node {
  id: number; title: string; category: string; page: number
  x: number; y: number; vx: number; vy: number
}
interface Edge {
  source: number; target: number; relation: string; type: string
}
interface GraphData {
  nodes: Node[]; edges: Edge[]
  categories: string[]
  stats: { total_requirements: number; domain_edges: number; semantic_edges: number }
}

export default function OntologyGraph() {
  const [graph, setGraph] = useState<GraphData | null>(null)
  const [loading, setLoading] = useState(false)
  const [hovered, setHovered] = useState<Node | null>(null)
  const [hoveredEdges, setHoveredEdges] = useState<Edge[]>([])
  const [selectedCat, setSelectedCat] = useState<string | null>(null)
  const [edgeFilter, setEdgeFilter] = useState<'all' | 'domain' | 'semantic'>('all')
  const svgRef = useRef<SVGSVGElement>(null)
  const animRef = useRef<number>(0)

  useEffect(() => { fetchGraph() }, [])

  async function fetchGraph() {
    setLoading(true)
    try {
      const res = await fetch(`${API}/requirements/graph`)
      const data = await res.json()
      // Initialize positions in a circle by category
      const catGroups: Record<string, number[]> = {}
      data.nodes.forEach((n: any) => {
        if (!catGroups[n.category]) catGroups[n.category] = []
        catGroups[n.category].push(n.id)
      })
      const cats = Object.keys(catGroups)
      const cx = 500, cy = 350, R = 280
      data.nodes.forEach((n: any) => {
        const catIdx = cats.indexOf(n.category)
        const catNodes = catGroups[n.category]
        const idxInCat = catNodes.indexOf(n.id)
        const catAngle = (catIdx / cats.length) * Math.PI * 2 - Math.PI / 2
        const spread = 0.4
        const angle = catAngle + (idxInCat - catNodes.length / 2) * spread / catNodes.length
        const r = R + (idxInCat % 2) * 30
        n.x = cx + Math.cos(angle) * r
        n.y = cy + Math.sin(angle) * r
        n.vx = 0; n.vy = 0
      })
      setGraph(data)
    } catch { /* ignore */ }
    finally { setLoading(false) }
  }

  // Simple force simulation
  const simulate = useCallback(() => {
    if (!graph) return
    const nodes = graph.nodes
    const edges = graph.edges
    const cx = 500, cy = 350

    for (let iter = 0; iter < 3; iter++) {
      // Repulsion between all nodes
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[j].x - nodes[i].x
          const dy = nodes[j].y - nodes[i].y
          const dist = Math.sqrt(dx * dx + dy * dy) || 1
          const force = 800 / (dist * dist)
          nodes[i].vx -= (dx / dist) * force
          nodes[i].vy -= (dy / dist) * force
          nodes[j].vx += (dx / dist) * force
          nodes[j].vy += (dy / dist) * force
        }
      }
      // Attraction along edges
      for (const e of edges) {
        const a = nodes.find(n => n.id === e.source)
        const b = nodes.find(n => n.id === e.target)
        if (!a || !b) continue
        const dx = b.x - a.x
        const dy = b.y - a.y
        const dist = Math.sqrt(dx * dx + dy * dy) || 1
        const force = (dist - 120) * 0.003
        a.vx += (dx / dist) * force
        a.vy += (dy / dist) * force
        b.vx -= (dx / dist) * force
        b.vy -= (dy / dist) * force
      }
      // Center gravity
      for (const n of nodes) {
        n.vx += (cx - n.x) * 0.002
        n.vy += (cy - n.y) * 0.002
        n.vx *= 0.85; n.vy *= 0.85
        n.x += n.vx; n.y += n.vy
        n.x = Math.max(30, Math.min(970, n.x))
        n.y = Math.max(30, Math.min(670, n.y))
      }
    }
    setGraph({ ...graph, nodes: [...nodes] })
    animRef.current = requestAnimationFrame(simulate)
  }, [graph])

  useEffect(() => {
    if (graph && graph.nodes.length > 0) {
      animRef.current = requestAnimationFrame(simulate)
      const timeout = setTimeout(() => cancelAnimationFrame(animRef.current), 8000)
      return () => { cancelAnimationFrame(animRef.current); clearTimeout(timeout) }
    }
  }, [graph?.stats?.total_requirements])

  function handleNodeHover(node: Node | null) {
    setHovered(node)
    if (node && graph) {
      setHoveredEdges(graph.edges.filter(e => e.source === node.id || e.target === node.id))
    } else {
      setHoveredEdges([])
    }
  }

  const filteredEdges = graph?.edges.filter(e => edgeFilter === 'all' || e.type === edgeFilter) || []
  const visibleNodes = graph?.nodes.filter(n => !selectedCat || n.category === selectedCat) || []
  const visibleNodeIds = new Set(visibleNodes.map(n => n.id))
  const visibleEdges = filteredEdges.filter(e => visibleNodeIds.has(e.source) && visibleNodeIds.has(e.target))

  if (loading) return <div style={{ padding: 40, color: '#888', textAlign: 'center' }}>Loading requirement ontology...</div>
  if (!graph) return <div style={{ padding: 40, color: '#555', textAlign: 'center' }}>No documents uploaded. Upload a regulatory PDF first.</div>

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* Controls */}
      <div style={{ padding: '12px 20px', borderBottom: '1px solid #2a2d35', display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
        <span style={{ fontSize: 14, color: '#aaa', fontWeight: 600 }}>Requirement Ontology</span>
        <span style={{ fontSize: 11, color: '#555' }}>
          {graph.stats.total_requirements} requirements &middot; {graph.stats.domain_edges} domain + {graph.stats.semantic_edges} semantic edges
        </span>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 4 }}>
          {['all', 'domain', 'semantic'].map(f => (
            <button key={f} onClick={() => setEdgeFilter(f as any)} style={{
              padding: '4px 12px', fontSize: 11, borderRadius: 4, cursor: 'pointer', border: '1px solid #2a2d35',
              background: edgeFilter === f ? '#1a3a5c' : 'transparent',
              color: edgeFilter === f ? '#4fc3f7' : '#888',
            }}>{f}</button>
          ))}
        </div>
      </div>

      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {/* Legend */}
        <div style={{ width: 200, padding: 12, borderRight: '1px solid #2a2d35', overflowY: 'auto' }}>
          <div style={{ fontSize: 11, color: '#888', marginBottom: 8, textTransform: 'uppercase' }}>Categories</div>
          <div style={{ cursor: 'pointer', padding: '4px 8px', borderRadius: 4, fontSize: 12, color: selectedCat ? '#888' : '#4fc3f7', background: selectedCat ? 'transparent' : '#1a3a5c', marginBottom: 4 }}
            onClick={() => setSelectedCat(null)}>All ({graph.nodes.length})</div>
          {graph.categories.map(cat => {
            const count = graph.nodes.filter(n => n.category === cat).length
            return (
              <div key={cat} onClick={() => setSelectedCat(selectedCat === cat ? null : cat)} style={{
                cursor: 'pointer', padding: '4px 8px', borderRadius: 4, fontSize: 12, marginBottom: 2,
                display: 'flex', alignItems: 'center', gap: 6,
                background: selectedCat === cat ? '#1e2130' : 'transparent',
                color: selectedCat === cat ? '#fff' : '#aaa',
              }}>
                <span style={{ width: 8, height: 8, borderRadius: '50%', background: CAT_COLORS[cat] || '#888', flexShrink: 0 }} />
                {cat} ({count})
              </div>
            )
          })}

          {hovered && (
            <div style={{ marginTop: 16, padding: 10, background: '#1e2130', borderRadius: 6, border: '1px solid #2a2d35' }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: '#4fc3f7' }}>Req {hovered.id}</div>
              <div style={{ fontSize: 11, color: '#ccc', marginTop: 4 }}>{hovered.title}</div>
              <div style={{ fontSize: 10, color: '#888', marginTop: 4 }}>p. {hovered.page} &middot; {hovered.category}</div>
              {hoveredEdges.length > 0 && (
                <div style={{ marginTop: 8, fontSize: 10, color: '#aaa' }}>
                  <div style={{ fontWeight: 600, marginBottom: 4 }}>Connections ({hoveredEdges.length}):</div>
                  {hoveredEdges.slice(0, 8).map((e, i) => {
                    const otherId = e.source === hovered.id ? e.target : e.source
                    return <div key={i} style={{ marginBottom: 2 }}>Req {otherId}: {e.relation.slice(0, 50)}</div>
                  })}
                  {hoveredEdges.length > 8 && <div>+{hoveredEdges.length - 8} more</div>}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Graph SVG */}
        <svg ref={svgRef} viewBox="0 0 1000 700" style={{ flex: 1, background: '#0a0d14' }}>
          {/* Edges */}
          {visibleEdges.map((e, i) => {
            const a = graph.nodes.find(n => n.id === e.source)
            const b = graph.nodes.find(n => n.id === e.target)
            if (!a || !b) return null
            const isHighlighted = hovered && (e.source === hovered.id || e.target === hovered.id)
            return (
              <line key={i} x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                stroke={isHighlighted ? '#4fc3f7' : e.type === 'domain' ? '#2a4a6b' : '#2a2d35'}
                strokeWidth={isHighlighted ? 2 : 1}
                opacity={hovered ? (isHighlighted ? 1 : 0.15) : 0.5}
              />
            )
          })}
          {/* Nodes */}
          {visibleNodes.map(n => {
            const isHovered = hovered?.id === n.id
            const isConnected = hovered && hoveredEdges.some(e => e.source === n.id || e.target === n.id)
            const dimmed = hovered && !isHovered && !isConnected
            return (
              <g key={n.id}
                onMouseEnter={() => handleNodeHover(n)}
                onMouseLeave={() => handleNodeHover(null)}
                style={{ cursor: 'pointer' }}
              >
                <circle cx={n.x} cy={n.y} r={isHovered ? 12 : 8}
                  fill={CAT_COLORS[n.category] || '#888'}
                  opacity={dimmed ? 0.15 : 1}
                  stroke={isHovered ? '#fff' : 'none'} strokeWidth={2}
                />
                <text x={n.x} y={n.y - 12} textAnchor="middle"
                  fontSize={isHovered ? 11 : 9} fill={dimmed ? '#333' : '#ccc'}
                  fontWeight={isHovered ? 700 : 400}
                >{n.id}</text>
              </g>
            )
          })}
        </svg>
      </div>
    </div>
  )
}
