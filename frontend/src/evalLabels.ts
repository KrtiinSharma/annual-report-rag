// Readable names for the evaluation runs in docs/results: [title, scope].
export const LABELS: Record<string, [string, string]> = {
  'hybrid-structure': ['Hybrid RAG, structure-aware chunks', 'All companies'],
  'hybrid-fixed': ['Hybrid RAG, fixed-size chunks', 'All companies'],
  'vector-indigo': ['Vector RAG', 'IndiGo FY24 report'],
  vectorless: ['Vectorless RAG', 'IndiGo FY24 report'],
  'retrieval-structure': ['Structure-aware chunks', 'Retrieval only'],
  'retrieval-fixed': ['Fixed-size chunks', 'Retrieval only'],
}

export const label = (name: string) => LABELS[name]?.[0] ?? name
