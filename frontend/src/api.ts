// Backend base URL: empty in dev (Vite proxies /api), the Cloud Run URL in production.
export const API_URL = (import.meta.env.VITE_API_URL ?? '').replace(/\/$/, '')

export interface Company {
  slug: string
  name: string
  nse: string
  bse: string
  sector: string
  ir_page: string
}

export interface ReportInfo {
  id: string
  company: string
  fiscal_year: string
  source_url: string
  indexed: boolean
  has_pdf: boolean
}

export interface Catalog {
  companies: Company[]
  reports: ReportInfo[]
  vectorless_reports: string[]
  examples: string[]
}

export interface Citation {
  id: number
  report_id: string
  company: string
  company_name: string
  fiscal_year: string
  page: number
  section: string
  chunk_type: string
  snippet: string
  score: number
  source_url: string
  cited: boolean
  matched_by: string[]
}

export interface Answer {
  question: string
  answer: string
  citations: Citation[]
  plan: Record<string, unknown>[]
  timings: Record<string, number>
  mode: 'hybrid' | 'vectorless'
}

export interface AskOptions {
  companies: string[]
  fiscal_years: string[]
  strategy: 'structure' | 'fixed'
  use_bm25: boolean
  use_vector: boolean
  use_rerank: boolean
  use_multi_query: boolean
}

export interface Experiment {
  name: string
  kind: 'rag' | 'retrieval'
  config: Record<string, unknown>
  n: number
  aggregate: Record<string, number | null>
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (typeof body.detail === 'string') detail = body.detail
    } catch {
      /* not JSON */
    }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

export const api = {
  catalog: () => request<Catalog>('/api/reports'),
  ask: (question: string, opts: AskOptions) =>
    request<Answer>('/api/ask', { method: 'POST', body: JSON.stringify({ question, ...opts }) }),
  askVectorless: (question: string, report_id: string) =>
    request<Answer>('/api/ask/vectorless', {
      method: 'POST',
      body: JSON.stringify({ question, report_id }),
    }),
  evals: () => request<{ experiments: Experiment[] }>('/api/eval'),
}

export function pdfLink(c: Pick<Citation, 'report_id' | 'page' | 'source_url'>, hasPdf: boolean) {
  if (hasPdf) return `${API_URL}/api/pdf/${c.report_id}#page=${c.page}`
  return c.source_url.toLowerCase().endsWith('.pdf') ? `${c.source_url}#page=${c.page}` : c.source_url
}
