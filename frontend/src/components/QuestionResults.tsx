import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Experiment } from '../api'
import { LABELS, label } from '../evalLabels'

type Page = [string, number]

export interface Row {
  id: string
  question: string
  answer: string
  reference: string
  cited_pages: Page[]
  reference_pages: Page[]
  faithfulness: number | null
  answer_relevancy: number | null
  context_recall: number | null
  'hit@5': number | null
  cited_page_hit: number | null
  latency_s: number | null
  tokens: number | null
}

const COMPANY: Record<string, string> = { indigo: 'IndiGo', tcs: 'TCS', infosys: 'Infosys' }
const pageName = ([report, page]: Page) => {
  const [company, year] = report.split('_')
  return `${COMPANY[company] ?? company} ${year ?? ''} p.${page}`
}
const samePage = (a: Page, b: Page) => a[0] === b[0] && a[1] === b[1]

const SCORES = [
  ['faithfulness', 'Faithfulness'],
  ['answer_relevancy', 'Relevancy'],
  ['context_recall', 'Recall'],
] as const

// Below this a score is flagged, so weak answers stand out when scanning the list.
const LOW = 0.5

function Score({ name, value }: { name: string; value: number | null }) {
  const low = value != null && value < LOW
  return (
    <span className={`whitespace-nowrap ${low ? 'text-warn' : 'text-ink-2'}`}>
      <span className="text-muted">{name} </span>
      <span className="font-mono">{value != null ? value.toFixed(2) : '–'}</span>
    </span>
  )
}

function PageList({ pages, reference }: { pages: Page[]; reference: Page[] }) {
  if (!pages.length) return <span className="text-muted">none</span>
  return (
    <span className="flex flex-wrap gap-1">
      {pages.map((p) => {
        const hit = reference.some((r) => samePage(p, r))
        return (
          <span
            key={p.join(':')}
            className={`rounded px-1.5 py-0.5 font-mono text-xs ${hit ? 'bg-accent-soft text-accent' : 'bg-surface-2 text-ink-2'}`}
            title={hit ? 'A reference page' : 'Not a reference page'}
          >
            {pageName(p)}
          </span>
        )
      })}
    </span>
  )
}

function QuestionRow({ row }: { row: Row }) {
  const hit = row.cited_page_hit === 1
  return (
    <details className="group border-b border-line last:border-0">
      <summary className="flex cursor-pointer list-none flex-col gap-1 px-4 py-3 select-none hover:bg-surface-2 sm:flex-row sm:items-center sm:gap-4">
        <span className="flex flex-1 items-start gap-2 text-sm text-ink">
          <span className="mt-0.5 text-muted transition-transform group-open:rotate-90" aria-hidden>
            ›
          </span>
          {row.question}
        </span>
        <span className="flex flex-wrap gap-x-3 gap-y-1 pl-5 text-xs sm:pl-0">
          {SCORES.map(([k, name]) => (
            <Score key={k} name={name} value={row[k]} />
          ))}
          <span className={`whitespace-nowrap ${hit ? 'text-ink-2' : 'text-warn'}`}>
            {hit ? '✓ right page cited' : '✗ right page not cited'}
          </span>
        </span>
      </summary>
      <div className="space-y-4 px-4 pb-4 pl-9 text-sm">
        <div>
          <div className="mb-1 text-xs font-medium text-muted">Answer</div>
          <div className="prose-answer text-ink">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{row.answer}</ReactMarkdown>
          </div>
        </div>
        <div>
          <div className="mb-1 text-xs font-medium text-muted">Reference answer</div>
          <p className="text-ink-2">{row.reference}</p>
        </div>
        <div className="grid gap-3 sm:grid-cols-2">
          <div>
            <div className="mb-1 text-xs font-medium text-muted">Pages the answer cited</div>
            <PageList pages={row.cited_pages} reference={row.reference_pages} />
          </div>
          <div>
            <div className="mb-1 text-xs font-medium text-muted">Reference pages</div>
            <PageList pages={row.reference_pages} reference={row.reference_pages} />
          </div>
        </div>
        <div className="text-xs text-muted">
          {row.latency_s?.toFixed(1)}s · {row.tokens?.toLocaleString('en-IN') ?? '–'} tokens · <span className="font-mono">{row.id}</span>
        </div>
      </div>
    </details>
  )
}

export function QuestionResults({ experiments, rows }: { experiments: Experiment[]; rows: Record<string, Row[]> }) {
  const runs = experiments.filter((e) => rows[e.name]?.length)
  const [run, setRun] = useState(runs[0]?.name ?? '')
  const [weakFirst, setWeakFirst] = useState(false)
  if (!runs.length) return null

  const list = [...(rows[run] ?? [])]
  if (weakFirst) list.sort((a, b) => (a.faithfulness ?? 0) - (b.faithfulness ?? 0))
  const low = list.filter((r) => SCORES.some(([k]) => r[k] != null && r[k]! < LOW)).length

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <div role="tablist" aria-label="Setup" className="flex flex-wrap gap-1 rounded-lg bg-surface-2 p-1">
          {runs.map((e) => (
            <button
              key={e.name}
              type="button"
              role="tab"
              aria-selected={run === e.name}
              onClick={() => setRun(e.name)}
              className={`rounded-md px-3 py-1.5 text-xs transition-colors ${
                run === e.name ? 'bg-surface font-medium text-ink shadow-sm' : 'text-ink-2 hover:text-ink'
              }`}
            >
              {label(e.name)}
            </button>
          ))}
        </div>
        <label className="ml-auto flex items-center gap-2 text-xs text-ink-2">
          <input type="checkbox" checked={weakFirst} onChange={(e) => setWeakFirst(e.target.checked)} />
          Lowest faithfulness first
        </label>
      </div>
      <p className="mb-2 text-xs text-muted">
        {LABELS[run]?.[1] ?? ''} · {list.length} questions · {low} with a score under {LOW}
      </p>
      <div className="rounded-xl border border-line bg-surface">
        {list.map((r) => (
          <QuestionRow key={r.id} row={r} />
        ))}
      </div>
    </div>
  )
}
