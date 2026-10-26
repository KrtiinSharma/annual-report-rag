import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { pdfLink, type Answer, type Citation } from '../api'

interface Props {
  answer: Answer
  pdfAvailable: Set<string>
  active: number | null
  onCite: (id: number) => void
}

// "[2]" / "[1, 3]" / "[4-5]" -> markdown links the renderer turns into citation chips
function linkCitations(text: string, max: number) {
  return text.replace(/\[(\d+(?:\s*[,–-]\s*\d+)*)\]/g, (whole, group: string) => {
    const ids: number[] = []
    for (const part of group.split(/\s*,\s*/)) {
      const range = part.split(/\s*[–-]\s*/).map(Number)
      if (range.length === 2) for (let i = range[0]; i <= range[1]; i++) ids.push(i)
      else ids.push(range[0])
    }
    const valid = ids.filter((i) => i >= 1 && i <= max)
    return valid.length ? valid.map((i) => `[${i}](#cite-${i})`).join('') : whole
  })
}

export function AnswerView({ answer, pdfAvailable, active, onCite }: Props) {
  const cited = answer.citations.filter((c) => c.cited)
  const others = answer.citations.filter((c) => !c.cited)
  return (
    <div className="space-y-6">
      <section className="rounded-xl border border-line bg-surface p-5 sm:p-6">
        <div className="mb-3 flex flex-wrap items-center gap-2 text-xs text-muted">
          <span className="rounded-full bg-accent-soft px-2 py-0.5 font-medium text-accent">
            {answer.mode === 'vectorless' ? 'Vectorless RAG' : 'Hybrid RAG'}
          </span>
          <span>{answer.timings.total?.toFixed(1)}s</span>
          <span>·</span>
          <span>
            {cited.length} cited of {answer.citations.length} passages
          </span>
        </div>
        <div className="prose-answer text-[15px] text-ink">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              a: ({ href, children }) => {
                const m = href?.match(/^#cite-(\d+)$/)
                if (!m) return <a href={href} target="_blank" rel="noreferrer" className="text-accent underline">{children}</a>
                const id = Number(m[1])
                return (
                  <button
                    type="button"
                    onClick={() => onCite(id)}
                    className={`mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded px-1 align-baseline font-mono text-[11px] font-medium transition-colors ${
                      active === id ? 'bg-accent text-accent-ink' : 'bg-accent-soft text-accent hover:bg-accent hover:text-accent-ink'
                    }`}
                    aria-label={`Show source ${id}`}
                  >
                    {id}
                  </button>
                )
              },
            }}
          >
            {linkCitations(answer.answer, answer.citations.length)}
          </ReactMarkdown>
        </div>
      </section>

      {cited.length > 0 && (
        <SourceList title="Cited sources" items={cited} pdfAvailable={pdfAvailable} active={active} />
      )}
      {others.length > 0 && (
        <details className="group">
          <summary className="cursor-pointer text-sm font-medium text-ink-2 select-none">
            Other retrieved passages ({others.length})
          </summary>
          <div className="mt-3">
            <SourceList items={others} pdfAvailable={pdfAvailable} active={active} />
          </div>
        </details>
      )}
    </div>
  )
}

function SourceList({ title, items, pdfAvailable, active }: { title?: string; items: Citation[]; pdfAvailable: Set<string>; active: number | null }) {
  return (
    <section>
      {title && <h3 className="mb-3 text-sm font-semibold text-ink-2">{title}</h3>}
      <div className="grid gap-3 md:grid-cols-2">
        {items.map((c) => (
          <SourceCard key={c.id} c={c} hasPdf={pdfAvailable.has(c.report_id)} active={active === c.id} />
        ))}
      </div>
    </section>
  )
}

function SourceCard({ c, hasPdf, active }: { c: Citation; hasPdf: boolean; active: boolean }) {
  return (
    <article
      id={`source-${c.id}`}
      className={`flex flex-col rounded-xl border bg-surface p-4 transition-shadow ${
        active ? 'border-accent shadow-[0_0_0_3px_var(--accent-soft)]' : 'border-line'
      }`}
    >
      <header className="mb-2 flex items-start gap-2">
        <span className="mt-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded bg-accent-soft px-1 font-mono text-[11px] font-medium text-accent">
          {c.id}
        </span>
        <div className="min-w-0 flex-1">
          <div className="text-sm font-semibold text-ink">
            {c.company_name} · {c.fiscal_year}
          </div>
          <div className="truncate text-xs text-muted" title={c.section}>
            Page {c.page}
            {c.section ? ` · ${c.section}` : ''}
            {c.chunk_type === 'table' ? ' · table' : ''}
          </div>
        </div>
        <a
          href={pdfLink(c, hasPdf)}
          target="_blank"
          rel="noreferrer"
          className="shrink-0 rounded-md border border-line px-2 py-1 text-xs font-medium text-ink-2 hover:border-accent hover:text-accent"
        >
          Open p.{c.page} ↗
        </a>
      </header>
      <div className="snippet max-h-48 overflow-auto rounded-md bg-surface-2 p-3 text-xs leading-relaxed whitespace-pre-wrap text-ink-2">
        {c.snippet}
      </div>
      {c.matched_by.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1">
          {c.matched_by.slice(0, 6).map((m) => (
            <span key={m} className="rounded bg-surface-2 px-1.5 py-0.5 font-mono text-[10px] text-muted">
              {m}
            </span>
          ))}
          <span className="ml-auto font-mono text-[10px] text-muted">score {c.score.toFixed(3)}</span>
        </div>
      )}
    </article>
  )
}
