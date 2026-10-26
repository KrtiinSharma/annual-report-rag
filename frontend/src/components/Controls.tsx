import type { AskOptions, Catalog } from '../api'

export type Mode = 'hybrid' | 'vectorless'

interface Props {
  catalog: Catalog
  mode: Mode
  setMode: (m: Mode) => void
  opts: AskOptions
  setOpts: (o: AskOptions) => void
  vectorlessReport: string
  setVectorlessReport: (id: string) => void
}

function toggle<T>(list: T[], item: T) {
  return list.includes(item) ? list.filter((x) => x !== item) : [...list, item]
}

function Chip({ on, onClick, children }: { on: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={on}
      className={`rounded-full border px-3 py-1 text-xs font-medium transition-colors ${
        on ? 'border-accent bg-accent text-accent-ink' : 'border-line bg-surface text-ink-2 hover:border-accent hover:text-accent'
      }`}
    >
      {children}
    </button>
  )
}

function Switch({ label, hint, on, onChange }: { label: string; hint: string; on: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="flex cursor-pointer items-start justify-between gap-3 py-1.5">
      <span>
        <span className="block text-sm text-ink">{label}</span>
        <span className="block text-xs text-muted">{hint}</span>
      </span>
      <input type="checkbox" className="peer sr-only" checked={on} onChange={(e) => onChange(e.target.checked)} />
      <span className="relative mt-0.5 h-5 w-9 shrink-0 rounded-full bg-line transition-colors peer-checked:bg-accent peer-focus-visible:ring-2 peer-focus-visible:ring-accent after:absolute after:top-0.5 after:left-0.5 after:h-4 after:w-4 after:rounded-full after:bg-surface after:transition-transform peer-checked:after:translate-x-4" />
    </label>
  )
}

const Heading = ({ children }: { children: React.ReactNode }) => (
  <h3 className="mb-2 text-[11px] font-semibold tracking-wider text-muted uppercase">{children}</h3>
)

export function Controls({ catalog, mode, setMode, opts, setOpts, vectorlessReport, setVectorlessReport }: Props) {
  const years = [...new Set(catalog.reports.map((r) => r.fiscal_year))].sort()
  const set = (patch: Partial<AskOptions>) => setOpts({ ...opts, ...patch })
  const reportLabel = (id: string) => {
    const r = catalog.reports.find((x) => x.id === id)
    const c = catalog.companies.find((x) => x.slug === r?.company)
    return r ? `${c?.name ?? r.company} ${r.fiscal_year}` : id
  }

  return (
    <div className="space-y-6">
      <div>
        <Heading>Mode</Heading>
        <div className="grid grid-cols-2 gap-1 rounded-lg bg-surface-2 p-1">
          {(['hybrid', 'vectorless'] as Mode[]).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setMode(m)}
              className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
                mode === m ? 'bg-surface text-ink shadow-sm' : 'text-ink-2 hover:text-ink'
              }`}
            >
              {m === 'hybrid' ? 'Hybrid RAG' : 'Vectorless'}
            </button>
          ))}
        </div>
        <p className="mt-2 text-xs text-muted">
          {mode === 'hybrid'
            ? 'BM25 + vector search over every report, fused and reranked.'
            : 'No embeddings: the LLM reads the contents tree of one report and picks pages.'}
        </p>
      </div>

      {mode === 'hybrid' ? (
        <>
          <div>
            <Heading>Companies</Heading>
            <div className="flex flex-wrap gap-1.5">
              {catalog.companies.map((c) => (
                <Chip key={c.slug} on={opts.companies.includes(c.slug)} onClick={() => set({ companies: toggle(opts.companies, c.slug) })}>
                  {c.name}
                </Chip>
              ))}
            </div>
            <p className="mt-1.5 text-xs text-muted">None selected = detect from the question.</p>
          </div>
          <div>
            <Heading>Fiscal year</Heading>
            <div className="flex flex-wrap gap-1.5">
              {years.map((y) => (
                <Chip key={y} on={opts.fiscal_years.includes(y)} onClick={() => set({ fiscal_years: toggle(opts.fiscal_years, y) })}>
                  {y}
                </Chip>
              ))}
            </div>
          </div>
          <div>
            <Heading>Pipeline</Heading>
            <div className="divide-y divide-line">
              <Switch label="Keyword search" hint="BM25" on={opts.use_bm25} onChange={(v) => set({ use_bm25: v || !opts.use_vector })} />
              <Switch label="Vector search" hint="bge-small embeddings" on={opts.use_vector} onChange={(v) => set({ use_vector: v || !opts.use_bm25 })} />
              <Switch label="Reranker" hint="ms-marco cross-encoder" on={opts.use_rerank} onChange={(v) => set({ use_rerank: v })} />
              <Switch label="Multi-query" hint="LLM rewrites + per-company searches" on={opts.use_multi_query} onChange={(v) => set({ use_multi_query: v })} />
            </div>
          </div>
          <div>
            <Heading>Chunking</Heading>
            <select
              value={opts.strategy}
              onChange={(e) => set({ strategy: e.target.value as AskOptions['strategy'] })}
              className="w-full rounded-lg border border-line bg-surface px-3 py-2 text-sm text-ink"
            >
              <option value="structure">Structure-aware (sections, whole tables)</option>
              <option value="fixed">Fixed size (1000 chars)</option>
            </select>
          </div>
        </>
      ) : (
        <div>
          <Heading>Report</Heading>
          {catalog.vectorless_reports.length ? (
            <select
              value={vectorlessReport}
              onChange={(e) => setVectorlessReport(e.target.value)}
              className="w-full rounded-lg border border-line bg-surface px-3 py-2 text-sm text-ink"
            >
              {catalog.vectorless_reports.map((id) => (
                <option key={id} value={id}>
                  {reportLabel(id)}
                </option>
              ))}
            </select>
          ) : (
            <p className="text-xs text-warn">No contents tree has been built yet.</p>
          )}
        </div>
      )}
    </div>
  )
}
