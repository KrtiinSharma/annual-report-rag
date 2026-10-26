import type { Answer } from '../api'

const STEP_LABELS: Record<string, string> = {
  plan: 'Query planning',
  retrieve: 'Retrieve + rerank',
  select_sections: 'Pick sections',
  select_pages: 'Pick pages',
  generate: 'Generate answer',
}

export function PipelinePanel({ answer, companyName }: { answer: Answer; companyName: (slug: string) => string }) {
  const steps = Object.entries(answer.timings).filter(([k]) => k !== 'total')
  const total = answer.timings.total || 1
  return (
    <section className="rounded-xl border border-line bg-surface p-5">
      <h3 className="mb-4 text-sm font-semibold text-ink-2">How this answer was made</h3>

      {answer.mode === 'hybrid' ? (
        <ol className="space-y-3">
          {answer.plan.map((t, i) => {
            const target = t as { queries: string[]; company: string | null; fiscal_year: string | null }
            return (
              <li key={i} className="rounded-lg bg-surface-2 p-3">
                <div className="mb-1.5 text-xs font-semibold text-ink">
                  Search {i + 1}: {target.company ? companyName(target.company) : 'all companies'}
                  {target.fiscal_year ? ` · ${target.fiscal_year}` : ' · all years'}
                </div>
                <ul className="space-y-1">
                  {target.queries.map((q, j) => (
                    <li key={j} className="flex gap-2 text-xs text-ink-2">
                      <span className="font-mono text-muted">q{j}</span>
                      <span>{q}</span>
                    </li>
                  ))}
                </ul>
              </li>
            )
          })}
        </ol>
      ) : (
        <ol className="space-y-3">
          {answer.plan.map((s, i) => {
            const step = s as { step: string; reasoning: string; chosen: (string | number)[] }
            return (
              <li key={i} className="rounded-lg bg-surface-2 p-3 text-xs">
                <div className="mb-1 font-semibold text-ink">
                  {step.step === 'sections' ? 'Sections chosen from the contents tree' : 'Pages read in full'}
                </div>
                <p className="mb-1.5 text-ink-2">{step.reasoning}</p>
                <div className="flex flex-wrap gap-1">
                  {step.chosen.map((c) => (
                    <span key={String(c)} className="rounded bg-surface px-1.5 py-0.5 font-mono text-[11px] text-ink-2">
                      {step.step === 'pages' ? `p.${c}` : c}
                    </span>
                  ))}
                </div>
              </li>
            )
          })}
        </ol>
      )}

      <h4 className="mt-5 mb-2 text-xs font-semibold text-ink-2">Time per step</h4>
      <div className="space-y-1.5">
        {steps.map(([k, v]) => (
          <div key={k} className="grid grid-cols-[7.5rem_1fr_3rem] items-center gap-2 text-xs">
            <span className="text-ink-2">{STEP_LABELS[k] ?? k}</span>
            <div className="h-2 rounded-full bg-surface-2">
              <div className="h-2 rounded-full bg-series-1" style={{ width: `${Math.max(2, (v / total) * 100)}%` }} />
            </div>
            <span className="text-right font-mono text-muted">{v.toFixed(1)}s</span>
          </div>
        ))}
      </div>
    </section>
  )
}
