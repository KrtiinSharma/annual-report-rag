import { useEffect, useState } from 'react'
import { api, type Experiment } from '../api'

const RAGAS = [
  ['faithfulness', 'Faithfulness', 'Answer claims supported by the retrieved passages'],
  ['answer_relevancy', 'Answer relevancy', 'Answer addresses the question asked'],
  ['context_precision', 'Context precision', 'Relevant passages ranked above irrelevant ones'],
  ['context_recall', 'Context recall', 'Reference answer covered by the retrieved passages'],
] as const

const RETRIEVAL = [
  ['hit@1', 'Hit@1'],
  ['hit@3', 'Hit@3'],
  ['hit@5', 'Hit@5'],
  ['hit@10', 'Hit@10'],
  ['mrr', 'MRR'],
] as const

function BarCell({ value }: { value: number | null | undefined }) {
  if (value == null) return <td className="px-3 py-2 text-right text-muted">–</td>
  return (
    <td className="px-3 py-2" title={value.toFixed(3)}>
      <div className="flex items-center gap-2">
        <div className="h-2 flex-1 rounded-full bg-surface-2">
          <div className="h-2 rounded-full bg-series-1" style={{ width: `${Math.max(2, value * 100)}%` }} />
        </div>
        <span className="w-10 text-right font-mono text-xs text-ink">{value.toFixed(2)}</span>
      </div>
    </td>
  )
}

function Table({ rows, cols, extra }: { rows: Experiment[]; cols: readonly (readonly [string, string, ...string[]])[]; extra?: boolean }) {
  return (
    <div className="overflow-x-auto rounded-xl border border-line bg-surface">
      <table className="w-full min-w-[640px] text-sm">
        <thead>
          <tr className="border-b border-line text-left text-xs text-muted">
            <th className="px-3 py-2 font-medium">Experiment</th>
            <th className="px-3 py-2 text-right font-medium">n</th>
            {cols.map(([k, label, hint]) => (
              <th key={k} className="px-3 py-2 font-medium" title={hint}>
                {label}
              </th>
            ))}
            {extra && <th className="px-3 py-2 text-right font-medium">Latency</th>}
          </tr>
        </thead>
        <tbody>
          {rows.map((e) => (
            <tr key={e.name} className="border-b border-line last:border-0">
              <td className="px-3 py-2 font-mono text-xs whitespace-nowrap text-ink">{e.name}</td>
              <td className="px-3 py-2 text-right font-mono text-xs text-muted">{e.n}</td>
              {cols.map(([k]) => (
                <BarCell key={k} value={e.aggregate[k]} />
              ))}
              {extra && (
                <td className="px-3 py-2 text-right font-mono text-xs text-ink-2">
                  {e.aggregate.latency_s != null ? `${e.aggregate.latency_s.toFixed(1)}s` : '–'}
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function VersusChart({ vector, vectorless }: { vector: Experiment; vectorless: Experiment }) {
  const metrics = [...RAGAS.map(([k, l]) => [k, l] as const), ['hit@5', 'Page hit@5'] as const]
  return (
    <div className="rounded-xl border border-line bg-surface p-5">
      <div className="mb-4 flex flex-wrap gap-4 text-xs text-ink-2">
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-sm bg-series-1" /> Vector RAG ({vector.name})
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-sm bg-series-2" /> Vectorless ({vectorless.name})
        </span>
      </div>
      <div className="space-y-4">
        {metrics.map(([k, label]) => (
          <div key={k}>
            <div className="mb-1 text-xs font-medium text-ink">{label}</div>
            {[
              [vector, 'bg-series-1', 'Vector'],
              [vectorless, 'bg-series-2', 'Vectorless'],
            ].map(([e, cls, name]) => {
              const v = (e as Experiment).aggregate[k]
              return (
                <div key={name as string} className="mb-1 flex items-center gap-2" title={`${name}: ${v?.toFixed(3) ?? 'n/a'}`}>
                  <div className="h-2.5 flex-1 rounded-full bg-surface-2">
                    <div className={`h-2.5 rounded-full ${cls}`} style={{ width: `${Math.max(1, (v ?? 0) * 100)}%` }} />
                  </div>
                  <span className="w-10 text-right font-mono text-xs text-ink-2">{v != null ? v.toFixed(2) : '–'}</span>
                </div>
              )
            })}
          </div>
        ))}
        <div className="grid grid-cols-2 gap-3 border-t border-line pt-3 text-xs">
          <div>
            <div className="text-muted">Vector latency</div>
            <div className="font-mono text-ink">{vector.aggregate.latency_s?.toFixed(1) ?? '–'}s</div>
          </div>
          <div>
            <div className="text-muted">Vectorless latency</div>
            <div className="font-mono text-ink">{vectorless.aggregate.latency_s?.toFixed(1) ?? '–'}s</div>
          </div>
        </div>
      </div>
    </div>
  )
}

export function EvalView() {
  const [data, setData] = useState<Experiment[] | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    api.evals().then((d) => setData(d.experiments)).catch((e) => setError(String(e.message ?? e)))
  }, [])

  if (error) return <p className="text-sm text-warn">Could not load results: {error}</p>
  if (!data) return <div className="loading-bar h-40 rounded-xl bg-surface-2" />

  const rag = data.filter((e) => e.kind === 'rag')
  const retrieval = data.filter((e) => e.kind === 'retrieval')
  const vectorless = rag.find((e) => e.config.mode === 'vectorless')
  const vectorOnSame = rag.find((e) => e.config.mode === 'hybrid' && e.config.subset === 'vectorless')

  if (!data.length)
    return <p className="text-sm text-ink-2">No evaluation results yet. Run <code>uv run arr-eval</code> and <code>arr-eval report</code>.</p>

  return (
    <div className="space-y-10">
      <section>
        <h2 className="text-lg font-semibold text-ink">End-to-end quality (RAGAS)</h2>
        <p className="mb-4 text-sm text-ink-2">
          Each run answers the test questions and is scored by RAGAS with a Groq judge. Every run is also an experiment in LangSmith.
        </p>
        <Table rows={rag} cols={[...RAGAS, ['hit@5', 'Page hit@5', 'A reference page is among the top 5 passages']]} extra />
      </section>

      {retrieval.length > 0 && (
        <section>
          <h2 className="text-lg font-semibold text-ink">Chunking strategies (retrieval only)</h2>
          <p className="mb-4 text-sm text-ink-2">
            How often a page holding the answer is among the top-k retrieved passages, for fixed-size versus structure-aware chunks.
          </p>
          <Table rows={retrieval} cols={RETRIEVAL} />
        </section>
      )}

      {vectorless && vectorOnSame && (
        <section>
          <h2 className="text-lg font-semibold text-ink">Vector vs vectorless RAG</h2>
          <p className="mb-4 text-sm text-ink-2">The same questions about one long report, answered both ways.</p>
          <VersusChart vector={vectorOnSame} vectorless={vectorless} />
        </section>
      )}
    </div>
  )
}
