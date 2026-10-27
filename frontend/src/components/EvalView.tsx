import { useEffect, useState } from 'react'
import type { Experiment } from '../api'
import { LABELS } from '../evalLabels'
import { QuestionResults, type Row } from './QuestionResults'

// Results are bundled into the site (written by `arr-eval report`), so this page never waits for
// the API to wake up.
interface EvalData {
  experiments: Experiment[]
  rows: Record<string, Row[]>
}

const RAGAS = [
  ['faithfulness', 'Faithfulness', 'Share of the claims in the answer that the retrieved passages support. Low means the answer says things the sources do not.'],
  ['answer_relevancy', 'Answer relevancy', 'How directly the answer addresses the question asked, rather than drifting or hedging.'],
  ['context_recall', 'Context recall', 'Share of the reference answer that can be found in the retrieved passages. Low means retrieval missed the information.'],
] as const

const PAGE_HIT = ['hit@5', 'Page hit@5', 'Share of questions where a page holding the answer is among the top 5 retrieved passages. Measured against reference pages, no judge involved.'] as const

const RETRIEVAL = [
  ['hit@1', 'Hit@1', 'A reference page is the top passage'],
  ['hit@3', 'Hit@3', 'A reference page is in the top 3'],
  ['hit@5', 'Hit@5', 'A reference page is in the top 5'],
  ['hit@10', 'Hit@10', 'A reference page is in the top 10'],
  ['mrr', 'MRR', 'Mean reciprocal rank: 1 if the first passage is a reference page, 0.5 if the second, and so on'],
] as const

const tokens = (v: number | null | undefined) => (v != null ? Math.round(v).toLocaleString('en-IN') : '–')
const pct = (v: number | null | undefined) => (v != null ? `${Math.round(v * 100)}%` : '–')
const fixed = (v: number | null | undefined, d = 2) => (v != null ? v.toFixed(d) : '–')

function ExperimentName({ e }: { e: Experiment }) {
  const [title, scope] = LABELS[e.name] ?? [e.name, '']
  return (
    <td className="px-3 py-2">
      <div className="font-medium whitespace-nowrap text-ink">{title}</div>
      <div className="text-xs whitespace-nowrap text-muted">
        {scope ? `${scope} · ` : ''}
        {e.n} questions · <span className="font-mono">{e.name}</span>
      </div>
    </td>
  )
}

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

function Table({ rows, cols, extra }: { rows: Experiment[]; cols: readonly (readonly [string, string, string])[]; extra?: boolean }) {
  return (
    <div className="overflow-x-auto rounded-xl border border-line bg-surface">
      <table className="w-full min-w-[720px] text-sm">
        <thead>
          <tr className="border-b border-line text-left text-xs text-muted">
            <th className="px-3 py-2 font-medium">Setup</th>
            {cols.map(([k, name, hint]) => (
              <th key={k} className="px-3 py-2 font-medium" title={hint}>
                {name}
              </th>
            ))}
            {extra && <th className="px-3 py-2 text-right font-medium">Latency</th>}
            {extra && (
              <th className="px-3 py-2 text-right font-medium" title="LLM tokens per question (input + output)">
                Tokens / q
              </th>
            )}
          </tr>
        </thead>
        <tbody>
          {rows.map((e) => (
            <tr key={e.name} className="border-b border-line last:border-0">
              <ExperimentName e={e} />
              {cols.map(([k]) => (
                <BarCell key={k} value={e.aggregate[k]} />
              ))}
              {extra && <td className="px-3 py-2 text-right font-mono text-xs text-ink-2">{fixed(e.aggregate.latency_s, 1)}s</td>}
              {extra && <td className="px-3 py-2 text-right font-mono text-xs text-ink-2">{tokens(e.aggregate.tokens)}</td>}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

interface Finding {
  stat: string
  statLabel: string
  title: string
  body: string
}

// Differences smaller than this are within the noise of a 17-20 question run.
const NOISE = 0.1
const gap = (a: number, b: number) => (Math.abs(a - b) < NOISE ? 'within noise for this sample size' : 'a clear gap')

function findings(data: EvalData): Finding[] {
  const agg = (name: string) => data.experiments.find((e) => e.name === name)?.aggregate
  const out: Finding[] = []
  const hybrid = agg('hybrid-structure')
  const vec = agg('vector-indigo')
  const vl = agg('vectorless')
  const fixedChunks = agg('hybrid-fixed')

  if (hybrid?.answer_relevancy != null && hybrid.faithfulness != null && hybrid['hit@5'] != null) {
    out.push({
      stat: fixed(hybrid.answer_relevancy),
      statLabel: 'answer relevancy',
      title: 'Hybrid RAG across all three companies',
      body: `Faithfulness is ${fixed(hybrid.faithfulness)}, and a page holding the answer is in the top 5 passages for ${pct(hybrid['hit@5'])} of questions, including comparisons across companies and years.`,
    })
  }
  if (vec?.['hit@5'] && vl?.['hit@5']) {
    const better = vec['hit@5'] >= vl['hit@5']
    out.push({
      stat: `${pct(vec['hit@5'])} vs ${pct(vl['hit@5'])}`,
      statLabel: 'right page in top 5',
      title: `Vector RAG finds the right page ${better ? 'more' : 'less'} often than vectorless`,
      body: `On the same IndiGo questions, context recall is ${fixed(vec.context_recall)} for vector RAG and ${fixed(vl.context_recall)} for vectorless, whose weak spot is picking the right section from the report's contents.`,
    })
  }
  if (vec?.tokens && vl?.tokens) {
    out.push({
      stat: `${(vl.tokens / vec.tokens).toFixed(1)}×`,
      statLabel: 'tokens per question',
      title: `Vectorless uses ${vl.tokens > vec.tokens ? 'more' : 'fewer'} tokens than vector RAG`,
      body: `${tokens(vl.tokens)} vs ${tokens(vec.tokens)} tokens per question: vectorless reads whole pages instead of short chunks. In exchange it needs no vector index, and it answered in ${fixed(vl.latency_s, 1)}s vs ${fixed(vec.latency_s, 1)}s.`,
    })
  }
  if (hybrid?.['hit@5'] != null && fixedChunks?.['hit@5'] != null && hybrid.faithfulness != null && fixedChunks.faithfulness != null) {
    out.push({
      stat: `${pct(hybrid['hit@5'])} vs ${pct(fixedChunks['hit@5'])}`,
      statLabel: 'right page in top 5',
      title: 'Chunking strategy: a trade-off, not a winner',
      body: `Structure-aware vs fixed-size chunks: page hit@5 ${pct(hybrid['hit@5'])} vs ${pct(fixedChunks['hit@5'])} (${gap(hybrid['hit@5'], fixedChunks['hit@5'])}), faithfulness ${fixed(hybrid.faithfulness)} vs ${fixed(fixedChunks.faithfulness)} (${gap(hybrid.faithfulness, fixedChunks.faithfulness)}).`,
    })
  }
  return out
}

function VersusChart({ vector, vectorless }: { vector: Experiment; vectorless: Experiment }) {
  const metrics = [...RAGAS, PAGE_HIT].map(([k, l]) => [k, l] as const)
  return (
    <div className="rounded-xl border border-line bg-surface p-5">
      <div className="mb-4 flex flex-wrap gap-4 text-xs text-ink-2">
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-sm bg-series-1" /> Vector RAG
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-sm bg-series-2" /> Vectorless RAG
        </span>
      </div>
      <div className="space-y-4">
        {metrics.map(([k, name]) => (
          <div key={k}>
            <div className="mb-1 text-xs font-medium text-ink">{name}</div>
            {[
              [vector, 'bg-series-1', 'Vector'],
              [vectorless, 'bg-series-2', 'Vectorless'],
            ].map(([e, cls, who]) => {
              const v = (e as Experiment).aggregate[k]
              return (
                <div key={who as string} className="mb-1 flex items-center gap-2" title={`${who}: ${v?.toFixed(3) ?? 'n/a'}`}>
                  <div className="h-2.5 flex-1 rounded-full bg-surface-2">
                    <div className={`h-2.5 rounded-full ${cls}`} style={{ width: `${Math.max(1, (v ?? 0) * 100)}%` }} />
                  </div>
                  <span className="w-10 text-right font-mono text-xs text-ink-2">{fixed(v)}</span>
                </div>
              )
            })}
          </div>
        ))}
        <div className="grid grid-cols-2 gap-3 border-t border-line pt-3 text-xs">
          <div>
            <div className="text-muted">Vector latency</div>
            <div className="font-mono text-ink">{fixed(vector.aggregate.latency_s, 1)}s</div>
          </div>
          <div>
            <div className="text-muted">Vectorless latency</div>
            <div className="font-mono text-ink">{fixed(vectorless.aggregate.latency_s, 1)}s</div>
          </div>
          <div>
            <div className="text-muted">Vector tokens / question</div>
            <div className="font-mono text-ink">{tokens(vector.aggregate.tokens)}</div>
          </div>
          <div>
            <div className="text-muted">Vectorless tokens / question</div>
            <div className="font-mono text-ink">{tokens(vectorless.aggregate.tokens)}</div>
          </div>
        </div>
        <p className="text-xs text-muted">
          Tokens are LLM input + output per question. Neither count includes one-off indexing: vector RAG embeds the report locally, and vectorless summarises its sections once with an LLM.
        </p>
      </div>
    </div>
  )
}

function Section({ title, intro, children }: { title: string; intro?: string; children: React.ReactNode }) {
  return (
    <section>
      <h2 className="text-lg font-semibold text-ink">{title}</h2>
      {intro && <p className="mb-4 max-w-3xl text-sm text-ink-2">{intro}</p>}
      {children}
    </section>
  )
}

export function EvalView() {
  const [data, setData] = useState<EvalData | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    import('../data/eval.json')
      .then((m) => setData(m.default as unknown as EvalData))
      .catch((e) => setError(String(e.message ?? e)))
  }, [])

  if (error) return <p className="text-sm text-warn">Could not load results: {error}</p>
  if (!data) return <div className="loading-bar h-40 rounded-xl bg-surface-2" />
  if (!data.experiments.length)
    return <p className="text-sm text-ink-2">No evaluation results yet. Run <code>uv run arr-eval</code> and <code>arr-eval report</code>.</p>

  // main configuration first, in the order of LABELS; unknown runs last
  const order = Object.keys(LABELS)
  const rank = (e: Experiment) => (order.includes(e.name) ? order.indexOf(e.name) : order.length)
  const sorted = [...data.experiments].sort((a, b) => rank(a) - rank(b))
  const rag = sorted.filter((e) => e.kind === 'rag')
  const retrieval = sorted.filter((e) => e.kind === 'retrieval')
  const vectorless = rag.find((e) => e.config.mode === 'vectorless')
  const vectorOnSame = rag.find((e) => e.config.mode === 'hybrid' && e.config.subset === 'vectorless')
  const chunking = rag.filter((e) => e.config.mode === 'hybrid' && e.config.subset === 'sample')

  return (
    <div className="space-y-10">
      <header>
        <h1 className="text-2xl font-semibold text-ink">Evaluation</h1>
        <p className="mt-2 max-w-3xl text-sm text-ink-2">
          A test set of 48 questions about the reports, each with a reference answer and the pages that hold it. Retrieval is scored on all 48. Full answers are scored with RAGAS on a fixed 20-question sample across all companies, and on 17 questions about the IndiGo FY24 report for the vector vs vectorless comparison. For the evaluation, Claude Haiku 5.5 writes the answers and also judges them (the live site answers with gpt-oss-120b on Groq). Every run is an experiment in LangSmith.
        </p>
      </header>

      <Section title="Key findings">
        <div className="grid gap-3 sm:grid-cols-2">
          {findings(data).map((f) => (
            <div key={f.title} className="rounded-xl border border-line bg-surface p-4">
              <div className="flex items-baseline gap-2">
                <span className="font-mono text-2xl font-semibold text-ink">{f.stat}</span>
                <span className="text-xs text-muted">{f.statLabel}</span>
              </div>
              <h3 className="mt-2 text-sm font-semibold text-ink">{f.title}</h3>
              <p className="mt-1 text-sm text-ink-2">{f.body}</p>
            </div>
          ))}
        </div>
      </Section>

      <Section title="How to read the scores" intro="All scores run from 0 to 1, higher is better.">
        <dl className="grid gap-x-6 gap-y-3 rounded-xl border border-line bg-surface p-4 text-sm sm:grid-cols-2">
          {[...RAGAS, PAGE_HIT].map(([k, name, hint]) => (
            <div key={k}>
              <dt className="font-medium text-ink">{name}</dt>
              <dd className="text-ink-2">{hint}</dd>
            </div>
          ))}
          <div>
            <dt className="font-medium text-ink">Latency and tokens</dt>
            <dd className="text-ink-2">Seconds to answer, and LLM tokens (input + output) used per question.</dd>
          </div>
        </dl>
        <div className="mt-3 rounded-xl border border-line bg-surface-2 p-4 text-sm text-ink-2">
          <span className="font-medium text-ink">Small samples are noisy. </span>
          Each run has 17 to 20 questions, so a single question moves an average by about 0.05. Treat differences under about 0.1 as a tie. The judge is an LLM, and re-judging the same answers moved scores by up to about 0.05.
        </div>
      </Section>

      <Section title="End-to-end quality (RAGAS)" intro="Each setup answers its questions, and RAGAS scores the answers against the retrieved passages and the reference answers.">
        <Table rows={rag} cols={[...RAGAS, PAGE_HIT]} extra />
      </Section>

      {vectorless && vectorOnSame && (
        <Section title="Vector vs vectorless RAG" intro="The same 17 questions about the IndiGo FY24 report, answered both ways. Vectorless reads a table-of-contents tree, picks sections and reads their pages; vector RAG searches embedded chunks.">
          <VersusChart vector={vectorOnSame} vectorless={vectorless} />
        </Section>
      )}

      {retrieval.length > 0 && (
        <Section title="Chunking strategies" intro="Fixed-size chunks split each page by length; structure-aware chunks follow headings and keep tables whole. First, how often retrieval puts a reference page near the top (all 48 questions):">
          <Table rows={retrieval} cols={RETRIEVAL} />
          {chunking.length > 1 && (
            <>
              <p className="mt-6 mb-4 text-sm text-ink-2">Then the same comparison end to end, with full hybrid RAG on the 20-question sample:</p>
              <Table rows={chunking} cols={[...RAGAS, PAGE_HIT]} />
            </>
          )}
        </Section>
      )}

      <Section title="Every question" intro="The answer each setup gave, its scores, and the pages it cited against the reference pages. Open a question to read the answer.">
        <QuestionResults experiments={rag} rows={data.rows} />
      </Section>
    </div>
  )
}
