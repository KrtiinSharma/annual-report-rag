import { useEffect, useMemo, useState } from 'react'
import { api, type Answer, type AskOptions, type Catalog } from './api'
import { AboutView } from './components/AboutView'
import { AnswerView } from './components/AnswerView'
import { Controls, type Mode } from './components/Controls'
import { EvalView } from './components/EvalView'
import { PipelinePanel } from './components/PipelinePanel'

type Tab = 'ask' | 'eval' | 'about'
const TABS: [Tab, string][] = [
  ['ask', 'Ask'],
  ['eval', 'Evaluation'],
  ['about', 'How it works'],
]

const DEFAULT_OPTS: AskOptions = {
  companies: [],
  fiscal_years: [],
  strategy: 'structure',
  use_bm25: true,
  use_vector: true,
  use_rerank: true,
  use_multi_query: true,
}

function useHashTab(): [Tab, (t: Tab) => void] {
  const read = () => (TABS.some(([t]) => `#${t}` === location.hash) ? (location.hash.slice(1) as Tab) : 'ask')
  const [tab, setTab] = useState<Tab>(read)
  useEffect(() => {
    const on = () => setTab(read())
    addEventListener('hashchange', on)
    return () => removeEventListener('hashchange', on)
  }, [])
  return [tab, (t) => (location.hash = t)]
}

function ThemeToggle() {
  const [theme, setTheme] = useState<string>(() => {
    try {
      return localStorage.getItem('theme') ?? ''
    } catch {
      return ''
    }
  })
  useEffect(() => {
    if (theme) document.documentElement.dataset.theme = theme
    else delete document.documentElement.dataset.theme
    try {
      localStorage.setItem('theme', theme)
    } catch {
      /* storage blocked */
    }
  }, [theme])
  const dark = theme ? theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches
  return (
    <button
      type="button"
      onClick={() => setTheme(dark ? 'light' : 'dark')}
      className="rounded-md border border-line px-2 py-1 text-xs text-ink-2 hover:text-ink"
      aria-label="Toggle dark mode"
    >
      {dark ? '☀ Light' : '☾ Dark'}
    </button>
  )
}

export default function App() {
  const [tab, setTab] = useHashTab()
  const [catalog, setCatalog] = useState<Catalog | null>(null)
  const [catalogError, setCatalogError] = useState('')
  const [waking, setWaking] = useState(false)

  const [mode, setMode] = useState<Mode>('hybrid')
  const [opts, setOpts] = useState<AskOptions>(DEFAULT_OPTS)
  const [vectorlessReport, setVectorlessReport] = useState('')
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState<Answer | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [active, setActive] = useState<number | null>(null)
  const [showControls, setShowControls] = useState(false)

  useEffect(() => {
    // A sleeping Hugging Face Space can take a minute to wake up.
    const slow = setTimeout(() => setWaking(true), 2500)
    const load = (attempt: number) =>
      api
        .catalog()
        .then((c) => {
          setCatalog(c)
          setVectorlessReport(c.vectorless_reports[0] ?? '')
          setWaking(false)
          setCatalogError('')
        })
        .catch((e) => {
          if (attempt < 8) setTimeout(() => load(attempt + 1), 8000)
          else setCatalogError(String(e.message ?? e))
        })
        .finally(() => clearTimeout(slow))
    load(0)
    return () => clearTimeout(slow)
  }, [])

  const pdfAvailable = useMemo(
    () => new Set(catalog?.reports.filter((r) => r.has_pdf).map((r) => r.id) ?? []),
    [catalog],
  )
  const companyName = (slug: string) => catalog?.companies.find((c) => c.slug === slug)?.name ?? slug

  async function ask(q = question) {
    if (q.trim().length < 3 || loading) return
    setQuestion(q)
    setLoading(true)
    setError('')
    setActive(null)
    try {
      const res = mode === 'vectorless' ? await api.askVectorless(q, vectorlessReport) : await api.ask(q, opts)
      setAnswer(res)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  function cite(id: number) {
    setActive(id)
    const el = document.getElementById(`source-${id}`)
    if (el?.closest('details')) (el.closest('details') as HTMLDetailsElement).open = true
    el?.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-10 border-b border-line bg-surface/90 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center gap-4 px-4 py-3">
          <a href="#ask" className="flex items-center gap-2">
            <img src={`${import.meta.env.BASE_URL}favicon.svg`} alt="" className="h-7 w-7" />
            <span className="text-[15px] font-semibold text-ink">Annual Report Analyst</span>
          </a>
          <nav className="ml-auto flex items-center gap-1">
            {TABS.map(([t, label]) => (
              <button
                key={t}
                type="button"
                onClick={() => setTab(t)}
                className={`rounded-md px-2.5 py-1.5 text-sm transition-colors ${
                  tab === t ? 'bg-surface-2 font-medium text-ink' : 'text-ink-2 hover:text-ink'
                }`}
              >
                {label}
              </button>
            ))}
            <span className="ml-2 hidden sm:block">
              <ThemeToggle />
            </span>
          </nav>
        </div>
      </header>

      {waking && !catalog && (
        <div className="border-b border-line bg-accent-soft px-4 py-2 text-center text-sm text-accent">
          Waking up the free server… this can take up to a minute the first time.
        </div>
      )}

      <main className="mx-auto max-w-7xl px-4 py-6">
        {tab === 'eval' && <EvalView />}
        {tab === 'about' && <AboutView />}
        {tab === 'ask' && (
          <div className="grid gap-6 lg:grid-cols-[280px_1fr]">
            <aside className="lg:sticky lg:top-20 lg:self-start">
              <button
                type="button"
                className="mb-3 w-full rounded-lg border border-line bg-surface px-3 py-2 text-left text-sm font-medium text-ink-2 lg:hidden"
                onClick={() => setShowControls(!showControls)}
              >
                {showControls ? 'Hide' : 'Show'} filters and pipeline options
              </button>
              <div className={`${showControls ? 'block' : 'hidden'} rounded-xl border border-line bg-surface p-4 lg:block`}>
                {catalog ? (
                  <Controls
                    catalog={catalog}
                    mode={mode}
                    setMode={setMode}
                    opts={opts}
                    setOpts={setOpts}
                    vectorlessReport={vectorlessReport}
                    setVectorlessReport={setVectorlessReport}
                  />
                ) : catalogError ? (
                  <p className="text-sm text-warn">Backend unreachable: {catalogError}</p>
                ) : (
                  <div className="loading-bar h-64 rounded-lg bg-surface-2" />
                )}
              </div>
            </aside>

            <div className="min-w-0 space-y-6">
              <section>
                <h1 className="mb-1 text-2xl font-semibold tracking-tight text-ink">Ask the annual reports</h1>
                <p className="mb-4 text-sm text-ink-2">
                  Questions across IndiGo, Infosys and TCS annual reports, answered with page-level citations.
                </p>
                <form
                  onSubmit={(e) => {
                    e.preventDefault()
                    ask()
                  }}
                  className="flex flex-col gap-2 rounded-xl border border-line bg-surface p-2 focus-within:border-accent sm:flex-row"
                >
                  <textarea
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault()
                        ask()
                      }
                    }}
                    rows={2}
                    maxLength={500}
                    placeholder="e.g. Compare the revenue growth of IndiGo and Infosys in FY24"
                    className="min-h-12 flex-1 resize-none bg-transparent px-2 py-1.5 text-[15px] text-ink outline-none placeholder:text-muted"
                    aria-label="Question"
                  />
                  <button
                    type="submit"
                    disabled={loading || question.trim().length < 3 || !catalog}
                    className="rounded-lg bg-accent px-5 py-2 text-sm font-semibold text-accent-ink transition-opacity disabled:opacity-40 sm:self-end"
                  >
                    {loading ? 'Thinking…' : 'Ask'}
                  </button>
                </form>
                {catalog && (
                  <div className="mt-3 flex flex-wrap gap-2">
                    {catalog.examples.map((ex) => (
                      <button
                        key={ex}
                        type="button"
                        onClick={() => ask(ex)}
                        disabled={loading}
                        className="rounded-full border border-line bg-surface px-3 py-1 text-xs text-ink-2 hover:border-accent hover:text-accent disabled:opacity-50"
                      >
                        {ex}
                      </button>
                    ))}
                  </div>
                )}
              </section>

              {error && (
                <div role="alert" className="rounded-xl border border-warn/40 bg-surface p-4 text-sm text-warn">
                  {error}
                </div>
              )}

              {loading && (
                <div className="space-y-3 rounded-xl border border-line bg-surface p-6">
                  {[80, 95, 70, 60].map((w, i) => (
                    <div key={i} className="loading-bar h-3 rounded bg-surface-2" style={{ width: `${w}%` }} />
                  ))}
                  <p className="pt-2 text-xs text-muted">
                    {mode === 'hybrid' ? 'Planning searches, retrieving, reranking and writing…' : 'Reading the contents tree and choosing pages…'}
                  </p>
                </div>
              )}

              {answer && !loading && (
                <div className="grid gap-6 xl:grid-cols-[1fr_320px]">
                  <AnswerView answer={answer} pdfAvailable={pdfAvailable} active={active} onCite={cite} />
                  <div className="xl:sticky xl:top-20 xl:self-start">
                    <PipelinePanel answer={answer} companyName={companyName} />
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      <footer className="mx-auto max-w-7xl px-4 pt-4 pb-10 text-xs text-muted">
        LangChain · Groq · FastEmbed · Chroma · LangSmith · RAGAS. Answers come from public annual reports and may contain errors.
      </footer>
    </div>
  )
}
