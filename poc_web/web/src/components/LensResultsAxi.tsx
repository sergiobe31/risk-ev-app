// Banda "Resultados" del lente AXI (shell B, DESIGN_UI_V2.md §3.1/§3.3/§3.4,
// semántica F4): fila L1 (f*+razón, P(pass) F4 ± IC95 con primer cruce (raw)
// en sub-leyenda, P(ruin) ± IC95 con semáforo, días-elegibilidad = máx(
// days_cal, min_days) — estimación calendario etiquetada —, trades de passes
// F4 vs suelo, maxDD p50), gráfico L1 P(pass)/P(ruin) vs f con bandas IC95,
// L2 colapsado, tabla por f (default cols L1; la vista completa añade el
// reparto 2a/2b/2c del cubo degradado), panel L3 "¿Cómo se ha calculado?"
// (onboarding-b1: sustituye a la nota de supuestos y puertas no modeladas),
// chips meta y warnings en bloque final.
//
// Suelos (§3.4, F4): regla doble E ≥ target ∧ T(d)-apertura ≥ suelo; si falta
// T el path continúa a f constante (2a → ruin, 2b → pass, 2c → sin resolver).
// El suelo de DÍAS no descuenta probabilidad: solo desplaza calendario (badge
// informativo, nunca ⚠).
import { useMemo, useState } from 'react'
import { BS1_TITLE, bs1At, type AxiReport, type AxiSimResult, type AxiSimParams, type ModelNote } from '../lib/api'
import type { DiagMetric } from '../lib/diag'
import { downloadText, runFilename, runToCsv, runToMarkdown, type RunLike } from '../lib/exporters'
import { Badge, Card, CardBody, CardHeader, Collapsible, Kpi } from './ui'
import { LineChart, type ChartSeries } from './charts'
import { SelectionDiagPanel } from './SelectionDiagPanel'
import { ModelNotesPanel, type RunText } from './ModelNotesPanel'
import { RecipeCard } from './RecipeCard'

const pct = (v: number | null | undefined, d = 1) =>
  v == null || !isFinite(v) ? '—' : `${(v * 100).toFixed(d)}%`
const num = (v: number | null | undefined, d = 0) =>
  v == null || !isFinite(v) ? '—' : v.toFixed(d)
const usd = (v: number | null | undefined) =>
  v == null || !isFinite(v)
    ? '—'
    : `${v.toLocaleString('es-ES', { maximumFractionDigits: 0 })} $`

// Badge de TRADES (§3.4, F4): el único que puede avisar ⚠. Sobre los passes
// F4 la mediana siempre ≥ suelo, así que el aviso real es "ningún path llega
// al target cumpliendo el suelo" (P(pass) F4 = 0).
function tradesBadge(median: number | null | undefined, floor: number | null) {
  if (floor == null) return 'sin suelo de trades en esta etapa'
  if (median == null || !isFinite(median))
    return `⚠ suelo ${floor}t: ningún pass llega al target con ${floor}+ trades`
  return median >= floor
    ? `mediana passes ${num(median)}t ≥ suelo ${floor}t ✓`
    : `⚠ mediana passes ${num(median)}t < suelo ${floor}t`
}

// Badge de DÍAS (§3.4): informativo, nunca ⚠ de probabilidad — el suelo de
// días solo desplaza calendario (esperar plano es gratis). days_cal es
// estimación calendario etiquetada (meta.days_cal), no exacta.
function daysBadge(p50: number | null | undefined, floor: number | null) {
  if (floor == null) return 'sin suelo de días en esta etapa'
  if (p50 == null || !isFinite(p50)) return `suelo ${floor}d — sin passes`
  return `próxima elegibilidad: día ${num(p50)} (suelo ${floor}d · estimación calendario)`
}

interface Props {
  result: AxiSimResult
  tolRuin: number // fracción (0..1)
  diag?: AxiSimResult | null // corrida gemela sin corte (ventana completa), P2
  nPaths?: number // recuento efectivo de paths para el test z
  params?: AxiSimParams // params de la corrida mostrada (export P4)
  modelNotes?: ModelNote[] // catalog.model_notes_axi (supuestos de la lente AXI)
}

export function LensResultsAxi({ result, tolRuin, diag, nPaths, params, modelNotes }: Props) {
  const [allCols, setAllCols] = useState(false)

  const star: AxiReport | undefined = useMemo(
    () => result.reports.find((r) => Math.abs(r.f - result.f_star) < 1e-9),
    [result],
  )

  // P2 — diagnóstico de selección: comparación al MISMO f (el f* OOS); el f*
  // que habrías elegido con la ventana completa va como fila propia.
  const diagMetrics: DiagMetric[] | null = useMemo(() => {
    if (!diag || star == null) return null
    const fullStar = diag.reports.find((r) => Math.abs(r.f - result.f_star) < 1e-9)
    if (fullStar == null) return null
    return [
      {
        metrica: 'P(pass) F4',
        kind: 'pct',
        higherBetter: true,
        full: fullStar.p_pass,
        oos: star.p_pass,
        test:
          nPaths && nPaths > 0
            ? { pFull: fullStar.p_pass, pOos: star.p_pass, nFull: nPaths, nOos: nPaths }
            : undefined,
      },
      {
        metrica: 'P(ruin)',
        kind: 'pct',
        higherBetter: false,
        full: fullStar.p_ruin,
        oos: star.p_ruin,
        test:
          nPaths && nPaths > 0
            ? { pFull: fullStar.p_ruin, pOos: star.p_ruin, nFull: nPaths, nOos: nPaths }
            : undefined,
      },
      {
        metrica: 'días a target p50 (primer cruce)',
        kind: 'days',
        higherBetter: false,
        full: fullStar.median_days_to_target,
        oos: star.median_days_to_target,
      },
      { metrica: 'f* recomendado', kind: 'pct', full: diag.f_star, oos: result.f_star },
    ]
  }, [diag, star, result.f_star, nPaths])

  // Feasibility del f* (review F2b: nada de verde hardcodeado en inviables).
  const feasible = result.recommendation != null

  // La serie raw (primer cruce, semántica motor) solo se dibuja si difiere de
  // la F4 en algún f.
  const rawDiffers = useMemo(
    () => result.reports.some((r) => Math.abs(r.p_pass_raw - r.p_pass) > 1e-12),
    [result],
  )

  const prSeries: ChartSeries[] = useMemo(() => {
    const mk = (pick: (r: AxiReport) => number, name: string, color: string, width: 1 | 2 | 3, dashed = false) =>
      ({
        name,
        color,
        width,
        dashed,
        data: result.reports.map((r, i) => ({ x: i, y: pick(r) })),
      }) as ChartSeries
    const series = [
      mk((r) => r.p_pass, 'P(pass)', '#34d399', 2),
      mk((r) => r.p_pass_ci[0], 'P(pass) IC95 inf', '#6ee7b7', 1, true),
      mk((r) => r.p_pass_ci[1], 'P(pass) IC95 sup', '#6ee7b7', 1, true),
      mk((r) => r.p_ruin, 'P(ruin)', '#fb7185', 2),
      mk((r) => r.p_ruin_ci[0], 'P(ruin) IC95 inf', '#fda4af', 1, true),
      mk((r) => r.p_ruin_ci[1], 'P(ruin) IC95 sup', '#fda4af', 1, true),
    ]
    if (rawDiffers)
      series.splice(1, 0, mk((r) => r.p_pass_raw, 'P(pass) primer cruce (raw)', '#a7f3d0', 1, true))
    return series
  }, [result, rawDiffers])

  const xLabels = result.reports.map((r) => pct(r.f, 2))
  const meta = result.meta
  const overTol = (star?.p_ruin ?? 0) > tolRuin
  const starRawDiffers =
    star != null && Math.abs(star.p_pass_raw - star.p_pass) > 1e-12
  // bloque-politis-white (fase B): sensibilidad con bloque 1 en f* (meta.bs1, lista con UNA fila: E16, S = {f*}). Sin fila
  // (bloque principal ≤ 1, modo fallback_iid o corrida sin la clave) no se pinta la línea: la cifra sería la misma.
  const bs1Star = bs1At(meta.bs1, result.f_star)
  // onboarding-b1 (enmienda 6): los textos de meta que vivían en el Collapsible
  // "Supuestos y puertas no modeladas" pasan literales a "En esta corrida".
  const runTexts: RunText[] = [
    { titulo: 'Suposición de política', texto: meta.policy_assumption },
    { titulo: 'Convención del flotante', texto: meta.float_convention },
    { titulo: 'Desplazamiento de f*', texto: meta.f_star_shift_note },
  ].filter((t) => !!t.texto)
  const passRuinHint =
    'Regla objetivo + trades mínimos; sin reducción de riesgo tras el objetivo · detalle en «¿Cómo se ha calculado?»'

  return (
    <div className="space-y-5">
      {/* P4 — export de la corrida actual */}
      <div className="flex justify-end gap-2">
        {([
          ['csv', 'CSV'],
          ['md', 'Markdown'],
        ] as const).map(([ext, label]) => {
          const like: RunLike = {
            lens: 'axi',
            meta: result.meta,
            f_star: result.f_star,
            recommendation: result.recommendation,
            reports: result.reports,
            warnings: result.warnings,
            request: params,
          }
          return (
            <button
              key={ext}
              onClick={() =>
                downloadText(
                  runFilename(like, ext),
                  ext === 'csv' ? runToCsv(like) : runToMarkdown(like),
                  ext === 'csv' ? 'text/csv' : 'text/markdown',
                )
              }
              className="rounded-md border border-zinc-700 px-2 py-1 text-[11px] font-medium text-zinc-400 hover:bg-zinc-800"
            >
              Exportar {label}
            </button>
          )
        })}
      </div>

      {/* Fila L1 — la decisión completa sin scroll */}
      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
        <Kpi
          label="f recomendado"
          value={pct(result.f_star, 2)}
          sub={result.recommendation?.reason ?? 'sin f factible bajo la tolerancia'}
          tone={feasible ? 'success' : 'neutral'}
        />
        <Kpi
          label="P(pass) @ f*"
          hint={passRuinHint}
          value={pct(star?.p_pass)}
          sub={`${
            starRawDiffers
              ? `primer cruce ${pct(star?.p_pass_raw)} → F4 ${pct(star?.p_pass)} · `
              : ''
          }IC95 ${pct(star?.p_pass_ci?.[0])} – ${pct(star?.p_pass_ci?.[1])}`}
        />
        <Kpi
          label="P(ruin) @ f*"
          hint={passRuinHint}
          value={pct(star?.p_ruin)}
          sub={`IC95 ${pct(star?.p_ruin_ci?.[0])} – ${pct(star?.p_ruin_ci?.[1])} · tol. ${pct(tolRuin, 0)}`}
          tone={overTol ? 'danger' : 'success'}
        />
        <Kpi
          label="Días hasta poder avanzar p50/p90"
          value={`${num(star?.median_days_eligibility)} / ${num(star?.p90_days_eligibility)}`}
          sub={daysBadge(star?.median_days_eligibility, meta.min_days)}
        />
        <Kpi
          label="Trades a target p50/p90 (passes)"
          value={`${num(star?.median_trades_qual)} / ${num(star?.p90_trades_qual)}`}
          sub={tradesBadge(star?.median_trades_qual, meta.min_unique_trades)}
        />
        <Kpi label="maxDD p50 @ f*" value={pct(star?.median_max_dd)} />
        {/* bloque-politis-white (fase B): línea gris de sensibilidad bajo los KPIs de f* (ocupa la fila entera de la rejilla) */}
        {bs1Star && (
          <p className="num col-span-full cursor-help text-xs text-zinc-400" title={BS1_TITLE}>
            Con bloque 1 (sin dependencia entre bloques): pasar {pct(bs1Star.p_pass)} · caer {pct(bs1Star.p_ruin)}
          </p>
        )}
      </div>

      {/* Gráfico L1 */}
      <Card>
        <CardHeader
          title="P(pass) / P(ruin) vs f"
          subtitle={`Barreras ${usd(meta.pass_line)} / ${usd(meta.ruin_line)} sobre allocation ${usd(meta.baseline)} · regla doble target + suelo de trades con continuación a f constante (F4) · bandas IC95 Wilson`}
        />
        <LineChart
          series={prSeries}
          xLabels={xLabels}
          yFormat={(v) => `${(v * 100).toFixed(0)}%`}
        />
      </Card>

      {/* P2 — diagnóstico de selección (solo con corrida gemela activada) */}
      {diagMetrics && <SelectionDiagPanel metrics={diagMetrics} />}

      {/* entrada-4-bloques F5 — receta para configurar el EA (solo carteras .sqx: las CSV no traen meta.recipe) */}
      {meta.recipe && <RecipeCard recipe={meta.recipe} fStar={result.f_star} />}

      {/* L2 — confianza (colapsado) */}
      <Collapsible title="Confianza y dispersión" subtitle="L2 · colas y terminal @ f*">
        <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
          <Kpi label="P(sin resolver) @ f*" value={pct(star?.p_sin_resolver)} sub="neither + timeout tras la continuación (§4.1)" />
          <Kpi label="maxDD p10 @ f*" value={pct(star?.p10_max_dd)} sub="cola adversa" tone="danger" />
          <Kpi label="Terminal p05" value={usd(star?.terminal_p05)} tone="danger" />
          <Kpi label="Terminal p50" value={usd(star?.terminal_p50)} />
          <Kpi label="Terminal p95" value={usd(star?.terminal_p95)} tone="success" />
        </div>
      </Collapsible>

      {/* Tabla por f — colapsada (L3); vista default = columnas L1 */}
      <Collapsible title="Tabla completa por f" subtitle="L3 · vista default: columnas L1">
        <div className="mb-2 flex items-center justify-between">
          <span className="text-xs text-zinc-500">
            {allCols ? 'Todas las columnas (L1+L2, incl. primer cruce y reparto 2a/2b/2c)' : 'Vista L1 — columnas de decisión'}
          </span>
          <button
            onClick={() => setAllCols((v) => !v)}
            className="rounded-md border border-zinc-700 px-2 py-1 text-[11px] font-medium text-zinc-400 hover:bg-zinc-800"
          >
            {allCols ? 'Solo columnas L1' : 'Todas las columnas'}
          </button>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-zinc-800 text-left text-[11px] uppercase tracking-wide text-zinc-500">
                <th className="px-4 py-2 font-medium">f</th>
                <th className="px-3 py-2 text-right font-medium">P(pass)</th>
                <th className="px-3 py-2 text-right font-medium">P(pass) 1er cruce</th>
                <th className="px-3 py-2 text-right font-medium">P(ruin)</th>
                <th className="px-3 py-2 text-right font-medium">días-eleg p50</th>
                <th className="px-3 py-2 text-right font-medium">días-eleg p90</th>
                <th className="px-3 py-2 text-right font-medium">trades p50</th>
                <th className="px-3 py-2 text-right font-medium">trades p90</th>
                <th className="px-4 py-2 text-right font-medium">maxDD p50</th>
                {allCols && (
                  <>
                    <th className="px-3 py-2 text-right font-medium">P(pass) IC95</th>
                    <th className="px-3 py-2 text-right font-medium">P(ruin) IC95</th>
                    <th className="px-3 py-2 text-right font-medium">P(sin res.)</th>
                    <th className="px-3 py-2 text-right font-medium">2a→ruin (n)</th>
                    <th className="px-3 py-2 text-right font-medium">2b→pass (n)</th>
                    <th className="px-3 py-2 text-right font-medium">2c→sin res. (n)</th>
                    <th className="px-3 py-2 text-right font-medium">días 1er cruce p50</th>
                    <th className="px-3 py-2 text-right font-medium">días 1er cruce p90</th>
                    <th className="px-3 py-2 text-right font-medium">trades 1er cruce p50</th>
                    <th className="px-3 py-2 text-right font-medium">trades 1er cruce p90</th>
                    <th className="px-3 py-2 text-right font-medium">maxDD p10</th>
                    <th className="px-3 py-2 text-right font-medium">terminal p05</th>
                    <th className="px-3 py-2 text-right font-medium">terminal p50</th>
                    <th className="px-4 py-2 text-right font-medium">terminal p95</th>
                  </>
                )}
              </tr>
            </thead>
            <tbody className="num">
              {result.reports.map((r) => {
                const isStar = Math.abs(r.f - result.f_star) < 1e-9
                return (
                  <tr
                    key={r.f}
                    className={`border-b border-zinc-800/60 ${
                      isStar ? 'bg-emerald-950/30' : 'hover:bg-zinc-800/30'
                    }`}
                  >
                    <td className="px-4 py-2 font-medium text-zinc-200">
                      {pct(r.f, 2)}
                      {isStar && <span className="ml-1.5 text-emerald-400">★</span>}
                    </td>
                    <td className="px-3 py-2 text-right text-emerald-400">{pct(r.p_pass)}</td>
                    <td className="px-3 py-2 text-right text-zinc-400">{pct(r.p_pass_raw)}</td>
                    <td className={`px-3 py-2 text-right ${r.p_ruin > tolRuin ? 'text-rose-400' : ''}`}>
                      {pct(r.p_ruin)}
                    </td>
                    <td className="px-3 py-2 text-right">{num(r.median_days_eligibility)}</td>
                    <td className="px-3 py-2 text-right text-zinc-400">{num(r.p90_days_eligibility)}</td>
                    <td className="px-3 py-2 text-right">{num(r.median_trades_qual)}</td>
                    <td className="px-3 py-2 text-right text-zinc-400">{num(r.p90_trades_qual)}</td>
                    <td className="px-4 py-2 text-right">{pct(r.median_max_dd)}</td>
                    {allCols && (
                      <>
                        <td className="px-3 py-2 text-right text-zinc-400">
                          {pct(r.p_pass_ci[0])} – {pct(r.p_pass_ci[1])}
                        </td>
                        <td className="px-3 py-2 text-right text-zinc-400">
                          {pct(r.p_ruin_ci[0])} – {pct(r.p_ruin_ci[1])}
                        </td>
                        <td className="px-3 py-2 text-right text-zinc-500">{pct(r.p_sin_resolver)}</td>
                        <td className="px-3 py-2 text-right text-rose-400/80">{r.post_target.n_ruin}</td>
                        <td className="px-3 py-2 text-right text-emerald-400/80">{r.post_target.n_pass}</td>
                        <td className="px-3 py-2 text-right text-zinc-500">{r.post_target.n_unresolved}</td>
                        <td className="px-3 py-2 text-right text-zinc-400">{num(r.median_days_to_target)}</td>
                        <td className="px-3 py-2 text-right text-zinc-500">{num(r.p90_days_to_target)}</td>
                        <td className="px-3 py-2 text-right text-zinc-400">{num(r.median_trades_to_target)}</td>
                        <td className="px-3 py-2 text-right text-zinc-500">{num(r.p90_trades_to_target)}</td>
                        <td className="px-3 py-2 text-right text-zinc-400">{pct(r.p10_max_dd)}</td>
                        <td className="px-3 py-2 text-right text-zinc-400">{usd(r.terminal_p05)}</td>
                        <td className="px-3 py-2 text-right">{usd(r.terminal_p50)}</td>
                        <td className="px-4 py-2 text-right text-zinc-400">{usd(r.terminal_p95)}</td>
                      </>
                    )}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </Collapsible>

      {/* onboarding-b1 — qué supone el modelo (policy, convenciones, f*, días y puertas) */}
      <ModelNotesPanel
        context={meta.model_context}
        runTexts={runTexts}
        profileNotes={modelNotes}
      />

      {/* Chips meta persistentes — separados de los KPIs */}
      <div className="flex flex-wrap items-center gap-2 text-xs text-zinc-400">
        <Badge>etapa {meta.etapa_display}</Badge>
        <Badge>allocation {usd(meta.baseline)}</Badge>
        <Badge>
          suelo {meta.min_unique_trades ?? '—'}t / {meta.min_days ?? '—'}d
        </Badge>
        <Badge>{meta.n_blocks} bloques</Badge>
        <Badge>cobertura {pct(meta.coverage)}</Badge>
        <Badge>crossing {meta.crossing}</Badge>
        <span
          title={meta.model_context?.block_size.reason}
          className={meta.model_context ? 'cursor-help' : undefined}
        >
          <Badge>bs {num(meta.block_size, 1)}</Badge>
        </span>
        <Badge>{meta.n_paths} paths</Badge>
        <Badge variant="outline">
          motor: {meta.motor} · sizing {meta.sizing}
        </Badge>
        <Badge variant="outline">semántica {meta.semantics}</Badge>
        <Badge variant="outline" className="max-w-full whitespace-normal">
          {meta.float_convention}
        </Badge>
        <span className="num ml-auto">{meta.elapsed_s.toFixed(1)} s · motor real</span>
      </div>

      {/* Warnings de ejecución — agrupados en un único bloque al final */}
      {result.warnings.length > 0 && (
        <Card className="border-amber-900/60 bg-amber-950/20">
          <CardBody className="space-y-1 py-2.5">
            {result.warnings.map((w, i) => (
              <p key={i} className="text-xs text-amber-300">
                ⚠ {w}
              </p>
            ))}
          </CardBody>
        </Card>
      )}
    </div>
  )
}
