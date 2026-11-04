// Banda "Resultados" del lente Prop Firm (shell B, DESIGN_UI_V2.md §3.2/§3.3):
// fila L1 de KPIs (decisión completa sin scroll), 2 gráficos L1, L2 colapsado,
// tabla por f (default = columnas L1, toggle "todas las columnas"), chips meta
// persistentes separados de los KPIs y warnings agrupados en un bloque final.
import { useMemo, useState } from 'react'
import { BS1_TITLE, bs1At, type Firm, type Report, type SimResult, type PropfirmSimParams } from '../lib/api'
import type { DiagMetric } from '../lib/diag'
import { downloadText, runFilename, runToCsv, runToMarkdown, type RunLike } from '../lib/exporters'
import { Badge, Card, CardBody, CardHeader, Collapsible, Kpi } from './ui'
import { LineChart, type ChartSeries } from './charts'
import { SelectionDiagPanel } from './SelectionDiagPanel'
import { PairPanel } from './PairPanel'
import { ProfileRulesPanel } from './ProfileRulesPanel'
import { ModelNotesPanel, type RunText } from './ModelNotesPanel'
import { RecipeCard } from './RecipeCard'

const pct = (v: number | null | undefined, d = 1) =>
  v == null || !isFinite(v) ? '—' : `${(v * 100).toFixed(d)}%`
const eur = (v: number | null | undefined) =>
  v == null || !isFinite(v)
    ? '—'
    : `${v.toLocaleString('es-ES', { maximumFractionDigits: 0 })} €`
const num = (v: number | null | undefined, d = 1) =>
  v == null || !isFinite(v) ? '—' : v.toFixed(d)

interface Props {
  result: SimResult
  tolNeg: number // fracción (0..1)
  diag?: SimResult | null // corrida gemela sin corte (ventana completa), P2
  nSeq?: number // recuento efectivo de secuencias para el test z
  params?: PropfirmSimParams // params de la corrida mostrada (par 2D opt-in F3)
  // Perfil de la CORRIDA (no el seleccionado en el desplegable; GT-9): reglas y
  // supuestos del panel "¿Cómo se ha calculado?" salen de aquí.
  firm?: Firm
}

export function LensResultsPropfirm({ result, tolNeg, diag, nSeq, params, firm }: Props) {
  const [allCols, setAllCols] = useState(false)
  // Fase 2 (FTMO Futures): economía por campaña — KPIs, tabla y gráficos
  // propios; sin trayectoria (sequence_ev_futures no la produce).
  const isFutures = result.meta.funded_model === 'futures'
  // onboarding-b1: etiquetas derivadas del perfil de la corrida (nada de 183d/30d fijos).
  const rules = firm?.rules
  const cycleRules = rules?.model === 'cycle' ? rules : null
  const extRules = rules && rules.model !== 'cycle' ? rules : null
  const campaignCapDays = extRules?.funded.subscription?.campaign_cap_days
  // perfiles-v2: con un reto de UNA fase (perfil `cycle` con 1 fase) P(P1∧P2) = P(P1): se rotula
  // "P(pasar)". Sale de las reglas del perfil de la CORRIDA, no del desplegable.
  const nPhases = cycleRules?.phases.length
  const onePhase = nPhases === 1
  const passLabel = onePhase ? 'P(pasar)' : 'P(P1∧P2)'
  // bloque-politis-white (fase B): rótulo de la probabilidad de pase en la línea «Con bloque 1».
  // PLAN-DESVIO: el plan y el brief fijan «F1+F2». Se usa el MISMO rótulo que el KPI vecino (`passLabel`: «P(P1∧P2)» con dos
  // fases, «P(pasar)» con una) y «P(P1)» en FTMO Futures (p_both = p_pass1, E9), para que la misma cantidad no salga con dos
  // nombres en el mismo panel (revisión de código de la fase B, H1).
  const bs1PassLabel = isFutures ? 'P(P1)' : passLabel
  const lifetimeDays = extRules?.evaluation.lifetime_days
  const spanDays = result.meta.model_context?.data_span_days
  const truncLabel = campaignCapDays != null ? `P(truncada ${campaignCapDays}d)` : 'P(truncada)'
  const truncColLabel = campaignCapDays != null ? `P(trunc ${campaignCapDays}d)` : 'P(trunc)'
  const expiredLabel = lifetimeDays != null ? `P(expirado ${lifetimeDays}d)` : 'P(expirado)'
  const runTexts: RunText[] = [
    ...(result.meta.ev_semantics ? [{ titulo: 'Qué mide el EV', texto: result.meta.ev_semantics }] : []),
    ...(result.meta.day_axis_note ? [{ titulo: 'Eje de días', texto: result.meta.day_axis_note }] : []),
  ]

  const star: Report | undefined = useMemo(
    () => result.reports.find((r) => Math.abs(r.f - result.f_star) < 1e-9),
    [result],
  )

  // P2 — diagnóstico de selección: se compara SIEMPRE al mismo f (el f* OOS,
  // la variable de decisión); el f* que habrías elegido con la ventana
  // completa va como fila propia.
  const diagMetrics: DiagMetric[] | null = useMemo(() => {
    if (!diag || star == null) return null
    const fullStar = diag.reports.find((r) => Math.abs(r.f - result.f_star) < 1e-9)
    if (fullStar == null) return null
    return [
      {
        metrica: isFutures ? 'EV/campaña' : 'EV/intento',
        kind: 'eur',
        higherBetter: true,
        full: fullStar.ev_per_attempt,
        oos: star.ev_per_attempt,
      },
      {
        metrica: 'P(neto<0)',
        kind: 'pct',
        higherBetter: false,
        full: fullStar.p_net_negative,
        oos: star.p_net_negative,
        test:
          nSeq && nSeq > 0
            ? { pFull: fullStar.p_net_negative, pOos: star.p_net_negative, nFull: nSeq, nOos: nSeq }
            : undefined,
      },
      {
        // Futures: fase única → P(P1∧P2) = P(P1) (enmienda E9).
        metrica: isFutures ? 'P(P1)' : passLabel,
        kind: 'pct',
        higherBetter: true,
        full: fullStar.p_both,
        oos: star.p_both,
        test:
          nSeq && nSeq > 0
            ? { pFull: fullStar.p_both, pOos: star.p_both, nFull: nSeq, nOos: nSeq }
            : undefined,
      },
      { metrica: 'f* recomendado', kind: 'pct', full: diag.f_star, oos: result.f_star },
    ]
  }, [diag, star, result.f_star, nSeq, isFutures, passLabel])

  const evSeries: ChartSeries[] = useMemo(() => {
    const mk = (pick: (r: Report) => number | null, name: string, color: string, width: 1 | 2 | 3, dashed = false) =>
      ({
        name,
        color,
        width,
        dashed,
        data: result.reports
          .map((r, i) => ({ x: i, y: pick(r) }))
          .filter((p): p is { x: number; y: number } => p.y != null),
      }) as ChartSeries
    return [
      mk((r) => r.ev_per_attempt, isFutures ? 'EV/campaña' : 'EV/intento', '#34d399', 2),
      mk((r) => r.ev_ci95[0], 'IC95 inf', '#6ee7b7', 1, true),
      mk((r) => r.ev_ci95[1], 'IC95 sup', '#6ee7b7', 1, true),
    ]
  }, [result, isFutures])

  const trajSeries: ChartSeries[] = useMemo(() => {
    // null en perfiles futures (E2): sin trayectoria no hay gráfico.
    const t = result.trajectory
    if (!t) return []
    const mk = (arr: number[], color: string, name: string, width: 1 | 2 | 3, dashed = false) =>
      ({ name, color, width, dashed, data: arr.map((y, x) => ({ x, y })) }) as ChartSeries
    return [
      mk(t.p95, '#38bdf8', 'p95', 1, true),
      mk(t.p50, '#34d399', 'p50', 2),
      mk(t.p05, '#fb7185', 'p05', 1, true),
    ]
  }, [result])

  const evXLabels = result.reports.map((r) => pct(r.f, 2))
  const trajXLabels = result.trajectory?.steps.map((s) => String(s)) ?? []
  const overTol = (star?.p_net_negative ?? 0) > tolNeg
  // bloque-politis-white (fase B): sensibilidad con bloque 1 en f* (meta.bs1, lista con UNA fila: E16, S = {f*}). Sin fila
  // (bloque principal ≤ 1, modo fallback_iid, secuencia sin caminos resueltos o corrida sin la clave) no se pinta la línea.
  const bs1Star = bs1At(result.meta.bs1, result.f_star)

  return (
    <div className="space-y-5">
      {/* P4 — export de la corrida actual */}
      <div className="flex justify-end gap-2">
        {([
          ['csv', 'CSV'],
          ['md', 'Markdown'],
        ] as const).map(([ext, label]) => {
          const like: RunLike = {
            lens: 'propfirm',
            meta: result.meta,
            f_star: result.f_star,
            recommendation: result.recommendation,
            reports: result.reports,
            warnings: result.warnings,
            request: params,
            nPhases,
            campaignCapDays,
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
      <div
        className={`grid grid-cols-2 gap-3 md:grid-cols-3 ${
          result.intentos_garantizados != null ? 'xl:grid-cols-6' : 'xl:grid-cols-5'
        }`}
      >
        <Kpi
          label="f recomendado"
          value={pct(result.f_star, 2)}
          sub={result.recommendation?.reason ?? 'sin f factible: punto central de la grilla'}
          tone={result.recommendation?.feasible ? 'success' : 'neutral'}
        />
        <Kpi
          label={isFutures ? 'EV/campaña @ f*' : 'EV/intento @ f*'}
          hint="Asume tu ventaja y tus costes del backtest · detalle en «¿Cómo se ha calculado?»"
          value={eur(star?.ev_per_attempt)}
          sub={`IC95 ${eur(star?.ev_ci95?.[0])} – ${eur(star?.ev_ci95?.[1])}`}
          tone={(star?.ev_per_attempt ?? 0) >= 0 ? 'success' : 'danger'}
        />
        <Kpi
          label="P(neto<0) @ f*"
          value={pct(star?.p_net_negative)}
          sub={`tolerancia ${pct(tolNeg, 0)}`}
          tone={overTol ? 'danger' : 'success'}
        />
        {isFutures ? (
          <>
            <Kpi
              label="EV/día @ f*"
              value={eur(star?.ev_per_day)}
              sub="EV/campaña ÷ días medios de campaña"
            />
            <Kpi
              label={`${truncLabel} @ f*`}
              value={pct(star?.p_truncated)}
              sub="truncado del modelo, no regla de la firma"
            />
            <Kpi
              label="Mensualidades @ f*"
              value={num(star?.mean_billings, 1)}
              sub={`${num(star?.mean_attempts, 1)} intentos · ${num(star?.mean_resets, 1)} resets (med)`}
            />
          </>
        ) : (
          <>
            <Kpi
              label={`${passLabel} @ f*`}
              value={pct(star?.p_both)}
              hint="Sobre las historias que se resuelven · detalle en «¿Cómo se ha calculado?»"
            />
            <Kpi
              label="Días al 1er payout p50"
              value={star?.median_days_first_payout == null ? '—' : `${num(star.median_days_first_payout, 0)} d`}
            />
          </>
        )}
        {result.intentos_garantizados != null && (
          <Kpi
            label="Intentos garantizados"
            value={String(result.intentos_garantizados)}
            sub={`⌊capital / fee⌋ · fee ${eur(result.meta.fee_eur)}`}
          />
        )}
        {!!star?.sizing_invalid && (
          <Kpi
            label="Sizing inválido @ f*"
            value="caps violados"
            sub="hay paths imposibles en la cuenta real (ver warnings)"
            tone="danger"
          />
        )}
        {/* bloque-politis-white (fase B): línea gris de sensibilidad bajo los KPIs de f* (ocupa la fila entera de la rejilla) */}
        {bs1Star && (
          <p className="num col-span-full cursor-help text-xs text-zinc-400" title={BS1_TITLE}>
            Con bloque 1 (sin dependencia entre bloques): {bs1PassLabel} {pct(bs1Star.p_both)} · P(neto&lt;0){' '}
            {pct(bs1Star.p_net_negative)} · EV {eur(bs1Star.ev_per_attempt)}
          </p>
        )}
      </div>

      {/* Gráficos L1 */}
      <div className={`grid grid-cols-1 gap-5 ${result.trajectory ? 'xl:grid-cols-2' : ''}`}>
        <Card>
          <CardHeader
            title={isFutures ? 'EV/campaña vs f' : 'EV por intento vs f'}
            subtitle={
              isFutures
                ? '€ por campaña de suscripción (EV incondicional) con IC95 bootstrap'
                : '€ por challenge (condicional a resolver) con IC95 bootstrap'
            }
          />
          <LineChart
            series={evSeries}
            xLabels={evXLabels}
            yFormat={(v) => `${Math.round(v).toLocaleString('es-ES')} €`}
          />
        </Card>
        {result.trajectory && (
          <Card>
            <CardHeader
              title={`Capital real · trayectorias @ f = ${pct(result.f_star, 2)}`}
              subtitle="p05 / p50 / p95 de la secuencia de challenges"
            />
            <LineChart
              series={trajSeries}
              xLabels={trajXLabels}
              yFormat={(v) => `${Math.round(v).toLocaleString('es-ES')} €`}
            />
          </Card>
        )}
      </div>

      {/* P2 — diagnóstico de selección (solo con corrida gemela activada) */}
      {diagMetrics && <SelectionDiagPanel metrics={diagMetrics} />}

      {/* entrada-4-bloques F5 — receta para configurar el EA (solo carteras .sqx: las CSV no traen meta.recipe) */}
      {result.meta.recipe && <RecipeCard recipe={result.meta.recipe} fStar={result.f_star} />}

      {/* L2 — confianza y sensibilidad (colapsado) */}
      <Collapsible title="Confianza y sensibilidad" subtitle="L2 · validez del EV y dispersión @ f*">
        {isFutures ? (
          <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
            <Kpi
              label="P(neither path) @ f*"
              hint="Incluye todas las historias · detalle en «¿Cómo se ha calculado?»"
              value={pct(star?.p_unresolved)}
              sub="EV incondicional: no excluye paths"
            />
            <Kpi
              label="P(neither intentos) @ f*"
              value={pct(star?.p_neither_attempts)}
              sub="neither / intentos de campaña"
            />
            <Kpi
              label="Días campaña (med)"
              value={num(star?.mean_days_campaign, 0)}
              sub="denominador del EV/día"
            />
            <Kpi label="Resets (med)" value={num(star?.mean_resets, 1)} />
            <Kpi
              label="Deferrals (med)"
              value={num(star?.mean_payout_deferrals, 1)}
              sub="payouts diferidos"
            />
            <Kpi
              label="Días a pase p50"
              value={num(star?.median_days_pass1, 0)}
              sub="campañas passed"
            />
          </div>
        ) : (
          <div
            className={`grid grid-cols-2 gap-3 md:grid-cols-3 ${
              cycleRules ? 'xl:grid-cols-7' : 'xl:grid-cols-6'
            }`}
          >
            <Kpi label="P(sin resolver) @ f*" value={pct(star?.p_unresolved)} sub="EV condicional a resolver" />
            <Kpi label="Challenges / payout" value={num(star?.challenges_per_payout, 1)} />
            <Kpi label="Neto p05" value={eur(star?.net_p05)} tone="danger" />
            <Kpi label="Neto p50" value={eur(star?.net_p50)} />
            <Kpi label="Neto p95" value={eur(star?.net_p95)} tone="success" />
            <Kpi
              label="Vivo al cap"
              value={pct(star?.alive_at_cap_frac)}
              sub={
                cycleRules
                  ? `llegó a ${cycleRules.funded.max_payout_cycles} ciclos`
                  : 'sesgo de truncado'
              }
              hint="Tope de la simulación y fin de tu backtest · detalle en «¿Cómo se ha calculado?»"
            />
            {cycleRules && (
              <Kpi
                label="Cortadas por los datos"
                value={pct(star?.funded_cut_by_data_frac)}
                sub={`fondeadas vivas al acabarse tu backtest${
                  spanDays != null ? ` (~${spanDays} d)` : ''
                }: sin más cobros`}
                hint="Tope de la simulación y fin de tu backtest · detalle en «¿Cómo se ha calculado?»"
              />
            )}
          </div>
        )}
        {/* Fase 6 (Apex): diagnósticos del perfil extendido @ f* */}
        {/* onboarding-b1: en FTMO CFD p_expired llega 0 (no NaN) → solo en extendidos */}
        {firm?.extended && star?.p_expired != null && (
          <div className="mt-3 grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
            <Kpi label={expiredLabel} value={pct(star.p_expired)} sub="resuelto: fee perdido" />
            <Kpi label="DLL hits (med)" value={num(star.mean_dll_hits, 1)} sub="por PA fondeada" />
            <Kpi
              label="Deferrals 50% (med)"
              value={num(star.mean_consistency_deferrals, 1)}
              sub="denegaciones de consistencia"
            />
            <Kpi label="P(muerte inact.)" value={pct(star.p_inactivity_death)} />
            <Kpi
              label="P(cierre escalera)"
              value={pct(star.p_closed_after_ladder)}
              sub="éxito terminal tras el último retiro"
            />
            <Kpi
              label="Violación caps"
              value={`${pct(star.frac_cap_violations_eval, 0)} / ${pct(star.frac_cap_violations_pa, 0)}`}
              sub={`eval / PA · máx ${num(star.max_simultaneous_micros, 0)} micro-eq simult.`}
              tone={star.sizing_invalid ? 'danger' : 'neutral'}
            />
          </div>
        )}
      </Collapsible>

      {/* Reglas del perfil de la corrida (read-only). Abierto por defecto solo en
          los perfiles extendidos (Apex, FTMO Futures); plegado en FTMO CFD. */}
      {firm?.rules && (
        <Collapsible
          title="Reglas del perfil"
          subtitle={`${firm.display_name} · read-only`}
          defaultOpen={firm.extended}
        >
          <ProfileRulesPanel firm={firm} />
        </Collapsible>
      )}

      {/* F3 — par 2D opt-in (usa los params exactos de esta corrida).
          FTMO Futures: el barrido 2D no aplica (economía por campaña). */}
      {params && firm?.funded_model !== 'futures' && <PairPanel params={params} nPhases={nPhases} />}

      {/* Tabla por f — colapsada (L3); vista default = columnas L1 */}
      <Collapsible title="Tabla completa por f" subtitle="L3 · vista default: columnas L1">
        <div className="mb-2 flex items-center justify-between">
          <span className="text-xs text-zinc-500">
            {allCols ? 'Todas las columnas (L1+L2)' : 'Vista L1 — columnas de decisión'}
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
                {isFutures ? (
                  <th className="px-3 py-2 text-right font-medium">P(neither path)</th>
                ) : (
                  <th className="px-3 py-2 text-right font-medium">{passLabel}</th>
                )}
                <th className="px-3 py-2 text-right font-medium">
                  {isFutures ? 'EV/campaña' : 'EV/intento'}
                </th>
                <th className="px-3 py-2 text-right font-medium">P(neto&lt;0)</th>
                <th className="px-4 py-2 text-right font-medium">
                  {isFutures ? 'días campaña (med)' : 'días payout p50'}
                </th>
                {allCols && (
                  <>
                    {isFutures && (
                      <>
                        <th className="px-3 py-2 text-right font-medium">EV/día</th>
                        <th className="px-3 py-2 text-right font-medium">{truncColLabel}</th>
                        <th className="px-3 py-2 text-right font-medium">mensualid.</th>
                        <th className="px-3 py-2 text-right font-medium">intentos</th>
                        <th className="px-3 py-2 text-right font-medium">resets</th>
                      </>
                    )}
                    {!onePhase && <th className="px-3 py-2 text-right font-medium">P(P1)</th>}
                    {!isFutures && (
                      <th className="px-3 py-2 text-right font-medium">P(neither)</th>
                    )}
                    <th className="px-3 py-2 text-right font-medium">EV IC95</th>
                    <th className="px-3 py-2 text-right font-medium">neto p05</th>
                    <th className="px-3 py-2 text-right font-medium">neto p50</th>
                    <th className="px-3 py-2 text-right font-medium">neto p95</th>
                    {!isFutures && (
                      <>
                        <th className="px-3 py-2 text-right font-medium">E[payout]</th>
                        <th className="px-3 py-2 text-right font-medium">P(payout|fund.)</th>
                        <th className="px-3 py-2 text-right font-medium">ch/payout</th>
                      </>
                    )}
                    <th className="px-3 py-2 text-right font-medium">días pass1 p50</th>
                    <th className="px-4 py-2 text-right font-medium">vivo cap</th>
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
                    <td className="px-3 py-2 text-right">
                      {isFutures ? pct(r.p_unresolved) : pct(r.p_both)}
                    </td>
                    <td
                      className={`px-3 py-2 text-right font-medium ${
                        r.ev_per_attempt == null
                          ? 'text-zinc-500'
                          : r.ev_per_attempt >= 0
                            ? 'text-emerald-400'
                            : 'text-rose-400'
                      }`}
                    >
                      {eur(r.ev_per_attempt)}
                    </td>
                    <td className={`px-3 py-2 text-right ${r.p_net_negative > tolNeg ? 'text-rose-400' : ''}`}>
                      {pct(r.p_net_negative)}
                    </td>
                    <td className="px-4 py-2 text-right">
                      {isFutures
                        ? num(r.mean_days_campaign, 0)
                        : num(r.median_days_first_payout, 0)}
                    </td>
                    {allCols && (
                      <>
                        {isFutures && (
                          <>
                            <td className="px-3 py-2 text-right">{eur(r.ev_per_day)}</td>
                            <td className="px-3 py-2 text-right">{pct(r.p_truncated)}</td>
                            <td className="px-3 py-2 text-right">{num(r.mean_billings, 1)}</td>
                            <td className="px-3 py-2 text-right">{num(r.mean_attempts, 1)}</td>
                            <td className="px-3 py-2 text-right">{num(r.mean_resets, 1)}</td>
                          </>
                        )}
                        {!onePhase && <td className="px-3 py-2 text-right">{pct(r.p_pass1)}</td>}
                        {!isFutures && (
                          <td className="px-3 py-2 text-right text-zinc-500">{pct(r.p_unresolved)}</td>
                        )}
                        <td className="px-3 py-2 text-right text-zinc-400">
                          {eur(r.ev_ci95[0])} – {eur(r.ev_ci95[1])}
                        </td>
                        <td className="px-3 py-2 text-right text-zinc-400">{eur(r.net_p05)}</td>
                        <td className="px-3 py-2 text-right">{eur(r.net_p50)}</td>
                        <td className="px-3 py-2 text-right text-zinc-400">{eur(r.net_p95)}</td>
                        {!isFutures && (
                          <>
                            <td className="px-3 py-2 text-right">{eur(r.mean_payout_eur)}</td>
                            <td className="px-3 py-2 text-right">{pct(r.p_payout_given_funded)}</td>
                            <td className="px-3 py-2 text-right">{num(r.challenges_per_payout, 1)}</td>
                          </>
                        )}
                        <td className="px-3 py-2 text-right">{num(r.median_days_pass1, 0)}</td>
                        <td className="px-4 py-2 text-right">{pct(r.alive_at_cap_frac)}</td>
                      </>
                    )}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </Collapsible>

      {/* onboarding-b1 — qué supone el modelo (antes de chips y avisos, que van al final) */}
      <ModelNotesPanel
        context={result.meta.model_context}
        runTexts={runTexts}
        profileNotes={firm?.caveats}
      />

      {/* Chips meta persistentes — separados de los KPIs */}
      <div className="flex flex-wrap items-center gap-2 text-xs text-zinc-400">
        <Badge>{result.meta.legs.length} patas</Badge>
        <Badge>{result.meta.n_blocks} bloques</Badge>
        <Badge>cobertura {pct(result.meta.coverage)}</Badge>
        <span
          title={result.meta.model_context?.block_size.reason}
          className={result.meta.model_context ? 'cursor-help' : undefined}
        >
          <Badge>bs {num(result.meta.block_size, 1)}</Badge>
        </span>
        <Badge>{result.meta.n_paths} paths</Badge>
        <Badge>{result.meta.firm}</Badge>
        {result.meta.ruleset_version && <Badge variant="outline">{result.meta.ruleset_version}</Badge>}
        {result.meta.day_axis && <Badge variant="outline">eje {result.meta.day_axis}</Badge>}
        {isFutures && (
          <>
            <Badge variant="outline">corte eje {result.meta.day_axis_cut_hhmm}</Badge>
            <Badge variant="outline">reset: {result.meta.reset_policy}</Badge>
            <Badge variant="outline">retirada: {result.meta.withdrawal_policy}</Badge>
          </>
        )}
        {(result.meta.n_copied_accounts ?? 1) > 1 && (
          <Badge variant="warn">{result.meta.n_copied_accounts} cuentas (corr. perfecta)</Badge>
        )}
        {!!result.meta.sizing_invalid_any && (
          <Badge variant="danger">sizing inválido en algún f</Badge>
        )}
        <Badge variant="outline">
          motor: float_mode={result.meta.float_mode} · sizing {result.meta.sizing}
        </Badge>
        <span className="num ml-auto">{result.meta.elapsed_s.toFixed(1)} s · motor real</span>
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
