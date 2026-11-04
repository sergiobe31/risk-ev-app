// L2 opt-in (F3): barrido 2D (f_challenge, f_funded) con ΔEV pareado vs la
// diagonal. El cómputo NO se lanza solo: botón "Calcular par 2D" con los
// MISMOS parámetros de la corrida 1D que generó los resultados visibles.
import { useMemo, useState } from 'react'
import { runPropfirmPair, type PairResult, type PropfirmSimParams } from '../lib/api'
import { Badge, Button, Card, CardBody, Collapsible, Kpi } from './ui'

const pct = (v: number | null | undefined, d = 1) =>
  v == null || !isFinite(v) ? '—' : `${(v * 100).toFixed(d)}%`
const eur = (v: number | null | undefined) =>
  v == null || !isFinite(v)
    ? '—'
    : `${v.toLocaleString('es-ES', { maximumFractionDigits: 0 })} €`

interface Props {
  params: PropfirmSimParams // params exactos de la corrida 1D mostrada
  nPhases?: number // fases del reto de la corrida (perfiles-v2): con 1, "P(pasar)" en vez de "P(P1∧P2)"
}

export function PairPanel({ params, nPhases }: Props) {
  const [res, setRes] = useState<PairResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  const run = async () => {
    setLoading(true)
    setErr(null)
    try {
      setRes(await runPropfirmPair(params))
    } catch (e) {
      setErr(String(e))
      setRes(null)
    } finally {
      setLoading(false)
    }
  }

  // Grilla y rango de EV para el gradiente de color.
  const { fCols, fRows, cellMap, evMin, evMax } = useMemo(() => {
    if (!res) return { fCols: [], fRows: [], cellMap: new Map<string, PairResult['cells'][number]>(), evMin: 0, evMax: 1 }
    const fs = [...new Set(res.cells.flatMap((c) => [c.f_challenge, c.f_funded]))].sort((a, b) => a - b)
    const map = new Map(res.cells.map((c) => [`${c.f_challenge}|${c.f_funded}`, c]))
    const evs = res.cells.map((c) => c.ev_per_attempt ?? 0)
    return {
      fCols: fs,
      fRows: fs,
      cellMap: map,
      evMin: Math.min(...evs),
      evMax: Math.max(...evs),
    }
  }, [res])

  const rec = res?.recommendation ?? null

  const cellBg = (ev: number | null): string => {
    if (ev == null || !isFinite(ev)) return ''
    if (evMax <= evMin) return 'rgba(52,211,153,0.25)'
    const t = (ev - evMin) / (evMax - evMin) // 0 peor → 1 mejor
    return ev >= 0
      ? `rgba(52,211,153,${(0.12 + 0.5 * t).toFixed(3)})` // emerald
      : `rgba(251,113,133,${(0.4 - 0.28 * t).toFixed(3)})` // rose
  }

  return (
    <Collapsible
      title="Par (f_challenge, f_funded)"
      subtitle="L2 · desdoblar el sizing challenge/fondeada — cómputo opt-in (F3)"
    >
      <div className="space-y-4">
        {!res && !loading && (
          <div className="flex flex-wrap items-center gap-3">
            <div className="md:w-[240px]">
              <Button onClick={run}>Calcular par 2D</Button>
            </div>
            <p className="max-w-xl text-xs text-zinc-500">
              Barrido {`f_challenge × f_funded`} con gaps fijos (CRN) e IC95 bootstrap
              pareado del ΔEV frente a la diagonal (un solo f). Puede tardar varios
 minutos: es una grilla completa de secuencias.
            </p>
          </div>
        )}
        {loading && <p className="text-sm text-zinc-400">Calculando grilla 2D…</p>}
        {err && (
          <p className="rounded-md border border-rose-900 bg-rose-950/60 px-2 py-1.5 text-xs text-rose-300">
            {err}
          </p>
        )}

        {res && rec && (
          <>
            {/* Banner decisión */}
            <Card
              className={
                rec.recommend_pair
                  ? 'border-emerald-800 bg-emerald-950/30'
                  : 'border-zinc-700 bg-zinc-900'
              }
            >
              <CardBody className="space-y-1 py-3">
                <p className="text-sm font-medium text-zinc-100">
                  {rec.recommend_pair
                    ? `Configura DOS f en el EA: challenge ${pct(rec.f_challenge, 2)} · fondeada ${pct(rec.f_funded, 2)}`
                    : `Quédate en un solo f (${pct(rec.f_single, 2)}): el ΔEV no supera el ruido`}
                </p>
                <p className="text-xs text-zinc-400">{rec.reason}</p>
                {(rec.on_boundary[0] || rec.on_boundary[1]) && (
                  <p className="text-xs text-amber-300">
                    ⚠ f recomendado en el borde de la grilla — amplía el rango de f para confirmar.
                  </p>
                )}
                {!rec.feasible && (
                  <p className="text-xs text-rose-300">
                    ⚠ Ninguna celda cumple las tolerancias; par mostrado por menor P(neto&lt;0).
                  </p>
                )}
              </CardBody>
            </Card>

            {/* KPIs del par */}
            <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
              <Kpi
                label="ΔEV vs diagonal"
                value={`${rec.delta_ev >= 0 ? '+' : ''}${eur(rec.delta_ev)}`}
                sub={`IC95 ${eur(rec.delta_ev_ci[0])} – ${eur(rec.delta_ev_ci[1])}`}
                tone={rec.delta_ev > 0 ? 'success' : 'danger'}
              />
              <Kpi label={`${nPhases === 1 ? 'P(pasar)' : 'P(P1∧P2)'} @ par`} value={pct(rec.p_pasar)} />
              <Kpi label="P(sin resolver) @ par" value={pct(rec.p_unresolved)} sub="veto > 5%" />
              <Kpi label="Vivo al cap @ par" value={pct(rec.alive_at_cap_frac)} />
            </div>

            {/* Tabla gradiente: filas f_challenge × columnas f_funded */}
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-zinc-800 text-left text-[11px] uppercase tracking-wide text-zinc-500">
                    <th className="px-4 py-2 font-medium">f_chal ↓ · f_fond →</th>
                    {fCols.map((ff) => (
                      <th key={ff} className="px-3 py-2 text-right font-medium">{pct(ff, 2)}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="num">
                  {fRows.map((fc) => (
                    <tr key={fc} className="border-b border-zinc-800/60">
                      <td className="px-4 py-2 font-medium text-zinc-200">{pct(fc, 2)}</td>
                      {fCols.map((ff) => {
                        const c = cellMap.get(`${fc}|${ff}`)
                        if (!c)
                          return (
                            <td key={ff} className="px-3 py-2 text-right text-zinc-600">—</td>
                          )
                        const isRec =
                          rec != null && fc === rec.f_challenge && ff === rec.f_funded
                        return (
                          <td
                            key={ff}
                            style={{ backgroundColor: cellBg(c.ev_per_attempt) }}
                            className={`relative px-3 py-2 text-right font-medium ${
                              isRec ? 'outline outline-1 outline-emerald-400' : 'text-zinc-100'
                            }`}
                          >
                            {eur(c.ev_per_attempt)}
                            {isRec && <span className="ml-1.5 text-emerald-300">★</span>}
                          </td>
                        )
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="text-xs text-zinc-500">
              Celda = EV/intento € (condicional a resolver). «—» = celda sin paths
              resueltos (mismo veto que el 1D).
            </p>

            {/* Sensibilidad al cap (solo 2 celdas de referencia) */}
            {res.cap_sensitivity.length > 0 && (
              <div>
                <p className="mb-1.5 text-xs font-medium uppercase tracking-wide text-zinc-500">
                  Sensibilidad al cap de ciclos (cota de truncado)
                </p>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-zinc-800 text-left text-[11px] uppercase tracking-wide text-zinc-500">
                      <th className="px-4 py-2 font-medium">celda</th>
                      <th className="px-3 py-2 text-right font-medium">EV cap máx</th>
                      <th className="px-3 py-2 text-right font-medium">EV cap10</th>
                      <th className="px-4 py-2 text-right font-medium">vivo al cap</th>
                    </tr>
                  </thead>
                  <tbody className="num">
                    {res.cap_sensitivity.map((s) => (
                      <tr key={s.celda} className="border-b border-zinc-800/60">
                        <td className="px-4 py-2 text-zinc-300">
                          {s.celda}{' '}
                          <span className="text-zinc-500">
                            ({pct(s.f_challenge, 2)} / {pct(s.f_funded, 2)})
                          </span>
                        </td>
                        <td className="px-3 py-2 text-right">{eur(s.ev_cap_max)}</td>
                        <td className="px-3 py-2 text-right text-zinc-400">{eur(s.ev_cap10)}</td>
                        <td className="px-4 py-2 text-right">{pct(s.vivo_al_cap)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="mt-1.5 text-xs text-zinc-500">
                  cap máx = payouts indefinidos (optimista); cap10 trunca a 10 ciclos.
                  La brecha entre ambos acota cuánto del EV depende de cobrar
                  indefinidamente.
                </p>
              </div>
            )}

            {/* Chips + warnings propios del 2D */}
            <div className="flex flex-wrap items-center gap-2 text-xs text-zinc-400">
              <Badge>grilla {res.meta.grid_n ?? '?'}×{res.meta.grid_n ?? '?'}</Badge>
              <Badge>{res.meta.cells_resolved ?? res.cells.length} celdas resueltas</Badge>
              <Badge variant="outline">bootstrap pareado n=2000</Badge>
            </div>
            {res.warnings.length > 0 && (
              <CardBody className="space-y-1 py-2.5 text-xs text-amber-300">
                {res.warnings.map((w, i) => (
                  <p key={i}>⚠ {w}</p>
                ))}
              </CardBody>
            )}
          </>
        )}
      </div>
    </Collapsible>
  )
}
