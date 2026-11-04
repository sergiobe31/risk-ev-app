// Panel global "Historial de corridas" (P4): lee /api/runs al abrirse (y tras
// cada corrida nueva vía refreshKey), filas con fecha/lente/patas/f*/métrica
// clave/badge de corte, acciones CSV y Markdown por fila y detalle expandible.
import { useEffect, useState } from 'react'
import { getRuns, type StoredRun } from '../lib/api'
import { runFilename, runToCsv, runToMarkdown, downloadText, type RunLike } from '../lib/exporters'
import { Badge, Collapsible } from './ui'

const fpct = (v: unknown): string =>
  v == null || !isFinite(Number(v)) ? '—' : `${(Number(v) * 100).toFixed(2)}%`
const feur = (v: unknown): string =>
  v == null || !isFinite(Number(v)) ? '—' : `${Math.round(Number(v)).toLocaleString('es-ES')} €`

function asRunLike(r: StoredRun): RunLike {
  return {
    lens: r.lens as RunLike['lens'],
    ts: r.ts,
    meta: (r.summary?.meta ?? {}) as Record<string, any>,
    f_star: Number(r.summary?.f_star ?? NaN),
    recommendation: (r.summary?.recommendation ?? null) as Record<string, any> | null,
    reports: (r.summary?.reports ?? []) as Record<string, any>[],
    warnings: [],
    request: r.request,
  }
}

// Métrica clave por lente en la fila (EV/intento @ f* o P(pass) @ f*).
function keyMetric(r: StoredRun): string {
  const s = r.summary ?? {}
  const star = (s.reports ?? []).find?.(
    (x: Record<string, any>) => Math.abs(x.f - Number(s.f_star)) < 1e-9,
  )
  if (!star) return '—'
  return r.lens === 'axi' ? `P(pass) ${fpct(star.p_pass)}` : `EV ${feur(star.ev_per_attempt)}`
}

interface Props {
  refreshKey: number // se incrementa tras cada corrida grabada
}

export function HistoryPanel({ refreshKey }: Props) {
  const [runs, setRuns] = useState<StoredRun[] | null>(null)
  const [open, setOpen] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  const load = () =>
    getRuns(undefined, 50)
      .then((res) => {
        setRuns(res.runs)
        setErr(null)
      })
      .catch((e) => setErr(String(e)))

  useEffect(() => {
    if (open) load()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, refreshKey])

  const exp = (r: StoredRun, ext: 'csv' | 'md') => {
    const like = asRunLike(r)
    downloadText(
      runFilename(like, ext),
      ext === 'csv' ? runToCsv(like) : runToMarkdown(like),
      ext === 'csv' ? 'text/csv' : 'text/markdown',
    )
  }

  return (
    <Collapsible
      title="Historial de corridas"
      subtitle="P4 · resúmenes ligeros en el servidor"
      onOpenChange={setOpen}
    >
      <div className="mb-2 flex items-center justify-between">
        <span className="text-xs text-zinc-500">
          Cada simulación exitosa se archiva automáticamente (sin trayectorias).
        </span>
        <button
          onClick={load}
          className="rounded-md border border-zinc-700 px-2 py-1 text-[11px] font-medium text-zinc-400 hover:bg-zinc-800"
        >
          Refrescar
        </button>
      </div>
      {err && <p className="text-xs text-rose-300">✗ {err}</p>}
      {!runs && !err && <p className="text-xs text-zinc-500">Sin historial cargado.</p>}
      {runs && runs.length === 0 && (
        <p className="text-xs text-zinc-500">
          Todavía no hay corridas registradas — cada simulación exitosa se archiva aquí.
        </p>
      )}
      {runs && runs.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-zinc-800 text-left text-[11px] uppercase tracking-wide text-zinc-500">
                <th className="px-4 py-2 font-medium">fecha</th>
                <th className="px-3 py-2 font-medium">lente</th>
                <th className="px-3 py-2 font-medium">patas</th>
                <th className="px-3 py-2 text-right font-medium">f*</th>
                <th className="px-3 py-2 text-right font-medium">métrica</th>
                <th className="px-3 py-2 font-medium">ventana</th>
                <th className="px-4 py-2 text-right font-medium">export</th>
              </tr>
            </thead>
            <tbody className="num">
              {runs.map((r) => {
                const legs = (r.summary?.meta?.legs ?? []).length
                const oos = r.request?.oos_start ?? r.summary?.meta?.oos_start
                return (
                  <tr key={r.id} className="border-b border-zinc-800/60 hover:bg-zinc-800/30">
                    <td className="px-4 py-2 text-zinc-300">{r.ts.replace('T', ' ').slice(0, 16)}</td>
                    <td className="px-3 py-2">
                      <Badge variant={r.lens === 'axi' ? 'default' : 'success'}>
                        {r.lens}
                      </Badge>
                      {r.diag_group_id && (
                        <Badge variant="outline" className="ml-1.5">diag</Badge>
                      )}
                    </td>
                    <td className="px-3 py-2">{legs || '—'}</td>
                    <td className="px-3 py-2 text-right">{fpct(r.summary?.f_star)}</td>
                    <td className="px-3 py-2 text-right text-zinc-400">{keyMetric(r)}</td>
                    <td className="px-3 py-2">
                      {oos ? (
                        <Badge variant="warn">OOS {String(oos).slice(0, 10)}</Badge>
                      ) : (
                        <Badge variant="outline">completa</Badge>
                      )}
                    </td>
                    <td className="px-4 py-2 text-right">
                      <button
                        onClick={() => exp(r, 'csv')}
                        className="mr-1.5 rounded-md border border-zinc-700 px-1.5 py-0.5 text-[11px] text-zinc-400 hover:bg-zinc-800"
                      >
                        CSV
                      </button>
                      <button
                        onClick={() => exp(r, 'md')}
                        className="rounded-md border border-zinc-700 px-1.5 py-0.5 text-[11px] text-zinc-400 hover:bg-zinc-800"
                      >
                        MD
                      </button>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </Collapsible>
  )
}
