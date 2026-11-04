// Panel L3 "Diagnóstico de selección" (P2): tabla métrica | ventana completa |
// OOS | Δ | lectura para la corrida gemela (misma request con/sin corte OOS).
// El titular SIEMPRE es la ventana OOS (regla M2-8) — caption fijo abajo.
import { Collapsible } from './ui'
import { diagLectura, type DiagMetric } from '../lib/diag'

function fmt(v: number | null, kind: DiagMetric['kind']): string {
  if (v == null || !isFinite(v)) return '—'
  switch (kind) {
    case 'pct':
      return `${(v * 100).toFixed(1)}%`
    case 'eur':
      return `${v.toLocaleString('es-ES', { maximumFractionDigits: 0 })} €`
    case 'usd':
      return `${v.toLocaleString('es-ES', { maximumFractionDigits: 0 })} $`
    case 'days':
      return `${v.toFixed(0)} d`
    default:
      return v.toFixed(2)
  }
}

export function SelectionDiagPanel({ metrics }: { metrics: DiagMetric[] }) {
  if (!metrics.length) return null
  return (
    <Collapsible
      title="Diagnóstico de selección"
      subtitle="L3 · ventana completa vs OOS — el titular siempre es OOS"
    >
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-zinc-800 text-left text-[11px] uppercase tracking-wide text-zinc-500">
              <th className="px-4 py-2 font-medium">métrica @ f*</th>
              <th className="px-3 py-2 text-right font-medium">ventana completa</th>
              <th className="px-3 py-2 text-right font-medium">OOS (titular)</th>
              <th className="px-3 py-2 text-right font-medium">Δ completa − OOS</th>
              <th className="px-4 py-2 text-right font-medium">lectura</th>
            </tr>
          </thead>
          <tbody className="num">
            {metrics.map((m) => {
              const d = m.full != null && m.oos != null ? m.full - m.oos : null
              const lec = diagLectura(m)
              return (
                <tr key={m.metrica} className="border-b border-zinc-800/60 hover:bg-zinc-800/30">
                  <td className="px-4 py-2 font-medium text-zinc-200">{m.metrica}</td>
                  <td className="px-3 py-2 text-right">{fmt(m.full, m.kind)}</td>
                  <td className="px-3 py-2 text-right font-medium text-emerald-400">
                    {fmt(m.oos, m.kind)}
                  </td>
                  <td className={`px-3 py-2 text-right ${lec.warn ? 'text-amber-300' : 'text-zinc-400'}`}>
                    {d == null ? '—' : `${d >= 0 ? '+' : ''}${fmt(d, m.kind)}`}
                  </td>
                  <td className={`px-4 py-2 text-right text-xs ${lec.warn ? 'text-amber-300' : 'text-zinc-500'}`}>
                    {lec.label}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-3 text-xs leading-relaxed text-zinc-500">
        Regla M2-8: el titular SIEMPRE es la ventana OOS, pase lo que pase aquí. Un Δ
        favorable puede ser fuga de selección, cambio de régimen o ruido — para decidir da
        igual cuál sea: lo evaluable fuera de muestra es lo que hay. Test z de dos
        proporciones (|z| ≥ 1.96) sobre recuentos efectivos de paths/seqs — aproximación
        diagnóstica, no contraste exacto.
      </p>
    </Collapsible>
  )
}
