// Gate del laboratorio σ-path (producto gap-régimen-ATR, pieza 3; triggers
// del estatuto, dossier §9b). INFORMATIVO: no dispara nada. Cuelga de
// PortfolioPanel (cero threading en App.tsx). Iconos solo ok/warn (✓/⚠).
import { useState } from 'react'
import type { LegDiagResult, PortfolioLoadResult } from '../lib/api'
import { triggerCompuesto } from '../lib/semaphore'
import { Collapsible, NumberInput } from './ui'

const pct = (v: number | null | undefined, d = 1) =>
  v == null || !isFinite(v) ? '—' : `${(v * 100).toFixed(d)}%`
const usd0 = (v: number | null | undefined) =>
  v == null || !isFinite(v) ? '—' : `$${Math.round(v).toLocaleString('en-US')}`

const GATE_OK = { icon: '✓', cls: 'text-emerald-400' }
const GATE_WARN = { icon: '⚠', cls: 'text-amber-400' }

interface Props {
  portfolio: PortfolioLoadResult
  diag: LegDiagResult | null
}

export function GatePanel({ portfolio, diag }: Props) {
  // Equity viva: default = capital de la cartera; sin localStorage (resetea
  // al recargar — aceptable para una lectura mensual).
  const [equity, setEquity] = useState(portfolio.account_size)
  const threshold = diag?.threshold_pctile ?? 90

  // Trigger 2 (estatuto): POR PATA, nunca mediana; los null no cuentan.
  const over20k = (diag?.rows ?? []).filter(
    (r) => r.e_star_p99 != null && isFinite(r.e_star_p99) && r.e_star_p99 > 20000,
  )
  // Línea informativa: semáforo compuesto (P ≥ umbral Y en clamp hoy).
  const semRows = diag?.rows ?? []
  const nSem = semRows.filter(
    (r) => triggerCompuesto(r.pctile_sl_live, r.would_be_clamp_now, threshold) === true,
  ).length

  const t1Activo = equity < 12000
  const s1 = t1Activo ? GATE_WARN : GATE_OK
  const s2 = over20k.length > 0 ? GATE_WARN : GATE_OK

  return (
    <Collapsible
      title="Gate del laboratorio σ-path · revisión mensual · informativo — no dispara nada"
      defaultOpen={false}
    >
      <div className="space-y-3 text-xs">
        <div className="flex items-center gap-2">
          <span className="text-zinc-400">Equity viva ($)</span>
          <NumberInput value={equity} step={500} min={0} onChange={setEquity} />
        </div>

        <div className="space-y-1.5">
          <p className="flex items-start gap-1.5">
            <span className={s1.cls}>{s1.icon}</span>
            <span className="text-zinc-300">
              Trigger 1 — equity &lt; $12,000:{' '}
              {t1Activo
                ? 'activo: con equity < $12k el canal de dinero sigue vivo; el gate está abierto mientras no decidas subir a $20k'
                : `no activo (equity ${usd0(equity)})`}
            </span>
          </p>
          <p className="flex items-start gap-1.5">
            <span className={diag ? s2.cls : 'text-zinc-500'}>
              {diag ? s2.icon : '·'}
            </span>
            <span className="text-zinc-300">
              Trigger 2 — patas con E*_p99 &gt; $20,000:{' '}
              {diag
                ? `${over20k.length}/${diag.rows.length} patas @ f = ${pct(diag.f, 2)}`
                : 'calcula el diagnóstico para evaluarlo'}
            </span>
          </p>
          {diag && over20k.length > 0 && (
            <ul className="ml-5 space-y-0.5 text-amber-300">
              {over20k.map((r) => (
                <li key={r.pata}>
                  {r.pata}: E*_p99 {usd0(r.e_star_p99)}
                </li>
              ))}
            </ul>
          )}
          <p className="flex items-start gap-1.5">
            <span className="text-zinc-500">·</span>
            <span className="text-zinc-300">
              Trigger 3 — renuncia a $20k: ¿has renunciado a subir a $20k? Si es
              así, el gate se abre (decisión tuya, no medida)
            </span>
          </p>
          <p className="flex items-start gap-1.5">
            <span className="text-zinc-500">·</span>
            <span className="text-zinc-400">
              semáforo encendido en{' '}
              {diag ? `${nSem}/${semRows.length}` : '—'} patas (P ≥ {threshold}{' '}
              y en clamp)
            </span>
          </p>
        </div>

        <div className="space-y-1 border-t border-zinc-800 pt-2 text-[11px] text-zinc-500">
          <p>
            C5: cross-check mensual con el journal del EA — PROGRAMADO, no
            opcional (seguimiento operativo)
          </p>
          <p>
            umbral P{threshold} calibrado con la curva
            GAP-ATR-CURVA 2026-10-12
          </p>
        </div>
      </div>
    </Collapsible>
  )
}
