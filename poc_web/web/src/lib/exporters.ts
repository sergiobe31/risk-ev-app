// Exportadores de corridas (P4): CSV (tabla completa de reports) y Markdown
// (vista L1+L2 con meta y recomendación). Operan sobre una forma normalizada
// RunLike que construyen tanto los resultados vivos como las filas del
// historial (mismo summary), así que un botón sirve para ambos.
import { bs1At, type Bs1RowAxi, type Bs1RowPropfirm } from './api'

export interface RunLike {
  lens: 'propfirm' | 'axi'
  ts?: string // ISO si viene del historial
  meta: Record<string, any>
  f_star: number
  recommendation: Record<string, any> | null
  reports: Record<string, any>[]
  warnings?: string[]
  request?: Record<string, any>
  // perfiles-v2: los rellena la corrida en vivo desde las reglas del perfil de la CORRIDA (el
  // historial no tiene catálogo: sin ellos, "P(P1∧P2)" y "P(truncada)" sin días).
  nPhases?: number // fases del reto cycle; con 1 -> "P(pasar)"
  campaignCapDays?: number | null // tope de campaña de la suscripción (futuros)
}

export function downloadText(filename: string, text: string, mime = 'text/plain'): void {
  const blob = new Blob([text], { type: `${mime};charset=utf-8` })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

const stamp = (r: RunLike): string =>
  (r.ts ?? new Date().toISOString()).slice(0, 16).replace(/[:T]/g, '')

// --- CSV: una fila por f, todas las columnas del report ----------------------
const PF_COLS = [
  'f', 'p_pass1', 'p_both', 'p_unresolved', 'ev_per_attempt',
  'p_net_negative', 'net_p05', 'net_p50', 'net_p95',
  'mean_payout_eur', 'p_payout_given_funded', 'challenges_per_payout',
  'median_days_first_payout', 'median_days_pass1', 'alive_at_cap_frac',
]
// Fase 2 (FTMO Futures): columnas ADITIVAS (la clave cruda ev_per_attempt se
// conserva — E8: sin renombrar nada existente).
const PF_FUTURES_COLS = [
  'ev_per_day', 'p_truncated', 'p_neither_attempts', 'mean_billings',
  'mean_resets', 'mean_attempts', 'mean_days_campaign', 'mean_payout_deferrals',
  'p_ruin', 'sizing_invalid', 'frac_cap_violations_eval', 'reset_policy',
  'withdrawal_policy',
]
const AXI_COLS = [
  'f', 'p_pass', 'p_pass_raw', 'p_ruin', 'p_ruin_raw', 'p_sin_resolver',
  'median_days_eligibility', 'p90_days_eligibility', 'median_trades_qual',
  'median_max_dd', 'p10_max_dd', 'terminal_p05', 'terminal_p50', 'terminal_p95',
]

function toCsv(cols: string[], rows: Record<string, any>[]): string {
  const esc = (v: unknown): string => {
    if (v == null) return ''
    const s = Array.isArray(v) ? v.join('|') : String(v)
    return /[";\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
  }
  return [cols.join(';'), ...rows.map((r) => cols.map((c) => esc(r[c])).join(';'))].join('\n')
}

export function runToCsv(r: RunLike): string {
  if (r.lens === 'axi') return toCsv(AXI_COLS, r.reports)
  const cols =
    r.meta?.funded_model === 'futures' ? [...PF_COLS, ...PF_FUTURES_COLS] : PF_COLS
  return toCsv(cols, r.reports)
}

// --- Markdown: vista L1+L2 ---------------------------------------------------
const fpct = (v: unknown, d = 2): string =>
  v == null || !isFinite(Number(v)) ? '—' : `${(Number(v) * 100).toFixed(d)}%`
const feur = (v: unknown): string =>
  v == null || !isFinite(Number(v)) ? '—' : `${Math.round(Number(v)).toLocaleString('es-ES')} €`
// bloque-politis-white (fase B): el bloque es un número real (p. ej. 3.064) -> un decimal, como el badge `bs` de la UI. Sirve
// también a las corridas antiguas del historial (block_size entero): salen como «10.0».
const fbs = (v: unknown): string => (v == null || !isFinite(Number(v)) ? '—' : Number(v).toFixed(1))

export function runToMarkdown(r: RunLike): string {
  const L: string[] = []
  const star = r.reports.find((x) => Math.abs(x.f - r.f_star) < 1e-9) ?? {}
  const m = r.meta as Record<string, any>
  const rec = r.recommendation as Record<string, any> | null

  L.push(`# Corrida ${r.lens === 'axi' ? 'AXI' : 'Prop firm'} · ${stamp(r)}`)
  if (m.oos_start) L.push(`\nCorte IS/OOS: **${m.oos_start}** (regla M2-8: titular = OOS)`)
  else L.push('\nVentana completa sin corte — métricas infladas si hubo selección (M2-8).')

  L.push(`\n## Meta\n`)
  for (const leg of m.legs ?? [])
    L.push(`- Pata **${leg.pata}** (${leg.símbolo}, ${leg.modo}): ${leg.trades} trades`)
  L.push(`- Bloques: ${m.n_blocks}${m.trades_in_window != null ? ` · trades en ventana: ${m.trades_in_window}` : ''}`)
  if (m.coverage != null && isFinite(m.coverage)) L.push(`- Cobertura: ${fpct(m.coverage)}`)
  L.push(`- block_size: ${fbs(m.block_size)} · paths: ${m.n_paths} · sizing: ${m.sizing ?? m.motor ?? '—'}${m.day_axis ? ` · eje: ${m.day_axis}` : ''}`)
  if (r.request?.day_axis === undefined && m.float_mode) L.push(`- float_mode: ${m.float_mode}`)
  if (m.funded_model === 'futures' && m.ev_semantics) L.push(`- ${m.ev_semantics}`)

  L.push(`\n## Decisión @ f* = ${fpct(r.f_star)}\n`)
  if (rec) {
    L.push(`**${rec.feasible || rec.recommend_pair ? 'Factible' : 'Sin f factible'}**: ${rec.reason}`)
    L.push('')
  }
  if (r.lens === 'axi') {
    L.push(`- P(pass) F4: ${fpct(star.p_pass)} (IC95 ${fpct(star.p_pass_ci?.[0])} – ${fpct(star.p_pass_ci?.[1])})`)
    L.push(`- P(ruin): ${fpct(star.p_ruin)} (IC95 ${fpct(star.p_ruin_ci?.[0])} – ${fpct(star.p_ruin_ci?.[1])})`)
    L.push(`- Días-elegibilidad p50/p90: ${star.median_days_eligibility ?? '—'} / ${star.p90_days_eligibility ?? '—'}`)
    L.push(`- maxDD p50: ${fpct(star.median_max_dd)} · terminal p05/p50/p95: ${feur(star.terminal_p05)} / ${feur(star.terminal_p50)} / ${feur(star.terminal_p95)}`)
    // bloque-politis-white (fase B): sensibilidad con bloque 1 en f*, con los mismos KPIs que la línea gris de la UI
    // (pasar / caer = P(pass) F4 / P(ruin)). Solo existe con un bloque principal > 1: corridas antiguas y bloque 1, sin línea.
    const bs1 = bs1At<Bs1RowAxi>(m.bs1, r.f_star)
    if (bs1)
      L.push(
        `- Con bloque 1 en f* (sin dependencia entre bloques): P(pass) F4: ${fpct(bs1.p_pass)} · P(ruin): ${fpct(bs1.p_ruin)}`,
      )
  } else {
    // E8: el label condicional aplica SOLO al Markdown; el CSV conserva la
    // clave cruda ev_per_attempt (renombrar rompería consumidores).
    const evLabel = m.funded_model === 'futures' ? 'EV/campaña' : 'EV/intento'
    L.push(`- ${evLabel}: ${feur(star.ev_per_attempt)} (IC95 ${feur(star.ev_ci95?.[0])} – ${feur(star.ev_ci95?.[1])})`)
    if (m.funded_model === 'futures')
      L.push(
        `- EV/día: ${feur(star.ev_per_day)} · P(truncada${
          r.campaignCapDays != null ? ` ${r.campaignCapDays}d` : ''
        }): ${fpct(star.p_truncated)}`,
      )
    L.push(
      m.funded_model === 'futures'
        ? `- P(neto<0): ${fpct(star.p_net_negative)} · P(neither path): ${fpct(star.p_unresolved)} · P(P1): ${fpct(star.p_pass1)}`
        : `- P(neto<0): ${fpct(star.p_net_negative)} · ${
            r.nPhases === 1 ? 'P(pasar)' : 'P(P1∧P2)'
          }: ${fpct(star.p_both)} · P(sin resolver): ${fpct(star.p_unresolved)}`,
    )
    L.push(`- Neto p05/p50/p95: ${feur(star.net_p05)} / ${feur(star.net_p50)} / ${feur(star.net_p95)}`)
    L.push(`- Días al 1er payout p50: ${star.median_days_first_payout ?? '—'} · vivo al cap: ${fpct(star.alive_at_cap_frac)}`)
    // bloque-politis-white (fase B): sensibilidad con bloque 1 en f*, con los mismos KPIs que la línea gris de la UI (pase,
    // P(neto<0), EV). El rótulo del pase es el de la línea de arriba (P(P1∧P2) / P(pasar) / P(P1) en futuros).
    const bs1 = bs1At<Bs1RowPropfirm>(m.bs1, r.f_star)
    if (bs1) {
      const bs1Pass = m.funded_model === 'futures' ? 'P(P1)' : r.nPhases === 1 ? 'P(pasar)' : 'P(P1∧P2)'
      L.push(
        `- Con bloque 1 en f* (sin dependencia entre bloques): ${bs1Pass}: ${fpct(bs1.p_both)} · P(neto<0): ${fpct(bs1.p_net_negative)} · ${evLabel}: ${feur(bs1.ev_per_attempt)}`,
      )
    }
  }

  if (r.warnings?.length) {
    L.push(`\n## Warnings\n`)
    for (const w of r.warnings) L.push(`- ⚠ ${w}`)
  }

  L.push(`\n## Reports por f\n`)
  L.push('```csv\n' + runToCsv(r) + '\n```')
  return L.join('\n') + '\n'
}

export function runFilename(r: RunLike, ext: 'csv' | 'md'): string {
  const oos = r.request?.oos_start ?? r.meta?.oos_start
  const tag = oos ? `oos-${oos}` : 'completa'
  return `corrida_${r.lens}_${stamp(r)}_${tag}.${ext}`
}
