// Diagnóstico de fuga de selección (P2): compara cada métrica evaluada en la
// VENTANA COMPLETA vs SOLO OOS (regla M2-8). Δ = completa − OOS; si Δ es
// favorable a la ventana completa y significativa, el minado dejó halago en el
// tramo donde se seleccionaron las estrategias. El titular SIEMPRE es OOS:
// esto es contexto de confianza, no otra decisión.

// z de dos proporciones (H0: p1 = p2, error estándar pooled). Aproximación
// diagnóstica: las paths/seqs comparten bloques re-muestreados, así que n es
// un recuento efectivo, no una n de observaciones independientes.
export function twoProportionZ(p1: number, n1: number, p2: number, n2: number): number {
  const p = (p1 * n1 + p2 * n2) / (n1 + n2)
  const se = Math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
  if (!isFinite(se) || se === 0) return 0
  return (p1 - p2) / se
}

export interface DiagMetric {
  metrica: string
  kind: 'pct' | 'eur' | 'usd' | 'days' | 'raw'
  // Dirección del "halago" para interpretar el signo de Δ en filas testables
  // (true: más es mejor, halago = Δ > 0; false: menos es mejor, halago = Δ < 0).
  higherBetter?: boolean
  full: number | null // valor en la ventana completa
  oos: number | null // valor en la ventana OOS (titular)
  // Solo filas de proporción: test z de dos proporciones con recuentos
  // efectivos (n seqs en prop firm, n paths en AXI).
  test?: { pFull: number; pOos: number; nFull: number; nOos: number }
}

export interface DiagReading {
  label: string
  warn: boolean
}

// Lectura de una fila: sin datos → '—'; sin test → informativa; con test,
// z ±1.96 separa ruido de diferencia real, y el signo frente a higherBetter
// decide si la diferencia favorece a la ventana completa (halago).
export function diagLectura(m: DiagMetric): DiagReading {
  if (m.full == null || m.oos == null) return { label: '—', warn: false }
  if (!m.test) return { label: 'informativo', warn: false }
  const d = m.test.pFull - m.test.pOos
  const z = twoProportionZ(m.test.pFull, m.test.nFull, m.test.pOos, m.test.nOos)
  if (!(Math.abs(z) >= 1.96)) return { label: 'compatible con ruido', warn: false }
  const flatters = m.higherBetter ? d > 0 : d < 0
  if (flatters) return { label: 'fuga apreciable (Δ significativo)', warn: true }
  return { label: 'reversal fuera de muestra (reportar tal cual)', warn: false }
}
