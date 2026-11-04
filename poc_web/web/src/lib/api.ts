// Tipos y cliente de la API web (shell B: cartera-como-entidad + lentes).

export interface Firm {
  key: string
  display_name: string
  account_size: number
  fee: number
  // Fase 6 (Apex): metadatos del perfil (aditivos; null/ausentes en FTMO)
  fee_ccy?: string // moneda declarada del fee (EUR | USD)
  ruleset_version?: string | null
  day_axis?: string | null // null = día calendario; 'et_session' = sesiones CME ET
  funded_model?: string // 'cycle' (FTMO) | 'ladder' (Apex) | 'futures' (FTMO Futures)
  // onboarding-b1: `extended` (normalizado a boolean en fetchCatalog; el API lo
  // manda 0/1) decide el perfil extendido, ya no la presencia de `rules`.
  extended: boolean
  kind?: 'cfd' | 'futures'
  rules?: ProfileRules | null // panel "Reglas del perfil" (todos los perfiles)
  caveats?: ModelNote[] // supuestos del modelo del perfil (antes string[])
  // onboarding-c: perfiles del usuario (el API manda los booleanos como 0/1 → usar !!x)
  official?: boolean | number
  based_on?: string | null
  verified_on?: string | null
  // perfiles-v2: compañía del perfil y marca de oculto (0/1 → !!x). Los ocultos
  // NO se excluyen del catálogo: el simulador los filtra y el detalle usa sus `rules`.
  company?: string | null
  hidden?: boolean | number
}

// Panel read-only "Reglas del perfil". Unión discriminada por `rules.model`:
// 'cycle' (FTMO CFD) frente a los extendidos, que NO traen `model` de primer
// nivel (su modelo de fondeada vive en `funded.model`).
export type ProfileRules = CycleRules | ExtendedRules

// onboarding-b1: reglas del perfil FTMO CFD (modelo `cycle`).
export interface CycleRules {
  model: 'cycle'
  ruleset_version: string | null
  phases: {
    name: string
    profit_target_pct: number
    max_daily_loss_pct: number
    max_loss_pct: number
    min_trading_days: number
  }[]
  funded: {
    model: 'cycle'
    profit_split_pct: number
    payout_cycle_days: number
    min_payout_profit_pct: number
    max_payout_cycles: number
    horizon_cap_days: number // truncado del modelo, no regla FTMO
  }
  refund_fee_on_first_payout: boolean | number // _jsonable: bool como 0/1
}

// Perfiles extendidos (Apex, FTMO Futures).
export interface ExtendedRules {
  model?: undefined
  ruleset_version: string
  day_axis: string | null
  day_axis_cut_hhmm?: string // corte ET del eje de sesiones ('18:00' Apex / '16:10' FTMO Futures)
  evaluation: {
    profit_target_usd: number | null
    lifetime_days: number | null
    on_horizon_exhausted: string | null
    trailing_usd: number | null
    max_contracts: { max_mini: number; max_micro: number } | null
    // Fase 3 (FTMO Futures; opcionales — solo perfiles futures)
    drawdown?: {
      type: string // 'eod_trailing'
      amount_usd: number
      lock_at_balance: number | null
      breach_on: string
    } | null
    daily_loss_usd?: number | null // DLL hard fatal en USD (solo Pro; null en Growth)
    consistency?: { max_best_day_share: number; strict: boolean } | null // puerta de pase
    min_trading_days?: number
  }
  funded: {
    model: string // 'cycle' (FTMO) | 'ladder' (Apex) | 'futures' (FTMO Futures)
    profit_split_pct: number
    trailing: {
      amount_usd: number
      lock_at_balance: number | null
      frozen_floor: number | null
    } | null
    dll_by_tier_usd: number[] | null
    dll_post_hit: string | null
    scaling_tiers: {
      level: number
      profit_lo: number
      profit_hi: number | null
      max_mini: number
      max_micro: number
    }[]
    payouts: {
      qualifying_day_min_usd: number
      days_per_payout: number
      ladder_usd: number[]
      min_payout_usd: number
      min_balance_first: number
      residual_floor: number
      max_payouts: number
      consistency: { max_best_day_share: number; strict: boolean; window: string }
    } | null
    inactivity: { window_days: number; min_days: number; min_profit_usd: number } | null
    activation: { fee_usd: number; deadline_days: number } | null
    // Fase 3 (FTMO Futures; opcionales — solo perfiles futures)
    drawdown?: {
      type: string // 'eod_trailing'
      amount_usd: number
      lock_at_balance: number | null // CAP del umbral (= account_size → lock permanente)
      breach_on: string
    } | null
    dll?: { type: string; amount_usd: number; post_hit: string } | null
    subscription?: {
      monthly_fee_usd: number
      reset_fee_usd: number
      billing_period_days: number
      campaign_cap_days: number // truncado del modelo, no regla FTMO
      funded_cap_days: number // truncado del modelo, no regla FTMO
      funded_start_delay_days: number
    } | null
    payouts_futures?: {
      qualifying_day_min_usd: number
      days_per_payout: number
      withdraw_share: number // fracción retirable del profit (0.5 Growth / 1.0 Pro)
      request_cap_usd: number
      min_new_profit_frac: number
      min_payout_usd: number
      withdrawal_policy: string // 'greedy' | 'buffer_preserving'
      buffer_usd: number | null
    } | null
  }
}

export interface Leg {
  path: string
  folder: string
  name: string
  kind?: string // 'csv' | 'sqx_futures' (onboarding-b1, GT-13)
  asset?: string | null // clase de activo del bróker de la ficha; null en un .sqx sin ficha
  // entrada-4-bloques F5 (solo patas .sqx; las CSV no traen estas claves): bróker de la ficha del símbolo,
  // estado de la ficha y motivo ('sin_ficha' y 'no_usable' llevan el porqué cuando lo hay).
  broker?: string | null
  ficha?: FichaState
  motivo?: string | null
}

export interface OosCampaign {
  key: string
  cut: string // ISO date del corte IS/OOS registrado
  source: string // procedencia (de dónde salió el corte)
}

export interface Catalog {
  firms: Firm[]
  model_notes_axi?: ModelNote[] // onboarding-b1: supuestos de la lente AXI
  axi_stages: AxiStage[]
  legs: Leg[]
  oos_campaigns: OosCampaign[]
}

export interface AxiStage {
  key: string
  display_name: string
  profit_target_pct: number | null
  min_days: number | null
  min_unique_trades: number | null
  max_loss_pct: number
  allocation_usd: number
}

// --- Cartera (POST /api/portfolio/load) ------------------------------------
export type GuardStatus = 'ok' | 'warn' | 'error'

export interface Guard {
  status: GuardStatus
  detail: string
}

export interface LegInfo {
  pata: string
  símbolo?: string
  modo?: string
  trades?: number
  status: GuardStatus
  warnings: string[]
  error?: string | null
  sqx?: string | null
  // entrada-4-bloques F5 (solo filas de patas .sqx cargadas; en las CSV y en las filas con error no existen)
  broker?: string | null // bróker de la ficha con que se leyó la pata
  ficha_sin_verificar?: string[] // campos de la ficha en needs_review (puede ser [])
  mm_backtest?: string | null // método de MM del backtest (lastSettings.xml); null si el .sqx no lo trae
  paridad_frac_ok?: number | null // fracción (0..1) de órdenes cuyo lote reproduce el riesgo fijo; null si el MM no es FixedAmount
}

export interface PortfolioLoadResult {
  portfolio_id: string
  ok: boolean
  legs: LegInfo[]
  guards: Record<string, Guard>
  warnings: string[]
  n_blocks: number
  coverage: number | null
  account_size: number
  f_ref: number
}

export interface LoadParams {
  legs: string[]
  account_size: number
  f_ref: number
  // Fase 6 (Apex): con una firm de sesiones ET, la carga añade la guarda de
  // compliance overnight (flat 16:59 ET) y el aviso de universo de datos.
  firm_key?: string
  source_tz?: string | null
}

// --- Lente prop firm (POST /api/simulate/propfirm) -------------------------
export interface LegRow {
  pata: string
  símbolo: string
  modo: string
  trades: number
}

// bloque-politis-white (fase B): sensibilidad con bloque 1 SOLO en f* (`meta.bs1`; plan §8 E15/E16, S = {f*}). Es una LISTA
// con UNA fila (la f* de la rejilla), no un objeto. Llega `null` si el bloque principal es ≤ 1 (+1e-9) o el modo es
// `fallback_iid` (la cifra sería la misma) y, en la lente prop firm, también si la secuencia no resuelve ningún camino. En el
// par 2D la clave NO existe. OJO: la clave es `bs1`, no `bs1_at_fstar` como decía §2.4 del plan (nota H6 de la revisión de
// código de la fase A). Los NaN del motor llegan como null (`_jsonable`).
export interface Bs1RowAxi {
  f: number
  p_pass: number
  p_ruin: number
}

export interface Bs1RowPropfirm {
  f: number
  p_both: number
  p_net_negative: number
  ev_per_attempt: number | null
}

export interface Meta {
  firm: string
  account_size: number
  fee_eur: number
  legs: LegRow[]
  n_blocks: number
  coverage: number
  n_no_valley: number
  pnl_w_no_valley: number
  // bloque-politis-white: ahora es un número REAL (p. ej. 3.064), ya no un entero; se pinta con un decimal.
  block_size: number
  bs1?: Bs1RowPropfirm[] | null // sensibilidad con bloque 1 en f* (ver Bs1RowPropfirm); ausente en el par 2D
  float_mode: string
  sizing: string
  n_paths: number
  elapsed_s: number
  // F2c (opcionales: presentes según request)
  oos_start?: string | null // corte IS/OOS aplicado (null = ventana completa)
  trades_in_window?: number // trades que sobreviven al corte
  day_axis?: string // 'ftmo' | 'servidor' | 'et_session' (perfil Apex)
  day_axis_note?: string // nota de optimismo solo a eje servidor
  // Fase 6 (Apex): eco del perfil extendido y del plumbing aplicado
  ruleset_version?: string | null
  source_tz?: string | null // tz usada para el etiquetado de sesiones ET
  trailing?: boolean | number // _jsonable serializa bool como 0/1
  n_copied_accounts?: number
  cap_violation_tol?: number
  sizing_invalid_any?: boolean | number // algún f superó el umbral de caps
  // Fase 2 (FTMO Futures; solo con funded.model == 'futures')
  funded_model?: string
  reset_policy?: string // 'billing_retry' | 'paid_reset' (eco del request)
  withdrawal_policy?: string // 'greedy' | 'buffer_preserving' (eco del request)
  day_axis_cut_hhmm?: string // '16:10' en FTMO Futures
  ev_semantics?: string // EV incondicional por campaña; p_pass1/p_unresolved sobre N paths
  model_context?: ModelContext // onboarding-b1
  recipe?: Recipe // entrada-4-bloques F4: solo con carteras .sqx (las CSV no llevan la clave)
}

// El shim serializa NaN/inf del motor como null (ver _jsonable en api/main.py),
// así que los campos derivados de payout/EV pueden llegar a null.
export interface Report {
  f: number
  n_paths: number
  p_pass1: number
  p_pass1_ci: [number | null, number | null]
  p_both: number
  p_both_ci: [number | null, number | null]
  p_unresolved: number
  ev_per_attempt: number | null
  ev_ci95: [number | null, number | null]
  net_mean: number | null
  p_net_negative: number
  net_p05: number | null
  net_p50: number | null
  net_p95: number | null
  mean_payout_eur: number | null
  p_payout_given_funded: number | null
  challenges_per_payout: number | null
  median_days_first_payout: number | null
  median_days_pass1: number | null
  alive_at_cap_frac: number | null
  funded_cut_by_data_frac?: number | null // onboarding-b1: solo cycle; fondeadas vivas al acabar los datos
  // Fase 6 (Apex; aditivos — null/0 en FTMO; _jsonable: nan→null, bool→0/1)
  p_expired?: number | null // P(expirado por lifetime | resuelto)
  mean_dll_hits?: number | null // hits medios de DLL por PA (ladder)
  mean_consistency_deferrals?: number | null // denegaciones 50% medias
  p_inactivity_death?: number | null // P(muerte por inactividad | fondeada)
  p_closed_after_ladder?: number | null // P(cierre feliz tras el último retiro)
  n_copied_accounts?: number
  sizing_invalid?: boolean | number // algún cap de contratos violado > tol
  frac_cap_violations_eval?: number | null
  frac_cap_violations_pa?: number | null
  max_simultaneous_micros?: number | null
  warnings?: string[] // warnings estructurados del motor (SIZING_INVALID, MULTI_CUENTA)
  // Fase 2 (FTMO Futures; aditivos — null/ausentes en perfiles no futures)
  ev_per_day?: number | null // EUR/día = ev_per_campaign / días campaña media
  p_truncated?: number | null // P(campaña truncada en el cap 183d — del modelo)
  p_neither_attempts?: number | null // neither / total de intentos de campaña
  mean_billings?: number | null // mensualidades medias por campaña
  mean_resets?: number | null // resets de pago medios por campaña
  mean_attempts?: number | null // intentos medios por campaña
  mean_days_campaign?: number | null // días medios consumidos por campaña
  mean_payout_deferrals?: number | null // payouts diferidos medios (paths passed)
  p_ruin?: number | null // P(bankroll arruinado) — solo con stop_on_insolvency
  reset_policy?: string // eco de la política aplicada
  withdrawal_policy?: string // eco de la política aplicada
}

export interface Recommendation {
  f: number
  feasible: boolean
  ev_per_attempt: number
  p_net_negative: number
  reason: string
}

export interface Trajectory {
  steps: number[]
  p05: number[]
  p50: number[]
  p95: number[]
}

export interface SimResult {
  meta: Meta
  warnings: string[]
  reports: Report[]
  recommendation: Recommendation | null
  // null en perfiles futures (sequence_ev_futures no produce trayectorias)
  trajectory: Trajectory | null
  f_star: number
  intentos_garantizados: number | null
}

export interface PropfirmSimParams {
  portfolio_id: string
  firm_key: string
  f_lo: number
  f_hi: number
  f_step: number
  n_paths: number
  n_challenges: number
  n_sequences: number
  capital0_eur: number
  fx_eur_usd: number
  tol_neg: number
  block_size: number
  stop_on_insolvency: boolean
  oos_start?: string | null
  day_axis?: 'ftmo' | 'servidor' // eje del daily-loss reset (default API = servidor)
  // Fase 6 (Apex): tz de los timestamps para sesiones ET (null = default del
  // backend, documentado en meta.source_tz), copy trading y umbral de caps.
  source_tz?: string | null
  n_copied_accounts?: number
  cap_violation_tol?: number
  // Fase 2 (FTMO Futures): políticas de la campaña de suscripción. El backend
  // las aplica solo con funded.model == 'futures'; en otros perfiles las ignora
  // (warning si no son los defaults).
  reset_policy?: 'billing_retry' | 'paid_reset'
  withdrawal_policy?: 'greedy' | 'buffer_preserving'
}

// --- F3: par 2D (POST /api/simulate/propfirm_pair) ---------------------------
// Celda (f_challenge, f_funded): report completo del motor (mismos campos que
// Report; f == f_challenge por compatibilidad) más el par que lo etiqueta.
export interface PairCell extends Report {
  f_challenge: number
  f_funded: number
}

export interface PairRecommendation {
  f_challenge: number
  f_funded: number
  f_single: number // ganador de la diagonal (baseline)
  ev_pair: number
  ev_single: number
  delta_ev: number
  delta_ev_ci: [number | null, number | null] // IC95 bootstrap pareado
  p_pasar: number
  p_unresolved: number
  alive_at_cap_frac: number
  on_boundary: [boolean, boolean] // (f_c en borde?, f_f en borde?) — llega como 0/1
  feasible: boolean
  recommend_pair: boolean // false → el IC del ΔEV toca 0: quedarse en diagonal
  reason: string
  n_boot_ok: number
}

export interface CapSensRow {
  celda: string
  f_challenge: number
  f_funded: number
  ev_cap_max: number | null
  ev_cap10: number | null
  vivo_al_cap: number | null
}

// bloque-politis-white (fase B): en el par 2D la clave `bs1` NO existe (sin sensibilidad con bloque 1, fuera de alcance).
export interface PairMeta extends Omit<Meta, 'bs1'> {
  grid_n?: number
  cells_resolved?: number
}

export interface PairResult {
  meta: PairMeta
  warnings: string[]
  cells: PairCell[]
  recommendation: PairRecommendation | null
  cap_sensitivity: CapSensRow[]
}

// --- onboarding-b1 ---------------------------------------------------------
// Supuestos del modelo (config/model_notes.yaml) y contexto de modelo por corrida
// (meta.model_context). OJO: `_jsonable` serializa los bool como 0/1, así que
// `has_real_clustering` llegaba como 0 | 1 | null (triestado). Desde el guardrail
// bloque-politis-white (2026-10-21) ya no se calcula y llega siempre `null`; se conserva la
// normalización y el tipo anulable. En el resto del front se usa el valor ya normalizado y
// nunca `=== true`.
export type Sesgo = 'neutral' | 'optimista' | 'conservador'

export interface ModelNote {
  id: string
  titulo: string
  texto: string // en `model_context.notes` ya viene rellenado con las cifras de la corrida
  sesgo: Sesgo
}

export interface ModelContext {
  block_size: {
    value: number // float (p. ej. 3.064), igual que `meta.block_size`
    mode: 'auto' | 'manual' | 'fallback_iid'
    // bloque-politis-white (fase B): método con que se eligió el bloque y eco del estimador. `series` es 'R' (resultado
    // por episodio en riesgos) en auto y fallback_iid; `opt_m`/`m` solo existen si el estimador llegó a correr (auto).
    method: 'politis_white' | 'manual' | 'fallback_iid'
    series: 'R' | null
    n_excluded_terms: number | null // operaciones sin riesgo por lote válido (no suman a su episodio)
    opt_m: number | null
    m: number | null
    reason: string
    has_real_clustering: boolean | null // desde bloque-politis-white siempre null (ya no se calcula)
    n_blocks: number
    calibrated_at_f: number | null // desde bloque-politis-white siempre null (la serie en R no tiene f)
  }
  data_span_days: number
  funded_horizon?: { cap_days: number | null; window_blocks: number } // solo prop firm
  notes: ModelNote[]
}

/** Booleano del API (0/1/true/false) -> boolean. */
export const asBool = (v: unknown): boolean => !!v

/** Triestado del API: null/undefined se conserva; el resto pasa por `!!`. */
export const asTriBool = (v: unknown): boolean | null => (v == null ? null : !!v)

/** `title` de la línea «Con bloque 1» bajo los KPIs de f* (bloque-politis-white, fase B): la comparten las dos lentes. */
export const BS1_TITLE =
  'Misma corrida en f* con bloques sueltos (bloque 1): la cifra prudente si la dependencia entre bloques de tu backtest no se mantiene.'

/**
 * Fila de `meta.bs1` (sensibilidad con bloque 1) que corresponde a la f dada, o `undefined` si no hay (`bs1` null, ausente o
 * sin esa f). Hoy la lista trae UNA fila, la de f* (E16: S = {f*}); buscar por f en vez de tomar `[0]` es lo que pide E16 si
 * algún día S pasa a ser la rejilla. Misma tolerancia (1e-9) que el resto del front para localizar f*.
 */
export function bs1At<R extends { f: number }>(rows: R[] | null | undefined, f: number): R | undefined {
  return Array.isArray(rows) ? rows.find((r) => Math.abs(r.f - f) < 1e-9) : undefined
}

function normalizeModelContext(mc: ModelContext | undefined): ModelContext | undefined {
  if (!mc) return mc
  return {
    ...mc,
    block_size: {
      ...mc.block_size,
      has_real_clustering: asTriBool(mc.block_size.has_real_clustering),
    },
  }
}

function normalizeMeta<M extends { model_context?: ModelContext }>(meta: M): M {
  return meta.model_context
    ? { ...meta, model_context: normalizeModelContext(meta.model_context) }
    : meta
}

async function handle<T>(res: Response): Promise<T> {
  const body = await res.json()
  if (!res.ok) {
    throw new Error(typeof body.detail === 'string' ? body.detail : `HTTP ${res.status}`)
  }
  return body as T
}

export async function fetchCatalog(): Promise<Catalog> {
  const c = await handle<Omit<Catalog, 'firms'> & { firms: (Omit<Firm, 'extended'> & { extended: unknown })[] }>(
    await fetch('/api/catalog'),
  )
  // onboarding-b1: `extended` llega como 0/1 (ver _jsonable) -> boolean.
  return { ...c, firms: c.firms.map((f) => ({ ...f, extended: asBool(f.extended) })) }
}

export async function loadPortfolio(params: LoadParams): Promise<PortfolioLoadResult> {
  return handle(
    await fetch('/api/portfolio/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    }),
  )
}

export async function runPropfirm(params: PropfirmSimParams): Promise<SimResult> {
  const r = await handle<SimResult>(
    await fetch('/api/simulate/propfirm', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    }),
  )
  return { ...r, meta: normalizeMeta(r.meta) }
}

export async function runPropfirmPair(params: PropfirmSimParams): Promise<PairResult> {
  const r = await handle<PairResult>(
    await fetch('/api/simulate/propfirm_pair', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    }),
  )
  return { ...r, meta: normalizeMeta(r.meta) }
}

// --- Lente AXI (POST /api/simulate/axi) --------------------------------------
// Semántica F4 (meta.semantics === "F4"; análisis y adjudicación en
// ANALISIS_CASOS_PASS_F4.md §9-§11): p_pass/p_ruin son POST-CONTINUACIÓN —
// regla doble E ≥ target ∧ T(d) por fecha de apertura ≥ min_unique_trades; si
// falta T, el path CONTINÚA a f constante hasta cualificar, arruinarse o
// timeout. p_pass_raw/p_ruin_raw = PRIMER CRUCE (semántica del motor, ancla de
// fidelidad tolerancia cero). p_pass_suelo = pass ya cualificado en el primer
// toque del target. post_target = reparto del cubo degradado (primer cruce
// pass con T < suelo): 2a → n_ruin / 2b → n_pass / 2c → n_unresolved.
// p_sin_resolver = neither + timeout post-continuación (pass+ruin+sin_resolver
// = 1; p_blown no se expone). días-elegibilidad = máx(days_cal, min_days) y
// trades, ambos sobre passes F4 (null si no hay); days_cal es ESTIMACIÓN
// calendario (meta.days_cal). Los campos *_to_target son RAW (primer cruce).
// NaN del motor → null.
export interface PostTarget {
  n_deg: number
  n_ruin: number
  n_pass: number
  n_unresolved: number
}

export interface AxiReport {
  f: number
  n_paths: number
  p_pass: number // F4 (post-continuación)
  p_pass_ci: [number, number]
  p_pass_raw: number // primer cruce (semántica motor)
  p_pass_raw_ci: [number, number]
  p_pass_suelo: number // pass ya cualificado en el primer toque
  p_ruin: number // F4 (incluye ruinas 2a destapadas por la continuación)
  p_ruin_ci: [number, number]
  p_ruin_raw: number
  p_ruin_raw_ci: [number, number]
  p_sin_resolver: number // neither + timeout
  post_target: PostTarget
  median_days_eligibility: number | null
  p90_days_eligibility: number | null
  median_trades_qual: number | null
  p90_trades_qual: number | null
  median_days_to_target: number | null // raw (primer cruce)
  p90_days_to_target: number | null
  median_trades_to_target: number | null
  p90_trades_to_target: number | null
  median_max_dd: number | null
  p10_max_dd: number | null
  terminal_p05: number | null
  terminal_p50: number | null
  terminal_p95: number | null
}

export interface AxiRecommendation {
  f: number
  feasible: boolean
  p_pass: number
  p_ruin: number
  median_days: number | null
  p90_days: number | null
  slow: boolean
  wide_ci: boolean
  reason: string
}

export interface DaysCalMeta {
  label: string
  formula: string
  gap_mean_days: number
  gap_mean_source: string
  nota: string
}

export interface AxiMeta {
  etapa: string
  etapa_display: string
  min_days: number | null
  min_unique_trades: number | null
  floors_applied: {
    min_days: number | null
    min_unique_trades: number | null
  }
  semantics: string
  float_convention: string
  f_star_shift_note: string
  policy_assumption: string
  days_cal: DaysCalMeta
  baseline: number
  pass_line: number
  ruin_line: number
  horizon_days: number | null
  legs: LegRow[]
  motor: string
  sizing: string
  // bloque-politis-white: ahora es un número REAL (p. ej. 3.064), ya no un entero; se pinta con un decimal.
  block_size: number
  bs1?: Bs1RowAxi[] | null // sensibilidad con bloque 1 en f* (ver Bs1RowPropfirm/Bs1RowAxi)
  n_paths: number
  n_blocks: number
  n_blocks_full: number
  crossing: number
  coverage: number | null
  elapsed_s: number
  // F2c (opcionales: presentes según request)
  oos_start?: string | null
  trades_in_window?: number
  model_context?: ModelContext // onboarding-b1
  recipe?: Recipe // entrada-4-bloques F4: solo con carteras .sqx (las CSV no llevan la clave)
}

export interface AxiSimResult {
  meta: AxiMeta
  warnings: string[]
  reports: AxiReport[]
  recommendation: AxiRecommendation | null
  f_star: number
}

export interface AxiSimParams {
  portfolio_id: string
  stage: string
  f_lo: number
  f_hi: number
  f_step: number
  n_paths: number
  tol_ruin: number
  block_size: number
  horizon_days?: number | null
  oos_start?: string | null // corte IS/OOS (regla M2-8); null = ventana completa
}

export async function runAxi(params: AxiSimParams): Promise<AxiSimResult> {
  const r = await handle<AxiSimResult>(
    await fetch('/api/simulate/axi', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    }),
  )
  return { ...r, meta: normalizeMeta(r.meta) }
}

// --- Historial de corridas (P4) ----------------------------------------------
// Resumen ligero elegido por el cliente: SIN trajectory ni seq.
export interface RunRecordIn {
  lens: 'propfirm' | 'axi'
  portfolio_id: string
  diag_group_id?: string | null // enlaza la gemela del diagnóstico
  request: Record<string, unknown>
  summary: Record<string, unknown> // meta + f_star + recommendation + reports
}

export interface StoredRun {
  id: string
  ts: string
  lens: string
  portfolio_id: string
  diag_group_id?: string | null
  git_sha?: string | null
  request: Record<string, any>
  summary: Record<string, any>
}

export async function postRun(rec: RunRecordIn): Promise<StoredRun> {
  return handle(
    await fetch('/api/runs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(rec),
    }),
  )
}

export async function getRuns(lens?: string, limit = 50): Promise<{ runs: StoredRun[] }> {
  const q = new URLSearchParams()
  if (lens) q.set('lens', lens)
  q.set('limit', String(limit))
  return handle(await fetch(`/api/runs?${q.toString()}`))
}

// --- Diagnóstico por pata (POST /api/portfolio/{id}/leg_diagnostics) ---------
export interface LegDiagRow {
  pata: string
  símbolo: string
  modo: string
  trades: number
  frac_min_lot_ref: number
  floor_risk_pct: number | null
  implied_cost_per_lot: number | null
  expected_cost_per_lot: number | null
  p_pass_solo: number | null
  p_ruin_solo: number | null
  warnings: string[]
  error: string | null
  // Régimen ATR / canal de dinero (aditivos — null en modo rápido)
  clamp_rate?: number | null
  floor_risk_med_pct_e?: number | null // ya en %
  floor_risk_p99_pct_e?: number | null // ya en %
  e_star_med?: number | null // $
  e_star_p99?: number | null // $
  sl_live_per_lot?: number | null
  pctile_sl_live?: number | null
  sl_live_asof?: string | null
  would_be_clamp_now?: boolean | number | null // _jsonable: bool como 0/1 (L197)
  regimen_warns?: string[]
}

export interface LegDiagParams {
  f: number
  stage: string
  n_paths: number
  block_size: number
}

export interface LegDiagResult {
  f: number
  stage: string
  stage_display: string
  baseline: number
  n_paths: number
  block_size: number
  float_convention: string // "bound (referencia motor)" — no comparar con el lente AXI (joint)
  rows: LegDiagRow[]
  // Régimen ATR (aditivos — ausentes en servidores sin la Fase 1)
  account_size?: number
  threshold_pctile?: number
  pending_calibration?: boolean | number // _jsonable: bool como 0/1
  threshold_source?: string
  reference_population?: string
  hybrid_label?: string
}

export async function fetchLegDiagnostics(
  portfolioId: string,
  params: LegDiagParams,
): Promise<LegDiagResult> {
  return handle(
    await fetch(`/api/portfolio/${portfolioId}/leg_diagnostics`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    }),
  )
}

// --- onboarding-c ---------------------------------------------------------
// Perfiles de prop firm del usuario (GET/POST/PUT/DELETE /api/profiles, ver
// poc_web/api/profiles.py: fuente de verdad de las formas JSON).
export interface ProfileView {
  key: string
  display_name: string
  official: boolean | number
  company: string | null // perfiles-v2
  hidden: boolean | number // perfiles-v2: oculto (no borrado)
  template: string | null // perfiles-v2: id de plantilla si se creó desde cero
  based_on: string | null // id del oficial raíz (null en los oficiales)
  verified_on: string | null // AAAA-MM-DD
  notes: string | null
  created: string | null
  updated: string | null
  model: string // 'cycle' | 'ladder' | 'futures'
  kind: 'cfd' | 'futures'
  ruleset_version: string | null
  needs_recreate: boolean | number
  spec: Record<string, any>
}

export interface ProfileFileError {
  key: string
  error: string
}

export interface ProfilesList {
  profiles: ProfileView[]
  errors: ProfileFileError[]
}

export interface ProfileSchemaField {
  path: string // p. ej. phases.0.profit_target_pct
  label: string
  unit: string
  type: 'number' | 'integer' | 'boolean' | 'string' | 'enum'
  min: number | null
  max: number | null
  min_exclusive: boolean | number
  section: string // Cuenta | Fase n | Fondeada | Suscripción
  reference?: unknown // valor del oficial raíz (GET /{key}/schema)
  example?: unknown // valor del oficial base (GET /templates/{tid}: `reference` se llama `example`)
  options?: string[] // solo type === 'enum'
}

// perfiles-v2: plantillas de "Nuevo perfil" desde cero (GET /api/profiles/templates[/{tid}]).
export interface TemplateInfo {
  id: string
  label: string
  kind: 'cfd' | 'futures'
  based_on: string
  n_phases: number
  n_fields: number
}

export interface TemplateDetail extends TemplateInfo {
  spec: Record<string, any> // la raíz con cada camino editable a null
  schema: ProfileSchemaField[]
}

export interface ProfileValidation {
  ok: boolean
  errors: string[]
}

export interface ProfileExport {
  format: string
  version: number
  key: string
  based_on: string
  template?: string | null
  verified_on: string | null
  notes: string | null
  spec: Record<string, any>
}

export interface ProfileImportResult {
  profile: ProfileView
  renamed_from: string | null
  message: string
}

// Los errores de perfiles llegan como detail:string o detail:{errors:[...]} (422).
export class ProfileApiError extends Error {
  status: number
  errors: string[]
  constructor(status: number, errors: string[]) {
    super(errors.join(' · '))
    this.status = status
    this.errors = errors
  }
}

// --- perfiles-v2: helpers de presentación de la vista Perfiles ---
export const NO_COMPANY = 'Sin compañía'

export const companyLabel = (p: { company: string | null }): string =>
  p.company?.trim() || NO_COMPANY

// "FTMO Challenge 50k (tuyo)" -> "FTMO Challenge 50k" (el "(tuyo)" se pinta aparte)
export const plainName = (p: { display_name: string }): string => p.display_name.replace(/\s*\(tuyo\)\s*$/, '')

const plural = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`

// "FTMO · 3 oficiales · 1 tuyo"
export function companySummary(name: string, items: { official: boolean | number }[]): string {
  const off = items.filter((p) => !!p.official).length
  const own = items.length - off
  return [name, off ? plural(off, 'oficial', 'oficiales') : null, own ? plural(own, 'tuyo', 'tuyos') : null]
    .filter(Boolean)
    .join(' · ')
}

// '2026-10-16' -> '16/10/2026' (sin pasar por Date: evita desfases de zona horaria)
export function fmtDate(iso: string | null): string {
  const m = iso ? /^(\d{4})-(\d{2})-(\d{2})/.exec(iso) : null
  return m ? `${m[3]}/${m[2]}/${m[1]}` : (iso ?? '')
}

// Texto legible de cualquier error de la API de perfiles (409 "deja al menos un perfil visible"...).
export function profileErrText(e: unknown): string {
  return e instanceof ProfileApiError ? e.errors.join(' · ') : String((e as Error)?.message ?? e)
}

async function profileHandle<T>(res: Response): Promise<T> {
  let body: any = null
  try {
    body = await res.json()
  } catch {
    body = null
  }
  if (!res.ok) {
    const d = body?.detail
    let errs: string[]
    if (typeof d === 'string') errs = [d]
    else if (Array.isArray(d?.errors)) errs = d.errors.map(String)
    else errs = [`HTTP ${res.status}`]
    throw new ProfileApiError(res.status, errs)
  }
  return body as T
}

const jsonInit = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
})

export async function listProfiles(): Promise<ProfilesList> {
  return profileHandle(await fetch('/api/profiles'))
}

export async function fetchProfileSchema(key: string): Promise<ProfileSchemaField[]> {
  return profileHandle(await fetch(`/api/profiles/${encodeURIComponent(key)}/schema`))
}

export async function validateProfile(body: {
  key: string
  spec: Record<string, any>
  based_on?: string // duplicado o edición de un duplicado
  template?: string // perfil desde plantilla (alternativa excluyente a based_on)
  editing?: boolean
  verified_on?: string | null
  notes?: string | null
}): Promise<ProfileValidation> {
  return profileHandle(await fetch('/api/profiles/validate', jsonInit('POST', body)))
}

// Duplicar (`base`) o crear desde cero (`template` + `spec` completa; `company` y
// `display_name` van DENTRO de la spec). Excluyentes.
export type CreateProfileBody =
  | { base: string; key: string; display_name?: string }
  | {
      template: string
      key: string
      spec: Record<string, any>
      verified_on?: string | null
      notes?: string | null
    }

export async function createProfile(body: CreateProfileBody): Promise<ProfileView> {
  return profileHandle(await fetch('/api/profiles', jsonInit('POST', body)))
}

export async function listTemplates(): Promise<TemplateInfo[]> {
  return profileHandle(await fetch('/api/profiles/templates'))
}

export async function fetchTemplate(tid: string): Promise<TemplateDetail> {
  return profileHandle(await fetch(`/api/profiles/templates/${encodeURIComponent(tid)}`))
}

// Oculta (`hidden: true`) o restaura (`false`) perfiles; atómico. 404 con ids
// desconocidos y 409 si no quedara ningún perfil visible (ProfileApiError.status).
export async function setProfilesHidden(
  keys: string[],
  hidden: boolean,
): Promise<{ hidden: string[] }> {
  return profileHandle(await fetch('/api/profiles/visibility', jsonInit('POST', { keys, hidden })))
}

export async function updateProfile(
  key: string,
  body: { spec: Record<string, any>; verified_on?: string | null; notes?: string | null },
): Promise<ProfileView> {
  return profileHandle(
    await fetch(`/api/profiles/${encodeURIComponent(key)}`, jsonInit('PUT', body)),
  )
}

export async function deleteProfile(key: string): Promise<{ deleted: string }> {
  return profileHandle(
    await fetch(`/api/profiles/${encodeURIComponent(key)}`, { method: 'DELETE' }),
  )
}

export async function exportProfile(key: string): Promise<ProfileExport> {
  return profileHandle(await fetch(`/api/profiles/${encodeURIComponent(key)}/export`))
}

// Se manda el TEXTO del fichero tal cual: el servidor lo parsea en modo estricto
// (rechaza NaN/Infinity y valida format/version).
export async function importProfile(rawJson: string): Promise<ProfileImportResult> {
  return profileHandle(
    await fetch('/api/profiles/import', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: rawJson,
    }),
  )
}

// --- entrada-4-bloques F5: fichas de símbolo -----------------------------------------------------
// Fuente de verdad de las formas JSON: poc_web/api/symbols.py (router /api/symbols) y, para la receta y las filas
// de carga, poc_web/api/main.py. Aquí NO se normalizan los booleanos: el router de símbolos los devuelve como
// booleanos JSON reales (no pasa por `_jsonable`).

/** Valor de `Leg.kind` de una pata `.sqx` (constante `KIND_SQX` del backend). */
export const SQX_LEG_KIND = 'sqx_futures'

export type FichaState = 'ok' | 'sin_ficha' | 'no_usable'
export type AssetClass = 'cfd' | 'futures'
export type ClockId = 'ny_plus_7' | 'ny_wall'
export type SourceKind = 'official' | 'mt5_sheet' | 'sqx' | 'manual'
export type CommissionKind = 'per_lot_rt' | 'pct_notional_rt'

/** Procedencia de un campo de la ficha. `date` puede ser null (p. ej. las oficiales de los YAML de costes). */
export interface FieldSource {
  from: SourceKind
  date: string | null
}

/** Exactamente una de las dos claves (ida y vuelta). */
export interface Commission {
  per_lot_rt?: number // USD por lote
  pct_notional_rt?: number // % del nocional
}

export interface BrokerView {
  id: string
  display_name: string
  company: string
  currency: string // solo 'USD' en la v1
  asset_class: AssetClass
  clock: ClockId
  ohlc_dir: string | null
  official: boolean
}

export interface SymbolView {
  broker: string
  sqx_symbol: string
  ticker: string
  asset_class: AssetClass
  point_value_usd: number
  lot_step: number
  min_lot: number
  max_lot: number | null
  commission: Commission
  spread_px: number | null // unidades de PRECIO (no pips ni puntos)
  ohlc_m1: string | null
  tick_size: number | null
  sources: Record<string, FieldSource>
  needs_review: string[] // campos sin verificar
  verified_on: string | null
  notes: string | null
  official: boolean
  usable: boolean
  motivo: string | null // por qué no es usable
  hidden: boolean
}

export interface SymbolsList {
  brokers: BrokerView[]
  symbols: SymbolView[]
  errors: string[]
  profiles: Record<string, string | null>
  lenses: Record<string, string>
}

export interface CommissionOption {
  id: CommissionKind
  label: string
  unit: string
  min: number
  max: number
  applies_to: AssetClass[]
}

export interface SymbolSchemaField {
  name: string
  label: string
  type: 'broker' | 'string' | 'number' | 'commission' | 'date' | 'text'
  required: boolean
  section: string
  editable_on_edit?: boolean // false = fijo al editar (bróker y símbolo SQX)
  help?: string
  unit?: string
  min?: number
  min_exclusive?: boolean
  max?: number
  fixed?: Partial<Record<AssetClass, number>> // p. ej. {futures: 1}: valor fijo en ese tipo de bróker
  options?: CommissionOption[] // solo type === 'commission'
}

export interface BrokerSchemaField {
  name: string
  label: string
  type: 'string' | 'enum'
  required: boolean
  pattern?: string
  max?: number
  options?: string[]
  default?: string
  help?: string
}

export interface EnumOption {
  id: string
  label: string
}

export interface SymbolSchema {
  version: number
  sheet_fields: SymbolSchemaField[]
  broker_fields: BrokerSchemaField[]
  enums: { clock: EnumOption[]; asset_class: EnumOption[]; source_from: EnumOption[] }
  limits: { min_lot_multiple_tolerance: number; export_format: string; export_version: number }
}

export interface SymbolValidation {
  ok: boolean
  errors: string[]
  sheet: Omit<SymbolView, 'hidden'> | null // la ficha normalizada
}

/** Conflicto entre la ficha existente y lo que propone una hoja de MT5 o un .sqx (enmienda E11). */
export interface SymbolConflict {
  field: string
  actual: unknown // número, o {per_lot_rt|pct_notional_rt: x} en la comisión
  propuesto: unknown
  fuente_actual: SourceKind | null
  fuente_propuesta: 'mt5_sheet' | 'sqx'
  texto: string // frase lista para mostrar (explica qué se puede hacer)
}

/**
 * Borrador plano de una ficha: una fila de la hoja de MT5 (`POST /mt5-sheet`) o lo que propone un `.sqx`
 * (`GET /draft-from-sqx`). No se ha guardado nada: se guarda con `POST ""` por ficha.
 */
export interface SymbolDraft {
  broker: string | null // null en el alta desde un .sqx (va en `faltan`)
  ticker: string | null
  sqx_symbol: string | null
  sqx_symbol_suggested: boolean
  candidates: string[] // sugerencias de símbolo SQX (hoja de MT5)
  asset_class: AssetClass | null
  currency: string | null // moneda de la columna "1 Pip Worth" de la hoja
  point_value_usd: number | null
  lot_step: number | null
  min_lot: number | null
  max_lot: number | null
  commission: Commission // {} si falta
  spread_px: number | null
  ohlc_m1: string | null
  tick_size: number | null
  sources: Record<string, FieldSource>
  needs_review: string[] // puede llevar 'currency' (que NO es un campo de la ficha)
  verified_on: string | null
  notes: string | null
  guardable: boolean
  motivo: string | null
  faltan: string[] // obligatorios sin valor (puede incluir 'broker' y 'sqx_symbol')
  warnings: string[]
  existing: { broker: string; sqx_symbol: string; official: boolean } | null
  editable: boolean // false: la ficha existente es oficial, de otro bróker u oculta
  conflicts: SymbolConflict[]
}

export interface Mt5SheetResult {
  broker: string
  drafts: SymbolDraft[]
}

export interface SymbolExport {
  format: string
  version: number
  broker: string
  sqx_symbol: string
  broker_def: Record<string, unknown> | null // solo en brókers de usuario
  sheet: Record<string, unknown>
}

export interface SymbolImportResult {
  sheet: SymbolView
  broker_created: boolean
  message: string
}

export interface BrokerBody {
  id: string
  display_name: string
  company: string
  currency?: string
  asset_class: AssetClass
  clock: ClockId
  ohlc_dir?: string
}

/** Cuerpo de alta/edición/validación de una ficha (claves del backend; ver `_BODY_KEYS` de symbols.py). */
export type SymbolBody = Record<string, unknown>

// Los errores de /api/symbols llegan como detail:string o detail:{errors:[...]} (400/404/409/422).
export class SymbolApiError extends Error {
  status: number
  errors: string[]
  constructor(status: number, errors: string[]) {
    super(errors.join(' · '))
    this.status = status
    this.errors = errors
  }
}

export function symbolErrText(e: unknown): string {
  return e instanceof SymbolApiError ? e.errors.join(' · ') : String((e as Error)?.message ?? e)
}

async function symbolHandle<T>(res: Response): Promise<T> {
  let body: any = null
  try {
    body = await res.json()
  } catch {
    body = null
  }
  if (!res.ok) {
    const d = body?.detail
    let errs: string[]
    if (typeof d === 'string') errs = [d]
    else if (Array.isArray(d?.errors)) errs = d.errors.map(String)
    else errs = [`HTTP ${res.status}`]
    throw new SymbolApiError(res.status, errs)
  }
  return body as T
}

export async function listSymbols(): Promise<SymbolsList> {
  return symbolHandle(await fetch('/api/symbols'))
}

export async function fetchSymbolSchema(): Promise<SymbolSchema> {
  return symbolHandle(await fetch('/api/symbols/schema'))
}

/** Valida sin guardar. Con `editing: true` el cuerpo es parcial y se fusiona con la ficha de usuario vigente. */
export async function validateSymbol(body: SymbolBody): Promise<SymbolValidation> {
  return symbolHandle(await fetch('/api/symbols/validate', jsonInit('POST', body)))
}

export async function createSymbol(body: SymbolBody): Promise<SymbolView> {
  return symbolHandle(await fetch('/api/symbols', jsonInit('POST', body)))
}

/** PARCIAL: lo no enviado se conserva y `null` borra un campo opcional (así se acepta un conflicto campo a campo). */
export async function updateSymbol(broker: string, sqxSymbol: string, body: SymbolBody): Promise<SymbolView> {
  return symbolHandle(
    await fetch(
      `/api/symbols/${encodeURIComponent(broker)}/${encodeURIComponent(sqxSymbol)}`,
      jsonInit('PUT', body),
    ),
  )
}

export async function deleteSymbol(broker: string, sqxSymbol: string): Promise<{ deleted: string }> {
  return symbolHandle(
    await fetch(`/api/symbols/${encodeURIComponent(broker)}/${encodeURIComponent(sqxSymbol)}`, {
      method: 'DELETE',
    }),
  )
}

/** Oculta (`hidden: true`) o muestra fichas; `keys` = "<bróker>/<símbolo SQX>". Atómico; 404 con claves desconocidas. */
export async function setSymbolsHidden(keys: string[], hidden: boolean): Promise<{ hidden: string[] }> {
  return symbolHandle(await fetch('/api/symbols/visibility', jsonInit('POST', { keys, hidden })))
}

export async function createBroker(body: BrokerBody): Promise<BrokerView> {
  return symbolHandle(await fetch('/api/symbols/brokers', jsonInit('POST', body)))
}

export async function exportSymbol(broker: string, sqxSymbol: string): Promise<SymbolExport> {
  return symbolHandle(
    await fetch(`/api/symbols/${encodeURIComponent(broker)}/${encodeURIComponent(sqxSymbol)}/export`),
  )
}

// Se manda el TEXTO del fichero tal cual: el servidor lo parsea en modo estricto y valida format/version.
export async function importSymbol(rawJson: string): Promise<SymbolImportResult> {
  return symbolHandle(
    await fetch('/api/symbols/import', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: rawJson,
    }),
  )
}

/** Borradores desde el texto de la hoja del script de SQX para MT5. NO guarda nada. */
export async function parseMt5Sheet(
  broker: string,
  text: string,
  mapping?: Record<string, string>,
): Promise<Mt5SheetResult> {
  const body: Record<string, unknown> = { broker, text }
  if (mapping && Object.keys(mapping).length > 0) body.mapping = mapping
  return symbolHandle(await fetch('/api/symbols/mt5-sheet', jsonInit('POST', body)))
}

/** Borrador de una ficha desde un .sqx del catálogo (alta guiada). NO guarda nada; 400 si la ruta no está en el catálogo. */
export async function fetchDraftFromSqx(path: string): Promise<SymbolDraft> {
  return symbolHandle(await fetch(`/api/symbols/draft-from-sqx?path=${encodeURIComponent(path)}`))
}

// --- Presentación de fichas (compartida por la vista Símbolos y el panel de cartera) ---
export const symbolKey = (broker: string, sqxSymbol: string): string => `${broker}/${sqxSymbol}`

/** Etiqueta de procedencia de un campo. "del .sqx (sin verificar)" es la del schema del backend. */
export const SOURCE_LABEL: Record<SourceKind, string> = {
  official: 'oficial',
  mt5_sheet: 'hoja MT5',
  sqx: 'del .sqx (sin verificar)',
  manual: 'manual',
}

export const ASSET_LABEL: Record<AssetClass, string> = { cfd: 'CFD', futures: 'Futuros' }

export const CLOCK_LABEL: Record<ClockId, string> = {
  ny_plus_7: 'Hora del servidor = Nueva York + 7 h',
  ny_wall: 'Reloj de pared de Nueva York',
}

// Mismas etiquetas que `_LABELS` de symbols.py (las usa donde no hay schema a mano: el panel de cartera, el detalle).
const SHEET_FIELD_LABEL: Record<string, string> = {
  broker: 'Bróker',
  sqx_symbol: 'Símbolo SQX',
  ticker: 'Ticker en el bróker',
  point_value_usd: 'Valor del punto',
  lot_step: 'Paso de lote',
  min_lot: 'Lote mínimo',
  max_lot: 'Lote máximo',
  commission: 'Comisión',
  spread_px: 'Spread típico',
  ohlc_m1: 'Fichero OHLC M1',
  tick_size: 'Tamaño del tick',
  currency: 'Moneda del valor del punto',
}

export const fieldLabel = (name: string): string => SHEET_FIELD_LABEL[name] ?? name

const sheetNumFmt = new Intl.NumberFormat('es-ES', { maximumFractionDigits: 10 })

/** Número de una ficha en español ('0,01', '100.000'); '—' si falta. */
export const fmtSheetNum = (v: number | null | undefined): string =>
  v == null || !Number.isFinite(v) ? '—' : sheetNumFmt.format(v)

export function fmtCommission(c: Commission | null | undefined): string {
  if (c?.per_lot_rt != null) return `${fmtSheetNum(c.per_lot_rt)} USD por lote (ida y vuelta)`
  if (c?.pct_notional_rt != null) return `${fmtSheetNum(c.pct_notional_rt)} % del nocional (ida y vuelta)`
  return '—'
}

/** Valor de un campo de ficha o de un conflicto, legible. */
export function fmtFieldValue(field: string, v: unknown): string {
  if (v == null) return '—'
  if (field === 'commission') return fmtCommission(v as Commission)
  if (typeof v === 'number') return fmtSheetNum(v)
  return String(v)
}

// --- entrada-4-bloques F4: receta para configurar el EA (meta.recipe) ------------------------------
// Va en la respuesta de simulate_propfirm y simulate_axi SOLO con carteras .sqx. Lleva TODA la rejilla de f: el
// front compone la de la f elegida sin pedir nada al backend (RecipeCard).
export interface RecipeNote {
  id: string
  texto: string
}

export interface RecipeMm {
  method: string // 'FixedAmount', ...
  params: Record<string, number> // RiskedMoney, Decimals, LotsIfNoMM, MaxLots
}

export interface RecipeParity {
  risked_money: number // X del backtest (RiskedMoney del lastSettings.xml)
  frac_ok: number // fracción de órdenes cuyo lote reproduce la regla de riesgo fijo
  n: number
  n_lifm: number // órdenes que en el backtest fueron a LotsIfNoMM
  frac_ok_sin_lifm: number | null
}

export interface RecipeLeg {
  pata: string
  symbol: string
  broker: string | null
  min_lot: number
  lot_step: number
  decimals: number // Decimals del EA = decimales de lot_step
  mmStep: number
  mmMultiplier: number
  LotsIfNoMM: number // = lote mínimo: la CONDICIÓN para que el EA haga lo simulado (E18)
  max_lot_by_f: number[] // MaxLots por f de la rejilla
  ficha_max_lot: number | null
  max_lot_excede_ficha_by_f: boolean[]
  frac_min_lot_by_f: number[] // fracción (0..1) de órdenes al lote mínimo
  p95_riesgo_real_pct_by_f: number[] // YA en % de la cuenta
  mm_backtest: RecipeMm | null
  paridad: RecipeParity | null
}

export interface Recipe {
  mm: string // 'FixedAmount'
  currency: string // 'USD'
  account_size: number
  f_grid: number[]
  risked_money_by_f: number[] // RiskedMoney = f · account_size
  legs: RecipeLeg[]
  notes: RecipeNote[]
}
