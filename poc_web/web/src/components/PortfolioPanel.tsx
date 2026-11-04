// Banda "Cartera" del shell B: selección de patas + carga/validación única
// (POST /api/portfolio/load) con semáforo de guardas y diagnóstico colapsado.
// Las patas se agrupan por carpeta en grupos PLEGABLES (colapsados por defecto,
// sin preselección): el catálogo puede traer ~80 patas repartidas en muchas
// carpetas y una lista plana entierra todo lo que no sea la primera carpeta.
import { useMemo, useState } from 'react'
import {
  fetchLegDiagnostics,
  fieldLabel,
  loadPortfolio,
  SQX_LEG_KIND,
  type Catalog,
  type GuardStatus,
  type Leg,
  type LegDiagParams,
  type LegDiagResult,
  type PortfolioLoadResult,
} from '../lib/api'
import { Badge, Button, Card, CardBody, CardHeader, Collapsible, Field, NumberInput, PercentInput, SmallBtn } from './ui'
import { GatePanel } from './GatePanel'
import { triggerCompuesto } from '../lib/semaphore'
const pct = (v: number | null | undefined, d = 1) =>
  v == null || !isFinite(v) ? '—' : `${(v * 100).toFixed(d)}%`
const usd2 = (v: number | null | undefined) =>
  v == null || !isFinite(v) ? '—' : `$${v.toFixed(2)}`
const usd0 = (v: number | null | undefined) =>
  v == null || !isFinite(v) ? '—' : `$${Math.round(v).toLocaleString('en-US')}`

const GUARD_LABELS: [string, string][] = [
  ['bloques', 'bloques'],
  ['cobertura', 'cobertura'],
  ['crossing', 'crossing'],
  ['valles', 'valles'],
  ['sqx', '.sqx'],
  ['capital', 'capital'],
  ['m1', 'M1'],
  ['overnight', 'overnight ET'], // Fase 6: solo con perfil de sesiones ET
  // onboarding-b1 (GT-8): guardas que gatean el `ok` de la carga y no tenían etiqueta.
  ['tipo_patas', 'tipo de patas'],
  ['ventanas', 'ventanas'],
  ['corte_1610', 'corte 16:10'],
  ['flotante_m1', 'flotante M1'],
  // entrada-4-bloques F4/F5: fichas de símbolo con campos sin verificar (solo carteras .sqx; nunca error)
  ['fichas', 'fichas'],
]

// entrada-4-bloques F5: una pata .sqx se puede marcar solo si su símbolo tiene ficha usable. Las CSV no traen `ficha`.
const legBlocked = (l: Leg): boolean => l.kind === SQX_LEG_KIND && !!l.ficha && l.ficha !== 'ok'

// Fila de una pata del selector. Pata .sqx: chip con el bróker de su ficha; si no hay ficha usable, badge «sin ficha» /
// «no usable» (con el motivo en el title), casilla DESHABILITADA y botón «Dar de alta» (abre la vista Símbolos con la ficha
// rellenada desde el .sqx). No se usa el Checkbox de ui.tsx porque no admite `disabled` (ui.tsx no se toca).
function LegRow({
  leg,
  checked,
  onChange,
  onRegister,
}: {
  leg: Leg
  checked: boolean
  onChange: (on: boolean) => void
  onRegister?: (path: string) => void
}) {
  const blocked = legBlocked(leg)
  const noUsable = leg.ficha === 'no_usable'
  // Lectura literal del brief: «Dar de alta» sale con `ficha != "ok"`, también en «no usable» (la ficha existe pero no sirve, p. ej. divisa
  // no USD). En ese caso, y si la ficha está oculta, el borrador sale NO guardable y la vista Símbolos explica por qué.
  const why =
    leg.motivo ??
    (noUsable
      ? 'La ficha de este símbolo no se puede usar.'
      : 'Este símbolo no tiene ficha: dala de alta para poder usar la pata.')
  return (
    <div className="flex items-center gap-1.5 rounded-md px-1.5 py-1 hover:bg-zinc-800/60">
      <label
        className={`flex min-w-0 flex-1 items-center gap-2 text-sm ${
          blocked ? 'cursor-not-allowed text-zinc-500' : 'cursor-pointer text-zinc-300'
        }`}
      >
        <input
          type="checkbox"
          className="size-3.5 accent-emerald-600 disabled:cursor-not-allowed disabled:opacity-50"
          checked={checked && !blocked}
          disabled={blocked}
          onChange={(e) => onChange(e.target.checked)}
        />
        <span className="truncate" title={leg.name}>
          {leg.name}
        </span>
      </label>
      {leg.kind === SQX_LEG_KIND && leg.broker && (
        <span title="Bróker de la ficha de este símbolo" className="shrink-0">
          <Badge variant="outline">{leg.broker}</Badge>
        </span>
      )}
      {blocked && (
        <span title={why} className="shrink-0">
          <Badge variant="warn">{noUsable ? 'no usable' : 'sin ficha'}</Badge>
        </span>
      )}
      {blocked && onRegister && (
        <SmallBtn tone="primary" onClick={() => onRegister(leg.path)} title="Abre la vista Símbolos con la ficha rellenada desde este .sqx">
          Dar de alta
        </SmallBtn>
      )}
    </div>
  )
}

const SEMAPHORE: Record<GuardStatus, { icon: string; cls: string }> = {
  ok: { icon: '✓', cls: 'text-emerald-400' },
  warn: { icon: '⚠', cls: 'text-amber-400' },
  error: { icon: '✗', cls: 'text-rose-400' },
}

// onboarding-b1 (enmienda 7): demo del repo exportado. Lleva las patas, el perfil
// y el capital EXPLÍCITOS: la carga no depende del estado que App acaba de poner.
export interface DemoSpec {
  folder: string
  label: string
  firmKey: string
  dayAxis?: 'ftmo' | 'servidor'
  legs: string[]
  accountSize: number
}

interface Props {
  catalog: Catalog
  selLegs: Set<string>
  onToggleLeg: (path: string, on: boolean) => void
  accountSize: number
  // Override manual del capital de replay (null = derivado de etapa/firm).
  onAccountSize: (v: number) => void
  portfolio: PortfolioLoadResult | null
  stale: boolean
  onLoaded: (p: PortfolioLoadResult, fRef: number) => void
  // Parámetros de la última simulación AXI (para "P sola @ f*"); null si no hay.
  axiDiag: LegDiagParams | null
  // Fase 6 (Apex): perfil de firm para la guarda overnight de la carga ('' =
  // carga agnóstica) y tz de los datos (null = default del backend).
  firmKey?: string
  sourceTz?: string | null
  // Demos presentes en el catálogo (vacío = sin botones) y callback que prepara
  // el estado de App (patas, lente, perfil, eje) y limpia los resultados.
  demos?: DemoSpec[]
  demoBusy?: boolean // hay una simulación en curso
  onDemoSelect?: (d: DemoSpec) => void
  // entrada-4-bloques F5: «Dar de alta» en una pata .sqx sin ficha -> App abre la vista Símbolos con el borrador del .sqx.
  onRegisterLeg?: (path: string) => void
}

export function PortfolioPanel({ catalog, selLegs, onToggleLeg, accountSize, onAccountSize, portfolio, stale, onLoaded, axiDiag, firmKey = '', sourceTz = null, demos = [], demoBusy = false, onDemoSelect, onRegisterLeg }: Props) {
  // Default 2%: con la cartera Seed actual f_min por pata ≈1.0-1.7%, y bajo
  // eso el grid es zona muerta (debate f-grid clamp-aware, 01-oct-2026).
  const [fRef, setFRef] = useState(0.02)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [diag, setDiag] = useState<LegDiagResult | null>(null)
  const [diagBusy, setDiagBusy] = useState(false)
  const [diagError, setDiagError] = useState<string | null>(null)
  // D2 — "f operada": null = usar la f de la última simulación AXI (f*).
  // Sin useEffect de sincronización (evita el warning set-state-in-effect).
  const [fOverride, setFOverride] = useState<number | null>(null)
  // Carpetas expandidas del selector (controlado: arranca todo colapsado).
  const [openFolders, setOpenFolders] = useState<Set<string>>(new Set())

  const legsByFolder = useMemo(() => {
    const map = new Map<string, Catalog['legs']>()
    for (const l of catalog.legs) {
      if (!map.has(l.folder)) map.set(l.folder, [])
      map.get(l.folder)!.push(l)
    }
    return map
  }, [catalog])

  const toggleFolder = (folder: string) =>
    setOpenFolders((prev) => {
      const next = new Set(prev)
      if (next.has(folder)) next.delete(folder)
      else next.add(folder)
      return next
    })

  // Todas/ninguna sobre una carpeta entera. «Todas» salta las patas .sqx sin ficha usable (no se pueden marcar).
  const setFolderLegs = (legs: Catalog['legs'], on: boolean) => {
    for (const l of legs) if (!on || !legBlocked(l)) onToggleLeg(l.path, on)
  }

  // `explicit` (demos): patas/capital/perfil explícitos, sin leer el estado
  // recién modificado por el padre.
  const load = async (explicit?: { legs: string[]; accountSize: number; firmKey: string }) => {
    setLoading(true)
    setError(null)
    try {
      const res = await loadPortfolio({
        legs: explicit?.legs ?? [...selLegs],
        account_size: explicit?.accountSize ?? accountSize,
        f_ref: fRef,
        firm_key: explicit?.firmKey ?? firmKey,
        source_tz: explicit ? null : sourceTz,
      })
      setDiag(null) // diagnósticos de una cartera anterior ya no aplican
      onLoaded(res, fRef)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  const loadDemo = (d: DemoSpec) => {
    onDemoSelect?.(d)
    setOpenFolders((prev) => new Set(prev).add(d.folder))
    setDiag(null)
    void load({ legs: d.legs, accountSize: d.accountSize, firmKey: d.firmKey })
  }

  const runDiag = async () => {
    if (!portfolio || !axiDiag) return
    setDiagBusy(true)
    setDiagError(null)
    try {
      // D2: todo el diagnóstico (incluida la P sola) va a la f operada.
      setDiag(await fetchLegDiagnostics(portfolio.portfolio_id,
        { ...axiDiag, f: fOverride ?? axiDiag.f }))
    } catch (e) {
      setDiagError(String(e))
    } finally {
      setDiagBusy(false)
    }
  }

  const okLegs = portfolio?.legs.filter((l) => l.status === 'ok') ?? []
  // F5: columnas extra del detalle SOLO si hay filas de patas .sqx cargadas (las CSV y las filas con error no traen `broker`)
  const sqxRows = !!portfolio?.legs.some((l) => l.broker !== undefined)

  return (
    <Card>
      <CardHeader
        title="Cartera"
        subtitle="Patas cargadas y validadas una vez; las guardas viven aquí"
        right={
          portfolio && (
            <Badge variant={portfolio.ok && !stale ? 'success' : portfolio.ok ? 'warn' : 'danger'}>
              {stale ? 'desactualizada — recarga' : portfolio.ok ? 'validada' : 'con errores'}
            </Badge>
          )
        }
      />
      <CardBody className="space-y-4">
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1fr_280px]">
          <div>
            <span className="mb-1 block text-xs font-medium text-zinc-400">
              Patas ({selLegs.size} seleccionadas)
            </span>
            <div className="max-h-72 space-y-0.5 overflow-y-auto rounded-md border border-zinc-800 bg-zinc-950/60 p-1">
              {[...legsByFolder.entries()].map(([folder, legs]) => {
                const open = openFolders.has(folder)
                const nSel = legs.filter((l) => selLegs.has(l.path)).length
                return (
                  <div key={folder}>
                    <div className="flex items-center gap-2 px-1.5 py-1">
                      <button
                        onClick={() => toggleFolder(folder)}
                        className="flex flex-1 cursor-pointer items-center gap-1.5 text-left"
                      >
                        <span
                          className={`text-[10px] text-zinc-500 transition-transform ${open ? 'rotate-90' : ''}`}
                        >
                          ▸
                        </span>
                        <span className="text-[10px] font-semibold uppercase tracking-wider text-zinc-600">
                          {folder}
                        </span>
                        <span className="num text-[10px] text-zinc-600">
                          {legs.length}
                          {nSel > 0 ? ` · ${nSel} sel` : ''}
                        </span>
                      </button>
                      <div className="flex gap-2 text-[10px] text-zinc-500">
                        <button
                          className="cursor-pointer hover:text-emerald-400"
                          onClick={() => setFolderLegs(legs, true)}
                        >
                          todas
                        </button>
                        <button
                          className="cursor-pointer hover:text-rose-400"
                          onClick={() => setFolderLegs(legs, false)}
                        >
                          ninguna
                        </button>
                      </div>
                    </div>
                    {open && (
                      <div className="mb-1 ml-3 space-y-0.5 border-l border-zinc-800 pl-2">
                        {legs.map((l) => (
                          <LegRow
                            key={l.path}
                            leg={l}
                            checked={selLegs.has(l.path)}
                            onChange={(on) => onToggleLeg(l.path, on)}
                            onRegister={onRegisterLeg}
                          />
                        ))}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </div>
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-2">
              <Field label="Tamaño cuenta ($)" hint="Capital sobre el que se replayea el sizing: la firm en prop firm, la allocation de la etapa en AXI. Editable: si lo cambias a mano, el override vive hasta que cambies de etapa/firm.">
                <NumberInput value={accountSize} step={500} min={100} onChange={onAccountSize} />
              </Field>
              <Field label="f_ref sizing (%)" hint="f del backtest para patas sin .sqx (modo rápido)">
                <PercentInput value={fRef} step={0.25} min={0.05} onChange={setFRef} />
              </Field>
            </div>
            <Button onClick={() => load()} disabled={loading || selLegs.size === 0}>
              {loading ? 'Validando…' : 'Cargar/validar cartera'}
            </Button>
            {demos.length > 0 && (
              <div className="space-y-1.5">
                <span className="block text-[11px] font-medium uppercase tracking-wide text-zinc-500">
                  Cargar demo
                </span>
                {demos.map((d) => (
                  <button
                    key={d.folder}
                    onClick={() => loadDemo(d)}
                    disabled={loading || demoBusy}
                    className="w-full rounded-md border border-zinc-700 px-3 py-1.5 text-left text-xs font-medium text-zinc-300 transition-colors hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    {d.label} · {d.legs.length} patas
                  </button>
                ))}
              </div>
            )}
            {error && (
              <p className="rounded-md border border-rose-900 bg-rose-950/60 px-2 py-1.5 text-xs text-rose-300">
                {error}
              </p>
            )}
          </div>
        </div>

        {portfolio && (
          <>
            {/* Semáforo de guardas (una línea) */}
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs">
              {GUARD_LABELS.map(([key, label]) => {
                const g = portfolio.guards[key]
                if (!g) return null
                const s = SEMAPHORE[g.status]
                return (
                  <span key={key} className="flex items-center gap-1" title={g.detail}>
                    <span className={s.cls}>{s.icon}</span>
                    <span className="text-zinc-400">{label}</span>
                  </span>
                )
              })}
              <span className="num ml-auto text-zinc-500">
                {portfolio.n_blocks} bloques · cobertura {pct(portfolio.coverage)} ·{' '}
                {okLegs.length} patas
              </span>
            </div>

            {portfolio.warnings.length > 0 && (
              <div className="space-y-1 rounded-md border border-amber-900/60 bg-amber-950/20 px-3 py-2">
                {portfolio.warnings.map((w, i) => (
                  <p key={i} className="text-xs text-amber-300">⚠ {w}</p>
                ))}
              </div>
            )}

            <Collapsible
              title="Detalle por pata"
              subtitle={sqxRows ? 'símbolo · bróker · modo · MM del backtest · paridad · trades · avisos' : 'símbolo · modo · trades · avisos'}
            >
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-zinc-800 text-left text-[11px] uppercase tracking-wide text-zinc-500">
                    <th className="px-2 py-1.5 font-medium">pata</th>
                    <th className="px-2 py-1.5 font-medium">símbolo</th>
                    {sqxRows && <th className="px-2 py-1.5 font-medium">bróker</th>}
                    <th className="px-2 py-1.5 font-medium">modo</th>
                    {sqxRows && <th className="px-2 py-1.5 font-medium">MM del backtest</th>}
                    {sqxRows && (
                      <th
                        className="px-2 py-1.5 text-right font-medium"
                        title="Fracción de órdenes del backtest cuyo lote coincide con el riesgo fijo de la app (solo si el MM del backtest es FixedAmount)"
                      >
                        paridad
                      </th>
                    )}
                    <th className="px-2 py-1.5 text-right font-medium">trades</th>
                    <th className="px-2 py-1.5 font-medium">avisos</th>
                  </tr>
                </thead>
                <tbody>
                  {portfolio.legs.map((l) => (
                    <tr key={l.pata} className="border-b border-zinc-800/60">
                      <td className="px-2 py-1.5 text-zinc-300">{l.pata}</td>
                      <td className="px-2 py-1.5 text-zinc-400">{l.símbolo ?? '—'}</td>
                      {sqxRows && (
                        <td className="px-2 py-1.5">
                          {l.broker ? <Badge variant="outline">{l.broker}</Badge> : <span className="text-zinc-600">—</span>}
                        </td>
                      )}
                      <td className="px-2 py-1.5">
                        {l.status === 'error' ? (
                          <Badge variant="danger">error</Badge>
                        ) : (
                          <Badge variant="outline">{l.modo ?? '—'}</Badge>
                        )}
                      </td>
                      {sqxRows && (
                        <td className="px-2 py-1.5 text-zinc-400">{l.mm_backtest ?? <span className="text-zinc-600">—</span>}</td>
                      )}
                      {sqxRows && (
                        <td className="num px-2 py-1.5 text-right">
                          {l.paridad_frac_ok == null ? <span className="text-zinc-600">—</span> : pct(l.paridad_frac_ok, 1)}
                        </td>
                      )}
                      <td className="num px-2 py-1.5 text-right">{l.trades ?? '—'}</td>
                      <td className="px-2 py-1.5 text-xs">
                        {l.error && <p className="text-rose-400">✗ {l.error}</p>}
                        {l.warnings.map((w, i) => (
                          <p key={i} className="text-amber-300">⚠ {w}</p>
                        ))}
                        {(l.ficha_sin_verificar?.length ?? 0) > 0 && (
                          <p className="text-amber-300" title={l.ficha_sin_verificar?.join(', ')}>
                            ⚠ Ficha con campos sin verificar: {l.ficha_sin_verificar?.map(fieldLabel).join(', ')}
                          </p>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Collapsible>
          </>
        )}

        {!!portfolio?.ok && !stale && (
          <Collapsible
            title="Diagnóstico por pata"
            subtitle="L2 · suelo min_lot · riesgo indeleble · coste · P sola @ f* (bajo demanda)"
          >
            <div className="space-y-3">
              <div className="flex flex-wrap items-center gap-3">
                <Button
                  onClick={runDiag}
                  disabled={!axiDiag || diagBusy}
                  className="w-auto px-3 py-1.5 text-xs"
                >
                  {diagBusy
                    ? 'Calculando MC por pata…'
                    : axiDiag
                      ? `Calcular @ f = ${pct(fOverride ?? axiDiag.f, 2)}`
                      : 'Lanza una simulación AXI para habilitar'}
                </Button>
                {axiDiag && (
                  <>
                    <Field label="f operada">
                      <PercentInput
                        value={fOverride ?? axiDiag.f}
                        step={0.05}
                        min={0.05}
                        onChange={setFOverride}
                      />
                    </Field>
                    {fOverride !== null && (
                      <button
                        onClick={() => setFOverride(null)}
                        className="cursor-pointer text-[11px] text-zinc-500 hover:text-emerald-400"
                      >
                        usar f* ({pct(axiDiag.f, 2)})
                      </button>
                    )}
                    <span className="text-xs text-zinc-500">
                      etapa {axiDiag.stage} · {axiDiag.n_paths} paths · bs {Number(axiDiag.block_size).toFixed(1)}
                    </span>
                  </>
                )}
              </div>
              {diagError && (
                <p className="rounded-md border border-rose-900 bg-rose-950/60 px-2 py-1.5 text-xs text-rose-300">
                  {diagError}
                </p>
              )}
              {diag && (
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-zinc-800 text-left text-[11px] uppercase tracking-wide text-zinc-500">
                      <th className="px-2 py-1.5 font-medium">pata</th>
                      <th className="px-2 py-1.5 font-medium">símbolo</th>
                      <th className="px-2 py-1.5 font-medium">modo</th>
                      <th className="px-2 py-1.5 text-right font-medium">trades</th>
                      <th className="px-2 py-1.5 text-right font-medium">% min_lot ref</th>
                      <th className="px-2 py-1.5 text-right font-medium">riesgo suelo % †</th>
                      <th className="px-2 py-1.5 text-right font-medium">coste impl/esper $/lote</th>
                      <th className="px-2 py-1.5 text-right font-medium">P(pass) solo</th>
                      <th className="px-2 py-1.5 text-right font-medium">P(ruin) solo</th>
                    </tr>
                  </thead>
                  <tbody>
                    {diag.rows.map((r) => (
                      <tr key={r.pata} className="border-b border-zinc-800/60 align-top">
                        <td className="px-2 py-1.5 text-zinc-300">
                          {r.pata}
                          {r.warnings.map((w, i) => (
                            <p key={i} className="text-[11px] text-amber-300">⚠ {w}</p>
                          ))}
                          {r.error && <p className="text-[11px] text-rose-400">✗ {r.error}</p>}
                        </td>
                        <td className="px-2 py-1.5 text-zinc-400">{r.símbolo}</td>
                        <td className="px-2 py-1.5"><Badge variant="outline">{r.modo}</Badge></td>
                        <td className="num px-2 py-1.5 text-right">{r.trades}</td>
                        <td className="num px-2 py-1.5 text-right">{pct(r.frac_min_lot_ref, 0)}</td>
                        <td className="num px-2 py-1.5 text-right">
                          {r.modo === 'exacto' ? (
                            <span
                              className="text-zinc-500"
                              title="pata exacta: usa 'suelo %E' en Canal de dinero y régimen (el back-derivado infravalora ×f_ref/risk)"
                            >
                              —
                            </span>
                          ) : (
                            r.floor_risk_pct == null || !isFinite(r.floor_risk_pct) ? '—' : r.floor_risk_pct.toFixed(2)
                          )}
                        </td>
                        <td className="num px-2 py-1.5 text-right">
                          {usd2(r.implied_cost_per_lot)} / {usd2(r.expected_cost_per_lot)}
                        </td>
                        <td className="num px-2 py-1.5 text-right text-emerald-400">{pct(r.p_pass_solo)}</td>
                        <td className="num px-2 py-1.5 text-right text-rose-400">{pct(r.p_ruin_solo)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
              {diag && diag.threshold_pctile === undefined && (
                <p className="rounded-md border border-amber-900/60 bg-amber-950/20 px-3 py-2 text-xs text-amber-300">
                  ⚠ el servidor no devuelve los campos de régimen — reinicia uvicorn
                </p>
              )}
              {diag && diag.threshold_pctile !== undefined && (
                <div className="space-y-1.5">
                  <p className="text-xs font-medium text-zinc-400">
                    Canal de dinero y régimen @ f = {pct(diag.f, 2)} · E = {usd0(diag.account_size)}
                  </p>
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-zinc-800 text-left text-[11px] uppercase tracking-wide text-zinc-500">
                        <th className="px-2 py-1.5 font-medium">pata</th>
                        <th className="px-2 py-1.5 text-right font-medium">clamp %</th>
                        <th className="px-2 py-1.5 text-right font-medium">suelo %E med / P99</th>
                        <th className="px-2 py-1.5 text-right font-medium">E* med / P99</th>
                        <th className="px-2 py-1.5 text-right font-medium">vol (as-of feed)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {diag.rows.map((r) => {
                        const trig = triggerCompuesto(
                          r.pctile_sl_live, r.would_be_clamp_now,
                          diag.threshold_pctile ?? 90)
                        return (
                          <tr key={r.pata} className="border-b border-zinc-800/60 align-top">
                            <td className="px-2 py-1.5 text-zinc-300">
                              {r.pata}
                              {(r.regimen_warns ?? []).map((w, i) => (
                                <p key={i} className="text-[11px] text-amber-300">⚠ {w}</p>
                              ))}
                            </td>
                            <td className="num px-2 py-1.5 text-right">{pct(r.clamp_rate, 0)}</td>
                            <td className="num px-2 py-1.5 text-right">
                              {r.floor_risk_med_pct_e == null || r.floor_risk_p99_pct_e == null
                                ? '—'
                                : `${r.floor_risk_med_pct_e.toFixed(2)}% / ${r.floor_risk_p99_pct_e.toFixed(2)}%`}
                            </td>
                            <td className="num px-2 py-1.5 text-right">
                              {r.e_star_med == null || r.e_star_p99 == null
                                ? '—'
                                : `${usd0(r.e_star_med)} / ${usd0(r.e_star_p99)}`}
                            </td>
                            <td
                              className="num px-2 py-1.5 text-right"
                              title={`sl vivo ${usd2(r.sl_live_per_lot)}/lote · as-of ${r.sl_live_asof ?? '—'} · ${diag.reference_population ?? ''}`}
                            >
                              {r.pctile_sl_live == null ? (
                                '—'
                              ) : (
                                <span className="inline-flex items-center justify-end gap-1.5 text-zinc-300">
                                  P{Math.round(r.pctile_sl_live)}
                                  {trig === true && (
                                    <Badge variant="outline">
                                      P{Math.round(r.pctile_sl_live)} · en clamp
                                    </Badge>
                                  )}
                                </span>
                              )}
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                  <div className="space-y-1 text-[11px] text-zinc-600">
                    <p>umbral P{diag.threshold_pctile} — {diag.threshold_source}</p>
                    <p>
                      informativo hasta medir su tasa de falsas señales o tener un
                      backtest (§9.3); no dispara nada
                    </p>
                    <p>{diag.hybrid_label}</p>
                    <p>población: {diag.reference_population}</p>
                    <p>
                      volatilidad as-of el final del feed M1 (
                      {diag.rows.find((r) => r.sl_live_asof)?.sl_live_asof ?? '—'});
                      actualiza el feed para una lectura actual
                    </p>
                    <p>
                      el clamp no es gratis: el riesgo pagado es min_lot×sl aunque f
                      baje
                    </p>
                  </div>
                </div>
              )}
              <p className="text-[11px] text-zinc-600">
                "riesgo suelo" = riesgo indeleble por trade aunque f → 0 (clamp de min_lot;
                cota inferior si el backtest ya estaba clamped). P sola = cada pata
                enfrentada sola a las barreras de la etapa @ f = {pct(diag?.f, 2)}.
                Convención del MC por pata: {diag?.float_convention ?? 'bound (referencia motor)'} — no
                comparable con el lente AXI (joint).
              </p>
              <p className="text-[11px] text-zinc-600">
                † back-derivado del modo rápido (cota inferior); en patas exactas ver la
                tabla de régimen.
              </p>
            </div>
          </Collapsible>
        )}

        {!!portfolio?.ok && !stale && (
          <GatePanel portfolio={portfolio} diag={diag} />
        )}
      </CardBody>
    </Card>
  )
}
