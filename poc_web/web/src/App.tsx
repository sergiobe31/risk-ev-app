// Shell B (DESIGN_UI_V2.md §2.2): la CARTERA es la entidad central, con dos
// LENTES encima: Prop firm (F1) y AXI (F2). Tres bandas apiladas:
// Cartera → Pregunta → Resultados. El motor no cambia: consume la API
// (POST /api/portfolio/load + POST /api/simulate/{propfirm,axi};
// el modo de flotante lo fija la API por tipo de pata y perfil: meta.float_mode de cada corrida).
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  fetchCatalog,
  postRun,
  runAxi,
  runPropfirm,
  SQX_LEG_KIND,
  type AxiSimParams,
  type AxiSimResult,
  type Catalog,
  type Firm,
  type PortfolioLoadResult,
  type PropfirmSimParams,
  type SimResult,
} from './lib/api'
import {
  Button,
  Card,
  CardBody,
  CardHeader,
  Checkbox,
  Collapsible,
  Field,
  NumberInput,
  PercentInput,
  Select,
  Slider,
  TextInput,
} from './components/ui'
import { PortfolioPanel, type DemoSpec } from './components/PortfolioPanel'
import { LensResultsPropfirm } from './components/LensResultsPropfirm'
import { LensResultsAxi } from './components/LensResultsAxi'
import { HistoryPanel } from './components/HistoryPanel'
import { ProfilesView } from './components/ProfilesView'
import { SymbolsView, type DraftRequest } from './components/SymbolsView'

// onboarding-b1 (GT-12): demos del repo exportado -> perfil que les corresponde.
// El botón solo aparece si el catálogo trae patas en esa carpeta.
const DEMO_DEFS: { folder: string; firmKey: string; label: string; dayAxis?: 'ftmo' }[] = [
  { folder: 'demo_cfd', firmKey: 'ftmo_50k', label: 'Demo CFD (FTMO 50k)', dayAxis: 'ftmo' },
  {
    folder: 'demo_futuros',
    firmKey: 'ftmo_futures_growth_50k',
    label: 'Demo futuros (FTMO Futures Growth 50k)',
  },
]

// Hint del eje de días en perfiles de sesiones ET: "{cut} -> {cut - 1 min}".
function minusOneMinute(hhmm: string): string {
  const [h, m] = hhmm.split(':').map(Number)
  if (!Number.isFinite(h) || !Number.isFinite(m)) return hhmm
  const t = (h * 60 + m - 1 + 1440) % 1440
  return `${String(Math.floor(t / 60)).padStart(2, '0')}:${String(t % 60).padStart(2, '0')}`
}

type Lens = 'propfirm' | 'axi'
// onboarding-c: vista Perfiles (editor de perfiles de firm); entrada-4-bloques F5: vista Símbolos (fichas por bróker)
type View = 'simulador' | 'perfiles' | 'simbolos'

// onboarding-c: los perfiles del usuario llevan el sufijo "(tuyo)" en el selector.
// El API manda los booleanos como 0/1 → !!x, nunca === true.
const firmName = (f: Firm) =>
  f.official != null && !f.official && !/\(tuyo\)\s*$/.test(f.display_name) ? `${f.display_name} (tuyo)` : f.display_name
// perfiles-v2: selector en dos niveles Firma (compañía) -> Reto. Los ocultos se filtran aquí
// (el catálogo los trae marcados `hidden`, 0/1 -> !!x).
const NO_COMPANY = 'Sin compañía'
const companyOf = (f: Firm) => f.company?.trim() || NO_COMPANY
const kindOf = (f: Firm): 'cfd' | 'futures' =>
  f.kind ?? (f.funded_model === 'cycle' ? 'cfd' : 'futures')
const firstVisibleKey = (c: Catalog) => c.firms.find((f) => !f.hidden)?.key ?? c.firms[0]?.key ?? ''
type Result =
  | { lens: 'propfirm'; data: SimResult; params: PropfirmSimParams }
  | { lens: 'axi'; data: AxiSimResult; params: AxiSimParams }

export default function App() {
  const [catalog, setCatalog] = useState<Catalog | null>(null)
  const [catalogError, setCatalogError] = useState<string | null>(null)
  // onboarding-b1 (GT-11): en frío el catálogo tarda ~1 min -> estado de carga + reintento.
  const [catalogLoading, setCatalogLoading] = useState(true)
  const [catalogReloading, setCatalogReloading] = useState(false) // recarga tras cambiar perfiles
  // onboarding-c: vista activa
  const [view, setView] = useState<View>('simulador')
  // F5: «Dar de alta» en una pata .sqx sin ficha -> la vista Símbolos lee el borrador de esa ruta y abre el formulario
  const [draftRequest, setDraftRequest] = useState<DraftRequest | null>(null)
  // F5 (H10): la vista Símbolos está montada siempre pero no pide sus datos hasta que se abre por primera vez
  const [symbolsOpened, setSymbolsOpened] = useState(false)

  // --- Banda Cartera --------------------------------------------------------
  const [selLegs, setSelLegs] = useState<Set<string>>(new Set())
  const selLegsRef = useRef(selLegs) // la selección vigente, para avisar de lo que quita la recarga del catálogo (H11)
  useEffect(() => {
    selLegsRef.current = selLegs
  }, [selLegs])
  const [legsNotice, setLegsNotice] = useState<string | null>(null) // F5 (H11): patas quitadas de la selección por su ficha
  const [portfolio, setPortfolio] = useState<PortfolioLoadResult | null>(null)
  const [stale, setStale] = useState(false)

  // --- Banda Pregunta ---------------------------------------------------------
  const [lens, setLens] = useState<Lens>('propfirm')
  // Contexto
  const [firmKey, setFirmKey] = useState('')
  const [stageKey, setStageKey] = useState('seed')
  // Parámetros compartidos (grilla f + paths)
  const [fLo, setFLo] = useState(0.0025)
  const [fHi, setFHi] = useState(0.0175)
  const [fStep, setFStep] = useState(0.0025)
  const [nPaths, setNPaths] = useState(800)
  // Prop firm
  const [nChallenges, setNChallenges] = useState(20)
  const [nSequences, setNSequences] = useState(1000)
  const [capital0, setCapital0] = useState(3000)
  const [fx, setFx] = useState(1.08)
  const [tolPct, setTolPct] = useState(50)
  const [stopRuin, setStopRuin] = useState(false)
  // AXI
  const [tolRuinPct, setTolRuinPct] = useState(10)
  const [bsAxi, setBsAxi] = useState(0) // 0 = automático (Politis-White)
  const [horizonOn, setHorizonOn] = useState(false)
  const [horizonDays, setHorizonDays] = useState(90)
  // F2c: corte IS/OOS (regla M2-8) + eje de días del daily-loss reset.
  // oosSel='' = sin corte (ventana completa, inflada si hubo selección).
  const [oosSel, setOosSel] = useState('')
  const [oosDate, setOosDate] = useState('') // fecha editable, prefill del registro
  const [dayAxis, setDayAxis] = useState<'ftmo' | 'servidor'>('ftmo')
  // Fase 6 (Apex): tz de los datos para sesiones ET ('' = default del backend,
  // documentado en meta.source_tz), copy trading y umbral del diagnóstico de caps.
  const [sourceTz, setSourceTz] = useState('')
  const [nAccts, setNAccts] = useState(1)
  const [capTolPct, setCapTolPct] = useState(0)
  // Fase 2 (FTMO Futures): políticas de la campaña de suscripción (solo
  // aplican con funded.model == 'futures'; el backend las ignora en el resto).
  const [resetPolicy, setResetPolicy] = useState<'billing_retry' | 'paid_reset'>(
    'billing_retry',
  )
  const [wdPolicy, setWdPolicy] = useState<'greedy' | 'buffer_preserving'>(
    'greedy',
  )
  const [resultsStale, setResultsStale] = useState(false)
  // P2: diagnóstico de selección — corrida gemela sin corte, opt-in por corrida.
  const [diagOn, setDiagOn] = useState(false)
  const [diagFull, setDiagFull] = useState<Result | null>(null)

  // --- Banda Resultados -------------------------------------------------------
  const [result, setResult] = useState<Result | null>(null)
  const [simLoading, setSimLoading] = useState(false)
  const [simError, setSimError] = useState<string | null>(null)
  const [historyKey, setHistoryKey] = useState(0) // refresca el historial (P4)

  const loadCatalog = useCallback(() => {
    fetchCatalog()
      .then((c) => {
        setCatalog(c)
        setFirmKey(firstVisibleKey(c))
        if (c.axi_stages.length) setStageKey(c.axi_stages[0].key)
        // Sin preselección de patas: se eligen carpeta a carpeta (grupos plegables).
      })
      .catch((e) => setCatalogError(String(e)))
      .finally(() => setCatalogLoading(false))
  }, [])
  useEffect(() => {
    loadCatalog()
  }, [loadCatalog])
  const retryCatalog = () => {
    setCatalogError(null)
    setCatalogLoading(true)
    loadCatalog()
  }

  // onboarding-c: tras cambiar perfiles se vuelve a pedir el catálogo; si el
  // firmKey seleccionado ya no existe (borrado) se selecciona el primero
  // (enmienda 13). Las reglas del perfil pudieron cambiar → cartera/resultados
  // desactualizados.
  // perfiles-v2: la recarga es INMEDIATA tras cada cambio (la pestaña Perfiles usa las `rules`
  // del catálogo); si el firmKey desaparece o se oculta se elige el primero visible.
  const reloadCatalog = (): Promise<void> => {
    setCatalogReloading(true) // demos desactivadas mientras recarga el catálogo
    return fetchCatalog()
      .then((c) => {
        setCatalog(c)
        setCatalogError(null)
        setFirmKey((prev) =>
          c.firms.some((f) => f.key === prev && !f.hidden) ? prev : firstVisibleKey(c),
        )
        // PLAN-DESVIO (adición): el brief no dice qué pasa con una pata ya marcada cuando su ficha deja de ser usable (se oculta o se
        // borra desde la vista Símbolos): su casilla queda deshabilitada y no se podría desmarcar, y la carga daría «sin ficha». Se la
        // quita de la selección al recargar el catálogo y se avisa de cuáles (revisión de código de la F5, H11). Sin patas que quitar
        // no se toca nada (no hay re-render).
        const blockedLegs = new Map(
          c.legs.filter((l) => l.kind === SQX_LEG_KIND && l.ficha && l.ficha !== 'ok').map((l) => [l.path, l.name] as const),
        )
        const dropped = [...selLegsRef.current].filter((p) => blockedLegs.has(p))
        if (dropped.length > 0) {
          setSelLegs((prev) => new Set([...prev].filter((p) => !blockedLegs.has(p))))
          const names = dropped.map((p) => blockedLegs.get(p)).join(', ')
          setLegsNotice(
            dropped.length === 1
              ? `He quitado 1 pata cuya ficha ya no es usable: ${names}. Revisa su ficha en la pestaña Símbolos.`
              : `He quitado ${dropped.length} patas cuyas fichas ya no son usables: ${names}. Revisa sus fichas en la pestaña Símbolos.`,
          )
        }
        if (portfolio) setStale(true)
        setResultsStale(true)
      })
      .catch((e) => setCatalogError(String(e)))
      .finally(() => setCatalogReloading(false))
  }
  const selectView = (v: View) => {
    setView(v)
    if (v === 'simbolos') setSymbolsOpened(true)
  }
  const registerLeg = (path: string) => {
    setSymbolsOpened(true)
    setDraftRequest({ path, nonce: Date.now() })
    setView('simbolos')
  }

  const firm = useMemo(
    () => catalog?.firms.find((f) => f.key === firmKey),
    [catalog, firmKey],
  )
  // Perfil extendido (Apex): reglas trailing/sesiones ET → el eje ftmo/servidor
  // no aplica (lo fija el day_labeler) y se exponen multi-cuenta/caps/source_tz.
  // onboarding-b1: criterio explícito del API (`extended`), ya no la presencia de
  // `rules` (que ahora existe también en FTMO CFD).
  const isExtendedProfile = !!firm?.extended
  const extendedRules = firm?.rules && firm.rules.model !== 'cycle' ? firm.rules : null
  // Perfil FTMO Futures: economía de suscripción por campaña (fase 2) — sin
  // fee de challenge, sin multi-cuenta y con selectores de políticas.
  const isFutures = firm?.funded_model === 'futures'
  // perfiles-v2 (R1): Firma = compañías con algún perfil visible (alfabético); Reto = sus
  // perfiles visibles (oficiales primero, luego los tuyos). `firmKey` sigue siendo la única
  // fuente de verdad: la compañía se deriva de él.
  const visibleFirms = useMemo(() => (catalog?.firms ?? []).filter((f) => !f.hidden), [catalog])
  const companies = useMemo(
    () => [...new Set(visibleFirms.map(companyOf))].sort((a, b) => a.localeCompare(b, 'es')),
    [visibleFirms],
  )
  const firmCompany = firm ? companyOf(firm) : (companies[0] ?? '')
  const retos = useMemo(() => {
    const mine = visibleFirms.filter((f) => companyOf(f) === firmCompany)
    const ord = (f: Firm) => (kindOf(f) === 'cfd' ? 0 : 2) + (f.official != null && !f.official ? 1 : 0)
    return [...mine].sort((a, b) => ord(a) - ord(b)) // sort estable: orden del catálogo dentro de cada bloque
  }, [visibleFirms, firmCompany])
  const retosHaveBothKinds = new Set(retos.map(kindOf)).size > 1
  const onCompanyChange = (c: string) => {
    const cand = visibleFirms.filter((f) => companyOf(f) === c)
    const sameKind = firm ? cand.find((f) => kindOf(f) === kindOf(firm)) : undefined
    const pick = sameKind ?? cand[0]
    if (!pick) return
    setFirmKey(pick.key)
    setStale(true) // account_size entra en la carga de la cartera
  }
  const stage = useMemo(
    () => catalog?.axi_stages.find((s) => s.key === stageKey),
    [catalog, stageKey],
  )
  // F2c: fecha efectiva del corte — editable tras elegir campaña; si el usuario
  // la vacía, cae al cut del registro. null = ventana completa (sin corte).
  const oosCampaign = useMemo(
    () => catalog?.oos_campaigns.find((c) => c.key === oosSel),
    [catalog, oosSel],
  )
  const oosStart = oosSel === '' ? null : (oosDate || oosCampaign?.cut || null)
  // Capital del replay de sizing (entra en la carga): la firm en prop firm;
  // la allocation de la etapa en AXI (convención Streamlit: replay equity =
  // baseline de la etapa — la cuenta AXI ES la allocation). El usuario puede
  // sobreescribirlo (override); si cambia etapa/firm, vuelve el valor derivado.
  const accountSizeDerived =
    lens === 'axi' ? (stage?.allocation_usd ?? 0) : (firm?.account_size ?? 0)
  const [accountOverride, setAccountOverride] = useState<number | null>(null)
  useEffect(() => setAccountOverride(null), [accountSizeDerived])
  const accountSize = accountOverride ?? accountSizeDerived

  const toggleLeg = (path: string, on: boolean) => {
    setSelLegs((prev) => {
      const next = new Set(prev)
      if (on) next.add(path)
      else next.delete(path)
      return next
    })
    setStale(true)
  }

  const onPortfolioLoaded = (p: PortfolioLoadResult, _fRef: number) => {
    setLegsNotice(null) // la cartera se acaba de cargar con la selección vigente
    setPortfolio(p)
    setStale(false)
    setResultsStale(false)
    setResult(null) // resultados de una cartera anterior ya no aplican
    setDiagFull(null)
  }

  const canSimulate = !!portfolio && portfolio.ok && !stale && !simLoading

  // GT-9: el panel de reglas y el de supuestos usan la firma de la CORRIDA, no la
  // seleccionada en el desplegable (que puede haber cambiado después de simular).
  const runFirmKey = result?.lens === 'propfirm' ? result.params.firm_key : null
  const runFirm = useMemo(
    () => (runFirmKey ? catalog?.firms.find((f) => f.key === runFirmKey) : undefined),
    [catalog, runFirmKey],
  )

  // Demos (enmienda 7): un botón por carpeta presente en el catálogo. Llevan las
  // patas, el perfil y el capital explícitos para que la carga no dependa del
  // estado que se acaba de poner (sin carreras).
  const demos: DemoSpec[] = useMemo(() => {
    const out: DemoSpec[] = []
    for (const d of DEMO_DEFS) {
      const legs = (catalog?.legs ?? []).filter((l) => l.folder === d.folder).map((l) => l.path)
      const f = catalog?.firms.find((x) => x.key === d.firmKey)
      if (legs.length === 0 || !f || !!f.hidden) continue // perfiles-v2: la demo exige el perfil VISIBLE
      out.push({ ...d, legs, accountSize: f.account_size })
    }
    return out
  }, [catalog])

  const onDemoSelect = (d: DemoSpec) => {
    setSelLegs(new Set(d.legs))
    setLens('propfirm')
    setFirmKey(d.firmKey)
    if (d.dayAxis) setDayAxis(d.dayAxis)
    setAccountOverride(null)
    setSourceTz('') // run() envía source_tz: la demo usa el default
    setStale(true)
    // un clic limpia los resultados en pantalla
    setResult(null)
    setDiagFull(null)
    setSimError(null)
    setResultsStale(false)
  }

  const run = async () => {
    if (!portfolio) return
    setSimLoading(true)
    setSimError(null)
    // P2: la corrida gemela del diagnóstico (sin corte = ventana completa) solo
    // tiene sentido con un corte seleccionado; dobla el tiempo de sim.
    const useDiag = diagOn && oosStart != null
    // P4: archivo silencioso de cada corrida exitosa (resumen SIN trajectory);
    // con diagnóstico activo, OOS y gemela comparten diag_group_id.
    const record = (
      request: Record<string, unknown>,
      data: SimResult | AxiSimResult,
      diagGroupId: string | null,
    ) => {
      postRun({
        lens,
        portfolio_id: portfolio.portfolio_id,
        diag_group_id: diagGroupId,
        request,
        summary: {
          meta: data.meta,
          f_star: data.f_star,
          recommendation: data.recommendation,
          reports: data.reports,
        },
      })
        .then(() => setHistoryKey((k) => k + 1))
        .catch(() => {}) // fallo del archivo ≠ error de sim
    }
    try {
      if (lens === 'propfirm') {
        const payload = {
          portfolio_id: portfolio.portfolio_id,
          firm_key: firmKey,
          f_lo: fLo,
          f_hi: fHi,
          f_step: fStep,
          n_paths: nPaths,
          n_challenges: nChallenges,
          n_sequences: nSequences,
          capital0_eur: capital0,
          fx_eur_usd: fx,
          tol_neg: tolPct / 100,
          block_size: 0,
          stop_on_insolvency: stopRuin,
          oos_start: oosStart,
          day_axis: dayAxis,
          // Fase 6 (Apex): el backend los ignora en perfiles clásicos.
          source_tz: sourceTz || null,
          n_copied_accounts: nAccts,
          cap_violation_tol: capTolPct / 100,
          // Fase 2: siempre presentes; el backend las aplica solo a perfiles
          // futures (en el resto: warning si no son los defaults, silencio si sí).
          reset_policy: resetPolicy,
          withdrawal_policy: wdPolicy,
        }
        const [data, full] = await Promise.all([
          runPropfirm(payload),
          useDiag ? runPropfirm({ ...payload, oos_start: null }) : Promise.resolve(null),
        ])
        setResult({ lens: 'propfirm', data, params: payload })
        setDiagFull(full ? { lens: 'propfirm', data: full, params: { ...payload, oos_start: null } } : null)
        const groupId = useDiag && full ? crypto.randomUUID() : null
        record(payload, data, groupId)
        if (full) record({ ...payload, oos_start: null }, full, groupId)
      } else {
        const payload = {
          portfolio_id: portfolio.portfolio_id,
          stage: stageKey,
          f_lo: fLo,
          f_hi: fHi,
          f_step: fStep,
          n_paths: nPaths,
          tol_ruin: tolRuinPct / 100,
          block_size: bsAxi,
          horizon_days: horizonOn ? horizonDays : null,
          oos_start: oosStart,
        }
        const [data, full] = await Promise.all([
          runAxi(payload),
          useDiag ? runAxi({ ...payload, oos_start: null }) : Promise.resolve(null),
        ])
        setResult({ lens: 'axi', data, params: payload })
        setDiagFull(full ? { lens: 'axi', data: full, params: { ...payload, oos_start: null } } : null)
        const groupId = useDiag && full ? crypto.randomUUID() : null
        record(payload, data, groupId)
        if (full) record({ ...payload, oos_start: null }, full, groupId)
      }
    } catch (e) {
      setSimError(String(e))
      setResult(null)
      setDiagFull(null)
    } finally {
      setSimLoading(false)
      setResultsStale(false)
    }
  }

  // Parámetros para el diagnóstico por pata bajo demanda (P sola @ f* de la
  // última simulación AXI sobre la cartera actualmente cargada).
  const axiDiag =
    result?.lens === 'axi' && portfolio && !stale
      ? {
          f: result.data.f_star,
          stage: stageKey,
          n_paths: nPaths,
          block_size: result.data.meta.block_size,
        }
      : null

  // F2c: control de corte IS/OOS compartido por ambos lentes (regla M2-8:
  // evaluar solo OOS). La fecha queda editable tras elegir campaña.
  const oosCutField = (
    <Field
      label="Corte IS/OOS"
      hint={
        oosCampaign
          ? `Procedencia del corte: ${oosCampaign.source}`
          : 'Sin corte se evalúa TODO el histórico: métricas infladas si las estrategias se seleccionaron sobre este tramo (regla M2-8)'
      }
    >
      <div className="space-y-1.5">
        <Select
          value={oosSel}
          onChange={(k) => {
            setOosSel(k)
            setOosDate(catalog?.oos_campaigns.find((c) => c.key === k)?.cut ?? '')
            setResultsStale(true)
          }}
          options={[
            { value: '', label: 'Sin corte — histórico completo (inflado)' },
            ...(catalog?.oos_campaigns ?? []).map((c) => ({
              value: c.key,
              label: `${c.key} · corte ${c.cut}`,
            })),
          ]}
        />
        {oosSel !== '' && (
          <>
            <TextInput
              type="date"
              value={oosDate}
              onChange={(v) => {
                setOosDate(v)
                setResultsStale(true)
              }}
            />
            <Checkbox
              checked={diagOn}
              onChange={setDiagOn}
              label="Diagnóstico de selección (dobra la sim: compara con ventana completa)"
            />
          </>
        )}
      </div>
    </Field>
  )

  return (
    <div className="mx-auto max-w-[1400px] space-y-5 px-6 py-6">
      {/* Header */}
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-lg font-semibold tracking-tight text-zinc-100">
            Riesgo de cartera <span className="text-zinc-500">· web</span>
          </h1>
          <p className="text-xs text-zinc-500">
            Cartera central + lentes (DESIGN_UI_V2 §2) · motor real{' '}
            <code className="text-zinc-400">src/risk</code> vía shim FastAPI
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {(['simulador', 'perfiles', 'simbolos'] as const).map((v) => (
            <button
              key={v}
              onClick={() => selectView(v)}
              aria-pressed={view === v}
              className={`rounded-md border px-3 py-1.5 text-sm font-medium transition-colors ${
                view === v
                  ? 'border-emerald-700 bg-emerald-950 text-emerald-300'
                  : 'border-zinc-700 text-zinc-400 hover:bg-zinc-800'
              }`}
            >
              {v === 'simulador' ? 'Simulador' : v === 'perfiles' ? 'Perfiles' : 'Símbolos'}
            </button>
          ))}
        </div>
      </header>

      {catalogLoading && (
        <Card>
          <CardBody className="py-2.5 text-xs text-zinc-400">
            Cargando estrategias… la primera vez puede tardar ~1 minuto
          </CardBody>
        </Card>
      )}
      {catalogError && (
        <Card className="border-rose-900 bg-rose-950/40">
          <CardBody className="flex flex-wrap items-center justify-between gap-3 py-2.5 text-xs text-rose-300">
            <span>✗ No se pudo cargar el catálogo: {catalogError}</span>
            <Button
              onClick={retryCatalog}
              disabled={catalogLoading}
              className="w-auto! px-3 py-1.5 text-xs"
            >
              Reintentar
            </Button>
          </CardBody>
        </Card>
      )}

      {/* onboarding-c: vista Perfiles (el simulador sigue montado pero oculto: conserva su estado) */}
      {/* montada siempre (oculta fuera de su vista): conserva tipo activo, compañías abiertas y selección */}
      <div className={view === 'perfiles' ? '' : 'hidden'}>
        <ProfilesView
          catalog={catalog}
          catalogLoading={catalogLoading || catalogReloading}
          onProfilesChanged={reloadCatalog}
        />
      </div>

      {/* entrada-4-bloques F5: vista Símbolos (fichas por bróker), también montada siempre y oculta con CSS. Al cambiar
          fichas el catálogo se recarga (como con los perfiles): el estado `ficha` de las patas .sqx cambia. */}
      <div className={view === 'simbolos' ? '' : 'hidden'}>
        <SymbolsView onSymbolsChanged={reloadCatalog} draftRequest={draftRequest} enabled={symbolsOpened} />
      </div>

      <div className={view === 'simulador' ? 'space-y-5' : 'hidden'}>
      {/* F5 (H11): aviso de las patas que la recarga del catálogo ha quitado de la selección por su ficha */}
      {legsNotice && (
        <Card className="border-amber-900 bg-amber-950/30">
          <CardBody className="flex items-start justify-between gap-3 py-2.5 text-xs text-amber-300">
            <span>⚠ {legsNotice}</span>
            <button
              onClick={() => setLegsNotice(null)}
              className="text-zinc-400 hover:text-zinc-200"
              aria-label="Cerrar aviso"
            >
              ✕
            </button>
          </CardBody>
        </Card>
      )}
      {/* ── Banda Cartera ─────────────────────────────────────────────────── */}
      {catalog && (
        <PortfolioPanel
          catalog={catalog}
          selLegs={selLegs}
          onToggleLeg={toggleLeg}
          accountSize={accountSize}
          onAccountSize={(v) => { setAccountOverride(v); setStale(true) }}
          portfolio={portfolio}
          stale={stale}
          onLoaded={onPortfolioLoaded}
          axiDiag={axiDiag}
          firmKey={lens === 'propfirm' ? firmKey : ''}
          sourceTz={sourceTz || null}
          demos={demos}
          demoBusy={simLoading || catalogReloading}
          onDemoSelect={onDemoSelect}
          onRegisterLeg={registerLeg}
        />
      )}

      {/* ── Banda Pregunta ────────────────────────────────────────────────── */}
      <Card>
        <CardHeader
          title="Pregunta"
          subtitle="Qué quieres saber de esta cartera y con qué parámetros"
        />
        <CardBody className="space-y-4">
          {/* Selector de lente */}
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => {
                setLens('propfirm')
                setStale(true) // account_size del lente entra en la carga
              }}
              className={`rounded-md border px-3 py-1.5 text-sm font-medium transition-colors ${
                lens === 'propfirm'
                  ? 'border-emerald-700 bg-emerald-950 text-emerald-300'
                  : 'border-zinc-700 text-zinc-400 hover:bg-zinc-800'
              }`}
            >
              ¿Juego el challenge de una prop firm?
            </button>
            <button
              onClick={() => {
                setLens('axi')
                setStale(true) // account_size del lente entra en la carga
              }}
              className={`rounded-md border px-3 py-1.5 text-sm font-medium transition-colors ${
                lens === 'axi'
                  ? 'border-emerald-700 bg-emerald-950 text-emerald-300'
                  : 'border-zinc-700 text-zinc-400 hover:bg-zinc-800'
              }`}
            >
              ¿Paso el reto AXI?
            </button>
          </div>

          {/* Contexto del lente */}
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            {lens === 'propfirm' && (
              <>
                <Field label="Firma" hint="Compañía de la prop firm (los perfiles ocultos no salen; se gestionan en la pestaña Perfiles)">
                  <Select
                    value={firmCompany}
                    onChange={onCompanyChange}
                    options={companies.map((c) => ({ value: c, label: c }))}
                  />
                </Field>
                <Field
                  label="Reto"
                  hint="Fija el tamaño de cuenta de la cartera (carga), el fee y las reglas del challenge"
                >
                  <Select
                    value={firmKey}
                    onChange={(k) => {
                      setFirmKey(k)
                      setStale(true) // account_size entra en la carga de la cartera
                    }}
                    options={retos.map((f) => ({
                      value: f.key,
                      group: retosHaveBothKinds ? (kindOf(f) === 'cfd' ? 'CFD' : 'Futuros') : undefined,
                      // Fase 2: los perfiles futures no tienen fee de challenge;
                      // el coste es la suscripción mensual (rules.funded.subscription).
                      label: (() => {
                        const sub =
                          f.funded_model === 'futures' && f.rules && f.rules.model !== 'cycle'
                            ? f.rules.funded.subscription
                            : null
                        return sub
                          ? `${firmName(f)} · suscr. $${sub.monthly_fee_usd.toLocaleString('es-ES')}/mes`
                          : `${firmName(f)} · fee ${f.fee.toLocaleString('es-ES')} ${f.fee_ccy ?? 'EUR'}`
                      })(),
                    }))}
                  />
                </Field>
              </>
            )}
            {lens === 'axi' && (
              <Field label="Etapa AXI" hint="Barreras y suelos de la etapa (config/axi_select.yaml). La cuenta del replay de sizing es la allocation de la etapa">
                <Select
                  value={stageKey}
                  onChange={(k) => {
                    setStageKey(k)
                    setStale(true) // la allocation entra en la carga de la cartera
                  }}
                  options={(catalog?.axi_stages ?? []).map((s) => ({
                    value: s.key,
                    label: `Etapa ${s.display_name} · target ${s.profit_target_pct ?? '—'}% · allocation ${s.allocation_usd.toLocaleString('es-ES')} $`,
                  }))}
                />
              </Field>
            )}
            {oosCutField}
            {lens === 'propfirm' && !isExtendedProfile && (
              <Field
                label="Eje de días"
                hint="FTMO resetea el daily loss a 00:00 CE(S)T = 01:00 servidor AXI. A eje servidor el P(pass) queda ~1–5pp optimista (solo comparación)"
              >
                <Select
                  value={dayAxis}
                  onChange={(k) => {
                    setDayAxis(k)
                    setResultsStale(true)
                  }}
                  options={[
                    { value: 'ftmo', label: 'FTMO (00:00 CE(S)T)' },
                    { value: 'servidor', label: 'Servidor AXI (00:00)' },
                  ]}
                />
              </Field>
            )}
            {lens === 'propfirm' && isExtendedProfile && (
              <Field
                label="Eje de días"
                hint={`El perfil agrupa por sesión (${extendedRules?.day_axis_cut_hhmm ?? '18:00'} → ${minusOneMinute(
                  extendedRules?.day_axis_cut_hhmm ?? '18:00',
                )} ET); el selector ftmo/servidor no aplica`}
              >
                <div className="w-full rounded-md border border-zinc-800 bg-zinc-950/60 px-2 py-1.5 text-sm text-zinc-400">
                  Sesiones ET (lo fija el perfil)
                </div>
              </Field>
            )}
            {lens === 'propfirm' && (
              <>
                <Field label="Capital real (€)">
                  <NumberInput value={capital0} step={500} min={0} onChange={setCapital0} />
                </Field>
                <Field label="EUR/USD">
                  <NumberInput value={fx} step={0.01} min={0.5} onChange={setFx} />
                </Field>
              </>
            )}
          </div>

          {/* Parámetros compartidos */}
          <div className="grid grid-cols-3 gap-3 md:grid-cols-6">
            <Field label="f min (%)">
              <PercentInput value={fLo} step={0.25} min={0.05} onChange={setFLo} />
            </Field>
            <Field label="f max (%)">
              <PercentInput value={fHi} step={0.25} min={0.1} onChange={setFHi} />
            </Field>
            <Field label="paso f (%)">
              <PercentInput value={fStep} step={0.05} min={0.05} onChange={setFStep} />
            </Field>
            <Field label="N paths">
              <NumberInput value={nPaths} step={100} min={100} onChange={setNPaths} />
            </Field>
            {lens === 'propfirm' && (
              <>
                {!isFutures && (
                  <Field label="N challenges">
                    <NumberInput value={nChallenges} step={5} min={1} onChange={setNChallenges} />
                  </Field>
                )}
                <Field
                  label="N secuencias"
                  hint={isFutures ? 'FTMO Futures: nº de campañas de suscripción simuladas' : undefined}
                >
                  <NumberInput value={nSequences} step={250} min={100} onChange={setNSequences} />
                </Field>
              </>
            )}
            {lens === 'axi' && (
              <Field
                label="block_size (0 = automático)"
                hint="0 = automático (método de Politis-White sobre el resultado de cada bloque en R; no depende de la f). Un número fija el tamaño medio a mano."
              >
                <NumberInput value={bsAxi} step={1} min={0} onChange={setBsAxi} />
              </Field>
            )}
          </div>

          {/* Tolerancia + extras por lente */}
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            {lens === 'propfirm' ? (
              <>
                <Field label={`Tolerancia P(neto<0): ${tolPct}%`}>
                  <Slider value={tolPct} onChange={setTolPct} min={5} max={95} step={5} />
                </Field>
                <div className="flex items-center">
                  <Checkbox
                    checked={stopRuin}
                    onChange={setStopRuin}
                    label="Stop por insolvencia (bankroll < fee)"
                  />
                </div>
              </>
            ) : (
              <Field label={`Tolerancia P(ruin): ${tolRuinPct}%`}>
                <Slider value={tolRuinPct} onChange={setTolRuinPct} min={2} max={50} step={2} />
              </Field>
            )}
          </div>

          {/* Fase 6 (Apex): copy trading, umbral de caps y tz de los datos */}
          {lens === 'propfirm' && isExtendedProfile && (
            <div className="grid grid-cols-1 gap-3 rounded-lg border border-zinc-800 bg-zinc-950/40 p-3 md:grid-cols-3">
              {!isFutures && (
                <Field
                  label="Cuentas copiadas (1-20)"
                  hint="Copy trading con correlación PERFECTA: pasan/queman a la vez; escala fees y payouts, no diversifica el riesgo de cola"
                >
                  <NumberInput value={nAccts} step={1} min={1} onChange={(v) => setNAccts(Math.max(1, Math.min(20, Math.round(v))))} />
                </Field>
              )}
              <Field label={`Tolerancia violación caps: ${capTolPct}%`}>
                <Slider value={capTolPct} onChange={setCapTolPct} min={0} max={50} step={5} />
              </Field>
              <Field
                label="tz de los datos (source_tz)"
                hint="Vacío = hora de servidor AXI (Europe/Athens, FX NY-close). Los futuros documentarán su tz (Fase 7)"
              >
                <TextInput
                  value={sourceTz}
                  onChange={setSourceTz}
                  placeholder="Europe/Athens (servidor AXI FX)"
                />
              </Field>
            </div>
          )}

          {/* Fase 2 (FTMO Futures): políticas de la campaña de suscripción */}
          {lens === 'propfirm' && isFutures && (
            <div className="grid grid-cols-1 gap-3 rounded-lg border border-zinc-800 bg-zinc-950/40 p-3 md:grid-cols-2">
              <Field
                label="Política de reset"
                hint="Tras un breach de evaluación: esperar a la siguiente fecha de facturación (gratis) o pagar un reset por reinicio inmediato"
              >
                <Select
                  value={resetPolicy}
                  onChange={setResetPolicy}
                  options={[
                    { value: 'billing_retry', label: 'Reintento en facturación (gratis)' },
                    { value: 'paid_reset', label: 'Reset de pago inmediato' },
                  ]}
                />
              </Field>
              <Field
                label="Política de retirada"
                hint="Qué se retira de la funded en cada payout: todo lo retirable o preservar un buffer contra el drawdown"
              >
                <Select
                  value={wdPolicy}
                  onChange={setWdPolicy}
                  options={[
                    { value: 'greedy', label: 'Greedy (retirar todo lo retirable)' },
                    { value: 'buffer_preserving', label: 'Buffer-preserving (preservar colchón)' },
                  ]}
                />
              </Field>
            </div>
          )}

          {lens === 'axi' && (
            <Collapsible title="Avanzado" subtitle="horizonte máximo opcional (§3.4: pregunta del usuario, no regla de la firm)">
              <div className="flex flex-wrap items-center gap-4">
                <Checkbox
                  checked={horizonOn}
                  onChange={setHorizonOn}
                  label="Limitar horizonte (¿pasaría en N días?)"
                />
                {horizonOn && (
                  <Field label="Horizonte (días)">
                    <NumberInput value={horizonDays} step={10} min={5} onChange={setHorizonDays} />
                  </Field>
                )}
              </div>
            </Collapsible>
          )}

          <div className="md:ml-auto md:w-[280px]">
            <Button onClick={run} disabled={!canSimulate}>
              {simLoading
                ? 'Simulando…'
                : !portfolio
                  ? 'Carga la cartera primero'
                  : stale
                    ? 'Cartera desactualizada — recarga'
                    : !portfolio.ok
                      ? 'Cartera con errores'
                      : resultsStale
                        ? 'Resultados desactualizados — re-simula'
                        : lens === 'propfirm'
                          ? 'Simular EV de la cartera'
                          : 'Simular P(pass) / P(ruin)'}
            </Button>
          </div>
          {simError && (
            <p className="rounded-md border border-rose-900 bg-rose-950/60 px-2 py-1.5 text-xs text-rose-300">
              {simError}
            </p>
          )}
        </CardBody>
      </Card>

      {/* ── Banda Resultados ──────────────────────────────────────────────── */}
      <section className="space-y-5">
        <h2 className="text-sm font-semibold text-zinc-300">Resultados</h2>
        {!result && !simLoading && (
          <Card className="flex h-40 items-center justify-center">
            <p className="text-sm text-zinc-500">
              Carga y valida la cartera, elige lente y pulsa{' '}
              <span className="text-zinc-300">Simular</span>.
            </p>
          </Card>
        )}
        {result?.lens === 'propfirm' && (
          <LensResultsPropfirm
            result={result.data}
            tolNeg={tolPct / 100}
            diag={diagFull?.lens === 'propfirm' ? diagFull.data : null}
            nSeq={nSequences}
            params={result.params}
            firm={runFirm}
          />
        )}
        {result?.lens === 'axi' && (
          <LensResultsAxi
            result={result.data}
            tolRuin={tolRuinPct / 100}
            diag={diagFull?.lens === 'axi' ? diagFull.data : null}
            nPaths={nPaths}
            params={result.params}
            modelNotes={catalog?.model_notes_axi}
          />
        )}

        {/* P4: historial global de corridas (siempre visible) */}
        <HistoryPanel refreshKey={historyKey} />
      </section>
      </div>
    </div>
  )
}
