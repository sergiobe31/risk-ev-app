// Formulario de una ficha de símbolo (guardrail entrada-4-bloques, F5). Mismo patrón que ProfileForm pero en un
// componente propio (ProfileForm no se toca): el formulario sale de GET /api/symbols/schema (etiquetas, unidades,
// secciones, qué es obligatorio) y se valida en vivo contra POST /api/symbols/validate (debounce 400 ms; las
// respuestas viejas se descartan). Tres usos, siempre en la COLUMNA DERECHA de la vista Símbolos:
//  - alta (`mode="new"`): vacía, copiada de otra ficha (duplicar) o desde un BORRADOR (fila de la hoja de MT5 o
//    alta guiada desde un .sqx): los campos obligatorios sin valor (`faltan`) salen vacíos y marcados;
//  - edición (`mode="edit"`) de una ficha TUYA: el bróker y el símbolo SQX van fijos;
//  - un borrador cuyo símbolo SQX YA tiene ficha (`draft.existing`): no hay formulario, sino la tabla de conflictos
//    «actual / propuesto / fuente» (enmienda E11). En una ficha tuya se acepta campo a campo (PUT con los elegidos);
//    en una oficial, de otro bróker u oculta solo se muestra el texto E11 y por qué no se puede guardar.
import { useEffect, useId, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import {
  ASSET_LABEL,
  createSymbol,
  fieldLabel,
  fmtFieldValue,
  fmtSheetNum,
  SOURCE_LABEL,
  SymbolApiError,
  updateSymbol,
  validateSymbol,
  type AssetClass,
  type BrokerView,
  type CommissionKind,
  type FieldSource,
  type SymbolConflict,
  type SymbolDraft,
  type SymbolSchema,
  type SymbolSchemaField,
  type SymbolView,
} from '../lib/api'
import { Badge, Button, Card, CardBody, CardHeader, Checkbox, Modal, SmallBtn } from './ui'
import { SourceBadge } from './SymbolDetail'

const DEBOUNCE_MS = 400
const NOTES_MAX = 2000
const inputCls =
  'w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-sm text-zinc-100 ' +
  'placeholder:text-zinc-600 focus:border-emerald-600 focus:outline-none'

/** Campos de datos de una ficha (los que tienen procedencia y marca de revisión; DATA_FIELDS de symbols.py). */
const DATA_FIELDS = [
  'ticker',
  'point_value_usd',
  'lot_step',
  'min_lot',
  'max_lot',
  'commission',
  'spread_px',
  'ohlc_m1',
  'tick_size',
] as const
type DataField = (typeof DATA_FIELDS)[number]
const isDataField = (s: string): s is DataField => (DATA_FIELDS as readonly string[]).includes(s)

type NumKey = 'point_value_usd' | 'lot_step' | 'min_lot' | 'max_lot' | 'spread_px' | 'tick_size'
const NUM_KEYS: readonly string[] = ['point_value_usd', 'lot_step', 'min_lot', 'max_lot', 'spread_px', 'tick_size']
const isNumKey = (s: string): s is NumKey => NUM_KEYS.includes(s)

export type FormOrigin = 'manual' | 'mt5_sheet' | 'sqx' | 'duplicate'

/** Cómo se abre un alta: de cero, copiada de otra ficha o desde un borrador. Debe ser un objeto ESTABLE (estado del padre). */
export interface FormPreset {
  origin: FormOrigin
  title?: string // cabecera ("Alta desde un .sqx", "Duplicar …")
  subtitle?: string
  broker?: string // bróker preseleccionado
  lockBroker?: boolean // el bróker destino viene fijado (la hoja de MT5 se leyó contra él)
  brokerFilter?: (b: BrokerView) => boolean // brókers elegibles (duplicar: los tuyos de la misma clase)
  draft?: SymbolDraft // fila de la hoja de MT5 o lo que propone un .sqx
  copyOf?: SymbolView // duplicar: ficha de la que se copian los datos
}

export interface SymbolFormProps {
  schema: SymbolSchema | null
  schemaError?: string | null
  brokers: BrokerView[]
  symbols: SymbolView[] // para localizar la ficha existente de un borrador
  mode: 'new' | 'edit'
  sheet?: SymbolView // mode === 'edit'
  preset?: FormPreset // mode === 'new'
  onSaved: (view: SymbolView, how: 'created' | 'updated') => void
  onCancel: () => void
  onDirtyChange?: (dirty: boolean) => void // debe ser estable (useCallback) en el padre
  // Un borrador cuyo símbolo SQX ya tiene ficha, pero OCULTA: el panel ofrece «Mostrar la ficha» (la muestra y cierra el borrador).
  onShowHidden?: (sheet: SymbolView) => Promise<void>
  cancelLabel?: string
}

interface FormValues {
  broker: string
  sqx_symbol: string
  ticker: string
  point_value_usd: number | null
  lot_step: number | null
  min_lot: number | null
  max_lot: number | null
  commission: { kind: CommissionKind; value: number | null }
  spread_px: number | null
  ohlc_m1: string
  tick_size: number | null
  verified_on: string
  notes: string
}

const EMPTY: FormValues = {
  broker: '',
  sqx_symbol: '',
  ticker: '',
  point_value_usd: null,
  lot_step: null,
  min_lot: null,
  max_lot: null,
  commission: { kind: 'per_lot_rt', value: null },
  spread_px: null,
  ohlc_m1: '',
  tick_size: null,
  verified_on: '',
  notes: '',
}

const todayIso = (): string => new Date().toLocaleDateString('sv-SE') // AAAA-MM-DD en hora local

function commissionState(c: { per_lot_rt?: number; pct_notional_rt?: number } | null | undefined): FormValues['commission'] {
  if (c?.per_lot_rt != null) return { kind: 'per_lot_rt', value: c.per_lot_rt }
  if (c?.pct_notional_rt != null) return { kind: 'pct_notional_rt', value: c.pct_notional_rt }
  return { kind: 'per_lot_rt', value: null }
}

function fromSheet(s: SymbolView): FormValues {
  return {
    broker: s.broker,
    sqx_symbol: s.sqx_symbol,
    ticker: s.ticker ?? '',
    point_value_usd: s.point_value_usd,
    lot_step: s.lot_step,
    min_lot: s.min_lot,
    max_lot: s.max_lot,
    commission: commissionState(s.commission),
    spread_px: s.spread_px,
    ohlc_m1: s.ohlc_m1 ?? '',
    tick_size: s.tick_size,
    verified_on: s.verified_on ?? '',
    notes: s.notes ?? '',
  }
}

function fromDraft(d: SymbolDraft, broker: string | undefined): FormValues {
  return {
    broker: d.broker ?? broker ?? '',
    sqx_symbol: d.sqx_symbol ?? '',
    ticker: d.ticker ?? '',
    point_value_usd: d.point_value_usd,
    lot_step: d.lot_step,
    min_lot: d.min_lot,
    max_lot: d.max_lot,
    commission: commissionState(d.commission),
    spread_px: d.spread_px,
    ohlc_m1: d.ohlc_m1 ?? '',
    tick_size: d.tick_size,
    verified_on: d.verified_on ?? '',
    notes: d.notes ?? '',
  }
}

function initialValues(mode: 'new' | 'edit', sheet: SymbolView | undefined, preset: FormPreset | undefined): FormValues {
  if (mode === 'edit' && sheet) return fromSheet(sheet)
  if (preset?.draft) return fromDraft(preset.draft, preset.broker)
  if (preset?.copyOf) {
    // Se copian los números; el símbolo SQX lo pone el usuario (es único entre brókers) y el OHLC es del bróker
    // original, no del nuevo.
    return {
      ...fromSheet(preset.copyOf),
      broker: preset.broker ?? '',
      sqx_symbol: '',
      ohlc_m1: '',
      verified_on: '',
      notes: `Copia de ${preset.copyOf.broker}/${preset.copyOf.sqx_symbol}.`,
    }
  }
  return { ...EMPTY, broker: preset?.broker ?? '' }
}

const nz = (s: string): string | null => (s.trim() ? s.trim() : null)

/** Valor de un campo de datos tal como viaja al backend (para comparar con la situación inicial). */
function fieldValue(v: FormValues, f: DataField): unknown {
  switch (f) {
    case 'commission':
      return v.commission.value == null ? null : { [v.commission.kind]: v.commission.value }
    case 'ticker':
      return nz(v.ticker)
    case 'ohlc_m1':
      return nz(v.ohlc_m1)
    default:
      return v[f]
  }
}

const sameField = (f: DataField, a: FormValues, b: FormValues): boolean =>
  JSON.stringify(fieldValue(a, f)) === JSON.stringify(fieldValue(b, f))

function buildBody(
  v: FormValues,
  broker: BrokerView | undefined,
  mode: 'new' | 'edit',
  initial: FormValues,
): Record<string, unknown> {
  const futures = broker?.asset_class === 'futures'
  const body: Record<string, unknown> = {
    broker: v.broker,
    sqx_symbol: v.sqx_symbol.trim(),
    ticker: nz(v.ticker),
    point_value_usd: v.point_value_usd,
    lot_step: futures ? 1 : v.lot_step,
    min_lot: futures ? 1 : v.min_lot,
    max_lot: v.max_lot,
    commission: v.commission.value == null ? null : { [v.commission.kind]: v.commission.value },
    spread_px: v.spread_px,
    tick_size: v.tick_size,
    verified_on: v.verified_on || null,
    notes: v.notes.trim() ? v.notes : null,
  }
  // Fichero de OHLC (revisión de código de la F5, H4). Con carpeta de OHLC en el bróker se manda tal cual. Sin ella el backend
  // rechaza un nombre de fichero, y un `null` en un PUT PARCIAL borraría en silencio el que tuviera la ficha: se omite la clave
  // (el PUT parcial conserva lo que haya) salvo que el usuario haya cambiado el campo a propósito (p. ej. para borrarlo).
  if (broker?.ohlc_dir) body.ohlc_m1 = nz(v.ohlc_m1)
  else if (mode === 'edit' && v.ohlc_m1 !== initial.ohlc_m1) body.ohlc_m1 = nz(v.ohlc_m1)
  return body
}

/** ¿El campo tiene algún valor? (para ofrecer «Ya lo he verificado» solo donde hay algo que verificar). */
function hasValue(name: string, v: FormValues): boolean {
  if (isNumKey(name)) return v[name] != null
  if (name === 'commission') return v.commission.value != null
  if (name === 'ohlc_m1') return nz(v.ohlc_m1) != null
  if (name === 'ticker') return nz(v.ticker) != null
  return true
}

function isMissing(name: string, v: FormValues, futures: boolean): boolean {
  switch (name) {
    case 'broker':
      return !v.broker
    case 'sqx_symbol':
      return !v.sqx_symbol.trim()
    case 'point_value_usd':
      return v.point_value_usd == null
    case 'lot_step':
      return !futures && v.lot_step == null
    case 'min_lot':
      return !futures && v.min_lot == null
    case 'commission':
      return v.commission.value == null
    default:
      return false
  }
}

// "point_value_usd (Valor del punto): falta el valor" -> campo + mensaje sin prefijo. "commission.per_lot_rt (...)" -> commission.
function splitError(e: string): { field: string | null; msg: string } {
  const m = /^([A-Za-z0-9_.]+)(?: \(|:)/.exec(e)
  if (m) {
    const field = m[1].split('.')[0]
    const i = e.indexOf('): ')
    const j = e.indexOf(': ')
    const msg = e.startsWith(`${m[1]} (`) && i >= 0 ? e.slice(i + 3) : j >= 0 ? e.slice(j + 2) : e
    return { field, msg }
  }
  if (e.startsWith('el símbolo SQX ')) return { field: 'sqx_symbol', msg: e }
  return { field: null, msg: e }
}

// --- Entradas ------------------------------------------------------------------------------------

/**
 * Accesibilidad de un control (revisión de código de la F5, H2). El control PRINCIPAL de cada fila lleva `id` y la etiqueta
 * de la fila apunta a él con `htmlFor` (nombre accesible = la etiqueta; al pulsarla, el foco va al control). Los controles
 * secundarios (p. ej. el tipo de comisión) no tienen etiqueta visible propia y llevan `aria-label`.
 */
interface Ctl {
  id?: string
  'aria-label'?: string
  'aria-describedby'?: string // la ayuda y el error del campo
  'aria-invalid'?: boolean
  'aria-required'?: boolean
}

function parseLoose(s: string): number | null {
  const t = s.trim().replace(',', '.')
  if (!t) return null
  const v = Number(t)
  return Number.isFinite(v) ? v : null
}

const fmtNum = (v: number): string => String(parseFloat(v.toPrecision(10)))

// Número que admite VACÍO (null): una ficha nueva parte de campos sin rellenar. Coma o punto decimal.
function OptNumberInput({
  value,
  onChange,
  placeholder,
  ...ctl
}: {
  value: number | null
  onChange: (v: number | null) => void
  placeholder?: string
} & Ctl) {
  const [draft, setDraft] = useState<string | null>(null)
  const shown = draft ?? (value == null ? '' : fmtNum(value))
  return (
    <input
      type="text"
      inputMode="decimal"
      className={inputCls}
      value={shown}
      placeholder={placeholder}
      {...ctl}
      onFocus={() => setDraft(value == null ? '' : fmtNum(value))}
      onChange={(e) => {
        setDraft(e.target.value)
        if (e.target.value.trim() === '') onChange(null)
        else {
          const v = parseLoose(e.target.value)
          if (v !== null) onChange(v) // con basura no se compromete: el usuario sigue escribiendo
        }
      }}
      onBlur={() => setDraft(null)}
    />
  )
}

// Equivalentes de TextInput, TextArea y Select de ui.tsx (mismo aspecto) que admiten `id` y atributos aria: ui.tsx no se toca.
function FormInput({
  value,
  onChange,
  type = 'text',
  placeholder,
  list,
  ...ctl
}: {
  value: string
  onChange: (v: string) => void
  type?: string
  placeholder?: string
  list?: string
} & Ctl) {
  return (
    <input
      type={type}
      className={inputCls}
      value={value}
      placeholder={placeholder}
      list={list}
      onChange={(e) => onChange(e.target.value)}
      {...ctl}
    />
  )
}

function FormTextArea({
  value,
  onChange,
  rows = 3,
  placeholder,
  maxLength,
  ...ctl
}: {
  value: string
  onChange: (v: string) => void
  rows?: number
  placeholder?: string
  maxLength?: number
} & Ctl) {
  return (
    <textarea
      className={inputCls}
      rows={rows}
      value={value}
      placeholder={placeholder}
      maxLength={maxLength}
      onChange={(e) => onChange(e.target.value)}
      {...ctl}
    />
  )
}

function FormSelect<T extends string>({
  value,
  onChange,
  options,
  ...ctl
}: {
  value: T
  onChange: (v: T) => void
  options: { value: T; label: string; group?: string }[]
} & Ctl) {
  const opt = (o: { value: T; label: string }) => (
    <option key={o.value} value={o.value}>
      {o.label}
    </option>
  )
  // las opciones del mismo grupo van seguidas (como en ui.Select)
  const groups: { name: string | null; items: typeof options }[] = []
  for (const o of options) {
    const name = o.group ?? null
    const last = groups[groups.length - 1]
    if (last && last.name === name) last.items.push(o)
    else groups.push({ name, items: [o] })
  }
  return (
    <select className={inputCls} value={value} onChange={(e) => onChange(e.target.value as T)} {...ctl}>
      {groups.map((g, i) =>
        g.name == null ? (
          g.items.map(opt)
        ) : (
          <optgroup key={`${g.name}-${i}`} label={g.name}>
            {g.items.map(opt)}
          </optgroup>
        ),
      )}
    </select>
  )
}

function Row({
  label,
  htmlFor,
  help,
  helpId,
  unit,
  error,
  errorId,
  review,
  missing,
  children,
}: {
  label: string
  htmlFor?: string // id del control principal de la fila (si no hay control, la etiqueta es solo texto)
  help?: string
  helpId?: string
  unit?: ReactNode
  error?: string
  errorId?: string
  review?: boolean // el campo está en needs_review
  missing?: boolean // obligatorio y vacío
  children: ReactNode
}) {
  return (
    <div className={`py-1.5 ${review ? 'rounded-md bg-amber-950/20 px-2' : ''}`}>
      <div className="grid grid-cols-1 items-start gap-x-3 gap-y-1 sm:grid-cols-[minmax(0,1fr)_210px_minmax(0,150px)]">
        <div className="min-w-0">
          <span className={`text-sm ${review ? 'text-amber-300' : 'text-zinc-300'}`}>
            {htmlFor ? <label htmlFor={htmlFor}>{label}</label> : label}
            {review && <span className="ml-1.5 text-[11px] text-amber-400">· sin verificar</span>}
            {missing && <span className="ml-1.5 text-[11px] text-rose-400">· obligatorio</span>}
          </span>
          {help && (
            <p id={helpId} className="text-[11px] leading-snug text-zinc-500">
              {help}
            </p>
          )}
        </div>
        <div className="space-y-1">{children}</div>
        <div className="flex flex-wrap items-center gap-1.5 pt-1.5 text-xs text-zinc-500">{unit}</div>
      </div>
      {error && (
        <p id={errorId} className="mt-0.5 text-xs text-rose-300">
          {error}
        </p>
      )}
    </div>
  )
}

const ReadOnly = ({ children }: { children: ReactNode }) => (
  <div className="w-full rounded-md border border-zinc-800 bg-zinc-950/60 px-2 py-1.5 text-sm text-zinc-300">{children}</div>
)

// --- Cuerpo del formulario -----------------------------------------------------------------------

function SymbolFormBody({
  schema,
  schemaError,
  brokers,
  mode,
  sheet,
  preset,
  onSaved,
  onCancel,
  onDirtyChange,
  cancelLabel = '← Volver',
}: SymbolFormProps) {
  const draft = preset?.draft
  const [initial] = useState<FormValues>(() => initialValues(mode, sheet, preset))
  const [values, setValues] = useState<FormValues>(initial)
  // campos tocados: en edición y con datos precargados se muestran todos los errores; en un alta vacía, solo los de lo tocado
  const [touched, setTouched] = useState<Set<string>>(() => {
    const s = new Set<string>()
    if (mode === 'edit' || preset?.draft || preset?.copyOf) {
      for (const k of Object.keys(initial) as (keyof FormValues)[]) {
        const v = initial[k]
        if (v != null && v !== '' && !(k === 'commission' && (v as FormValues['commission']).value == null)) s.add(k)
      }
    }
    return s
  })
  const [verified, setVerified] = useState<Set<string>>(new Set()) // campos en needs_review que el usuario da por buenos
  const [validated, setValidated] = useState<{ sig: string; errors: string[] } | null>(null)
  const [saving, setSaving] = useState(false)
  const [saveErrors, setSaveErrors] = useState<string[]>([])
  const [confirmCancel, setConfirmCancel] = useState(false)

  const brokerObj = brokers.find((b) => b.id === values.broker)
  const hasBroker = !!values.broker
  const assetClass: AssetClass = brokerObj?.asset_class ?? 'cfd'
  const futures = assetClass === 'futures'
  const formId = useId() // prefijo de los `id` de los controles (la etiqueta de cada fila apunta a su control)
  // El cuerpo viaja como JSON: la firma de la validación en vivo y las dependencias del efecto son cadenas (estables), no objetos.
  const bodyJson = JSON.stringify(buildBody(values, brokerObj, mode, initial))
  const sig = `${mode}|${bodyJson}`
  const dirty = JSON.stringify(values) !== JSON.stringify(initial) || verified.size > 0
  const pending = hasBroker && validated?.sig !== sig
  const rawErrors = useMemo(() => (validated && validated.sig === sig ? validated.errors : []), [validated, sig])

  // campos sin verificar: los de la ficha que se edita, o los que propone el borrador (menos 'currency', que no es de la ficha)
  const reviewFields = useMemo(() => {
    const src = mode === 'edit' ? (sheet?.needs_review ?? []) : (draft?.needs_review ?? [])
    return new Set(src.filter(isDataField))
  }, [mode, sheet, draft])
  const provenance: Record<string, FieldSource> | undefined = mode === 'edit' ? sheet?.sources : draft?.sources

  const required = useMemo(() => (schema?.sheet_fields ?? []).filter((f) => f.required).map((f) => f.name), [schema])
  const missing = required.filter((n) => isMissing(n, values, futures))

  // Se avisa al padre de que hay cambios sin guardar (para su confirmación de salida con Modal).
  useEffect(() => {
    onDirtyChange?.(dirty)
  }, [dirty, onDirtyChange])
  useEffect(() => () => onDirtyChange?.(false), [onDirtyChange])

  // Validación en vivo: debounce de 400 ms; las respuestas viejas se descartan. Sin bróker no hay qué validar.
  useEffect(() => {
    if (!hasBroker) return
    let alive = true
    const mySig = sig
    const t = setTimeout(() => {
      const payload = JSON.parse(bodyJson) as Record<string, unknown>
      validateSymbol({ ...payload, editing: mode === 'edit' })
        .then((r) => alive && setValidated({ sig: mySig, errors: r.errors }))
        .catch((e) => alive && setValidated({ sig: mySig, errors: [e instanceof SymbolApiError ? e.errors.join(' · ') : String(e)] }))
    }, DEBOUNCE_MS)
    return () => {
      alive = false
      clearTimeout(t)
    }
  }, [bodyJson, mode, sig, hasBroker])

  // Errores a mostrar: en edición y con datos precargados, todos; en un alta vacía, solo los de campos tocados y los
  // globales. Los de un campo obligatorio vacío los cubre el aviso «Faltan N campos».
  const { fieldErrors, shownErrors } = useMemo(() => {
    const byField = new Map<string, string>()
    const shown: string[] = []
    for (const e of rawErrors) {
      const { field, msg } = splitError(e)
      if (field && schema?.sheet_fields.some((f) => f.name === field)) {
        if (required.includes(field) && isMissing(field, values, futures)) continue
        if (!touched.has(field)) continue
        if (!byField.has(field)) byField.set(field, msg)
      }
      shown.push(e)
    }
    return { fieldErrors: byField, shownErrors: shown }
  }, [rawErrors, schema, required, values, futures, touched])

  const hardBlock = !!draft && !draft.guardable && draft.needs_review.includes('currency')
  const canSave =
    !hardBlock && !!schema && !!values.broker && !pending && rawErrors.length === 0 && !saving &&
    (mode === 'new' ? missing.length === 0 : dirty)

  const upd = (patch: Partial<FormValues>) => {
    setValues((v) => ({ ...v, ...patch }))
    setTouched((s) => {
      const n = new Set(s)
      for (const k of Object.keys(patch)) n.add(k)
      return n
    })
  }
  const setNumber = (key: NumKey, v: number | null) => upd({ [key]: v } as Partial<FormValues>)
  const setBroker = (id: string) => {
    const b = brokers.find((x) => x.id === id)
    setValues((v) => ({
      ...v,
      broker: id,
      // % del nocional solo vale en CFD; sin carpeta de OHLC en el bróker no hay fichero M1
      commission: b?.asset_class === 'futures' && v.commission.kind === 'pct_notional_rt' ? { ...v.commission, kind: 'per_lot_rt' } : v.commission,
      ohlc_m1: b && !b.ohlc_dir ? '' : v.ohlc_m1,
    }))
    setTouched((s) => new Set(s).add('broker'))
  }

  const save = async () => {
    setSaving(true)
    setSaveErrors([])
    const body = buildBody(values, brokerObj, mode, initial)
    try {
      if (mode === 'edit' && sheet) {
        // Edición PARCIAL: se mandan los campos y, explícitos, sus campos sin verificar (el backend no sabe que el valor
        // «verificado» no ha cambiado). Lo cambiado pasa a manual; lo dado por bueno, también.
        const changed = DATA_FIELDS.filter((f) => !sameField(f, values, initial))
        const sources: Record<string, FieldSource> = {}
        for (const f of verified) sources[f] = { from: 'manual', date: todayIso() }
        const needs = sheet.needs_review.filter((f) => !(verified.has(f) || (isDataField(f) && changed.includes(f))))
        const view = await updateSymbol(sheet.broker, sheet.sqx_symbol, { ...body, sources, needs_review: needs })
        onSaved(view, 'updated')
      } else {
        // Alta. Con un borrador se conserva la procedencia (hoja MT5 / .sqx) de lo que el usuario NO ha tocado y esos campos
        // siguen sin verificar; lo que cambia o da por bueno pasa a manual (el backend lo marca así si no se manda).
        // H3 (revisión de código de la F5): `sources` solo lleva la PROCEDENCIA; `needs_review` sale de recorrer el
        // `needs_review` del borrador completo, así un campo sin entrada en `sources` no pierde su marca de «sin verificar».
        const sources: Record<string, FieldSource> = {}
        const needs: string[] = []
        if (draft) {
          const untouched = (f: DataField) => !verified.has(f) && sameField(f, values, initial)
          for (const f of DATA_FIELDS) {
            const src = draft.sources[f]
            if (src && untouched(f)) sources[f] = { from: src.from, date: src.date }
          }
          for (const f of draft.needs_review) {
            // 'currency' no es un campo de la ficha; un campo sin valor no tiene nada que verificar
            if (isDataField(f) && untouched(f) && hasValue(f, values)) needs.push(f)
          }
        }
        const view = await createSymbol(draft ? { ...body, sources, needs_review: needs } : body)
        onSaved(view, 'created')
      }
    } catch (e) {
      setSaveErrors(e instanceof SymbolApiError ? e.errors : [String(e)])
    } finally {
      setSaving(false)
    }
  }

  const cancel = () => (dirty ? setConfirmCancel(true) : onCancel())

  const sections = useMemo(() => {
    const order: string[] = []
    const by = new Map<string, SymbolSchemaField[]>()
    for (const f of schema?.sheet_fields ?? []) {
      if (!by.has(f.section)) {
        by.set(f.section, [])
        order.push(f.section)
      }
      by.get(f.section)!.push(f)
    }
    return order.map((name) => ({ name, fields: by.get(name)! }))
  }, [schema])

  const eligibleBrokers = brokers.filter(preset?.brokerFilter ?? (() => true))

  const brokerSelect = (a11y: Ctl): { input: ReactNode; labelled: boolean } => {
    if (mode === 'edit' || preset?.lockBroker) {
      return { input: <ReadOnly>{brokerObj ? `${brokerObj.display_name} (${brokerObj.id})` : values.broker}</ReadOnly>, labelled: false }
    }
    const mk = (b: BrokerView) => ({
      value: b.id,
      label: `${b.display_name} · ${ASSET_LABEL[b.asset_class]}`,
      group: b.official ? 'Oficiales' : 'Tuyos',
    })
    return {
      input: (
        <FormSelect
          value={values.broker}
          onChange={setBroker}
          options={[
            ...(values.broker ? [] : [{ value: '', label: 'Elige un bróker…', group: undefined }]),
            ...eligibleBrokers.filter((b) => b.official).map(mk),
            ...eligibleBrokers.filter((b) => !b.official).map(mk),
          ]}
          {...a11y}
        />
      ),
      labelled: true,
    }
  }

  // `a11y` lleva el `id` y las descripciones del control PRINCIPAL de la fila; `labelled` dice si lo ha usado (si no, la fila
  // no tiene control: valor de solo lectura). Los controles secundarios llevan su propio `aria-label`.
  const renderInput = (f: SymbolSchemaField, a11y: Ctl): { input: ReactNode; unit?: ReactNode; labelled: boolean } => {
    if (f.type === 'broker') return brokerSelect(a11y)
    if (f.name === 'sqx_symbol') {
      if (mode === 'edit') return { input: <ReadOnly>{values.sqx_symbol}</ReadOnly>, labelled: false }
      const cands = draft?.candidates ?? []
      const listId = `${formId}-candidates`
      return {
        input: (
          <>
            <FormInput
              value={values.sqx_symbol}
              onChange={(v) => upd({ sqx_symbol: v })}
              list={cands.length ? listId : undefined}
              placeholder="p. ej. XAUUSD_mibroker"
              {...a11y}
            />
            {cands.length > 0 && (
              <datalist id={listId}>
                {cands.map((c) => (
                  <option key={c} value={c} />
                ))}
              </datalist>
            )}
          </>
        ),
        unit: draft?.sqx_symbol_suggested ? <Badge variant="outline">sugerido</Badge> : undefined,
        labelled: true,
      }
    }
    if (f.name === 'ticker') {
      return {
        input: <FormInput value={values.ticker} onChange={(v) => upd({ ticker: v })} placeholder="opcional" {...a11y} />,
        labelled: true,
      }
    }
    if (f.type === 'number' && isNumKey(f.name)) {
      const fixed = f.fixed?.[assetClass]
      if (fixed != null) {
        return {
          input: <ReadOnly>{fmtSheetNum(fixed)}</ReadOnly>,
          unit: <span>fijo en {ASSET_LABEL[assetClass].toLowerCase()}</span>,
          labelled: false,
        }
      }
      return {
        input: <OptNumberInput value={values[f.name]} onChange={(v) => setNumber(f.name as NumKey, v)} {...a11y} />,
        unit: f.unit ? <span>{f.unit}</span> : undefined,
        labelled: true,
      }
    }
    if (f.type === 'commission') {
      const opts = (f.options ?? []).filter((o) => o.applies_to.includes(assetClass))
      const cur = opts.find((o) => o.id === values.commission.kind) ?? opts[0]
      return {
        input: (
          <>
            {opts.length > 1 && (
              <FormSelect<CommissionKind>
                value={values.commission.kind}
                onChange={(k) => upd({ commission: { ...values.commission, kind: k } })}
                options={opts.map((o) => ({ value: o.id, label: o.label }))}
                aria-label="Tipo de comisión"
              />
            )}
            <OptNumberInput
              value={values.commission.value}
              placeholder="0 si no hay comisión"
              onChange={(v) => upd({ commission: { ...values.commission, value: v } })}
              {...a11y}
            />
          </>
        ),
        unit: <span>{opts.length > 1 ? (cur?.unit ?? '') : (cur?.label ?? '')}</span>,
        labelled: true,
      }
    }
    if (f.name === 'ohlc_m1') {
      const hasDir = !!brokerObj?.ohlc_dir
      // Sin carpeta de OHLC no hay fichero que elegir: solo lectura. Salvo que la ficha ya lo traiga (H4): entonces se ve y se puede
      // borrar a propósito; el backend dirá que no se puede conservar sin carpeta si se deja.
      if (!hasDir && !values.ohlc_m1) {
        return {
          input: <ReadOnly>—</ReadOnly>,
          unit: <span>{values.broker ? 'este bróker no tiene carpeta de OHLC (sin flotante por minuto)' : 'elige antes el bróker'}</span>,
          labelled: false,
        }
      }
      return {
        input: (
          <FormInput value={values.ohlc_m1} onChange={(v) => upd({ ohlc_m1: v })} placeholder="p. ej. XAUUSD_mibroker.csv" {...a11y} />
        ),
        unit: hasDir ? undefined : <span>este bróker no tiene carpeta de OHLC: el fichero no se usa; bórralo para poder guardar</span>,
        labelled: true,
      }
    }
    if (f.type === 'date') {
      return {
        input: (
          <div className="flex gap-2">
            <FormInput type="date" value={values.verified_on} onChange={(v) => upd({ verified_on: v })} {...a11y} />
            <button
              type="button"
              aria-label="Poner la fecha de hoy en «Verificada el»"
              onClick={() => upd({ verified_on: todayIso() })}
              className="shrink-0 rounded-md border border-zinc-700 px-2 text-xs text-zinc-300 hover:bg-zinc-800"
            >
              hoy
            </button>
          </div>
        ),
        labelled: true,
      }
    }
    if (f.type === 'text') {
      return {
        input: (
          <FormTextArea
            value={values.notes}
            rows={3}
            maxLength={f.max ?? NOTES_MAX}
            placeholder="De dónde salen los números, qué cambió, etc."
            onChange={(v) => upd({ notes: v })}
            {...a11y}
          />
        ),
        unit: <span>{values.notes.length}/{f.max ?? NOTES_MAX}</span>,
        labelled: true,
      }
    }
    return { input: <ReadOnly>—</ReadOnly>, labelled: false }
  }

  const title =
    mode === 'edit'
      ? `Editar ficha · ${sheet?.sqx_symbol ?? ''}`
      : (preset?.title ?? 'Nueva ficha') + (values.sqx_symbol.trim() ? ` · ${values.sqx_symbol.trim()}` : '')
  const subtitle =
    mode === 'edit'
      ? `bróker ${brokerObj?.display_name ?? values.broker} (${values.broker})`
      : preset?.subtitle

  return (
    <Card>
      <CardHeader
        title={title}
        subtitle={subtitle}
        right={
          <button
            onClick={cancel}
            className="rounded-md border border-zinc-700 px-2.5 py-1 text-xs text-zinc-300 hover:bg-zinc-800"
          >
            {cancelLabel}
          </button>
        }
      />
      <CardBody className="space-y-4">
        <OriginNote preset={preset} mode={mode} />

        {draft && draft.warnings.length > 0 && (
          <ul className="list-disc space-y-0.5 rounded-md border border-amber-900 bg-amber-950/40 py-2 pl-7 pr-3 text-xs text-amber-300">
            {draft.warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        )}

        {draft && !draft.guardable && draft.motivo && !hardBlock && (
          <p className="rounded-md border border-amber-900 bg-amber-950/40 px-3 py-2 text-xs text-amber-300">
            Este borrador aún no se puede guardar: {draft.motivo}.
          </p>
        )}

        <div className="grid grid-cols-1 gap-5 xl:grid-cols-[minmax(0,1fr)_300px]">
          <div className="space-y-4">
            {schemaError && (
              <p className="rounded-md border border-rose-900 bg-rose-950/60 px-2 py-1.5 text-xs text-rose-300">
                No se pudo cargar el formulario: {schemaError}
              </p>
            )}
            {!schema && !schemaError && <p className="text-sm text-zinc-500">Cargando campos…</p>}
            {sections.map((sec) => (
              <section key={sec.name} className="rounded-lg border border-zinc-800 bg-zinc-950/40">
                <h4 className="border-b border-zinc-800 px-3 py-2 text-xs font-semibold uppercase tracking-wide text-zinc-400">
                  {sec.name}
                </h4>
                <div className="divide-y divide-zinc-800/60 px-3">
                  {sec.fields.map((f) => {
                    const cid = `${formId}-${f.name}` // id del control principal de la fila (H2)
                    const err = fieldErrors.get(f.name)
                    const { input, unit, labelled } = renderInput(f, {
                      id: cid,
                      'aria-describedby': [f.help ? `${cid}-help` : '', err ? `${cid}-err` : ''].filter(Boolean).join(' ') || undefined,
                      'aria-invalid': err ? true : undefined,
                      'aria-required': f.required ? true : undefined,
                    })
                    const review = isDataField(f.name) && reviewFields.has(f.name)
                    const src = provenance?.[f.name]
                    return (
                      <Row
                        key={f.name}
                        label={f.label}
                        htmlFor={labelled ? cid : undefined}
                        help={f.help}
                        helpId={`${cid}-help`}
                        error={err}
                        errorId={`${cid}-err`}
                        review={review}
                        missing={mode === 'new' && f.required && isMissing(f.name, values, futures)}
                        unit={
                          <>
                            {unit}
                            {src && !(mode === 'new' && !draft) && <SourceBadge source={src} />}
                          </>
                        }
                      >
                        {input}
                        {/* PLAN-DESVIO (adición): el brief no pide un control para dar por verificado un campo, pero sin él un campo
                            en `needs_review` (lo que propone el .sqx, la comisión 0 de la hoja) no se podría confirmar sin cambiarle
                            el valor: la guarda `fichas` de la carga avisaría para siempre. Al guardar, el campo pasa a procedencia
                            «manual» y sale de `needs_review` (el backend lo acepta explícito en el PUT/POST). */}
                        {review && hasValue(f.name, values) && (
                          <Checkbox
                            checked={verified.has(f.name)}
                            onChange={(on) =>
                              setVerified((s) => {
                                const n = new Set(s)
                                if (on) n.add(f.name)
                                else n.delete(f.name)
                                return n
                              })
                            }
                            label={
                              <span className="text-xs text-amber-300">
                                Ya lo he verificado<span className="sr-only"> el campo «{f.label}»</span>
                              </span>
                            }
                          />
                        )}
                      </Row>
                    )
                  })}
                </div>
              </section>
            ))}
          </div>

          <aside className="space-y-3 xl:sticky xl:top-4 xl:self-start">
            {mode === 'new' && missing.length > 0 && (
              <div
                className="rounded-md border border-amber-900 bg-amber-950/40 px-3 py-2 text-xs text-amber-300"
                aria-live="polite"
              >
                {missing.length === 1 ? 'Falta 1 campo por rellenar' : `Faltan ${missing.length} campos por rellenar`}
                {': '}
                {missing.map(fieldLabel).join(', ')}
              </div>
            )}

            {!values.broker ? (
              <div className="rounded-md border border-zinc-800 bg-zinc-950/60 px-3 py-2 text-xs text-zinc-400">
                Elige un bróker para validar la ficha.
              </div>
            ) : (
              !(mode === 'new' && missing.length > 0 && shownErrors.length === 0 && !pending) && (
                <div
                  className={`rounded-md border px-3 py-2 text-xs ${
                    shownErrors.length
                      ? 'border-rose-900 bg-rose-950/50 text-rose-300'
                      : pending
                        ? 'border-zinc-800 bg-zinc-950/60 text-zinc-400'
                        : rawErrors.length > 0
                          ? 'border-rose-900 bg-rose-950/50 text-rose-300'
                          : 'border-emerald-900 bg-emerald-950/40 text-emerald-300'
                  }`}
                  aria-live="polite"
                >
                  {shownErrors.length > 0 ? (
                    <>
                      <div className="mb-1 font-semibold">
                        {shownErrors.length === 1 ? '1 error' : `${shownErrors.length} errores`}
                      </div>
                      <ul className="list-disc space-y-0.5 pl-4">
                        {shownErrors.map((e, i) => (
                          <li key={i}>{e}</li>
                        ))}
                      </ul>
                    </>
                  ) : pending ? (
                    'Validando…'
                  ) : rawErrors.length > 0 ? (
                    'Hay errores en campos que aún no has tocado: revisa el formulario.'
                  ) : (
                    'Sin errores: la ficha es válida.'
                  )}
                </div>
              )
            )}

            {saveErrors.length > 0 && (
              <ul className="list-disc space-y-0.5 rounded-md border border-rose-900 bg-rose-950/60 py-1.5 pl-6 pr-2 text-xs text-rose-300">
                {saveErrors.map((e, i) => (
                  <li key={i}>{e}</li>
                ))}
              </ul>
            )}

            {hardBlock ? (
              <div className="rounded-md border border-amber-900 bg-amber-950/40 px-3 py-2 text-xs text-amber-300">
                <div className="mb-1 font-semibold">No se puede guardar</div>
                {/* PLAN-DESVIO (menor): el brief dice «si guardable == false, el motivo en vez del botón Guardar». Se aplica así solo
                    cuando el borrador es imposible de arreglar en el formulario (la hoja no trae una moneda USD: E6). Si solo faltan
                    datos (símbolo SQX, lote mínimo…) el botón Guardar sale, desactivado hasta que la validación en vivo pase, y el
                    motivo va arriba: con la regla literal, rellenar los `faltan` no llevaría a ningún sitio. */}
                <p>{draft?.motivo}</p>
              </div>
            ) : (
              <Button onClick={save} disabled={!canSave}>
                {saving
                  ? mode === 'new'
                    ? 'Creando…'
                    : 'Guardando…'
                  : mode === 'new'
                    ? 'Crear ficha'
                    : dirty
                      ? 'Guardar'
                      : 'Sin cambios'}
              </Button>
            )}
          </aside>
        </div>
      </CardBody>

      {confirmCancel && (
        <Modal title="Cambios sin guardar" onClose={() => setConfirmCancel(false)}>
          <p className="text-sm text-zinc-300">
            Hay cambios sin guardar en la ficha. Si sales ahora se pierden.
          </p>
          <div className="flex justify-end gap-2 pt-1">
            <SmallBtn onClick={() => setConfirmCancel(false)}>Seguir editando</SmallBtn>
            <SmallBtn
              tone="danger"
              onClick={() => {
                setConfirmCancel(false)
                onCancel()
              }}
            >
              Descartar los cambios
            </SmallBtn>
          </div>
        </Modal>
      )}
    </Card>
  )
}

function OriginNote({ preset, mode }: { preset?: FormPreset; mode: 'new' | 'edit' }) {
  let text: string | null = null
  if (mode === 'edit') {
    text =
      'Cambia los números que quieras. Los campos en ámbar siguen sin verificar: márcalos como verificados cuando los hayas comprobado.'
  } else if (preset?.origin === 'sqx') {
    text =
      'Borrador leído de tu .sqx: lo que trae el .sqx solo se propone y queda marcado como sin verificar hasta que lo confirmes. El lote mínimo no está en el .sqx: tecléalo tú. Elige también el bróker de la ficha.'
  } else if (preset?.origin === 'mt5_sheet') {
    text =
      'Borrador de la hoja de MT5. Revisa los datos y teclea los que faltan (la hoja exporta la comisión a 0 si no se tecleó). Nada se guarda hasta que pulses «Crear ficha».'
  } else if (preset?.origin === 'duplicate') {
    text =
      'Copia de una ficha. Cambia lo que quieras y ponle un símbolo SQX propio: tiene que ser igual que el de tus .sqx de ese bróker y distinto del original (el símbolo SQX es único entre brókers).'
  }
  if (!text) return null
  return <p className="rounded-md border border-zinc-800 bg-zinc-950/60 px-3 py-2 text-xs text-zinc-400">{text}</p>
}

// --- Borrador cuyo símbolo SQX YA tiene ficha: conflictos (E11) -------------------------------------

function ConflictRows({ conflicts, chosen, onToggle }: { conflicts: SymbolConflict[]; chosen?: Set<string>; onToggle?: (field: string, on: boolean) => void }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-zinc-800 text-left text-[11px] uppercase tracking-wide text-zinc-500">
            <th className="px-3 py-1.5 font-medium">Campo</th>
            <th className="px-3 py-1.5 font-medium">Actual</th>
            <th className="px-3 py-1.5 font-medium">Propuesto</th>
            <th className="px-3 py-1.5 font-medium">Fuente (actual → propuesta)</th>
            {onToggle && <th className="px-3 py-1.5 font-medium">Quedarme con el propuesto</th>}
          </tr>
        </thead>
        <tbody>
          {conflicts.map((c) => (
            <tr key={c.field} className="border-b border-zinc-800/60 align-top">
              <td className="px-3 py-1.5 text-zinc-300">{fieldLabel(c.field)}</td>
              <td className="num px-3 py-1.5 text-zinc-100">{fmtFieldValue(c.field, c.actual)}</td>
              <td className="num px-3 py-1.5 text-amber-300">{fmtFieldValue(c.field, c.propuesto)}</td>
              <td className="px-3 py-1.5">
                <div className="flex flex-wrap items-center gap-1.5">
                  <Badge variant="outline">{c.fuente_actual ? SOURCE_LABEL[c.fuente_actual] : '—'}</Badge>
                  <span className="text-zinc-500">→</span>
                  <Badge variant={c.fuente_propuesta === 'sqx' ? 'warn' : 'default'}>{SOURCE_LABEL[c.fuente_propuesta]}</Badge>
                </div>
              </td>
              {onToggle && (
                <td className="px-3 py-1.5">
                  <Checkbox
                    checked={chosen?.has(c.field) ?? false}
                    onChange={(on) => onToggle(c.field, on)}
                    label={
                      <span className="text-xs text-zinc-300">
                        Aceptar<span className="sr-only"> el valor propuesto de «{fieldLabel(c.field)}»</span>
                      </span>
                    }
                  />
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function ExistingDraftPanel({ preset, symbols, onSaved, onCancel, onShowHidden, cancelLabel = '← Volver' }: SymbolFormProps) {
  const draft = preset!.draft!
  const ex = draft.existing!
  const current = symbols.find((s) => s.broker === ex.broker && s.sqx_symbol === ex.sqx_symbol)
  const [showing, setShowing] = useState(false)
  // Solo una ficha TUYA, visible y del bróker de destino se puede tocar (el backend ya lo resolvió en `editable`).
  const canApply = draft.editable && !!current && !current.official
  const [chosen, setChosen] = useState<Set<string>>(new Set())
  const [saving, setSaving] = useState(false)
  const [errors, setErrors] = useState<string[]>([])

  const toggle = (field: string, on: boolean) =>
    setChosen((s) => {
      const n = new Set(s)
      if (on) n.add(field)
      else n.delete(field)
      return n
    })

  const apply = async () => {
    if (!current) return
    setSaving(true)
    setErrors([])
    try {
      // PUT con SOLO los campos elegidos, con la procedencia de lo que propone el borrador (hoja MT5 / .sqx). Lo que viene del
      // .sqx queda sin verificar; lo que se acepta de la hoja deja de estarlo (salvo lo que el borrador ya marcó).
      const body: Record<string, unknown> = {}
      const sources: Record<string, FieldSource> = {}
      for (const c of draft.conflicts) {
        if (!chosen.has(c.field)) continue
        body[c.field] = c.propuesto
        sources[c.field] = { from: c.fuente_propuesta, date: draft.sources[c.field]?.date ?? todayIso() }
      }
      body.sources = sources
      body.needs_review = [
        ...current.needs_review.filter((f) => !chosen.has(f)),
        ...[...chosen].filter((f) => draft.needs_review.includes(f)),
      ]
      onSaved(await updateSymbol(ex.broker, ex.sqx_symbol, body), 'updated')
    } catch (e) {
      setErrors(e instanceof SymbolApiError ? e.errors : [String(e)])
    } finally {
      setSaving(false)
    }
  }

  return (
    <Card>
      <CardHeader
        title={`${preset?.title ?? 'Borrador'} · ${draft.sqx_symbol ?? ''}`}
        subtitle={`El símbolo SQX ya tiene ficha en el bróker '${ex.broker}' (${ex.official ? 'oficial' : 'tuya'})`}
        right={
          <button
            onClick={onCancel}
            className="rounded-md border border-zinc-700 px-2.5 py-1 text-xs text-zinc-300 hover:bg-zinc-800"
          >
            {cancelLabel}
          </button>
        }
      />
      <CardBody className="space-y-4">
        {!canApply && draft.motivo && (
          <div className="rounded-md border border-amber-900 bg-amber-950/40 px-3 py-2 text-xs text-amber-300">
            <div className="mb-1 font-semibold">No se puede guardar</div>
            <p>{draft.motivo}</p>
          </div>
        )}
        {/* H9 (revisión de código de la F5): el backend dice que la ficha existente es editable (tuya, visible, del mismo bróker)
            pero no está en la lista de la vista (lista desincronizada): sin casillas ni motivo el usuario no sabría por qué. */}
        {draft.editable && !current && (
          <p className="rounded-md border border-amber-900 bg-amber-950/40 px-3 py-2 text-xs text-amber-300">
            No encuentro la ficha: recarga la lista.
          </p>
        )}
        {/* Ficha existente NO USABLE (p. ej. divisa no USD): es lo que hace que la pata salga «no usable». */}
        {current && !current.usable && (
          <p className="rounded-md border border-amber-900 bg-amber-950/40 px-3 py-2 text-xs text-amber-300">
            La ficha existente no se puede usar: {current.motivo ?? 'motivo no indicado'}.
          </p>
        )}
        {/* PLAN-DESVIO (adición): ficha OCULTA. Por eso las patas de este símbolo salen «sin ficha»; el texto del backend dice
            «ocúltala» (vale para una oficial visible) y aquí lo útil es mostrarla de nuevo: pista y botón «Mostrar la ficha». */}
        {current?.hidden && (
          <div className="flex flex-wrap items-center justify-between gap-3 rounded-md border border-zinc-700 bg-zinc-950/60 px-3 py-2 text-xs text-zinc-300">
            <span>
              La ficha de este símbolo está <span className="font-semibold text-zinc-100">oculta</span>: por eso las patas de este
              símbolo salen sin ficha. Muéstrala para volver a usarla.
            </span>
            {onShowHidden && (
              <SmallBtn
                tone="primary"
                disabled={showing}
                onClick={() => {
                  setShowing(true)
                  void onShowHidden(current).finally(() => setShowing(false))
                }}
              >
                {showing ? 'Mostrando…' : 'Mostrar la ficha'}
              </SmallBtn>
            )}
          </div>
        )}
        {draft.warnings.filter((w) => w !== draft.motivo).length > 0 && (
          <ul className="list-disc space-y-0.5 rounded-md border border-amber-900 bg-amber-950/40 py-2 pl-7 pr-3 text-xs text-amber-300">
            {draft.warnings
              .filter((w) => w !== draft.motivo)
              .map((w, i) => (
                <li key={i}>{w}</li>
              ))}
          </ul>
        )}
        {draft.conflicts.length === 0 ? (
          <p className="text-sm text-zinc-400">
            Los datos del borrador coinciden con la ficha existente: no hay nada que cambiar.
          </p>
        ) : (
          <>
            <ConflictRows conflicts={draft.conflicts} chosen={canApply ? chosen : undefined} onToggle={canApply ? toggle : undefined} />
            {!canApply && (
              <ul className="list-disc space-y-1 pl-5 text-xs text-zinc-400">
                {draft.conflicts.map((c) => (
                  <li key={c.field}>{c.texto}</li>
                ))}
              </ul>
            )}
          </>
        )}
        {canApply && draft.conflicts.length > 0 && (
          <div className="space-y-2">
            <p className="text-xs text-zinc-500">
              Marca los campos en los que prefieres el valor propuesto: se aplican a tu ficha, con su procedencia. Lo que no
              marques se queda como está.
            </p>
            {errors.length > 0 && (
              <ul className="list-disc space-y-0.5 rounded-md border border-rose-900 bg-rose-950/60 py-1.5 pl-6 pr-2 text-xs text-rose-300">
                {errors.map((e, i) => (
                  <li key={i}>{e}</li>
                ))}
              </ul>
            )}
            <div className="md:w-[320px]">
              <Button onClick={apply} disabled={chosen.size === 0 || saving}>
                {saving ? 'Aplicando…' : chosen.size === 0 ? 'Elige algún campo' : `Aplicar ${chosen.size} campo${chosen.size > 1 ? 's' : ''} a mi ficha`}
              </Button>
            </div>
          </div>
        )}
      </CardBody>
    </Card>
  )
}

export function SymbolForm(props: SymbolFormProps) {
  // Un borrador cuyo símbolo SQX ya tiene ficha no se edita: se resuelve campo a campo (o se explica por qué no se puede).
  if (props.mode === 'new' && props.preset?.draft?.existing) return <ExistingDraftPanel {...props} />
  return <SymbolFormBody {...props} />
}
