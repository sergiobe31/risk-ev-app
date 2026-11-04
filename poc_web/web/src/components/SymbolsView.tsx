// Vista "Símbolos" (guardrail entrada-4-bloques, F5): fichas de símbolo por bróker. ÁRBOL + DETALLE, con el mismo
// esquema que la vista Perfiles. Columna izquierda: acciones («+ Bróker», «+ Símbolo», «Importar hoja de MT5»,
// «Importar fichas»), los errores del fichero de fichas, el árbol bróker -> símbolos (badge «oficial» / «tuyo») y la
// sección plegada «Ocultos (n)». Columna derecha: el detalle de la ficha elegida (SymbolDetail), el formulario
// (SymbolForm) o la importación de la hoja de MT5 (SymbolImportDialog).
// Ocultar != borrar: las oficiales se ocultan y se restauran (no se editan); las tuyas además se editan y se borran.
// Toda confirmación va con Modal (ningún diálogo nativo del navegador). El backend es poc_web/api/symbols.py.
//
// Montada SIEMPRE por App (oculta fuera de su vista): conserva el árbol, la selección y el formulario abierto. Tras
// cada cambio avisa a App (`onSymbolsChanged`) para recargar el catálogo, porque el estado `ficha` de las patas cambia.
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import {
  ASSET_LABEL,
  CLOCK_LABEL,
  createBroker,
  deleteSymbol,
  exportSymbol,
  fetchDraftFromSqx,
  fetchSymbolSchema,
  importSymbol,
  listSymbols,
  setSymbolsHidden,
  symbolErrText,
  symbolKey,
  type AssetClass,
  type BrokerView,
  type ClockId,
  type SymbolDraft,
  type SymbolSchema,
  type SymbolsList,
  type SymbolView,
} from '../lib/api'
import { Badge, Card, CardBody, CardHeader, Field, Modal, Select, SmallBtn, TextInput } from './ui'
import { SymbolDetail } from './SymbolDetail'
import { SymbolForm, type FormPreset } from './SymbolForm'
import { SymbolImportDialog } from './SymbolImportDialog'

type Notice = { tone: 'ok' | 'error'; text: string }
interface Confirm {
  title: string
  text: string
  confirmLabel: string
  tone: 'primary' | 'danger'
  run: () => Promise<void>
}
// Panel de la columna derecha. `id` cambia en cada apertura: remonta el formulario (estado limpio).
type Pane =
  | { kind: 'detail' }
  | { kind: 'edit'; id: number; sheet: SymbolView }
  | { kind: 'new'; id: number; preset: FormPreset }
  | { kind: 'import'; id: number }

/** Petición de «Dar de alta» desde una pata .sqx sin ficha (App la manda; `nonce` distingue peticiones de la misma ruta). */
export interface DraftRequest {
  path: string
  nonce: number
}

const BROKER_ID_RE = /^[a-z0-9_]{3,40}$/
const ID_HINT = 'El id debe cumplir ^[a-z0-9_]{3,40}$ (minúsculas, dígitos y _; 3 a 40 caracteres)'

const sqxPreset = (d: SymbolDraft, path: string): FormPreset => ({
  origin: 'sqx',
  title: 'Alta desde un .sqx',
  subtitle: path.split(/[\\/]/).pop() ?? path,
  draft: d,
})

function Chevron({ open }: { open: boolean }) {
  return <span className={`inline-block text-zinc-500 transition-transform ${open ? 'rotate-90' : ''}`}>▸</span>
}

function SymbolRow({
  s,
  selected,
  onSelect,
  right,
  showBroker = false,
}: {
  s: SymbolView
  selected: boolean
  onSelect: () => void
  right?: ReactNode
  showBroker?: boolean
}) {
  return (
    <div
      className={`flex items-center gap-1.5 border-l-2 px-2 py-1 ${
        selected ? 'border-emerald-500 bg-emerald-950/40' : 'border-transparent hover:bg-zinc-800/50'
      }`}
    >
      <button
        type="button"
        onClick={onSelect}
        aria-current={selected ? 'true' : undefined}
        className="flex min-w-0 flex-1 flex-wrap items-center gap-1.5 text-left text-xs"
      >
        <span className={`truncate ${selected ? 'text-emerald-200' : 'text-zinc-200'}`}>{s.sqx_symbol}</span>
        {s.official ? <Badge variant="outline">oficial</Badge> : <Badge variant="success">tuyo</Badge>}
        {s.needs_review.length > 0 && <Badge variant="warn">sin verificar</Badge>}
        {!s.usable && <Badge variant="danger">no usable</Badge>}
        {showBroker && <span className="shrink-0 text-zinc-500">· {s.broker}</span>}
      </button>
      {right}
    </div>
  )
}

function SymbolTree({
  tree,
  hiddenList,
  selectedKey,
  onSelect,
  expanded,
  onToggleBroker,
  hiddenOpen,
  onToggleHidden,
  onShow,
  busy,
}: {
  tree: { broker: BrokerView; items: SymbolView[] }[]
  hiddenList: SymbolView[]
  selectedKey: string | null
  onSelect: (s: SymbolView) => void
  expanded: Set<string>
  onToggleBroker: (id: string) => void
  hiddenOpen: boolean
  onToggleHidden: () => void
  onShow: (keys: string[]) => void
  busy: boolean
}) {
  return (
    <div className="max-h-[75vh] space-y-2 overflow-y-auto pr-0.5">
      {tree.map(({ broker, items }) => {
        const open = expanded.has(broker.id)
        return (
          <div key={broker.id} className="overflow-hidden rounded-lg border border-zinc-800 bg-zinc-950/40">
            <button
              type="button"
              onClick={() => onToggleBroker(broker.id)}
              aria-expanded={open}
              className="flex w-full items-center gap-2 px-2.5 py-2 text-left"
            >
              <Chevron open={open} />
              <span className="min-w-0 flex-1">
                <span className="flex flex-wrap items-center gap-1.5">
                  <span className="truncate text-sm font-medium text-zinc-100">{broker.display_name}</span>
                  <span className="num text-xs text-zinc-500">({items.length})</span>
                  {broker.official ? <Badge variant="outline">oficial</Badge> : <Badge variant="success">tuyo</Badge>}
                </span>
                <span className="block truncate text-[11px] text-zinc-500">
                  {ASSET_LABEL[broker.asset_class]} · {CLOCK_LABEL[broker.clock]}
                </span>
              </span>
            </button>
            {open && (
              <div className="divide-y divide-zinc-800/60 border-t border-zinc-800">
                {items.length === 0 && <p className="px-3 py-2 text-xs text-zinc-500">Este bróker aún no tiene fichas.</p>}
                {items.map((s) => (
                  <SymbolRow
                    key={s.sqx_symbol}
                    s={s}
                    selected={symbolKey(s.broker, s.sqx_symbol) === selectedKey}
                    onSelect={() => onSelect(s)}
                  />
                ))}
              </div>
            )}
          </div>
        )
      })}

      {hiddenList.length > 0 && (
        <div className="overflow-hidden rounded-lg border border-zinc-800 bg-zinc-950/40">
          <div className="flex items-center gap-2 px-2.5 py-2">
            <button
              type="button"
              onClick={onToggleHidden}
              aria-expanded={hiddenOpen}
              className="flex min-w-0 flex-1 items-center gap-2 text-left text-sm font-medium text-zinc-400"
            >
              <Chevron open={hiddenOpen} />
              Ocultos ({hiddenList.length})
            </button>
            {hiddenOpen && (
              <SmallBtn onClick={() => onShow(hiddenList.map((s) => symbolKey(s.broker, s.sqx_symbol)))} disabled={busy}>
                Mostrar todos
              </SmallBtn>
            )}
          </div>
          {hiddenOpen && (
            <div className="divide-y divide-zinc-800/60 border-t border-zinc-800">
              {hiddenList.map((s) => (
                <SymbolRow
                  key={symbolKey(s.broker, s.sqx_symbol)}
                  s={s}
                  showBroker
                  selected={symbolKey(s.broker, s.sqx_symbol) === selectedKey}
                  onSelect={() => onSelect(s)}
                  right={
                    <SmallBtn onClick={() => onShow([symbolKey(s.broker, s.sqx_symbol)])} disabled={busy}>
                      Mostrar
                    </SmallBtn>
                  }
                />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// Alta de un bróker propio (POST /api/symbols/brokers). Campos y etiquetas salen de `broker_fields` del schema.
function BrokerModal({
  schema,
  onClose,
  onCreated,
}: {
  schema: SymbolSchema | null
  onClose: () => void
  onCreated: (b: BrokerView) => Promise<void>
}) {
  const [id, setId] = useState('')
  const [name, setName] = useState('')
  const [company, setCompany] = useState('')
  const [assetClass, setAssetClass] = useState<AssetClass>('cfd')
  const [clock, setClock] = useState<ClockId>('ny_plus_7')
  const [ohlcDir, setOhlcDir] = useState('')
  const [saving, setSaving] = useState(false)
  const [errors, setErrors] = useState<string[]>([])
  const label = (n: string, fallback: string) => schema?.broker_fields.find((f) => f.name === n)?.label ?? fallback
  const pattern = schema?.broker_fields.find((f) => f.name === 'id')?.pattern
  const idOk = (pattern ? new RegExp(pattern) : BROKER_ID_RE).test(id)
  const enumOpts = (kind: 'asset_class' | 'clock', fallback: Record<string, string>) =>
    (schema?.enums[kind] ?? Object.entries(fallback).map(([i, l]) => ({ id: i, label: l }))).map((o) => ({
      value: o.id,
      label: o.label,
    }))
  const canSave = idOk && name.trim() !== '' && company.trim() !== '' && !saving

  const submit = async () => {
    setSaving(true)
    setErrors([])
    try {
      const b = await createBroker({
        id,
        display_name: name.trim(),
        company: company.trim(),
        currency: 'USD',
        asset_class: assetClass,
        clock,
        ...(ohlcDir.trim() ? { ohlc_dir: ohlcDir.trim() } : {}),
      })
      await onCreated(b)
    } catch (e) {
      setErrors([symbolErrText(e)])
    } finally {
      setSaving(false)
    }
  }

  return (
    <Modal title="Nuevo bróker" onClose={() => !saving && onClose()}>
      <p className="text-xs text-zinc-400">
        Un bróker agrupa las fichas de tus símbolos y fija su clase de activo, el reloj de su servidor y la carpeta de sus
        OHLC de 1 minuto. Las fichas oficiales no se editan: para cambiarlas, duplícalas en un bróker tuyo.
      </p>
      <Field label={label('id', 'Id')} hint="Minúsculas, dígitos y _ (3 a 40 caracteres)">
        <TextInput value={id} onChange={setId} placeholder="p. ej. mi_broker" />
      </Field>
      {id !== '' && !idOk && <p className="text-xs text-rose-300">{ID_HINT}</p>}
      <Field label={label('display_name', 'Nombre')}>
        <TextInput value={name} onChange={setName} placeholder="p. ej. Mi bróker (MT5)" />
      </Field>
      <Field label={label('company', 'Compañía')}>
        <TextInput value={company} onChange={setCompany} placeholder="p. ej. MiBroker" />
      </Field>
      <div className="grid grid-cols-2 gap-3">
        <Field label={label('asset_class', 'Clase de activo')}>
          <Select<AssetClass> value={assetClass} onChange={setAssetClass} options={enumOpts('asset_class', ASSET_LABEL) as { value: AssetClass; label: string }[]} />
        </Field>
        <Field label={label('currency', 'Divisa de la cuenta')} hint="La v1 solo admite cuentas en USD">
          <div className="w-full rounded-md border border-zinc-800 bg-zinc-950/60 px-2 py-1.5 text-sm text-zinc-300">USD</div>
        </Field>
      </div>
      <Field
        label={label('clock', 'Reloj del servidor')}
        hint="Casi todos los MT5 (también FTMO y AXI) marcan Nueva York + 7 h; los datos de futuros de SQX, el reloj de pared de Nueva York"
      >
        <Select<ClockId> value={clock} onChange={setClock} options={enumOpts('clock', CLOCK_LABEL) as { value: ClockId; label: string }[]} />
      </Field>
      <Field
        label={`${label('ohlc_dir', 'Carpeta de OHLC')} (opcional)`}
        hint="Ruta relativa a la raíz del repo con los OHLC de 1 minuto de este bróker. Sin ella no hay flotante por minuto."
      >
        <TextInput value={ohlcDir} onChange={setOhlcDir} placeholder="p. ej. risk_app/inputs/ohlc_mi_broker" />
      </Field>
      {errors.length > 0 && (
        <p className="rounded-md border border-rose-900 bg-rose-950/60 px-2 py-1.5 text-xs text-rose-300">{errors.join(' · ')}</p>
      )}
      <div className="flex justify-end gap-2 pt-1">
        <SmallBtn onClick={onClose} disabled={saving}>
          Cancelar
        </SmallBtn>
        <SmallBtn tone="primary" onClick={() => void submit()} disabled={!canSave}>
          {saving ? 'Creando…' : 'Crear bróker'}
        </SmallBtn>
      </div>
    </Modal>
  )
}

export function SymbolsView({
  onSymbolsChanged,
  draftRequest,
  enabled,
}: {
  // App vuelve a pedir el catálogo EN EL MOMENTO (el estado `ficha` de las patas cambia); se resuelve al terminar
  onSymbolsChanged: () => Promise<void>
  draftRequest: DraftRequest | null
  // H10 (revisión de código de la F5): la vista está montada siempre (conserva su estado) pero no pide las fichas ni el schema hasta
  // que App dice que se abre por primera vez (la pestaña, o un «Dar de alta» desde el selector de patas).
  enabled: boolean
}) {
  const [data, setData] = useState<SymbolsList | null>(null)
  const [schema, setSchema] = useState<SymbolSchema | null>(null)
  const [schemaError, setSchemaError] = useState<string | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [notice, setNotice] = useState<Notice | null>(null)
  const [pane, setPane] = useState<Pane>({ kind: 'detail' })
  const [selectedKey, setSelectedKey] = useState<string | null>(null)
  const [expanded, setExpanded] = useState<Set<string>>(new Set()) // ids de bróker abiertos; vacío = todo plegado
  const [hiddenOpen, setHiddenOpen] = useState(false)
  const [confirm, setConfirm] = useState<Confirm | null>(null)
  const [leave, setLeave] = useState<(() => void) | null>(null)
  const [brokerModal, setBrokerModal] = useState(false)
  const [busy, setBusy] = useState(false)
  const [draftDone, setDraftDone] = useState(0) // nonce de la última petición de borrador ya resuelta
  const dirtyRef = useRef(false) // el formulario abierto tiene cambios sin guardar
  const idRef = useRef(0)
  const fileRef = useRef<HTMLInputElement>(null)

  const reload = useCallback(
    () =>
      listSymbols()
        .then((d) => {
          setData(d)
          setLoadError(null)
        })
        .catch((e) => setLoadError(symbolErrText(e))),
    [],
  )

  // tras un cambio: catálogo (App) y lista, en paralelo
  const refresh = async () => {
    await Promise.all([onSymbolsChanged().catch(() => undefined), reload()])
  }

  useEffect(() => {
    if (!enabled) return
    let alive = true
    listSymbols()
      .then((d) => alive && setData(d))
      .catch((e) => alive && setLoadError(symbolErrText(e)))
    fetchSymbolSchema()
      .then((s) => alive && setSchema(s))
      .catch((e) => alive && setSchemaError(symbolErrText(e)))
    return () => {
      alive = false
    }
  }, [enabled])

  const brokers = useMemo(() => data?.brokers ?? [], [data])
  const symbols = useMemo(() => data?.symbols ?? [], [data])
  const selected = selectedKey ? (symbols.find((s) => symbolKey(s.broker, s.sqx_symbol) === selectedKey) ?? null) : null
  const tree = useMemo(() => {
    const by = new Map<string, SymbolView[]>()
    for (const s of symbols) {
      if (s.hidden) continue
      if (!by.has(s.broker)) by.set(s.broker, [])
      by.get(s.broker)!.push(s)
    }
    // oficiales primero, después los tuyos (orden del API dentro de cada bloque); símbolos por nombre
    const ordered = [...brokers.filter((b) => b.official), ...brokers.filter((b) => !b.official)]
    return ordered.map((broker) => ({
      broker,
      items: (by.get(broker.id) ?? []).sort((a, b) => a.sqx_symbol.localeCompare(b.sqx_symbol, 'es')),
    }))
  }, [brokers, symbols])
  const hiddenList = useMemo(() => symbols.filter((s) => s.hidden), [symbols])

  const onDirty = useCallback((d: boolean) => {
    dirtyRef.current = d
  }, [])

  // PLAN-DESVIO (adición): si hay un formulario con cambios sin guardar, pide confirmación (Modal) antes de salir de él. El brief solo
  // exige un Modal para borrar; ProfilesView protege así el formulario abierto (con el diálogo nativo del navegador) y aquí se hace igual, con Modal.
  const guard = useCallback((action: () => void) => {
    if (!dirtyRef.current) action()
    else setLeave(() => action) // () => action: setState con una función la ejecutaría como actualizador
  }, [])

  const reveal = (s: SymbolView) => {
    setExpanded((x) => new Set(x).add(s.broker))
    if (s.hidden) setHiddenOpen(true)
    setSelectedKey(symbolKey(s.broker, s.sqx_symbol))
  }

  const select = (s: SymbolView) =>
    guard(() => {
      setSelectedKey(symbolKey(s.broker, s.sqx_symbol))
      setPane({ kind: 'detail' })
      setNotice(null)
    })

  const toggleBroker = (id: string) =>
    setExpanded((x) => {
      const n = new Set(x)
      if (n.has(id)) n.delete(id)
      else n.add(id)
      return n
    })

  // --- «Dar de alta» desde una pata .sqx sin ficha (App manda la ruta) ---------------------------
  const draftPath = draftRequest?.path ?? null
  const draftNonce = draftRequest?.nonce ?? 0
  useEffect(() => {
    if (!draftPath) return
    let alive = true
    fetchDraftFromSqx(draftPath)
      .then((d) => {
        if (!alive) return
        setNotice(null)
        guard(() => setPane({ kind: 'new', id: ++idRef.current, preset: sqxPreset(d, draftPath) }))
        setDraftDone(draftNonce)
      })
      .catch((e) => {
        if (!alive) return
        setNotice({ tone: 'error', text: `No se pudo leer el borrador del .sqx: ${symbolErrText(e)}` })
        setDraftDone(draftNonce)
      })
    return () => {
      alive = false
    }
  }, [draftPath, draftNonce, guard])
  const draftLoading = draftNonce !== 0 && draftDone !== draftNonce

  // --- Acciones ------------------------------------------------------------------------------
  const doHide = async (keys: string[], hidden: boolean, okText: string) => {
    setBusy(true)
    try {
      await setSymbolsHidden(keys, hidden)
      await refresh()
      setNotice({ tone: 'ok', text: okText })
    } catch (e) {
      setNotice({ tone: 'error', text: symbolErrText(e) })
    } finally {
      setBusy(false)
    }
  }

  const askDelete = (s: SymbolView) =>
    setConfirm({
      title: 'Borrar ficha',
      text: `¿Borrar la ficha ${s.sqx_symbol} del bróker ${s.broker}? No se puede deshacer (exporta la ficha antes si quieres conservarla) y las patas .sqx de este símbolo dejarán de cargarse hasta que la des de alta otra vez.`,
      confirmLabel: 'Borrar',
      tone: 'danger',
      run: async () => {
        setBusy(true)
        try {
          await deleteSymbol(s.broker, s.sqx_symbol)
          if (selectedKey === symbolKey(s.broker, s.sqx_symbol)) setSelectedKey(null)
          setPane({ kind: 'detail' })
          await refresh()
          setNotice({ tone: 'ok', text: `Ficha '${s.sqx_symbol}' borrada.` })
        } catch (e) {
          setNotice({ tone: 'error', text: symbolErrText(e) })
        } finally {
          setBusy(false)
        }
      },
    })

  const runConfirm = async () => {
    const c = confirm
    setConfirm(null)
    if (c) await c.run()
  }

  const doExport = async (s: SymbolView) => {
    try {
      const env = await exportSymbol(s.broker, s.sqx_symbol)
      const blob = new Blob([JSON.stringify(env, null, 2) + '\n'], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${s.broker}__${s.sqx_symbol.replace(/[^A-Za-z0-9_.@-]/g, '_')}.json`
      document.body.appendChild(a)
      a.click()
      a.remove()
      URL.revokeObjectURL(url)
      setNotice({ tone: 'ok', text: `Exportada la ficha ${s.sqx_symbol} (${a.download}).` })
    } catch (e) {
      setNotice({ tone: 'error', text: `No se pudo exportar: ${symbolErrText(e)}` })
    }
  }

  const doImportFile = async (file: File) => {
    setBusy(true)
    try {
      const res = await importSymbol(await file.text())
      await refresh()
      reveal(res.sheet)
      setPane({ kind: 'detail' })
      setNotice({
        tone: 'ok',
        text: `Ficha '${res.sheet.sqx_symbol}' importada en el bróker '${res.sheet.broker}'${
          res.broker_created ? ' (se ha creado también el bróker)' : ''
        }.`,
      })
    } catch (e) {
      setNotice({ tone: 'error', text: `No se pudo importar ${file.name}: ${symbolErrText(e)}` })
    } finally {
      setBusy(false)
      if (fileRef.current) fileRef.current.value = ''
    }
  }

  const openNew = () =>
    guard(() => {
      setNotice(null)
      setPane({
        kind: 'new',
        id: ++idRef.current,
        preset: { origin: 'manual', title: 'Nueva ficha', broker: selected?.broker ?? [...expanded][0] },
      })
    })

  const openEdit = (s: SymbolView) =>
    guard(() => {
      setNotice(null)
      setPane({ kind: 'edit', id: ++idRef.current, sheet: s })
    })

  const openDuplicate = (s: SymbolView) => {
    const eligible = brokers.filter((b) => !b.official && b.asset_class === s.asset_class)
    if (eligible.length === 0) {
      setNotice({
        tone: 'error',
        text: `Para duplicar la ficha necesitas un bróker tuyo de tipo ${ASSET_LABEL[s.asset_class]}: créalo antes con «+ Bróker».`,
      })
      return
    }
    guard(() => {
      setNotice(null)
      setPane({
        kind: 'new',
        id: ++idRef.current,
        preset: {
          origin: 'duplicate',
          title: `Duplicar ${s.sqx_symbol}`,
          subtitle: `copia de la ficha ${s.broker}/${s.sqx_symbol}`,
          copyOf: s,
          broker: eligible[0].id,
          brokerFilter: (b) => !b.official && b.asset_class === s.asset_class,
        },
      })
    })
  }

  const openImport = () =>
    guard(() => {
      setNotice(null)
      setPane({ kind: 'import', id: ++idRef.current })
    })

  const onFormSaved = async (view: SymbolView, how: 'created' | 'updated') => {
    await refresh()
    reveal(view)
    setPane({ kind: 'detail' })
    setNotice({
      tone: 'ok',
      text:
        how === 'created'
          ? `Ficha '${view.sqx_symbol}' creada en el bróker '${view.broker}'. Vuelve al Simulador y recarga la cartera para usarla.`
          : `Ficha '${view.sqx_symbol}' guardada.`,
    })
  }

  // Desde la importación de la hoja: se refresca y se avisa, pero se sigue en la tabla de borradores.
  const onImportSaved = async (view: SymbolView, how: 'created' | 'updated') => {
    await refresh()
    setNotice({
      tone: 'ok',
      text:
        how === 'created'
          ? `Ficha '${view.sqx_symbol}' creada en el bróker '${view.broker}'.`
          : `Ficha '${view.sqx_symbol}' actualizada con los campos elegidos.`,
    })
  }

  // --- Columna derecha -----------------------------------------------------------------------
  let right: ReactNode
  if (pane.kind === 'import') {
    right = (
      <SymbolImportDialog
        key={pane.id}
        brokers={brokers}
        schema={schema}
        schemaError={schemaError}
        symbols={symbols}
        defaultBroker={selected?.broker}
        onSaved={onImportSaved}
        onClose={() => setPane({ kind: 'detail' })}
        onDirtyChange={onDirty}
        onNewBroker={() => setBrokerModal(true)}
      />
    )
  } else if (pane.kind === 'new') {
    right = (
      <SymbolForm
        key={pane.id}
        schema={schema}
        schemaError={schemaError}
        brokers={brokers}
        symbols={symbols}
        mode="new"
        preset={pane.preset}
        onSaved={onFormSaved}
        onCancel={() => setPane({ kind: 'detail' })}
        onDirtyChange={onDirty}
        onShowHidden={async (s) => {
          await doHide([symbolKey(s.broker, s.sqx_symbol)], false, `'${s.sqx_symbol}' vuelve a estar visible.`)
          reveal({ ...s, hidden: false })
          setPane({ kind: 'detail' })
        }}
      />
    )
  } else if (pane.kind === 'edit') {
    right = (
      <SymbolForm
        key={pane.id}
        schema={schema}
        schemaError={schemaError}
        brokers={brokers}
        symbols={symbols}
        mode="edit"
        sheet={pane.sheet}
        cancelLabel="← Volver al detalle"
        onSaved={onFormSaved}
        onCancel={() => setPane({ kind: 'detail' })}
        onDirtyChange={onDirty}
      />
    )
  } else {
    right = (
      <SymbolDetail
        sheet={selected}
        broker={selected ? brokers.find((b) => b.id === selected.broker) : undefined}
        busy={busy}
        onEdit={() => selected && openEdit(selected)}
        onDuplicate={() => selected && openDuplicate(selected)}
        onExport={() => selected && void doExport(selected)}
        onToggleHidden={() =>
          selected &&
          void doHide(
            [symbolKey(selected.broker, selected.sqx_symbol)],
            !selected.hidden,
            selected.hidden ? `'${selected.sqx_symbol}' vuelve a estar visible.` : `'${selected.sqx_symbol}' oculta.`,
          )
        }
        onDelete={() => selected && askDelete(selected)}
      />
    )
  }

  return (
    <div className="space-y-5">
      <Card>
        <CardHeader
          title="Fichas de símbolo"
          subtitle="Las especificaciones de cada símbolo en su bróker: valor del punto, pasos de lote, comisión y spread. Las oficiales son de solo lectura (se ocultan o se duplican); las tuyas se editan. El .sqx solo propone valores: los marca como sin verificar hasta que los confirmas."
        />
      </Card>

      {notice && (
        <Card className={notice.tone === 'ok' ? 'border-emerald-900 bg-emerald-950/30' : 'border-rose-900 bg-rose-950/40'}>
          <CardBody
            className={`flex items-start justify-between gap-3 py-2.5 text-xs ${
              notice.tone === 'ok' ? 'text-emerald-300' : 'text-rose-300'
            }`}
          >
            <span>{notice.text}</span>
            <button onClick={() => setNotice(null)} className="text-zinc-400 hover:text-zinc-200" aria-label="Cerrar aviso">
              ✕
            </button>
          </CardBody>
        </Card>
      )}

      {draftLoading && (
        <Card>
          <CardBody className="py-2.5 text-xs text-zinc-400">Leyendo el .sqx para proponer la ficha…</CardBody>
        </Card>
      )}

      {loadError && (
        <Card className="border-rose-900 bg-rose-950/40">
          <CardBody className="py-2.5 text-xs text-rose-300">No se pudieron cargar las fichas: {loadError}</CardBody>
        </Card>
      )}

      {data && data.errors.length > 0 && (
        <Card className="border-amber-900 bg-amber-950/30">
          <CardBody className="space-y-1.5 py-3 text-xs text-amber-300">
            <div className="font-semibold">
              Hay {data.errors.length === 1 ? 'un error' : `${data.errors.length} errores`} al cargar las fichas (lo
              afectado no se usa)
            </div>
            <ul className="list-disc space-y-0.5 pl-5">
              {data.errors.map((e, i) => (
                <li key={i}>{e}</li>
              ))}
            </ul>
          </CardBody>
        </Card>
      )}

      {!data && !loadError && <p className="text-sm text-zinc-500">Cargando fichas…</p>}

      {data && (
        <div className="grid grid-cols-1 items-start gap-5 lg:grid-cols-[340px_minmax(0,1fr)]">
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-1.5">
              <SmallBtn tone="primary" onClick={() => setBrokerModal(true)} disabled={busy}>
                + Bróker
              </SmallBtn>
              <SmallBtn tone="primary" onClick={openNew} disabled={busy}>
                + Símbolo
              </SmallBtn>
              <SmallBtn onClick={openImport} disabled={busy} title="Lee el CSV que exporta el script de SQX para MT5">
                Importar hoja de MT5
              </SmallBtn>
              <SmallBtn onClick={() => fileRef.current?.click()} disabled={busy} title="Importa una ficha exportada (JSON)">
                Importar fichas
              </SmallBtn>
              <input
                ref={fileRef}
                type="file"
                accept=".json,application/json"
                className="hidden"
                aria-label="Importar fichas (JSON)"
                onChange={(e) => {
                  const f = e.target.files?.[0]
                  if (f) void doImportFile(f)
                }}
              />
            </div>
            <SymbolTree
              tree={tree}
              hiddenList={hiddenList}
              selectedKey={selectedKey}
              onSelect={select}
              expanded={expanded}
              onToggleBroker={toggleBroker}
              hiddenOpen={hiddenOpen}
              onToggleHidden={() => setHiddenOpen((v) => !v)}
              onShow={(keys) =>
                void doHide(keys, false, keys.length === 1 ? `'${keys[0].split('/')[1]}' vuelve a estar visible.` : `${keys.length} fichas restauradas.`)
              }
              busy={busy}
            />
          </div>
          <div className="min-w-0">{right}</div>
        </div>
      )}

      {brokerModal && (
        <BrokerModal
          schema={schema}
          onClose={() => setBrokerModal(false)}
          onCreated={async (b) => {
            setBrokerModal(false)
            await refresh()
            setExpanded((x) => new Set(x).add(b.id))
            setNotice({
              tone: 'ok',
              text: `Bróker '${b.id}' creado. Añade sus símbolos con «+ Símbolo» o «Importar hoja de MT5».`,
            })
          }}
        />
      )}

      {confirm && (
        <Modal title={confirm.title} onClose={() => setConfirm(null)}>
          <p className="text-sm text-zinc-300">{confirm.text}</p>
          <div className="flex justify-end gap-2 pt-1">
            <SmallBtn onClick={() => setConfirm(null)}>Cancelar</SmallBtn>
            <SmallBtn tone={confirm.tone} onClick={() => void runConfirm()}>
              {confirm.confirmLabel}
            </SmallBtn>
          </div>
        </Modal>
      )}

      {leave && (
        <Modal title="Formulario abierto" onClose={() => setLeave(null)}>
          <p className="text-sm text-zinc-300">Hay un formulario con cambios sin guardar. Si sigues, se descartan.</p>
          <div className="flex justify-end gap-2 pt-1">
            <SmallBtn onClick={() => setLeave(null)}>Seguir editando</SmallBtn>
            <SmallBtn
              tone="danger"
              onClick={() => {
                const go = leave
                dirtyRef.current = false
                setLeave(null)
                go()
              }}
            >
              Descartar y continuar
            </SmallBtn>
          </div>
        </Modal>
      )}
    </div>
  )
}
