// Importación de la hoja del script de SQX para MT5 (guardrail entrada-4-bloques, F5; plan §2.2): columna derecha
// de la vista Símbolos. Elegir el CSV de la hoja -> elegir el bróker destino -> POST /api/symbols/mt5-sheet ->
// tabla de BORRADORES. Cada fila: ticker, símbolo SQX propuesto (editable; los `candidates` salen como sugerencias),
// estado (guardable o el motivo), lo que falta y los conflictos con una ficha existente. «Revisar y guardar» abre el
// SymbolForm con el borrador. Nada se guarda sin que el usuario lo confirme ahí.
//
// La hoja se vuelve a leer (con `mapping` = los símbolos SQX que el usuario ha tecleado) cada vez que cambia un
// símbolo: es el backend quien dice si la ficha ya existe, si hay conflictos y si el borrador es guardable.
import { useEffect, useMemo, useRef, useState } from 'react'
import {
  fieldLabel,
  parseMt5Sheet,
  symbolErrText,
  type BrokerView,
  type Mt5SheetResult,
  type SymbolDraft,
  type SymbolSchema,
  type SymbolView,
} from '../lib/api'
import { Badge, Card, CardBody, CardHeader, Field, Select, SmallBtn } from './ui'
import { SymbolForm, type FormPreset } from './SymbolForm'

const DEBOUNCE_MS = 350
const inputCls =
  'w-full rounded-md border bg-zinc-950 px-2 py-1 text-sm text-zinc-100 placeholder:text-zinc-600 ' +
  'focus:outline-none'
// 1 a 60 caracteres sin espacios ni barras (el backend valida el resto: SQX_SYMBOL_RE de symbols.py)
const SQX_OK = /^[^\s/\\]{1,60}$/

interface Outcome {
  sig: string // petición a la que responde (descarta las respuestas viejas)
  data?: Mt5SheetResult
  error?: string
}

export function SymbolImportDialog({
  brokers,
  schema,
  schemaError,
  symbols,
  defaultBroker,
  onSaved,
  onClose,
  onDirtyChange,
  onNewBroker,
}: {
  brokers: BrokerView[]
  schema: SymbolSchema | null
  schemaError?: string | null
  symbols: SymbolView[]
  defaultBroker?: string
  onSaved: (view: SymbolView, how: 'created' | 'updated') => void // se ha guardado una ficha desde aquí
  onClose: () => void
  onDirtyChange?: (dirty: boolean) => void
  onNewBroker?: () => void // abre el alta de bróker («+ Bróker»); H8: sin brókeres CFD no hay dónde guardar
}) {
  const cfdBrokers = useMemo(() => brokers.filter((b) => b.asset_class === 'cfd'), [brokers])
  // Bróker destino: lo elegido, si sigue siendo uno de los CFD; si no, el de la vista o el primero. Derivado (no se copia a un estado
  // aparte) para que, si aparece un bróker CFD con el diálogo abierto (H8), ya quede elegido.
  const [brokerPick, setBrokerPick] = useState<string>(defaultBroker ?? '')
  const brokerId = cfdBrokers.some((b) => b.id === brokerPick) ? brokerPick : (cfdBrokers[0]?.id ?? '')
  const [file, setFile] = useState<{ name: string; text: string; seq: number } | null>(null)
  const [fileError, setFileError] = useState<string | null>(null)
  const [mapping, setMapping] = useState<Record<string, string>>({}) // ticker -> símbolo SQX tecleado
  const [outcome, setOutcome] = useState<Outcome | null>(null)
  const [reviewing, setReviewing] = useState<{ key: number; preset: FormPreset } | null>(null)
  const [saved, setSaved] = useState<Set<string>>(new Set()) // símbolos SQX guardados desde esta hoja
  const fileRef = useRef<HTMLInputElement>(null)
  const reviewIdRef = useRef(0) // cada apertura del formulario es un `key` nuevo (estado limpio)
  const fileSeqRef = useRef(0) // cada fichero elegido es una lectura nueva (aunque se llame y pese igual)

  // solo viajan los símbolos válidos; los inválidos se marcan en su fila y el backend propone el suyo
  const cleanMapping = useMemo(() => {
    const out: Record<string, string> = {}
    for (const [k, v] of Object.entries(mapping)) if (SQX_OK.test(v)) out[k] = v
    return out
  }, [mapping])
  const reqSig = file && brokerId ? `${brokerId}|${file.seq}|${file.name}|${JSON.stringify(cleanMapping)}` : ''
  const loading = !!reqSig && outcome?.sig !== reqSig
  const data = outcome?.data // mientras se relee, se sigue mostrando la tabla anterior (atenuada)
  const errorNow = outcome && outcome.sig === reqSig ? outcome.error : undefined

  // PLAN-DESVIO (menor): el brief describe «elegir el fichero → elegir el bróker destino → POST /mt5-sheet». Aquí la lectura es automática en
  // cuanto hay fichero y bróker (el bróker destino viene preelegido), y se repite al cambiar el bróker o un símbolo SQX tecleado: leer la
  // hoja no guarda nada, y es el backend quien dice si la ficha ya existe, si hay conflictos y si el borrador es guardable.
  useEffect(() => {
    if (!file || !brokerId) return
    let alive = true
    const mySig = reqSig
    // la primera lectura es inmediata; las siguientes (el usuario teclea símbolos) esperan a que pare
    const t = setTimeout(
      () => {
        parseMt5Sheet(brokerId, file.text, cleanMapping)
          .then((d) => alive && setOutcome({ sig: mySig, data: d }))
          .catch((e) => alive && setOutcome({ sig: mySig, error: symbolErrText(e) }))
      },
      Object.keys(cleanMapping).length ? DEBOUNCE_MS : 0,
    )
    return () => {
      alive = false
      clearTimeout(t)
    }
  }, [file, brokerId, cleanMapping, reqSig])

  const pickFile = async (f: File) => {
    setFileError(null)
    try {
      const text = await f.text()
      setFile({ name: f.name, text, seq: ++fileSeqRef.current })
      setMapping({})
      setSaved(new Set())
    } catch (e) {
      setFileError(`No se pudo leer ${f.name}: ${String((e as Error)?.message ?? e)}`)
    } finally {
      if (fileRef.current) fileRef.current.value = ''
    }
  }

  const review = (d: SymbolDraft) =>
    setReviewing({
      key: ++reviewIdRef.current,
      preset: {
        origin: 'mt5_sheet',
        title: `Hoja de MT5 · ${d.ticker ?? ''}`,
        subtitle: `${file?.name ?? ''} → bróker ${d.broker ?? brokerId}`,
        broker: d.broker ?? brokerId,
        lockBroker: true,
        draft: d,
      },
    })

  if (reviewing) {
    return (
      <SymbolForm
        key={reviewing.key}
        schema={schema}
        schemaError={schemaError}
        brokers={brokers}
        symbols={symbols}
        mode="new"
        preset={reviewing.preset}
        cancelLabel="← Volver a la tabla"
        onDirtyChange={onDirtyChange}
        onCancel={() => setReviewing(null)}
        onSaved={(view, how) => {
          setSaved((s) => new Set(s).add(view.sqx_symbol))
          setReviewing(null)
          onSaved(view, how)
        }}
      />
    )
  }

  const drafts = errorNow ? [] : (data?.drafts ?? [])
  return (
    <Card>
      <CardHeader
        title="Importar la hoja de MT5"
        subtitle="El script de SQX para MT5 exporta «Human Readable Exported Information Sheet.csv» con los datos de los símbolos de tu bróker."
        right={
          <button
            onClick={onClose}
            className="rounded-md border border-zinc-700 px-2.5 py-1 text-xs text-zinc-300 hover:bg-zinc-800"
          >
            ← Cerrar
          </button>
        }
      />
      <CardBody className="space-y-4">
        <p className="rounded-md border border-zinc-800 bg-zinc-950/60 px-3 py-2 text-xs text-zinc-400">
          Elige la hoja (CSV) y el bróker donde guardar las fichas. La app te propone una ficha por fila: revisa cada una
          y guarda las que quieras. <span className="text-zinc-300">Nada se guarda sin que lo confirmes</span>; la hoja
          exporta la comisión a 0 si no se tecleó en el script, así que la comisión siempre hay que confirmarla a mano.
        </p>

        {/* H8 (revisión de código de la F5): sin ningún bróker CFD no hay dónde guardar las fichas de la hoja. Aviso con la acción. */}
        {cfdBrokers.length === 0 && (
          <div className="flex flex-wrap items-center justify-between gap-3 rounded-md border border-amber-900 bg-amber-950/40 px-3 py-2 text-xs text-amber-300">
            <span>
              No hay ningún bróker de tipo CFD donde guardar las fichas de la hoja: crea uno con «+ Bróker» (clase de activo CFD) y
              vuelve aquí.
            </span>
            {onNewBroker && (
              <SmallBtn tone="primary" onClick={onNewBroker}>
                + Bróker
              </SmallBtn>
            )}
          </div>
        )}

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div>
            <span className="mb-1 block text-xs font-medium text-zinc-400">1. Hoja de MT5 (CSV)</span>
            <div className="flex flex-wrap items-center gap-2">
              <SmallBtn tone="primary" onClick={() => fileRef.current?.click()}>
                Elegir fichero…
              </SmallBtn>
              <span className="min-w-0 truncate text-xs text-zinc-400">{file ? file.name : 'ningún fichero elegido'}</span>
              <input
                ref={fileRef}
                type="file"
                accept=".csv,text/csv"
                className="hidden"
                aria-label="Hoja de MT5 (CSV)"
                onChange={(e) => {
                  const f = e.target.files?.[0]
                  if (f) void pickFile(f)
                }}
              />
            </div>
            {fileError && <p className="mt-1 text-xs text-rose-300">{fileError}</p>}
          </div>
          <Field label="2. Bróker destino (solo CFD)">
            <Select
              value={brokerId}
              onChange={setBrokerPick}
              options={cfdBrokers.map((b) => ({
                value: b.id,
                label: `${b.display_name} (${b.id})`,
                group: b.official ? 'Oficiales' : 'Tuyos',
              }))}
            />
          </Field>
        </div>

        {errorNow && (
          <p className="rounded-md border border-rose-900 bg-rose-950/60 px-2 py-1.5 text-xs text-rose-300">{errorNow}</p>
        )}
        {loading && file && <p className="text-xs text-zinc-500">Leyendo la hoja…</p>}

        {drafts.length > 0 && (
          <div className={`space-y-2 ${loading ? 'opacity-60' : ''}`} aria-busy={loading}>
            <p className="text-xs text-zinc-400">
              {drafts.length === 1 ? '1 símbolo en la hoja' : `${drafts.length} símbolos en la hoja`} · bróker {data?.broker}. El
              símbolo SQX tiene que ser igual, carácter a carácter, que el de tus .sqx de ese bróker: si no lo conoce la app,
              tecléalo.
            </p>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-zinc-800 text-left text-[11px] uppercase tracking-wide text-zinc-500">
                    <th className="px-2 py-1.5 font-medium">Ticker</th>
                    <th className="px-2 py-1.5 font-medium">Símbolo SQX propuesto</th>
                    <th className="px-2 py-1.5 font-medium">Estado</th>
                    <th className="px-2 py-1.5 font-medium">Faltan</th>
                    <th className="px-2 py-1.5 font-medium">Conflictos</th>
                    <th className="px-2 py-1.5 font-medium" />
                  </tr>
                </thead>
                <tbody>
                  {drafts.map((d, i) => {
                    const ticker = d.ticker ?? ''
                    const shown = mapping[ticker] ?? d.sqx_symbol ?? ''
                    const invalid = shown !== '' && !SQX_OK.test(shown)
                    const done = d.sqx_symbol != null && saved.has(d.sqx_symbol)
                    // H1 (revisión de código de la F5): el borrador de la fila solo vale si es el de la ÚLTIMA lectura y se calculó con
                    // el símbolo que se ve en la caja. Mientras se relee (`loading`), o si lo tecleado no es lo que dice el borrador
                    // (inválido, vacío o aún sin releer), abrir el formulario guardaría con un símbolo o un bróker equivocados.
                    const outdated = loading || shown !== (d.sqx_symbol ?? '')
                    return (
                      <tr key={`${ticker}-${i}`} className="border-b border-zinc-800/60 align-top">
                        <td className="px-2 py-1.5 font-medium text-zinc-200">{ticker}</td>
                        <td className="min-w-48 px-2 py-1.5">
                          <input
                            className={`${inputCls} ${invalid ? 'border-rose-700' : 'border-zinc-700 focus:border-emerald-600'}`}
                            value={shown}
                            list={d.candidates.length ? `cand-${i}` : undefined}
                            placeholder="símbolo SQX"
                            aria-label={`Símbolo SQX de ${ticker}`}
                            onChange={(e) => setMapping((m) => ({ ...m, [ticker]: e.target.value }))}
                          />
                          {d.candidates.length > 0 && (
                            <datalist id={`cand-${i}`}>
                              {d.candidates.map((c) => (
                                <option key={c} value={c} />
                              ))}
                            </datalist>
                          )}
                          {invalid && <p className="mt-0.5 text-[11px] text-rose-300">sin espacios ni barras (1 a 60 caracteres)</p>}
                          {!invalid && d.sqx_symbol_suggested && mapping[ticker] === undefined && (
                            <p className="mt-0.5 text-[11px] text-zinc-500">sugerido por el ticker</p>
                          )}
                        </td>
                        <td className="max-w-72 px-2 py-1.5 text-xs">
                          {done ? (
                            <Badge variant="success">guardada</Badge>
                          ) : d.guardable && !d.existing ? (
                            <Badge variant="success">guardable</Badge>
                          ) : d.guardable && d.existing ? (
                            <Badge variant="warn">ya tiene ficha tuya</Badge>
                          ) : (
                            <div className="space-y-0.5">
                              <Badge variant="danger">no guardable</Badge>
                              {d.motivo && <p className="text-zinc-400">{d.motivo}</p>}
                            </div>
                          )}
                        </td>
                        <td className="px-2 py-1.5 text-xs text-amber-300">
                          {!done && d.faltan.length ? d.faltan.map(fieldLabel).join(', ') : <span className="text-zinc-600">—</span>}
                        </td>
                        <td className="px-2 py-1.5 text-xs">
                          {!done && d.conflicts.length ? (
                            <Badge variant="warn">
                              <span title={d.conflicts.map((c) => c.texto).join('\n')}>
                                {d.conflicts.length === 1 ? '1 conflicto' : `${d.conflicts.length} conflictos`}
                              </span>
                            </Badge>
                          ) : (
                            <span className="text-zinc-600">—</span>
                          )}
                        </td>
                        <td className="px-2 py-1.5 text-right">
                          {!done && (
                            <SmallBtn
                              tone="primary"
                              onClick={() => review(d)}
                              disabled={outdated}
                              title={
                                loading
                                  ? 'Leyendo la hoja…'
                                  : outdated
                                    ? 'El símbolo SQX tecleado aún no se ha leído (o no es válido): corrígelo o espera un momento'
                                    : undefined
                              }
                            >
                              Revisar y guardar
                            </SmallBtn>
                          )}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </CardBody>
    </Card>
  )
}
