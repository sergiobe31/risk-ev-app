// Formulario de un perfil de usuario (guardrail onboarding-c C-2 + perfiles-v2 fase 2).
// Dos modos, siempre en la COLUMNA DERECHA de la vista Perfiles:
//  - editar (`profile`): el formulario sale de GET /api/profiles/{key}/schema (lista blanca
//    del servidor: label, unidad, rango y valor del oficial raíz);
//  - nuevo desde plantilla (`newProfile`): parte de GET /api/profiles/templates/{tid} (spec con
//    null, ejemplo en gris, contador "faltan N campos", errores solo de los campos tocados).
// Validación en vivo contra POST /api/profiles/validate (debounce 400 ms); "Guardar" queda
// desactivado mientras haya errores, falten campos o la validación esté pendiente.
import { useEffect, useMemo, useState } from 'react'
import {
  createProfile,
  fetchProfileSchema,
  ProfileApiError,
  updateProfile,
  validateProfile,
  type ProfileSchemaField,
  type ProfileView,
  type TemplateDetail,
} from '../lib/api'
import {
  Button,
  Card,
  CardBody,
  CardHeader,
  Checkbox,
  Field,
  NumberInput,
  Select,
  TextArea,
  TextInput,
} from './ui'

const DEBOUNCE_MS = 400
const NOTES_MAX = 2000
const inputCls =
  'w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-sm text-zinc-100 ' +
  'placeholder:text-zinc-600 focus:border-emerald-600 focus:outline-none'

type Spec = Record<string, any>

interface Draft {
  spec: Spec
  verifiedOn: string
  notes: string
}

interface Validated {
  sig: string
  errors: string[]
}

export interface NewProfileInit {
  key: string
  template: TemplateDetail
  company: string
  displayName: string
}

function getByPath(spec: unknown, path: string): unknown {
  let node: any = spec
  for (const t of path.split('.')) {
    if (node == null) return undefined
    node = Array.isArray(node) ? node[Number(t)] : node[t]
  }
  return node
}

// Devuelve una copia de spec con path = value (los tokens numéricos indexan listas).
function setByPath(spec: Spec, path: string, value: unknown): Spec {
  const next = structuredClone(spec)
  const toks = path.split('.')
  let node: any = next
  for (let i = 0; i < toks.length - 1; i++) {
    node = Array.isArray(node) ? node[Number(toks[i])] : node[toks[i]]
  }
  const last = toks[toks.length - 1]
  if (Array.isArray(node)) node[Number(last)] = value
  else node[last] = value
  return next
}

function fmtValue(v: unknown, unit: string, isBool: boolean): string {
  if (v == null) return '—'
  if (isBool) return v ? 'sí' : 'no' // 0/1 numérico NO es booleano fuera de campos boolean
  if (typeof v === 'number') return `${v.toLocaleString('es-ES')}${unit ? ` ${unit}` : ''}`
  return String(v)
}

function rangeHint(f: ProfileSchemaField): string | undefined {
  if (f.type === 'boolean' || f.type === 'string' || f.type === 'enum') return undefined
  const lo = f.min != null ? `${f.min_exclusive ? '>' : '≥'} ${f.min}` : null
  const hi = f.max != null ? `≤ ${f.max}` : null
  const parts = [f.type === 'integer' ? 'entero' : null, lo, hi].filter(Boolean)
  return parts.length ? `Rango: ${parts.join(', ')}` : undefined
}

function sameValue(a: unknown, b: unknown): boolean {
  if (typeof a === 'boolean' || typeof b === 'boolean') return !!a === !!b
  return a === b
}

const isEmpty = (v: unknown): boolean => v == null || (typeof v === 'string' && v.trim() === '')

function todayIso(): string {
  return new Date().toLocaleDateString('sv-SE') // AAAA-MM-DD en hora local
}

// Parseo tolerante (coma o punto decimal); null ante vacío o basura.
function parseLoose(s: string): number | null {
  const t = s.trim().replace(',', '.')
  if (!t) return null
  const v = Number(t)
  return Number.isFinite(v) ? v : null
}

const fmtNum = (v: number): string => String(parseFloat(v.toPrecision(10)))

// Número que admite VACÍO (null): el modo plantilla parte de campos sin rellenar.
function OptNumberInput({
  value,
  onChange,
  placeholder,
}: {
  value: number | null
  onChange: (v: number | null) => void
  placeholder?: string
}) {
  const [draft, setDraft] = useState<string | null>(null)
  const shown = draft ?? (value == null ? '' : fmtNum(value))
  return (
    <input
      type="text"
      inputMode="decimal"
      className={inputCls}
      value={shown}
      placeholder={placeholder}
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

function FieldRow({
  field,
  value,
  onChange,
  isNew,
  error,
  unitNow,
}: {
  unitNow?: string
  field: ProfileSchemaField
  value: unknown
  onChange: (v: unknown) => void
  isNew: boolean
  error?: string
}) {
  const ref = isNew ? field.example : field.reference
  // en edición: el nombre difiere del oficial por construcción, no se marca "cambiado"
  const changed = !isNew && field.type !== 'string' && !sameValue(value, ref)
  const unit = field.unit
  const exTxt = `ej. ${fmtValue(ref, '', field.type === 'boolean')}`
  return (
    <div className="py-1.5">
      <div className="grid grid-cols-1 items-center gap-x-3 gap-y-1 sm:grid-cols-[minmax(0,1fr)_150px_48px_minmax(0,130px)]">
        <span
          className={`text-sm ${changed ? 'text-amber-300' : 'text-zinc-300'}`}
          title={[field.path, rangeHint(field)].filter(Boolean).join(' · ')}
        >
          {field.label}
          {changed && <span className="ml-1.5 text-[11px] text-amber-400">· cambiado</span>}
        </span>
        <div>
          {field.type === 'boolean' ? (
            isNew ? (
              <Select
                value={value == null ? '' : value ? 'si' : 'no'}
                onChange={(v) => onChange(v === '' ? null : v === 'si')}
                options={[
                  { value: '', label: 'Elige…' },
                  { value: 'si', label: 'Sí' },
                  { value: 'no', label: 'No' },
                ]}
              />
            ) : (
              <Checkbox
                checked={!!value}
                onChange={(v) => onChange(v)}
                label={value ? 'Sí' : 'No'}
              />
            )
          ) : field.type === 'enum' ? (
            <Select
              value={typeof value === 'string' ? value : ''}
              onChange={(v) => onChange(v === '' ? null : v)}
              options={[
                ...(isEmpty(value) ? [{ value: '', label: 'Elige…' }] : []),
                ...(field.options ?? []).map((o) => ({ value: o, label: o })),
              ]}
            />
          ) : field.type === 'string' ? (
            <TextInput
              value={String(value ?? '')}
              placeholder={isNew ? exTxt : undefined}
              onChange={(v) => onChange(isNew && v.trim() === '' ? null : v)}
            />
          ) : isNew ? (
            <OptNumberInput
              value={typeof value === 'number' ? value : null}
              placeholder={exTxt}
              onChange={onChange}
            />
          ) : (
            <NumberInput
              value={typeof value === 'number' ? value : Number(value ?? 0)}
              step={field.type === 'integer' ? 1 : undefined}
              onChange={(v) => onChange(v)}
            />
          )}
        </div>
        <span className="text-xs text-zinc-500">{field.type === 'boolean' ? '' : (unitNow ?? unit)}</span>
        {isNew ? (
          <span className="num text-xs text-zinc-600" title="Valor del perfil oficial de referencia">
            {exTxt}
            {field.type !== 'boolean' && field.type !== 'string' && field.type !== 'enum' && unit
              ? ` ${unit}`
              : ''}
          </span>
        ) : (
          <span className="num text-xs text-zinc-500" title="Valor del perfil oficial base">
            oficial: {fmtValue(ref, unit, field.type === 'boolean')}
          </span>
        )}
      </div>
      {error && <p className="mt-0.5 text-xs text-rose-300">{error}</p>}
    </div>
  )
}

// "phases.0.profit_target_pct (Objetivo): fuera de rango..." -> camino y mensaje sin prefijo.
function splitError(e: string): { path: string | null; msg: string } {
  const m = /^([A-Za-z0-9_.]+) \(/.exec(e)
  if (!m) return { path: null, msg: e }
  const i = e.indexOf('): ')
  return { path: m[1], msg: i >= 0 ? e.slice(i + 3) : e }
}

export function ProfileForm({
  profile,
  newProfile,
  onSaved,
  onCancel,
  cancelLabel = '← Volver al detalle',
}: {
  profile?: ProfileView
  newProfile?: NewProfileInit
  onSaved: (p: ProfileView) => void
  onCancel: () => void
  cancelLabel?: string
}) {
  const isNew = !!newProfile
  const profKey = (newProfile?.key ?? profile?.key) as string
  const initial = useMemo<Draft>(() => {
    if (newProfile) {
      return {
        spec: {
          ...structuredClone(newProfile.template.spec),
          company: newProfile.company,
          display_name: newProfile.displayName,
        },
        verifiedOn: '',
        notes: '',
      }
    }
    return {
      spec: structuredClone((profile as ProfileView).spec),
      verifiedOn: profile?.verified_on ?? '',
      notes: profile?.notes ?? '',
    }
  }, [profile, newProfile])
  const [draft, setDraft] = useState<Draft>(initial)
  const [remoteSchema, setRemoteSchema] = useState<ProfileSchemaField[] | null>(null)
  const [schemaError, setSchemaError] = useState<string | null>(null)
  const [validated, setValidated] = useState<Validated | null>(null)
  const [touched, setTouched] = useState<Set<string>>(
    () => new Set(newProfile ? ['company', 'display_name'] : []),
  )
  const [saving, setSaving] = useState(false)
  const [saveErrors, setSaveErrors] = useState<string[]>([])

  const schema = newProfile ? newProfile.template.schema : remoteSchema
  const sig = JSON.stringify(draft)
  const dirty = sig !== JSON.stringify(initial)
  const pending = validated?.sig !== sig
  const rawErrors = useMemo(
    () => (validated && !pending ? validated.errors : []),
    [validated, pending],
  )

  const missing = useMemo(
    () => (isNew ? (schema ?? []).filter((f) => isEmpty(getByPath(draft.spec, f.path))).length : 0),
    [isNew, schema, draft.spec],
  )

  // Errores a mostrar: en edición, todos; en plantilla, solo los de campos tocados (y los
  // globales, sin camino), sin los "falta el valor" (ya cuenta el contador).
  const { fieldErrors, shownErrors } = useMemo(() => {
    const byPath = new Map<string, string>()
    const shown: string[] = []
    const known = new Set((schema ?? []).map((f) => f.path))
    for (const e of rawErrors) {
      const { path, msg } = splitError(e)
      if (isNew && /falta el valor/.test(e)) continue
      if (path && known.has(path)) {
        if (isNew && !touched.has(path)) continue
        if (!byPath.has(path)) byPath.set(path, msg)
      }
      shown.push(e)
    }
    return { fieldErrors: byPath, shownErrors: shown }
  }, [rawErrors, schema, isNew, touched])

  const canSave =
    !!schema && !pending && rawErrors.length === 0 && !saving && (isNew ? missing === 0 : dirty)

  useEffect(() => {
    if (isNew) return
    let alive = true
    fetchProfileSchema(profKey)
      .then((s) => alive && setRemoteSchema(s))
      .catch((e) => alive && setSchemaError(String(e.message ?? e)))
    return () => {
      alive = false
    }
  }, [isNew, profKey])

  // Validación en vivo: debounce de 400 ms; las respuestas viejas se descartan.
  useEffect(() => {
    let alive = true
    const mySig = JSON.stringify(draft)
    const t = setTimeout(() => {
      const common = {
        key: profKey,
        spec: draft.spec,
        verified_on: draft.verifiedOn || null,
        notes: draft.notes || null,
      }
      const body = newProfile
        ? { ...common, template: newProfile.template.id }
        : profile?.template
          ? { ...common, template: profile.template, editing: true }
          : { ...common, based_on: profile?.based_on ?? profKey, editing: true }
      validateProfile(body)
        .then((r) => alive && setValidated({ sig: mySig, errors: r.errors }))
        .catch((e) =>
          alive && setValidated({ sig: mySig, errors: [String(e.message ?? e)] }),
        )
    }, DEBOUNCE_MS)
    return () => {
      alive = false
      clearTimeout(t)
    }
  }, [draft, profKey, profile, newProfile])

  const sections = useMemo(() => {
    const order: string[] = []
    const by = new Map<string, ProfileSchemaField[]>()
    for (const f of schema ?? []) {
      if (!by.has(f.section)) {
        by.set(f.section, [])
        order.push(f.section)
      }
      by.get(f.section)!.push(f)
    }
    return order.map((name) => ({ name, fields: by.get(name)! }))
  }, [schema])

  const setField = (path: string, value: unknown) => {
    setDraft((d) => ({ ...d, spec: setByPath(d.spec, path, value) }))
    setTouched((s) => (s.has(path) ? s : new Set(s).add(path)))
  }

  const save = async () => {
    setSaving(true)
    setSaveErrors([])
    try {
      const view = newProfile
        ? await createProfile({
            template: newProfile.template.id,
            key: newProfile.key,
            spec: draft.spec,
            verified_on: draft.verifiedOn || null,
            notes: draft.notes || null,
          })
        : await updateProfile(profKey, {
            spec: draft.spec,
            verified_on: draft.verifiedOn || null,
            notes: draft.notes || null,
          })
      onSaved(view)
    } catch (e) {
      setSaveErrors(e instanceof ProfileApiError ? e.errors : [String(e)])
    } finally {
      setSaving(false)
    }
  }

  const cancel = () => {
    if (dirty && !window.confirm('Hay cambios sin guardar. ¿Descartarlos?')) return
    onCancel()
  }

  const name = String(draft.spec.display_name ?? '') || profKey
  return (
    <Card>
      <CardHeader
        title={`${isNew ? 'Nuevo perfil' : 'Editar perfil'} · ${name}`}
        subtitle={
          isNew
            ? `id ${profKey} · plantilla ${newProfile!.template.label}`
            : `id ${profKey} · basado en ${profile?.based_on ?? '—'}${
                profile?.template ? ` · plantilla ${profile.template}` : ''
              }`
        }
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
        <p className="rounded-md border border-amber-900 bg-amber-950/40 px-3 py-2 text-xs text-amber-300">
          {isNew
            ? 'Rellena todos los números a mano (en gris, el valor de ejemplo del perfil oficial de referencia). Solo se pueden definir reglas que la app ya sabe simular.'
            : 'Solo puedes cambiar números de reglas que la app ya sabe simular. Para una regla nueva hace falta código.'}
        </p>

        <div className="grid grid-cols-1 gap-5 xl:grid-cols-[minmax(0,1fr)_300px]">
          {/* Campos por sección */}
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
                  {sec.fields.map((f) => (
                    <FieldRow
                      key={f.path}
                      field={f}
                      // la unidad de la tarifa sigue a la moneda elegida (fee_ccy del borrador); el ejemplo conserva la suya
                      unitNow={
                        f.path === 'fee' && typeof getByPath(draft.spec, 'fee_ccy') === 'string'
                          ? (getByPath(draft.spec, 'fee_ccy') as string)
                          : undefined
                      }
                      isNew={isNew}
                      value={getByPath(draft.spec, f.path)}
                      error={fieldErrors.get(f.path)}
                      onChange={(v) => setField(f.path, v)}
                    />
                  ))}
                </div>
              </section>
            ))}
          </div>

          {/* Metadatos, validación y acciones */}
          <aside className="space-y-3 xl:sticky xl:top-4 xl:self-start">
            <Field
              label="Verificado en la web de la firma el"
              hint="Fecha en que comprobaste estos números contra la web oficial de la firma"
            >
              <div className="flex gap-2">
                <TextInput
                  type="date"
                  value={draft.verifiedOn}
                  onChange={(v) => setDraft((d) => ({ ...d, verifiedOn: v }))}
                />
                <button
                  onClick={() => setDraft((d) => ({ ...d, verifiedOn: todayIso() }))}
                  className="shrink-0 rounded-md border border-zinc-700 px-2 text-xs text-zinc-300 hover:bg-zinc-800"
                >
                  hoy
                </button>
              </div>
            </Field>
            <Field label={`Notas (${draft.notes.length}/${NOTES_MAX})`}>
              <TextArea
                value={draft.notes}
                rows={4}
                maxLength={NOTES_MAX}
                placeholder="De dónde salen los números, qué cambió, etc."
                onChange={(v) => setDraft((d) => ({ ...d, notes: v }))}
              />
            </Field>

            {isNew && missing > 0 && (
              <div
                className="rounded-md border border-amber-900 bg-amber-950/40 px-3 py-2 text-xs text-amber-300"
                aria-live="polite"
              >
                {missing === 1 ? 'Falta 1 campo por rellenar' : `Faltan ${missing} campos por rellenar`}
              </div>
            )}

            {!(isNew && missing > 0 && shownErrors.length === 0) && (
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
                  'Sin errores: el perfil es válido y simulable.'
                )}
              </div>
            )}

            {saveErrors.length > 0 && (
              <ul className="list-disc space-y-0.5 rounded-md border border-rose-900 bg-rose-950/60 py-1.5 pl-6 pr-2 text-xs text-rose-300">
                {saveErrors.map((e, i) => (
                  <li key={i}>{e}</li>
                ))}
              </ul>
            )}

            <Button onClick={save} disabled={!canSave}>
              {saving
                ? isNew
                  ? 'Creando…'
                  : 'Guardando…'
                : isNew
                  ? 'Crear perfil'
                  : dirty
                    ? 'Guardar'
                    : 'Sin cambios'}
            </Button>
          </aside>
        </div>
      </CardBody>
    </Card>
  )
}
