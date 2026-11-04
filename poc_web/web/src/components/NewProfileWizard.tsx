// "Nuevo perfil" desde cero (guardrail perfiles-v2, §2.6), en la columna derecha, en dos pasos:
//  1. tipo, familia de reglas (tarjetas con la descripción de GET /templates), compañía (las
//     existentes de ese tipo + "Otra…"), nombre e id (sugerido, único, ^[a-z0-9_]{3,40}$);
//  2. el formulario parte de GET /templates/{tid}: campos vacíos, ejemplo en gris, contador
//     "faltan N campos" y "Crear" activo solo con todo completo y válido (ProfileForm).
import { useEffect, useMemo, useState } from 'react'
import {
  companyLabel,
  fetchTemplate,
  listTemplates,
  NO_COMPANY,
  profileErrText,
  type ProfileView,
  type TemplateInfo,
} from '../lib/api'
import { Button, Card, CardBody, CardHeader, Field, Segmented, Select, TextInput } from './ui'
import { ProfileForm, type NewProfileInit } from './ProfileForm'

const KEY_RE = /^[a-z0-9_]{3,40}$/
const OTHER = '__otra__'
const COMPANY_MAX = 40
const NAME_MAX = 80

type Kind = 'cfd' | 'futures'

// Misma normalización que el servidor: recorta y colapsa espacios internos.
const normalizeCompany = (s: string) => s.trim().replace(/\s+/g, ' ')

function slug(s: string): string {
  return s
    .normalize('NFD')
    .replace(/\p{M}/gu, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
}

// <compañía>_<familia>, único entre los ids existentes, dentro del formato.
function suggestKey(company: string, tid: string, taken: Set<string>): string {
  const base = `${slug(company) || 'perfil'}_${tid}`.slice(0, 40)
  if (KEY_RE.test(base) && !taken.has(base)) return base
  for (let n = 2; n < 1000; n++) {
    const suffix = `_${n}`
    const cand = base.slice(0, 40 - suffix.length) + suffix
    if (KEY_RE.test(cand) && !taken.has(cand)) return cand
  }
  return ''
}

export function NewProfileWizard({
  initialKind,
  profiles,
  onCancel,
  onCreated,
}: {
  initialKind: Kind
  profiles: ProfileView[] // todos (para compañías existentes e ids ocupados)
  onCancel: () => void
  onCreated: (view: ProfileView) => void
}) {
  const [kind, setKind] = useState<Kind>(initialKind)
  const [templates, setTemplates] = useState<TemplateInfo[] | null>(null)
  const [tplError, setTplError] = useState<string | null>(null)
  const [tid, setTid] = useState<string | null>(null)
  const [companySel, setCompanySel] = useState<string>('')
  const [companyOther, setCompanyOther] = useState('')
  const [displayName, setDisplayName] = useState('')
  const [keyInput, setKeyInput] = useState<string | null>(null) // null = usa la sugerida
  const [init, setInit] = useState<NewProfileInit | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    listTemplates()
      .then((t) => alive && setTemplates(t))
      .catch((e) => alive && setTplError(profileErrText(e)))
    return () => {
      alive = false
    }
  }, [])

  const taken = useMemo(() => new Set(profiles.map((p) => p.key)), [profiles])
  const companies = useMemo(() => {
    const set = new Set<string>()
    for (const p of profiles) {
      if (p.kind !== kind) continue
      const c = companyLabel(p)
      if (c !== NO_COMPANY) set.add(c)
    }
    return [...set].sort((a, b) => a.localeCompare(b, 'es'))
  }, [profiles, kind])
  const kindTemplates = useMemo(() => (templates ?? []).filter((t) => t.kind === kind), [templates, kind])

  // compañía efectiva: la elegida de la lista o el texto libre de "Otra…"
  const sel = companySel === '' ? (companies[0] ?? OTHER) : companySel
  const company = normalizeCompany(sel === OTHER ? companyOther : sel)
  const companyErr = !company
    ? 'Indica la compañía'
    : company.length > COMPANY_MAX
      ? `Máximo ${COMPANY_MAX} caracteres`
      : null
  const key = keyInput ?? (tid ? suggestKey(company, tid, taken) : '')
  const keyErr = !tid
    ? null
    : !KEY_RE.test(key)
      ? 'El id debe cumplir ^[a-z0-9_]{3,40}$ (minúsculas, dígitos y _; 3 a 40 caracteres)'
      : taken.has(key)
        ? `El id '${key}' ya existe`
        : null
  const name = displayName.trim()
  const nameErr = !name ? 'Indica el nombre del perfil' : name.length > NAME_MAX ? `Máximo ${NAME_MAX} caracteres` : null
  const ready = !!tid && !companyErr && !keyErr && !nameErr && !loading

  const changeKind = (k: Kind) => {
    setKind(k)
    setTid(null)
    setCompanySel('')
    setKeyInput(null)
  }

  const next = async () => {
    if (!ready || !tid) return
    setLoading(true)
    setError(null)
    try {
      const detail = await fetchTemplate(tid)
      setInit({ key, template: detail, company, displayName: name })
    } catch (e) {
      setError(profileErrText(e))
    } finally {
      setLoading(false)
    }
  }

  if (init) {
    return (
      <ProfileForm
        key={init.key}
        newProfile={init}
        cancelLabel="← Cambiar familia o datos"
        onCancel={() => setInit(null)}
        onSaved={onCreated}
      />
    )
  }

  return (
    <Card>
      <CardHeader
        title="Nuevo perfil"
        subtitle="Paso 1 de 2 · tipo, familia de reglas y datos básicos. Después rellenas los números a mano."
        right={
          <button
            onClick={onCancel}
            className="rounded-md border border-zinc-700 px-2.5 py-1 text-xs text-zinc-300 hover:bg-zinc-800"
          >
            Cancelar
          </button>
        }
      />
      <CardBody className="space-y-5">
        <Field label="Tipo de producto">
          <div>
            <Segmented
              value={kind}
              onChange={changeKind}
              options={[
                { value: 'cfd', label: 'CFD' },
                { value: 'futures', label: 'Futuros' },
              ]}
            />
          </div>
        </Field>

        <div className="space-y-1.5">
          <span className="block text-xs font-medium text-zinc-400">Familia de reglas</span>
          {tplError && (
            <p className="rounded-md border border-rose-900 bg-rose-950/60 px-2 py-1.5 text-xs text-rose-300">
              No se pudieron cargar las plantillas: {tplError}
            </p>
          )}
          {!templates && !tplError && <p className="text-sm text-zinc-500">Cargando familias…</p>}
          <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
            {kindTemplates.map((t) => (
              <button
                key={t.id}
                type="button"
                onClick={() => {
                  setTid(t.id)
                  setKeyInput(null)
                }}
                aria-pressed={tid === t.id}
                className={`rounded-lg border px-3 py-2.5 text-left transition-colors ${
                  tid === t.id
                    ? 'border-emerald-700 bg-emerald-950/50'
                    : 'border-zinc-700 hover:bg-zinc-800/60'
                }`}
              >
                <span className={`block text-sm font-medium ${tid === t.id ? 'text-emerald-200' : 'text-zinc-200'}`}>
                  {t.label}
                </span>
                <span className="mt-1 block text-xs text-zinc-500">
                  {t.n_phases === 1 ? '1 fase' : `${t.n_phases} fases`} · {t.n_fields} campos a rellenar
                </span>
              </button>
            ))}
          </div>
          {templates && kindTemplates.length === 0 && (
            <p className="text-sm text-zinc-500">No hay familias de reglas para este tipo.</p>
          )}
        </div>

        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          <Field label="Compañía" hint="Agrupa los retos en el selector del simulador y en el árbol">
            <Select
              value={sel}
              onChange={(v) => {
                setCompanySel(v)
                setKeyInput(null)
              }}
              options={[
                ...companies.map((c) => ({ value: c, label: c })),
                { value: OTHER, label: 'Otra…' },
              ]}
            />
          </Field>
          {sel === OTHER && (
            <Field label="Nombre de la compañía">
              <TextInput value={companyOther} onChange={setCompanyOther} placeholder="p. ej. FundedNext" />
            </Field>
          )}
          <Field label="Nombre del perfil">
            <TextInput
              value={displayName}
              onChange={setDisplayName}
              placeholder={`p. ej. ${company || 'FundedNext'} 50k`}
            />
          </Field>
          <Field label="Id del perfil" hint="Minúsculas, dígitos y _ (3 a 40 caracteres); único">
            <TextInput value={key} onChange={setKeyInput} />
          </Field>
        </div>

        {tid && (
          <div className="space-y-0.5 text-xs text-rose-300">
            {companyErr && <p>{companyErr}</p>}
            {nameErr && <p>{nameErr}</p>}
            {keyErr && <p>{keyErr}</p>}
          </div>
        )}
        {error && (
          <p className="rounded-md border border-rose-900 bg-rose-950/60 px-2 py-1.5 text-xs text-rose-300">{error}</p>
        )}

        <div className="md:ml-auto md:w-[240px]">
          <Button onClick={next} disabled={!ready}>
            {loading ? 'Cargando…' : !tid ? 'Elige una familia' : 'Continuar →'}
          </Button>
        </div>
      </CardBody>
    </Card>
  )
}
