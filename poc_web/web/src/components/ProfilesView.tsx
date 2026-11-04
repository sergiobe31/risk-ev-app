// Vista "Perfiles" (guardrails onboarding-c C-2 y perfiles-v2 fase 2): ÁRBOL + DETALLE.
// Cabecera con el segmentado CFD | Futuros, "+ Nuevo perfil", "Importar…" y el menú ⋯;
// columna izquierda = compañías plegadas con sus retos (+ "Ocultos (n)"); columna derecha =
// el reto elegido (reglas y acciones), el editor o el asistente de "Nuevo perfil".
// Ocultar != borrar (D1): los oficiales se ocultan y se restauran; los tuyos además se
// borran. El backend es poc_web/api/profiles.py (fuente de verdad de las formas JSON).
import { useEffect, useMemo, useRef, useState } from 'react'
import {
  companyLabel,
  createProfile,
  deleteProfile,
  exportProfile,
  importProfile,
  listProfiles,
  plainName,
  profileErrText,
  setProfilesHidden,
  type Catalog,
  type ProfileView,
  type ProfilesList,
} from '../lib/api'
import { Card, CardBody, CardHeader, Field, MenuButton, Modal, Segmented, SmallBtn, TextInput } from './ui'
import { ProfileForm } from './ProfileForm'
import { ProfileDetail } from './ProfileDetail'
import { ProfileTree } from './ProfileTree'
import { NewProfileWizard } from './NewProfileWizard'

const KEY_RE = /^[a-z0-9_]{3,40}$/

type Kind = 'cfd' | 'futures'
type Mode = 'detail' | 'edit' | 'new'
type Notice = { tone: 'ok' | 'error'; text: string }
interface Confirm {
  title: string
  text: string
  confirmLabel: string
  tone: 'primary' | 'danger'
  run: () => Promise<void>
}

const KIND_LABEL: Record<Kind, string> = { cfd: 'CFD', futures: 'Futuros' }

// Sugerencia de id para un duplicado: <base>_mio, <base>_mio_2, ... dentro del formato.
function suggestKey(base: string, taken: Set<string>): string {
  for (let n = 1; n < 1000; n++) {
    const suffix = n === 1 ? '_mio' : `_mio_${n}`
    const cand = base.slice(0, 40 - suffix.length) + suffix
    if (KEY_RE.test(cand) && !taken.has(cand)) return cand
  }
  return ''
}

export function ProfilesView({
  catalog,
  catalogLoading,
  onProfilesChanged,
}: {
  catalog: Catalog | null
  catalogLoading: boolean
  // App vuelve a pedir el catálogo EN EL MOMENTO (el detalle usa sus `rules`); la promesa
  // se resuelve cuando el catálogo ya está actualizado.
  onProfilesChanged: () => Promise<void>
}) {
  const [data, setData] = useState<ProfilesList | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [notice, setNotice] = useState<Notice | null>(null)
  const [kind, setKind] = useState<Kind>('cfd')
  const [mode, setMode] = useState<Mode>('detail')
  const [selectedKey, setSelectedKey] = useState<string | null>(null)
  const [expanded, setExpanded] = useState<Set<string>>(new Set()) // `${kind}|${compañía}`; vacío = todo plegado
  const [hiddenOpen, setHiddenOpen] = useState<Set<Kind>>(new Set())
  const [confirm, setConfirm] = useState<Confirm | null>(null)
  const [dupFrom, setDupFrom] = useState<ProfileView | null>(null)
  const [dupKey, setDupKey] = useState('')
  const [dupName, setDupName] = useState('')
  const [dupError, setDupError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  const reload = () =>
    listProfiles()
      .then((d) => {
        setData(d)
        setLoadError(null)
      })
      .catch((e) => setLoadError(profileErrText(e)))

  // tras un cambio: catálogo (App) y lista, en paralelo
  const refresh = async () => {
    await Promise.all([onProfilesChanged().catch(() => undefined), reload()])
  }

  useEffect(() => {
    let alive = true
    listProfiles()
      .then((d) => alive && setData(d))
      .catch((e) => alive && setLoadError(profileErrText(e)))
    return () => {
      alive = false
    }
  }, [])

  const profiles = useMemo(() => data?.profiles ?? [], [data])
  const taken = useMemo(() => new Set(profiles.map((p) => p.key)), [profiles])
  const ofKind = useMemo(() => profiles.filter((p) => p.kind === kind), [profiles, kind])
  const visibleCount = (k: Kind) => profiles.filter((p) => p.kind === k && !p.hidden).length
  const visibleTotal = profiles.filter((p) => !p.hidden).length
  const selected = profiles.find((p) => p.key === selectedKey) ?? null

  // Guarda de la UI: no ofrecer ocultar/borrar si dejaría cero perfiles visibles (el backend da 409).
  const canHide = (keys: string[]) => {
    const vis = profiles.filter((p) => !p.hidden && keys.includes(p.key)).length
    return visibleTotal - vis >= 1
  }
  const canDelete = (p: ProfileView) => !!p.hidden || visibleTotal - 1 >= 1

  // Si hay un formulario abierto, avisa antes de salir de él.
  const leaveForm = (): boolean =>
    mode === 'detail' || window.confirm('Hay un formulario abierto. ¿Descartarlo?')

  const goKind = (k: Kind) => {
    if (k === kind || !leaveForm()) return
    setKind(k)
    setMode('detail')
    setSelectedKey(null)
  }

  const select = (key: string) => {
    if (key !== selectedKey && !leaveForm()) return
    setSelectedKey(key)
    setMode('detail')
    setNotice(null)
  }

  const toggleCompany = (company: string) =>
    setExpanded((s) => {
      const id = `${kind}|${company}`
      const n = new Set(s)
      if (n.has(id)) n.delete(id)
      else n.add(id)
      return n
    })
  const expandedHere = useMemo(() => {
    const out = new Set<string>()
    for (const id of expanded) if (id.startsWith(`${kind}|`)) out.add(id.slice(kind.length + 1))
    return out
  }, [expanded, kind])

  const reveal = (p: ProfileView) => {
    // asegura que el perfil se vea en el árbol: tipo activo y compañía desplegada
    setKind(p.kind)
    setExpanded((s) => new Set(s).add(`${p.kind}|${companyLabel(p)}`))
    setSelectedKey(p.key)
  }

  const doHide = async (keys: string[], hidden: boolean, okText: string) => {
    setBusy(true)
    try {
      await setProfilesHidden(keys, hidden)
      await refresh()
      setNotice({ tone: 'ok', text: okText })
    } catch (e) {
      setNotice({ tone: 'error', text: profileErrText(e) }) // incluye el 409 "deja al menos un perfil visible"
    } finally {
      setBusy(false)
    }
  }

  const askHideCompany = (company: string, keys: string[]) =>
    setConfirm({
      title: `Ocultar ${company}`,
      text: `Se ocultan ${keys.length === 1 ? 'el perfil' : `los ${keys.length} perfiles`} de ${company} del simulador y de esta lista. No se borra nada: lo encuentras en «Ocultos» y puedes restaurarlo.`,
      confirmLabel: 'Ocultar compañía',
      tone: 'primary',
      run: () => doHide(keys, true, `${company} oculta (${keys.length}).`),
    })

  const visibleOfKind = ofKind.filter((p) => !p.hidden).map((p) => p.key)
  const hiddenOfKind = ofKind.filter((p) => !!p.hidden).map((p) => p.key)
  const askHideKind = () =>
    setConfirm({
      title: `Ocultar todos los de ${KIND_LABEL[kind]}`,
      text: `Se ocultan los ${visibleOfKind.length} perfiles de ${KIND_LABEL[kind]} visibles del simulador y de esta lista. No se borra nada: están en «Ocultos» y puedes restaurarlos.`,
      confirmLabel: 'Ocultar todos',
      tone: 'primary',
      run: () => doHide(visibleOfKind, true, `Ocultos todos los de ${KIND_LABEL[kind]} (${visibleOfKind.length}).`),
    })

  const askDelete = (p: ProfileView) =>
    setConfirm({
      title: 'Borrar perfil',
      text: `¿Borrar ${plainName(p)} (${p.key})? Esta acción no se puede deshacer; exporta el perfil antes si quieres conservarlo.`,
      confirmLabel: 'Borrar',
      tone: 'danger',
      run: async () => {
        setBusy(true)
        try {
          await deleteProfile(p.key)
          if (selectedKey === p.key) setSelectedKey(null)
          setMode('detail')
          await refresh()
          setNotice({ tone: 'ok', text: `Perfil '${p.key}' borrado.` })
        } catch (e) {
          setNotice({ tone: 'error', text: profileErrText(e) })
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

  const openDuplicate = (p: ProfileView) => {
    setDupFrom(p)
    setDupKey(suggestKey(p.key, taken))
    setDupName(`${plainName(p)} (tuyo)`)
    setDupError(null)
  }

  const dupIdError = !dupFrom
    ? null
    : !KEY_RE.test(dupKey)
      ? 'El id debe cumplir ^[a-z0-9_]{3,40}$ (minúsculas, dígitos y _; 3 a 40 caracteres)'
      : taken.has(dupKey)
        ? `El id '${dupKey}' ya existe`
        : null

  const doDuplicate = async () => {
    if (!dupFrom || dupIdError) return
    setBusy(true)
    setDupError(null)
    try {
      const view = await createProfile({
        base: dupFrom.key,
        key: dupKey,
        display_name: dupName.trim() || undefined,
      })
      setDupFrom(null)
      await refresh()
      reveal(view)
      setNotice({ tone: 'ok', text: `Perfil '${view.key}' creado. Ajusta sus números y guarda.` })
      setMode('edit') // directo al editor (columna derecha): lo habitual tras duplicar
    } catch (e) {
      setDupError(profileErrText(e))
    } finally {
      setBusy(false)
    }
  }

  const doExport = async (p: ProfileView) => {
    try {
      const env = await exportProfile(p.key)
      const blob = new Blob([JSON.stringify(env, null, 2) + '\n'], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${p.key}.json`
      document.body.appendChild(a)
      a.click()
      a.remove()
      URL.revokeObjectURL(url)
      setNotice({ tone: 'ok', text: `Exportado ${p.key}.json` })
    } catch (e) {
      setNotice({ tone: 'error', text: `No se pudo exportar: ${profileErrText(e)}` })
    }
  }

  const doImport = async (file: File) => {
    setBusy(true)
    try {
      const res = await importProfile(await file.text())
      await refresh()
      reveal(res.profile)
      setMode('detail')
      setNotice({
        tone: 'ok',
        text: res.renamed_from
          ? `El id '${res.renamed_from}' ya existía: importado como '${res.profile.key}'.`
          : `Perfil '${res.profile.key}' importado.`,
      })
    } catch (e) {
      setNotice({ tone: 'error', text: `No se pudo importar ${file.name}: ${profileErrText(e)}` })
    } finally {
      setBusy(false)
      if (fileRef.current) fileRef.current.value = ''
    }
  }

  const startNew = () => {
    if (!leaveForm()) return
    setNotice(null)
    setMode('new')
  }

  // --- Columna derecha --------------------------------------------------------------------
  let right: React.ReactNode
  if (mode === 'new') {
    right = (
      <NewProfileWizard
        initialKind={kind}
        profiles={profiles}
        onCancel={() => setMode('detail')}
        onCreated={async (view) => {
          await refresh()
          reveal(view)
          setMode('detail')
          setNotice({ tone: 'ok', text: `Perfil '${view.key}' creado.` })
        }}
      />
    )
  } else if (mode === 'edit' && selected && !selected.official) {
    right = (
      <ProfileForm
        key={selected.key}
        profile={selected}
        onCancel={() => setMode('detail')}
        onSaved={async (view) => {
          await refresh()
          setMode('detail')
          setNotice({ tone: 'ok', text: `Perfil '${view.key}' guardado.` })
        }}
      />
    )
  } else {
    right = (
      <ProfileDetail
        profile={selected}
        firm={selected ? catalog?.firms.find((f) => f.key === selected.key) : undefined}
        catalogLoading={catalogLoading}
        busy={busy}
        canHide={canHide([selected?.key ?? ''])}
        canDelete={selected ? canDelete(selected) : false}
        onDuplicate={() => selected && openDuplicate(selected)}
        onEdit={() => {
          setNotice(null)
          setMode('edit')
        }}
        onExport={() => selected && doExport(selected)}
        onToggleHidden={() =>
          selected &&
          doHide(
            [selected.key],
            !selected.hidden,
            selected.hidden ? `'${selected.key}' vuelve a estar visible.` : `'${selected.key}' oculto.`,
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
          title="Perfiles de prop firm"
          subtitle="Los oficiales son de solo lectura: se ocultan (y se restauran), o se duplican para cambiar sus números. También puedes crear un perfil desde cero con reglas que la app ya sabe simular."
          right={
            <div className="flex flex-wrap items-center justify-end gap-2">
              <Segmented<Kind>
                value={kind}
                onChange={goKind}
                options={[
                  { value: 'cfd', label: `CFD (${visibleCount('cfd')})` },
                  { value: 'futures', label: `Futuros (${visibleCount('futures')})` },
                ]}
              />
              <SmallBtn tone="primary" onClick={startNew} disabled={busy}>
                + Nuevo perfil
              </SmallBtn>
              <SmallBtn onClick={() => fileRef.current?.click()} disabled={busy}>
                Importar…
              </SmallBtn>
              <input
                ref={fileRef}
                type="file"
                accept=".json,application/json"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0]
                  if (f) doImport(f)
                }}
              />
              <MenuButton
                items={[
                  {
                    label: `Ocultar todos los de ${KIND_LABEL[kind]}`,
                    onClick: askHideKind,
                    disabled: busy || visibleOfKind.length === 0 || !canHide(visibleOfKind),
                    title:
                      visibleOfKind.length > 0 && !canHide(visibleOfKind)
                        ? 'Dejaría cero perfiles visibles'
                        : undefined,
                  },
                  {
                    label: `Mostrar todos los ocultos de ${KIND_LABEL[kind]}`,
                    onClick: () => {
                      doHide(hiddenOfKind, false, `Restaurados los ocultos de ${KIND_LABEL[kind]} (${hiddenOfKind.length}).`)
                    },
                    disabled: busy || hiddenOfKind.length === 0,
                  },
                ]}
              />
            </div>
          }
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

      {loadError && (
        <Card className="border-rose-900 bg-rose-950/40">
          <CardBody className="py-2.5 text-xs text-rose-300">No se pudieron cargar los perfiles: {loadError}</CardBody>
        </Card>
      )}

      {data && data.errors.length > 0 && (
        <Card className="border-amber-900 bg-amber-950/30">
          <CardBody className="space-y-1.5 py-3 text-xs text-amber-300">
            <div className="font-semibold">
              Hay {data.errors.length === 1 ? 'una entrada' : `${data.errors.length} entradas`} de tu
              fichero de perfiles que no se cargan
            </div>
            <ul className="list-disc space-y-0.5 pl-5">
              {data.errors.map((e, i) => (
                <li key={i}>
                  <code className="text-amber-200">{e.key ?? '(fichero)'}</code>: {e.error}
                </li>
              ))}
            </ul>
          </CardBody>
        </Card>
      )}

      {!data && !loadError && <p className="text-sm text-zinc-500">Cargando perfiles…</p>}

      {data && (
        <div className="grid grid-cols-1 items-start gap-5 lg:grid-cols-[340px_minmax(0,1fr)]">
          <ProfileTree
            profiles={ofKind}
            kindLabel={KIND_LABEL[kind]}
            selectedKey={selectedKey}
            onSelect={select}
            expanded={expandedHere}
            onToggleCompany={toggleCompany}
            hiddenOpen={hiddenOpen.has(kind)}
            onToggleHidden={() =>
              setHiddenOpen((s) => {
                const n = new Set(s)
                if (n.has(kind)) n.delete(kind)
                else n.add(kind)
                return n
              })
            }
            onHideCompany={askHideCompany}
            onShow={(keys) =>
              void doHide(keys, false, keys.length === 1 ? `'${keys[0]}' vuelve a estar visible.` : `${keys.length} perfiles restaurados.`)
            }
            canHide={canHide}
            busy={busy}
            onNew={startNew}
          />
          <div className="min-w-0">{right}</div>
        </div>
      )}

      {dupFrom && (
        <Modal title={`Duplicar ${dupFrom.display_name}`} onClose={() => !busy && setDupFrom(null)}>
          <Field label="Id del nuevo perfil" hint="Minúsculas, dígitos y _ (3 a 40 caracteres)">
            <TextInput value={dupKey} onChange={setDupKey} />
          </Field>
          {dupIdError && <p className="text-xs text-rose-300">{dupIdError}</p>}
          <Field label="Nombre">
            <TextInput value={dupName} onChange={setDupName} />
          </Field>
          {dupError && (
            <p className="rounded-md border border-rose-900 bg-rose-950/60 px-2 py-1.5 text-xs text-rose-300">{dupError}</p>
          )}
          <div className="flex justify-end gap-2 pt-1">
            <SmallBtn onClick={() => setDupFrom(null)} disabled={busy}>
              Cancelar
            </SmallBtn>
            <SmallBtn tone="primary" onClick={doDuplicate} disabled={busy || !!dupIdError}>
              {busy ? 'Creando…' : 'Duplicar'}
            </SmallBtn>
          </div>
        </Modal>
      )}

      {confirm && (
        <Modal title={confirm.title} onClose={() => setConfirm(null)}>
          <p className="text-sm text-zinc-300">{confirm.text}</p>
          <div className="flex justify-end gap-2 pt-1">
            <SmallBtn onClick={() => setConfirm(null)}>Cancelar</SmallBtn>
            <SmallBtn tone={confirm.tone} onClick={runConfirm}>
              {confirm.confirmLabel}
            </SmallBtn>
          </div>
        </Modal>
      )}
    </div>
  )
}
