// Árbol de la vista Perfiles (guardrail perfiles-v2, §2.6): columna izquierda. Compañías del
// tipo elegido, PLEGADAS por defecto; dentro, los retos en filas compactas; al final, la
// sección plegada "Ocultos (n)" con "Mostrar". El estado (plegado, selección) vive en
// ProfilesView; este componente solo pinta y avisa.
import { useMemo } from 'react'
import { companyLabel, companySummary, plainName, type ProfileView } from '../lib/api'
import { Badge, LockIcon, SmallBtn } from './ui'

function Chevron({ open }: { open: boolean }) {
  return (
    <span className={`inline-block text-zinc-500 transition-transform ${open ? 'rotate-90' : ''}`}>▸</span>
  )
}

function Row({
  p,
  selected,
  onSelect,
  right,
  showCompany = false,
}: {
  p: ProfileView
  selected: boolean
  onSelect: () => void
  right?: React.ReactNode
  showCompany?: boolean
}) {
  const official = !!p.official
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
        className="flex min-w-0 flex-1 items-center gap-1.5 text-left text-xs"
      >
        {official ? (
          <span className="text-zinc-500" title="Perfil oficial (solo lectura)">
            <LockIcon />
          </span>
        ) : null}
        <span className={`truncate ${selected ? 'text-emerald-200' : 'text-zinc-200'}`}>{plainName(p)}</span>
        {!official && <span className="shrink-0 text-emerald-400">(tuyo)</span>}
        {showCompany && <span className="shrink-0 text-zinc-500">· {companyLabel(p)}</span>}
        {!official && !p.verified_on && (
          <Badge variant="warn" className="shrink-0">
            sin verificar
          </Badge>
        )}
      </button>
      {right}
    </div>
  )
}

export function ProfileTree({
  profiles,
  kindLabel,
  selectedKey,
  onSelect,
  expanded,
  onToggleCompany,
  hiddenOpen,
  onToggleHidden,
  onHideCompany,
  onShow,
  canHide,
  busy,
  onNew,
}: {
  profiles: ProfileView[] // todos los del tipo activo (visibles y ocultos)
  kindLabel: string // "CFD" | "Futuros"
  selectedKey: string | null
  onSelect: (key: string) => void
  expanded: Set<string>
  onToggleCompany: (company: string) => void
  hiddenOpen: boolean
  onToggleHidden: () => void
  onHideCompany: (company: string, keys: string[]) => void
  onShow: (keys: string[]) => void
  canHide: (keys: string[]) => boolean // false si dejaría cero perfiles visibles
  busy: boolean
  onNew: () => void
}) {
  const visible = useMemo(() => profiles.filter((p) => !p.hidden), [profiles])
  const hidden = useMemo(() => profiles.filter((p) => !!p.hidden), [profiles])
  const groups = useMemo(() => {
    const by = new Map<string, ProfileView[]>()
    for (const p of visible) {
      const c = companyLabel(p)
      if (!by.has(c)) by.set(c, [])
      by.get(c)!.push(p)
    }
    return [...by.entries()]
      .sort(([a], [b]) => a.localeCompare(b, 'es'))
      .map(([name, items]) => ({
        name,
        // oficiales primero, después los tuyos (orden estable del API dentro de cada bloque)
        items: [...items.filter((p) => !!p.official), ...items.filter((p) => !p.official)],
      }))
  }, [visible])

  return (
    <div className="space-y-2">
      {visible.length === 0 && (
        <div className="space-y-2 rounded-lg border border-dashed border-zinc-700 px-4 py-5 text-center">
          <p className="text-sm text-zinc-300">No hay perfiles de {kindLabel} visibles</p>
          <div className="flex justify-center gap-2">
            {hidden.length > 0 && (
              <SmallBtn onClick={() => onShow(hidden.map((p) => p.key))} disabled={busy}>
                Mostrar ocultos
              </SmallBtn>
            )}
            <SmallBtn tone="primary" onClick={onNew}>
              + Nuevo perfil
            </SmallBtn>
          </div>
        </div>
      )}

      {groups.map((g) => {
        const open = expanded.has(g.name)
        const keys = g.items.map((p) => p.key)
        const allowed = canHide(keys)
        return (
          <div key={g.name} className="overflow-hidden rounded-lg border border-zinc-800 bg-zinc-950/40">
            <div className="flex items-center gap-2 px-2.5 py-2">
              <button
                type="button"
                onClick={() => onToggleCompany(g.name)}
                aria-expanded={open}
                className="flex min-w-0 flex-1 items-center gap-2 text-left text-sm font-medium text-zinc-100"
              >
                <Chevron open={open} />
                <span className="truncate">{companySummary(g.name, g.items)}</span>
              </button>
              <SmallBtn
                onClick={() => onHideCompany(g.name, keys)}
                disabled={busy || !allowed}
                title={allowed ? 'Oculta todos los perfiles de esta compañía (se pueden restaurar)' : 'Dejaría cero perfiles visibles'}
              >
                Ocultar compañía
              </SmallBtn>
            </div>
            {open && (
              <div className="divide-y divide-zinc-800/60 border-t border-zinc-800">
                {g.items.map((p) => (
                  <Row key={p.key} p={p} selected={p.key === selectedKey} onSelect={() => onSelect(p.key)} />
                ))}
              </div>
            )}
          </div>
        )
      })}

      {hidden.length > 0 && (
        <div className="overflow-hidden rounded-lg border border-zinc-800 bg-zinc-950/40">
          <div className="flex items-center gap-2 px-2.5 py-2">
            <button
              type="button"
              onClick={onToggleHidden}
              aria-expanded={hiddenOpen}
              className="flex min-w-0 flex-1 items-center gap-2 text-left text-sm font-medium text-zinc-400"
            >
              <Chevron open={hiddenOpen} />
              Ocultos ({hidden.length})
            </button>
            {hiddenOpen && (
              <SmallBtn onClick={() => onShow(hidden.map((p) => p.key))} disabled={busy}>
                Mostrar todos
              </SmallBtn>
            )}
          </div>
          {hiddenOpen && (
            <div className="divide-y divide-zinc-800/60 border-t border-zinc-800">
              {hidden.map((p) => (
                <Row
                  key={p.key}
                  p={p}
                  selected={p.key === selectedKey}
                  onSelect={() => onSelect(p.key)}
                  showCompany
                  right={
                    <SmallBtn onClick={() => onShow([p.key])} disabled={busy}>
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
