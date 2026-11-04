// Detalle de la vista Perfiles (guardrail perfiles-v2, §2.6): columna derecha. Solo el reto
// elegido: cabecera con insignias, reglas en solo lectura (ProfileRulesPanel con las `rules`
// del catálogo, también para un oculto) y las acciones. "Editar" y "Nuevo" abren su formulario
// en esta misma columna (lo gestiona ProfilesView).
import { companyLabel, fmtDate, plainName, type Firm, type ProfileView } from '../lib/api'
import { Badge, Card, CardBody, LockIcon, SmallBtn } from './ui'
import { ProfileRulesPanel } from './ProfileRulesPanel'

// Plantillas de "Nuevo perfil" (perfiles-v2 §2.3): rótulo corto para la insignia.
const TEMPLATE_LABEL: Record<string, string> = {
  cfd_2f: 'CFD 2 fases',
  cfd_1f: 'CFD 1 fase',
  fut_suscripcion: 'futuros con suscripción',
  fut_trailing: 'futuros con trailing',
}

export function ProfileDetail({
  profile,
  firm,
  catalogLoading,
  busy,
  canHide,
  canDelete,
  onDuplicate,
  onEdit,
  onExport,
  onToggleHidden,
  onDelete,
}: {
  profile: ProfileView | null
  firm?: Firm // entrada del catálogo con las `rules`
  catalogLoading: boolean
  busy: boolean
  canHide: boolean // false si ocultarlo dejaría cero perfiles visibles
  canDelete: boolean // false si borrarlo dejaría cero perfiles visibles
  onDuplicate: () => void
  onEdit: () => void
  onExport: () => void
  onToggleHidden: () => void
  onDelete: () => void
}) {
  if (!profile) {
    return (
      <Card className="flex h-48 items-center justify-center">
        <p className="text-sm text-zinc-500">Elige un reto a la izquierda</p>
      </Card>
    )
  }
  const official = !!profile.official
  const hidden = !!profile.hidden
  const needs = !!profile.needs_recreate
  const hideBlocked = !hidden && !canHide
  const size = typeof profile.spec?.account_size === 'number' ? profile.spec.account_size : null
  return (
    <Card>
      <div className="space-y-3 border-b border-zinc-800 px-4 py-3">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0 space-y-1">
            <div className="flex flex-wrap items-baseline gap-2">
              <h3 className="text-sm font-semibold text-zinc-100">{plainName(profile)}</h3>
              {!official && <span className="text-sm text-emerald-400">(tuyo)</span>}
              <code className="text-[11px] text-zinc-500">{profile.key}</code>
            </div>
            <div className="text-xs text-zinc-400">
              {companyLabel(profile)}
              {size != null && <span className="num text-zinc-500"> · {size.toLocaleString('es-ES')} $</span>}
            </div>
            <div className="flex flex-wrap items-center gap-1.5 text-xs">
              {official ? (
                <Badge variant="outline" className="gap-1">
                  <LockIcon />
                  oficial{profile.ruleset_version ? ` · ${profile.ruleset_version}` : ''}
                </Badge>
              ) : (
                <>
                  <Badge variant="success">tuyo</Badge>
                  {profile.template ? (
                    <Badge variant="outline">desde cero · {TEMPLATE_LABEL[profile.template] ?? profile.template}</Badge>
                  ) : (
                    <span className="text-zinc-400">basado en {profile.based_on}</span>
                  )}
                  {profile.verified_on ? (
                    <span className="text-zinc-400">verificado el {fmtDate(profile.verified_on)}</span>
                  ) : (
                    <Badge variant="warn">sin verificar</Badge>
                  )}
                </>
              )}
              {hidden && <Badge variant="default">oculto</Badge>}
            </div>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {!official && (
              <SmallBtn
                onClick={onEdit}
                disabled={needs || busy}
                title={needs ? 'El oficial base cambió: duplícalo de nuevo' : 'Editar los números del perfil'}
              >
                Editar
              </SmallBtn>
            )}
            <SmallBtn tone="primary" onClick={onDuplicate} disabled={busy} title="Crea una copia editable">
              Duplicar
            </SmallBtn>
            <SmallBtn onClick={onExport} title="Descarga el perfil como JSON">
              Exportar
            </SmallBtn>
            <SmallBtn
              onClick={onToggleHidden}
              disabled={busy || hideBlocked}
              title={
                hidden
                  ? 'Vuelve a mostrarlo en el simulador'
                  : hideBlocked
                    ? 'Dejaría cero perfiles visibles'
                    : 'Lo quita del simulador y de la lista (se puede restaurar)'
              }
            >
              {hidden ? 'Mostrar' : 'Ocultar'}
            </SmallBtn>
            {!official && (
              <SmallBtn
                tone="danger"
                onClick={onDelete}
                disabled={busy || !canDelete}
                title={canDelete ? 'Borra el perfil (no se puede deshacer)' : 'Dejaría cero perfiles visibles'}
              >
                Borrar
              </SmallBtn>
            )}
          </div>
        </div>
        {needs && (
          <p className="rounded-md border border-amber-900 bg-amber-950/40 px-3 py-2 text-xs text-amber-300">
            El perfil oficial base ({profile.based_on}) cambió desde que creaste este perfil, así que ya no
            es coherente con él y no se puede editar. Vuelve a duplicar el oficial y reaplica tus
            números; después puedes borrar este.
          </p>
        )}
        {!official && profile.notes && (
          <p className="whitespace-pre-line text-xs text-zinc-500">{profile.notes}</p>
        )}
      </div>
      <CardBody>
        {firm?.rules ? (
          <ProfileRulesPanel firm={firm} />
        ) : (
          <p className="text-sm text-zinc-500">
            {catalogLoading ? 'Cargando reglas…' : 'Este perfil no tiene reglas que mostrar en el catálogo.'}
          </p>
        )}
      </CardBody>
    </Card>
  )
}
