// Panel "¿Cómo se ha calculado?" (plan onboarding-b1, §2.5): junto a los números,
// qué supone el modelo. Un Collapsible L3 plegado por defecto con dos secciones:
//  1. "En esta corrida": las notas de meta.model_context (ya con las cifras de la
//     corrida) y los textos de meta que antes no se pintaban o vivían en otro panel
//     (ev_semantics, day_axis_note; en AXI policy_assumption, float_convention,
//     f_star_shift_note). Texto literal, sin reescribir.
//  2. "Supuestos del perfil": las notas de config/model_notes.yaml agrupadas por
//     sesgo. OMITE los ids que ya salen en "En esta corrida" (sin duplicados).
// Se usa en las dos lentes; el catálogo de la lente AXI llega en
// catalog.model_notes_axi y el de prop firm en firm.caveats (firma de la CORRIDA).
import type { ModelContext, ModelNote, Sesgo } from '../lib/api'
import { Collapsible } from './ui'

/** Texto de meta que se pinta tal cual en "En esta corrida". */
export interface RunText {
  titulo: string
  texto: string
}

interface Props {
  context?: ModelContext // meta.model_context de la corrida
  runTexts?: RunText[] // ev_semantics, day_axis_note, policy_assumption…
  profileNotes?: ModelNote[] // caveats de la firma de la corrida | model_notes_axi
}

const GROUPS: { sesgo: Sesgo; titulo: string; cls: string; dot: string }[] = [
  {
    sesgo: 'optimista',
    titulo: 'Pueden inflar el resultado',
    cls: 'text-amber-400',
    dot: 'bg-amber-500',
  },
  {
    sesgo: 'conservador',
    titulo: 'Pueden quedarse cortas',
    cls: 'text-sky-400',
    dot: 'bg-sky-500',
  },
  { sesgo: 'neutral', titulo: 'Para saber', cls: 'text-zinc-400', dot: 'bg-zinc-500' },
]

function NoteItem({ n, dot }: { n: ModelNote; dot: string }) {
  return (
    <li className="flex gap-2">
      <span className={`mt-1.5 size-1.5 shrink-0 rounded-full ${dot}`} />
      <p className="text-xs leading-relaxed text-zinc-400">
        <span className="font-medium text-zinc-200">{n.titulo}.</span> {n.texto}
      </p>
    </li>
  )
}

export function ModelNotesPanel({ context, runTexts = [], profileNotes = [] }: Props) {
  const runNotes = context?.notes ?? []
  const shownIds = new Set(runNotes.map((n) => n.id))
  const staticNotes = profileNotes.filter((n) => !shownIds.has(n.id))
  if (runNotes.length === 0 && runTexts.length === 0 && staticNotes.length === 0) return null

  const dotOf = (s: Sesgo) => GROUPS.find((g) => g.sesgo === s)?.dot ?? 'bg-zinc-500'

  return (
    <Collapsible title="¿Cómo se ha calculado?" subtitle="L3 · qué supone el modelo detrás de cada número">
      <div className="space-y-4">
        {(runNotes.length > 0 || runTexts.length > 0) && (
          <section>
            <h4 className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-zinc-500">
              En esta corrida
            </h4>
            <ul className="space-y-1.5">
              {runNotes.map((n) => (
                <NoteItem key={n.id} n={n} dot={dotOf(n.sesgo)} />
              ))}
              {runTexts.map((t) => (
                <li key={t.titulo} className="flex gap-2">
                  <span className="mt-1.5 size-1.5 shrink-0 rounded-full bg-zinc-500" />
                  <p className="text-xs leading-relaxed text-zinc-400">
                    <span className="font-medium text-zinc-200">{t.titulo}:</span> {t.texto}
                  </p>
                </li>
              ))}
            </ul>
          </section>
        )}

        {staticNotes.length > 0 && (
          <section className="space-y-3">
            <h4 className="text-[11px] font-semibold uppercase tracking-wide text-zinc-500">
              Supuestos del perfil
            </h4>
            {GROUPS.map((g) => {
              const items = staticNotes.filter((n) => n.sesgo === g.sesgo)
              if (items.length === 0) return null
              return (
                <div key={g.sesgo}>
                  <div className={`mb-1 text-xs font-semibold ${g.cls}`}>{g.titulo}</div>
                  <ul className="space-y-1.5">
                    {items.map((n) => (
                      <NoteItem key={n.id} n={n} dot={g.dot} />
                    ))}
                  </ul>
                </div>
              )
            })}
          </section>
        )}
      </div>
    </Collapsible>
  )
}
