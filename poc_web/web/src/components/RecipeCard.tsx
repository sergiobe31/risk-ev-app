// Receta para configurar el EA (guardrail entrada-4-bloques, F5; plan §2.8 y §2.9, enmiendas E3, E12, E14, E18).
// Solo aparece con carteras .sqx (el backend añade `meta.recipe` en las dos lentes). La receta trae TODA la rejilla
// de f: aquí se compone la de la f elegida (por defecto, la f* del resultado o la más cercana de la rejilla), sin
// pedir nada al backend. Muestra, para cada pata, los parámetros del MM «FixedAmount» del EA:
//   RiskedMoney (USD) = f · cuenta · Decimals · mmStep · mmMultiplier · LotsIfNoMM · MaxLots,
// más cuántas órdenes van al lote mínimo y el riesgo real p95 (E12). LotsIfNoMM = lote mínimo es una CONDICIÓN para que
// el EA haga lo simulado, nunca una recomendación (E18).
import { useState } from 'react'
import type { Recipe, RecipeLeg } from '../lib/api'
import { Badge, Card, CardBody, CardHeader, Collapsible, Select, SmallBtn } from './ui'

const NOTES_OPEN_KEY = 'riskapp.recipe.notes.open' // '1' abiertas, '0' cerradas; sin valor = nunca se tocaron (abiertas)

// Las notas de la receta llegan como {id, texto}; los títulos son los de config/model_notes.yaml.
const NOTE_TITLE: Record<string, string> = {
  receta_ea: 'Receta para configurar tu EA',
  fichas_broker: 'De dónde salen las especificaciones de cada símbolo',
  axi_sqx_flotante: 'Flotante de tus .sqx en la lente AXI',
}

// PLAN-DESVIO (interpretación): el brief dice «Notas (`recipe.notes`) en un `Collapsible` abierto la primera vez» sin decir qué es la
// «primera vez». Se interpreta así: abiertas mientras el usuario no las toque; a partir de ahí se recuerda su ÚLTIMA decisión (abrirlas
// o cerrarlas, revisión de código de la F5, H6) en localStorage y las próximas recetas salen como las dejó. Un navegador nuevo, o sin
// localStorage, las ve abiertas.
function notesStartOpen(): boolean {
  try {
    return localStorage.getItem(NOTES_OPEN_KEY) !== '0'
  } catch {
    return true // sin localStorage (modo privado, etc.): siempre abiertas
  }
}

function rememberNotesOpen(open: boolean): void {
  try {
    localStorage.setItem(NOTES_OPEN_KEY, open ? '1' : '0')
  } catch {
    /* sin localStorage no se recuerda */
  }
}

function nearestIndex(grid: number[], x: number): number {
  let best = 0
  for (let i = 1; i < grid.length; i++) if (Math.abs(grid[i] - x) < Math.abs(grid[best] - x)) best = i
  return best
}

// Parámetros del EA: con PUNTO decimal (es lo que se teclea en MT5 y lo que pega el botón «Copiar receta»).
const plain = (v: number | null | undefined, maxDigits = 6): string =>
  v == null || !Number.isFinite(v) ? '—' : String(parseFloat(v.toFixed(maxDigits)))
const lots = (v: number | null | undefined, decimals: number): string =>
  v == null || !Number.isFinite(v) ? '—' : v.toFixed(Math.max(0, decimals))
const pct1 = (v: number | null | undefined): string => (v == null || !Number.isFinite(v) ? '—' : `${(v * 100).toFixed(1)} %`)
const pctPts = (v: number | null | undefined): string => (v == null || !Number.isFinite(v) ? '—' : `${v.toFixed(2)} %`)

const USD_WARNING =
  'Confirma en MT5 que tu cuenta está en USD antes de teclear RiskedMoney; si no, la receta no vale (v1 solo USD).'

function recipeText(r: Recipe, idx: number): string {
  const f = r.f_grid[idx]
  const x = r.risked_money_by_f[idx]
  const out = [
    `Receta para el EA · MM ${r.mm} · ${r.currency}`,
    `f = ${(f * 100).toFixed(2)} % de una cuenta de ${plain(r.account_size, 2)} ${r.currency} -> RiskedMoney = ${plain(x)} ${r.currency}`,
    USD_WARNING,
    '',
  ]
  r.legs.forEach((l, i) => {
    out.push(`EA ${i + 1} · ${l.symbol} · pata ${l.pata}${l.broker ? ` · bróker ${l.broker}` : ''}`)
    out.push(`  RiskedMoney = ${plain(x)}`)
    out.push(`  Decimals = ${l.decimals}`)
    out.push(`  mmStep = ${plain(l.mmStep)}`)
    out.push(`  mmMultiplier = ${plain(l.mmMultiplier)}`)
    out.push(`  LotsIfNoMM = ${lots(l.LotsIfNoMM, l.decimals)}  (condición: el lote mínimo del símbolo)`)
    out.push(`  MaxLots = ${lots(l.max_lot_by_f[idx], l.decimals)}`)
    out.push(
      `  (a esta f, ${pct1(l.frac_min_lot_by_f[idx])} de las órdenes van al lote mínimo; riesgo real p95 = ${pctPts(l.p95_riesgo_real_pct_by_f[idx])} de la cuenta)`,
    )
    if (l.max_lot_excede_ficha_by_f[idx]) {
      out.push(`  AVISO: MaxLots supera el máximo del bróker (${plain(l.ficha_max_lot)}): el EA toparía en ese máximo y la app no.`)
    }
  })
  return out.join('\n')
}

function legBacktest(l: RecipeLeg): string | null {
  const parts: string[] = []
  if (l.mm_backtest) parts.push(`MM del backtest: ${l.mm_backtest.method}`)
  if (l.paridad) parts.push(`paridad ${pct1(l.paridad.frac_ok)}`)
  return parts.length ? parts.join(' · ') : null
}

export function RecipeCard({ recipe, fStar }: { recipe: Recipe; fStar: number }) {
  const starIdx = nearestIndex(recipe.f_grid, fStar)
  // la f elegida se olvida cuando llega otra receta (otra corrida): se compara por identidad, sin efecto de reseteo
  const [picked, setPicked] = useState<{ recipe: Recipe; idx: number } | null>(null)
  const [copied, setCopied] = useState<{ recipe: Recipe; idx: number; ok: boolean } | null>(null)
  const [notesOpen, setNotesOpen] = useState<boolean>(notesStartOpen) // en sincronía con el <details> (H6)
  if (recipe.f_grid.length === 0) return null

  const idx = picked && picked.recipe === recipe ? Math.min(picked.idx, recipe.f_grid.length - 1) : starIdx
  const f = recipe.f_grid[idx]
  const x = recipe.risked_money_by_f[idx]
  const text = recipeText(recipe, idx)
  const copyState = copied && copied.recipe === recipe && copied.idx === idx ? copied : null
  const excedidas = recipe.legs.filter((l) => l.max_lot_excede_ficha_by_f[idx])

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text)
      setCopied({ recipe, idx, ok: true })
    } catch {
      setCopied({ recipe, idx, ok: false }) // sin permiso de portapapeles: se enseña el texto para copiarlo a mano
    }
  }

  return (
    <Card>
      <CardHeader
        title="Receta para el EA"
        subtitle="Cómo configurar tus EAs para que hagan el riesgo fijo que simula la app"
        right={
          <label className="flex items-center gap-2 text-xs text-zinc-400">
            <span>f de la receta</span>
            <div className="w-56">
              <Select
                value={String(idx)}
                onChange={(v) => setPicked({ recipe, idx: Number(v) })}
                options={recipe.f_grid.map((g, i) => ({
                  value: String(i),
                  label: `${(g * 100).toFixed(2)} %${i === starIdx ? ' ★ f*' : ''} · ${plain(recipe.risked_money_by_f[i])} ${recipe.currency}`,
                }))}
              />
            </div>
          </label>
        }
      />
      <CardBody className="space-y-4">
        <div className="space-y-1 rounded-lg border border-emerald-900 bg-emerald-950/30 px-4 py-3">
          <p className="text-sm text-zinc-200">
            Configura cada EA con <code className="rounded bg-zinc-800 px-1 text-emerald-300">FixedAmount</code> y{' '}
            <code className="rounded bg-zinc-800 px-1 text-emerald-300">RiskedMoney</code> ={' '}
            <span className="num text-lg font-semibold text-emerald-300">
              {plain(x)} {recipe.currency}
            </span>
          </p>
          <p className="text-xs text-zinc-400">
            f = {(f * 100).toFixed(2)} % de una cuenta de {recipe.account_size.toLocaleString('es-ES')} {recipe.currency}
            {idx === starIdx ? ' (la f* de esta corrida)' : ''} · {recipe.legs.length === 1 ? '1 pata' : `${recipe.legs.length} patas`}
          </p>
          <p className="text-xs text-amber-300">{USD_WARNING}</p>
        </div>

        {excedidas.map((l) => (
          <p key={l.pata} className="rounded-md border border-rose-900 bg-rose-950/60 px-3 py-2 text-xs text-rose-300">
            ✗ {l.symbol} ({l.pata}): el lote máximo simulado ({lots(l.max_lot_by_f[idx], l.decimals)}) supera el máximo del
            bróker en su ficha ({plain(l.ficha_max_lot)}) a esta f: el EA toparía en ese máximo y la app no.
          </p>
        ))}

        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-zinc-800 text-left text-[11px] uppercase tracking-wide text-zinc-500">
                <th className="px-2 py-1.5 font-medium">Símbolo</th>
                <th className="px-2 py-1.5 text-right font-medium" title="Dinero que arriesga cada orden hasta su stop inicial (USD)">
                  RiskedMoney
                </th>
                <th className="px-2 py-1.5 text-right font-medium" title="Decimales del lote = decimales del paso de lote">
                  Decimals
                </th>
                <th className="px-2 py-1.5 text-right font-medium" title="Paso de lote del símbolo">
                  mmStep
                </th>
                <th className="px-2 py-1.5 text-right font-medium">mmMultiplier</th>
                <th
                  className="px-2 py-1.5 text-right font-medium"
                  title="Condición para que el EA haga lo simulado: el lote mínimo del símbolo. Cuando el lote calculado redondea a cero, el EA abre LotsIfNoMM; con el valor por defecto de SQX (1,0) abriría un lote entero."
                >
                  LotsIfNoMM (condición)
                </th>
                <th
                  className="px-2 py-1.5 text-right font-medium"
                  title="El mayor lote que la simulación usa a esta f: un tope de seguridad que nunca recorta lo simulado"
                >
                  MaxLots
                </th>
                <th
                  className="px-2 py-1.5 text-right font-medium"
                  title="Órdenes cuyo riesgo X no llega al lote mínimo: se abren igualmente al mínimo"
                >
                  % de órdenes al lote mínimo
                </th>
                <th
                  className="px-2 py-1.5 text-right font-medium"
                  title="Percentil 95 del riesgo real por orden (con el lote mínimo y el redondeo), en % de la cuenta"
                >
                  Riesgo real p95
                </th>
              </tr>
            </thead>
            <tbody>
              {recipe.legs.map((l) => {
                const excede = !!l.max_lot_excede_ficha_by_f[idx]
                const bt = legBacktest(l)
                return (
                  <tr key={l.pata} className="border-b border-zinc-800/60 align-top">
                    <td className="max-w-64 px-2 py-1.5">
                      <div className="font-medium text-zinc-200">{l.symbol}</div>
                      <div className="truncate text-[11px] text-zinc-500" title={l.pata}>
                        {l.pata}
                      </div>
                      {bt && <div className="text-[11px] text-zinc-600">{bt}</div>}
                    </td>
                    <td className="num px-2 py-1.5 text-right text-emerald-300">
                      {plain(x)} <span className="text-xs text-zinc-500">{recipe.currency}</span>
                    </td>
                    <td className="num px-2 py-1.5 text-right">{l.decimals}</td>
                    <td className="num px-2 py-1.5 text-right">{plain(l.mmStep)}</td>
                    <td className="num px-2 py-1.5 text-right">{plain(l.mmMultiplier)}</td>
                    <td className="num px-2 py-1.5 text-right">
                      {lots(l.LotsIfNoMM, l.decimals)}{' '}
                      <Badge variant="outline" className="align-middle">
                        condición
                      </Badge>
                    </td>
                    <td className={`num px-2 py-1.5 text-right ${excede ? 'font-semibold text-rose-400' : ''}`}>
                      {lots(l.max_lot_by_f[idx], l.decimals)}
                    </td>
                    <td className="num px-2 py-1.5 text-right">{pct1(l.frac_min_lot_by_f[idx])}</td>
                    <td className="num px-2 py-1.5 text-right">{pctPts(l.p95_riesgo_real_pct_by_f[idx])}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
        <p className="text-[11px] leading-relaxed text-zinc-500">
          LotsIfNoMM es una condición, no una sugerencia: debe ser el lote mínimo del símbolo para que el EA haga lo que
          simula la app (también cubre las órdenes sin stop). A esta f, la columna «% de órdenes al lote mínimo» dice
          cuántas órdenes arriesgan más de X porque no cabe un lote menor, y «Riesgo real p95» cuánto arriesga de verdad
          una orden de las grandes, en % de la cuenta.
        </p>

        {recipe.notes.length > 0 && (
          <Collapsible
            title="Notas de la receta"
            subtitle="qué tener en cuenta al configurar el EA"
            defaultOpen={notesOpen}
            onOpenChange={(open) => {
              setNotesOpen(open)
              rememberNotesOpen(open)
            }}
          >
            <ul className="space-y-2">
              {recipe.notes.map((n) => (
                <li key={n.id} className="flex gap-2">
                  <span className="mt-1.5 size-1.5 shrink-0 rounded-full bg-zinc-500" />
                  <p className="text-xs leading-relaxed text-zinc-400">
                    {NOTE_TITLE[n.id] && <span className="font-medium text-zinc-200">{NOTE_TITLE[n.id]}. </span>}
                    {n.texto}
                  </p>
                </li>
              ))}
            </ul>
          </Collapsible>
        )}

        <div className="space-y-2">
          <div className="flex flex-wrap items-center gap-3">
            <SmallBtn tone="primary" onClick={() => void copy()} title="Copia los parámetros de cada EA como texto plano">
              Copiar receta
            </SmallBtn>
            {copyState?.ok && <span className="text-xs text-emerald-400">✓ Receta copiada (f = {(f * 100).toFixed(2)} %)</span>}
            {copyState && !copyState.ok && (
              <span className="text-xs text-amber-300">El navegador no deja copiar: selecciona el texto de abajo y cópialo.</span>
            )}
          </div>
          {copyState && !copyState.ok && (
            <textarea
              readOnly
              rows={Math.min(14, text.split('\n').length)}
              value={text}
              onFocus={(e) => e.currentTarget.select()}
              className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 font-mono text-xs text-zinc-300"
            />
          )}
        </div>
      </CardBody>
    </Card>
  )
}
