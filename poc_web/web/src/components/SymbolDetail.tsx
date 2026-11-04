// Detalle de una ficha de símbolo (guardrail entrada-4-bloques, F5): columna derecha de la vista Símbolos.
// Cada campo con su PROCEDENCIA (badge "oficial", "hoja MT5", "del .sqx (sin verificar)" o "manual") y los
// campos de `needs_review` resaltados en ámbar con "sin verificar". Las acciones (Editar, Duplicar, Exportar,
// Ocultar/Mostrar, Borrar) las gestiona SymbolsView; "Editar" y "Borrar" solo existen en las fichas tuyas.
import {
  ASSET_LABEL,
  CLOCK_LABEL,
  fmtCommission,
  fmtDate,
  fmtSheetNum,
  SOURCE_LABEL,
  type BrokerView,
  type FieldSource,
  type SourceKind,
  type SymbolView,
} from '../lib/api'
import { Badge, Card, LockIcon, SmallBtn } from './ui'

const SOURCE_VARIANT: Record<SourceKind, 'default' | 'success' | 'warn' | 'outline'> = {
  official: 'outline',
  mt5_sheet: 'default',
  sqx: 'warn',
  manual: 'success',
}

/** Procedencia de un campo: badge por origen + fecha (si la hay). Sin procedencia no pinta nada. */
export function SourceBadge({ source }: { source: FieldSource | undefined }) {
  if (!source) return null
  return (
    <span className="inline-flex flex-wrap items-center gap-1">
      <Badge variant={SOURCE_VARIANT[source.from]}>{SOURCE_LABEL[source.from]}</Badge>
      {source.date && <span className="num text-[11px] text-zinc-500">{fmtDate(source.date)}</span>}
    </span>
  )
}

interface Row {
  name: string // clave del campo (la de sources / needs_review)
  label: string
  value: string
  unit?: string
}

function rowsOf(s: SymbolView): Row[] {
  const lots = (v: number | null) => (v == null ? '—' : fmtSheetNum(v))
  return [
    { name: 'point_value_usd', label: 'Valor del punto', value: fmtSheetNum(s.point_value_usd), unit: 'USD por 1,0 de precio y lote' },
    { name: 'lot_step', label: 'Paso de lote', value: lots(s.lot_step), unit: 'lotes' },
    { name: 'min_lot', label: 'Lote mínimo', value: lots(s.min_lot), unit: 'lotes' },
    { name: 'max_lot', label: 'Lote máximo', value: lots(s.max_lot), unit: s.max_lot == null ? undefined : 'lotes' },
    { name: 'commission', label: 'Comisión', value: fmtCommission(s.commission) },
    {
      name: 'spread_px',
      label: 'Spread típico',
      value: fmtSheetNum(s.spread_px),
      unit: s.spread_px == null ? undefined : 'en unidades de precio',
    },
    { name: 'ohlc_m1', label: 'Fichero OHLC M1', value: s.ohlc_m1 ?? '—' },
    { name: 'tick_size', label: 'Tamaño del tick', value: fmtSheetNum(s.tick_size), unit: s.tick_size == null ? undefined : 'precio' },
    { name: 'ticker', label: 'Ticker en el bróker', value: s.ticker || '—' },
  ]
}

export function SymbolDetail({
  sheet,
  broker,
  busy,
  onEdit,
  onDuplicate,
  onExport,
  onToggleHidden,
  onDelete,
}: {
  sheet: SymbolView | null
  broker?: BrokerView
  busy: boolean
  onEdit: () => void
  onDuplicate: () => void
  onExport: () => void
  onToggleHidden: () => void
  onDelete: () => void
}) {
  if (!sheet) {
    return (
      <Card className="flex h-48 items-center justify-center">
        <p className="text-sm text-zinc-500">Elige un símbolo a la izquierda</p>
      </Card>
    )
  }
  const official = sheet.official
  const rows = rowsOf(sheet)
  const pending = sheet.needs_review.length
  return (
    <Card>
      <div className="space-y-3 border-b border-zinc-800 px-4 py-3">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0 space-y-1">
            <div className="flex flex-wrap items-baseline gap-2">
              <h3 className="text-sm font-semibold text-zinc-100">{sheet.sqx_symbol}</h3>
              {!official && <span className="text-sm text-emerald-400">(tuyo)</span>}
              <code className="text-[11px] text-zinc-500">{sheet.broker}</code>
            </div>
            <div className="text-xs text-zinc-400">
              {broker ? `${broker.display_name} · ${ASSET_LABEL[broker.asset_class]}` : sheet.broker}
              {broker && <span className="text-zinc-500"> · {CLOCK_LABEL[broker.clock]}</span>}
            </div>
            <div className="flex flex-wrap items-center gap-1.5 text-xs">
              {official ? (
                <Badge variant="outline" className="gap-1">
                  <LockIcon />
                  oficial
                </Badge>
              ) : (
                <Badge variant="success">tuyo</Badge>
              )}
              {sheet.hidden && <Badge variant="default">oculta</Badge>}
              {!sheet.usable && <Badge variant="danger">no usable</Badge>}
              {pending > 0 && (
                <Badge variant="warn">
                  {pending === 1 ? '1 campo sin verificar' : `${pending} campos sin verificar`}
                </Badge>
              )}
              {sheet.verified_on && (
                <span className="text-zinc-400">verificada el {fmtDate(sheet.verified_on)}</span>
              )}
            </div>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {!official && (
              <SmallBtn onClick={onEdit} disabled={busy} title="Editar los datos de la ficha">
                Editar
              </SmallBtn>
            )}
            <SmallBtn
              tone="primary"
              onClick={onDuplicate}
              disabled={busy}
              title="Crea una copia editable en uno de tus brókers (con otro símbolo SQX)"
            >
              Duplicar
            </SmallBtn>
            <SmallBtn onClick={onExport} title="Descarga la ficha como JSON">
              Exportar
            </SmallBtn>
            <SmallBtn
              onClick={onToggleHidden}
              disabled={busy}
              title={sheet.hidden ? 'Vuelve a usar esta ficha' : 'Deja de usar esta ficha (se puede restaurar)'}
            >
              {sheet.hidden ? 'Mostrar' : 'Ocultar'}
            </SmallBtn>
            {!official && (
              <SmallBtn tone="danger" onClick={onDelete} disabled={busy} title="Borra la ficha (no se puede deshacer)">
                Borrar
              </SmallBtn>
            )}
          </div>
        </div>
        {!sheet.usable && (
          <p className="rounded-md border border-amber-900 bg-amber-950/40 px-3 py-2 text-xs text-amber-300">
            Esta ficha no se puede usar: {sheet.motivo ?? 'motivo no indicado'}.
          </p>
        )}
        {official && (
          <p className="text-xs text-zinc-500">
            Las fichas oficiales son de solo lectura: se ocultan o se duplican en un bróker tuyo para cambiar sus datos.
          </p>
        )}
        {sheet.notes && <p className="whitespace-pre-line text-xs text-zinc-500">{sheet.notes}</p>}
      </div>
      <div className="overflow-x-auto px-4 py-3">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-zinc-800 text-left text-[11px] uppercase tracking-wide text-zinc-500">
              <th className="px-3 py-1.5 font-medium">Campo</th>
              <th className="px-3 py-1.5 font-medium">Valor</th>
              <th className="px-3 py-1.5 font-medium">Procedencia</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const review = sheet.needs_review.includes(r.name)
              return (
                <tr key={r.name} className={`border-b border-zinc-800/60 ${review ? 'bg-amber-950/30' : ''}`}>
                  <td className={`px-3 py-1.5 ${review ? 'text-amber-300' : 'text-zinc-300'}`}>{r.label}</td>
                  <td className="num px-3 py-1.5 text-zinc-100">
                    {r.value}
                    {r.unit && <span className="ml-1.5 text-xs text-zinc-500">{r.unit}</span>}
                  </td>
                  <td className="px-3 py-1.5">
                    <div className="flex flex-wrap items-center gap-1.5">
                      <SourceBadge source={sheet.sources[r.name]} />
                      {review && <Badge variant="warn">sin verificar</Badge>}
                    </div>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
        {pending > 0 && (
          <p className="mt-2 text-xs text-amber-300">
            Los campos en ámbar vienen de una fuente sin confirmar (por ejemplo el .sqx o una comisión a 0 de la
            hoja). Confírmalos en «Editar» para que la cartera deje de avisar.
          </p>
        )}
      </div>
    </Card>
  )
}
