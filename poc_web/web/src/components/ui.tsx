// Primitivas de UI estilo shadcn (hand-rolled, Tailwind).
import { clsx } from 'clsx'
import { useEffect, useRef, useState } from 'react'
import type { ReactNode } from 'react'

export function Card({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={clsx('rounded-xl border border-zinc-800 bg-zinc-900 shadow-sm', className)}>
      {children}
    </div>
  )
}

export function CardHeader({ title, subtitle, right }: { title: string; subtitle?: string; right?: ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-3 border-b border-zinc-800 px-4 py-3">
      <div>
        <h3 className="text-sm font-semibold text-zinc-100">{title}</h3>
        {subtitle && <p className="mt-0.5 text-xs text-zinc-500">{subtitle}</p>}
      </div>
      {right}
    </div>
  )
}

export function CardBody({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={clsx('px-4 py-3', className)}>{children}</div>
}

type BadgeVariant = 'default' | 'success' | 'danger' | 'warn' | 'outline'

export function Badge({ children, variant = 'default', className }: { children: ReactNode; variant?: BadgeVariant; className?: string }) {
  const styles: Record<BadgeVariant, string> = {
    default: 'bg-zinc-800 text-zinc-300 border-zinc-700',
    success: 'bg-emerald-950 text-emerald-400 border-emerald-800',
    danger: 'bg-rose-950 text-rose-400 border-rose-800',
    warn: 'bg-amber-950 text-amber-400 border-amber-800',
    outline: 'bg-transparent text-zinc-400 border-zinc-700',
  }
  return (
    <span className={clsx('inline-flex items-center rounded-md border px-1.5 py-0.5 text-[11px] font-medium', styles[variant], className)}>
      {children}
    </span>
  )
}

export function Button({ children, onClick, disabled, className }: { children: ReactNode; onClick?: () => void; disabled?: boolean; className?: string }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={clsx(
        'inline-flex w-full items-center justify-center gap-2 rounded-md bg-emerald-600 px-4 py-2 text-sm font-semibold text-zinc-50',
        'transition-colors hover:bg-emerald-500 disabled:cursor-not-allowed disabled:opacity-50',
        className,
      )}
    >
      {children}
    </button>
  )
}

export function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs font-medium text-zinc-400" title={hint}>{label}</span>
      {children}
    </label>
  )
}

const inputCls =
  'w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-sm text-zinc-100 ' +
  'focus:border-emerald-600 focus:outline-none'

// Parseo tolerante para los drafts de NumberInput: acepta coma y punto como
// separador decimal (teclado ES vs API que espera punto) y devuelve null ante
// basura/vacío sin romper el estado — el caller conserva el último valor.
function parseLoose(s: string): number | null {
  const t = s.trim().replace(',', '.')
  if (!t) return null
  const v = Number(t)
  return Number.isFinite(v) ? v : null
}

function fmtNum(v: number): string {
  return Number.isFinite(v) ? String(parseFloat(v.toPrecision(10))) : ''
}

export function NumberInput({ value, onChange, step, min, max }: { value: number; onChange: (v: number) => void; step?: number; min?: number; max?: number }) {
  // Mientras el campo tiene foco manda el draft (string): el campo puede
  // quedarse vacío a mitad de edición sin que salte un 0 fantasma (antes:
  // input number nativo con Number("")=0 repintando en cada tecla). El estado
  // solo se compromete cuando el draft parsea.
  const [draft, setDraft] = useState<string | null>(null)
  const shown = draft ?? fmtNum(value)
  const clamp = (v: number) => Math.min(max ?? Infinity, Math.max(min ?? -Infinity, v))
  return (
    <input
      type="text"
      inputMode="decimal"
      className={inputCls}
      value={shown}
      onFocus={() => setDraft(fmtNum(value))}
      onChange={(e) => {
        setDraft(e.target.value)
        const v = parseLoose(e.target.value)
        if (v !== null) onChange(clamp(v))
        // con draft no parseable no comprometemos: el usuario sigue escribiendo
      }}
      onKeyDown={(e) => {
        if (!step || (e.key !== 'ArrowUp' && e.key !== 'ArrowDown')) return
        e.preventDefault()
        const cur = parseLoose(draft ?? '') ?? value
        const nv = clamp(cur + (e.key === 'ArrowUp' ? step : -step))
        setDraft(fmtNum(nv))
        onChange(nv)
      }}
      onBlur={() => {
        if (draft === null) return
        const v = parseLoose(draft)
        setDraft(null)
        if (v !== null) onChange(clamp(v)) // normaliza al salir: coma→punto, respeta min/max
      }}
    />
  )
}

export function TextInput({ value, onChange, type = 'text', placeholder }: { value: string; onChange: (v: string) => void; type?: string; placeholder?: string }) {
  return (
    <input
      type={type}
      className={inputCls}
      value={value}
      placeholder={placeholder}
      onChange={(e) => onChange(e.target.value)}
    />
  )
}

export function TextArea({ value, onChange, rows = 3, placeholder, maxLength }: { value: string; onChange: (v: string) => void; rows?: number; placeholder?: string; maxLength?: number }) {
  return (
    <textarea
      className={inputCls}
      rows={rows}
      value={value}
      placeholder={placeholder}
      maxLength={maxLength}
      onChange={(e) => onChange(e.target.value)}
    />
  )
}

// f es fracción interna (convención del motor); la UI la muestra/edita en
// puntos porcentuales (0.0025 ↔ 0.25). step/min se pasan en puntos %. Se
// apoya en NumberInput (tolerante coma/punto, sin 0 fantasma) y convierte
// unidades alrededor.
export function PercentInput({ value, onChange, step, min }: { value: number; onChange: (v: number) => void; step?: number; min?: number }) {
  return (
    <NumberInput
      value={parseFloat((value * 100).toFixed(6))}
      onChange={(v) => onChange(v / 100)}
      step={step}
      min={min}
    />
  )
}

// perfiles-v2: `group` opcional por opción -> <optgroup> (las opciones del mismo grupo
// deben ir seguidas; sin `group` el comportamiento es el de siempre).
export function Select<T extends string>({ value, onChange, options }: { value: T; onChange: (v: T) => void; options: { value: T; label: string; group?: string }[] }) {
  const opt = (o: { value: T; label: string }) => (
    <option key={o.value} value={o.value}>{o.label}</option>
  )
  const groups: { name: string | null; items: typeof options }[] = []
  for (const o of options) {
    const name = o.group ?? null
    const last = groups[groups.length - 1]
    if (last && last.name === name) last.items.push(o)
    else groups.push({ name, items: [o] })
  }
  return (
    <select className={inputCls} value={value} onChange={(e) => onChange(e.target.value as T)}>
      {groups.map((g, i) =>
        g.name == null ? (
          g.items.map(opt)
        ) : (
          <optgroup key={`${g.name}-${i}`} label={g.name}>
            {g.items.map(opt)}
          </optgroup>
        ),
      )}
    </select>
  )
}

// perfiles-v2: control segmentado (p. ej. CFD | Futuros).
export function Segmented<T extends string>({ value, onChange, options }: { value: T; onChange: (v: T) => void; options: { value: T; label: string }[] }) {
  return (
    <div className="inline-flex rounded-md border border-zinc-700 p-0.5" role="group">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          aria-pressed={o.value === value}
          onClick={() => onChange(o.value)}
          className={clsx(
            'rounded px-3 py-1 text-xs font-medium transition-colors',
            o.value === value ? 'bg-emerald-950 text-emerald-300' : 'text-zinc-400 hover:bg-zinc-800',
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

// perfiles-v2: menú desplegable `⋯` (se cierra al hacer clic fuera o con Escape).
export function MenuButton({ items, label = 'Más acciones' }: { items: { label: string; onClick: () => void; disabled?: boolean; title?: string }[]; label?: string }) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])
  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        aria-label={label}
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        className="rounded-md border border-zinc-700 px-2.5 py-1 text-xs font-medium text-zinc-300 hover:bg-zinc-800"
      >
        ⋯
      </button>
      {open && (
        <div role="menu" className="absolute right-0 z-40 mt-1 min-w-56 rounded-md border border-zinc-700 bg-zinc-900 py-1 shadow-xl">
          {items.map((it) => (
            <button
              key={it.label}
              type="button"
              role="menuitem"
              disabled={it.disabled}
              title={it.title}
              onClick={() => {
                setOpen(false)
                it.onClick()
              }}
              className="block w-full px-3 py-1.5 text-left text-xs text-zinc-200 hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {it.label}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

export function Checkbox({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: ReactNode }) {
  return (
    <label className="flex cursor-pointer items-center gap-2 rounded-md px-1.5 py-1 text-sm text-zinc-300 hover:bg-zinc-800/60">
      <input
        type="checkbox"
        className="size-3.5 accent-emerald-600"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
      {label}
    </label>
  )
}

export function Slider({ value, onChange, min, max, step }: { value: number; onChange: (v: number) => void; min: number; max: number; step: number }) {
  return (
    <input
      type="range"
      className="w-full accent-emerald-600"
      value={value}
      min={min}
      max={max}
      step={step}
      onChange={(e) => onChange(Number(e.target.value))}
    />
  )
}

export function Collapsible({ title, subtitle, children, defaultOpen = false, onOpenChange }: { title: string; subtitle?: string; children: ReactNode; defaultOpen?: boolean; onOpenChange?: (open: boolean) => void }) {
  return (
    <details
      className="group rounded-xl border border-zinc-800 bg-zinc-900"
      open={defaultOpen}
      onToggle={(e) => onOpenChange?.((e.target as HTMLDetailsElement).open)}
    >
      <summary className="flex cursor-pointer list-none items-center gap-2 px-4 py-3 text-sm font-semibold text-zinc-200 marker:hidden [&::-webkit-details-marker]:hidden">
        <span className="text-zinc-500 transition-transform group-open:rotate-90">▸</span>
        {title}
        {subtitle && <span className="text-xs font-normal text-zinc-500">{subtitle}</span>}
      </summary>
      <div className="border-t border-zinc-800 px-4 py-3">{children}</div>
    </details>
  )
}

export function Kpi({ label, value, sub, tone = 'neutral', hint }: { label: string; value: string; sub?: string; tone?: 'neutral' | 'success' | 'danger'; hint?: string }) {
  const tones = {
    neutral: 'text-zinc-100',
    success: 'text-emerald-400',
    danger: 'text-rose-400',
  }
  return (
    <Card className="px-4 py-3">
      {/* onboarding-b1 (enmienda 8): `hint` = supuestos que más mueven el número (title) */}
      <div
        className={clsx('text-[11px] font-medium uppercase tracking-wide text-zinc-500', hint && 'cursor-help')}
        title={hint}
      >
        {label}
        {hint && <span className="ml-1 text-zinc-600">ⓘ</span>}
      </div>
      <div className={clsx('num mt-1 text-xl font-semibold', tones[tone])} title={hint}>{value}</div>
      {sub && <div className="num mt-0.5 text-xs text-zinc-500">{sub}</div>}
    </Card>
  )
}

// perfiles-v2: botón pequeño, modal y candado, compartidos por las piezas de la vista Perfiles
// (antes locales de ProfilesView).
export function SmallBtn({
  children,
  onClick,
  disabled,
  tone = 'default',
  title,
}: {
  children: ReactNode
  onClick: () => void
  disabled?: boolean
  tone?: 'default' | 'primary' | 'danger'
  title?: string
}) {
  const tones = {
    default: 'border-zinc-700 text-zinc-300 hover:bg-zinc-800',
    primary: 'border-emerald-800 text-emerald-300 hover:bg-emerald-950',
    danger: 'border-rose-900 text-rose-300 hover:bg-rose-950',
  }
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={clsx(
        'rounded-md border px-2.5 py-1 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40',
        tones[tone],
      )}
    >
      {children}
    </button>
  )
}

export function Modal({ title, children, onClose }: { title: string; children: ReactNode; onClose: () => void }) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 px-4"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose()
      }}
      onKeyDown={(e) => {
        if (e.key === 'Escape') onClose()
      }}
      role="dialog"
      aria-modal="true"
    >
      <div className="w-full max-w-md rounded-xl border border-zinc-700 bg-zinc-900 shadow-xl">
        <div className="border-b border-zinc-800 px-4 py-3 text-sm font-semibold text-zinc-100">{title}</div>
        <div className="space-y-3 px-4 py-4">{children}</div>
      </div>
    </div>
  )
}

export function LockIcon() {
  return (
    <svg
      viewBox="0 0 16 16"
      className="inline-block size-3.5 align-[-2px]"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      aria-hidden="true"
    >
      <rect x="3.5" y="7" width="9" height="6.5" rx="1.2" />
      <path d="M5.5 7V5a2.5 2.5 0 0 1 5 0v2" />
    </svg>
  )
}
