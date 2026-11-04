// Panel "Reglas del perfil" read-only (plan Apex Fase 6): bloques del perfil
// extendido (trailing + safety net, DLL por tier, escalera de retiros,
// consistencia, cualificados, inactividad, caps). onboarding-b1: también las
// reglas de los perfiles FTMO CFD (`rules.model === 'cycle'`: fases + fondeada).
// Los avisos del modelo ya NO viven aquí: pasan a ModelNotesPanel
// ("¿Cómo se ha calculado?"). Sin editor (el editor de perfiles es otra vista).
import type { CycleRules, ExtendedRules, Firm } from '../lib/api'
import { Badge } from './ui'

const usd = (v: number | null | undefined) =>
  v == null ? '—' : `$${v.toLocaleString('es-ES', { maximumFractionDigits: 0 })}`

function Block({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-950/50 px-3 py-2.5">
      <div className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-zinc-500">
        {title}
      </div>
      <div className="space-y-1 text-xs text-zinc-300">{children}</div>
    </div>
  )
}

function Row({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <p>
      <span className="text-zinc-500">{k}:</span> <span className="num">{v}</span>
    </p>
  )
}

const pctTxt = (v: number) => `${v.toLocaleString('es-ES', { maximumFractionDigits: 2 })}%`

// Reglas del perfil FTMO CFD (modelo `cycle`): Fase 1, Fase 2 y fondeada.
function CycleRulesView({ rules }: { rules: CycleRules }) {
  const f = rules.funded
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        {rules.ruleset_version && (
          <Badge variant="outline">ruleset {rules.ruleset_version}</Badge>
        )}
        <Badge variant="outline">split {f.profit_split_pct}%</Badge>
      </div>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
        {rules.phases.map((ph, i) => (
          <Block key={ph.name} title={`Fase ${i + 1} · ${ph.name}`}>
            <Row k="Objetivo" v={pctTxt(ph.profit_target_pct)} />
            <Row k="Pérdida diaria máx." v={pctTxt(ph.max_daily_loss_pct)} />
            <Row k="Pérdida máxima" v={pctTxt(ph.max_loss_pct)} />
            <Row k="Días mínimos de trading" v={ph.min_trading_days} />
          </Block>
        ))}
        <Block title="Cuenta fondeada">
          <Row k="Reparto del beneficio" v={`${f.profit_split_pct}%`} />
          <Row k="Ciclo de cobro" v={`${f.payout_cycle_days} días`} />
          <Row
            k="Beneficio mínimo para cobrar"
            v={`${pctTxt(f.min_payout_profit_pct)} de la cuenta`}
          />
          <Row
            k="Tope de la simulación"
            v={`${f.max_payout_cycles} ciclos ≈ ${f.horizon_cap_days} días — truncado del modelo, no regla de la firma`}
          />
          <Row
            k="Reembolso del fee"
            v={rules.refund_fee_on_first_payout ? 'sí, con el primer cobro' : 'no'}
          />
        </Block>
      </div>
    </div>
  )
}

export function ProfileRulesPanel({ firm }: { firm: Firm }) {
  const rules = firm.rules
  if (!rules) return null
  if (rules.model === 'cycle') return <CycleRulesView rules={rules} />
  return <ExtendedRulesView rules={rules} />
}

function ExtendedRulesView({ rules }: { rules: ExtendedRules }) {
  const ev = rules.evaluation
  const f = rules.funded
  const isFutures = f.model === 'futures'
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        {rules.ruleset_version && (
          <Badge variant="outline">ruleset {rules.ruleset_version}</Badge>
        )}
        {rules.day_axis && (
          <Badge variant="outline">
            eje: sesiones ET · corte {rules.day_axis_cut_hhmm ?? '18:00'}
          </Badge>
        )}
        {f.activation && (
          <Badge variant="outline">
            activación {usd(f.activation.fee_usd)} (deadline {f.activation.deadline_days}d)
          </Badge>
        )}
        <Badge variant="outline">split {f.profit_split_pct}%</Badge>
      </div>

      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
        <Block title="Evaluación">
          <Row k="Target" v={usd(ev.profit_target_usd)} />
          {isFutures && ev.drawdown && (
            <Row
              k="Drawdown EOD-trailing"
              v={`${usd(ev.drawdown.amount_usd)} · lock (cap) ${usd(
                ev.drawdown.lock_at_balance,
              )} · breach ${ev.drawdown.breach_on}`}
            />
          )}
          {!isFutures && ev.trailing_usd != null && (
            <Row k="Trailing intradía (equity)" v={usd(ev.trailing_usd)} />
          )}
          {isFutures && ev.daily_loss_usd != null && (
            <Row
              k="Daily loss (hard fatal)"
              v={`${usd(ev.daily_loss_usd)} por día — la cuenta muere al primer toque`}
            />
          )}
          {isFutures && ev.consistency && (
            <Row
              k="Consistencia (puerta de pase)"
              v={`mejor día ≤ ${ev.consistency.max_best_day_share * 100}%${
                ev.consistency.strict ? ' (estricto)' : ' (inclusiva)'
              } — retrasa el pase, nunca breach`}
            />
          )}
          {isFutures && ev.min_trading_days != null && (
            <Row k="Min días de trading" v={ev.min_trading_days} />
          )}
          {ev.lifetime_days != null && (
            <Row
              k="Lifetime"
              v={`${ev.lifetime_days} días naturales · expirar = ${
                ev.on_horizon_exhausted === 'fail' ? 'fee perdido' : 'excluido'
              }`}
            />
          )}
          {ev.max_contracts && (
            <Row
              k="Cap contratos"
              v={`${ev.max_contracts.max_mini} mini / ${ev.max_contracts.max_micro} micro`}
            />
          )}
        </Block>

        {!isFutures && f.trailing && (
          <Block title="Trailing de la PA">
            <Row k="Importe" v={usd(f.trailing.amount_usd)} />
            <Row
              k="Congelado"
              v={
                f.trailing.lock_at_balance != null
                  ? `al alcanzar ${usd(f.trailing.lock_at_balance)} → suelo fijo ${usd(
                      f.trailing.frozen_floor,
                    )} (safety net)`
                  : '—'
              }
            />
          </Block>
        )}

        {!isFutures && f.dll_by_tier_usd && (
          <Block title={`Daily loss (pausa, no fatal · ${f.dll_post_hit ?? 'clamp'})`}>
            <Row k="Por tier" v={f.dll_by_tier_usd.map((v) => usd(v)).join(' / ')} />
            <p className="text-zinc-500">
              Hit = liquidación + pausa hasta las 18:00 ET; PnL de sesión clampado a −DLL.
            </p>
          </Block>
        )}

        {/* Fase 3 (FTMO Futures): suscripción, drawdown/dll EOD y payouts */}
        {isFutures && f.subscription && (
          <Block title="Suscripción">
            <Row
              k="Mensual / reset de pago"
              v={`${usd(f.subscription.monthly_fee_usd)} / ${usd(f.subscription.reset_fee_usd)}`}
            />
            <Row
              k="Ciclo de facturación"
              v={`${f.subscription.billing_period_days} días, anclado a t=0; cancelada al pasar`}
            />
            <Row
              k="Retraso de la funded"
              v={`${f.subscription.funded_start_delay_days} días tras el pase`}
            />
            <Row
              k="Cap campaña / funded"
              v={`${f.subscription.campaign_cap_days}d / ${f.subscription.funded_cap_days}d — truncado del modelo, no regla de la firma`}
            />
          </Block>
        )}

        {isFutures && f.drawdown && (
          <Block title="Drawdown de la funded (EOD-trailing)">
            <Row k="Importe" v={usd(f.drawdown.amount_usd)} />
            <Row
              k="Lock (cap del umbral)"
              v={`${usd(f.drawdown.lock_at_balance)} = capital inicial → lock permanente`}
            />
            <Row k="Breach" v={`${f.drawdown.breach_on} intradía (balance + flotante)`} />
          </Block>
        )}

        {isFutures && f.dll && (
          <Block title={`Daily loss funded (${f.dll.type})`}>
            <Row k="Importe" v={usd(f.dll.amount_usd)} />
            <p className="text-zinc-500">
              {f.dll.type === 'hard_fatal'
                ? 'La cuenta muere al primer toque.'
                : 'Cierra todo y suspende el día (PnL clampado al suelo); la cuenta sigue viva.'}
            </p>
          </Block>
        )}

        {isFutures && f.payouts_futures && (
          <Block title="Payouts (Sim-Funded)">
            <Row
              k="Qualifying day"
              v={`≥ ${usd(f.payouts_futures.qualifying_day_min_usd)} de P&L cerrado EOD`}
            />
            <Row k="Días por payout" v={f.payouts_futures.days_per_payout} />
            <Row
              k="Retirable"
              v={`${f.payouts_futures.withdraw_share * 100}% del profit sobre el inicial`}
            />
            <Row k="Cap por request" v={usd(f.payouts_futures.request_cap_usd)} />
            <Row
              k="Mín request / buffer"
              v={`${usd(f.payouts_futures.min_payout_usd)} / ${usd(f.payouts_futures.buffer_usd)}`}
            />
            <Row k="Política (default)" v={f.payouts_futures.withdrawal_policy} />
          </Block>
        )}

        {f.scaling_tiers.length > 0 && (
          <Block title="Escalera de contratos (tiers PA)">
            {f.scaling_tiers.map((t) => (
              <Row
                key={t.level}
                k={`N${t.level} · ${usd(t.profit_lo)}–${t.profit_hi == null ? '∞' : usd(t.profit_hi)}`}
                v={`${t.max_mini} mini / ${t.max_micro} micro`}
              />
            ))}
          </Block>
        )}

        {f.payouts && (
          <Block title="Escalera de retiros">
            <Row k="Escalera" v={f.payouts.ladder_usd.map((v) => usd(v)).join(' · ')} />
            <Row
              k="Mín / suelo residual"
              v={`${usd(f.payouts.min_payout_usd)} / ${usd(f.payouts.residual_floor)}`}
            />
            <Row k="1er retiro exige" v={usd(f.payouts.min_balance_first)} />
            <Row
              k="Días cualificados"
              v={`${f.payouts.days_per_payout} días ≥ ${usd(f.payouts.qualifying_day_min_usd)} por retiro`}
            />
            <Row k="Máx retiros" v={f.payouts.max_payouts} />
            <Row
              k="Consistencia"
              v={`mejor día < ${f.payouts.consistency.max_best_day_share * 100}%${
                f.payouts.consistency.strict ? ' (estricto)' : ''
              } · ${f.payouts.consistency.window}`}
            />
          </Block>
        )}

        {f.inactivity && (
          <Block title="Inactividad">
            <Row
              k="Regla"
              v={`≥ ${f.inactivity.min_days} días con beneficio ≥ ${usd(
                f.inactivity.min_profit_usd,
              )} en ventana rolling de ${f.inactivity.window_days} d naturales`}
            />
          </Block>
        )}
      </div>
    </div>
  )
}
