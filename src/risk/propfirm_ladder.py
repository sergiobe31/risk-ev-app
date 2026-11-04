"""Cuenta PA (fondeada) Apex — máquina de estados del modelo ``ladder``.

Plan Apex 150K Fase 4 (adjudicado 2026-09-14; ver
plan apex-150k y la sección FASE 4 del plan de implementación). Spec: ``docs/referencia/APEX_150K_Reglas_Motor_de_Riesgo-2.md``
§3 (reglas PA) y §5 (JSON de parámetros).

**Mecánica de anclaje (adjudicada, ronda 2).** Los arrays ``rel_*`` del
:class:`src.risk.propfirm.BlockDayArrays` son puro trading-PnL relativo al
inicio de bloque; los ajustes NO-trading (payouts y clamps de DLL) entran en
``saldo`` a fin de sesión (patrón ``saldo -= ajuste`` de
``evaluate_funded_path``:724, FTMO)::

    saldo -= payout        (al pagar — NO hay reset a baseline)
    saldo -= clamp_adj     con clamp_adj = (rel_closed[d] − rel_before[d]) − (−dll)  (si hit DLL y > 0)

así ``balance(d) = account + saldo + rel_closed[d]`` queda correcto para todos
los días posteriores. Check trailing diario::

    max(hwm − (account + saldo_inicio_día + rel_low[d]), rel_exc[d]) >= amount

con ``hwm`` actualizado por sesión como ``max(hwm, account + saldo_inicio_día +
rel_high[d])`` — el update usa el saldo de INICIO de sesión (pre-ajustes del
día) ⇒ ``hwm`` es cota superior del HWM real y el término 1 subsume
conservadoramente la parte cross-día de ``rel_exc`` tras ajustes (demostrado en
la adjudicación; NO hace falta ``rel_exc + adj``).

**Congelado (safety net).** ``hwm >= lock_at_balance`` (154,100) ⇒ el suelo
queda fijo en ``frozen_floor`` (150,100) PERMANENTE y el trailing deja de
apretar: el check pasa a ser ``low_abs <= frozen_floor`` (``rel_exc`` ya no
aplica). El congelado se evalúa sobre la cota superior del pico (``rel_high``),
mismo convenio que el update del ``hwm``.

**Orden por día-sesión** (adjudicado):

0. **Inactividad al INICIO de sesión**: si ya han transcurrido ``window_days``
   naturales completos, la ventana rolling ``(s − window, s]`` (s = inicio de
   la sesión en días desde el arranque de la PA) debe contener ``>= min_days``
   sesiones con PnL neto (clamped) ``>= min_profit_usd``; si no, muerte ANTES
   de operar la sesión. Evaluar al inicio (en vez de al cierre) cubre gaps
   largos sin trades: el calendario incluye días sin actividad vía
   ``elapsed``/``durations``/``gaps``. Supuesto documentado: SIN grace period
   (la spec no lo explicita; la fase durmiente de 15 días es solo prealerta).
1. **Trailing** (con congelado) → muerte. Precedencia sobre el DLL: el check
   trailing del propio día del hit usa los arrays CRUDOS (sobre-flag
   conservador, documentado — no "corregir").
2. **DLL** (pausa no mortal): hit si ``rel_low[d] − rel_before[d] <= −dll_hoy``
   (anclado a inicio de sesión; subsume el caso solo-realizado porque
   ``rel_low[d] <= rel_closed[d]``). Efecto ``post_hit="clamp"``: el PnL de la
   sesión queda ``min(observado, −dll)`` vía ``clamp_adj`` en ``saldo``;
   ``dll_hits++``. Caveat: liquidación forzosa sin slippage (el clamp puede
   quedarse corto frente al fill real).
3. **Tier del día siguiente** desde el balance de cierre POST-ajustes del día
   (clamp y payout incluidos: el retiro reduce el balance real del que se lee
   el rango — por eso los retiros degradan tier). Suelo nivel 1.
4. **Día cualificado**: PnL neto de sesión (clamped) >=
   ``qualifying_day_min_usd`` (300). Un día con hit DLL nunca cualifica (su
   neto clamped es <= −dll < 0). Caveat: neto de comisiones — si el backtest
   es gross, cualificados e inactividad son levemente optimistas.
5. (La inactividad se chequea al inicio de la sesión siguiente — ver paso 0.)
6. **Payout** si ``qual_days >= days_per_payout`` y hay retirable:
   - importe ``min(ladder[n_payouts], balance − residual_floor)``, exigido
     ``>= min_payout_usd``; el PRIMERO exige además ``balance >=
     min_balance_first`` (154,600). Si no se cumple, no hay deferral: se
     reevalúa cada sesión (con ``days_per_payout=5`` y umbral de cualificado
     300, ``min_payout_usd`` queda subsumido por la consistencia estricta:
     ventana > 2×300 = 600 > 500 — se mantiene el check por generalidad).
   - consistencia: ``max(pnl_sesión_ventana) / beneficio_ventana < 0.50``
     ESTRICTO (igualdad = denegado; ventana = PnL por sesión CLAMPED desde el
     último retiro o el inicio). Si falla → ``consistency_deferrals++`` y
     reevaluación cada sesión. SUPUESTO A VERIFICAR contra el help center
     (ligado a ``ruleset_version``): los ``qual_days`` se CONSERVAN tras la
     denegación (no se resetea el ciclo).
   - al pagar: ``saldo -= payout`` (importe íntegro; en el array se registra
     ``payout × profit_split`` — con split 100% es identidad, pero se aplica
     igualmente por generalidad), reset de ventana y ``qual_days``;
     ``n_payouts == max_payouts`` ⇒ cierre terminal feliz
     (``closed_after_ladder=True``).
7. **Horizonte agotado vivo** ⇒ ``alive_at_cap`` (convención de truncado:
   ``max_payout_cycles`` solo fija el ancho del array de payouts para encajar
   con :class:`src.risk.propfirm.FundedOutcomes`; el truncado real es el fin
   del backtest — el path no murió ni cerró la escalera).

Nunca hereda barreras de fase (``PropFirmPhase``): recibe solo
``cfg_funded`` (``FundedRules``). FTMO (``model="cycle"``) sigue por
``evaluate_funded_path``, byte-idéntico — el dispatch vive en
:func:`src.risk.propfirm.sweep_funded` (import lazy de este módulo para evitar
circularidad).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.risk.propfirm import BlockDayArrays, FundedRules


@dataclass
class LadderPathResult:
    """Resultado por path de la PA Apex (modelo ladder)."""

    payouts: np.ndarray          # (max_payout_cycles,) USD; ceros tras morir/cerrar
    days_to_first_payout: float  # nan si ninguno
    alive_at_cap: bool           # vivo al agotar el horizonte (truncado)
    dll_hits: int
    consistency_deferrals: int
    inactivity_death: bool
    closed_after_ladder: bool    # cierre terminal feliz tras el último retiro
    payout_days: np.ndarray      # (max_payouts,) día (elapsed+1) de cada retiro; nan
    days: float                  # días naturales hasta el stop/cierre/fin del horizonte


def _tier_level(tiers, profit_close: float) -> int:
    """Nivel de scaling por beneficio acumulado (suelo 1; huecos → nivel previo)."""
    lvl = 1
    for t in tiers:
        if profit_close >= t.profit_lo:
            lvl = max(lvl, t.level)
    return lvl


def evaluate_funded_path_ladder(
    block_order: np.ndarray,
    day_arrays: list[BlockDayArrays],
    durations: np.ndarray,
    gaps: np.ndarray | None = None,
    *,
    cfg_funded: FundedRules,
    account_size: float,
    profit_split: float,
) -> LadderPathResult:
    """Simula la PA Apex sobre un path: trailing con congelado + DLL por tier
    (pausa con clamp) + días cualificados + consistencia estricta + escalera de
    retiros + inactividad rolling. Ver el docstring del módulo.

    ``gaps`` (opcional): días flat por transición de bloque (alineado con
    ``block_order``), mismo convenio que ``evaluate_funded_path`` — el
    calendario de la inactividad incluye días sin trades.

    Bloque sin arrays trailing (``rel_low``/``rel_high``/``rel_exc`` a None) →
    raise ruidoso (nunca degradación silenciosa): precomputar con
    ``block_day_arrays_portfolio(..., trailing=True)``.
    """
    pay = cfg_funded.payouts
    dd = cfg_funded.drawdown
    if pay is None or dd is None:
        raise ValueError(
            "evaluate_funded_path_ladder exige cfg_funded.payouts y "
            "cfg_funded.drawdown (modelo ladder; para 'cycle' usa "
            "evaluate_funded_path)"
        )
    if pay.max_payouts > cfg_funded.max_payout_cycles:
        raise ValueError(
            f"payouts.max_payouts ({pay.max_payouts}) > max_payout_cycles "
            f"({cfg_funded.max_payout_cycles}): el array de payouts no cabe"
        )
    dll = cfg_funded.daily_loss
    inact = cfg_funded.inactivity
    tiers = cfg_funded.scaling_tiers
    cons = pay.consistency
    amount = float(dd.amount_usd)
    lock = dd.lock_at_balance
    frozen_floor = dd.frozen_floor

    payouts = np.zeros(cfg_funded.max_payout_cycles)
    payout_days = np.full(pay.max_payouts, np.nan)
    days_first = float("nan")
    saldo = 0.0
    hwm = float(account_size)
    frozen = False
    dll_hoy = float(dll.by_tier_usd[0]) if dll is not None else None
    qual_days = 0
    window_pnl: list[float] = []          # PnL por sesión (clamped) desde el último retiro
    act_marks: list[tuple[float, bool]] = []  # (fin de sesión, día activo) para inactividad
    n_payouts = 0
    dll_hits = 0
    deferrals = 0
    dead = False
    inactivity_death = False
    closed = False
    elapsed_prev = 0.0
    total_days = 0.0

    for pos, bi in enumerate(block_order):
        ba = day_arrays[bi]
        if ba.rel_low is None or ba.rel_high is None or ba.rel_exc is None:
            raise ValueError(
                f"bloque {bi} sin arrays trailing (rel_low/rel_high/rel_exc a "
                "None): la PA ladder exige block_day_arrays_portfolio(..., "
                "trailing=True) — nunca degradación silenciosa"
            )
        day0 = int(ba.day_ord[0]) if len(ba.day_ord) else 0
        for j in range(len(ba.day_ord)):
            s = elapsed_prev + float(int(ba.day_ord[j]) - day0)  # inicio de sesión
            e = s + 1.0                                          # fin de sesión
            # 0) Inactividad al INICIO de sesión (ventana completa que cierra
            #    ayer): muere antes de operar; cubre gaps largos sin trades.
            if inact is not None and s >= inact.window_days:
                lo = s - inact.window_days
                n_act = sum(1 for (em, a) in act_marks if a and lo < em <= s)
                if n_act < inact.min_days:
                    dead = True
                    inactivity_death = True
                    total_days = s  # muere al abrir la sesión (18:00 ET)
                    break
            s_start = account_size + saldo  # ancla de INICIO de sesión (pre-ajustes del día)
            total_days = e
            # 1) Trailing: hwm por sesión con el saldo de inicio; congelado
            #    permanente al cruzar lock_at_balance; precedencia sobre DLL.
            hwm = max(hwm, s_start + float(ba.rel_high[j]))
            if not frozen and lock is not None and hwm >= lock:
                frozen = True
            low_abs = s_start + float(ba.rel_low[j])
            if frozen:
                breach = low_abs <= frozen_floor
            else:
                breach = max(hwm - low_abs, float(ba.rel_exc[j])) >= amount
            if breach:
                dead = True
                break
            # 2) DLL (pausa no mortal; clamp conservador, sin slippage).
            rel_before = float(ba.rel_before[j])
            rel_closed = float(ba.rel_closed[j])
            session_net = rel_closed - rel_before
            if dll is not None and float(ba.rel_low[j]) - rel_before <= -dll_hoy:
                dll_hits += 1
                clamp_adj = session_net + dll_hoy
                if clamp_adj > 0.0:
                    session_net -= clamp_adj
                    saldo -= clamp_adj  # PnL sesión := min(observado, −dll)
            # 4) Día cualificado (neto de sesión clamped; un día con hit DLL
            #    queda <= −dll < umbral y nunca cualifica).
            if session_net >= pay.qualifying_day_min_usd:
                qual_days += 1
            window_pnl.append(session_net)
            if inact is not None:
                act_marks.append((e, session_net >= inact.min_profit_usd))
            # 6) Payout (consistencia estricta sobre la ventana clamped).
            if n_payouts < pay.max_payouts and qual_days >= pay.days_per_payout:
                balance = account_size + saldo + rel_closed
                amount_req = min(pay.ladder_usd[n_payouts],
                                 balance - pay.residual_floor)
                eligible = (
                    amount_req >= pay.min_payout_usd
                    and (n_payouts > 0 or balance >= pay.min_balance_first)
                )
                if eligible:
                    wsum = float(sum(window_pnl))
                    best = max(window_pnl)
                    ok = wsum > 0.0 and (
                        best < cons.max_best_day_share * wsum
                        if cons.strict
                        else best <= cons.max_best_day_share * wsum
                    )
                    if not ok:
                        deferrals += 1  # qual_days se CONSERVAN (supuesto, ver docstring)
                    else:
                        payouts[n_payouts] = amount_req * profit_split
                        payout_days[n_payouts] = e
                        saldo -= amount_req  # NO reset a baseline
                        if np.isnan(days_first):
                            days_first = e
                        window_pnl = []
                        qual_days = 0
                        n_payouts += 1
                        if n_payouts == pay.max_payouts:
                            closed = True  # cierre terminal feliz tras el último retiro
            # 3) Tier del día siguiente desde el balance de cierre POST-ajustes
            #    del día (clamp y payout incluidos — los retiros degradan tier).
            if dll is not None and tiers:
                lvl = _tier_level(tiers, saldo + rel_closed)
                dll_hoy = float(dll.by_tier_usd[min(lvl, len(dll.by_tier_usd)) - 1])
            if closed:
                break
        if dead or closed:
            break
        # saldo acumula el PnL del bloque; los ajustes mid-bloque (payouts y
        # clamps) ya están restados, así que sumar total_pnl cuadra (patrón FTMO).
        saldo += ba.total_pnl
        elapsed_prev += durations[bi] + (gaps[pos] if gaps is not None else 0.0)
    return LadderPathResult(
        payouts=payouts,
        days_to_first_payout=days_first,
        alive_at_cap=not dead and not closed,
        dll_hits=dll_hits,
        consistency_deferrals=deferrals,
        inactivity_death=inactivity_death,
        closed_after_ladder=closed,
        payout_days=payout_days,
        days=total_days,
    )
