"""Código específico FTMO Futures (plan ``ftmo-futures-plan``, aprobado
2026-09-29): helpers puros, walker de evaluación EOD-trail (Fase 2) y, en
fases posteriores, máquina funded + economía de suscripción.

Convenciones adjudicadas en el plan (§2.2/§2.9) y PINEADAS por tests:

- Los arrays ``rel_*`` son relativos al ANCLA del bloque; ``S_d =
  account_size + saldo`` (ancla + saldo, NO balance de inicio de día).
- El umbral del día d usa SOLO cierres de balance ESTRICTAMENTE anteriores:
  el update del HWM va DESPUÉS del check de breach del día d (§2.9.2).
- Breach al TOCAR el umbral (``<=``). Granularidad M1.
- Caveat comisiones (plan v2 C2, ligado a ``ruleset_version``): ``rel_low``
  es GROSS de comisiones; la equity de la regla descuenta comisiones → el
  chequeo de breach es levemente OPTIMISTA (mismo caveat declarado del
  ladder Apex).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.risk.propfirm import (
    BlockDayArrays,
    PhasePathResult,
    PropFirmPhase,
)

# Tolerancia del borde inclusivo de la consistencia ("40% or less" = al tocar
# pasa; §2.9.7 del plan: comparación con tolerancia 1e-9 para no perder el
# borde por flotación — p.ej. 2000/0.40 = 5000.000000000001 en float).
_CONSISTENCY_TOL = 1e-9


def dll_eff_low(rel_low: float, rel_before: float, dll: float):
    """Trigger DLL + peor punto efectivo del día (plan §2.5 paso 2, v2 #1).

    Trigger (con el ``rel_low`` CRUDO, §2.9.13): ``hit ⟺ rel_low − rel_before
    <= −dll``. Al dispararse, la firm liquida todo y la equity deja de caer:
    el peor punto A EFECTOS del Max DD es el suelo del DLL, en UNIDADES DE
    BLOQUE: ``eff_low = max(rel_low, rel_before − dll)`` (la forma
    dimensionalmente correcta — ``rel_before``/``rel_low`` son relativos al
    ancla del bloque, no al inicio de día; la forma ``max(rel_low, −dll)``
    solo vale para ``rel_before = 0``). Sin hit → ``eff_low = rel_low``.
    """
    if rel_low - rel_before <= -dll:
        return max(rel_low, rel_before - dll), True
    return rel_low, False


def soft_dll_clamp(observed: float, dll: float):
    """Liquidación en el suelo del soft-DLL, signo corregido (plan v2 #5).

    La liquidación se produce EN el suelo: ``session_net := max(observado,
    −dll)`` — si el backtest cerró peor que −dll, el clamp lo MEJORA a −dll;
    si cerró mejor, el neto se conserva TAL CUAL (caso no-op: el día puede
    cualificar; supuesto declarado v2 #15). Contabilidad:
    ``clamp_adj = max(0, −dll − observado)`` sumado a ``saldo``.

    Divergencia deliberada con el clamp ``min`` del ladder Apex (cota
    conservadora, shipped y NO tocada): aquí la semántica es de liquidación
    en el suelo (adjudicado en el plan).
    """
    return max(observed, -dll), max(0.0, -dll - observed)


def payout_request_max(*, profit_over_initial: float,
                       new_profit_cycle: float, share: float, cap: float,
                       min_new_frac: float, balance: float,
                       threshold: float, policy: str,
                       buffer_usd: float) -> float:
    """Máximo retirable en un request (puro; golden oficial AC-10).

    ``request ≤ min(cap, share × profit_over_initial, new_profit_cycle /
    min_new_frac)`` — la regla min-new-profit ("≥50% de lo solicitado debe
    ser profit NUEVO del ciclo", §2.9.8: el ciclo se mide desde el último
    payout, post-clamps) — y, con ``policy == "buffer_preserving"``, además
    ``request ≤ balance − T_d − buffer_usd`` (preservar el colchón sobre el
    umbral vigente). Devuelve 0 (no retirable) si alguna cota es ≤ 0
    (p.ej. ``new_profit_cycle ≤ 0``, §2.5 paso 6).

    Cotas simultáneas y monotonía (ceteris paribus): no-decreciente en
    ``profit_over_initial`` y en ``new_profit_cycle`` (property AC-10).
    """
    request = min(float(cap), float(share) * float(profit_over_initial),
                  float(new_profit_cycle) / float(min_new_frac))
    if policy == "buffer_preserving":
        request = min(request, float(balance) - float(threshold)
                      - float(buffer_usd))
    return max(0.0, request)


@dataclass
class FuturesFundedResult:
    """Resultado de la Sim-Funded FTMO Futures (plan §2.5)."""

    payouts: np.ndarray        # (max_payout_cycles,) USD (ratio 0.9 aplicado)
    days_to_first_payout: float  # nan si ninguno
    alive_at_cap: bool
    dll_hits: int              # días con trigger DLL (soft y hard)
    payout_deferrals: int      # días elegibles con request < min_payout ($20)
    payout_overflows: int      # requests beyond max_payout_cycles (truncado)
    days: float                # días naturales hasta muerte/cap/fin de backtest
    final_balance: float       # diagnóstico: balance al cierre (conservación
                               # AC-15 y políticas AC-16; no es regla FTMO)


def evaluate_funded_path_futures(
    block_order: np.ndarray,
    day_arrays: list[BlockDayArrays],
    durations: np.ndarray,
    gaps: np.ndarray | None = None,
    *,
    cfg_funded,
    account_size: float,
    profit_split: float,           # fracción 0-1 (ratio 90/10 → 0.9)
    horizon_days: float | None,    # funded_cap_days (truncado; None = sin cap)
    withdrawal_policy: str,        # "greedy" | "buffer_preserving"
) -> FuturesFundedResult:
    """Máquina de estados de la Sim-Funded FTMO Futures (plan §2.5; v2 #1/#5).

    Arranque: cuenta nueva al capital inicial (HWM = account_size). NO hereda
    barreras de fase; recibe solo ``cfg_funded`` (+ ratio de payout).

    Orden por día-sesión (adjudicado):

    1. ``T_d`` al inicio del día desde el HWM de cierres EOD (cap = inicial
       tras el lock). El HWM se alimenta de cierres PRE-retiro (paso 7) y
       NUNCA baja.
    2. **Trigger DLL con ``rel_low`` CRUDO → ``eff_low``** (v2 #1, ver
       :func:`dll_eff_low`): al dispararse el soft-DLL la equity deja de caer
       (la firm liquida todo), así que el peor punto A EFECTOS del Max DD es
       el suelo del DLL. Pro (``hard_fatal``): hit → muerte directa.
    3. **Breach Max DD con ``eff_low``**: ``S_d + eff_low ≤ T_d`` → muerte.
       (Usar el crudo mataría cuentas que la liquidación en el suelo salvaría
       — gemelos AC-12.)
    4. **Efecto soft-DLL (Growth)**: clamp con signo ``max`` vía
       :func:`soft_dll_clamp`: ``saldo += clamp_adj``, ``dll_hits++`` (los
       hits hard también se cuentan, contados al disparar el trigger).
    5. **Qualifying day**: ``session_net`` (P&L cerrado EOD, POST-clamp si
       hubo soft-DLL) ≥ ``qualifying_day_min_usd`` → ``qual_days++``. La
       cualificación usa el neto post-clamp (v2 #15: un día con hit y cierre
       ≥ umbral CUALIFICA — supuesto declarado).
    6. **Payout** si ``qual_days ≥ days_per_payout``:
       ``new_profit_cycle`` = Σ ``session_net`` post-clamp desde el último
       payout; ``profit_over_initial = balance − account_size`` con
       ``balance = account_size + saldo + rel_closed[d]`` (saldo ya con el
       clamp del día). ``request = payout_request_max(...)`` (política).
       Si ``request ≥ min_payout_usd`` (20): el retiro rebaja el balance por
       el importe ÍNTEGRO (``saldo −= request``); el payout REGISTRADO es
       ``request × profit_split`` (ratio 90/10 sobre lo solicitado); reset de
       ``qual_days`` y ``new_profit_cycle``. Si no: ``payout_deferrals++`` y
       los ``qual_days`` SE CONSERVAN (reevaluación diaria, misma convención
       documentada que Apex). Si el request desborda ``max_payout_cycles``:
       ``payout_overflows++`` y se trata como ``alive_at_cap`` (truncado).
    7. **Update EOD**: ``HWM = max(HWM, balance PRE-retiro del día)`` — el
       cierre del día del retiro cuenta con el balance anterior al retiro
       (§2.9.3: retirar no baja el umbral; el colchón sí baja exactamente en
       el importe del retiro).
    8. **Horizonte** (``funded_cap_days``): convención espejo del trailing —
       el día se procesa entero y, si ``total_days > horizon_days`` →
       ``alive_at_cap`` (truncado reportado, v2 C1; no es regla FTMO).

    Sin consistencia en funded (textual: solo eval), sin inactividad, sin
    escalera (la cuenta vive hasta breach o cap).

    Invariante de conservación del dinero (property AC-15): en todo punto
    ``balance == account_size + Σ session_net (post-clamp) − Σ request``.

    Caveat comisiones (v2 C2): ``rel_low``/``rel_closed`` son GROSS → breach
    y cualificados levemente optimistas (ligado a ``ruleset_version``).
    Caveat liquidación soft-DLL sin slippage (el clamp puede quedarse
    corto/largo frente al fill real — mismo caveat declarado que Apex).
    """
    pay = cfg_funded.payouts_futures
    if pay is None:
        raise ValueError(
            "evaluate_funded_path_futures exige funded.payouts_futures"
        )
    dll_spec = cfg_funded.daily_loss
    if dll_spec is None or len(dll_spec.by_tier_usd) != 1:
        raise ValueError(
            "evaluate_funded_path_futures exige funded.daily_loss de UN tier"
        )
    dll = float(dll_spec.by_tier_usd[0])
    dll_hard = dll_spec.type == "hard_fatal"
    dd = cfg_funded.drawdown
    if dd is None or dd.type != "eod_trailing":
        raise ValueError(
            "evaluate_funded_path_futures exige funded.drawdown.type "
            "'eod_trailing'"
        )
    amount = float(dd.amount_usd)
    cap = dd.lock_at_balance
    buffer_usd = pay.buffer_usd if pay.buffer_usd is not None else amount
    max_cycles = int(cfg_funded.max_payout_cycles)
    if withdrawal_policy not in ("greedy", "buffer_preserving"):
        raise ValueError(
            f"withdrawal_policy '{withdrawal_policy}' no soportada "
            "(solo 'greedy' | 'buffer_preserving')"
        )

    payouts = np.zeros(max_cycles)
    hwm = float(account_size)
    saldo = 0.0
    elapsed_prev = 0.0
    qual_days = 0
    new_profit_cycle = 0.0
    dll_hits = 0
    deferrals = 0
    overflows = 0
    n_payouts = 0
    days_first = float("nan")
    total_days = 0.0
    last_balance = float(account_size)

    def _ret(alive_at_cap: bool) -> FuturesFundedResult:
        return FuturesFundedResult(
            payouts=payouts, days_to_first_payout=days_first,
            alive_at_cap=alive_at_cap, dll_hits=dll_hits,
            payout_deferrals=deferrals, payout_overflows=overflows,
            days=total_days, final_balance=last_balance,
        )

    for pos, bi in enumerate(block_order):
        ba = day_arrays[bi]
        if ba.rel_low is None:
            raise ValueError(
                f"bloque {bi} sin rel_low: la Sim-Funded exige la infra "
                "trailing por minuto (block_day_arrays_portfolio(..., "
                "trailing=True)) — nunca degradación silenciosa"
            )
        day0 = int(ba.day_ord[0]) if len(ba.day_ord) else 0
        for j in range(len(ba.day_ord)):
            # Ancla del día = account + saldo ACTUAL (con los clamps/payouts
            # de los días previos, dentro y fuera del bloque): los arrays
            # rel_* son relativos al ancla del BLOQUE y los ajustes mid-bloque
            # entran en saldo (patrón cross-bloque verificado en Apex).
            anchor = account_size + saldo
            elapsed = elapsed_prev + float(int(ba.day_ord[j]) - day0)
            total_days = elapsed + 1.0
            t_day = hwm - amount
            if cap is not None:
                t_day = min(t_day, float(cap))
            # 2) trigger DLL (rel_low CRUDO) → eff_low; los hits se cuentan al
            #    disparar (diagnóstico), antes de saber si el día mata.
            eff_low, hit = dll_eff_low(ba.rel_low[j], ba.rel_before[j], dll)
            if hit:
                dll_hits += 1
            # 3) breach Max DD con eff_low (la liquidación en el suelo frena
            #    la caída a efectos del umbral — gemelos AC-12).
            if anchor + eff_low <= t_day:
                last_balance = anchor + ba.rel_closed[j]
                return _ret(alive_at_cap=False)
            # 4) efecto DLL: hard → muerte; soft → clamp (signo max, v2 #5).
            session_net = ba.rel_closed[j] - ba.rel_before[j]
            if hit:
                if dll_hard:
                    last_balance = anchor + ba.rel_closed[j]
                    return _ret(alive_at_cap=False)
                session_net, clamp_adj = soft_dll_clamp(session_net, dll)
                saldo += clamp_adj
            balance_pre = account_size + saldo + ba.rel_closed[j]
            last_balance = balance_pre
            # 5) día cualificado (neto cerrado EOD post-clamp).
            if session_net >= pay.qualifying_day_min_usd:
                qual_days += 1
            new_profit_cycle += session_net
            # 6) payout si se alcanzaron los qualifying days.
            if qual_days >= pay.days_per_payout:
                request = payout_request_max(
                    profit_over_initial=balance_pre - account_size,
                    new_profit_cycle=new_profit_cycle,
                    share=pay.withdraw_share,
                    cap=pay.request_cap_usd,
                    min_new_frac=pay.min_new_profit_frac,
                    balance=balance_pre,
                    threshold=t_day,
                    policy=withdrawal_policy,
                    buffer_usd=buffer_usd,
                )
                if request >= pay.min_payout_usd:
                    if n_payouts >= max_cycles:
                        overflows += 1
                        return _ret(alive_at_cap=True)
                    payouts[n_payouts] = request * profit_split
                    n_payouts += 1
                    if np.isnan(days_first):
                        days_first = total_days
                    saldo -= request
                    qual_days = 0
                    new_profit_cycle = 0.0
                    # El balance post-retiro queda reflejado en el diagnóstico
                    # (conservación AC-15: baja exactamente en `request`).
                    last_balance = account_size + saldo + ba.rel_closed[j]
                else:
                    deferrals += 1   # qual_days y ciclo se CONSERVAN
            # 7) update EOD: HWM con el cierre PRE-retiro del día (§2.9.3).
            hwm = max(hwm, balance_pre)
            # 8) horizonte funded (convención espejo del trailing: el día se
            #    procesa entero; alive_at_cap reporta el truncado, v2 C1).
            if horizon_days is not None and total_days > horizon_days:
                return _ret(alive_at_cap=True)
        saldo += ba.total_pnl
        elapsed_prev += durations[bi] + (gaps[pos] if gaps is not None else 0.0)
    return _ret(alive_at_cap=False)


# ---------------------------------------------------------------------------
# Nivel 2: economía de suscripción (plan §2.6; v2 #2/#3/#4/#6/#9/#11, C1)
# ---------------------------------------------------------------------------


def to_eur(usd: float, fx_eur_usd: float) -> float:
    """Conversión única USD→EUR de los flujos de nivel 2 (v2 #9).

    Convención del motor: ``fx_eur_usd`` = USD por 1 EUR → EUR = USD / fx,
    el MISMO sentido que :func:`src.risk.propfirm.resolve_fee_eur` (los fees
    se normalizan UNA vez en el setup; los payouts pasan por aquí). Pineado
    por test: ``to_eur(110, 1.10) == 100``.
    """
    if fx_eur_usd is None or fx_eur_usd <= 0:
        raise ValueError("fx_eur_usd debe ser > 0")
    return float(usd) / float(fx_eur_usd)


def billing_count(days: float, period_days: int = 30) -> int:
    """Nº de mensualidades devengadas en ``days`` días de campaña (§2.6).

    Fechas de facturación t = 0, period, 2·period, …: la campaña viva paga en
    cada fecha estrictamente anterior a su fin, y la suscripción se CANCELA al
    pasar (no hay mensualidad en/tras el pase). ``billing_count(T) =
    ceil(T/period)`` para T > 0 — bordes pineados AC-17: 5→1, 30→1, 31→2,
    60→2, 61→3. T ≤ 0 → 0 (no ocurre: todo intento dura ≥ 1 día). Monótona
    no-decreciente con salto exacto tras cada múltiplo del periodo.
    """
    if days <= 0:
        return 0
    return int(np.ceil(float(days) / float(period_days)))


def _draw_attempt_index(seed: int, c: int, i: int, n_pool: int) -> int:
    """CRN POR ÍNDICE (v2 #3): el intento i de la campaña c sortea su path de
    una stream derivada de (seed, c, i).

    Invariante adjudicado: el sorteo del (c, i)-ésimo intento es IDÉNTICO
    bajo cualquier política e independiente de cuántos intentos consumieron
    otras campañas — el consumo secuencial de UN rng rompería el pareado
    cuando las políticas difieren en nº de intentos (hallazgo #3 de la
    revisión adversarial).

    Caveat (revisión cruzada GLM-5.3 2026-09-30): el sorteo es CON
    REEMPLAZO — una campaña puede re-dibujar el mismo path en distintos
    intentos (distinto ``i``). El invariante CRN por índice se mantiene
    (cada (c, i) tiene su stream propia); el reemplazo es correcto aquí
    porque cada intento es una cuenta nueva e independiente, muestreada de
    la misma población de paths.
    """
    if n_pool <= 0:
        raise ValueError("n_pool debe ser > 0")
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), int(c), int(i)])
    )
    return int(rng.integers(0, n_pool))


@dataclass
class _CampaignOutcome:
    """Resultado de UNA campaña (secuencia de intentos hasta pase/cap)."""

    net_eur: float
    days: float             # días_campaña (denominador EV/día, §2.6 #6)
    truncated: bool         # cap de campaña = TRUNCADO (v2 C1), no "fallida"
    ruined: bool
    passed: bool
    n_attempts: int
    n_neither: int
    n_billings: int
    n_resets: int
    payout_eur: float
    days_to_pass: float     # nan si no pass


def _simulate_campaign(*, draw, outcomes, cfg, fx_eur_usd, reset_policy,
                       stop_on_insolvency, bankroll):
    """Simula una campaña = secuencias de intentos hasta el pase o el cap.

    ``draw(i)`` inyecta el sorteo del path del intento i (en producción:
    :func:`_draw_attempt_index` — CRN por índice; en los tests: secuencias
    deterministas). ``bankroll``: lista mutable de un float compartida entre
    campañas; con ``stop_on_insolvency`` es barrera absorbente (el trader no
    puede pagar el siguiente cargo → la campaña se trunca en ruina).

    Políticas de reset (§2.6):
    - ``billing_retry`` (default, GRATIS): el intento que breachea reintenta
      en la primera fecha de facturación posterior: ``T = period·ceil(T/period)``
      (la espera está incluida en el elapsed; sin coste extra).
    - ``paid_reset``: reintento inmediato (``T += d``) con ``reset_fee`` por
      fallo; la facturación sigue anclada a t = 0 (el reset no altera el
      ciclo — textual).

    Intento ``neither`` (v2 #4): consume reloj con su duración completa de
    backtest (``days_eval1``) y la campaña continúa inmediatamente — sin
    espera de facturación ni reset (no hubo breach). Caveats: el intento
    nuevo arranca con cuenta fresca (estado del umbral reiniciado) y la
    duración es cota inferior del elapsed real (coste levemente optimista).

    Truncado (v2 C1): si ``T + d > campaign_cap_days`` el reloj se corta EN
    el cap (incluido mid-intento): la campaña tiene neto = −(mensualidades
    devengadas hasta el cap) − resets pagados, SE INCLUYE en el EV y alimenta
    ``p_truncated``. NO es una "campaña fallida": el cap de 6 meses NO es
    regla FTMO, es horizonte de truncado del modelo (decisión de diseño).

    Pase: suscripción CANCELADA en T (FAQ explícita); la fase funded arranca
    en ``T + funded_start_delay_days`` (revisión/KYC, 3 días por defecto) y
    contribuye al reloj de campaña sus días REALES hasta muerte/cap/fin de
    backtest — ``outcomes.days_funded[k]`` (propagado desde
    ``FuturesFundedResult.days`` por ``sweep_funded``; revisión cruzada
    GLM-5.3 2026-09-30) —, no el ``funded_cap_days`` fijo: ``days_campaña =
    reloj_eval + delay + days_funded[k]``. Fallback documentado: si el campo
    falta (legacy) o es nan, se usa ``funded_cap_days`` (horizonte, cota
    superior). El dinero del EV es exacto en ambos casos.

    Orden determinista de cargos el mismo día (v2 #11): (1) mensualidad
    (fecha de facturación), (2) reset (momento del fallo), (3) abono de
    payout. El abono se colapsa al día del primer payout de la funded (los
    ciclos posteriores no cambian el total ni el orden relevante).
    """
    sub = cfg.subscription
    period = float(sub.billing_period_days)
    cap_days = float(sub.campaign_cap_days)
    fcap = float(sub.funded_cap_days)
    delay = float(sub.funded_start_delay_days)
    monthly = float(sub.monthly_fee_eur)
    reset_fee = float(sub.reset_fee_eur)

    T = 0.0
    n_attempts = 0
    n_neither = 0
    n_resets = 0
    n_billings = 0
    payout_eur = 0.0
    ruined = False
    next_bill = 0.0

    def charge(amount) -> bool:
        nonlocal ruined
        if stop_on_insolvency and bankroll[0] < amount:
            ruined = True
            return False
        bankroll[0] -= amount
        return True

    def advance_billings(up_to, include_equal) -> bool:
        """Carga las fechas de facturación < up_to (o <= si include_equal)."""
        nonlocal next_bill, n_billings
        limit = up_to + (1e-9 if include_equal else -1e-9)
        while next_bill <= limit:
            if not charge(monthly):
                return False
            n_billings += 1
            next_bill += period
        return True

    def outcome(days, truncated, passed, days_to_pass):
        return _CampaignOutcome(
            net_eur=payout_eur - n_billings * monthly - n_resets * reset_fee,
            days=days, truncated=truncated, ruined=ruined, passed=passed,
            n_attempts=n_attempts, n_neither=n_neither, n_billings=n_billings,
            n_resets=n_resets, payout_eur=payout_eur, days_to_pass=days_to_pass,
        )

    # Mensualidad de t = 0 cargada al arrancar la campaña.
    if not charge(monthly):
        return outcome(days=0.0, truncated=False, passed=False,
                       days_to_pass=float("nan"))
    n_billings = 1
    next_bill = period

    i = 0
    while True:
        if n_attempts > 100_000:
            raise RuntimeError("campaña sin fin: revisa days_eval1/política")
        k = draw(i)
        i += 1
        n_attempts += 1
        d = float(outcomes.days_eval1[k])
        t_end = T + d
        if t_end > cap_days + 1e-9:
            # Truncado mid-intento (v2 C1): reloj cortado EN el cap.
            advance_billings(cap_days, include_equal=False)
            return outcome(days=cap_days, truncated=True, passed=False,
                           days_to_pass=float("nan"))
        if outcomes.pass1[k]:
            # Pase: la suscripción se cancela en t_end (fechas >= t_end no).
            advance_billings(t_end, include_equal=False)
            T = t_end
            payout_eur = to_eur(float(outcomes.funded_payouts[k].sum()),
                                fx_eur_usd)
            bankroll[0] += payout_eur   # abono (3º en el orden del día)
            fdays = fcap                # fallback: horizonte (legacy/nan)
            if outcomes.days_funded is not None:
                raw = float(outcomes.days_funded[k])
                if not np.isnan(raw):
                    fdays = raw         # días reales hasta muerte/cap/fin
            return outcome(days=T + delay + fdays, truncated=False,
                           passed=True, days_to_pass=T)
        # fail / neither: la suscripción SIGUE (facturación <= t_end).
        if not advance_billings(t_end, include_equal=True):
            return outcome(days=T, truncated=False, passed=False,
                           days_to_pass=float("nan"))
        T = t_end
        if outcomes.neither1[k]:
            n_neither += 1
            continue                    # continúa inmediatamente (v2 #4)
        # fail
        if reset_policy == "paid_reset":
            if not charge(reset_fee):   # 2º en el orden del día (v2 #11)
                return outcome(days=T, truncated=False, passed=False,
                               days_to_pass=float("nan"))
            n_resets += 1               # reintento inmediato (T ya avanzó)
        else:
            # billing_retry: espera gratis hasta la próxima fecha de
            # facturación (la fecha ES la del reintento; §2.9.4).
            T = period * np.ceil(T / period)


@dataclass
class FuturesSequenceResult:
    """Economía de suscripción sobre n_paths campañas (plan §2.6, firma v2 #11).

    EV INCONDICIONAL (v2 #2/C1): el denominador común es el índice completo
    de campañas (0..n_paths−1) — el cap trunca SIN invalidar (toda campaña
    tiene neto), así que no hay exclusión dependiente de la política y el
    estimando es el económicamente correcto (el trader paga las campañas
    truncadas). EV/campaña y EV/día sobre el MISMO set (v2 #6).

    Nota de nivel (revisión cruzada GLM-5.3 2026-09-30): las métricas funded
    heredadas (``median_days_first_payout``, ``p_payout_given_funded``,
    ``alive_at_cap_frac``, ``mean_dll_hits``, ``mean_payout_deferrals``) se
    calculan sobre el POOL de paths passed (una observación por path del
    backtest), NO sobre las campañas simuladas — una campaña passed puede
    re-dibujar paths ya contados (draw con reemplazo). El EV y sus
    contadores de campaña (``mean_attempts``, ``mean_billings``,
    ``p_truncated``, ``mean_days_campaign``) sí son por campaña.
    """

    ev_per_campaign: float      # EUR (incondicional)
    ev_per_day: float           # EUR/día = ev_per_campaign / mean_days_campaign
    ev_ci95: tuple              # IC95 bootstrap sobre el índice de campañas
    net_mean_eur: float
    net_p05: float
    net_p50: float
    net_p95: float
    p_net_negative: float
    p_truncated: float          # v2 C1: fracción de campañas truncadas en cap
    p_neither_attempts: float   # v2 #4: neither / total de intentos
    mean_billings: float
    mean_resets: float
    mean_attempts: float
    mean_days_campaign: float   # denominador pineado del EV/día (v2 #6)
    median_days_to_pass: float  # campañas passed
    median_days_first_payout: float   # paths passed con payout (días funded)
    p_payout_given_funded: float      # paths passed con payout > 0
    alive_at_cap_frac: float          # paths passed alive_at_cap
    mean_dll_hits: float
    mean_payout_deferrals: float
    p_ruin: float               # > 0 solo con stop_on_insolvency
    n_paths: int
    warnings: list


def sequence_ev_futures(outcomes, cfg, *, fx_eur_usd, n_paths, n_boot=2000,
                        initial_capital_eur, seed=0,
                        reset_policy="billing_retry",
                        stop_on_insolvency=False) -> FuturesSequenceResult:
    """EV de campañas de suscripción FTMO Futures (plan §2.6).

    Cada campaña c consume intentos sorteados por :func:`_draw_attempt_index`
    (CRN por índice: reproducible bit-exact con misma seed e idéntico bajo
    cualquier política). Exige ``days_eval1`` relleno en TODOS los outcomes
    (pass/fail/neither — v2 #4; raise ruidoso si falta) y ``cfg.subscription``
    resuelto a EUR (:func:`src.risk.propfirm.resolve_fee_eur` — raise si los
    ``*_eur`` siguen a None, patrón activation).

    ``stop_on_insolvency``: bankroll del trader compartido entre campañas
    (orden creciente de índice); barrera absorbente cuando no cubre el
    siguiente cargo en el orden mensualidad → reset → abono (v2 #11). Una
    campaña arruinada reporta su neto realizado (cargos no pagados no se
    devengan) y las siguientes caen en ruina inmediata (p_ruin reportado).
    """
    if fx_eur_usd is None or fx_eur_usd <= 0:
        raise ValueError("fx_eur_usd debe ser > 0")
    if outcomes.days_eval1 is None:
        raise ValueError(
            "sequence_ev_futures exige days_eval1 relleno en TODOS los "
            "outcomes (pass/fail/neither, v2 #4): los neither consumen "
            "reloj de campaña — vuelve a generar outcomes con "
            "simulate_challenge (campo aditivo desde el plan FTMO Futures)"
        )
    sub = cfg.subscription
    if sub is None:
        raise ValueError("sequence_ev_futures exige cfg.subscription")
    if sub.monthly_fee_eur is None or sub.reset_fee_eur is None:
        raise ValueError(
            "sequence_ev_futures exige la suscripción resuelta a EUR "
            "(resolve_fee_eur antes de llamar — patrón activation)"
        )
    if reset_policy not in ("billing_retry", "paid_reset"):
        raise ValueError(
            f"reset_policy '{reset_policy}' no soportada "
            "(solo 'billing_retry' | 'paid_reset')"
        )
    if not (len(outcomes.pass1) == len(outcomes.neither1)
            == len(outcomes.days_eval1)):
        raise ValueError("pass1/neither1/days_eval1 deben alinearse por path")

    pool = len(outcomes.pass1)
    bankroll = [float(initial_capital_eur)]
    camps = [
        _simulate_campaign(
            draw=lambda i, c=c: _draw_attempt_index(seed, c, i, pool),
            outcomes=outcomes, cfg=cfg, fx_eur_usd=fx_eur_usd,
            reset_policy=reset_policy,
            stop_on_insolvency=stop_on_insolvency, bankroll=bankroll,
        )
        for c in range(int(n_paths))
    ]
    nets = np.array([c.net_eur for c in camps])
    days = np.array([c.days for c in camps])
    attempts = np.array([c.n_attempts for c in camps])
    ev_per_campaign = float(nets.mean())
    # Denominador EV/día (v2 #6): solo campañas que consumieron reloj —
    # las arruinadas EN EL ARRANQUE (bankroll < 1 mensualidad, n_attempts=0,
    # days=0) quedan EXCLUIDAS: no representan días de producto comprometidos
    # (convención elegida en la revisión cruzada GLM-5.3 2026-09-30).
    clock_mask = attempts > 0
    mean_days = (float(days[clock_mask].mean()) if clock_mask.any()
                 else float("nan"))
    if not clock_mask.any():
        warnings0 = [
            "ALL_CAMPAIGNS_RUINED_AT_START: ninguna campaña consumió reloj "
            "(bankroll < 1 mensualidad); EV/día no es definible"
        ]
    else:
        warnings0 = []
    ev_per_day = (ev_per_campaign / mean_days
                  if mean_days and mean_days > 0 else float("nan"))

    # IC95 bootstrap sobre el índice de campaña (EV incondicional, v2 #2/C1).
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 777_777]))
    boot = nets[rng.integers(0, len(nets), size=(int(n_boot), len(nets)))
                ].mean(axis=1)
    ev_ci95 = (float(np.percentile(boot, 2.5)),
               float(np.percentile(boot, 97.5)))

    total_attempts = sum(c.n_attempts for c in camps)
    n_neither = sum(c.n_neither for c in camps)
    passed_idx = np.flatnonzero(outcomes.pass1)
    got_payout = (outcomes.funded_payouts[passed_idx].sum(axis=1) > 0) \
        if len(passed_idx) else np.array([], dtype=bool)
    warnings: list = list(warnings0)
    p_neither_att = (n_neither / total_attempts) if total_attempts else 0.0
    if p_neither_att > 0.2:
        warnings.append(
            "HIGH_NEITHER_ATTEMPTS: la fracción de intentos neither es alta "
            "— el backtest es corto para este producto (misma señal que "
            "p_unresolved clásico)"
        )
    return FuturesSequenceResult(
        ev_per_campaign=ev_per_campaign,
        ev_per_day=ev_per_day,
        ev_ci95=ev_ci95,
        net_mean_eur=float(nets.mean()),
        net_p05=float(np.percentile(nets, 5)),
        net_p50=float(np.percentile(nets, 50)),
        net_p95=float(np.percentile(nets, 95)),
        p_net_negative=float((nets < 0).mean()),
        p_truncated=float(np.mean([c.truncated for c in camps])),
        p_neither_attempts=float(p_neither_att),
        mean_billings=float(np.mean([c.n_billings for c in camps])),
        mean_resets=float(np.mean([c.n_resets for c in camps])),
        mean_attempts=float(np.mean([c.n_attempts for c in camps])),
        mean_days_campaign=mean_days,
        median_days_to_pass=float(np.nanmedian([c.days_to_pass for c in camps])),
        median_days_first_payout=(
            float(np.nanmedian(outcomes.days_first_payout[passed_idx]))
            if len(passed_idx) else float("nan")),
        p_payout_given_funded=(float(got_payout.mean()) if len(passed_idx)
                               else float("nan")),
        alive_at_cap_frac=(
            float(np.mean(outcomes.alive_at_cap[passed_idx]))
            if len(passed_idx) else float("nan")),
        mean_dll_hits=(
            float(np.mean(outcomes.dll_hits[passed_idx]))
            if len(passed_idx) and outcomes.dll_hits is not None
            else float("nan")),
        mean_payout_deferrals=(
            float(np.mean(outcomes.payout_deferrals[passed_idx]))
            if len(passed_idx) and outcomes.payout_deferrals is not None
            else float("nan")),
        p_ruin=float(np.mean([c.ruined for c in camps])),
        n_paths=int(n_paths),
        warnings=warnings,
    )


@dataclass
class PolicyComparison:
    """Comparación pareada de políticas (reset o retirada) sobre campañas.

    Validez del IC pareado (exigido por la adjudicación ronda 2, ligado a
    v2 C1): el remuestreo pareado por índice de campaña es válido porque
    TODA campaña tiene neto en ambas políticas — el cap trunca sin
    invalidar —, los intentos neither consumidos son los mismos paths (CRN
    por índice) y la retirada solo actúa dentro de la funded de campañas
    passed (no cambia el conjunto de campañas ni los denominadores). Así el
    índice común es el completo y el estimando incondicional aplica.
    """

    kind: str                   # "reset" | "withdrawal"
    label_a: str
    label_b: str
    ev_mean_a: float            # EV/campaña EUR (incondicional)
    ev_mean_b: float
    delta_ev: float             # a − b: media de las diferencias PAREADAS
    ci95_paired: tuple          # IC95 bootstrap remuestreando el índice común
    n_paths: int
    validity_note: str = (
        "IC pareado válido por v2 C1: toda campaña tiene neto en ambas "
        "políticas (el cap trunca incluyendo su coste en el EV) y los "
        "sorteos son CRN por índice — el índice común es el completo."
    )


def _campaign_nets(outcomes, cfg, *, fx_eur_usd, n_paths, seed, reset_policy,
                   stop_on_insolvency, initial_capital_eur) -> np.ndarray:
    pool = len(outcomes.pass1)
    bankroll = [float(initial_capital_eur)]
    return np.array([
        _simulate_campaign(
            draw=lambda i, c=c: _draw_attempt_index(seed, c, i, pool),
            outcomes=outcomes, cfg=cfg, fx_eur_usd=fx_eur_usd,
            reset_policy=reset_policy,
            stop_on_insolvency=stop_on_insolvency, bankroll=bankroll,
        ).net_eur
        for c in range(int(n_paths))
    ])


def compare_policies_futures(outcomes_a, outcomes_b, cfg, *, fx_eur_usd,
                             n_paths, n_boot=2000, initial_capital_eur,
                             seed=0, kind, label_a, label_b,
                             policy_a="billing_retry", policy_b=None,
                             stop_on_insolvency=False) -> PolicyComparison:
    """ΔEV + IC95 bootstrap PAREADO sobre el índice común de campañas (v2 #2).

    - ``kind="reset"``: ``outcomes_a is outcomes_b`` (mismos paths); las
      políticas son ``policy_a``/``policy_b`` (billing_retry vs paid_reset).
    - ``kind="withdrawal"``: outcomes de DOS barridos de la funded sobre los
      mismos day_arrays (greedy vs buffer_preserving, vía sweep_funded con
      configs reemplazadas); la política de reset debe ser LA MISMA en ambos
      lados — si ``policy_a != policy_b`` se hace raise ruidoso: la
      comparación mezclaría resets silenciosamente (hallazgo MATERIAL de la
      revisión cruzada GLM-5.3 2026-09-30). Con ``policy_b=None`` se copia
      ``policy_a`` (default seguro).

    Misma ``seed`` ⇒ mismos sorteos por índice ⇒ la comparación es pareada
    campaña a campaña; el IC remuestrea el índice común completo (nunca un
    subconjunto dependiente de la política — ver ``validity_note``).
    """
    if kind not in ("reset", "withdrawal"):
        raise ValueError("kind debe ser 'reset' | 'withdrawal'")
    if policy_b is None:
        policy_b = policy_a
    if kind == "withdrawal" and policy_a != policy_b:
        raise ValueError(
            f"kind='withdrawal' compara políticas de RETIRADA sobre outcomes "
            f"de dos barridos de la funded: la política de reset debe ser la "
            f"misma en ambos lados (policy_a == policy_b; recibido "
            f"'{policy_a}' vs '{policy_b}') — si no, la comparación mezcla "
            f"resets silenciosamente"
        )
    nets_a = _campaign_nets(outcomes_a, cfg, fx_eur_usd=fx_eur_usd,
                            n_paths=n_paths, seed=seed,
                            reset_policy=policy_a,
                            stop_on_insolvency=stop_on_insolvency,
                            initial_capital_eur=initial_capital_eur)
    nets_b = _campaign_nets(outcomes_b, cfg, fx_eur_usd=fx_eur_usd,
                            n_paths=n_paths, seed=seed,
                            reset_policy=policy_b,
                            stop_on_insolvency=stop_on_insolvency,
                            initial_capital_eur=initial_capital_eur)
    delta = nets_a - nets_b
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 888_888]))
    boot = delta[rng.integers(0, len(delta), size=(int(n_boot), len(delta)))
                 ].mean(axis=1)
    return PolicyComparison(
        kind=kind, label_a=label_a, label_b=label_b,
        ev_mean_a=float(nets_a.mean()), ev_mean_b=float(nets_b.mean()),
        delta_ev=float(delta.mean()),
        ci95_paired=(float(np.percentile(boot, 2.5)),
                     float(np.percentile(boot, 97.5))),
        n_paths=int(n_paths),
    )


def eod_trailing_thresholds(closed, initial, amount, cap) -> np.ndarray:
    """Umbrales EOD-trailing por día (plan §2.2; golden oficial AC-3).

    Definición textual verificada: al inicio del día d,

        T_d = min( max(cierres_balance_{d'<d}, inicial) − amount , cap )

    con ``cap`` None = sin cap (genérico para futuras firms; FTMO declara
    cap = account_size → lock permanente del umbral en el inicial). El HWM
    de cierres nunca baja.

    Devuelve ``len(closed)+1`` umbrales: el índice d (0..len(closed)−1) es
    el umbral APLICABLE al día d (usa cierres estrictamente anteriores);
    el último índice es el umbral que aplicaría al día posterior al último
    cierre (golden del lock y walker funded tras el EOD final).
    """
    closed = np.asarray(closed, dtype=float)
    n = len(closed)
    out = np.empty(n + 1, dtype=float)
    hwm = float(initial)
    amount = float(amount)
    for d in range(n + 1):
        t = hwm - amount
        if cap is not None:
            t = min(t, float(cap))
        out[d] = t
        if d < n:
            hwm = max(hwm, float(closed[d]))
    return out


def evaluate_phase_path_eod_trail(
    block_order: np.ndarray,
    day_arrays: list[BlockDayArrays],
    durations: np.ndarray,
    gaps: np.ndarray | None = None,
    *,
    account_size: float,
    phase: PropFirmPhase,
) -> PhasePathResult:
    """Evaluación de fase con Max DD EOD-trailing (FTMO Futures, plan §2.2-2.4).

    Espejo estructural de :func:`src.risk.propfirm.evaluate_phase_path_trailing`
    pero con umbral diario desde el HWM de CIERRES de balance (no de equity) y
    breach intradía sobre equity (``breach_on: equity``, default; ``balance``
    chequea el cierre EOD y también es usable por el camino mono).

    Orden por día (adjudicado en el plan):

    1. ``T_d`` desde el HWM de cierres (estado; se actualiza AL FINAL del
       día — §2.9.2: mutar el cierre del día d no cambia T_d).
    2. **Breach Max DD**: ``S_d + rel_low[d] <= T_d`` (equity) o
       ``S_d + rel_closed[d] <= T_d`` (balance) → ``fail``. Subsumido el
       chequeo de cierre (``rel_low <= rel_closed`` por construcción).
    3. **DLL hard de fase** (Pro, ``phase.daily_loss_usd``):
       ``rel_low[d] − rel_before[d] <= −dll`` → ``fail`` (trigger con el
       ``rel_low`` CRUDO; Growth no declara DLL en eval).
    4. **Target + consistencia** (§2.4; puerta de continuación, NUNCA
       produce fail): ``total = S_d + rel_closed[d] − account_size``,
       ``best_day = max_d (rel_closed[d] − rel_before[d])``; pasa el primer
       día con ``total >= max(target_usd, best_day/share)`` (borde inclusivo,
       tolerancia 1e-9) y ``trading_days >= min_trading_days``. El best_day
       puede crecer después y retrasar el pase — nunca genera fail.
    5. ``neither`` al agotar el backtest.

    ``PhasePathResult.end_pos`` (plan ftmo-fases-secuencias): posición, dentro
    de ``block_order``, del bloque de resolución (``fail`` / ``pass``); en
    ``neither``, ``len(block_order) - 1``. La fase siguiente / la fondeada
    arrancan en ``propfirm._next_start(end_pos)``.

    Estado cross-bloque: HWM ABSOLUTO (arranca en ``account_size``) y
    ``saldo`` con la convención de los walkers existentes
    (``saldo += total_pnl`` al cerrar bloque; ``S_d = account_size + saldo``
    dentro del bloque). Bloque sin ``rel_low`` con breach equity / DLL en
    USD → raise ruidoso (§2.9.11: nunca degradación silenciosa; la infra
    trailing por minuto es ``block_day_arrays_portfolio(..., trailing=True)``).
    """
    dd = phase.drawdown
    if dd is None or dd.type != "eod_trailing":
        raise ValueError(
            "evaluate_phase_path_eod_trail exige phase.drawdown.type "
            "'eod_trailing'"
        )
    amount = float(dd.amount_usd)
    cap = dd.lock_at_balance          # None = sin cap
    breach_on = dd.breach_on
    dll = phase.daily_loss_usd
    cons = phase.consistency
    target_usd = account_size * phase.profit_target_pct / 100.0
    needs_low = breach_on == "equity" or dll is not None

    hwm = float(account_size)
    saldo = 0.0
    elapsed_prev = 0.0
    trading_days = 0
    best_day = 0.0
    terminal = float(account_size)
    total_days = 0.0
    for pos, bi in enumerate(block_order):
        ba = day_arrays[bi]
        if needs_low and ba.rel_low is None:
            raise ValueError(
                f"bloque {bi} sin rel_low: eod_trailing breach_on 'equity' y "
                "el DLL en USD exigen la infra trailing por minuto "
                "(block_day_arrays_portfolio(..., trailing=True)) — nunca "
                "degradación silenciosa"
            )
        s_block = account_size + saldo
        day0 = int(ba.day_ord[0]) if len(ba.day_ord) else 0
        for j in range(len(ba.day_ord)):
            elapsed = elapsed_prev + float(int(ba.day_ord[j]) - day0)
            total_days = elapsed + 1.0
            t_day = hwm - amount
            if cap is not None:
                t_day = min(t_day, float(cap))
            rel_low = ba.rel_low[j] if ba.rel_low is not None else None
            closed_abs = s_block + ba.rel_closed[j]
            terminal = closed_abs
            # 1) breach Max DD (equity intradía o balance EOD).
            if breach_on == "equity":
                breach = s_block + rel_low <= t_day
            else:
                breach = closed_abs <= t_day
            if breach:
                return PhasePathResult("fail", total_days, trading_days,
                                       terminal, end_pos=pos)
            # 2) DLL hard de fase (Pro; trigger con rel_low CRUDO, §2.3).
            if dll is not None and rel_low - ba.rel_before[j] <= -dll:
                return PhasePathResult("fail", total_days, trading_days,
                                       terminal, end_pos=pos)
            if ba.opens_day[j] > 0:
                trading_days += 1
            # 3) target + consistencia (puerta de pase; borde inclusivo).
            session_net = ba.rel_closed[j] - ba.rel_before[j]
            if session_net > best_day:
                best_day = session_net
            total = closed_abs - account_size
            required = target_usd
            if cons is not None and best_day > 0:
                # "Best Day represents 40% or less of your total profit"
                # (inclusivo): total >= best_day / share.
                required = max(required, best_day / cons.max_best_day_share)
            if (
                total + _CONSISTENCY_TOL >= required
                and trading_days >= phase.min_trading_days
            ):
                return PhasePathResult("pass", total_days, trading_days,
                                       terminal, end_pos=pos)
            # 4) update EOD: el HWM se alimenta del cierre del día DESPUÉS
            #    del check (cierres estrictamente anteriores, §2.9.2).
            hwm = max(hwm, closed_abs)
        saldo += ba.total_pnl
        elapsed_prev += durations[bi] + (gaps[pos] if gaps is not None else 0.0)
    return PhasePathResult("neither", total_days, trading_days, terminal,
                           end_pos=len(block_order) - 1)
