"""Simulador EV de prop firms (enfoque extractivo) sobre calendario real.

Nivel 1 (trade-level): barreras de fase FTMO-like sobre la serie DIARIA real
(bloques flat-flat barajados, valle flotante worst por dia calendario desde
OHLC M1 — mismo patron que :mod:`src.risk.mc_portfolio`, NO ``floor(n×mitd)``;
ver PLAN_PROPFIRM_REVIEW.md, CRITICO-2 de la revision GLM):

    max loss   : estatico — equity (valle o cierre) <= inicial*(1-max_loss%) -> fail
    daily loss : dinamico — equity (valle o cierre) < balance_inicio_dia
                 - daily%*inicial -> fail. El suelo se recalcula cada dia
                 calendario desde el balance de inicio de ese dia.
    pass       : equity cerrada >= inicial*(1+target%) Y trading_days >=
                 min_trading_days. Si el target llega antes de min_days, el
                 path CONTINUA a riesgo normal (cota conservadora; en la
                 realidad se operan microlotes — P3 de la revision).

Nivel 2 (sequence-level): secuencias de N challenges con muestreo CONJUNTO por
path (F1[k] -> F2[k] -> funded[k]; CRITICO-1): la dependencia intra-challenge
se preserva y P(P1 ∧ P2) se reporta conjunta, no factorizada.

CONTINUACION de la secuencia (plan ftmo-fases-secuencias, 2026-10-16): F1, F2 y
fondeada de un path NO reusan la misma fila de bloques desde la posicion 0 (eso
daba P(F2|F1)=1 por construccion y hacia que la fondeada reviviera los mismos
bloques con los que paso la evaluacion). Son ventanas de n bloques de UNA fila
prolongada (:class:`PhaseTape`, 3n bloques): la F1 es la ventana [0, n) — la
fila del bootstrap de hoy, bit a bit —, la F2 empieza en el bloque siguiente
al de resolucion de la F1 y la fondeada en el siguiente al de resolucion de la
ultima fase de evaluacion (:func:`_next_start`). La prolongacion es el mismo
bootstrap estacionario (:func:`src.risk.mc.extend_stationary_rows`), asi que la
dependencia entre fases es la del propio bootstrap (rachas de bs bloques).
Aproximaciones residuales: se salta el resto del bloque en que se resuelve la
fase; no hay espera entre fases; sin regimen mas largo que las rachas.

Sizing: lotes discretos por ``f`` via :func:`src.risk.mm.simulate_f` (re-floor
lot_step, clamp min/max), conservados por trade al barajar (Modelo A: valido
cerca del baseline; FTMO es ±10%, mismo orden que AXI Seed).

CAVEAT EPISTEMICO (aplica a toda la app): la distribucion de neto es
CONDICIONAL al edge del backtest; la incertidumbre sobre si ese edge persiste
en forward NO esta en estos numeros.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.risk.mc_portfolio import StrategyMC, detect_flatflat_blocks
from src.risk.propfirm_trailing import (
    align_trailing_days,
    build_trailing_columns,
    trailing_block_result,
)
from src.risk.ruin import wilson_ci

_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "propfirms.yaml"

# Réplicas del bootstrap del IC95 del EV en sequence_ev (M2-15 revisión GLM r2:
# constante nombrada y congelada — el stream de rng se comparte con los draws de
# secuencia, así que este valor forma parte de la reproducibilidad).
_EV_CI_N_BOOT = 2000


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrailingDrawdown:
    """Drawdown trailing intradía sobre equity (flotante incluido).

    ``type`` admite:
    - ``trailing_intraday_unrealized`` (Apex): el suelo sube con el
      high-water mark de equity; en la fondeada se congela al alcanzar
      ``lock_at_balance`` y queda fijo en ``frozen_floor``.
    - ``eod_trailing`` (FTMO Futures): el umbral se recalcula al INICIO de
      cada día desde ``max(cierres de balance anteriores, inicial) − amount``
      (HWM de cierres, nunca baja); ``lock_at_balance`` se reinterpreta como
      CAP del umbral (FTMO = account_size, lock permanente) y
      ``frozen_floor`` está PROHIBIDO (el loader raisea).
    ``breach_on``: "equity" (default; breach intradía sobre balance +
    flotante, FTMO Futures) | "balance" (cierre EOD, no usado por estos
    perfiles). Importes en USD (moneda de la cuenta de la firm).
    """

    amount_usd: float
    lock_at_balance: float | None = None
    frozen_floor: float | None = None
    type: str = "trailing_intraday_unrealized"
    breach_on: str = "equity"


@dataclass(frozen=True)
class SubscriptionSpec:
    """Suscripción mensual FTMO Futures (nivel 2, ``model: futures``).

    Economía verificada 2026-09-29: suscripción cancelada al pasar; breach
    de eval → cuenta nueva GRATIS en la siguiente fecha de facturación;
    reset de pago opcional para reinicio inmediato (no altera el ciclo).
    ``*_eur`` quedan rellenos por :func:`resolve_fee_eur` (conversión USD→EUR
    UNA vez, patrón ``ActivationSpec.fee_eur``); None hasta entonces.
    ``campaign_cap_days`` / ``funded_cap_days`` son horizontes de TRUNCADO
    del modelo (decisión de diseño 2026-09-29, plan v2 C1), NO reglas FTMO.
    """

    monthly_fee_usd: float
    reset_fee_usd: float
    billing_period_days: int = 30
    campaign_cap_days: int = 183
    funded_cap_days: int = 365
    funded_start_delay_days: int = 3
    monthly_fee_eur: float | None = None
    reset_fee_eur: float | None = None


@dataclass(frozen=True)
class FuturesPayouts:
    """Payouts de la Sim-Funded FTMO Futures (``model: futures``).

    Qualifying days sobre P&L cerrado EOD; retirable = ``withdraw_share`` ×
    profit sobre el capital INICIAL; regla min-new-profit: ≥
    ``min_new_profit_frac`` de lo solicitado debe ser profit nuevo del ciclo
    (desde el último payout) → ``request ≤ new_profit_cycle / min_new_frac``.
    El ratio 90/10 aplica sobre lo solicitado (``profit_split_pct`` del perfil;
    el "Withdraw 100%" de Pro es la fracción retirable, no el ratio — caveat
    ligado a ``ruleset_version``, plan v2 C4).
    """

    qualifying_day_min_usd: float
    days_per_payout: int
    withdraw_share: float
    request_cap_usd: float
    min_new_profit_frac: float = 0.5
    min_payout_usd: float = 20.0
    withdrawal_policy: str = "greedy"      # "greedy" | "buffer_preserving"
    buffer_usd: float | None = None        # default = amount del Max DD


@dataclass(frozen=True)
class ActivationSpec:
    """Fee de activación de la PA (solo se cobra en paths que pasan).

    ``fee_usd`` es el importe declarado en el YAML (USD). Tras
    :func:`resolve_fee_eur`, ``fee_eur`` queda relleno con la conversión y es
    el valor que debe consumir el motor (None hasta entonces).
    """

    fee_usd: float
    deadline_days: int
    fee_eur: float | None = None


@dataclass(frozen=True)
class DailyLossSpec:
    """Daily loss de la fondeada tipo Apex: pausa no fatal, por sesión."""

    type: str                        # p.ej. "pause_not_fatal"
    by_tier_usd: tuple[float, ...]   # DLL por tier (índice = level - 1)
    reset: str = "session"
    post_hit: str = "clamp"          # PnL sesión := min(observado, -DLL)


@dataclass(frozen=True)
class ScalingTier:
    """Cap de contratos por tramo de beneficio acumulado de la PA."""

    level: int
    profit_lo: float
    profit_hi: float | None          # None = tramo abierto por arriba
    max_mini: int
    max_micro: int


@dataclass(frozen=True)
class ContractCap:
    """Cap de contratos simultáneos de una fase de evaluación.

    ``max_mini``/``max_micro`` son los límites declarados por la firm (Apex
    150K evaluación: 12 mini / 120 micro). NO se exige que
    ``max_micro == 10 * max_mini`` (como tampoco en ``ScalingTier``): se
    conservan tal como los declara la firm y el diagnóstico trabaja en
    micro-equivalentes (``max_micro``).
    """

    max_mini: int
    max_micro: int


@dataclass(frozen=True)
class ConsistencySpec:
    """Regla de consistencia (mejor día vs beneficio de la ventana)."""

    max_best_day_share: float
    strict: bool                     # True: igualdad al umbral = denegado
    window: str                      # p.ej. "since_last_payout"


@dataclass(frozen=True)
class LadderPayouts:
    """Escalera de retiros de la PA Apex (modelo ``ladder``)."""

    qualifying_day_min_usd: float
    days_per_payout: int
    ladder_usd: tuple[float, ...]
    min_payout_usd: float
    min_balance_first: float
    residual_floor: float            # safety net: nunca se retira por debajo
    max_payouts: int
    consistency: ConsistencySpec


@dataclass(frozen=True)
class InactivitySpec:
    """Muerte por inactividad: ventana rolling con mínimo de días activos."""

    window_days: int
    min_days: int
    min_profit_usd: float


@dataclass(frozen=True)
class PropFirmPhase:
    name: str
    profit_target_pct: float
    max_daily_loss_pct: float | None   # null = sin DLL (0.0 PROHIBIDO por el loader)
    max_loss_pct: float | None         # null con drawdown trailing (barrera estática off)
    min_trading_days: int
    drawdown: TrailingDrawdown | None = None
    lifetime_days: int | None = None
    on_horizon_exhausted: str = "exclude"  # "exclude" (FTMO) / "fail" (Apex)
    max_contracts: ContractCap | None = None  # cap de contratos simultáneos (Fase 5)
    daily_loss_usd: float | None = None  # DLL en USD (FTMO Futures Pro eval,
                                         # hard fatal); XOR max_daily_loss_pct
    consistency: ConsistencySpec | None = None  # puerta de pase, solo eval
                                         # (FTMO Futures; nunca produce breach)


@dataclass(frozen=True)
class FundedRules:
    payout_cycle_days: int             # solo model "cycle" (0 = no aplica en ladder)
    min_payout_profit_pct: float
    max_payout_cycles: int
    model: str = "cycle"               # "cycle" (FTMO) / "ladder" (Apex) / "futures"
    drawdown: TrailingDrawdown | None = None
    daily_loss: DailyLossSpec | None = None
    scaling_tiers: tuple[ScalingTier, ...] = ()
    payouts: LadderPayouts | None = None
    inactivity: InactivitySpec | None = None
    payouts_futures: FuturesPayouts | None = None  # solo model "futures"


@dataclass(frozen=True)
class PropFirmConfig:
    key: str
    display_name: str
    account_size: float          # USD
    fee: float                   # moneda segun fee_ccy; EUR tras resolve_fee_eur
    profit_split_pct: float
    refund_fee_on_first_payout: bool
    phases: tuple[PropFirmPhase, ...]
    funded: FundedRules
    ruleset_version: str = ""
    fee_ccy: str = "EUR"         # moneda de `fee` tal como sale del loader
    activation: ActivationSpec | None = None
    day_axis: str | None = None  # None = dia calendario; "et_session" (Apex)
    day_axis_cut_hhmm: str = "18:00"  # corte ET del eje de sesiones (solo con
                                 # day_axis "et_session"; "18:00" = Apex
                                 # bit-identico, FTMO Futures "16:10")
    subscription: SubscriptionSpec | None = None  # solo model "futures"


_SUPPORTED_DRAWDOWN_TYPES = ("trailing_intraday_unrealized", "eod_trailing")
_SUPPORTED_FEE_CCY = ("EUR", "USD")
_SUPPORTED_ON_HORIZON = ("exclude", "fail")
_SUPPORTED_FUNDED_MODELS = ("cycle", "ladder", "futures")
_SUPPORTED_DAY_AXIS = ("et_session",)
_SUPPORTED_DLL_POST_HIT = ("clamp",)
_SUPPORTED_BREACH_ON = ("equity", "balance")
_SUPPORTED_DLL_TYPES = ("pause_not_fatal", "hard_fatal")
_SUPPORTED_RESET_POLICIES = ("billing_retry", "paid_reset")
_SUPPORTED_WITHDRAWAL_POLICIES = ("greedy", "buffer_preserving")

# Formato del corte del eje de sesiones (misma regla que
# session_axis._roll_from_cut; "00:00" se rechaza aparte: degenera el roll).
_CUT_HHMM_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


def _opt_float(v) -> float | None:
    return None if v is None else float(v)


def _parse_trailing_drawdown(raw: dict | None, ctx: str) -> TrailingDrawdown | None:
    if raw is None:
        return None
    dd_type = str(raw.get("type", "trailing_intraday_unrealized"))
    if dd_type not in _SUPPORTED_DRAWDOWN_TYPES:
        raise ValueError(
            f"{ctx}: drawdown.type '{dd_type}' no soportado "
            f"(solo {_SUPPORTED_DRAWDOWN_TYPES})"
        )
    if raw.get("amount_usd") is None:
        raise ValueError(f"{ctx}: drawdown requiere amount_usd")
    amount = float(raw["amount_usd"])
    if amount <= 0:
        raise ValueError(f"{ctx}: drawdown.amount_usd debe ser > 0")
    breach_on = str(raw.get("breach_on", "equity"))
    if breach_on not in _SUPPORTED_BREACH_ON:
        raise ValueError(
            f"{ctx}: drawdown.breach_on '{breach_on}' no soportado "
            f"(solo {_SUPPORTED_BREACH_ON})"
        )
    lock = _opt_float(raw.get("lock_at_balance"))
    floor = _opt_float(raw.get("frozen_floor"))
    if dd_type == "eod_trailing":
        # lock_at_balance = CAP del umbral (FTMO = account_size); frozen_floor
        # prohibido (el umbral nunca baja). La validación del cap
        # (>= account_size - amount, plan v2 #10) se hace en
        # load_propfirm_config: aquí no se conoce account_size.
        if floor is not None:
            raise ValueError(
                f"{ctx}: eod_trailing prohibe frozen_floor (no aplica: el "
                f"umbral de cierres nunca baja); lock_at_balance es el CAP"
            )
    else:
        if (lock is None) != (floor is None):
            raise ValueError(
                f"{ctx}: lock_at_balance y frozen_floor van juntos (ambos o ninguno)"
            )
        if lock is not None and not floor < lock:
            raise ValueError(
                f"{ctx}: frozen_floor ({floor}) debe ser < lock_at_balance ({lock})"
            )
    return TrailingDrawdown(
        amount_usd=amount, lock_at_balance=lock, frozen_floor=floor,
        type=dd_type, breach_on=breach_on,
    )


def _parse_daily_loss(raw: dict | None, ctx: str) -> DailyLossSpec | None:
    if raw is None:
        return None
    tiers = tuple(float(x) for x in (raw.get("by_tier_usd") or ()))
    if not tiers or any(x <= 0 for x in tiers):
        raise ValueError(f"{ctx}: daily_loss.by_tier_usd debe ser no vacia y de importes > 0")
    post_hit = str(raw.get("post_hit", "clamp"))
    if post_hit not in _SUPPORTED_DLL_POST_HIT:
        raise ValueError(f"{ctx}: daily_loss.post_hit '{post_hit}' no soportado")
    dll_type = str(raw.get("type", "pause_not_fatal"))
    if dll_type not in _SUPPORTED_DLL_TYPES:
        raise ValueError(
            f"{ctx}: daily_loss.type '{dll_type}' no soportado "
            f"(solo {_SUPPORTED_DLL_TYPES})"
        )
    return DailyLossSpec(
        type=dll_type,
        by_tier_usd=tiers,
        reset=str(raw.get("reset", "session")),
        post_hit=post_hit,
    )


def _parse_scaling_tiers(raw_list: list | None, ctx: str) -> tuple[ScalingTier, ...]:
    tiers: list[ScalingTier] = []
    for t in raw_list or ():
        lo, hi = t["profit_range"]
        lo = float(lo)
        hi = _opt_float(hi)
        if hi is not None and hi < lo:
            raise ValueError(f"{ctx}: scaling_tiers profit_range con hi < lo")
        max_mini = int(t["max_mini"])
        max_micro = int(t["max_micro"])
        if max_mini <= 0 or max_micro <= 0:
            raise ValueError(f"{ctx}: scaling_tiers max_mini/max_micro deben ser > 0")
        tiers.append(
            ScalingTier(
                level=int(t["level"]), profit_lo=lo, profit_hi=hi,
                max_mini=max_mini, max_micro=max_micro,
            )
        )
    return tuple(tiers)


def _parse_consistency(raw: dict, ctx: str) -> ConsistencySpec:
    share = float(raw["max_best_day_share"])
    if not 0 < share <= 1:
        raise ValueError(f"{ctx}: consistency.max_best_day_share debe estar en (0, 1]")
    return ConsistencySpec(
        max_best_day_share=share,
        strict=bool(raw.get("strict", True)),
        window=str(raw.get("window", "since_last_payout")),
    )


def _parse_ladder_payouts(raw: dict | None, ctx: str) -> LadderPayouts | None:
    if raw is None:
        return None
    ladder = tuple(float(x) for x in (raw.get("ladder_usd") or ()))
    if not ladder or any(x <= 0 for x in ladder):
        raise ValueError(f"{ctx}: payouts.ladder_usd debe ser no vacia y de importes > 0")
    max_payouts = int(raw["max_payouts"])
    if max_payouts != len(ladder):
        raise ValueError(
            f"{ctx}: payouts.max_payouts ({max_payouts}) debe ser == "
            f"len(ladder_usd) ({len(ladder)})"
        )
    min_payout = float(raw["min_payout_usd"])
    if min_payout <= 0:
        raise ValueError(f"{ctx}: payouts.min_payout_usd debe ser > 0")
    residual = float(raw["residual_floor"])
    min_first = float(raw["min_balance_first"])
    if min_first < residual + min_payout:
        raise ValueError(
            f"{ctx}: payouts.min_balance_first ({min_first}) debe ser >= "
            f"residual_floor + min_payout_usd ({residual + min_payout}); "
            f"residual_floor es el safety net bajo el que nunca se retira"
        )
    days_per = int(raw["days_per_payout"])
    if days_per <= 0:
        raise ValueError(f"{ctx}: payouts.days_per_payout debe ser > 0")
    qual_min = float(raw["qualifying_day_min_usd"])
    if qual_min < 0:
        raise ValueError(f"{ctx}: payouts.qualifying_day_min_usd debe ser >= 0")
    return LadderPayouts(
        qualifying_day_min_usd=qual_min,
        days_per_payout=days_per,
        ladder_usd=ladder,
        min_payout_usd=min_payout,
        min_balance_first=min_first,
        residual_floor=residual,
        max_payouts=max_payouts,
        consistency=_parse_consistency(raw["consistency"], ctx),
    )


def _parse_inactivity(raw: dict | None, ctx: str) -> InactivitySpec | None:
    if raw is None:
        return None
    window = int(raw["window_days"])
    min_days = int(raw["min_days"])
    min_profit = float(raw["min_profit_usd"])
    if window <= 0 or not 0 < min_days <= window or min_profit < 0:
        raise ValueError(
            f"{ctx}: inactivity invalida (window_days>0, 0<min_days<=window_days, "
            f"min_profit_usd>=0)"
        )
    return InactivitySpec(window_days=window, min_days=min_days, min_profit_usd=min_profit)


def _parse_activation(raw: dict | None, ctx: str) -> ActivationSpec | None:
    if raw is None:
        return None
    fee = float(raw["fee_usd"])
    deadline = int(raw["deadline_days"])
    if fee <= 0 or deadline <= 0:
        raise ValueError(f"{ctx}: activation.fee_usd y deadline_days deben ser > 0")
    return ActivationSpec(fee_usd=fee, deadline_days=deadline)


def _parse_contract_cap(raw: dict | None, ctx: str) -> ContractCap | None:
    if raw is None:
        return None
    max_mini = int(raw["max_mini"])
    max_micro = int(raw["max_micro"])
    if max_mini <= 0 or max_micro <= 0:
        raise ValueError(f"{ctx}: max_contracts max_mini/max_micro deben ser > 0")
    return ContractCap(max_mini=max_mini, max_micro=max_micro)


def _parse_subscription(raw: dict | None, ctx: str) -> SubscriptionSpec | None:
    if raw is None:
        return None
    monthly = float(raw["monthly_fee_usd"])
    reset = float(raw["reset_fee_usd"])
    if monthly <= 0 or reset < 0:
        raise ValueError(
            f"{ctx}: subscription.monthly_fee_usd debe ser > 0 y "
            f"reset_fee_usd >= 0"
        )
    period = int(raw.get("billing_period_days", 30))
    cap = int(raw.get("campaign_cap_days", 183))
    fcap = int(raw.get("funded_cap_days", 365))
    delay = int(raw.get("funded_start_delay_days", 3))
    if period <= 0 or cap <= 0 or fcap <= 0 or delay < 0:
        raise ValueError(
            f"{ctx}: subscription invalida (billing_period_days>0, "
            f"campaign_cap_days>0, funded_cap_days>0, funded_start_delay_days>=0)"
        )
    return SubscriptionSpec(
        monthly_fee_usd=monthly,
        reset_fee_usd=reset,
        billing_period_days=period,
        campaign_cap_days=cap,
        funded_cap_days=fcap,
        funded_start_delay_days=delay,
    )


def _parse_futures_payouts(raw: dict | None, ctx: str,
                           default_buffer_usd: float | None) -> FuturesPayouts | None:
    if raw is None:
        return None
    share = float(raw["withdraw_share"])
    if not 0 < share <= 1:
        raise ValueError(f"{ctx}: payouts_futures.withdraw_share debe estar en (0, 1]")
    cap = float(raw["request_cap_usd"])
    if cap <= 0:
        raise ValueError(f"{ctx}: payouts_futures.request_cap_usd debe ser > 0")
    min_new = float(raw.get("min_new_profit_frac", 0.5))
    if not 0 < min_new <= 1:
        raise ValueError(
            f"{ctx}: payouts_futures.min_new_profit_frac debe estar en (0, 1]"
        )
    min_payout = float(raw.get("min_payout_usd", 20.0))
    if min_payout < 0:
        raise ValueError(f"{ctx}: payouts_futures.min_payout_usd debe ser >= 0")
    days_per = int(raw["days_per_payout"])
    if days_per <= 0:
        raise ValueError(f"{ctx}: payouts_futures.days_per_payout debe ser > 0")
    qual = float(raw["qualifying_day_min_usd"])
    if qual < 0:
        raise ValueError(f"{ctx}: payouts_futures.qualifying_day_min_usd debe ser >= 0")
    policy = str(raw.get("withdrawal_policy", "greedy"))
    if policy not in _SUPPORTED_WITHDRAWAL_POLICIES:
        raise ValueError(
            f"{ctx}: payouts_futures.withdrawal_policy '{policy}' no soportada "
            f"(solo {_SUPPORTED_WITHDRAWAL_POLICIES})"
        )
    buffer = _opt_float(raw.get("buffer_usd"))
    if buffer is None:
        buffer = default_buffer_usd  # default = amount del Max DD (conservar el DD)
    return FuturesPayouts(
        qualifying_day_min_usd=qual,
        days_per_payout=days_per,
        withdraw_share=share,
        request_cap_usd=cap,
        min_new_profit_frac=min_new,
        min_payout_usd=min_payout,
        withdrawal_policy=policy,
        buffer_usd=buffer,
    )


def parse_firm_spec(key: str, spec: dict) -> PropFirmConfig:
    """Valida UN perfil (el ``spec`` de ``firms.<key>`` del YAML) y construye su config.

    Es el cuerpo del bucle de :func:`load_propfirm_config`, sin cambiar su logica
    (guardian: ``tests/test_risk_propfirm_parse_spec.py``, golden de los 10 oficiales).
    Una clave obligatoria ausente se traduce de ``KeyError`` a ``ValueError`` con el
    perfil y la clave; el resto de excepciones (``TypeError``/``AttributeError`` por
    tipos mal formados) se propaga tal cual: la seguridad de tipos de lo que manda un
    usuario la da el llamador antes de invocar este parser (enmienda 1 de onboarding-c).
    Las claves desconocidas (p. ej. ``meta``) se ignoran.
    """
    try:
        return _parse_firm_spec(key, spec)
    except KeyError as exc:
        missing = exc.args[0] if exc.args else "?"
        raise ValueError(f"{key}: falta '{missing}'") from exc


def _parse_firm_spec(key: str, spec: dict) -> PropFirmConfig:
    phases = tuple(
        PropFirmPhase(
            name=str(ph["name"]),
            profit_target_pct=float(ph["profit_target_pct"]),
            max_daily_loss_pct=_opt_float(ph.get("max_daily_loss_pct")),
            max_loss_pct=_opt_float(ph.get("max_loss_pct")),
            min_trading_days=int(ph["min_trading_days"]),
            drawdown=_parse_trailing_drawdown(ph.get("drawdown"), f"{key}/{ph['name']}"),
            lifetime_days=(
                None if ph.get("lifetime_days") is None else int(ph["lifetime_days"])
            ),
            on_horizon_exhausted=str(ph.get("on_horizon_exhausted", "exclude")),
            max_contracts=_parse_contract_cap(
                ph.get("max_contracts"), f"{key}/{ph['name']}"
            ),
            daily_loss_usd=_opt_float(ph.get("daily_loss_usd")),
            consistency=(
                _parse_consistency(ph["consistency"], f"{key}/{ph['name']}")
                if ph.get("consistency") is not None else None
            ),
        )
        for ph in spec["phases"]
    )
    if not 1 <= len(phases) <= 2:
        raise ValueError(f"{key}: se soportan 1-2 fases, hay {len(phases)}")
    for ph in phases:
        ctx = f"{key}/{ph.name}"
        if ph.profit_target_pct <= 0:
            raise ValueError(f"{ctx}: target debe ser > 0")
        if ph.max_loss_pct is not None and ph.max_loss_pct <= 0:
            raise ValueError(f"{ctx}: max_loss debe ser > 0 o null")
        if ph.max_daily_loss_pct is not None and ph.max_daily_loss_pct <= 0:
            raise ValueError(
                f"{ctx}: max_daily_loss_pct debe ser null o > 0 (0.0 prohibido: "
                f"el evaluador mataria cualquier dia rojo si el dispatch fallara)"
            )
        if (
            ph.max_daily_loss_pct is not None
            and ph.max_loss_pct is not None
            and ph.max_daily_loss_pct >= ph.max_loss_pct
        ):
            raise ValueError(
                f"{ctx}: daily loss ({ph.max_daily_loss_pct}%) debe ser "
                f"< max loss ({ph.max_loss_pct}%) o la cuenta seria inoperable"
            )
        if ph.drawdown is not None and ph.max_loss_pct is not None:
            raise ValueError(
                f"{ctx}: con drawdown trailing, max_loss_pct debe ser null "
                f"(manda el trailing)"
            )
        if ph.on_horizon_exhausted not in _SUPPORTED_ON_HORIZON:
            raise ValueError(
                f"{ctx}: on_horizon_exhausted '{ph.on_horizon_exhausted}' "
                f"no soportado (solo {_SUPPORTED_ON_HORIZON})"
            )
        if ph.lifetime_days is not None and ph.lifetime_days <= 0:
            raise ValueError(f"{ctx}: lifetime_days debe ser > 0 o null")
        if ph.daily_loss_usd is not None and ph.daily_loss_usd <= 0:
            raise ValueError(f"{ctx}: daily_loss_usd debe ser > 0")
        if ph.daily_loss_usd is not None and ph.max_daily_loss_pct is not None:
            raise ValueError(
                f"{ctx}: daily_loss_usd y max_daily_loss_pct son mutuamente "
                f"excluyentes (elige uno)"
            )
    fnd = spec["funded"]
    model = str(fnd.get("model", "cycle"))
    if model not in _SUPPORTED_FUNDED_MODELS:
        raise ValueError(f"{key}: funded.model '{model}' no soportado (solo {_SUPPORTED_FUNDED_MODELS})")
    payouts = _parse_ladder_payouts(fnd.get("payouts"), f"{key}/funded")
    if model == "ladder" and payouts is None:
        raise ValueError(f"{key}: funded.model 'ladder' requiere el bloque payouts completo")
    funded_drawdown = _parse_trailing_drawdown(fnd.get("drawdown"), f"{key}/funded")
    funded = FundedRules(
        payout_cycle_days=int(fnd.get("payout_cycle_days", 0)),
        min_payout_profit_pct=float(fnd.get("min_payout_profit_pct", 0.0)),
        max_payout_cycles=int(fnd.get("max_payout_cycles", 50)),
        model=model,
        drawdown=funded_drawdown,
        daily_loss=_parse_daily_loss(fnd.get("daily_loss"), f"{key}/funded"),
        scaling_tiers=_parse_scaling_tiers(fnd.get("scaling_tiers"), f"{key}/funded"),
        payouts=payouts,
        inactivity=_parse_inactivity(fnd.get("inactivity"), f"{key}/funded"),
        payouts_futures=_parse_futures_payouts(
            fnd.get("payouts_futures"), f"{key}/funded",
            None if funded_drawdown is None else funded_drawdown.amount_usd,
        ),
    )
    if model == "cycle" and funded.payout_cycle_days <= 0:
        raise ValueError(f"{key}: funded.model 'cycle' requiere payout_cycle_days > 0")
    subscription = _parse_subscription(spec.get("subscription"), key)
    if model == "futures":
        if funded.payouts_futures is None:
            raise ValueError(
                f"{key}: funded.model 'futures' requiere el bloque payouts_futures"
            )
        if funded.drawdown is None or funded.drawdown.type != "eod_trailing":
            raise ValueError(
                f"{key}: funded.model 'futures' exige funded.drawdown.type "
                f"'eod_trailing'"
            )
        if subscription is None:
            raise ValueError(
                f"{key}: funded.model 'futures' requiere el bloque subscription"
            )
        if funded.daily_loss is not None and len(funded.daily_loss.by_tier_usd) != 1:
            raise ValueError(
                f"{key}: futures daily_loss.by_tier_usd de UN elemento "
                f"(sin tiers en FTMO Futures)"
            )
    fee_ccy = str(spec.get("fee_ccy", "EUR")).upper()
    if fee_ccy not in _SUPPORTED_FEE_CCY:
        raise ValueError(f"{key}: fee_ccy '{fee_ccy}' no soportada (solo {_SUPPORTED_FEE_CCY})")
    day_axis = spec.get("day_axis")
    if day_axis is not None:
        day_axis = str(day_axis)
        if day_axis not in _SUPPORTED_DAY_AXIS:
            raise ValueError(f"{key}: day_axis '{day_axis}' no soportado (solo {_SUPPORTED_DAY_AXIS})")
    raw_cut = spec.get("day_axis_cut_hhmm")
    if raw_cut is None:
        day_axis_cut = "18:00"
    else:
        if day_axis != "et_session":
            raise ValueError(
                f"{key}: day_axis_cut_hhmm solo aplica con day_axis 'et_session'"
            )
        day_axis_cut = str(raw_cut)
        if not _CUT_HHMM_RE.match(day_axis_cut) or day_axis_cut == "00:00":
            raise ValueError(
                f"{key}: day_axis_cut_hhmm '{day_axis_cut}' invalido "
                f"(formato 'HH:MM', 00:01-23:59)"
            )
    cfg = PropFirmConfig(
        key=key,
        display_name=str(spec.get("display_name", key)),
        account_size=float(spec["account_size"]),
        fee=float(spec["fee"]),
        profit_split_pct=float(spec["profit_split_pct"]),
        refund_fee_on_first_payout=bool(spec.get("refund_fee_on_first_payout", False)),
        phases=phases,
        funded=funded,
        ruleset_version=str(spec.get("ruleset_version", "")),
        fee_ccy=fee_ccy,
        activation=_parse_activation(spec.get("activation"), key),
        day_axis=day_axis,
        day_axis_cut_hhmm=day_axis_cut,
        subscription=subscription,
    )
    if cfg.account_size <= 0 or cfg.fee < 0 or not (0 < cfg.profit_split_pct <= 100):
        raise ValueError(f"{key}: account_size/fee/profit_split invalidos")
    for dd in [p.drawdown for p in phases] + [funded.drawdown]:
        if (
            dd is not None and dd.type == "eod_trailing"
            and dd.lock_at_balance is not None
            and dd.lock_at_balance < cfg.account_size - dd.amount_usd
        ):
            raise ValueError(
                f"{key}: eod_trailing lock_at_balance (cap {dd.lock_at_balance}) "
                f"debe ser >= account_size - amount "
                f"({cfg.account_size - dd.amount_usd})"
            )
    return cfg


def validate_runnable(cfg: PropFirmConfig) -> list[str]:
    """Combinaciones que el loader acepta pero el motor NO puede correr (GT-5).

    Devuelve mensajes en espanol; lista vacia = ejecutable. Se aplica a los perfiles
    de USUARIO (los 10 oficiales dan ``[]``, test). Lo que ya cubren los ``_parse_*``
    / el loader NO se duplica: ``len(ladder_usd) == max_payouts``,
    ``min_balance_first >= residual_floor + min_payout_usd``,
    ``frozen_floor < lock_at_balance``, ``daily_loss.by_tier_usd`` todos > 0 y
    ``payout_cycle_days > 0`` en ``cycle``.
    """
    errs: list[str] = []
    key = cfg.key
    if cfg.funded.model == "cycle":
        for ph in cfg.phases:
            if ph.drawdown is None and (
                ph.max_loss_pct is None or ph.max_daily_loss_pct is None
            ):
                errs.append(
                    f"{key}/{ph.name}: una fase sin drawdown trailing exige "
                    f"max_loss_pct y max_daily_loss_pct (no nulos): el motor los usa "
                    f"como barreras"
                )
    if cfg.funded.max_payout_cycles < 1:
        errs.append(f"{key}: funded.max_payout_cycles debe ser >= 1")
    if cfg.funded.model == "ladder" and cfg.funded.payouts is not None:
        if cfg.funded.payouts.max_payouts > cfg.funded.max_payout_cycles:
            errs.append(
                f"{key}: payouts.max_payouts ({cfg.funded.payouts.max_payouts}) no puede "
                f"superar funded.max_payout_cycles ({cfg.funded.max_payout_cycles}): "
                f"el array de payouts no cabe"
            )
    if (cfg.activation is not None or cfg.subscription is not None) and cfg.fee_ccy != "USD":
        errs.append(
            f"{key}: con activation o subscription (importes en USD) fee_ccy debe ser "
            f"'USD'; con EUR la conversion a EUR no se rellena y la simulacion falla"
        )
    for ph in cfg.phases:
        if ph.drawdown is not None and ph.drawdown.amount_usd >= cfg.account_size:
            errs.append(
                f"{key}/{ph.name}: drawdown.amount_usd ({ph.drawdown.amount_usd:g}) debe "
                f"ser < account_size ({cfg.account_size:g})"
            )
    fdd = cfg.funded.drawdown
    if fdd is not None and fdd.amount_usd >= cfg.account_size:
        errs.append(
            f"{key}/funded: drawdown.amount_usd ({fdd.amount_usd:g}) debe ser < "
            f"account_size ({cfg.account_size:g})"
        )
    return errs


def load_propfirm_config(path: str | Path | None = None) -> dict[str, PropFirmConfig]:
    """Carga los perfiles de prop firm del YAML y los valida (errores duros)."""
    import yaml

    p = Path(path) if path else _DEFAULT_CONFIG
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    out: dict[str, PropFirmConfig] = {}
    for key, spec in (raw.get("firms") or {}).items():
        out[key] = parse_firm_spec(key, spec)
    if not out:
        raise ValueError(f"{p}: no hay perfiles bajo 'firms:'")
    return out


def resolve_fee_eur(cfg: PropFirmConfig, fx_eur_usd: float | None) -> PropFirmConfig:
    """Normaliza a EUR las fees declaradas en USD (UNA sola vez, en el setup).

    La conversión por touchpoint quedó descartada (adjudicado 2026-09-14): hay
    consumidores escondidos como ``stop_on_insolvency`` que comparan bankroll
    EUR contra ``cfg.fee``. Convención: ``fx_eur_usd`` = USD por 1 EUR (la
    misma de ``sequence_ev``: EUR = USD / fx).

    Con ``fee_ccy == "EUR"`` devuelve ``cfg`` sin tocar. Con USD exige
    ``fx_eur_usd > 0`` y devuelve una copia con ``fee`` en EUR,
    ``fee_ccy == "EUR"``, ``activation.fee_eur`` relleno (``fee_usd`` se
    conserva tal como se declaró en el YAML) y, si hay suscripción
    (FTMO Futures), ``subscription.monthly_fee_eur`` / ``reset_fee_eur``
    rellenos con el MISMO sentido de conversión (EUR = USD / fx).
    """
    if cfg.fee_ccy == "EUR":
        return cfg
    if fx_eur_usd is None or fx_eur_usd <= 0:
        raise ValueError(
            f"{cfg.key}: fee_ccy={cfg.fee_ccy} exige fx_eur_usd > 0 para "
            f"normalizar el fee a EUR"
        )
    activation = cfg.activation
    if activation is not None:
        activation = replace(activation, fee_eur=activation.fee_usd / fx_eur_usd)
    subscription = cfg.subscription
    if subscription is not None:
        subscription = replace(
            subscription,
            monthly_fee_eur=subscription.monthly_fee_usd / fx_eur_usd,
            reset_fee_eur=subscription.reset_fee_usd / fx_eur_usd,
        )
    return replace(
        cfg, fee=cfg.fee / fx_eur_usd, fee_ccy="EUR", activation=activation,
        subscription=subscription,
    )


# ---------------------------------------------------------------------------
# Bloques flat-flat mono-estrategia con fechas de apertura
# ---------------------------------------------------------------------------

@dataclass
class DayBlock:
    """Episodio flat-flat de UNA estrategia, con fechas por trade."""

    idx: int
    start: pd.Timestamp
    end: pd.Timestamp
    duration_days: float
    trade_local_id: np.ndarray   # (k,) indice del trade en la estrategia
    open_ord: np.ndarray         # (k,) ordinal de fecha de apertura
    close_ord: np.ndarray        # (k,) ordinal de fecha de cierre
    valle_precio: list           # (k,) dict {date: valle_precio} (f-indep, <=0)
    pnl_per_lot: np.ndarray      # (k,) f-indep
    n_trades: int


def build_day_blocks(
    strategy: StrategyMC,
    oos_start: pd.Timestamp | None = None,
    day_labeler=None,
) -> list[DayBlock]:
    """Construye los bloques flat-flat de una estrategia (calendario real).

    ``day_labeler``: callable ts→date opcional (eje de sesiones ET, ver
    :mod:`src.risk.session_axis`) aplicado a ``open_ord``. ``close_d`` y las
    claves de ``valle`` ya llegan etiquetadas desde ``precompute_strategy``
    con el MISMO labeler (un solo punto de remap). None = día calendario
    (bit-idéntico al comportamiento histórico).
    """
    blocks = detect_flatflat_blocks({0: (strategy.open_t, strategy.close_t)})
    if day_labeler is None:
        open_ord = pd.to_datetime(strategy.open_t).normalize().to_numpy()
    else:
        open_ord = np.asarray(day_labeler(pd.to_datetime(strategy.open_t)))
    out: list[DayBlock] = []
    for bstart, bend in blocks:
        if oos_start is not None and pd.Timestamp(bstart) < oos_start:
            continue
        tloc, oo, cc, vp, ppl = [], [], [], [], []
        for i in range(strategy.n_trades):
            ot, ct = strategy.open_t[i], strategy.close_t[i]
            if pd.isna(ot) or pd.isna(ct):
                continue
            if ot >= bstart and ct <= bend:
                # BAJO-3 (revision GLM): trade sin valle (tramo OHLC vacio) se
                # INCLUYE con valle vacio — su PnL cerrado cuenta; descartarlo
                # sesgaria el total_pnl del bloque silenciosamente.
                tloc.append(i)
                oo.append(pd.Timestamp(open_ord[i]).toordinal())
                cc.append(pd.Timestamp(strategy.close_d[i]).toordinal())
                vp.append(strategy.valle.get(i, {}))
                ppl.append(strategy.pnl_per_lot[i])
        if not tloc:
            continue
        out.append(DayBlock(
            idx=len(out), start=bstart, end=bend,
            duration_days=(pd.Timestamp(bend) - pd.Timestamp(bstart)).total_seconds() / 86400.0,
            trade_local_id=np.array(tloc, dtype=int),
            open_ord=np.array(oo, dtype=int),
            close_ord=np.array(cc, dtype=int),
            valle_precio=vp,
            pnl_per_lot=np.array(ppl, dtype=float),
            n_trades=len(tloc),
        ))
    for j, b in enumerate(out):
        b.idx = j
    return out


def empirical_gaps(blocks: list[DayBlock]) -> np.ndarray:
    """Gaps (dias flat) entre bloques consecutivos del historial.

    CRITICO-1 (revision GLM): bajo barajado, el elapsed entre bloques debe
    incluir los gaps flat para que los ciclos de payout (dias CALENDARIO en
    FTMO) tictaqueen a cadencia real. La alternativa de ordinales absolutos
    es invalida bajo barajado (elapsed negativo cuando un bloque viejo sigue
    a uno nuevo — verificado 490/513 transiciones en XAUUSD). La correccion
    correcta es muestrear estos gaps empiricos por transicion.
    """
    if len(blocks) < 2:
        return np.zeros(1)
    gaps = [
        (blocks[i + 1].start - blocks[i].end).total_seconds() / 86400.0
        for i in range(len(blocks) - 1)
    ]
    return np.maximum(np.asarray(gaps, dtype=float), 0.0)


# ---------------------------------------------------------------------------
# Arrays diarios por bloque (precomputo por f; lineal en saldo_start)
# ---------------------------------------------------------------------------

@dataclass
class BlockDayArrays:
    """Series diarias de un bloque RELATIVAS al saldo de inicio del bloque.

    ``rel_before``/``rel_closed`` = PnL cerrado acumulado antes/al final del
    dia; ``rel_float`` = peor flotante del dia (incluye el cerrado previo).
    equity_abs(d) = account_size + saldo_start + rel_*(d).

    Campos trailing (plan Apex Fase 2, default None → FTMO no paga memoria ni
    CPU; solo se rellenan con ``block_day_arrays_portfolio(...,
    trailing=True)``): ``rel_low``/``rel_high``/``rel_exc`` alineados a
    ``day_ord`` = min lower_t / max upper_t / max excursión del día(-sesión)
    con cummax DE BLOQUE (no reseteado por día); ``block_low``/``block_high``/
    ``block_exc`` = escalares del bloque (min_t lower_t, max_t upper_t,
    max_t(cummax(upper) − lower_t)). Ver :mod:`src.risk.propfirm_trailing`.
    """

    day_ord: np.ndarray      # (d,) ordinales de dia, ordenados
    rel_before: np.ndarray   # (d,) PnL cerrado antes del dia
    rel_closed: np.ndarray   # (d,) PnL cerrado al final del dia
    rel_float: np.ndarray    # (d,) peor equity flotante del dia (relativa)
    opens_day: np.ndarray    # (d,) trades abiertos ese dia
    total_pnl: float
    rel_low: np.ndarray | None = None    # (d,) min lower_t del dia-sesion
    rel_high: np.ndarray | None = None   # (d,) max upper_t del dia-sesion
    rel_exc: np.ndarray | None = None    # (d,) max excursion (cummax de bloque)
    block_low: float | None = None       # min_t lower_t (escalar de bloque)
    block_high: float | None = None      # max_t upper_t
    block_exc: float | None = None       # max_t(cummax(upper)_t − lower_t)


def block_day_arrays(b: DayBlock, lots_f: np.ndarray, pv: float) -> BlockDayArrays:
    """Precomputa las series diarias de un bloque a un ``f`` (lotes fijos)."""
    k = b.n_trades
    lots = lots_f[b.trade_local_id]
    pnl_trade = lots * b.pnl_per_lot

    # PnL cerrado por dia de cierre.
    order = np.argsort(b.close_ord, kind="mergesort")
    cd_s, pt_s = b.close_ord[order], pnl_trade[order]
    uniq_c, inv = np.unique(cd_s, return_inverse=True)
    pnl_per_close_day = np.zeros(len(uniq_c))
    np.add.at(pnl_per_close_day, inv, pt_s)
    closed_cum = np.cumsum(pnl_per_close_day)
    close_idx = {int(d): i for i, d in enumerate(uniq_c)}

    # Flotante worst por dia (suma de valles de todos los trades abiertos).
    day_float: dict[int, float] = {}
    for j in range(k):
        for d, v in b.valle_precio[j].items():
            o = pd.Timestamp(d).toordinal()
            day_float[o] = day_float.get(o, 0.0) + float(v) * lots[j] * pv

    # Opens por dia.
    day_opens: dict[int, int] = {}
    for o in b.open_ord:
        day_opens[int(o)] = day_opens.get(int(o), 0) + 1

    all_days = np.array(sorted(set(day_float) | set(close_idx) | set(day_opens)))
    rel_before = np.zeros(len(all_days))
    rel_closed = np.zeros(len(all_days))
    rel_float = np.zeros(len(all_days))
    opens = np.zeros(len(all_days), dtype=int)
    for j, d in enumerate(all_days):
        d = int(d)
        ci = close_idx.get(d)
        rel_before[j] = closed_cum[ci - 1] if (ci is not None and ci > 0) else (
            closed_cum[_prev_close_idx(uniq_c, d)] if _prev_close_idx(uniq_c, d) is not None else 0.0
        )
        rel_closed[j] = closed_cum[ci] if ci is not None else rel_before[j]
        rel_float[j] = rel_before[j] + day_float.get(d, 0.0)
        opens[j] = day_opens.get(d, 0)
    return BlockDayArrays(
        day_ord=all_days.astype(int), rel_before=rel_before,
        rel_closed=rel_closed, rel_float=rel_float, opens_day=opens,
        total_pnl=float(closed_cum[-1]) if len(closed_cum) else 0.0,
    )


def _prev_close_idx(uniq_c: np.ndarray, d: int) -> int | None:
    """Indice del ultimo dia de cierre estrictamente anterior a ``d`` (o None)."""
    i = int(np.searchsorted(uniq_c, d)) - 1
    return i if i >= 0 else None


def precompute_day_arrays(
    blocks: list[DayBlock], lots: np.ndarray, pv: float
) -> list[list[BlockDayArrays]]:
    """Para cada f de la grilla, los arrays diarios de todos los bloques."""
    return [[block_day_arrays(b, lots[fi], pv) for b in blocks] for fi in range(lots.shape[0])]


# ---------------------------------------------------------------------------
# Bloques flat-flat de CARTERA (multi-pata) — plan propfirm-carteras v3 paso 1
# ---------------------------------------------------------------------------

@dataclass
class PortfolioDayBlock:
    """Episodio flat-flat de una CARTERA (varias estrategias), con fechas por trade.

    Como :class:`DayBlock` pero multi-pata: ``sym_id`` identifica la pata de cada
    trade y ``n_no_valley`` cuenta los trades sin valle OHLC (paridad BAJO-3:
    se incluyen con valle vacío — su PnL cerrado cuenta — y se cuentan porque su
    ``open_t``/``close_t`` (desde Fase 2): timestamps por trade — el
    constructor trailing (``src.risk.propfirm_trailing``) los necesita para el
    calendario conjunto (incluye TODOS los ct, cubiertos o no).
    """

    idx: int
    start: pd.Timestamp
    end: pd.Timestamp
    duration_days: float
    sym_id: np.ndarray           # (k,) índice de pata por trade
    trade_local_id: np.ndarray   # (k,) índice del trade en su pata
    open_ord: np.ndarray         # (k,) ordinal de fecha de apertura
    close_ord: np.ndarray        # (k,) ordinal de fecha de cierre
    valle_precio: list           # (k,) dict {date: valle_precio} (f-indep, <=0)
    pnl_per_lot: np.ndarray      # (k,) f-indep (USD por lote)
    n_trades: int
    n_no_valley: int
    open_t: np.ndarray | None = None   # (k,) timestamp de apertura (trailing, Fase 2)
    close_t: np.ndarray | None = None  # (k,) timestamp de cierre (trailing, Fase 2)


def build_day_blocks_portfolio(
    strategies: dict,
    oos_start: pd.Timestamp | None = None,
    day_labeler=None,
) -> list[PortfolioDayBlock]:
    """Bloques flat-flat del calendario CONJUNTO de una cartera (calendario real).

    Versión multi-pata de :func:`build_day_blocks`: ``detect_flatflat_blocks``
    sobre todas las patas y, por bloque, adjunta los trades de TODAS las patas
    contenidos (open >= bstart ∧ close <= bend), incluidos los de valle vacío
    (BAJO-3, contados en ``n_no_valley``). ``oos_start`` descarta bloques enteros
    anteriores al corte (anti-lookahead, M2-8: el denominador de coverage del
    batch cuenta los trades de esos bloques como missing, N8).

    ``day_labeler``: callable ts→date opcional (eje de sesiones ET, ver
    :mod:`src.risk.session_axis`) aplicado a ``open_ord`` de cada pata; las
    patas deben venir precomputadas con el MISMO labeler (``close_d`` y claves
    de ``valle`` ya etiquetadas). None = día calendario (bit-idéntico).
    """
    oc = {k: (s.open_t, s.close_t) for k, s in strategies.items()}
    blocks = detect_flatflat_blocks(oc)
    if day_labeler is None:
        open_ord_by_key = {
            k: pd.to_datetime(s.open_t).normalize().to_numpy()
            for k, s in strategies.items()
        }
    else:
        open_ord_by_key = {
            k: np.asarray(day_labeler(pd.to_datetime(s.open_t)))
            for k, s in strategies.items()
        }
    out: list[PortfolioDayBlock] = []
    for bstart, bend in blocks:
        if oos_start is not None and pd.Timestamp(bstart) < oos_start:
            continue
        sym_id, tloc, oo, cc, vp, ppl = [], [], [], [], [], []
        ots, cts = [], []
        n_nv = 0
        for si, (key, s) in enumerate(strategies.items()):
            open_ord = open_ord_by_key[key]
            for i in range(s.n_trades):
                ot, ct = s.open_t[i], s.close_t[i]
                if pd.isna(ot) or pd.isna(ct):
                    continue
                if ot >= bstart and ct <= bend:
                    v = s.valle.get(i, {})
                    if not v:
                        n_nv += 1
                    sym_id.append(si)
                    tloc.append(i)
                    oo.append(pd.Timestamp(open_ord[i]).toordinal())
                    cc.append(pd.Timestamp(s.close_d[i]).toordinal())
                    vp.append(v)
                    ppl.append(s.pnl_per_lot[i])
                    ots.append(ot)
                    cts.append(ct)
        if not sym_id:
            continue
        out.append(PortfolioDayBlock(
            idx=len(out), start=bstart, end=bend,
            duration_days=(pd.Timestamp(bend) - pd.Timestamp(bstart)).total_seconds() / 86400.0,
            sym_id=np.array(sym_id, dtype=int),
            trade_local_id=np.array(tloc, dtype=int),
            open_ord=np.array(oo, dtype=int),
            close_ord=np.array(cc, dtype=int),
            valle_precio=vp,
            pnl_per_lot=np.array(ppl, dtype=float),
            n_trades=len(sym_id),
            n_no_valley=n_nv,
            open_t=np.array(ots, dtype="datetime64[ns]"),
            close_t=np.array(cts, dtype="datetime64[ns]"),
        ))
    for j, b in enumerate(out):
        b.idx = j
    return out


def _day_float_bound(b: PortfolioDayBlock, lots: np.ndarray,
                     pv_by_sym: dict[int, float]) -> dict[int, float]:
    """Flotante diario de la COTA (corr=1 intra-día): suma de los valles por día
    de todos los trades abiertos (M2-1; inutilizable a f≥1% — ver float_mode
    "joint")."""
    day_float: dict[int, float] = {}
    for j in range(b.n_trades):
        pv = pv_by_sym[b.sym_id[j]]
        for d, v in b.valle_precio[j].items():
            o = pd.Timestamp(d).toordinal()
            day_float[o] = day_float.get(o, 0.0) + float(v) * lots[j] * pv
    return day_float


def _assemble_block_day_arrays(b: PortfolioDayBlock, lots: np.ndarray,
                               day_float: dict[int, float]) -> BlockDayArrays:
    """Ensambla las series diarias de un bloque de cartera dado el flotante.

    Común a ambos ``float_mode``: PnL cerrado por día de cierre, rel_before /
    rel_closed / opens_day (unión entre patas = trading day a nivel cuenta,
    regla FTMO literal) y ``rel_float = rel_before + day_float``.
    """
    pnl_trade = lots * b.pnl_per_lot

    # PnL cerrado por dia de cierre.
    order = np.argsort(b.close_ord, kind="mergesort")
    cd_s, pt_s = b.close_ord[order], pnl_trade[order]
    uniq_c, inv = np.unique(cd_s, return_inverse=True)
    pnl_per_close_day = np.zeros(len(uniq_c))
    np.add.at(pnl_per_close_day, inv, pt_s)
    closed_cum = np.cumsum(pnl_per_close_day)
    close_idx = {int(d): i for i, d in enumerate(uniq_c)}

    # Opens por dia.
    day_opens: dict[int, int] = {}
    for o in b.open_ord:
        day_opens[int(o)] = day_opens.get(int(o), 0) + 1

    all_days = np.array(sorted(set(day_float) | set(close_idx) | set(day_opens)))
    rel_before = np.zeros(len(all_days))
    rel_closed = np.zeros(len(all_days))
    rel_float = np.zeros(len(all_days))
    opens = np.zeros(len(all_days), dtype=int)
    for j, d in enumerate(all_days):
        d = int(d)
        ci = close_idx.get(d)
        rel_before[j] = closed_cum[ci - 1] if (ci is not None and ci > 0) else (
            closed_cum[_prev_close_idx(uniq_c, d)] if _prev_close_idx(uniq_c, d) is not None else 0.0
        )
        rel_closed[j] = closed_cum[ci] if ci is not None else rel_before[j]
        rel_float[j] = rel_before[j] + day_float.get(d, 0.0)
        opens[j] = day_opens.get(d, 0)
    return BlockDayArrays(
        day_ord=all_days.astype(int), rel_before=rel_before,
        rel_closed=rel_closed, rel_float=rel_float, opens_day=opens,
        total_pnl=float(closed_cum[-1]) if len(closed_cum) else 0.0,
    )


def _joint_float_columns(b: PortfolioDayBlock,
                         minute_by_sym: dict[int, dict]) -> tuple | None:
    """Columnas de flotante M1 por trade (f-INDEPENDIENTE, unidades de precio).

    Calendario conjunto = unión de los timestamps M1 de las series por minuto
    de los trades del bloque (``minute_valle``, ver
    :func:`src.risk.mc_portfolio.attach_minute_valleys`). Por trade, su serie
    con ffill físico dentro de [open, close] (una posición viva sin barra en un
    minuto conserva su último flotante; entre la última barra y el cierre
    también) y 0 fuera. Trades sin valle: no aportan (paridad BAJO-3).

    Devuelve ``(uniq_ords, day_starts, cols)`` — ordinales de día de los
    minutos, offsets de inicio de día para ``np.minimum.reduceat`` y la lista
    de columnas — o None si ningún trade del bloque tiene serie.
    """
    series = []
    for j in range(b.n_trades):
        mv = minute_by_sym.get(int(b.sym_id[j])) or {}
        entry = mv.get(int(b.trade_local_id[j]))
        if entry is None:
            continue
        ts, vals, ot, ct = entry
        series.append((pd.Timestamp(ot), pd.Timestamp(ct), ts, vals))
    if not series:
        return None
    cal = np.unique(np.concatenate([x[2] for x in series]))
    m = len(cal)
    # Ordinal de día por minuto (date.toordinal() = día unix + 719163).
    ords = cal.astype("datetime64[D]").astype(np.int64) + 719163
    day_starts = np.flatnonzero(np.r_[True, np.diff(ords) != 0])
    cols = []
    for ot, ct, ts, vals in series:
        pos = cal.searchsorted(ts)
        col = np.full(m, np.nan)
        col[pos] = vals
        # ffill dentro de [ot, ct]; fuera, 0 (numpy puro: último índice válido).
        src = np.where(np.isnan(col), 0, np.arange(m))
        np.maximum.accumulate(src, out=src)
        col = col[src]
        col[np.isnan(col)] = 0.0
        col[(cal < ot.to_datetime64()) | (cal > ct.to_datetime64())] = 0.0
        cols.append(col)
    return ords[day_starts], day_starts, cols


def _joint_day_float_block(b: PortfolioDayBlock, lots: np.ndarray,
                           pv_by_sym: dict[int, float],
                           minute_by_sym: dict[int, dict],
                           joint_cols: tuple | None = None) -> dict[int, float]:
    """``day_float`` conjunto intradía de un bloque a un ``f`` (variante M2-1).

    min por día de calendario de Σ_trades vivos flotante_M1(trade, t)×lotes×pv.
    ``joint_cols`` opcional: salida cacheada de :func:`_joint_float_columns`
    para reusar las columnas entre los f de una grilla (lo hace
    :func:`precompute_day_arrays_portfolio`).
    """
    if joint_cols is None:
        joint_cols = _joint_float_columns(b, minute_by_sym)
    if joint_cols is None:
        return {}
    uniq_ords, day_starts, cols = joint_cols
    acc = np.zeros(len(cols[0]))
    k = 0
    for j in range(b.n_trades):
        mv = minute_by_sym.get(int(b.sym_id[j])) or {}
        if mv.get(int(b.trade_local_id[j])) is None:
            continue
        acc += cols[k] * (lots[j] * pv_by_sym[b.sym_id[j]])
        k += 1
    mins = np.minimum.reduceat(acc, day_starts)
    return {int(o): float(v) for o, v in zip(uniq_ords, mins)}


def _windows_day_float(b: PortfolioDayBlock, lots: np.ndarray,
                       adverse_by_sym: dict[int, np.ndarray]) -> dict[int, float]:
    """``day_float`` por VENTANAS de trade de un bloque a un ``f`` (``float_mode="windows"``).

    Por trade j (``n_j = lots[j]``): ``w_j = -adverse[sym_j][local_j] * n_j``
    (peor flotante NETO de la comisión RT, USD por lote >= 0 en
    ``adverse_by_sym``) y ``pnl_j = pnl_per_lot[j] * n_j``. Eventos
    ``(sesión, instante, delta)``:

    - ``(open_ord, open_t, w)``: el trade queda VIVO en su peor estado;
    - para cada sesión ``s`` con ``open_ord < s <= close_ord``, un evento de
      arrastre ``(s, -inf, w)`` (el trade ya vive al abrir la sesión ``s``);
    - ``(close_ord, close_t, pnl - w)``: al cerrar pasa de ``w`` a ``pnl``.

    Orden total ``(sesión, instante, delta)``: ``-inf`` antes de cualquier
    instante y, en empate de instante, deltas negativos primero (conservador:
    un cierre y una apertura simultáneos cuentan ambos en estado vivo).
    ``day_float[s]`` = mínimo de la suma acumulada de los eventos de ``s``
    desde 0. Es una cota inferior de ``min_t(realized_<=t + flotante_t)`` de
    la sesión relativa al balance de inicio de sesión, válida SIN suponer
    cuándo ocurrió el MAE (prueba: plan float-valleys-ventanas §W2).

    El delta de cierre ``pnl - w`` puede quedar ligeramente negativo dentro de
    la tolerancia del extractor (enmienda V2): NO se afirma ``>= 0``.

    Devuelve ``{ordinal de sesión: day_float}``; incluye las sesiones de
    arrastre intermedias (fines de semana/festivos incluidos: conservador y
    sin breaches nuevos, el umbral T no cambia sin cierres).
    """
    if b.open_t is None or b.close_t is None:
        raise ValueError("float_mode='windows' requiere open_t/close_t por trade "
                         "en el PortfolioDayBlock")
    if pd.isna(b.open_t).any() or pd.isna(b.close_t).any():
        # NaT → int64 min colisionaría con el -inf de los arrastres.
        raise ValueError("float_mode='windows': open_t/close_t con NaT")
    k = b.n_trades
    lots = np.asarray(lots, dtype=float)
    sym = np.asarray(b.sym_id, dtype=int)
    loc = np.asarray(b.trade_local_id, dtype=int)
    adv = np.empty(k, dtype=float)
    for s in np.unique(sym):
        a = adverse_by_sym.get(int(s))
        if a is None:
            raise ValueError(
                f"float_mode='windows': adverse_by_sym sin la pata {int(s)}"
            )
        a = np.asarray(a, dtype=float)
        m = sym == s
        if loc[m].min() < 0 or loc[m].max() >= len(a):
            raise ValueError(
                f"float_mode='windows': adverse_by_sym[{int(s)}] (len {len(a)}) "
                f"no cubre el id local de trade {int(loc[m].max())}"
            )
        adv[m] = a[loc[m]]
    w = -adv * lots
    pnl = np.asarray(b.pnl_per_lot, dtype=float) * lots
    o_ord = np.asarray(b.open_ord, dtype=np.int64)
    c_ord = np.asarray(b.close_ord, dtype=np.int64)
    n_drag = c_ord - o_ord
    if (n_drag < 0).any():
        raise ValueError("float_mode='windows': trade con close_ord < open_ord "
                         "(labeler inconsistente entre open y close)")
    o_ts = np.asarray(b.open_t, dtype="datetime64[ns]").astype(np.int64)
    c_ts = np.asarray(b.close_t, dtype="datetime64[ns]").astype(np.int64)
    # Arrastres: una fila por (trade, sesión s en (open_ord, close_ord]).
    tot = int(n_drag.sum())
    rep = np.repeat(np.arange(k), n_drag)
    starts = np.cumsum(n_drag) - n_drag
    off = np.arange(tot, dtype=np.int64) - np.repeat(starts, n_drag) + 1
    day = np.concatenate([o_ord, c_ord, o_ord[rep] + off])
    ts = np.concatenate([o_ts, c_ts, np.full(tot, np.iinfo(np.int64).min)])
    delta = np.concatenate([w, pnl - w, w[rep]])
    order = np.lexsort((delta, ts, day))   # clave primaria: day; luego ts; luego delta
    day, delta = day[order], delta[order]
    seg = np.flatnonzero(np.r_[True, day[1:] != day[:-1]])
    cs = np.cumsum(delta)
    base = np.r_[0.0, cs[:-1]][seg]        # acumulado antes de cada sesión
    level = cs - np.repeat(base, np.diff(np.r_[seg, len(day)]))
    mins = np.minimum.reduceat(level, seg)
    return {int(d): float(v) for d, v in zip(day[seg], mins)}


def _minute_columns(b: PortfolioDayBlock, minute_by_sym: dict[int, dict],
                    day_labeler=None):
    """Calendario + columnas adversas por trade del modo ``"minute"`` (f-INDEPENDIENTE).

    Reutiliza el constructor adjudicado del trailing
    (:func:`src.risk.propfirm_trailing.build_trailing_columns`, que NO se
    modifica) con la serie adversa como adversa Y favorable, y descarta la
    favorable en cuanto se construye (``fav_cols = adv_cols``: alias, no se
    guarda una segunda copia; el coste extra es solo de CPU y transitorio).

    Fail-loud: ``ValueError`` si algún trade del bloque no tiene serie en
    ``minute_by_sym`` (``{sym_id: {local_id: (ts, vals, ot, ct)}}``). Sin ella
    un trade sin cobertura aportaría 0 al flotante y los días anteriores al
    primer minuto cubierto quedarían a 0 en :func:`align_trailing_days`.
    """
    for j in range(b.n_trades):
        si, tl = int(b.sym_id[j]), int(b.trade_local_id[j])
        if (minute_by_sym.get(si) or {}).get(tl) is None:
            raise ValueError(
                f"float_mode='minute': el trade {tl} de la pata {si} no tiene "
                "serie en minute_by_sym (se exige serie de TODOS los trades "
                "del bloque)"
            )
    tcols = build_trailing_columns(b, minute_by_sym, minute_by_sym,
                                   day_labeler=day_labeler)
    tcols.fav_cols = tcols.adv_cols
    return tcols


def _minute_day_arrays(b: PortfolioDayBlock, lots: np.ndarray,
                       pv_by_sym: dict[int, float], tcols,
                       day_labeler=None) -> BlockDayArrays:
    """Series diarias de un bloque a un ``f`` en modo ``"minute"`` (ver
    :func:`block_day_arrays_portfolio`).

    ``rel_float[d] = min_{t en minutos del día d}(realized_≤t + Σ vivos
    adv_j(t)·lots_j·pv_j)`` relativo al saldo de inicio del bloque. El
    conjunto de días = días con minutos ∪ TODOS los días ``open_ord_j <= d <=
    close_ord_j`` (fines de semana y festivos con posiciones abiertas
    incluidos): FTMO recalcula el suelo diario cada medianoche, y el estado
    arrastrado de un día sin minutos propios se evalúa contra ese suelo nuevo.
    """
    res = trailing_block_result(b, lots, pv_by_sym, None, None,
                                day_labeler=day_labeler, tcols=tcols)
    days = {int(d) for d in res.day_ord}
    for o, c in zip(b.open_ord, b.close_ord):
        days.update(range(int(o), int(c) + 1))
    bda = _assemble_block_day_arrays(b, lots, {d: 0.0 for d in days})
    bda.rel_float = align_trailing_days(res, bda.day_ord)[0]
    return bda


def block_day_arrays_portfolio(
    b: PortfolioDayBlock,
    lots_by_sym: dict[int, np.ndarray],
    pv_by_sym: dict[int, float],
    f_idx: int,
    float_mode: str = "bound",
    minute_by_sym: dict[int, dict] | None = None,
    trailing: bool = False,
    minute_fav_by_sym: dict[int, dict] | None = None,
    day_labeler=None,
    trailing_cols=None,
    adverse_by_sym: dict[int, np.ndarray] | None = None,
    minute_cols=None,
) -> BlockDayArrays:
    """Precomputa las series diarias de un bloque de CARTERA a un ``f``.

    Misma lógica que :func:`block_day_arrays` pero por trade j usa
    ``lots_by_sym[sym_j][f_idx, local_j]`` y ``pv_by_sym[sym_j]``. Devuelve el
    mismo :class:`BlockDayArrays` (opens_day = unión entre patas = trading day a
    nivel cuenta, regla FTMO literal).

    ``float_mode``:
      - ``"bound"`` (default, comportamiento histórico): el flotante diario
        suma los valles por día de todas las patas abiertas (cota corr=1
        intra-día, M2-1 — el diagnóstico midió su sesgo: ΔP(pass) hasta
        +26pp a f=1.5%; queda como referencia/comparación).
      - ``"joint"``: flotante conjunto intradía (veredicto M2-1 implementado):
        min por día de la suma de flotantes M1 de todas las posiciones vivas,
        con ffill físico dentro de [open, close]. Requiere ``minute_by_sym``
        = ``{sym_id: StrategyMC.minute_valle}`` (ver
        :func:`src.risk.mc_portfolio.attach_minute_valleys`).
      - ``"windows"`` (plan float-valleys-ventanas, patas de futuros nativas
        desde ``.sqx``): cota de VENTANAS por trade (:func:`_windows_day_float`)
        a partir de ``adverse_by_sym = {sym_id: array (n_trades_pata,)}``
        con el peor flotante adverso NETO de comisión RT en USD por lote
        (``(mae_usd + comm_usd)/lots``, >= 0; ver
        :class:`src.risk.futures_legs.FuturesLegData`), indexado por el id
        local del trade en su pata. Semántica:

        * en windows ``rel_float`` INCLUYE el realized intradía (la suma de
          eventos arranca en el balance de inicio de sesión y los cierres
          inyectan su PnL), a diferencia de ``bound``/``joint``; por eso
          ``rel_low = rel_float`` (copia). ``rel_high``/``rel_exc`` y los
          escalares ``block_*`` quedan None;
        * es cota inferior válida SIN suponer cuándo ocurrió el MAE (prueba
          en el plan, W2) y ``rel_low <= rel_closed`` por sesión;
        * es NETA de la comisión RT (la equity de SQX la descuenta mientras
          el trade vive): el caveat C2 "rel_low gross" NO aplica, y puede
          ser algo más conservadora que FTMO si este solo descuenta la
          comisión de entrada (V4);
        * los arrastres en fines de semana o festivos son conservadores y no
          crean breaches nuevos (T no cambia sin cierres);
        * solo es válido para walkers ``eod_trailing``
          (``evaluate_phase_path_eod_trail``); el walker trailing Apex rechaza
          estos bloques porque ``block_low is None``. ``trailing=True`` con
          windows es ``ValueError`` (V3);
        * fail-loud: ``ValueError`` si falta ``adverse_by_sym``, una pata o un
          id local.
      - ``"minute"`` (plan cfd-sqx-minuto-par, patas .sqx CFD con OHLC M1):
        flotante POR MINUTO con el realizado intradía incluido,
        ``rel_float[d] = min_{t en minutos del día d}(realized_≤t + Σ trades
        vivos adv_j(t)·lots_j·pv_j)`` relativo al saldo de inicio del bloque.
        Requiere ``minute_by_sym = {sym_id: {local_id: (ts, vals, ot, ct)}}``
        (mismo formato que ``StrategyMC.minute_valle``) con serie de TODOS los
        trades del bloque; las series van en unidades de PRECIO y ya NETAS de
        costes (spread, comisión, swap): el motor no sabe de ellos. Reutiliza
        el constructor del trailing (:mod:`src.risk.propfirm_trailing`) con
        ``fav = adv`` descartada. Semántica:

        * convenciones heredadas del trailing: envolvente activa en
          ``[ot, ct)``, realizado inyectado EN ct, ffill físico entre barras;
          los días sin minutos propios toman el estado del último minuto del
          día anterior con minutos (realizado + flotante arrastrados);
        * el conjunto de días incluye TODOS los ordinales
          ``open_ord_j <= d <= close_ord_j`` de algún trade (fines de semana y
          festivos con posiciones abiertas): FTMO recalcula el suelo diario
          cada medianoche, y un viernes con beneficio cerrado deja el suelo
          del sábado más alto con la misma equity arrastrada;
        * es una COTA por minuto (el peor precio de cada minuto para todas las
          posiciones a la vez y los costes cargados desde la apertura), NO la
          equity exacta. Residual ANTI-conservador heredado (R2 del plan
          Apex): el vaivén entre la última barra y ct no se acota (≈ 0 con
          cierres en minuto exacto);
        * solo vale para perfiles ESTÁTICOS (suelo diario + max loss
          estático): ``rel_low``/``rel_high``/``rel_exc`` y ``block_*`` quedan
          None;
        * fail-loud (``ValueError``): sin ``minute_by_sym``, con
          ``trailing=True``, o si falta la serie de algún trade del bloque.
        ``minute_cols`` opcional: salida cacheada de :func:`_minute_columns`
        (f-independiente) para reuso entre f.

    ``trailing=True`` (plan Apex Fase 2): ADEMÁS de los campos de siempre
    (``rel_float`` sigue gobernado por ``float_mode``, intacto), rellena
    ``rel_low``/``rel_high``/``rel_exc`` y los escalares ``block_low``/
    ``block_high``/``block_exc`` con el constructor SEPARADO de
    :mod:`src.risk.propfirm_trailing` (envolvente activa en [ot, ct), realized
    inyectado EN ct — NO reutiliza la máscara del joint-float). Requiere
    ``minute_by_sym`` (adversas) Y ``minute_fav_by_sym`` (favorables, de
    ``attach_minute_valleys(..., with_peaks=True)``); si falta alguna pata o
    picos, raise ruidoso. ``day_labeler`` etiqueta los minutos por sesión ET
    (None = día calendario). ``trailing_cols`` opcional: salida cacheada de
    ``build_trailing_columns`` (f-independiente) para reuso entre f.

    Camino solo-portfolio: el mono se obtiene como portfolio de 1 pata (la
    paridad mono ≡ joint de 1 pata ya está testeada), así que
    :func:`block_day_arrays` mono NO lleva trailing.
    """
    k = b.n_trades
    lots = np.array([
        lots_by_sym[b.sym_id[j]][f_idx, b.trade_local_id[j]] for j in range(k)
    ])
    if float_mode == "bound":
        day_float = _day_float_bound(b, lots, pv_by_sym)
    elif float_mode == "joint":
        if minute_by_sym is None:
            raise ValueError("float_mode='joint' requiere minute_by_sym "
                             "(attach_minute_valleys por pata)")
        day_float = _joint_day_float_block(b, lots, pv_by_sym, minute_by_sym)
    elif float_mode == "minute":
        if trailing:
            raise ValueError(
                "float_mode='minute' no admite trailing=True: es solo para "
                "perfiles estáticos (rel_low/rel_high/rel_exc quedan None)"
            )
        if minute_by_sym is None:
            raise ValueError("float_mode='minute' requiere minute_by_sym "
                             "({sym_id: {local_id: (ts, vals, ot, ct)}})")
        tcols = (minute_cols if minute_cols is not None
                 else _minute_columns(b, minute_by_sym, day_labeler))
        return _minute_day_arrays(b, lots, pv_by_sym, tcols, day_labeler)
    elif float_mode == "windows":
        if trailing:
            raise ValueError(
                "float_mode='windows' no admite trailing=True (V3): rel_low "
                "windows es la cota por eventos para walkers eod_trailing; "
                "el trailing Apex usa la infra por minuto"
            )
        if adverse_by_sym is None:
            raise ValueError("float_mode='windows' requiere adverse_by_sym "
                             "({sym_id: adverso neto USD/lote por trade})")
        day_float = _windows_day_float(b, lots, adverse_by_sym)
        bda = _assemble_block_day_arrays(b, lots, day_float)
        bda.rel_low = bda.rel_float.copy()
        return bda
    else:
        raise ValueError("float_mode desconocido: " + repr(float_mode))
    bda = _assemble_block_day_arrays(b, lots, day_float)
    if trailing:
        if minute_by_sym is None or minute_fav_by_sym is None:
            raise ValueError(
                "trailing=True requiere minute_by_sym (adversas) Y "
                "minute_fav_by_sym (favorables): attach_minute_valleys(..., "
                "with_peaks=True) por pata"
            )
        res = trailing_block_result(
            b, lots, pv_by_sym, minute_by_sym, minute_fav_by_sym,
            day_labeler=day_labeler, tcols=trailing_cols,
        )
        rel_low, rel_high, rel_exc = align_trailing_days(res, bda.day_ord)
        bda.rel_low, bda.rel_high, bda.rel_exc = rel_low, rel_high, rel_exc
        bda.block_low, bda.block_high, bda.block_exc = (
            res.block_low, res.block_high, res.block_exc,
        )
    return bda


def precompute_day_arrays_portfolio(
    blocks: list[PortfolioDayBlock],
    lots_by_sym: dict[int, np.ndarray],
    pv_by_sym: dict[int, float],
    n_f: int,
    float_mode: str = "bound",
    minute_by_sym: dict[int, dict] | None = None,
    trailing: bool = False,
    minute_fav_by_sym: dict[int, dict] | None = None,
    day_labeler=None,
    adverse_by_sym: dict[int, np.ndarray] | None = None,
) -> list[list[BlockDayArrays]]:
    """Para cada f de la grilla, los arrays diarios de todos los bloques de cartera.

    Con ``float_mode="windows"`` (cota de ventanas por trade, ver
    :func:`block_day_arrays_portfolio`) requiere ``adverse_by_sym`` y
    ``trailing=False`` (``ValueError`` si no); bucle por bloque y f, sin caché
    especial (el coste es una ordenación de eventos por bloque y f).

    Con ``float_mode="joint"`` las columnas M1 por trade (f-independientes) se
    construyen UNA vez por bloque y se ponderan por lotes(f)×pv en cada f — el
    min diario de la suma ponderada no es prefactorizable (re-floor discreto de
    lotes), pero la parte cara (calendario conjunto + ffill) sí.

    Con ``trailing=True`` (plan Apex Fase 2) cada bloque lleva además
    ``rel_low``/``rel_high``/``rel_exc`` y los escalares de bloque (ver
    :func:`block_day_arrays_portfolio`); el calendario trailing y las columnas
    adv/fav (f-independientes) también se construyen una vez por bloque —
    el stream realized sí es f-dependiente (lotes×pnl_per_lot). Las columnas
    joint de ``rel_float`` (si ``float_mode="joint"``) se cachean igual.

    Con ``float_mode="minute"`` (ver :func:`block_day_arrays_portfolio`) las
    columnas por minuto (f-independientes) se construyen UNA vez por bloque,
    bloque a bloque (la favorable se descarta al construirse y las columnas
    del bloque se liberan al pasar al siguiente), y se ponderan por f.
    ``ValueError`` con ``trailing=True``, sin ``minute_by_sym`` o si falta la
    serie de algún trade.
    """
    if float_mode == "minute":
        if trailing:
            raise ValueError(
                "float_mode='minute' no admite trailing=True: es solo para "
                "perfiles estáticos"
            )
        if minute_by_sym is None:
            # Validado aquí además de por bloque: sin bloques no habría llamada.
            raise ValueError("float_mode='minute' requiere minute_by_sym "
                             "({sym_id: {local_id: (ts, vals, ot, ct)}})")
        out_m: list[list[BlockDayArrays]] = [[] for _ in range(n_f)]
        for b in blocks:
            mcols = _minute_columns(b, minute_by_sym, day_labeler)
            for fi in range(n_f):
                out_m[fi].append(block_day_arrays_portfolio(
                    b, lots_by_sym, pv_by_sym, fi, float_mode="minute",
                    minute_by_sym=minute_by_sym, day_labeler=day_labeler,
                    minute_cols=mcols,
                ))
        return out_m
    if float_mode == "windows":
        if trailing or adverse_by_sym is None:
            # Validado aquí además de por bloque: sin bloques no habría llamada.
            raise ValueError(
                "float_mode='windows' requiere adverse_by_sym y trailing=False"
            )
        out_w: list[list[BlockDayArrays]] = [[] for _ in range(n_f)]
        for b in blocks:
            for fi in range(n_f):
                out_w[fi].append(block_day_arrays_portfolio(
                    b, lots_by_sym, pv_by_sym, fi, float_mode="windows",
                    adverse_by_sym=adverse_by_sym,
                ))
        return out_w
    if trailing and (minute_by_sym is None or minute_fav_by_sym is None):
        raise ValueError(
            "trailing=True requiere minute_by_sym (adversas) Y "
            "minute_fav_by_sym (favorables): attach_minute_valleys(..., "
            "with_peaks=True) por pata"
        )
    if float_mode == "bound" and not trailing:
        return [
            [block_day_arrays_portfolio(b, lots_by_sym, pv_by_sym, fi) for b in blocks]
            for fi in range(n_f)
        ]
    if float_mode not in ("bound", "joint"):
        raise ValueError("float_mode desconocido: " + repr(float_mode))
    if float_mode == "joint" and minute_by_sym is None:
        raise ValueError("float_mode='joint' requiere minute_by_sym "
                         "(attach_minute_valleys por pata)")
    # Cachés f-independientes por bloque: columnas joint (rel_float) y
    # calendario + columnas trailing (adv/fav).
    joint_cols_by_block = (
        [_joint_float_columns(b, minute_by_sym) for b in blocks]
        if float_mode == "joint" else [None] * len(blocks)
    )
    tcols_by_block = (
        [build_trailing_columns(b, minute_by_sym, minute_fav_by_sym,
                                day_labeler=day_labeler) for b in blocks]
        if trailing else [None] * len(blocks)
    )
    out: list[list[BlockDayArrays]] = [[] for _ in range(n_f)]
    for b, joint_cols, tcols in zip(blocks, joint_cols_by_block, tcols_by_block):
        for fi in range(n_f):
            lots = np.array([
                lots_by_sym[b.sym_id[j]][fi, b.trade_local_id[j]]
                for j in range(b.n_trades)
            ])
            if float_mode == "joint":
                day_float = _joint_day_float_block(b, lots, pv_by_sym,
                                                   minute_by_sym,
                                                   joint_cols=joint_cols)
            else:
                day_float = _day_float_bound(b, lots, pv_by_sym)
            bda = _assemble_block_day_arrays(b, lots, day_float)
            if trailing:
                res = trailing_block_result(
                    b, lots, pv_by_sym, minute_by_sym, minute_fav_by_sym,
                    day_labeler=day_labeler, tcols=tcols,
                )
                rel_low, rel_high, rel_exc = align_trailing_days(res, bda.day_ord)
                bda.rel_low, bda.rel_high, bda.rel_exc = rel_low, rel_high, rel_exc
                bda.block_low, bda.block_high, bda.block_exc = (
                    res.block_low, res.block_high, res.block_exc,
                )
            out[fi].append(bda)
    return out


# ---------------------------------------------------------------------------
# Evaluacion de paths
# ---------------------------------------------------------------------------

@dataclass
class PhasePathResult:
    outcome: str          # 'pass' | 'fail' | 'neither' | 'expired' (trailing)
    days: float           # dias calendario hasta el stop (o total si neither)
    trading_days: int
    terminal: float       # equity absoluta al stop
    expired: bool = False  # True si outcome == 'expired' (lifetime agotado)
    # Posicion (dentro de ``block_order``) del bloque en que la fase se resolvio
    # (pass / fail / expired); en 'neither', ``len(block_order) - 1``. Campo
    # ADITIVO (plan ftmo-fases-secuencias): la fase siguiente arranca en
    # ``_next_start(end_pos)``. -1 = no informado (constructores externos).
    end_pos: int = -1


def evaluate_phase_path(
    block_order: np.ndarray,
    day_arrays: list[BlockDayArrays],
    durations: np.ndarray,
    gaps: np.ndarray | None = None,
    *,
    account_size: float,
    phase: PropFirmPhase,
) -> PhasePathResult:
    """Recorre los dias del path aplicando max loss estatico + daily loss
    dinamico (suelo desde balance de inicio de dia) + target + min_days.

    Post-target: el path CONTINUA a riesgo normal hasta cumplir min_days
    (cota conservadora, P3 de la revision).

    ``gaps`` (opcional): dias flat muestreados por transicion de bloque
    (alineado con ``block_order``). Sin el, el tiempo entre bloques queda
    comprimido (CRITICO-1 de la revision de codigo).
    """
    max_loss_line = account_size * (1.0 - phase.max_loss_pct / 100.0)
    target_line = account_size * (1.0 + phase.profit_target_pct / 100.0)
    daily_abs = phase.max_daily_loss_pct / 100.0 * account_size

    saldo = 0.0
    elapsed_prev = 0.0
    trading_days = 0
    target_hit = False
    terminal = account_size
    total_days = 0.0
    for pos, bi in enumerate(block_order):
        ba = day_arrays[bi]
        day0 = int(ba.day_ord[0]) if len(ba.day_ord) else 0
        for j in range(len(ba.day_ord)):
            elapsed = elapsed_prev + float(int(ba.day_ord[j]) - day0)
            day_start_eq = account_size + saldo + ba.rel_before[j]
            daily_floor = day_start_eq - daily_abs
            worst = account_size + saldo + ba.rel_float[j]
            closed = account_size + saldo + ba.rel_closed[j]
            terminal = closed
            total_days = elapsed + 1.0
            if worst <= max_loss_line or closed <= max_loss_line:
                return PhasePathResult("fail", total_days, trading_days, terminal,
                                       end_pos=pos)
            if worst < daily_floor or closed < daily_floor:
                return PhasePathResult("fail", total_days, trading_days, terminal,
                                       end_pos=pos)
            if ba.opens_day[j] > 0:
                trading_days += 1
            if not target_hit and closed >= target_line:
                target_hit = True
            if target_hit and trading_days >= phase.min_trading_days:
                return PhasePathResult("pass", total_days, trading_days, terminal,
                                       end_pos=pos)
        saldo += ba.total_pnl
        elapsed_prev += durations[bi] + (gaps[pos] if gaps is not None else 0.0)
    return PhasePathResult("neither", total_days, trading_days, terminal,
                           end_pos=len(block_order) - 1)


def evaluate_phase_path_trailing(
    block_order: np.ndarray,
    day_arrays: list[BlockDayArrays],
    durations: np.ndarray,
    gaps: np.ndarray | None = None,
    *,
    account_size: float,
    phase: PropFirmPhase,
) -> PhasePathResult:
    """Evaluación de fase con drawdown TRAILING intradía (plan Apex Fase 3).

    Espejo de :func:`evaluate_phase_path` (que queda congelada para FTMO) pero
    con el suelo trailing de ``phase.drawdown.amount_usd`` sobre el high-water
    mark de equity (flotante incluido), usando los escalares/arrays trailing
    del :class:`BlockDayArrays` (rellenados por
    ``block_day_arrays_portfolio(..., trailing=True)``).

    Estado: ``hwm`` ABSOLUTO (arranca en ``account_size``) y ``saldo`` a inicio
    de bloque; ``S = account_size + saldo`` es el ancla de cada bloque.

    Breach (liquidación instantánea), chequeado por DÍA con la condición de
    bloque de Fase 2 a granularidad diaria::

        max((hwm − S) − rel_low[d], rel_exc[d]) >= amount

    (agregada sobre los días del bloque ES la fórmula adjudicada
    ``max((hwm − S) − block_low, block_exc) >= amount``: ``block_low`` es el
    min de los ``rel_low`` y ``block_exc`` el max de los ``rel_exc``). Se
    evalúa por día y NO como pre-check de bloque porque un target alcanzado en
    un día anterior del mismo bloque debe ganar a un breach posterior (la
    evaluación termina al pasar). Esta forma SUBSUMA el check de cerrado
    ``closed <= max(hwm, S + rel_high[d]) − amount`` (demostrable: el último
    minuto del día tiene ``lower == realized == rel_closed[d]`` y
    ``cummax(upper) >= rel_high[d]``), así que un breach intradía con
    recuperación al cierre también se flaggea (cota conservadora de T1).

    Orden de checks por día (adjudicado): **breach → target → expiración** —
    el breach gana al target el mismo día; pasar en el último día del lifetime
    es válido (target se chequea antes que la expiración).

    ``lifetime_days`` (días NATURALES wall-clock vía ``durations`` + ``gaps``,
    como el ``elapsed`` de FTMO): si se agota sin target → outcome NUEVO
    ``"expired"`` (distinto de ``"neither"``: expirar es resolver con fee
    perdido; ``neither`` sigue siendo horizonte de backtest insuficiente).
    ``min_trading_days = 0`` (Apex) funciona: pasa el mismo día del target.

    ``hwm := max(hwm, S + block_high)`` al final de cada bloque (incluye el
    realized vía ``upper``) y ``saldo += total_pnl``.

    Fase trailing pero bloque sin escalares/arrays trailing (None) → raise
    ruidoso (nunca degradación silenciosa): precomputar con
    ``block_day_arrays_portfolio(..., trailing=True)``.
    """
    dd = phase.drawdown
    if dd is None:
        raise ValueError(
            "evaluate_phase_path_trailing exige phase.drawdown "
            "(para fases sin trailing usa evaluate_phase_path)"
        )
    amount = float(dd.amount_usd)
    target_line = account_size * (1.0 + phase.profit_target_pct / 100.0)
    lifetime = phase.lifetime_days

    hwm = float(account_size)
    saldo = 0.0
    elapsed_prev = 0.0
    trading_days = 0
    target_hit = False
    terminal = account_size
    total_days = 0.0
    for pos, bi in enumerate(block_order):
        ba = day_arrays[bi]
        if (
            ba.block_low is None or ba.block_high is None or ba.block_exc is None
            or ba.rel_low is None or ba.rel_high is None or ba.rel_exc is None
        ):
            raise ValueError(
                f"bloque {bi} sin datos trailing (block_low/block_high/"
                "block_exc/rel_low/rel_high/rel_exc a None): una fase trailing "
                "exige block_day_arrays_portfolio(..., trailing=True) — "
                "nunca degradación silenciosa"
            )
        s_block = account_size + saldo
        day0 = int(ba.day_ord[0]) if len(ba.day_ord) else 0
        for j in range(len(ba.day_ord)):
            elapsed = elapsed_prev + float(int(ba.day_ord[j]) - day0)
            closed = s_block + ba.rel_closed[j]
            terminal = closed
            total_days = elapsed + 1.0
            # 1) breach trailing (liquidación instantánea; gana al target).
            if max((hwm - s_block) - ba.rel_low[j], ba.rel_exc[j]) >= amount:
                return PhasePathResult("fail", total_days, trading_days, terminal,
                                       end_pos=pos)
            if ba.opens_day[j] > 0:
                trading_days += 1
            # 2) target sobre balance cerrado.
            if not target_hit and closed >= target_line:
                target_hit = True
            if target_hit and trading_days >= phase.min_trading_days:
                return PhasePathResult("pass", total_days, trading_days, terminal,
                                       end_pos=pos)
            # 3) expiración del lifetime (pasar en el último día es válido).
            if lifetime is not None and elapsed + 1.0 > lifetime:
                return PhasePathResult("expired", total_days, trading_days,
                                       terminal, expired=True, end_pos=pos)
        hwm = max(hwm, s_block + ba.block_high)
        saldo += ba.total_pnl
        elapsed_prev += durations[bi] + (gaps[pos] if gaps is not None else 0.0)
    return PhasePathResult("neither", total_days, trading_days, terminal,
                           end_pos=len(block_order) - 1)


@dataclass
class FundedPathResult:
    cycle_payouts: np.ndarray   # (max_cycles,) payout USD por ciclo (0 si muerto)
    days_to_first_payout: float # nan si ninguno
    alive_at_cap: bool
    # Plan onboarding-b1 §2.4 (aditivo): True si la fondeada NO murió y se acabó
    # la ventana de datos antes del cap de ciclos (``cycle < max_cycles``). Es
    # disjunto de ``alive_at_cap`` (que exige cycle >= max_cycles).
    cut_by_data: bool = False


def evaluate_funded_path(
    block_order: np.ndarray,
    day_arrays: list[BlockDayArrays],
    durations: np.ndarray,
    gaps: np.ndarray | None = None,
    *,
    account_size: float,
    phase: PropFirmPhase,          # barreras de la fondeada (daily + max loss)
    funded: FundedRules,
    profit_split: float,           # fraccion 0-1
) -> FundedPathResult:
    """Simula la cuenta fondeada: ciclos de payout, reset a baseline tras cada
    payout, corte por breach (daily/max loss) o por cap de ciclos.

    El profit NO retirado (ciclo en negativo) NO se resetea: la equity
    continua. Tras un payout la equity vuelve a ``account_size`` (el max loss
    estatico de FTMO se mantiene sobre el inicial).
    """
    max_cycles = funded.max_payout_cycles
    max_loss_line = account_size * (1.0 - phase.max_loss_pct / 100.0)
    daily_abs = phase.max_daily_loss_pct / 100.0 * account_size
    min_profit = funded.min_payout_profit_pct / 100.0 * account_size

    payouts = np.zeros(max_cycles)
    days_first = float("nan")
    saldo = 0.0
    elapsed_prev = 0.0
    cycle = 0
    alive_at_cap = False
    dead = False
    for pos, bi in enumerate(block_order):
        if dead or cycle >= max_cycles:
            break
        ba = day_arrays[bi]
        day0 = int(ba.day_ord[0]) if len(ba.day_ord) else 0
        for j in range(len(ba.day_ord)):
            if cycle >= max_cycles:
                break
            elapsed = elapsed_prev + float(int(ba.day_ord[j]) - day0)
            day_start_eq = account_size + saldo + ba.rel_before[j]
            daily_floor = day_start_eq - daily_abs
            worst = account_size + saldo + ba.rel_float[j]
            closed = account_size + saldo + ba.rel_closed[j]
            if worst <= max_loss_line or closed <= max_loss_line:
                dead = True
                break
            if worst < daily_floor or closed < daily_floor:
                dead = True
                break
            # ¿Cruce de fin de ciclo? (el dia elapsed+1 cierra el ciclo `cycle`)
            while cycle < max_cycles and (elapsed + 1.0) >= (cycle + 1) * funded.payout_cycle_days:
                profit = closed - account_size
                if profit > min_profit:
                    payouts[cycle] = profit * profit_split
                    saldo -= profit  # reset a baseline
                    closed = account_size
                    if np.isnan(days_first):
                        days_first = elapsed + 1.0
                cycle += 1
        # saldo acumula el PnL del bloque; si hubo resets mid-bloque, saldo ya
        # lleva restados los profits retirados, asi que sumar total_pnl cuadra.
        if not dead:
            saldo += ba.total_pnl
        elapsed_prev += durations[bi] + (gaps[pos] if gaps is not None else 0.0)
    if not dead and cycle >= max_cycles:
        alive_at_cap = True
    cut_by_data = (not dead) and cycle < max_cycles
    return FundedPathResult(payouts, days_first, alive_at_cap, cut_by_data)


# ---------------------------------------------------------------------------
# Nivel 1: barrido de la firm completa sobre paths compartidos
# ---------------------------------------------------------------------------

@dataclass
class ContractCapDiag:
    """Diagnóstico de caps de contratos simultáneos (plan Apex Fase 5).

    Unidades: MICRO-EQUIVALENTES (1 mini = 10 micros). ``path_max[k]`` es el
    máximo de micro-equivalentes simultáneos del path k (suma cross-instrumento
    ponderada por la tabla de equivalencias; nan en paths fuera de ``mask``).
    ``frac_violating`` = fracción de paths evaluados con ``path_max >
    cap_micros`` (violación ESTRICTA: el cap es inclusivo). ``scope`` =
    "evaluation" (cap fijo de la fase, p.ej. 120 micro) | "funded" (cap del
    tier de la PA — v1: cota conservadora con el tier MÍNIMO, ver
    :func:`contract_cap_diagnostic`).
    """

    scope: str
    cap_micros: float
    n_paths: int                 # paths evaluados (tras mask)
    frac_violating: float
    max_observed: float          # nan si no se evaluó ningún path
    path_max: np.ndarray         # (N,) micro-eq simultáneos máximos por path


@dataclass
class FirmPathOutcomes:
    """Outcomes por path (mismo indice k en los tres escenarios)."""

    pass1: np.ndarray               # (N,) bool
    pass2: np.ndarray               # (N,) bool (True si no hay fase 2)
    neither1: np.ndarray            # (N,) bool — F1 ni pass ni fail (horizonte finito)
    neither2: np.ndarray            # (N,) bool — F2 neither (solo si pass1)
    funded_payouts: np.ndarray      # (N, max_cycles) USD
    days_first_payout: np.ndarray   # (N,) nan si ninguno
    alive_at_cap: np.ndarray        # (N,) bool
    days_to_pass1: np.ndarray       # (N,) nan si no pass
    expired1: np.ndarray | None = None  # (N,) bool — F1 expirada (lifetime; trailing)
    expired2: np.ndarray | None = None  # (N,) bool — F2 expirada (solo si pass1)
    # Diagnósticos ladder (plan Apex Fase 4; None en el camino cycle/FTMO):
    dll_hits: np.ndarray | None = None             # (N,) int
    consistency_deferrals: np.ndarray | None = None  # (N,) int
    inactivity_deaths: np.ndarray | None = None    # (N,) bool
    closed_after_ladder: np.ndarray | None = None  # (N,) bool
    # Diagnósticos de caps de contratos (plan Apex Fase 5; None = no evaluado,
    # FTMO intacto). Se calculan FUERA de simulate_challenge/sweep_funded (ver
    # :func:`contract_cap_diagnostic`) y se acoplan aquí.
    contract_cap_eval: ContractCapDiag | None = None  # vs cap de la evaluación
    contract_cap_pa: ContractCapDiag | None = None    # vs cap del tier de la PA
    days_eval1: np.ndarray | None = None  # propagado de ChallengeOutcomes
        # (FTMO Futures; None en caminos legacy — misma semántica que arriba)
    funded_start: np.ndarray | None = None  # (N,) int — bloque de la cinta en
        # que arranca la fondeada (propagado de ChallengeOutcomes; plan
        # ftmo-fases-secuencias). None si los outcomes no vienen de la cinta.
    funded_windows: np.ndarray | None = None  # (N, n_blocks) int32 — ventanas de
        # la fondeada = ``tape.windows(funded_start)`` (la API calcula con ellas el
        # cap de la fondeada). Lo rellena simulate_firm_paths; None en el resto.
    payout_deferrals: np.ndarray | None = None  # propagado de FundedOutcomes
        # (FTMO Futures; None en caminos legacy)
    payout_overflows: np.ndarray | None = None  # propagado de FundedOutcomes
        # (FTMO Futures; None en caminos legacy)
    days_funded: np.ndarray | None = None  # (N,) días naturales de la funded
        # hasta muerte/cap/fin de backtest (FTMO Futures, desde
        # FuturesFundedResult.days); nan en no-passers; None en legacy.
        # Denominador real del EV/día de sequence_ev_futures (revisión
        # cruzada GLM-5.3 2026-09-30: sustituye el fcap fijo del plan).
    cut_by_data: np.ndarray | None = None  # (N,) bool — fondeada viva cuando se
        # acabó la ventana de datos, sin llegar al cap (plan onboarding-b1 §2.4;
        # solo modelo cycle; None en ladder/futures y en caminos legacy).

    @property
    def both(self) -> np.ndarray:
        """Pass F1 y pass F2 (el path llega a fondeada)."""
        return self.pass1 & self.pass2

    @property
    def expired(self) -> np.ndarray:
        """Paths expirados por lifetime (trailing). RESUELTOS con EV = −fee:
        no son ``unresolved`` (eso es solo neither = horizonte de backtest)."""
        n = len(self.pass1)
        e1 = self.expired1 if self.expired1 is not None else np.zeros(n, dtype=bool)
        e2 = self.expired2 if self.expired2 is not None else np.zeros(n, dtype=bool)
        return e1 | e2

    @property
    def unresolved(self) -> np.ndarray:
        """Paths sin resolver (neither en alguna fase) — excluidos del EV."""
        return self.neither1 | self.neither2

    @property
    def resolved(self) -> np.ndarray:
        return ~self.unresolved


@dataclass
class ChallengeOutcomes:
    """Resultado de F1+F2 a un ``f_challenge``, sobre los paths compartidos."""

    pass1: np.ndarray               # (N,) bool
    pass2: np.ndarray               # (N,) bool (True si no hay fase 2)
    neither1: np.ndarray            # (N,) bool
    neither2: np.ndarray            # (N,) bool
    days_to_pass1: np.ndarray       # (N,) nan si no pass
    expired1: np.ndarray | None = None  # (N,) bool — F1 expirada (trailing)
    expired2: np.ndarray | None = None  # (N,) bool — F2 expirada (solo si pass1)
    contract_cap: ContractCapDiag | None = None  # diagnóstico cap evaluación (Fase 5)
    days_eval1: np.ndarray | None = None  # (N,) días calendario hasta la
        # resolución de F1 (pass/fail) o hasta el fin del backtest (neither);
        # relleno en TODOS los outcomes (FTMO Futures, plan v2 #4). None en
        # perfiles legacy; los consumidores nuevos lo exigen con raise ruidoso.
    funded_start: np.ndarray | None = None  # (N,) int — primer bloque de la
        # cinta de la fondeada de cada path (plan ftmo-fases-secuencias):
        # 2 fases: s2 + _next_start(r2.end_pos); 1 fase: _next_start(r1.end_pos).
        # Definido para TODOS los paths (la fondeada solo se evalua en passers).

    @property
    def both(self) -> np.ndarray:
        return self.pass1 & self.pass2

    @property
    def expired(self) -> np.ndarray:
        """Paths expirados por lifetime (trailing) — resueltos con EV = −fee."""
        n = len(self.pass1)
        e1 = self.expired1 if self.expired1 is not None else np.zeros(n, dtype=bool)
        e2 = self.expired2 if self.expired2 is not None else np.zeros(n, dtype=bool)
        return e1 | e2

    @property
    def unresolved(self) -> np.ndarray:
        return self.neither1 | self.neither2

    @property
    def resolved(self) -> np.ndarray:
        return ~self.unresolved


@dataclass
class FundedOutcomes:
    """Fondeada a un ``f_funded``, evaluada solo en los passers de un ``f_c``.

    Diagnósticos del modelo ``ladder`` (plan Apex Fase 4): arrays por path,
    None en el camino ``cycle`` (FTMO no paga memoria ni cambia).
    """

    funded_payouts: np.ndarray      # (N, max_cycles) USD; no-passers a 0
    days_first_payout: np.ndarray   # (N,) nan si ninguno
    alive_at_cap: np.ndarray        # (N,) bool
    dll_hits: np.ndarray | None = None             # (N,) int — hits de DLL (ladder)
    consistency_deferrals: np.ndarray | None = None  # (N,) int — denegaciones 50%
    inactivity_deaths: np.ndarray | None = None    # (N,) bool — muerte por inactividad
    closed_after_ladder: np.ndarray | None = None  # (N,) bool — cierre feliz tras el último retiro
    contract_cap: ContractCapDiag | None = None  # diagnóstico cap PA (Fase 5; tier mínimo)
    payout_deferrals: np.ndarray | None = None   # (N,) int — días elegibles con
                                                 # request < min (FTMO Futures)
    payout_overflows: np.ndarray | None = None   # (N,) int — requests beyond
                                                 # max_payout_cycles (truncado)
    days_funded: np.ndarray | None = None        # (N,) días naturales de la
                                                 # funded hasta muerte/cap/fin
                                                 # (FTMO Futures; nan en
                                                 # no-passers; None en legacy)
    cut_by_data: np.ndarray | None = None        # (N,) bool — fondeada viva al
                                                 # acabarse los datos, sin cap
                                                 # (cycle; None en ladder/futures)


def combine_challenge_funded(ch: ChallengeOutcomes, fu: FundedOutcomes) -> FirmPathOutcomes:
    """Ensambla el ``FirmPathOutcomes`` que consume :func:`sequence_ev`.

    Puente del refactor (PLAN_RECOMMEND_F_PAIR §2a): la maquinaria de Nivel 2
    (``sequence_ev``, ``firm_report``) no cambia de firma; solo cambia como se
    producen los outcomes (dos pasadas con ``f`` distinto por fase, o una).
    """
    return FirmPathOutcomes(
        pass1=ch.pass1, pass2=ch.pass2, neither1=ch.neither1,
        neither2=ch.neither2, funded_payouts=fu.funded_payouts,
        days_first_payout=fu.days_first_payout, alive_at_cap=fu.alive_at_cap,
        days_to_pass1=ch.days_to_pass1,
        expired1=ch.expired1, expired2=ch.expired2,
        dll_hits=fu.dll_hits, consistency_deferrals=fu.consistency_deferrals,
        inactivity_deaths=fu.inactivity_deaths,
        closed_after_ladder=fu.closed_after_ladder,
        contract_cap_eval=ch.contract_cap, contract_cap_pa=fu.contract_cap,
        days_eval1=ch.days_eval1,
        funded_start=ch.funded_start,
        payout_deferrals=fu.payout_deferrals,
        payout_overflows=fu.payout_overflows,
        days_funded=fu.days_funded,
        cut_by_data=fu.cut_by_data,
    )


@dataclass(frozen=True)
class PhaseTape:
    """Fila prolongada de cada path: F1 = [0, n); la F2 y la fondeada son ventanas de n bloques que empiezan donde se
    resolvió la fase anterior. ``idx[:, :n]`` es ``idx_matrix`` BIT A BIT; ``gaps[:, :n]`` son los gaps de la F1 de hoy.

    Cotas: ``end_pos <= n - 1`` => la F2 empieza en ``s2 <= n``; la fondeada, en ``s_f <= 2n`` (2 fases) o ``<= n``
    (1 fase); la ventana termina en ``s_f + n <= 3n``. Cada fase conserva su ventana de n bloques (mismo horizonte).

    ``window()`` devuelve VISTAS de la cinta (sin copia): los evaluadores no deben mutarlas.
    """

    idx: np.ndarray            # (n_paths, 3n) int32
    gaps: np.ndarray | None    # (n_paths, 3n), o None si no hay gaps empíricos
    n_blocks: int              # n

    def window(self, p: int, start: int) -> tuple[np.ndarray, np.ndarray | None]:
        """Ventana de n bloques del path ``p`` que arranca en ``start``: (orden de bloques, gaps o None)."""
        n = self.n_blocks
        start = int(start)
        if start < 0 or start + n > self.idx.shape[1]:
            raise ValueError(
                f"ventana fuera de la cinta: start={start}, n={n}, ancho={self.idx.shape[1]}")
        g = self.gaps[p, start:start + n] if self.gaps is not None else None
        return self.idx[p, start:start + n], g

    def windows(self, starts: np.ndarray) -> np.ndarray:
        """Ventanas de n bloques de TODOS los paths, ``(n_paths, n)``, que arrancan en ``starts[p]``.
        Para el diagnóstico de caps de la fondeada (``contract_cap_diagnostic``)."""
        starts = np.asarray(starts, dtype=np.int64)
        n = self.n_blocks
        cols = starts[:, None] + np.arange(n, dtype=np.int64)[None, :]
        return np.take_along_axis(self.idx, cols, axis=1)


def build_phase_tape(
    idx_matrix: np.ndarray,
    idx_ext: np.ndarray,
    gaps_f1: np.ndarray | None,
    gaps_ext: np.ndarray | None,
) -> PhaseTape:
    """Concatena ``[idx_matrix | idx_ext]`` y ``[gaps_f1 | gaps_ext]`` y valida formas.

    ``idx_ext``: ``(n_paths, 2n)`` (:func:`src.risk.mc.extend_stationary_rows`). ``gaps_f1``/``gaps_ext``: ambos None
    (sin gaps empíricos) o ambos dados, ``(n_paths, n)`` y ``(n_paths, 2n)``. Índices en int32 (ahorro de RAM).
    """
    idx_matrix = np.asarray(idx_matrix)
    if idx_matrix.ndim != 2:
        raise ValueError("idx_matrix debe ser (n_paths, n_blocks)")
    n_paths, n = idx_matrix.shape
    if idx_ext is None or np.asarray(idx_ext).shape != (n_paths, 2 * n):
        raise ValueError(
            f"idx_ext debe ser (n_paths, 2*n_blocks) = ({n_paths}, {2 * n}) "
            "(extend_stationary_rows(idx_matrix, n_trades, bs, 2*n_blocks, rng))")
    if (gaps_f1 is None) != (gaps_ext is None):
        raise ValueError("gaps_f1 y gaps_ext: o ambos None o ambos dados")
    idx = np.empty((n_paths, 3 * n), dtype=np.int32)
    idx[:, :n] = idx_matrix
    idx[:, n:] = idx_ext
    gaps = None
    if gaps_f1 is not None:
        gaps_f1 = np.asarray(gaps_f1)
        gaps_ext = np.asarray(gaps_ext)
        if gaps_f1.shape != (n_paths, n) or gaps_ext.shape != (n_paths, 2 * n):
            raise ValueError(
                f"gaps_f1 debe ser ({n_paths}, {n}) y gaps_ext ({n_paths}, {2 * n})")
        gaps = np.empty((n_paths, 3 * n), dtype=np.result_type(gaps_f1, gaps_ext))
        gaps[:, :n] = gaps_f1
        gaps[:, n:] = gaps_ext
    return PhaseTape(idx=idx, gaps=gaps, n_blocks=n)


def _draw_gap_tape(
    rng: np.random.Generator,
    gap_days: np.ndarray,
    n_paths: int,
    n_blocks: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Gaps de la cinta. Helper ÚNICO (1D y par 2D). Orden de consumo del ``rng`` (contrato): PRIMERO la llamada de
    siempre ``rng.choice(gap_days, size=(n_paths, n_blocks), replace=True)`` (los gaps de la F1, idénticos a los de
    antes del cambio) y DESPUÉS la prolongación ``(n_paths, 2 * n_blocks)`` para la F2 y la fondeada."""
    g1 = rng.choice(gap_days, size=(n_paths, n_blocks), replace=True)
    g_ext = rng.choice(gap_days, size=(n_paths, 2 * n_blocks), replace=True)
    return g1, g_ext


def _next_start(end_pos: int) -> int:
    """Primer bloque de la fase siguiente: el siguiente al de resolución. ÚNICO punto que define la continuación
    (parchearlo a ``return 0`` reproduce el comportamiento anterior: F2 y fondeada sobre la fila de la F1)."""
    return end_pos + 1


def simulate_challenge(
    tape: PhaseTape,
    day_arrays: list[BlockDayArrays],
    durations: np.ndarray,
    cfg: PropFirmConfig,
    *,
    portfolio: bool = False,
) -> ChallengeOutcomes:
    """Evalua F1 y F2 a un ``f_challenge`` sobre la cinta de cada path (CONTINUACION de la secuencia).

    ``tape``: :class:`PhaseTape` (``build_phase_tape``), muestreada UNA vez
    fuera (CRN global para toda la grilla — PLAN_RECOMMEND_F_PAIR §2b); sus
    gaps (None = sin gaps) evitan que el tiempo entre bloques quede comprimido
    (CRITICO-1 de la revision de codigo). La F1 se evalua sobre la ventana 0
    (``idx_matrix[p]`` y ``gaps_f1[p]`` de siempre, bit a bit); la F2, sobre la
    ventana que arranca en ``_next_start(r1.end_pos)`` — el bloque siguiente al
    de resolucion de la F1 —, para TODOS los paths (los arrays quedan completos;
    ``neither2``/``expired2`` solo cuentan en passers de la F1). Si no es un
    ``PhaseTape`` -> ``ValueError`` (reutilizar la fila de la F1 hacia
    P(F2|F1)=1). ``ChallengeOutcomes.funded_start`` da el bloque de arranque de
    la fondeada de cada path.

    Dispatch (plan Apex Fase 3): una fase con ``drawdown`` trailing se evalúa
    con :func:`evaluate_phase_path_trailing`; sin trailing, el camino FTMO es
    byte-idéntico. El camino trailing es SOLO-portfolio (``portfolio=True`` y
    ``day_arrays`` de ``precompute_day_arrays_portfolio(..., trailing=True)``;
    el trailing exige alineación por minuto del calendario conjunto): una fase
    trailing por el camino mono (``portfolio=False``) hace raise ruidoso — el
    mono es una cartera de 1 pata (paridad ya testeada).

    Dispatch FTMO Futures (plan fase 2): ``drawdown.type == "eod_trailing"`` →
    :func:`src.risk.propfirm_futures.evaluate_phase_path_eod_trail` (import
    lazy, mismo anti-circularidad que ``sweep_funded`` con ``propfirm_ladder``).
    ``breach_on: "balance"`` admite el camino mono; ``"equity"`` exige
    portfolio (misma regla que el trailing Apex).

    ``days_eval1`` (plan v2 #4): campo ADITIVO relleno SIEMPRE con los días
    calendario de la evaluación de F1 — días a la resolución (pass/fail/
    expired) o duración completa del backtest en ``neither`` (los intentos
    neither consumen reloj de campaña en la economía de suscripción). Los
    perfiles legacy lo reciben a None en constructores externos.
    """
    if not isinstance(tape, PhaseTape):
        raise ValueError(
            "simulate_challenge recibe un PhaseTape (build_phase_tape): reutilizar la "
            "fila de la F1 para la F2 hace P(F2|F1)=1")
    n_paths = tape.idx.shape[0]
    acc = cfg.account_size
    phase1 = cfg.phases[0]
    phase2 = cfg.phases[1] if len(cfg.phases) > 1 else None

    def _evaluator(phase: PropFirmPhase):
        if phase.drawdown is None:
            return evaluate_phase_path
        dd = phase.drawdown
        # eod_trailing con breach sobre BALANCE (cierre EOD) es evaluable por
        # el camino mono (no exige rel_low); equity intradía y el trailing
        # Apex exigen la infra por minuto del camino portfolio (FTMO Futures
        # solo usa equity → mismo raise ruidoso que el trailing Apex).
        needs_portfolio = dd.type != "eod_trailing" or dd.breach_on == "equity"
        if needs_portfolio and not portfolio:
            raise ValueError(
                f"{cfg.key}/{phase.name}: las fases con drawdown trailing "
                "(eod_trailing breach_on 'equity') solo se evalúan por el "
                "camino portfolio (portfolio=True con day_arrays de "
                "precompute_day_arrays_portfolio(..., trailing=True)) — "
                "el mono es una cartera de 1 pata"
            )
        if dd.type == "eod_trailing":
            from src.risk.propfirm_futures import evaluate_phase_path_eod_trail
            return evaluate_phase_path_eod_trail
        return evaluate_phase_path_trailing

    eval1 = _evaluator(phase1)
    eval2 = _evaluator(phase2) if phase2 is not None else None

    pass1 = np.zeros(n_paths, dtype=bool)
    neither1 = np.zeros(n_paths, dtype=bool)
    neither2 = np.zeros(n_paths, dtype=bool)
    expired1 = np.zeros(n_paths, dtype=bool)
    expired2 = np.zeros(n_paths, dtype=bool)
    pass2 = np.ones(n_paths, dtype=bool)
    days_pass1 = np.full(n_paths, np.nan)
    days_eval1 = np.full(n_paths, np.nan)
    funded_start = np.zeros(n_paths, dtype=np.int64)

    for p in range(n_paths):
        order, gaps_p = tape.window(p, 0)
        r1 = eval1(order, day_arrays, durations, gaps_p,
                   account_size=acc, phase=phase1)
        pass1[p] = r1.outcome == "pass"
        neither1[p] = r1.outcome == "neither"
        expired1[p] = r1.outcome == "expired"
        days_eval1[p] = r1.days   # pass/fail: días a resolución; neither: fin
                                  # del backtest (entra al reloj de campaña, v2 #4)
        if pass1[p]:
            days_pass1[p] = r1.days
        if phase2 is not None:
            s2 = _next_start(r1.end_pos)
            order2, gaps2 = tape.window(p, s2)
            r2 = eval2(order2, day_arrays, durations, gaps2,
                       account_size=acc, phase=phase2)
            pass2[p] = r2.outcome == "pass"
            neither2[p] = pass1[p] and r2.outcome == "neither"
            expired2[p] = pass1[p] and r2.outcome == "expired"
            funded_start[p] = s2 + _next_start(r2.end_pos)
        else:
            funded_start[p] = _next_start(r1.end_pos)
    return ChallengeOutcomes(pass1, pass2, neither1, neither2, days_pass1,
                             expired1, expired2, days_eval1=days_eval1,
                             funded_start=funded_start)


def sweep_funded(
    *,
    tape: PhaseTape,
    funded_start: np.ndarray,
    day_arrays_f_grid: list[list[BlockDayArrays]],
    passers_mask: np.ndarray,
    durations: np.ndarray,
    cfg: PropFirmConfig,
    f_grid: np.ndarray,
) -> dict[float, FundedOutcomes]:
    """Barre la fondeada sobre la grilla ``f_grid``, solo en paths passers.

    SOLO palabras clave (cualquier llamada posicional antigua falla con
    ``TypeError``: no puede dar en silencio el numero viejo). Cada passer ``p``
    se evalua sobre ``tape.window(p, funded_start[p])`` — la CONTINUACION de su
    fila tras la evaluacion, no la fila con la que paso (plan
    ftmo-fases-secuencias); ``funded_start`` sale de
    ``ChallengeOutcomes.funded_start``.

    Una llamada por ``f_c`` (PLAN_RECOMMEND_F_PAIR §2c): evalua cada ``f_f``
    sobre la misma cinta (CRN) y devuelve ``{f_f: FundedOutcomes}``.
    Los no-passers quedan a 0 (su outcome no se lee — misma convencion lazy
    que el codigo original; no rompe el muestreo conjunto).

    Dispatch por ``cfg.funded.model`` (plan Apex Fase 4): ``"cycle"`` →
    :func:`evaluate_funded_path` con las barreras de ``phase1`` (camino FTMO,
    byte-idéntico); ``"ladder"`` →
    :func:`src.risk.propfirm_ladder.evaluate_funded_path_ladder` (import lazy —
    evita la circularidad propfirm↔ladder), que recibe SOLO ``cfg.funded`` y
    NUNCA hereda barreras de ``phase1``; ``"futures"`` (plan FTMO Futures
    Fase 3) → :func:`src.risk.propfirm_futures.evaluate_funded_path_futures`
    (import lazy, misma anti-circularidad), con el horizonte
    ``subscription.funded_cap_days`` y la política de retirada default del
    YAML (``payouts_futures.withdrawal_policy``; el batch puede correr ambas
    llamando por separado).
    """
    acc = cfg.account_size
    phase1 = cfg.phases[0]
    if not isinstance(tape, PhaseTape):
        raise ValueError("sweep_funded recibe un PhaseTape (build_phase_tape)")
    n_paths = tape.idx.shape[0]
    passers = np.flatnonzero(passers_mask)
    ladder_mode = cfg.funded.model == "ladder"
    futures_mode = cfg.funded.model == "futures"
    if ladder_mode:
        from src.risk.propfirm_ladder import evaluate_funded_path_ladder
    if futures_mode:
        from src.risk.propfirm_futures import evaluate_funded_path_futures
        horizon = (cfg.subscription.funded_cap_days
                   if cfg.subscription is not None else 365)
        wd_policy = cfg.funded.payouts_futures.withdrawal_policy
    out: dict[float, FundedOutcomes] = {}
    for fi, day_arrays_f in enumerate(day_arrays_f_grid):
        funded = np.zeros((n_paths, cfg.funded.max_payout_cycles))
        days_first = np.full(n_paths, np.nan)
        alive = np.zeros(n_paths, dtype=bool)
        cut_data = (np.zeros(n_paths, dtype=bool)
                    if not (ladder_mode or futures_mode) else None)
        dll_hits = np.zeros(n_paths, dtype=int) if (ladder_mode or futures_mode) else None
        deferrals = np.zeros(n_paths, dtype=int) if ladder_mode else None
        inact_deaths = np.zeros(n_paths, dtype=bool) if ladder_mode else None
        closed_ladder = np.zeros(n_paths, dtype=bool) if ladder_mode else None
        pay_def = np.zeros(n_paths, dtype=int) if futures_mode else None
        pay_ovf = np.zeros(n_paths, dtype=int) if futures_mode else None
        days_funded = np.full(n_paths, np.nan) if futures_mode else None
        for p in passers:
            order, gaps_p = tape.window(p, funded_start[p])
            if ladder_mode:
                rf = evaluate_funded_path_ladder(
                    order, day_arrays_f, durations, gaps_p,
                    cfg_funded=cfg.funded, account_size=acc,
                    profit_split=cfg.profit_split_pct / 100.0)
                funded[p] = rf.payouts
                days_first[p] = rf.days_to_first_payout
                alive[p] = rf.alive_at_cap
                dll_hits[p] = rf.dll_hits
                deferrals[p] = rf.consistency_deferrals
                inact_deaths[p] = rf.inactivity_death
                closed_ladder[p] = rf.closed_after_ladder
            elif futures_mode:
                rf = evaluate_funded_path_futures(
                    order, day_arrays_f, durations, gaps_p,
                    cfg_funded=cfg.funded, account_size=acc,
                    profit_split=cfg.profit_split_pct / 100.0,
                    horizon_days=horizon, withdrawal_policy=wd_policy)
                funded[p] = rf.payouts
                days_first[p] = rf.days_to_first_payout
                alive[p] = rf.alive_at_cap
                dll_hits[p] = rf.dll_hits
                pay_def[p] = rf.payout_deferrals
                pay_ovf[p] = rf.payout_overflows
                days_funded[p] = rf.days
            else:
                rf = evaluate_funded_path(order, day_arrays_f, durations, gaps_p,
                                          account_size=acc, phase=phase1,
                                          funded=cfg.funded,
                                          profit_split=cfg.profit_split_pct / 100.0)
                funded[p] = rf.cycle_payouts
                days_first[p] = rf.days_to_first_payout
                alive[p] = rf.alive_at_cap
                cut_data[p] = rf.cut_by_data
        out[float(f_grid[fi])] = FundedOutcomes(
            funded, days_first, alive,
            dll_hits=dll_hits, consistency_deferrals=deferrals,
            inactivity_deaths=inact_deaths, closed_after_ladder=closed_ladder,
            payout_deferrals=pay_def, payout_overflows=pay_ovf,
            days_funded=days_funded, cut_by_data=cut_data)
    return out


# ---------------------------------------------------------------------------
# Diagnóstico de caps de contratos simultáneos (plan Apex Fase 5)
# ---------------------------------------------------------------------------
#
# Vive en este módulo (y no en propfirm_ladder) porque consume datos de
# Nivel 1 — el calendario open/close por trade de los ``PortfolioDayBlock`` y
# los lotes por f (``lots_arr`` de ``src.risk.mm.simulate_f``, en forma de
# ``lots_by_sym``) — y NO el estado del ladder. Consecuencia: el cap de la PA
# se evalúa contra el tier MÍNIMO (cota conservadora); el tier vigente cada
# día es estado interno del ladder y reportar contra él exigiría rehecer la
# máquina de estados (descartado en v1 — ``max_observed`` queda reportado para
# comparar a mano contra tiers altos).
#
# Unidades: MICRO-EQUIVALENTES (1 mini = 10 micros). ``lots_arr`` está en
# "lotes del símbolo" (convención del broker); la tabla
# ``micro_equiv_by_sym`` (micro-eq por 1.0 lote de cada pata) convierte a la
# unidad común. Si todo el universo son micros, la tabla es 1.0 (default) y la
# suma es directa — LIMITACIÓN v1: con instrumentos de tick muy distinto el
# "micro-equivalente" es solo una convención de conteo de contratos, no de
# riesgo económico.
#
# Reducción path = max sobre sus bloques: los bloques son episodios flat-flat
# de la cartera y en un path se concatenan SECUENCIALMENTE (con gaps), así que
# no hay solapamiento cross-bloque y el máximo del path es el máximo de los
# máximos de bloque. Esto hace el diagnóstico O(n_bloques) por f, no por path.

def block_max_simultaneous_micros(
    b: PortfolioDayBlock,
    lots: np.ndarray,
    micro_equiv: np.ndarray | float = 1.0,
) -> float:
    """Máximo de contratos simultáneos (micro-equivalentes) de UN bloque.

    ``lots``: (k,) lotes de cada trade del bloque a un ``f`` concreto (ya
    re-floor/clamp por ``mm.simulate_f``). ``micro_equiv``: escalar o (k,)
    micro-equivalentes por 1.0 lote por trade.

    Granularidad de TIMESTAMP (``open_t``/``close_t`` del bloque, Fase 2) —
    dos trades el mismo día(-sesión) pero en horas disjuntas NO solapan.
    Convenio de borde (el del trailing, Fase 2): la posición está viva en
    ``[ot, ct)``; un trade que cierra en el mismo instante en que otro abre NO
    es simultáneo con él (los cierres se procesan antes en empates).
    """
    if b.open_t is None or b.close_t is None:
        raise ValueError(
            "block_max_simultaneous_micros exige open_t/close_t en el bloque "
            "(camino portfolio: build_day_blocks_portfolio; el mono es una "
            "cartera de 1 pata)"
        )
    w = np.asarray(lots, dtype=float) * np.asarray(micro_equiv, dtype=float)
    if w.shape != (b.n_trades,):
        raise ValueError(
            f"lots/micro_equiv deben tener shape ({b.n_trades},) "
            f"(hay {w.shape})"
        )
    if b.n_trades == 0:
        return 0.0
    # Eventos: cierres (−w) primero que aperturas (+w) en empate de timestamp
    # (concatenación en ese orden + argsort estable = convenio [ot, ct)).
    times = np.concatenate([b.close_t, b.open_t]).astype("datetime64[ns]")
    deltas = np.concatenate([-w, w])
    order = np.argsort(times, kind="mergesort")
    run = np.cumsum(deltas[order])
    return float(run.max())


def contract_cap_diagnostic(
    idx_matrix: np.ndarray,
    blocks: list[PortfolioDayBlock],
    lots_by_sym: dict[int, np.ndarray],
    f_idx: int,
    *,
    cap_micros: float,
    micro_equiv_by_sym: dict[int, float] | None = None,
    scope: str = "evaluation",
    mask: np.ndarray | None = None,
) -> ContractCapDiag:
    """Fracción de paths que violan el cap de contratos + máximo observado.

    ``idx_matrix``: (n_paths, n_blocks) órdenes de bloques: para scope
    "evaluation" la fila de la F1 (``idx_matrix``); para scope "funded", las
    ventanas de la fondeada (``PhaseTape.windows(funded_start)``). ``blocks``: los ``PortfolioDayBlock`` del universo.
    ``lots_by_sym``: ``{sym_id: lots_arr (n_f, n_trades)}`` de
    ``mm.simulate_f`` por pata; ``f_idx`` el ``f`` evaluado (en modo par,
    ``f_c`` para scope "evaluation" y ``f_f`` para "funded").
    ``cap_micros``: cap en micro-equivalentes (evaluación:
    ``eval_contract_cap_micros(cfg)``; PA: ``pa_min_tier_cap_micros(cfg)``).
    ``micro_equiv_by_sym``: micro-eq por 1.0 lote de cada pata (None = 1.0,
    todo micros, suma directa — ver la nota de unidades del módulo).
    ``mask``: bool (n_paths,) para evaluar un subconjunto (p.ej. solo passers
    para la PA); la fracción y el máximo se calculan sobre los paths enmascarados.

    Política de invalidación: la marca ``sizing_invalid`` NO se decide aquí
    sino en :func:`sequence_ev` (umbral ``cap_violation_tol``), donde el
    diagnóstico ya está acoplado a los outcomes.
    """
    if cap_micros <= 0:
        raise ValueError("cap_micros debe ser > 0")
    if scope not in ("evaluation", "funded"):
        raise ValueError(f"scope '{scope}' no soportado (evaluation|funded)")
    equiv = micro_equiv_by_sym or {}
    block_max = np.array([
        block_max_simultaneous_micros(
            b,
            np.array([
                lots_by_sym[int(b.sym_id[j])][f_idx, int(b.trade_local_id[j])]
                for j in range(b.n_trades)
            ]),
            np.array([
                float(equiv.get(int(b.sym_id[j]), 1.0)) for j in range(b.n_trades)
            ]),
        )
        for b in blocks
    ])
    n_paths = idx_matrix.shape[0]
    if mask is None:
        mask = np.ones(n_paths, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (n_paths,):
            raise ValueError(f"mask debe tener shape ({n_paths},)")
    # path max = max de los máximos de bloque (bloques flat-flat secuenciales).
    path_max_all = block_max[idx_matrix].max(axis=1)
    path_max = np.where(mask, path_max_all, np.nan)
    n_eval = int(mask.sum())
    vals = path_max_all[mask]
    frac = float((vals > cap_micros).mean()) if n_eval else float("nan")
    return ContractCapDiag(
        scope=scope,
        cap_micros=float(cap_micros),
        n_paths=n_eval,
        frac_violating=frac,
        max_observed=float(vals.max()) if n_eval else float("nan"),
        path_max=path_max,
    )


def eval_contract_cap_micros(cfg: PropFirmConfig) -> float | None:
    """Cap de contratos de la EVALUACIÓN en micro-equivalentes.

    Mínimo entre las fases que lo declaran (conservador si F1 y F2 declaran
    caps distintos). None si el perfil no declara ``max_contracts`` (FTMO).
    """
    caps = [ph.max_contracts.max_micro for ph in cfg.phases if ph.max_contracts]
    return float(min(caps)) if caps else None


def pa_min_tier_cap_micros(cfg: PropFirmConfig) -> float | None:
    """Cota CONSERVADORA del cap de la PA: el mínimo ``max_micro`` de los
    scaling tiers (Apex: tier 1 = 40). None si el perfil no tiene tiers.

    Limitación v1 (documentada en la cabecera de esta sección): el tier
    vigente cada día depende del estado del ladder (beneficio acumulado post
    retiros/clamps) y no es accesible sin rehecerlo; comparar contra el tier
    mínimo sobre-flaggea paths que en tiers altos serían legales —
    ``max_observed`` permite la comparación manual contra tiers superiores.
    """
    tiers = cfg.funded.scaling_tiers
    return float(min(t.max_micro for t in tiers)) if tiers else None


def simulate_firm_paths(
    idx_matrix: np.ndarray,
    day_arrays: list[BlockDayArrays],
    durations: np.ndarray,
    cfg: PropFirmConfig,
    gap_days: np.ndarray | None = None,
    seed: int = 0,
    gaps_fixed: np.ndarray | None = None,
    portfolio: bool = False,
    *,
    idx_ext: np.ndarray | None = None,
    gaps_ext: np.ndarray | None = None,
) -> FirmPathOutcomes:
    """Evalua F1, F2 y fondeada sobre los MISMOS paths (muestreo conjunto), en CONTINUACION de la secuencia.

    La F1 va sobre ``idx_matrix[p]`` (la fila de siempre); la F2 y la fondeada,
    sobre ventanas posteriores de la MISMA fila prolongada (``idx_ext``,
    :func:`src.risk.mc.extend_stationary_rows`; ver :class:`PhaseTape`). Sin
    ``idx_ext`` -> ``ValueError``: reutilizar la fila de la F1 hacia
    P(F2|F1)=1.

    Wrapper back-compatible (PLAN_RECOMMEND_F_PAIR §2d): sin ``gaps_fixed``
    dibuja los gaps dentro con ``seed`` (:func:`_draw_gap_tape`: los gaps de la
    F1 son los de siempre, la prolongacion va despues); con ``gaps_fixed``
    inyectado, ``gaps_ext`` es OBLIGATORIO (``ValueError`` si falta) y equivale
    exactamente a ``combine_challenge_funded(simulate_challenge(...),
    sweep_funded(...)[f])``.

    ``gap_days``: distribucion empirica de gaps inter-bloque (dias flat) de
    :func:`empirical_gaps`; se muestrea una por transicion para que el tiempo
    calendario de los ciclos de payout sea realista (CRITICO-1 revision de
    codigo). La fondeada solo se evalua en paths con pass1 ∧ pass2 (lazy: su
    outcome no se lee en los demas — no rompe el muestreo conjunto). Los
    outcomes devueltos llevan ``funded_start`` y ``funded_windows``
    (= ``tape.windows(funded_start)``, int32) para que la API calcule el cap de la
    fondeada; la cinta en sí queda local a esta función.
    """
    if idx_ext is None:
        raise ValueError(
            "simulate_firm_paths exige idx_ext (extend_stationary_rows): reutilizar la "
            "fila de la F1 para la F2 y la fondeada hace P(F2|F1)=1")
    n_paths, n_blocks = idx_matrix.shape
    if gaps_fixed is None:
        rng = np.random.default_rng(seed)
        if gap_days is not None and len(gap_days):
            gaps_fixed, gaps_ext = _draw_gap_tape(rng, gap_days, n_paths, n_blocks)
        else:
            gaps_fixed, gaps_ext = None, None
    elif gaps_ext is None:
        raise ValueError(
            "simulate_firm_paths: con gaps_fixed hay que pasar gaps_ext "
            "(_draw_gap_tape devuelve ambos)")
    tape = build_phase_tape(idx_matrix, idx_ext, gaps_fixed, gaps_ext)
    ch = simulate_challenge(tape, day_arrays, durations, cfg,
                            portfolio=portfolio)
    fu = sweep_funded(tape=tape, funded_start=ch.funded_start,
                      day_arrays_f_grid=[day_arrays], passers_mask=ch.both,
                      durations=durations, cfg=cfg, f_grid=[0.0])[0.0]
    out = combine_challenge_funded(ch, fu)
    out.funded_windows = tape.windows(ch.funded_start)
    return out


@dataclass
class SequenceResult:
    ev_per_attempt: float           # EUR por challenge (condicional a resolver)
    ev_ci95: tuple[float, float]    # IC95 bootstrap de la media de ev_vec
    net_mean: float                 # EUR neto medio tras n_challenges
    p_net_negative: float
    net_p05: float
    net_p50: float
    net_p95: float
    p_pass1: float
    p_both: float                   # P(P1 ∧ P2) conjunta
    p_unresolved: float             # P(neither) — paths excluidos del EV
    p_payout_given_funded: float
    mean_payout_eur: float          # E[payouts | llega a fondeada] (a cap)
    mean_payout_eur_untruncated: float  # idem solo paths no truncados
    challenges_per_payout: float    # nan si ninguna secuencia cobro
    median_days_first_payout: float # nan si ninguno
    median_days_pass1: float        # nan si ninguno
    alive_at_cap_frac: float        # condicional a llegar a fondeada
    trajectory_p05: np.ndarray      # (n_challenges+1,) percentiles del capital real
    trajectory_p50: np.ndarray
    trajectory_p95: np.ndarray
    p_ruin: float = float("nan")        # P(bankroll < fee) si stop_on_insolvency
    mean_attempts: float = float("nan")  # intentos medios ejecutados si stop
    p_expired: float = float("nan")     # P(expired | resolved) — regla lifetime trailing
    # Diagnósticos ladder (plan Apex Fase 4), condicionales a llegar a fondeada;
    # nan si el camino es cycle/FTMO (arrays None en los outcomes).
    mean_dll_hits: float = float("nan")           # hits medios de DLL por PA
    mean_consistency_deferrals: float = float("nan")  # denegaciones 50% medias
    p_inactivity_death: float = float("nan")      # P(muerte por inactividad | fondeada)
    p_closed_after_ladder: float = float("nan")   # P(cierre feliz tras el 6º retiro | fondeada)
    # Fase 5: multi-cuenta y diagnóstico de caps de contratos.
    n_copied_accounts: int = 1          # cuentas copiadas (correlación perfecta)
    sizing_invalid: bool = False        # True: hay paths imposibles (cap contratos)
    frac_cap_violations_eval: float = float("nan")   # fracción violando el cap de evaluación
    frac_cap_violations_pa: float = float("nan")     # fracción violando el cap PA (tier mín.)
    max_simultaneous_micros: float = float("nan")    # máximo observado (micro-eq, ambos scopes)
    warnings: tuple[str, ...] = ()      # warnings estructurados para la UI (no prints)


def ev_vector_all(
    outcomes: FirmPathOutcomes,
    cfg: PropFirmConfig,
    *,
    fx_eur_usd: float,
    cap_cycles: int | None = None,
) -> np.ndarray:
    """EV por path (EUR) sobre los N paths, SIN filtrar por ``resolved``.

    Formula de ``sequence_ev`` extraida para reuso en ``recommend_f_pair``:
    el bootstrap pareado del ΔEV la aplica sobre el pool comun de paths, y
    cada celda condiciona a SU ``resolved`` (PLAN_RECOMMEND_F_PAIR §3 paso 6).

    Activation fee (plan Apex Fase 3): los paths ``both`` (llegan a fondeada)
    descuentan además ``cfg.activation.fee_eur`` — la PA se paga solo al
    pasar. ``fee_eur`` debe venir resuelto a EUR por :func:`resolve_fee_eur`
    en el setup (conversión UNA vez); si hay activation pero ``fee_eur`` es
    None → raise ruidoso. Caveat adjudicado: el deadline de activación de 7
    días NO se modela (siempre activamos). FTMO (``activation=None``) es
    byte-idéntico a la fórmula histórica.
    """
    cap = min(cap_cycles or cfg.funded.max_payout_cycles,
              outcomes.funded_payouts.shape[1])
    both = outcomes.pass1 & outcomes.pass2
    payout_usd = outcomes.funded_payouts[:, :cap].sum(axis=1)
    got_payout = (outcomes.funded_payouts[:, :cap] > 0).any(axis=1)
    refund = cfg.fee if cfg.refund_fee_on_first_payout else 0.0
    ev = -cfg.fee + both.astype(float) * (payout_usd / fx_eur_usd + refund * got_payout)
    act = cfg.activation
    if act is not None:
        if act.fee_eur is None:
            raise ValueError(
                f"{cfg.key}: hay activation fee pero fee_eur es None — resuelve "
                "la config con resolve_fee_eur(cfg, fx_eur_usd) en el setup "
                "(la conversión USD→EUR se hace UNA vez, no por touchpoint)"
            )
        ev = ev - both.astype(float) * act.fee_eur
    return ev


def sequence_ev(
    outcomes: FirmPathOutcomes,
    cfg: PropFirmConfig,
    *,
    fx_eur_usd: float,
    n_challenges: int,
    n_sequences: int,
    initial_capital_eur: float,
    seed: int = 0,
    cap_cycles: int | None = None,
    stop_on_insolvency: bool = False,
    n_copied_accounts: int = 1,
    cap_violation_tol: float = 0.0,
) -> SequenceResult:
    """Nivel 2: secuencias de N challenges con muestreo conjunto por path
    (la F2 y la fondeada de cada path CONTINUAN la secuencia de su F1: ver la
    cabecera del módulo y :class:`PhaseTape`).

    Cada challenge sortea un path k RESUELTO: paga el fee; si pass1[k] ∧
    pass2[k], cobra los payouts (USD->EUR via fx) + refund del fee si la
    fondeada produjo >= 1 payout (refund condicionado).

    ``stop_on_insolvency``: si es True, cada trayectoria se DETIENE cuando su
    bankroll antes de un intento es < fee (no puede pagarlo). Sin el, el modelo
    asume fondos infinitos: ``n_challenges > floor(capital0/fee)`` simula
    intentos impagables y ``p_net_negative`` se calcula sobre un horizonte que
    el usuario quizá no puede ejecutar. Con el stop, el neto se mide al stop
    (o al horizonte si nunca ocurre) y se reportan ``p_ruin`` y
    ``mean_attempts``. Regla práctica sin stop: n_challenges <= capital0/fee.

    ``n_copied_accounts`` (plan Apex Fase 5; 1-20): copy trading — el intento
    opera N cuentas a la vez, así que fee, activation, refund y payouts se
    multiplican por N (el fee efectivo por intento pasa a ser ``N × cfg.fee``
    y la insolvencia compara contra ese fee escalado). CAVEAT de CORRELACIÓN
    PERFECTA: todas las cuentas pasan/queman a la vez; el riesgo de cola NO se
    diversifica y ``p_net_negative`` no mejora por copiar — el multiplicador
    escala ganancias Y pérdidas. Con ``stop_on_insolvency`` el horizonte
    ejecutable se ACORTA (cada intento cuesta N×fee), así que ``p_ruin`` y el
    neto sí cambian. N=1 es no-op exacto.

    ``cap_violation_tol`` (plan Apex Fase 5; default 0.0): si los outcomes
    llevan diagnósticos de caps (:class:`ContractCapDiag`) y la fracción de
    paths violadores de alguno supera el umbral (ESTRICTO), el resultado se
    marca ``sizing_invalid=True`` con un warning estructurado en ``warnings``
    (prefijo ``SIZING_INVALID``) — esos paths son IMPOSIBLES en la cuenta real
    (la firm bloquea la apertura), no una aproximación (adjudicado). Se marca,
    no se aborta: las métricas se reportan igualmente. FTMO (diagnósticos a
    None) queda intacto: ``sizing_invalid=False`` y fracciones nan.

    CRITICO-2 (revision de codigo): los paths ``neither`` (ni pass ni fail en
    el horizonte finito) se EXCLUYEN del EV — en la realidad el challenge
    seguiria operando hasta resolver, y tratarlos como fee perdido subestima
    el EV de forma diferencial a ``f`` bajo. El EV es pues CONDICIONAL a
    resolver el challenge en el horizonte; ``p_unresolved`` reporta la
    fraccion excluida (warning en UI si > 5%).

    Fases con ``on_horizon_exhausted="fail"`` (trailing con lifetime, plan
    Apex Fase 3): los paths ``expired`` (lifetime agotado sin target) cuentan
    como RESUELTOS con EV = −fee — son el "fail por reloj", no horizonte de
    backtest insuficiente. Solo ``neither`` alimenta ``p_unresolved`` (el veto
    de :func:`recommend_f_pair` queda intacto); ``p_expired`` reporta la
    fracción de resueltos que murió por la regla de días.
    """
    if fx_eur_usd <= 0:
        raise ValueError("fx_eur_usd debe ser > 0")
    if (
        not isinstance(n_copied_accounts, (int, np.integer))
        or isinstance(n_copied_accounts, bool)
        or not 1 <= int(n_copied_accounts) <= 20
    ):
        raise ValueError(
            f"n_copied_accounts debe ser un entero en [1, 20] "
            f"(recibido {n_copied_accounts!r})"
        )
    n_acct = int(n_copied_accounts)
    if not 0.0 <= cap_violation_tol <= 1.0:
        raise ValueError("cap_violation_tol debe estar en [0, 1]")
    cap = cap_cycles or cfg.funded.max_payout_cycles
    cap = min(cap, outcomes.funded_payouts.shape[1])
    n = len(outcomes.pass1)
    both = outcomes.both
    unresolved = outcomes.unresolved
    resolved = ~unresolved
    if not resolved.any():
        raise ValueError(
            "Ningun path resolvio (todos neither): amplia el backtest, sube f "
            "o reduce el horizonte. El EV condicional a resolver no es estimable."
        )
    p_unresolved = float(unresolved.mean())
    # Expired (lifetime trailing agotado, on_horizon_exhausted="fail") cuenta
    # como RESUELTO con EV = −fee (pass1=False en ellos); solo neither alimenta
    # p_unresolved (el veto de recommend_f_pair queda semánticamente intacto).
    p_expired = float(outcomes.expired[resolved].mean())

    # Multi-cuenta (Fase 5): payouts y fee de reporte escalados por N; el EV
    # por path se escala entero más abajo (fee + activation + refund incluidos).
    payout_usd = outcomes.funded_payouts[:, :cap].sum(axis=1)
    got_payout = (outcomes.funded_payouts[:, :cap] > 0).any(axis=1)
    payout_eur = payout_usd / fx_eur_usd * n_acct
    refund = cfg.fee * n_acct if cfg.refund_fee_on_first_payout else 0.0

    ev_vec_all = ev_vector_all(outcomes, cfg, fx_eur_usd=fx_eur_usd,
                               cap_cycles=cap_cycles) * n_acct  # EV por path (EUR)
    ev_vec = ev_vec_all[resolved]
    ev_per_attempt = float(ev_vec.mean())

    # IC95 bootstrap de la media del EV (paths resueltos).
    rng = np.random.default_rng(seed)
    n_res = len(ev_vec)
    boot = ev_vec[rng.integers(0, n_res, size=(_EV_CI_N_BOOT, n_res))].mean(axis=1)
    ev_ci95 = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

    idx_res = np.flatnonzero(resolved)
    draws = idx_res[rng.integers(0, n_res, size=(n_sequences, n_challenges))]
    gains = ev_vec_all[draws]

    # Stop por insolvencia: la trayectoria solo ejecuta el intento i si su
    # bankroll ANTES del intento cubre el fee EFECTIVO (N×fee con copy trading;
    # paseo con barrera absorbente; los payouts reponen bankroll y pueden
    # alargar la secuencia mas alla del minimo garantizado
    # floor(capital0/(N×fee))).
    executed = np.ones((n_sequences, n_challenges), dtype=bool)
    ruined = np.zeros(n_sequences, dtype=bool)
    fee_attempt = cfg.fee * n_acct
    stop_applied = stop_on_insolvency and fee_attempt > 0
    if stop_applied:
        pre = initial_capital_eur + np.concatenate(
            [np.zeros((n_sequences, 1)), np.cumsum(gains, axis=1)[:, :-1]], axis=1
        )
        insolvent = pre < fee_attempt
        ruined = insolvent.any(axis=1)
        stop_idx = insolvent.argmax(axis=1)  # 1er intento impagable (0 si ninguno)
        executed = np.arange(n_challenges) < stop_idx[:, None]
        executed[~ruined] = True

    gains_exec = gains * executed
    traj = initial_capital_eur + np.concatenate(
        [np.zeros((n_sequences, 1)), np.cumsum(gains_exec, axis=1)], axis=1
    )
    net = traj[:, -1] - initial_capital_eur

    # challenges por payout: challenge que llego a fondeada Y cobro >= 1 payout
    # (solo intentos ejecutados; con stop, cpp usa los intentos reales).
    hits = both[draws] & got_payout[draws] & executed
    n_hits = hits.sum(axis=1)
    n_attempts = executed.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cpp = np.where(n_hits > 0, n_attempts / np.maximum(n_hits, 1), np.nan)
    cpp_vals = cpp[np.isfinite(cpp)]

    funded_reached = both & resolved
    p_payout = float(got_payout[funded_reached].mean()) if funded_reached.any() else float("nan")
    mean_payout = (
        float((payout_eur[funded_reached] + refund * got_payout[funded_reached]).mean())
        if funded_reached.any() else float("nan")
    )
    untr = funded_reached & ~outcomes.alive_at_cap
    mean_payout_untr = (
        float((payout_eur[untr] + refund * got_payout[untr]).mean())
        if untr.any() else float("nan")
    )
    alive_frac = (
        float(outcomes.alive_at_cap[funded_reached].mean())
        if funded_reached.any() else float("nan")
    )

    # Diagnósticos ladder (Fase 4): medias condicionales a llegar a fondeada;
    # nan si el modelo es cycle (arrays None) o nadie llegó.
    def _diag_mean(arr):
        if arr is None or not funded_reached.any():
            return float("nan")
        return float(arr[funded_reached].mean())

    mean_dll_hits = _diag_mean(outcomes.dll_hits)
    mean_deferrals = _diag_mean(outcomes.consistency_deferrals)
    p_inact_death = _diag_mean(outcomes.inactivity_deaths)
    p_closed_ladder = _diag_mean(outcomes.closed_after_ladder)
    dfp = outcomes.days_first_payout[funded_reached]
    dfp = dfp[np.isfinite(dfp)]
    dp1 = outcomes.days_to_pass1[outcomes.pass1 & resolved]
    dp1 = dp1[np.isfinite(dp1)]

    # Diagnósticos de caps de contratos (Fase 5): marca no silenciable si la
    # fracción de paths violadores supera el umbral — esos paths son
    # imposibles en la cuenta real (la firm bloquea la apertura).
    warnings: list[str] = []
    cap_diags = {
        "eval": outcomes.contract_cap_eval,
        "pa": outcomes.contract_cap_pa,
    }
    frac_eval = float("nan")
    frac_pa = float("nan")
    max_sim = float("nan")
    for tag, diag in cap_diags.items():
        if diag is None:
            continue
        if tag == "eval":
            frac_eval = diag.frac_violating
        else:
            frac_pa = diag.frac_violating
        max_sim = diag.max_observed if not np.isfinite(max_sim) else max(
            max_sim, diag.max_observed
        )
        if diag.frac_violating > cap_violation_tol:
            warnings.append(
                f"SIZING_INVALID[{diag.scope}]: {diag.frac_violating:.1%} de "
                f"paths violan el cap de {diag.cap_micros:.0f} micro-eq "
                f"(máx observado {diag.max_observed:.0f}; umbral "
                f"{cap_violation_tol:.1%}) — esos paths son imposibles en la "
                f"cuenta real: el resultado NO es operable con este sizing"
            )
    sizing_invalid = any(
        d is not None and d.frac_violating > cap_violation_tol
        for d in cap_diags.values()
    )
    if n_acct > 1:
        warnings.append(
            f"MULTI_CUENTA: N={n_acct} cuentas copiadas con CORRELACIÓN "
            f"PERFECTA — pasan/queman a la vez; P(neto<0) no se diversifica "
            f"y el riesgo de cola tampoco: el multiplicador escala ganancias "
            f"Y pérdidas de fees"
        )

    return SequenceResult(
        ev_per_attempt=ev_per_attempt,
        ev_ci95=ev_ci95,
        net_mean=float(net.mean()),
        p_net_negative=float((net < 0).mean()),
        net_p05=float(np.percentile(net, 5)),
        net_p50=float(np.percentile(net, 50)),
        net_p95=float(np.percentile(net, 95)),
        p_pass1=float(outcomes.pass1[resolved].mean()),
        p_both=float(both[resolved].mean()),
        p_unresolved=p_unresolved,
        p_payout_given_funded=p_payout,
        mean_payout_eur=mean_payout,
        mean_payout_eur_untruncated=mean_payout_untr,
        challenges_per_payout=float(np.nanmean(cpp_vals)) if len(cpp_vals) else float("nan"),
        median_days_first_payout=float(np.median(dfp)) if len(dfp) else float("nan"),
        median_days_pass1=float(np.median(dp1)) if len(dp1) else float("nan"),
        alive_at_cap_frac=alive_frac,
        trajectory_p05=np.percentile(traj, 5, axis=0),
        trajectory_p50=np.percentile(traj, 50, axis=0),
        trajectory_p95=np.percentile(traj, 95, axis=0),
        p_ruin=float(ruined.mean()) if stop_applied else float("nan"),
        mean_attempts=float(n_attempts.mean()) if stop_applied else float("nan"),
        p_expired=p_expired,
        mean_dll_hits=mean_dll_hits,
        mean_consistency_deferrals=mean_deferrals,
        p_inactivity_death=p_inact_death,
        p_closed_after_ladder=p_closed_ladder,
        n_copied_accounts=n_acct,
        sizing_invalid=sizing_invalid,
        frac_cap_violations_eval=frac_eval,
        frac_cap_violations_pa=frac_pa,
        max_simultaneous_micros=max_sim,
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------
# Reporte + recomendacion
# ---------------------------------------------------------------------------

@dataclass
class FirmReport:
    """Resultado por ``f`` (nivel 1 agregado + nivel 2)."""

    f: float
    n_paths: int
    p_pass1: float
    p_pass1_ci: tuple[float, float]
    p_both: float
    p_both_ci: tuple[float, float]
    p_unresolved: float
    ev_per_attempt: float
    ev_ci95: tuple[float, float]
    net_mean: float
    p_net_negative: float
    net_p05: float
    net_p50: float
    net_p95: float
    mean_payout_eur: float
    p_payout_given_funded: float
    challenges_per_payout: float
    median_days_first_payout: float
    median_days_pass1: float
    alive_at_cap_frac: float
    seq: SequenceResult = field(repr=False, default=None)
    f_challenge: float | None = None   # modo par: f del challenge (None = f unico)
    f_funded: float | None = None      # modo par: f de la fondeada
    p_expired: float = float("nan")    # P(expired | resolved) — trailing lifetime
    # Diagnósticos ladder (Fase 4; nan en FTMO). to_row() los surfcea de forma
    # ADITIVA (Fase 6) sin tocar las claves históricas.
    mean_dll_hits: float = float("nan")
    mean_consistency_deferrals: float = float("nan")
    p_inactivity_death: float = float("nan")
    p_closed_after_ladder: float = float("nan")
    # Fase 5 (también aditivos en to_row() desde Fase 6):
    n_copied_accounts: int = 1
    sizing_invalid: bool = False
    frac_cap_violations_eval: float = float("nan")
    frac_cap_violations_pa: float = float("nan")
    max_simultaneous_micros: float = float("nan")
    warnings: tuple[str, ...] = ()
    # Plan onboarding-b1 §2.4 (aditivo): fracción de fondeadas ``cycle`` aún vivas
    # cuando se acabó la ventana de datos, sin llegar al cap de ciclos. Mismo
    # denominador que ``alive_at_cap_frac`` (``funded_reached = both & resolved``,
    # ver ``sequence_ev``). None si el modelo no es ``cycle`` (ladder/futures);
    # nan si nadie llegó a fondeada. NO entra en ``to_row()`` (I1).
    funded_cut_by_data_frac: float | None = None

    def to_row(self) -> dict:
        row = {
            "f": self.f,
            "P(P1)": self.p_pass1,
            "P(P1∧P2)": self.p_both,
            "P(sin resolver)": self.p_unresolved,
            "EV/intento €": self.ev_per_attempt,
            "EV IC95 €": f"[{self.ev_ci95[0]:,.0f},{self.ev_ci95[1]:,.0f}]",
            "neto medio €": self.net_mean,
            "P(neto<0)": self.p_net_negative,
            "neto p05 €": self.net_p05,
            "neto p50 €": self.net_p50,
            "neto p95 €": self.net_p95,
            "E[payout €]": self.mean_payout_eur,
            "P(payout|fond.)": self.p_payout_given_funded,
            "challenges/payout": self.challenges_per_payout,
            "dias a payout (med)": self.median_days_first_payout,
            "dias a pase F1 (med)": self.median_days_pass1,
            "vivo_al_cap": self.alive_at_cap_frac,
        }
        if self.f_challenge is not None:
            row = {"f_chal.": self.f_challenge, "f_fond.": self.f_funded, **row}
        # Campos Apex (Fases 3-5), ADITIVOS: las claves históricas no cambian;
        # en FTMO llegan como nan/False/1 (defaults del motor).
        row.update({
            "P(expirado)": self.p_expired,
            "DLL hits (med)": self.mean_dll_hits,
            "deferrals 50% (med)": self.mean_consistency_deferrals,
            "P(muerte inact.)": self.p_inactivity_death,
            "P(cierre escalera)": self.p_closed_after_ladder,
            "n cuentas": self.n_copied_accounts,
            "sizing_invalid": self.sizing_invalid,
            "frac viol. cap eval": self.frac_cap_violations_eval,
            "frac viol. cap PA": self.frac_cap_violations_pa,
            "máx micros simult.": self.max_simultaneous_micros,
        })
        return row


def firm_report(f: float, outcomes: FirmPathOutcomes, seq: SequenceResult) -> FirmReport:
    n = len(outcomes.pass1)
    # M2-10 (revisión GLM r1): los puntos estimados (seq.p_pass1/p_both) van
    # sobre paths RESOLVED; los IC deben usar el MISMO denominador o la tabla
    # mezcla estimandos distintos cuando p_unresolved > 0.
    resolved = outcomes.resolved
    n_res = int(resolved.sum())
    cut_frac = None
    if outcomes.cut_by_data is not None:
        # Misma máscara que ``alive_frac`` en sequence_ev (both & resolved).
        funded_reached = outcomes.both & resolved
        cut_frac = (float(outcomes.cut_by_data[funded_reached].mean())
                    if funded_reached.any() else float("nan"))
    return FirmReport(
        f=float(f), n_paths=n,
        p_pass1=seq.p_pass1,
        p_pass1_ci=wilson_ci(int(outcomes.pass1[resolved].sum()), n_res),
        p_both=seq.p_both,
        p_both_ci=wilson_ci(int((outcomes.pass1 & outcomes.pass2)[resolved].sum()), n_res),
        p_unresolved=seq.p_unresolved,
        ev_per_attempt=seq.ev_per_attempt, ev_ci95=seq.ev_ci95,
        net_mean=seq.net_mean,
        p_net_negative=seq.p_net_negative,
        net_p05=seq.net_p05, net_p50=seq.net_p50, net_p95=seq.net_p95,
        mean_payout_eur=seq.mean_payout_eur,
        p_payout_given_funded=seq.p_payout_given_funded,
        challenges_per_payout=seq.challenges_per_payout,
        median_days_first_payout=seq.median_days_first_payout,
        median_days_pass1=seq.median_days_pass1,
        alive_at_cap_frac=seq.alive_at_cap_frac,
        seq=seq,
        p_expired=seq.p_expired,
        mean_dll_hits=seq.mean_dll_hits,
        mean_consistency_deferrals=seq.mean_consistency_deferrals,
        p_inactivity_death=seq.p_inactivity_death,
        p_closed_after_ladder=seq.p_closed_after_ladder,
        n_copied_accounts=seq.n_copied_accounts,
        sizing_invalid=seq.sizing_invalid,
        frac_cap_violations_eval=seq.frac_cap_violations_eval,
        frac_cap_violations_pa=seq.frac_cap_violations_pa,
        max_simultaneous_micros=seq.max_simultaneous_micros,
        warnings=seq.warnings,
        funded_cut_by_data_frac=cut_frac,
    )


def firm_report_pair(f_c: float, f_f: float, outcomes: FirmPathOutcomes,
                     seq: SequenceResult) -> FirmReport:
    """Report de una celda ``(f_c, f_f)`` del barrido 2D.

    Las metricas de challenge (``p_pass1``, ``p_both``…) vienen de ``f_c``; los
    payouts, de ``f_f``. ``f`` queda en ``f_c`` por compatibilidad con la tabla
    actual; los campos ``f_challenge``/``f_funded`` marcan el par.
    """
    r = firm_report(f_c, outcomes, seq)
    r.f_challenge = float(f_c)
    r.f_funded = float(f_f)
    return r


@dataclass
class EVRecommendation:
    f: float
    feasible: bool
    ev_per_attempt: float
    p_net_negative: float
    reason: str


def recommend_f_ev(reports: list[FirmReport], tol_p_negative: float) -> EVRecommendation | None:
    """Max EV/intento sujeto a P(neto<0) <= tol (MEDIO-3 de la revision).

    Fallback: si ningun ``f`` cumple la restriccion, el de menor P(neto<0).
    Empate en EV -> el ``f`` mas bajo.
    """
    if not reports:
        return None
    feasible = [r for r in reports if r.p_net_negative <= tol_p_negative]
    if feasible:
        best = max(feasible, key=lambda r: (r.ev_per_attempt, -r.f))
        reason = f"máx EV con P(neto<0) ≤ {tol_p_negative:.0%}"
    else:
        best = min(reports, key=lambda r: (r.p_net_negative, r.f))
        reason = (f"ningún f cumple P(neto<0) ≤ {tol_p_negative:.0%}; "
                  "se muestra el de menor P(neto<0)")
    return EVRecommendation(
        f=best.f, feasible=bool(feasible),
        ev_per_attempt=best.ev_per_attempt,
        p_net_negative=best.p_net_negative, reason=reason,
    )


# ---------------------------------------------------------------------------
# Recomendacion por par (f_challenge, f_funded) — PLAN_RECOMMEND_F_PAIR §3
# ---------------------------------------------------------------------------

@dataclass
class PairRecommendation:
    """Recomendacion del barrido 2D ``(f_challenge, f_funded)``."""

    f_challenge: float
    f_funded: float
    f_single: float                  # f ganador de la diagonal (baseline)
    ev_pair: float
    ev_single: float                 # mejor EV de la diagonal (mismos filtros)
    delta_ev: float
    delta_ev_ci: tuple[float, float]  # IC95 bootstrap PAREADO (pool comun)
    p_pasar: float
    p_unresolved: float
    alive_at_cap_frac: float
    on_boundary: tuple[bool, bool]   # (f_c en borde?, f_f en borde?)
    feasible: bool
    recommend_pair: bool             # False -> IC incluye 0: quedarse en diagonal
    reason: str
    n_boot_ok: int = 0               # replicas validas del bootstrap pareado


def recommend_f_pair(
    reports_2d: dict[tuple[float, float], FirmReport],
    cells: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]],
    tol_p_negative: float,
    *,
    p_unresolved_max: float = 0.05,
    n_boot: int = 2000,
    seed: int = 0,
    delta_econ: float = 0.0,
) -> PairRecommendation | None:
    """Max EV sobre la grilla ``(f_c, f_f)`` s.a. P(neto<0)<=tol y veto por
    ``p_unresolved``, con ΔEV vs la diagonal y su IC95 bootstrap PAREADO.

    ``cells[(f_c, f_f)] = (ev_vec_all, resolved)`` por celda (``ev_vec_all`` de
    :func:`ev_vector_all`, sin filtrar por ``resolved``). El bootstrap sortea
    UN vector de indices del pool comun ``{0..N-1}`` por replica y cada celda
    promedia sobre su interseccion con SU ``resolved`` (ratio estimator:
    mimifica el estimador puntual del ΔEV; NO se usa la interseccion de
    resueltos, que cambiaria el estimando — PLAN_RECOMMEND_F_PAIR §3 paso 6).

    Recomienda el par solo si el IC95 del ΔEV queda por encima de
    ``max(0, delta_econ)``; si no, recomienda el ganador de la diagonal (que se
    calcula con los MISMOS filtros de factibilidad y veto — N7). Fallback (nada
    factible): la celda de menor P(neto<0), como :func:`recommend_f_ev`.
    """
    if not reports_2d:
        return None

    def _pick(reps: dict) -> tuple[tuple[float, float] | None, bool]:
        """(celda ganadora, hay_factible) bajo filtro de factibilidad + veto."""
        surv = {k: r for k, r in reps.items()
                if r.p_net_negative <= tol_p_negative
                and r.p_unresolved <= p_unresolved_max}
        if surv:
            return max(surv, key=lambda k: (surv[k].ev_per_attempt, -k[0], -k[1])), True
        if reps:
            return min(reps, key=lambda k: (reps[k].p_net_negative, k[0], k[1])), False
        return None, False

    fs = sorted({k[0] for k in reports_2d} | {k[1] for k in reports_2d})
    f_lo, f_hi = fs[0], fs[-1]
    diag = {k: r for k, r in reports_2d.items() if k[0] == k[1]}

    best_key, feasible = _pick(reports_2d)
    diag_key, _ = _pick(diag)
    if diag_key is None:
        diag_key = best_key  # grilla degenerada sin diagonal

    best = reports_2d[best_key]
    ev_pair = best.ev_per_attempt
    ev_single = reports_2d[diag_key].ev_per_attempt
    delta_ev = ev_pair - ev_single

    # IC95 pareado del ΔEV (pool comun; cada celda condiciona a SU resolved).
    delta_ci = (float("nan"), float("nan"))
    n_ok = 0
    pair_cell = cells.get(best_key)
    diag_cell = cells.get(diag_key)
    if pair_cell is not None and diag_cell is not None and n_boot > 0:
        ev_a, res_a = pair_cell
        ev_b, res_b = diag_cell
        n = len(ev_a)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=(n_boot, n))
        ra, rb = res_a[idx], res_b[idx]
        ca, cb = ra.sum(axis=1), rb.sum(axis=1)
        ok = (ca > 0) & (cb > 0)
        n_ok = int(ok.sum())
        if n_ok:
            d = ((ev_a[idx] * ra)[ok].sum(axis=1) / ca[ok]
                 - (ev_b[idx] * rb)[ok].sum(axis=1) / cb[ok])
            delta_ci = (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))

    thr = max(0.0, delta_econ)
    recommend_pair = bool(
        feasible and best_key != diag_key
        and np.isfinite(delta_ci[0]) and delta_ci[0] > thr
    )
    if not feasible:
        rec_key = best_key
        reason = (f"ninguna celda cumple P(neto<0) ≤ {tol_p_negative:.0%} y "
                  f"P(sin resolver) ≤ {p_unresolved_max:.0%}; se muestra la de "
                  "menor P(neto<0)")
    elif recommend_pair:
        rec_key = best_key
        reason = (f"máx EV con P(neto<0) ≤ {tol_p_negative:.0%} y P(sin resolver) "
                  f"≤ {p_unresolved_max:.0%}; ΔEV = {delta_ev:+,.0f} € "
                  f"(IC95 [{delta_ci[0]:+,.0f}, {delta_ci[1]:+,.0f}])")
    else:
        rec_key = diag_key
        reason = (f"ΔEV = {delta_ev:+,.0f} € no significativo "
                  f"(IC95 [{delta_ci[0]:+,.0f}, {delta_ci[1]:+,.0f}]) → f único "
                  "de la diagonal")

    rec = reports_2d[rec_key]
    return PairRecommendation(
        f_challenge=rec_key[0], f_funded=rec_key[1],
        f_single=diag_key[0],
        ev_pair=ev_pair, ev_single=ev_single,
        delta_ev=delta_ev, delta_ev_ci=delta_ci,
        p_pasar=rec.p_both, p_unresolved=rec.p_unresolved,
        alive_at_cap_frac=rec.alive_at_cap_frac,
        on_boundary=(rec_key[0] in (f_lo, f_hi), rec_key[1] in (f_lo, f_hi)),
        feasible=feasible, recommend_pair=recommend_pair,
        reason=reason, n_boot_ok=n_ok,
    )
