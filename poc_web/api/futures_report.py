"""Adaptador de reporte FTMO Futures para la API (plan §3.2, capa API).

El motor (``src.risk``) está CONGELADO: este módulo traduce los resultados
del motor (``sequence_ev_futures`` → :class:`FuturesSequenceResult`) al
espejo de :class:`src.risk.propfirm.FirmReport` que consume el frontend,
con los extras futures como campos aditivos. Convenciones (adjudicadas):

- El EV es INCONDICIONAL por campaña (cap 183d = truncado del modelo, no
  regla FTMO): los IC de ``p_pass1``/``p_both`` usan ``wilson_ci`` sobre
  los N paths COMPLETOS del backtest (sin filtrar resolved).
- ``p_both`` = tasa de ``pass1`` bajo ``assert len(cfg.phases) == 1`` (los
  perfiles futures son monofásicos → ``pass2`` siempre True).
- ``n_paths`` = paths del backtest (``len(outcomes.pass1)``), NO campañas.
- ``sizing_invalid``: umbral ESTRICTO ``frac_eval > cap_violation_tol``
  (replica la semántica de ``sequence_ev``), warning con el formato
  ``SIZING_INVALID`` del motor, leyendo ``outcomes.contract_cap_eval``
  (acoplado por el endpoint para AMBAS ramas). El cap PA no aplica (los
  perfiles futures no tienen scaling tiers).
- ``warnings``: las del motor (``seq.warnings``) + el sizing si aplica.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from src.risk.propfirm import wilson_ci

__all__ = ["FuturesReport", "firm_report_futures", "withdrawal_policy_cfg"]


@dataclass
class FuturesReport:
    """Espejo de ``FirmReport`` (mismas claves de tabla/JSON) + extras futures.

    ``ev_per_attempt`` es un ALIAS de ``ev_per_campaign`` (semántica
    etiquetada en ``meta.ev_semantics``); ``p_unresolved`` = fracción de
    neither INCONDICIONAL sobre N paths.
    """

    f: float
    n_paths: int
    p_pass1: float
    p_pass1_ci: tuple
    p_both: float
    p_both_ci: tuple
    p_unresolved: float
    ev_per_attempt: float
    ev_ci95: tuple
    net_mean: float
    p_net_negative: float
    net_p05: float
    net_p50: float
    net_p95: float
    mean_payout_eur: float = float("nan")
    p_payout_given_funded: float = float("nan")
    challenges_per_payout: float = float("nan")
    median_days_first_payout: float = float("nan")
    median_days_pass1: float = float("nan")
    alive_at_cap_frac: float = float("nan")
    # Extras futures (aditivos):
    ev_per_day: float = float("nan")
    p_truncated: float = float("nan")
    p_neither_attempts: float = float("nan")
    mean_billings: float = float("nan")
    mean_resets: float = float("nan")
    mean_attempts: float = float("nan")
    mean_days_campaign: float = float("nan")
    mean_dll_hits: float = float("nan")
    mean_payout_deferrals: float = float("nan")
    p_ruin: float = float("nan")
    sizing_invalid: bool = False
    frac_cap_violations_eval: float = float("nan")
    reset_policy: str = ""
    withdrawal_policy: str = ""
    warnings: tuple = ()


def withdrawal_policy_cfg(cfg, policy: str):
    """Política de retirada aplicada a la cfg (espejo del batch CLI
    ``sqx/exp_ftmo_futures_batch._withdrawal_policy_cfg``: ``dataclasses.replace``
    sobre ``payouts_futures.withdrawal_policy``). Debe aplicarse ANTES de
    ``simulate_firm_paths`` — ``sweep_funded`` la lee en la evaluación."""
    return replace(
        cfg,
        funded=replace(
            cfg.funded,
            payouts_futures=replace(
                cfg.funded.payouts_futures, withdrawal_policy=policy
            ),
        ),
    )


def firm_report_futures(f, outcomes, seq, *, cfg, reset_policy,
                        withdrawal_policy, cap_violation_tol=0.0
                        ) -> FuturesReport:
    """Reporte de un ``f`` para perfiles futures a partir de los outcomes
    del backtest y la secuencia de campañas del motor."""
    assert len(cfg.phases) == 1, (
        "firm_report_futures asume perfil monofásico (p_both = p_pass1): "
        f"{len(cfg.phases)} fases"
    )
    # N = paths del backtest (E9): p_pass1/p_both/p_unresolved INCONDICIONALES.
    n = len(outcomes.pass1)
    both = outcomes.pass1 & outcomes.pass2
    frac_eval = float("nan")
    sizing_invalid = False
    warnings: list = list(seq.warnings)
    diag = outcomes.contract_cap_eval
    if diag is not None:
        frac_eval = float(diag.frac_violating)
        if diag.frac_violating > cap_violation_tol:  # umbral ESTRICTO
            sizing_invalid = True
            warnings.append(
                f"SIZING_INVALID[{diag.scope}]: {diag.frac_violating:.1%} de "
                f"paths violan el cap de {diag.cap_micros:.0f} micro-eq "
                f"(máx observado {diag.max_observed:.0f}; umbral "
                f"{cap_violation_tol:.1%}) — esos paths son imposibles en la "
                f"cuenta real: el resultado NO es operable con este sizing"
            )
    return FuturesReport(
        f=float(f), n_paths=n,
        p_pass1=float(outcomes.pass1.mean()),
        p_pass1_ci=wilson_ci(int(outcomes.pass1.sum()), n),
        p_both=float(both.mean()),
        p_both_ci=wilson_ci(int(both.sum()), n),
        p_unresolved=float(outcomes.neither1.mean()),
        ev_per_attempt=seq.ev_per_campaign,
        ev_ci95=tuple(seq.ev_ci95),
        net_mean=seq.net_mean_eur,
        p_net_negative=seq.p_net_negative,
        net_p05=seq.net_p05, net_p50=seq.net_p50, net_p95=seq.net_p95,
        p_payout_given_funded=seq.p_payout_given_funded,
        median_days_first_payout=seq.median_days_first_payout,
        median_days_pass1=seq.median_days_to_pass,
        alive_at_cap_frac=seq.alive_at_cap_frac,
        ev_per_day=seq.ev_per_day,
        p_truncated=seq.p_truncated,
        p_neither_attempts=seq.p_neither_attempts,
        mean_billings=seq.mean_billings,
        mean_resets=seq.mean_resets,
        mean_attempts=seq.mean_attempts,
        mean_days_campaign=seq.mean_days_campaign,
        mean_dll_hits=seq.mean_dll_hits,
        mean_payout_deferrals=seq.mean_payout_deferrals,
        p_ruin=seq.p_ruin,
        sizing_invalid=sizing_invalid,
        frac_cap_violations_eval=frac_eval,
        reset_policy=reset_policy,
        withdrawal_policy=withdrawal_policy,
        warnings=tuple(warnings),
    )
