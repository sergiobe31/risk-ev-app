"""Lente AXI F4 — camino con continuidad post-target (regla DOBLE).

Contexto: ``poc_web/ANALISIS_CASOS_PASS_F4.md`` §9 (propuesta) y §10 (GO de
Claude, 2026-08-22). El motor ``src/risk/mc_portfolio.evaluate_path`` corta el
camino en el primer cruce de ``pass_line`` sin mirar los suelos de etapa
(``min_unique_trades``/``min_days``), dejando inobservables los destinos
post-target (casos 2a/2b/2c de §4 del análisis). Este módulo implementa el
walker con continuidad aprobado en §10:

- **Regla DOBLE de parada:** ``pass`` solo el primer día con
  ``m ≥ pass_line`` ∧ ``T(d) ≥ min_unique_trades``; si falta T, el camino
  CONTINÚA con el mismo f (política declarada, §10.3-1/§10.4-1). D
  (``min_days``) nunca gatea el pass: esperar plano es gratis ⇒ se reporta como
  día de elegibilidad (``days_cal``), no como riesgo.
- ``ruin`` idéntico al motor (primer ``m ≤ ruin_line``); orden de chequeos por
  día conservado: ruin → pass → timeout (§10.3-6). Agotar ``horizon_days`` con
  T aún bajo ⇒ ``"timeout"`` (sin resolver).
- **T(d) cuenta trades COLOCADOS (fecha de APERTURA ≤ d, §10.3-2)**, vía
  :func:`precompute_open_days` (``sym_id`` + ``trade_local_id`` → ``open_t``).
  El T block-level del motor se registra aparte (``trades_block`` /
  ``fc_trades_block``) para la fidelidad raw (§10.3-5).
- **``first_cross``**: snapshot del primer cruce pass/ruin (día, T block-level
  y por día, max_dd, terminal) ⇒ los campos "raw" del payload se derivan sin
  recomputar (:func:`raw_outcome`; V4, tolerancia cero vs ``run_portfolio_mc``).
- **``days_cal``**: estimación calendario de la edad de etapa =
  ``days_busy + (n_bloques_consumidos − 1) · gap_mean`` (§10.3-3: huecos
  inter-bloque completados; el sobreconteo intrabloque — ``days_seen`` suma la
  duración completa del bloque en curso tras el toque — queda acotado por la
  duración media de bloque). La edad simulada arranca en el inicio de la
  ventana de datos, no en el alta real de etapa. ``gap_mean`` se deriva del
  dato en runtime con :func:`mean_inter_block_gap`, nunca hardcodeado (§10.2).

Dependencia deliberada (§10.3-4): importa la función PRIVADA
``_block_day_eval`` del motor para tener una única fuente de verdad de la
convención merged (saldo cerrado previo al día + peor flotante intradía,
validada vs ``dailyEquity.bin``). Guardianes contra deriva: V1/V2 en
``test_axi_path.py`` y el bloque 1 de ``check_axi.py``. ``src/risk/`` permanece
intocado.

Limitación conocida (E7, §10.1): la extensión de horizonte vía
``stationary_block_indices(length=…)`` NO preserva la ``idx_matrix`` compartida
(la desincronización del stream rng entre paths desplaza todas las filas) y el
empalme por concatenación truncaría el bloque en curso en la costura (no
equivale a un path continuo de 2n). Por eso F4 no la usa; si hiciera falta más
ventana para resolver 2c, requiere construcción con rng independiente.

Limitación heredada (§9.2): la serie merged BOUND guarda un único peor-punto por
día ⇒ toques intradía de target con cierre por debajo son invisibles
(subdetección de pass). En F5b (GO condicionado §8.6 de
``poc_web/PROPUESTA_F5_PRECISION_MOTOR.md``) queda como variante INTERNA de
fidelidad (``mtm="bound"``); el walker expuesto usa la serie JOINT.

F5b — convención DUAL (§8.4-Q3, §8.6-1):

- **Walker expuesto 100% joint** (``mtm="joint"``): las decisiones
  (ruin/pass/timeout) recorren la serie merged con flotante JOINT = min diario
  de la suma de flotantes M1 de las posiciones vivas (construcción adjudicada
  de ``src.risk.propfirm``, reusada vía shim de tipos — ``BlockRecord`` no
  tiene ``n_trades`` — sobre ``_joint_float_columns``/``_joint_day_float_block``;
  ``src/risk/`` sigue intocado). El precomputo por (bloque, f) vive en
  :func:`joint_columns_cache` (f-indep, una vez por simulación) y
  :func:`precompute_joint` (merged con saldo 0; el walk suma el saldo).
- **``first_cross``/``fc_*`` SIEMPRE en bound** (semántica motor): en modo
  joint un sub-pase bound con semántica EXACTA del motor
  (:func:`_bound_motor_walk`, espejo de ``evaluate_path``) produce el snapshot
  y el ``raw_path`` motor-exacto ⇒ ``raw_outcome`` conserva V4 y el bloque 1 de
  ``check_axi.py`` sobrevive intacto.
- OJO (§8.2): la dirección bound↔joint NO es monótona (el joint puede ser más
  adverso en días concretos) — ningún texto del payload/UI afirma dirección.

D2 (guardrail ``csv-d1-d2``, fase 2; plan csv-d1-d2 §2.3 y Enmiendas v2 E6-E7, E10): la serie
joint de arriba NO suma el realizado del día (``merged0(d)`` = prefijo cerrado ANTES de d + mínimo del
flotante de las posiciones vivas): una pérdida cerrada por la mañana y otra flotante por la tarde no se
suman en el peor punto del día. :func:`minute_columns_cache` + :func:`precompute_minute` construyen la
misma tupla ``(days, merged0, total_pnl)`` por bloque con el modo POR MINUTO del motor
(``realizado_≤t + Σ flotante de las vivas``, mínimo por día), reutilizando el constructor del trailing
(``src.risk.propfirm_trailing``, sin tocarlo). El walker no cambia (``mtm="joint"`` con ``joint_pre``);
las funciones del joint se quedan (A/B y ``pareado_bound_joint_f5.py``).
"""
from __future__ import annotations

import datetime as dt
import bisect
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd

# Dependencia privada deliberada (§10.3-4): única fuente de verdad de la
# convención merged. Guardián: V1/V2 (test_axi_path.py) + check_axi bloque 1.
from src.risk.mc_portfolio import PathOutcome, _block_day_eval
# Dependencia privada deliberada (§8.6-1 de PROPUESTA_F5): única fuente de
# verdad de la construcción JOINT adjudicada (calendario conjunto + ffill
# físico + min diario de la suma ponderada). El shim de tipos lo aporta
# :func:`_block_shim` (BlockRecord → lo que esperan las funciones de
# PortfolioDayBlock). Guardián: tests de equivalencia en test_axi_path.py.
from src.risk.propfirm import _joint_day_float_block, _joint_float_columns
# D2 (csv-d1-d2): constructor del trailing, reusado tal cual por el modo "minute" del motor
# (``propfirm._minute_columns``); aquí solo hace falta ``rel_low`` (realizado ≤ t + flotante de las vivas).
from src.risk.propfirm_trailing import (align_trailing_days, build_trailing_columns,
                                        trailing_block_result)


@dataclass
class PathOutcomeAXI:
    """Outcome del walker F4 (continuación post-target, regla doble).

    ``outcome``/``days``/``max_dd``/``terminal``/``trades_block`` tienen la
    semántica del motor en el stop FINAL (para paths fuera del cubo degradado
    son bit-idénticos a ``evaluate_path`` — V2). ``trades`` es T(día del stop)
    por APERTURA (el que gobierna la decisión de suelo y ``trades_qual``).
    ``first_cross`` + ``fc_*``: snapshot del primer cruce (semántica raw del
    motor) para derivar los campos raw sin recomputar (V4). ``n_blocks`` =
    bloques consumidos en el stop (alimenta ``days_cal``).
    """
    outcome: str             # 'pass' | 'ruin' | 'timeout' | 'neither'
    days: float              # días busy acumulados hasta el stop (como motor)
    trades: int              # T(día del stop) por APERTURA
    max_dd: float            # fracción (negativo) sobre el baseline
    terminal: float          # merged del stop / último merged del path
    trades_block: int        # T block-level en el stop (semántica motor)
    n_blocks: int            # bloques consumidos en el stop
    days_cal: float          # days + (n_blocks−1)·gap_mean (ESTIMACIÓN, A3)
    first_cross: str | None  # 'pass' | 'ruin' | None (primer cruce, raw bound)
    fc_days: float           # snapshot del primer cruce (NaN/0 si no hubo)
    fc_trades_block: int
    fc_trades_day: int
    fc_max_dd: float
    fc_terminal: float
    # Modo joint: outcome motor-exacto del sub-pase bound (§8.6-1). None en
    # modo bound (raw se deriva del snapshot, lógica F4 intacta).
    raw_path: PathOutcome | None = None


def precompute_open_days(blocks, strategies) -> list[np.ndarray]:
    """Ordinales de fecha de APERTURA por bloque (ordenados), para T(d).

    ``strategies`` es el dict de ``StrategyMC`` en el MISMO orden de inserción
    con que se construyeron los bloques (``sym_id`` indexa ese orden). Sale un
    array por bloque, paralelo a la lista ``blocks`` (posición = la que usa el
    walker). Trades con apertura NaT no pueden pertenecer a un bloque
    (``build_block_records`` exige valle, que exige ``open_t`` válido); aun así
    se filtran ordinales < 0 por defensa.
    """
    ord_by_sym = []
    for s in strategies.values():
        ords = np.full(len(s.open_t), -1, dtype=np.int64)
        for i, t in enumerate(s.open_t):
            if not pd.isna(t):
                ords[i] = pd.Timestamp(t).toordinal()
        ord_by_sym.append(ords)
    out: list[np.ndarray] = []
    for b in blocks:
        od = np.array([ord_by_sym[b.sym_id[j]][b.trade_local_id[j]]
                       for j in range(len(b.sym_id))], dtype=np.int64)
        out.append(np.sort(od[od >= 0]))
    return out


def mean_inter_block_gap(blocks) -> float:
    """Hueco medio entre bloques consecutivos del calendario original (días).

    ḡ = (span − Σ duration_days) / (n_bloques − 1), derivado SIEMPRE del dato
    en runtime (§10.2: nunca hardcodeado). Denominador = span entre el inicio
    del primer bloque y el fin del último (ventana de trading de los bloques).
    """
    if len(blocks) < 2:
        return 0.0
    span = (pd.Timestamp(blocks[-1].end)
            - pd.Timestamp(blocks[0].start)).total_seconds() / 86400.0
    dur = sum(b.duration_days for b in blocks)
    return max((span - dur) / (len(blocks) - 1), 0.0)


# ---------------------------------------------------------------------------
# F5b — serie JOINT (flotante conjunto intradía) para el walker (§8.6-1)
# ---------------------------------------------------------------------------
def _block_shim(b) -> SimpleNamespace:
    """Lo mínimo que las funciones joint de ``propfirm`` esperan del bloque
    (``PortfolioDayBlock.n_trades/sym_id/trade_local_id``; ``BlockRecord`` no
    tiene ``n_trades`` — shim de tipos, §8.5)."""
    return SimpleNamespace(n_trades=len(b.sym_id), sym_id=b.sym_id,
                           trade_local_id=b.trade_local_id)


def joint_columns_cache(blocks, minute_by_sym) -> list:
    """Columnas M1 f-INDEPENDIENTES por bloque (una vez por simulación).

    Salida de ``_joint_float_columns`` por bloque (None si ningún trade del
    bloque tiene serie M1). Es la parte cara (calendario conjunto + ffill
    físico); la ponderación por f la hace :func:`_block_day_eval_joint`.
    NO se persiste en ``_portfolios`` (§8.6-9): se reconstruye por simulación.
    """
    return [_joint_float_columns(_block_shim(b), minute_by_sym)
            for b in blocks]


def _block_day_eval_joint(b, lots, pv_by_sym, minute_by_sym, joint_cols):
    """Espejo de ``_block_day_eval`` con flotante JOINT y saldo_start = 0.

    Misma convención de prefijo (saldo cerrado ANTES del día d: los closes del
    propio día no entran) y mismo orden de cómputo — pero el flotante del día
    es el min diario de la suma M1 ponderada por lotes×pv (construcción
    adjudicada de ``propfirm``). Devuelve ``(days, merged0, total_pnl)``; el
    walk suma el saldo real (``merged = merged0 + saldo``). La serie joint no
    ancla bit-identidad contra el motor, así que el precomputo por (bloque, f)
    es válido aquí — en bound NO se hace (reordenar la suma flotante rompería
    la bit-identidad V1/V2/V4).
    """
    pnl_trade = lots * b.pnl_per_lot
    cd = b.close_d
    order = np.argsort(cd, kind="mergesort")
    cd_s, pt_s = cd[order], pnl_trade[order]
    uniq, inv = np.unique(cd_s, return_inverse=True)
    pnl_per_d = np.zeros(len(uniq))
    np.add.at(pnl_per_d, inv, pt_s)
    cs = np.cumsum(pnl_per_d)
    day_float_ord = _joint_day_float_block(_block_shim(b), lots, pv_by_sym,
                                           minute_by_sym, joint_cols)
    day_float = {dt.date.fromordinal(o): v for o, v in day_float_ord.items()}
    all_d = sorted(set(uniq.tolist()) | set(day_float.keys()))
    uniq_list = uniq.tolist()
    merged0 = []
    for d in all_d:
        idx = bisect.bisect_left(uniq_list, d)
        prefix = float(cs[idx - 1]) if idx > 0 else 0.0
        merged0.append(prefix + day_float.get(d, 0.0))
    total_pnl = float(cs[-1]) if len(cs) else 0.0
    return np.array(all_d), np.array(merged0), total_pnl


def precompute_joint(blocks, lots_by_f_sym, pv_by_sym, f_idx, minute_by_sym,
                     cols_cache) -> list:
    """``(days, merged0, total_pnl)`` por bloque a un ``f`` (saldo_start = 0)."""
    out = []
    for bi, b in enumerate(blocks):
        lots = np.array([
            lots_by_f_sym[b.sym_id[j]][f_idx, b.trade_local_id[j]]
            for j in range(len(b.sym_id))], dtype=float)
        out.append(_block_day_eval_joint(b, lots, pv_by_sym, minute_by_sym,
                                         cols_cache[bi]))
    return out


# ---------------------------------------------------------------------------
# D2 — serie diaria POR MINUTO con el realizado intradía (csv-d1-d2, fase 2)
# ---------------------------------------------------------------------------
def _minute_shim(b, strategies=None) -> SimpleNamespace:
    """Lo mínimo que el constructor del trailing espera de un bloque (``PortfolioDayBlock``).

    ``BlockRecord`` no tiene ``n_trades`` ni ``open_t``/``close_t`` (shim de tipos, como
    :func:`_block_shim`). Con ``strategies`` (LISTA de ``StrategyMC`` en el orden de ``sym_id``,
    es decir ``list(strats.values())``) añade ``open_t``/``close_t`` por trade, que
    :func:`build_trailing_columns` necesita para el calendario y para la máscara ``[ot, ct)``.
    Sin ellas basta para ponderar por lotes (:func:`trailing_block_result` con ``tcols``).
    """
    k = len(b.sym_id)
    shim = SimpleNamespace(n_trades=k, sym_id=b.sym_id,
                           trade_local_id=b.trade_local_id,
                           pnl_per_lot=b.pnl_per_lot)
    if strategies is not None:
        def _at(attr: str) -> np.ndarray:
            return np.array(
                [getattr(strategies[int(b.sym_id[j])], attr)[int(b.trade_local_id[j])]
                 for j in range(k)], dtype="datetime64[ns]")

        shim.open_t = _at("open_t")
        shim.close_t = _at("close_t")
    return shim


def minute_columns_cache(blocks, strategies, minute_by_sym) -> list:
    """Calendario conjunto + columnas por minuto f-INDEPENDIENTES, por bloque (una vez por simulación).

    ``strategies``: LISTA de ``StrategyMC`` en el orden de ``enumerate(strats.values())`` — el mismo
    que ``lots_by``/``pv_by``/``minute_by_sym`` (E7). ``minute_by_sym`` = ``{sym_id: {local_id: (ts, vals,
    ot, ct)}}`` (formato de ``StrategyMC.minute_valle``) con serie de los trades de los bloques que se
    quieran acotar; un trade sin serie no aporta flotante (su ``ct`` sí entra en el calendario).

    Por bloque llama a ``build_trailing_columns(shim, minute_by_sym, minute_by_sym, day_labeler=None)``:
    la serie adversa hace de adversa Y de favorable (la favorable se descarta: ``fav_cols`` es un alias
    de ``adv_cols``, como en ``propfirm._minute_columns``); día de calendario, igual que el joint de
    hoy. Convención heredada del trailing (E10): la envolvente de cada trade está activa en ``[ot, ct)``
    — CERO en ``ct`` — y el realizado se inyecta EN ``ct``. NO se persiste en ``_portfolios``: se
    reconstruye por simulación.
    """
    if not isinstance(strategies, (list, tuple)):
        raise TypeError("minute_columns_cache: `strategies` debe ser una LISTA en el orden de sym_id "
                        "(list(strats.values())); un dict por nombre no sirve (E7)")
    out = []
    for b in blocks:
        tcols = build_trailing_columns(_minute_shim(b, strategies), minute_by_sym,
                                       minute_by_sym, day_labeler=None)
        tcols.fav_cols = tcols.adv_cols
        out.append(tcols)
    return out


def precompute_minute(blocks, lots_by_f_sym, pv_by_sym, f_idx, cache) -> list:
    """``(days, merged0, total_pnl)`` por bloque a un ``f`` (saldo_start = 0), con el MISMO formato que
    :func:`precompute_joint` y por tanto sin tocar el walker.

    ``merged0(d)`` = ``rel_low[d]`` del constructor del trailing = ``min_{t del día d}(realizado_≤t +
    Σ flotante de las posiciones vivas)`` relativo al saldo de inicio del bloque; el walk suma el saldo
    real (``merged = merged0 + saldo``). A diferencia de :func:`precompute_joint`, el realizado del
    propio día YA está dentro del mínimo (no hay "prefijo cerrado antes de d" aparte).

    Conjunto de días (E6): los días con minutos propios (``res.day_ord``) ∪ los días de cierre del
    bloque, igual que el joint de hoy; NO se rellenan los días entre la apertura y el cierre (un fin
    de semana con una posición abierta no es una fila: así ``days_seen``, T(d) y el orden
    ruin→pass→timeout no cambian de contrato). Un día sin minutos propios hereda el estado del último
    minuto del día anterior (``align_trailing_days``). ``days`` sale como objetos ``date``, ORDENADOS
    (con ``assert`` de monotonía), que es lo que espera el walker (``d.toordinal()``).

    ``cache``: salida de :func:`minute_columns_cache` (una entrada por bloque).
    """
    if len(cache) != len(blocks):
        raise ValueError(f"cache de columnas por minuto ({len(cache)}) y bloques "
                         f"({len(blocks)}) no coinciden: reconstruye la caché")
    out = []
    for bi, b in enumerate(blocks):
        lots = np.array([
            lots_by_f_sym[b.sym_id[j]][f_idx, b.trade_local_id[j]]
            for j in range(len(b.sym_id))], dtype=float)
        shim = _minute_shim(b)
        res = trailing_block_result(shim, lots, pv_by_sym, None, None,
                                    tcols=cache[bi])
        close_ords = {pd.Timestamp(d).toordinal() for d in b.close_d}
        day_ord = np.array(sorted({int(o) for o in res.day_ord} | close_ords),
                           dtype=np.int64)
        assert len(day_ord) > 0 and bool(np.all(np.diff(day_ord) > 0)), (
            "precompute_minute: días no estrictamente crecientes")
        merged0 = align_trailing_days(res, day_ord)[0]
        days = np.array([dt.date.fromordinal(int(o)) for o in day_ord])
        total_pnl = float(np.sum(lots * b.pnl_per_lot))
        out.append((days, merged0, total_pnl))
    return out


def _bound_motor_walk(block_order, blocks, open_days, lots_by_f_sym, pv_by_sym,
                      f_idx, *, pass_line, ruin_line, baseline, horizon_days):
    """Sub-pase BOUND con semántica EXACTA del motor (``evaluate_path``) +
    snapshot del primer cruce (con T(d) por apertura).

    Espejo línea a línea de ``evaluate_path`` (mismo orden de acumulación y de
    chequeos; ``_block_day_eval`` en vivo con el saldo real) ⇒ el ``raw_path``
    devuelto es motor-exacto (V4). Devuelve ``(PathOutcome, first_cross, fc)``.
    """
    saldo = 0.0
    days_seen = 0.0
    n_unique = 0
    peak = 0.0
    max_dd = 0.0
    last_merged = 0.0
    first_cross: str | None = None
    fc: dict = {}
    for bi in block_order:
        b = blocks[bi]
        od = open_days[bi]
        days, merged, total_pnl = _block_day_eval(b, lots_by_f_sym, pv_by_sym,
                                                  f_idx, saldo)
        t_prev = n_unique
        n_unique += b.n_unique_trades
        days_seen += b.duration_days
        if len(merged):
            cumpeak = np.maximum.accumulate(np.maximum(merged, peak))
            max_dd = min(max_dd, float(np.min(merged - cumpeak)))
            peak = float(cumpeak[-1])
            last_merged = float(merged[-1])
            for d, m in zip(days, merged):
                if first_cross is None:
                    cross = ("ruin" if m <= ruin_line
                             else "pass" if m >= pass_line else None)
                    if cross is not None:
                        first_cross = cross
                        fc = {"days": days_seen, "trades_block": n_unique,
                              "trades_day": t_prev + int(np.searchsorted(
                                  od, d.toordinal(), side="right")),
                              "max_dd": max_dd / baseline,
                              "terminal": float(m)}
                if m <= ruin_line:
                    return (PathOutcome("ruin", days_seen, n_unique,
                                        max_dd / baseline, float(m)),
                            first_cross, fc)
                if m >= pass_line:
                    return (PathOutcome("pass", days_seen, n_unique,
                                        max_dd / baseline, float(m)),
                            first_cross, fc)
                if horizon_days is not None and days_seen > horizon_days:
                    return (PathOutcome("timeout", days_seen, n_unique,
                                        max_dd / baseline, float(m)),
                            first_cross, fc)
        saldo += total_pnl
    return (PathOutcome("neither", days_seen, n_unique, max_dd / baseline,
                        last_merged), first_cross, fc)


def _evaluate_joint(block_order, blocks, open_days, joint_pre, *,
                    pass_line, ruin_line, baseline, min_unique_trades,
                    gap_mean, horizon_days, lots_by_f_sym, pv_by_sym, f_idx):
    """Rama JOINT del walker F4 (§8.6-1): decisiones (ruin→pass→timeout, regla
    doble, continuación a f constante) sobre la serie merged JOINT precomputada
    (``merged0 + saldo``); ``first_cross``/``fc_*``/``raw_path`` del sub-pase
    bound (semántica motor, inmutable entre convenciones).
    """
    raw_path, first_cross, fc = _bound_motor_walk(
        block_order, blocks, open_days, lots_by_f_sym, pv_by_sym, f_idx,
        pass_line=pass_line, ruin_line=ruin_line, baseline=baseline,
        horizon_days=horizon_days)
    saldo = 0.0
    days_seen = 0.0
    n_unique = 0
    peak = 0.0
    max_dd = 0.0
    last_merged = 0.0
    n_blocks = 0

    def _out(kind: str, t_day: int, m: float) -> PathOutcomeAXI:
        return PathOutcomeAXI(
            outcome=kind, days=days_seen, trades=t_day,
            max_dd=max_dd / baseline, terminal=float(m),
            trades_block=n_unique, n_blocks=n_blocks,
            days_cal=days_seen + (n_blocks - 1) * gap_mean,
            first_cross=first_cross,
            fc_days=fc.get("days", float("nan")),
            fc_trades_block=fc.get("trades_block", 0),
            fc_trades_day=fc.get("trades_day", 0),
            fc_max_dd=fc.get("max_dd", float("nan")),
            fc_terminal=fc.get("terminal", float("nan")),
            raw_path=raw_path)

    for bi in block_order:
        b = blocks[bi]
        n_blocks += 1
        od = open_days[bi]
        days, merged0, total_pnl = joint_pre[bi]
        t_prev = n_unique
        n_unique += b.n_unique_trades
        days_seen += b.duration_days
        if len(merged0):
            merged = merged0 + saldo
            cumpeak = np.maximum.accumulate(np.maximum(merged, peak))
            max_dd = min(max_dd, float(np.min(merged - cumpeak)))
            peak = float(cumpeak[-1])
            last_merged = float(merged[-1])
            for d, m in zip(days, merged):  # days en orden temporal
                t_day = t_prev + int(np.searchsorted(od, d.toordinal(),
                                                     side="right"))
                if m <= ruin_line:
                    return _out("ruin", t_day, m)
                if m >= pass_line:
                    if min_unique_trades is None or t_day >= min_unique_trades:
                        return _out("pass", t_day, m)
                if horizon_days is not None and days_seen > horizon_days:
                    return _out("timeout", t_day, m)
        saldo += total_pnl
    return _out("neither", n_unique, last_merged)


def evaluate_path_axi(block_order, blocks, open_days, lots_by_f_sym, pv_by_sym,
                      f_idx, *, pass_line: float, ruin_line: float,
                      baseline: float, min_unique_trades: int | None,
                      gap_mean: float,
                      horizon_days: float | None = None,
                      mtm: str = "bound",
                      joint_pre: list | None = None) -> PathOutcomeAXI:
    """Walker F4: recorre ``block_order`` como ``evaluate_path``, pero con
    regla doble (pass solo si ``m ≥ pass_line`` ∧ ``T(d) ≥ min_unique_trades``)
    y continuidad post-target a f constante.

    Con ``min_unique_trades is None`` degenera de forma bit-idéntica a
    ``evaluate_path`` (V1). ``open_days`` = salida de
    :func:`precompute_open_days`. Réplica la mecánica del motor paso a paso
    (mismo orden de acumulación y de chequeos) para que los paths fuera del
    cubo degradado sean bit-idénticos (V2).

    ``mtm`` (§8.6-1):
      - ``"bound"`` (default): serie merged de peor-punto-por-día — la variante
        de fidelidad (V1/V2/V4, raw y check_axi bloque 1).
      - ``"joint"``: decisiones sobre la serie merged JOINT precomputada
        (``joint_pre``, salida de :func:`precompute_joint` a este f);
        ``first_cross``/``fc_*``/``raw_path`` siguen en bound (sub-pase
        :func:`_bound_motor_walk`, motor-exacto). El walker expuesto es joint.
    """
    if mtm == "joint":
        if joint_pre is None:
            raise ValueError("mtm='joint' requiere joint_pre "
                             "(precompute_joint por bloque a este f)")
        return _evaluate_joint(
            block_order, blocks, open_days, joint_pre,
            pass_line=pass_line, ruin_line=ruin_line, baseline=baseline,
            min_unique_trades=min_unique_trades, gap_mean=gap_mean,
            horizon_days=horizon_days, lots_by_f_sym=lots_by_f_sym,
            pv_by_sym=pv_by_sym, f_idx=f_idx)
    if mtm != "bound":
        raise ValueError("mtm desconocido: " + repr(mtm))

    saldo = 0.0
    days_seen = 0.0
    n_unique = 0
    peak = 0.0
    max_dd = 0.0
    last_merged = 0.0
    n_blocks = 0
    first_cross: str | None = None
    fc: dict = {}

    def _snap(kind: str, t_day: int, m: float) -> None:
        nonlocal first_cross, fc
        if first_cross is None:
            first_cross = kind
            fc = {"days": days_seen, "trades_block": n_unique,
                  "trades_day": t_day, "max_dd": max_dd / baseline,
                  "terminal": float(m)}

    def _out(kind: str, t_day: int, m: float) -> PathOutcomeAXI:
        return PathOutcomeAXI(
            outcome=kind, days=days_seen, trades=t_day,
            max_dd=max_dd / baseline, terminal=float(m),
            trades_block=n_unique, n_blocks=n_blocks,
            days_cal=days_seen + (n_blocks - 1) * gap_mean,
            first_cross=first_cross,
            fc_days=fc.get("days", float("nan")),
            fc_trades_block=fc.get("trades_block", 0),
            fc_trades_day=fc.get("trades_day", 0),
            fc_max_dd=fc.get("max_dd", float("nan")),
            fc_terminal=fc.get("terminal", float("nan")))

    for bi in block_order:
        b = blocks[bi]
        n_blocks += 1
        od = open_days[bi]
        days, merged, total_pnl = _block_day_eval(b, lots_by_f_sym, pv_by_sym,
                                                  f_idx, saldo)
        t_prev = n_unique  # trades abiertos en bloques ya consumidos
        n_unique += b.n_unique_trades
        days_seen += b.duration_days
        if len(merged):
            cumpeak = np.maximum.accumulate(np.maximum(merged, peak))
            max_dd = min(max_dd, float(np.min(merged - cumpeak)))
            peak = float(cumpeak[-1])
            last_merged = float(merged[-1])
            for d, m in zip(days, merged):  # days en orden temporal
                t_day = t_prev + int(np.searchsorted(od, d.toordinal(),
                                                     side="right"))
                if m <= ruin_line:
                    _snap("ruin", t_day, m)
                    return _out("ruin", t_day, m)
                if m >= pass_line:
                    _snap("pass", t_day, m)
                    if min_unique_trades is None or t_day >= min_unique_trades:
                        return _out("pass", t_day, m)
                if horizon_days is not None and days_seen > horizon_days:
                    return _out("timeout", t_day, m)
        saldo += total_pnl
    return _out("neither", n_unique, last_merged)


def raw_outcome(o: PathOutcomeAXI) -> PathOutcome:
    """``PathOutcome`` con semántica del MOTOR (primer cruce): agregar sobre
    estos outcomes reproduce ``run_portfolio_mc`` con tolerancia cero (V4),
    sin recomputar paths.

    Modo joint: el sub-pase bound ya produjo el outcome motor-exacto
    (``o.raw_path``). Modo bound: se deriva del snapshot ``first_cross``; sin
    cruce (``first_cross is None``) el stop final ya coincide con el del motor
    (``neither``/``timeout`` con los mismos valores).
    """
    if o.raw_path is not None:
        return o.raw_path
    if o.first_cross is None:
        return PathOutcome(o.outcome, o.days, o.trades_block, o.max_dd,
                           o.terminal)
    return PathOutcome(o.first_cross, o.fc_days, o.fc_trades_block,
                       o.fc_max_dd, o.fc_terminal)


def post_target_split(outcomes: list[PathOutcomeAXI],
                      min_unique_trades: int | None) -> dict:
    """Reparto del cubo degradado (caso 2 de §4): paths con primer cruce
    ``pass`` y T(d) por apertura < suelo, clasificados por su destino F4 final
    (2a → ruin, 2b → pass, 2c → sin resolver). Identidades exactas (V3):
    ``n_deg = n_ruin + n_pass + n_unresolved``,
    ``n_deg = n·(p_pass_raw − p_pass_suelo)``.
    """
    if min_unique_trades is None:
        return {"n_deg": 0, "n_ruin": 0, "n_pass": 0, "n_unresolved": 0}
    deg = [o for o in outcomes
           if o.first_cross == "pass" and o.fc_trades_day < min_unique_trades]
    n_ruin = sum(1 for o in deg if o.outcome == "ruin")
    n_pass = sum(1 for o in deg if o.outcome == "pass")
    return {"n_deg": len(deg), "n_ruin": n_ruin, "n_pass": n_pass,
            "n_unresolved": len(deg) - n_ruin - n_pass}
