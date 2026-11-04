"""Tests Fase A (plan float-valleys-ventanas) — sqx/sqx_trades.py.

- A1 GOLD sintético: writer inverso del formato .sqx (bloques 0x7A/0x77, registros
  de 149 B) con un largo, un corto (SL por encima) y un registro de otro test
  (``CrossCheck_1``, con valores que romperían las validaciones si no se filtrara).
  Valores elegidos a mano, representables exactamente en float32.
- A2 Fail-loud: un test por regla.
- A3 Integración real (skip si no existe el banco del árbol principal): 65/65
  ficheros, 131.076 trades, comisión/lote 1,12 ± 0,02, 0 violaciones de MAE.
"""
from __future__ import annotations

import hashlib
import struct
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from sqx.d1.extract_sqx import SQXFormatError, _depacketize, _parse_orders_full
from sqx.sqx_trades import (
    DEFAULT_CFD_COSTS_PATH,
    DEFAULT_COSTS_PATH,
    MM_CLASSES,
    SQXFutureLegError,
    costs_path_for_symbol,
    main,
    read_backtest_mm,
    read_instrument_info,
    read_trade_table,
    read_trade_table_spec,
)

FMT = b"SQOrderFileFormat:11"
REC = 149
REAL_BANK = Path(
    "/nonexistent/private-data"
)
PORTFOLIO_XML = b"<Strategy><Rules/></Strategy>"


# --------------------------------------------------------------------------
# writer inverso del formato .sqx (patrón de tests/test_float_valleys.py)
# --------------------------------------------------------------------------
def _packetize(payload: bytes) -> bytes:
    out = bytearray(b"\xac\xed\x00\x05")
    off = 0
    while off < len(payload):
        chunk = payload[off : off + 1024]
        if len(chunk) == 1024:
            out += b"\x7a" + struct.pack(">I", len(chunk)) + chunk
        elif len(chunk) < 256:
            out += b"\x77" + bytes([len(chunk)]) + chunk
        else:
            out += b"\x77" + struct.pack(">H", len(chunk)) + chunk
        off += len(chunk)
    return bytes(out)


def _ms(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def _rec(t: dict) -> bytes:
    rec = bytearray(REC)
    rec[0] = 4
    rec[1] = t.get("test_idx", 2)
    rec[2] = 3
    rec[3] = t.get("flag", 1)
    rec[23] = 6
    # [15:23] colocación de la orden ("order", por defecto = "open": orden a mercado); [32:40] ejecución.
    struct.pack_into(">q", rec, 15, _ms(t.get("order", t["open"])))
    struct.pack_into(">f", rec, 24, t["lots"])
    struct.pack_into(">f", rec, 28, t["open_px"])
    struct.pack_into(">q", rec, 32, _ms(t["open"]))
    struct.pack_into(">q", rec, 44, _ms(t["close"]))
    struct.pack_into(">f", rec, 52, t["close_px"])
    struct.pack_into(">f", rec, 56, t["sl_px"])
    struct.pack_into(">f", rec, 66, t["pnl"])
    struct.pack_into(">f", rec, 99, t["mae"])
    rec[98] = 1
    rec[135] = 1
    rec[144] = 0xFF
    return bytes(rec)


def _settings(symbol: str, instrument: str, pv: float, tick: float, precision: int,
              capital: float) -> bytes:
    return (
        '<?xml version="1.0"?><ResultsGroup ResultName="Strategy 9.9.9">'
        "<ResultsMap><Results>"
        f'<Result resultKey="Main: {symbol}/M30" special="false">'
        f'<ValuesMap><Symbol type="String">{symbol}</Symbol></ValuesMap>'
        '<SettingsMap><StrategyName type="String">Strategy 9.9.9</StrategyName>'
        f'<BacktestPrecision type="Integer">{precision}</BacktestPrecision>'
        f'<MoneyManagement.InitialCapital type="Double">{capital}</MoneyManagement.InitialCapital>'
        "</SettingsMap></Result>"
        "</Results></ResultsMap>"
        f'<SymbolsMap><SymbolInfo symbolName="{symbol}" instrumentName="{instrument}">'
        f'<InstrumentInfo instrument="{instrument}" tickSize="{tick}" pointValue="{pv}" />'
        "</SymbolInfo></SymbolsMap></ResultsGroup>"
    ).encode()


def write_sqx(
    path: Path,
    trades: list[dict],
    *,
    symbol: str = "@micro_nasdaq",
    instrument: str = "NQ_ftmo",
    pv: float = 2.0,
    tick: float = 0.25,
    precision: int = 1,
    capital: float = 50000.0,
    with_main_label: bool = True,
    payload_hook=None,
    portfolio_xml: str | None = None,
    last_settings: str | None = None,
) -> Path:
    """Escribe un ``.sqx`` sintético. ``portfolio_xml`` sustituye al ``strategy_Portfolio.xml`` fijo
    (``PORTFOLIO_XML``) y ``last_settings`` añade un ``lastSettings.xml`` (el MM del backtest). Con ``None`` (los
    valores por defecto) el ZIP es el de siempre: mismas entradas, mismo orden, SIN ``lastSettings.xml``."""
    strings = [b"", b"Main: " + symbol.encode() + b"/M30" if with_main_label else b"Other",
               b"Strategy 9.9.9", symbol.encode(), b"CrossCheck_1"]
    payload = struct.pack(">H", len(FMT)) + FMT + b"\x00" * 24
    payload += struct.pack(">ii", 1, len(strings))
    for s in strings:
        payload += struct.pack(">H", len(s)) + s
    for t in trades:
        payload += _rec(t)
    if payload_hook is not None:  # p.ej. truncar el payload para tests de robustez
        payload = payload_hook(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("settings.xml", _settings(symbol, instrument, pv, tick, precision, capital))
        zf.writestr("orders.bin", _packetize(payload))
        zf.writestr("strategy_Portfolio.xml", PORTFOLIO_XML if portfolio_xml is None else portfolio_xml.encode())
        if last_settings is not None:
            zf.writestr("lastSettings.xml", last_settings)
    return path


def patch_instrument_info(path: Path, **attrs: str) -> Path:
    """Añade atributos al ``InstrumentInfo`` del ``settings.xml`` de un ``.sqx`` sintético (``orderSizeStep``,
    ``defaultSpread``, ``commissions``…), reescribiendo el ZIP. Los valores se escapan como los escribe SQX
    (``commissions`` lleva el XML de los métodos dentro del atributo)."""
    from xml.sax.saxutils import quoteattr
    with zipfile.ZipFile(path) as zf:
        items = [(i.filename, zf.read(i.filename)) for i in zf.infolist()]
    out = []
    for name, data in items:
        if name == "settings.xml":
            text = data.decode()
            extra = "".join(f" {k}={quoteattr(v)}" for k, v in attrs.items())
            assert text.count(" />", text.index("<InstrumentInfo")) >= 1
            head, tail = text.split("<InstrumentInfo", 1)
            body, rest = tail.split(" />", 1)
            data = (head + "<InstrumentInfo" + body + extra + " />" + rest).encode()
        out.append((name, data))
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in out:
            zf.writestr(name, data)
    return path


def long_win(**kw) -> dict:
    """Largo ganador: +10 pts * 2 USD/pt = 20 bruto; comisión 1,125 => neto 18,875."""
    t = dict(open=datetime(2024, 3, 4, 10, 0), close=datetime(2024, 3, 5, 11, 30), lots=1.0,
             open_px=18000.0, close_px=18010.0, sl_px=17990.0, pnl=18.875, mae=25.0, flag=1)
    t.update(kw)
    return t


def short_loss(**kw) -> dict:
    """Corto perdedor, 2 lotes: +5 pts en contra * 2 * 2 = 20 bruto; comisión 2,25 => -22,25."""
    t = dict(open=datetime(2024, 3, 4, 14, 0), close=datetime(2024, 3, 4, 15, 0), lots=2.0,
             open_px=18000.0, close_px=18005.0, sl_px=18020.0, pnl=-22.25, mae=30.0, flag=2)
    t.update(kw)
    return t


def crosscheck(**kw) -> dict:
    """Registro de otro test, con valores que violarían varias reglas si no se filtrase."""
    t = dict(open=datetime(2024, 3, 6, 9, 0), close=datetime(2024, 3, 6, 9, 30), lots=0.0,
             open_px=18000.0, close_px=18000.0, sl_px=18000.0, pnl=9999.0, mae=0.0, flag=1,
             test_idx=5)
    t.update(kw)
    return t


# --------------------------------------------------------------------------
# A1 — golden sintético
# --------------------------------------------------------------------------
def test_a1_golden_tabla_y_meta(tmp_path):
    p = write_sqx(tmp_path / "g.sqx", [long_win(), short_loss(), crosscheck()])
    df, meta = read_trade_table(p)

    assert list(df.columns) == ["open_ts", "close_ts", "lots", "pnl_net", "mae_usd", "open_px",
                                "close_px", "sl_px", "is_buy", "comm_usd", "dir_flag", "order_ts"]
    assert (df["order_ts"] == df["open_ts"]).all() and meta["frac_pending_entry"] == 0.0  # a mercado
    assert len(df) == 2  # el CrossCheck se filtra
    # Orden por close_ts: el corto (04-mar 15:00) antes que el largo (05-mar 11:30).
    assert df["close_ts"].dtype == np.dtype("datetime64[ns]")
    assert df["open_ts"].dt.tz is None and df["close_ts"].dt.tz is None
    assert list(df["close_ts"]) == [pd.Timestamp("2024-03-04 15:00"), pd.Timestamp("2024-03-05 11:30")]
    assert list(df["open_ts"]) == [pd.Timestamp("2024-03-04 14:00"), pd.Timestamp("2024-03-04 10:00")]

    exp = {
        "lots": [2.0, 1.0],
        "pnl_net": [-22.25, 18.875],
        "mae_usd": [30.0, 25.0],
        "open_px": [18000.0, 18000.0],
        "close_px": [18005.0, 18010.0],
        "sl_px": [18020.0, 17990.0],
        # comm = (close-open)*signo*pv*lots - pnl_net
        #   corto: 5*(-1)*2*2 - (-22.25) = -20 + 22.25 = 2.25 ; largo: 10*2*1 - 18.875 = 1.125
        "comm_usd": [2.25, 1.125],
    }
    for col, vals in exp.items():
        np.testing.assert_allclose(df[col].to_numpy(), vals, atol=1e-6, err_msg=col)
    assert list(df["is_buy"]) == [False, True]
    assert df["is_buy"].dtype == bool
    assert list(df["dir_flag"]) == [2, 1]
    assert df["dir_flag"].dtype == np.uint8

    assert meta["strategy_name"] == "Strategy 9.9.9"
    assert meta["symbol"] == "@micro_nasdaq"  # symbolName, NO instrumentName (V1)
    assert meta["pv"] == 2.0
    assert meta["tick_size"] == 0.25
    assert meta["initial_capital"] == 50000.0
    assert meta["backtest_precision"] == 1
    assert meta["sha1"] == hashlib.sha1(PORTFOLIO_XML).hexdigest()
    assert meta["n_trades"] == 2
    # riesgo inicial: corto 20*2*2 = 80 (MAE 30 no lo supera); largo 10*2 = 20 (MAE 25 > 20,5)
    assert meta["frac_mae_gt_initial_risk"] == pytest.approx(0.5)
    # comisión 1,125/lote vs YAML 1,12: |dif| 0,005 <= 0,02 => sin aviso de coste
    assert not any("total_round_trip" in w for w in meta["warnings"])
    # flag 2 (no observado): solo se registra. Los cortos ya no avisan: el lado sale
    # del SL y está verificado con cortos reales (riesgo R1 cerrado el 2026-10-12).
    assert any("flag" in w for w in meta["warnings"])
    assert not any("cortos" in w for w in meta["warnings"])
    assert meta["asset_class"] == "futures"
    assert "swap_usd" not in df.columns


def test_a1_warning_comision_distinta_del_yaml(tmp_path):
    # comisión 1,50/lote (constante) => no es error, sí warning vs total_round_trip 1,12
    t = long_win(pnl=20.0 - 1.5)
    df, meta = read_trade_table(write_sqx(tmp_path / "c.sqx", [t]))
    np.testing.assert_allclose(df["comm_usd"], [1.5], atol=1e-6)
    assert any("total_round_trip" in w for w in meta["warnings"])


def test_yaml_sqx_symbol_coherente_con_sqx_policy():
    costs = yaml.safe_load(DEFAULT_COSTS_PATH.read_text(encoding="utf-8"))
    mapping = {k: v["sqx_symbol"] for k, v in costs["symbols"].items()}
    assert mapping == {"MNQ": "@micro_nasdaq", "M2K": "@micro_rusell",
                       "MGC": "@micro_oro", "M6E": "@micro_eurusd"}
    assert set(mapping.values()) == set(costs["sqx_policy"])


# --------------------------------------------------------------------------
# A2 — fail-loud, un test por regla
# --------------------------------------------------------------------------
def test_a2_mae_menor_que_perdida_bruta(tmp_path):
    # largo perdedor: bruto -20 (10 pts * 2); MAE 10 < 20 - tolerancia
    t = long_win(close_px=17990.0, sl_px=17980.0, pnl=-21.125, mae=10.0)
    with pytest.raises(SQXFormatError, match="MAE menor"):
        read_trade_table(write_sqx(tmp_path / "m.sqx", [t]))


def test_a2_mae_igual_a_perdida_bruta_ok(tmp_path):
    # stop exacto: MAE == pérdida bruta (74% de perdedoras reales) => válido
    t = long_win(close_px=17990.0, sl_px=17990.0 - 0.25, pnl=-21.125, mae=20.0)
    df, meta = read_trade_table(write_sqx(tmp_path / "ok.sqx", [t]))
    assert len(df) == 1 and bool(df["is_buy"][0])


def test_a2_sl_igual_a_open(tmp_path):
    with pytest.raises(SQXFormatError, match="sl_px == open_px"):
        read_trade_table(write_sqx(tmp_path / "s.sqx", [long_win(sl_px=18000.0)]))


def test_a2_simbolo_fuera_de_allowlist(tmp_path):
    p = write_sqx(tmp_path / "x.sqx", [long_win()], symbol="NAS100", instrument="NAS100")
    with pytest.raises(SQXFormatError, match="allowlist"):
        read_trade_table(p)


def test_a2_allowlist_compara_symbolname_no_instrumentname(tmp_path):
    # instrumentName "NQ_ftmo" no está en la allowlist; symbolName sí => OK (enmienda V1)
    p = write_sqx(tmp_path / "v1.sqx", [long_win()], symbol="@micro_nasdaq", instrument="NQ_ftmo")
    assert read_trade_table(p)[1]["symbol"] == "@micro_nasdaq"


def test_a2_pv_incoherente_con_yaml(tmp_path):
    p = write_sqx(tmp_path / "pv.sqx", [long_win()], pv=5.0)
    with pytest.raises(SQXFormatError, match="pointValue"):
        read_trade_table(p)


def test_flag1_con_sl_por_encima_es_un_corto(tmp_path):
    """El byte [3] no es la dirección: los cortos reales llevan flag 1 (8.467 cortos
    CFD EURUSD, 2026-10-12). Con el SL por encima es un corto, sin error."""
    t = short_loss(flag=1)
    df, meta = read_trade_table(write_sqx(tmp_path / "f.sqx", [t]))
    assert list(df["is_buy"]) == [False]
    np.testing.assert_allclose(df["comm_usd"], [2.25], atol=1e-6)
    assert not any("flag" in w for w in meta["warnings"])


def test_a2_lotes_no_positivos(tmp_path):
    with pytest.raises(SQXFormatError, match="lotes <= 0"):
        read_trade_table(write_sqx(tmp_path / "l.sqx", [long_win(lots=0.0)]))


def test_a2_mae_negativo(tmp_path):
    # ganador con MAE -5 (< -0,011*lotes): el MAE es >= 0
    with pytest.raises(SQXFutureLegError, match="MAE negativo"):
        read_trade_table(write_sqx(tmp_path / "mn.sqx", [long_win(mae=-5.0)]))
    with pytest.raises(SQXFutureLegError, match="MAE negativo"):
        read_trade_table(write_sqx(tmp_path / "mn1.sqx", [long_win(mae=-0.001)]))


def test_a2_mae_negativo_es_estricto(tmp_path):
    # sin tolerancia: -0,005 (dentro de la antigua banda de 0,011) también falla
    with pytest.raises(SQXFutureLegError, match="MAE negativo"):
        read_trade_table(write_sqx(tmp_path / "mn2.sqx", [long_win(mae=-0.005)]))
    # MAE == 0 exacto es válido
    df, _ = read_trade_table(write_sqx(tmp_path / "mn0.sqx", [long_win(mae=0.0)]))
    assert df["mae_usd"][0] == 0.0


def test_a2_comision_ruido_negativo_se_fija_a_cero_con_warning(tmp_path):
    # backtest sin comisión: pnl_net = bruto + 0,01 (ruido) => comm = -0,01 en [-0,02*lots, 0)
    t = long_win(pnl=20.0 + 0.01)
    df, meta = read_trade_table(write_sqx(tmp_path / "cz.sqx", [t]))
    assert df["comm_usd"][0] == 0.0
    assert (df["comm_usd"] >= 0).all()
    assert any("fijada a 0.0 en 1 trades" in w for w in meta["warnings"])
    # comm exactamente 0 no genera el aviso
    df0, meta0 = read_trade_table(write_sqx(tmp_path / "c0.sqx", [long_win(pnl=20.0)]))
    assert df0["comm_usd"][0] == 0.0
    assert not any("fijada a 0.0" in w for w in meta0["warnings"])


def test_a2_comision_menos_0001_a_cero_con_warning(tmp_path):
    df, meta = read_trade_table(write_sqx(tmp_path / "cm1.sqx", [long_win(pnl=20.001)]))
    assert df["comm_usd"][0] == 0.0
    assert any("fijada a 0.0 en 1 trades" in w for w in meta["warnings"])


def test_a2_comision_menos_05_por_lote_es_error(tmp_path):
    with pytest.raises(SQXFutureLegError, match="comisión implícita negativa"):
        read_trade_table(write_sqx(tmp_path / "cm5.sqx", [long_win(pnl=20.5)]))


def test_a2_comision_negativa_bajo_umbral_es_error(tmp_path):
    # comm = -0,05 (2 lotes: umbral -0,04) y -0,03 (1 lote: umbral -0,02)
    with pytest.raises(SQXFutureLegError, match="comisión implícita negativa"):
        read_trade_table(write_sqx(tmp_path / "cn1.sqx", [long_win(pnl=20.0 + 0.03)]))
    t2 = short_loss(pnl=-20.0 + 0.05)
    with pytest.raises(SQXFutureLegError, match="comisión implícita negativa"):
        read_trade_table(write_sqx(tmp_path / "cn2.sqx", [t2]))


def test_a2_sl_cero(tmp_path):
    with pytest.raises(SQXFutureLegError, match="sl_px <= 0"):
        read_trade_table(write_sqx(tmp_path / "sl0.sqx", [long_win(sl_px=0.0)]))


@pytest.mark.parametrize("campo", ["open_px", "close_px"])
def test_a2_precios_open_close_no_positivos(tmp_path, campo):
    with pytest.raises(SQXFutureLegError, match=f"{campo} <= 0"):
        read_trade_table(write_sqx(tmp_path / "p0.sqx", [long_win(**{campo: 0.0})]))


def _header_len() -> int:
    # 2 (uint16 len) + len(FMT) + 24 reservados + 8 (n1, n_strings)
    return 2 + len(FMT) + 24 + 8


def test_a2_payload_truncado_en_tabla_de_strings(tmp_path):
    # corta a mitad del primer string de la tabla: antes escapaba struct.error/basura
    p = write_sqx(tmp_path / "t1.sqx", [long_win()],
                  payload_hook=lambda b: b[: _header_len() + 2 + 0])  # sin longitud completa
    with pytest.raises(SQXFutureLegError, match="truncado"):
        read_trade_table(p)
    # longitud del string presente pero string cortado a mitad
    p = write_sqx(tmp_path / "t2.sqx", [long_win()],
                  payload_hook=lambda b: b[: _header_len() + 2 + 3])
    with pytest.raises(SQXFutureLegError, match="truncado"):
        read_trade_table(p)


def test_a2_payload_truncado_en_cabecera(tmp_path):
    for n in (1, 5, 2 + len(FMT) + 10, _header_len() - 3):
        p = write_sqx(tmp_path / f"h{n}.sqx", [long_win()], payload_hook=lambda b, n=n: b[:n])
        with pytest.raises(SQXFutureLegError):
            read_trade_table(p)


def test_a2_cuerpo_no_multiplo_de_registro(tmp_path):
    p = write_sqx(tmp_path / "bm.sqx", [long_win()], payload_hook=lambda b: b[:-7])
    with pytest.raises(SQXFutureLegError, match="no múltiplo"):
        read_trade_table(p)


def test_a2_yaml_sin_clave_de_coste(tmp_path):
    costs = yaml.safe_load(DEFAULT_COSTS_PATH.read_text(encoding="utf-8"))
    del costs["symbols"]["MNQ"]["tick_value_usd"]
    cp = tmp_path / "costs_roto.yaml"
    cp.write_text(yaml.safe_dump(costs), encoding="utf-8")
    with pytest.raises(SQXFutureLegError, match="tick_value_usd"):
        read_trade_table(write_sqx(tmp_path / "y.sqx", [long_win()]), costs_path=cp)
    costs["symbols"]["MNQ"]["tick_value_usd"] = 0.5
    del costs["symbols"]["MNQ"]["total_round_trip"]
    cp = tmp_path / "costs_roto2.yaml"  # load_costs_policy cachea por ruta: otro fichero
    cp.write_text(yaml.safe_dump(costs), encoding="utf-8")
    with pytest.raises(SQXFutureLegError, match="total_round_trip"):
        read_trade_table(write_sqx(tmp_path / "y2.sqx", [long_win()]), costs_path=cp)


def test_fallback_sin_etiqueta_main_un_solo_test_id(tmp_path):
    # sin string "Main:" y un único test id => todo vale
    p = write_sqx(tmp_path / "nm.sqx", [long_win(), short_loss()], with_main_label=False)
    df, meta = read_trade_table(p)
    assert len(df) == 2 and meta["n_trades"] == 2


def test_fallback_sin_etiqueta_main_varios_test_ids_falla(tmp_path):
    p = write_sqx(tmp_path / "nm2.sqx",
                  [long_win(), long_win(test_idx=5)], with_main_label=False)
    with pytest.raises(SQXFutureLegError, match="varios test ids"):
        read_trade_table(p)


def test_a2_ningun_trade_main(tmp_path):
    p = write_sqx(tmp_path / "n.sqx", [crosscheck(lots=1.0, sl_px=17990.0, pnl=18.875)])
    with pytest.raises(SQXFormatError, match="ningún trade Main"):
        read_trade_table(p)


def test_a2_comision_por_lote_no_constante(tmp_path):
    # segundo trade con comisión 3,0/lote frente a 1,125 => desviación > 0,02
    t2 = short_loss(pnl=-20.0 - 3.0 * 2)
    with pytest.raises(SQXFormatError, match="no constante"):
        read_trade_table(write_sqx(tmp_path / "k.sqx", [long_win(), t2]))


def test_a2_errores_son_subclase_de_sqxformaterror():
    assert issubclass(SQXFutureLegError, SQXFormatError)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def test_cli_bank_parquet_y_fallos_listados(tmp_path, capsys):
    bank = tmp_path / "bank"
    write_sqx(bank / "a.sqx", [long_win(), short_loss()])
    write_sqx(bank / "b.sqx", [long_win()])
    write_sqx(bank / "bad.sqx", [long_win(sl_px=18000.0)])  # SL == open
    out = tmp_path / "out" / "trades.parquet"
    rc = main(["--bank", str(bank), "--out", str(out)])
    err = capsys.readouterr().err
    assert rc == 1  # hay fallos: nunca silencioso
    assert "bad.sqx" in err and "sl_px == open_px" in err
    allt = pd.read_parquet(out)
    assert len(allt) == 3
    assert set(allt["file"]) == {"a.sqx", "b.sqx"}
    assert (allt["sha1"] == hashlib.sha1(PORTFOLIO_XML).hexdigest()).all()
    meta = pd.read_parquet(tmp_path / "out" / "trades_meta.parquet")
    assert sorted(meta["file"]) == ["a.sqx", "b.sqx"]
    assert set(meta["n_trades"]) == {1, 2}


def test_cli_bank_sin_fallos_rc0(tmp_path):
    bank = tmp_path / "bank"
    write_sqx(bank / "a.sqx", [long_win()])
    assert main(["--bank", str(bank), "--out", str(tmp_path / "t.parquet")]) == 0


# --------------------------------------------------------------------------
# A3 — integración real (paridad externa con extract_sqx._parse_orders_full)
# --------------------------------------------------------------------------
@pytest.mark.skipif(not REAL_BANK.exists(), reason="banco real MNQ M30 no disponible")
def test_a3_banco_real_mnq_m30():
    files = sorted(REAL_BANK.glob("*.sqx"))
    assert len(files) == 65
    total = 0
    n_gt = 0.0
    n_pend = 0
    for f in files:
        df, meta = read_trade_table(f)  # 65/65 sin error (0 violaciones de MAE incluidas)
        total += len(df)
        assert meta["symbol"] == "@micro_nasdaq", f.name
        assert meta["pv"] == 2.0, f.name
        assert df["is_buy"].all(), f.name
        assert (df["dir_flag"] == 1).all(), f.name
        cpl = (df["comm_usd"] / df["lots"]).to_numpy()
        assert np.abs(cpl - 1.12).max() <= 0.02, f.name
        # close >= open (hay trades de duración 0: entrada y salida en la misma barra)
        assert (df["close_ts"] >= df["open_ts"]).all(), f.name
        gross = df["pnl_net"] + df["comm_usd"]
        assert (df["mae_usd"] >= -gross - 0.011 * df["lots"]).all(), f.name
        n_gt += meta["frac_mae_gt_initial_risk"] * len(df)
        assert (df["open_ts"] >= df["order_ts"]).all(), f.name
        n_pend += int((df["open_ts"] > df["order_ts"]).sum())
    assert total == 131_076
    assert n_pend == 13_936  # 10,6% entran con orden pendiente (medido 2026-10-14)
    # informativo (plan §0: 0,034% de los trades con MAE > riesgo inicial + $0,5)
    assert n_gt / total < 0.001


@pytest.mark.skipif(not REAL_BANK.exists(), reason="banco real MNQ M30 no disponible")
def test_a3_paridad_con_parse_orders_full():
    """nº de trades Main y pnl_net por trade idénticos al parser independiente
    ``_parse_orders_full`` (emparejado por (hora de colocación, cierre): ese parser lee [15:23])."""
    files = sorted(REAL_BANK.glob("*.sqx"))[:: max(1, 65 // 3)][:3]
    assert len(files) == 3
    for f in files:
        df, meta = read_trade_table(f)
        with zipfile.ZipFile(f) as zf:
            ref = _parse_orders_full(_depacketize(zf.read("orders.bin")))
        assert len(df) == len(ref) == meta["n_trades"], f.name
        open_ms = df["order_ts"].astype("int64").to_numpy() // 1_000_000  # ns -> ms
        close_ms = df["close_ts"].astype("int64").to_numpy() // 1_000_000
        mine: dict[tuple[int, int], list[float]] = {}
        for o, c, pnl in zip(open_ms, close_ms, df["pnl_net"].to_numpy()):
            mine.setdefault((int(o), int(c)), []).append(float(pnl))
        for o, c, _lots, pnl in ref:
            cands = mine[(o, c)]
            j = int(np.argmin([abs(x - pnl) for x in cands]))
            assert cands.pop(j) == pytest.approx(pnl, abs=1e-6), f.name
        assert all(not v for v in mine.values()), f.name


# --------------------------------------------------------------------------
# Fin del backtest y tolerancia del MAE con lotes fraccionarios (2026-10-12)
# --------------------------------------------------------------------------
def test_trade_abierto_al_final_del_backtest_se_descarta(tmp_path):
    """Último cierre con PnL exactamente 0 y precio movido = trade que SQX deja
    abierto al final del test: se descarta con aviso."""
    eob = long_win(open=datetime(2024, 3, 8, 10, 0), close=datetime(2024, 3, 8, 20, 0),
                   close_px=17980.0, pnl=0.0, mae=40.0)
    df, meta = read_trade_table(write_sqx(tmp_path / "e.sqx", [long_win(), eob]))
    assert len(df) == 1 and meta["n_trades"] == 1
    assert any("final del backtest" in w for w in meta["warnings"])


# --------------------------------------------------------------------------
# Hora de ejecución [32:40] frente a hora de colocación [15:23] (2026-10-14)
# --------------------------------------------------------------------------
def test_open_ts_es_la_ejecucion_y_order_ts_la_colocacion(tmp_path):
    """Orden stop colocada a las 08:00 y ejecutada a las 10:00: el trade abre a las 10:00."""
    pend = long_win(order=datetime(2024, 3, 4, 8, 0))
    df, meta = read_trade_table(write_sqx(tmp_path / "p.sqx", [pend, short_loss()]))
    largo = df[df["is_buy"]].iloc[0]
    assert largo["open_ts"] == pd.Timestamp("2024-03-04 10:00")
    assert largo["order_ts"] == pd.Timestamp("2024-03-04 08:00")
    assert df["order_ts"].dtype == np.dtype("datetime64[ns]")
    assert meta["frac_pending_entry"] == 0.5


def test_ejecucion_anterior_a_la_colocacion_es_error(tmp_path):
    with pytest.raises(SQXFutureLegError, match="horas incoherentes"):
        read_trade_table(write_sqx(tmp_path / "e.sqx", [long_win(order=datetime(2024, 3, 4, 11, 0))]))


def test_cierre_anterior_a_la_ejecucion_es_error(tmp_path):
    t = long_win(open=datetime(2024, 3, 5, 12, 0), order=datetime(2024, 3, 4, 9, 0))  # cierre 05-mar 11:30
    with pytest.raises(SQXFutureLegError, match="horas incoherentes"):
        read_trade_table(write_sqx(tmp_path / "c.sqx", [t]))


def test_pnl_cero_que_no_es_el_ultimo_cierre_se_conserva(tmp_path):
    # bruto = comisión (0,5625 pts · 2 USD = 1,125) → PnL neto 0 legítimo, y no es el
    # último cierre: se conserva.
    t0 = long_win(close_px=18000.5625, pnl=0.0, mae=5.0)
    df, _ = read_trade_table(write_sqx(tmp_path / "z.sqx", [t0, long_win(
        open=datetime(2024, 3, 6, 10, 0), close=datetime(2024, 3, 6, 11, 0))]))
    assert len(df) == 2


# --------------------------------------------------------------------------
# CFD FTMO (config/ftmo_cfd_costs.yaml)
# --------------------------------------------------------------------------
CFD_PV = 100000.0


def cfd_short(day: int, *, hour=10, close_hour=12, nights=0, lots=0.25,
              open_px=1.125, move=0.00390625, win=False, swap=0.0, comm_lot=5.0, mae=None):
    """Corto EURUSD CFD con precios binarios exactos en float32. ``move`` en precio
    (0,00390625 = 39,0625 pips). ``swap`` = USD totales (positivo = abono)."""
    close_px = open_px - move if win else open_px + move
    gross = (open_px - close_px) * CFD_PV * lots
    o = datetime(2024, 3, day, hour, 0)
    c = datetime(2024, 3, day + nights, close_hour, 0)
    return dict(open=o, close=c, lots=lots, open_px=open_px, close_px=close_px,
                sl_px=open_px + 2 * move, pnl=gross - comm_lot * lots + swap,
                mae=(-gross if not win else 10.0) if mae is None else mae, flag=1)


def write_cfd(path, trades):
    return write_sqx(path, trades, symbol="EURUSD_ftmo", instrument="EURUSD_cfd_ftmo",
                     pv=CFD_PV, tick=0.0001)


def test_costs_path_for_symbol_elige_el_yaml_por_allowlist():
    assert costs_path_for_symbol("@micro_nasdaq") == DEFAULT_COSTS_PATH
    assert costs_path_for_symbol("EURUSD_ftmo") == DEFAULT_CFD_COSTS_PATH
    assert costs_path_for_symbol("XYZ") is None and costs_path_for_symbol(None) is None


def test_cfd_comision_estimada_y_swap_separado(tmp_path):
    intradia = [cfd_short(d, win=(d % 2 == 0)) for d in (4, 5, 6, 7, 8)]
    noches = cfd_short(11, nights=2, swap=0.10 * 2 * 0.25)      # abono corto 0,10/lote/noche
    df, meta = read_trade_table(write_cfd(tmp_path / "c.sqx", intradia + [noches]),
                                costs_path=DEFAULT_CFD_COSTS_PATH)
    assert meta["asset_class"] == "cfd" and meta["lot_step"] == 0.01 and meta["min_lot"] == 0.01
    assert meta["commission_per_lot"] == pytest.approx(5.0, abs=1e-6)
    assert list(df.columns)[-1] == "swap_usd"
    np.testing.assert_allclose(df["comm_usd"], 5.0 * df["lots"], atol=1e-6)
    np.testing.assert_allclose(df["swap_usd"].iloc[-1], 0.05, atol=1e-4)
    np.testing.assert_allclose(df["swap_usd"].iloc[:-1], 0.0, atol=1e-4)
    assert not df["is_buy"].any()
    assert not any("total_round_trip" in w for w in meta["warnings"])


def test_cfd_comision_no_constante_sin_rollover_es_error(tmp_path):
    tr = [cfd_short(d) for d in (4, 5, 6, 7)] + [cfd_short(8, comm_lot=7.0)]
    with pytest.raises(SQXFutureLegError, match="sin rollover"):
        read_trade_table(write_cfd(tmp_path / "n.sqx", tr), costs_path=DEFAULT_CFD_COSTS_PATH)


def test_cfd_pocos_trades_sin_rollover_usa_el_yaml_con_aviso(tmp_path):
    tr = [cfd_short(4), cfd_short(5, nights=1, swap=0.025)]
    df, meta = read_trade_table(write_cfd(tmp_path / "p.sqx", tr), costs_path=DEFAULT_CFD_COSTS_PATH)
    assert meta["commission_per_lot"] == 5.0
    assert any("sin rollover" in w for w in meta["warnings"])
    np.testing.assert_allclose(df["swap_usd"], [0.0, 0.025], atol=1e-4)


def test_cfd_trade_que_cruza_las_23_no_cuenta_como_sin_rollover(tmp_path):
    """Un trade del mismo día que cruza las 23:00 lleva swap: no entra en la
    estimación de la comisión (si entrara, la desviación sería error)."""
    tr = [cfd_short(d) for d in (4, 5, 6, 7, 8)]
    tr.append(cfd_short(11, hour=20, close_hour=23, swap=-0.5))   # cargo de swap
    df, meta = read_trade_table(write_cfd(tmp_path / "r.sqx", tr), costs_path=DEFAULT_CFD_COSTS_PATH)
    assert meta["commission_per_lot"] == pytest.approx(5.0, abs=1e-6)
    np.testing.assert_allclose(df["swap_usd"].iloc[-1], -0.5, atol=1e-4)


def test_cfd_orden_colocada_antes_de_las_23_y_ejecutada_despues_cuenta_como_sin_rollover(tmp_path):
    """Lo que manda para el swap es la hora de EJECUCIÓN: una orden pendiente colocada a las 20:00 que se
    ejecuta a las 23:10 y cierra a las 23:40 no pasa ninguna noche (2026-10-14). Con 4 trades más, solo hay
    5 sin rollover (los mínimos para estimar la comisión) si este cuenta."""
    tr = [cfd_short(d) for d in (4, 5, 6, 7)]
    t = cfd_short(11, hour=23, close_hour=23)
    t.update(order=datetime(2024, 3, 11, 20, 0), open=datetime(2024, 3, 11, 23, 10),
             close=datetime(2024, 3, 11, 23, 40))
    df, meta = read_trade_table(write_cfd(tmp_path / "o.sqx", tr + [t]), costs_path=DEFAULT_CFD_COSTS_PATH)
    assert not any("sin rollover" in w for w in meta["warnings"])
    np.testing.assert_allclose(df["swap_usd"], 0.0, atol=1e-4)


def test_cfd_trade_abierto_justo_a_las_23_paga_swap_y_no_cuenta_como_sin_rollover(tmp_path):
    """SQX cobra el swap de la noche a un trade abierto exactamente a las 23:00:00 (rollover) y cerrado después
    (NASDAQ M30, 2026-10-14): no entra en la estimación de la comisión."""
    tr = [cfd_short(d) for d in (4, 5, 6, 7, 8)]
    tr.append(cfd_short(11, hour=23, close_hour=23, swap=-0.5))
    tr[-1]["close"] = tr[-1]["close"].replace(minute=30)
    df, meta = read_trade_table(write_cfd(tmp_path / "q.sqx", tr), costs_path=DEFAULT_CFD_COSTS_PATH)
    assert meta["commission_per_lot"] == pytest.approx(5.0, abs=1e-6)
    np.testing.assert_allclose(df["swap_usd"].iloc[-1], -0.5, atol=1e-4)


def test_mae_con_lotes_fraccionarios_tolera_redondeo_a_centimos(tmp_path):
    """MAE redondeado a céntimos (0,004 USD por debajo de la pérdida bruta) con
    0,25 lotes: dentro del suelo absoluto de 0,011 USD (antes, 0,011·0,25 = 0,0028)."""
    t = cfd_short(4)
    t["mae"] = t["mae"] - 0.00390625      # 0,0039 USD por debajo (float32 exacto)
    tr = [t] + [cfd_short(d) for d in (5, 6, 7, 8)]
    df, _ = read_trade_table(write_cfd(tmp_path / "m.sqx", tr), costs_path=DEFAULT_CFD_COSTS_PATH)
    assert len(df) == 5


def test_cfd_mae_menor_que_perdida_bruta_sigue_siendo_error(tmp_path):
    t = cfd_short(4)
    t["mae"] = t["mae"] - 1.0
    tr = [t] + [cfd_short(d) for d in (5, 6, 7, 8)]
    with pytest.raises(SQXFutureLegError, match="MAE menor"):
        read_trade_table(write_cfd(tmp_path / "x.sqx", tr), costs_path=DEFAULT_CFD_COSTS_PATH)


def test_cfd_simbolo_fuera_de_la_allowlist_cfd(tmp_path):
    with pytest.raises(SQXFutureLegError, match="allowlist"):
        read_trade_table(write_cfd(tmp_path / "s.sqx", [cfd_short(4)]))   # YAML de futuros


# Oro FTMO: comisión PercentageBased de SQX (0,0014% del nocional de apertura, 2026-10-14)
GOLD_PV, GOLD_PCT = 100.0, 0.0014


def gold_long(day: int, open_px: float, *, nights=0, lots=0.25, move=4.0, win=True, swap=0.0,
              comm_lot=None):
    """Largo de oro con precios exactos en float32; comisión por defecto = la proporcional."""
    close_px = open_px + move if win else open_px - move
    gross = (close_px - open_px) * GOLD_PV * lots
    comm = (GOLD_PCT / 100 * open_px * GOLD_PV if comm_lot is None else comm_lot) * lots
    return dict(open=datetime(2024, 3, day, 10, 0), close=datetime(2024, 3, day + nights, 12, 0), lots=lots,
                open_px=open_px, close_px=close_px, sl_px=open_px - 2 * move, pnl=gross - comm + swap,
                mae=10.0 if win else -gross, flag=1)


def write_gold(path, trades):
    return write_sqx(path, trades, symbol="XAUUSD_ftmo(2)", instrument="XAUUSD_cfd_ftmo", pv=GOLD_PV, tick=0.01)


def test_cfd_oro_comision_proporcional_al_precio_y_swap_separado(tmp_path):
    precios = (1200.0, 1800.0, 2400.0, 3000.0, 4096.0)
    tr = [gold_long(d, px, win=(d % 2 == 0)) for d, px in zip((4, 5, 6, 7, 8), precios)]
    tr.append(gold_long(11, 2048.0, nights=2, swap=-68.56 * 0.01 * GOLD_PV * 2 * 0.25 / 100))
    df, meta = read_trade_table(write_gold(tmp_path / "g.sqx", tr), costs_path=DEFAULT_CFD_COSTS_PATH)
    assert meta["asset_class"] == "cfd" and meta["commission_pct"] == GOLD_PCT
    esperado = GOLD_PCT / 100 * df["open_px"] * GOLD_PV * df["lots"]
    np.testing.assert_allclose(df["comm_usd"], esperado, atol=1e-6)
    np.testing.assert_allclose(df["swap_usd"].iloc[:-1], 0.0, atol=1e-3)
    np.testing.assert_allclose(df["swap_usd"].iloc[-1], -0.3428, atol=1e-3)
    assert not any("total_round_trip" in w for w in meta["warnings"])


def test_cfd_oro_comision_constante_no_cuadra_con_la_proporcional(tmp_path):
    tr = [gold_long(d, px, comm_lot=5.84) for d, px in zip((4, 5, 6, 7, 8), (1200.0, 1800.0, 2400.0, 3000.0, 4096.0))]
    with pytest.raises(SQXFutureLegError, match="commission_pct"):
        read_trade_table(write_gold(tmp_path / "k.sqx", tr), costs_path=DEFAULT_CFD_COSTS_PATH)


def test_artefacto_con_pnl_casi_cero_tambien_se_descarta(tmp_path):
    """Revisión DeepSeek H1: |PnL| <= 0,01 USD en el último cierre, no solo 0,0 exacto."""
    eob = long_win(open=datetime(2024, 3, 8, 10, 0), close=datetime(2024, 3, 8, 20, 0),
                   close_px=17980.0, pnl=0.00390625, mae=40.0)
    df, meta = read_trade_table(write_sqx(tmp_path / "e2.sqx", [long_win(), eob]))
    assert len(df) == 1 and any("final del backtest" in w for w in meta["warnings"])


# --------------------------------------------------------------------------
# F2 (guardrail entrada-4-bloques): lectura contra una FICHA de símbolo, rechazos de ATM y de salidas
# parciales, MM del backtest e InstrumentInfo. Todo sintético.
# --------------------------------------------------------------------------
MNQ_SPEC = {"symbol": "@micro_nasdaq", "asset_class": "futures", "point_value_usd": 2.0, "tick_size": 0.25,
            "commission": {"per_lot_rt": 1.12}, "lot_step": 1.0, "min_lot": 1.0,
            "source": {"point_value_usd": "official", "tick_size": "official", "commission": "official"}}
EUR_SPEC = {"symbol": "EURUSD_ftmo", "asset_class": "cfd", "point_value_usd": 10.0 / 0.0001, "tick_size": 0.0001,
            "commission": {"per_lot_rt": 5.0}, "lot_step": 0.01, "min_lot": 0.01,
            "source": {"point_value_usd": "official", "tick_size": "official", "commission": "official"}}
GOLD_SPEC = {"symbol": "XAUUSD_ftmo(2)", "asset_class": "cfd", "point_value_usd": 1.0 / 0.01, "tick_size": 0.01,
             "commission": {"pct_notional_rt": GOLD_PCT}, "lot_step": 0.01, "min_lot": 0.01,
             "source": {"point_value_usd": "official", "tick_size": "official", "commission": "official"}}


def test_f2_write_sqx_por_defecto_es_el_zip_de_siempre(tmp_path):
    base = write_sqx(tmp_path / "base.sqx", [long_win()])
    with zipfile.ZipFile(base) as zf:
        assert zf.namelist() == ["settings.xml", "orders.bin", "strategy_Portfolio.xml"]
        assert zf.read("strategy_Portfolio.xml") == PORTFOLIO_XML
        ref = {n: zf.read(n) for n in zf.namelist()}
    # con portfolio_xml y last_settings: solo cambia el strategy_Portfolio.xml y se añade lastSettings.xml al final
    extra = write_sqx(tmp_path / "extra.sqx", [long_win()], portfolio_xml="<Strategy><X/></Strategy>",
                      last_settings="<root/>")
    with zipfile.ZipFile(extra) as zf:
        assert zf.namelist() == ["settings.xml", "orders.bin", "strategy_Portfolio.xml", "lastSettings.xml"]
        assert zf.read("settings.xml") == ref["settings.xml"] and zf.read("orders.bin") == ref["orders.bin"]
        assert zf.read("strategy_Portfolio.xml") == b"<Strategy><X/></Strategy>"
        assert zf.read("lastSettings.xml") == b"<root/>"
    # last_settings=None sigue sin lastSettings.xml aunque se cambie el portfolio_xml
    only_pf = write_sqx(tmp_path / "pf.sqx", [long_win()], portfolio_xml="<Strategy/>")
    with zipfile.ZipFile(only_pf) as zf:
        assert "lastSettings.xml" not in zf.namelist()


# --- la lectura contra una ficha da lo mismo que contra el YAML -----------------------------------------------
def test_f2_spec_de_la_ficha_igual_que_el_yaml_en_futuros_cfd_y_oro(tmp_path):
    fut = write_sqx(tmp_path / "f.sqx", [long_win(), short_loss()])
    cfd = write_cfd(tmp_path / "c.sqx", [cfd_short(d, win=(d % 2 == 0)) for d in (4, 5, 6, 7, 8)]
                    + [cfd_short(11, nights=2, swap=0.10 * 2 * 0.25)])
    gold = write_gold(tmp_path / "g.sqx", [gold_long(d, px, win=(d % 2 == 0))
                                           for d, px in zip((4, 5, 6, 7, 8), (1200.0, 1800.0, 2400.0, 3000.0, 4096.0))])
    for path, spec, costs in ((fut, MNQ_SPEC, DEFAULT_COSTS_PATH), (cfd, EUR_SPEC, DEFAULT_CFD_COSTS_PATH),
                              (gold, GOLD_SPEC, DEFAULT_CFD_COSTS_PATH)):
        (df_y, meta_y) = read_trade_table(path, costs_path=costs)
        (df_s, meta_s) = read_trade_table_spec(path, spec)
        pd.testing.assert_frame_equal(df_y, df_s)
        assert meta_y == meta_s, path.name
        assert list(meta_s)[:3] == ["strategy_name", "symbol", "pv"]          # mismo contrato de meta, sin claves nuevas


def test_f2_spec_futuros_no_necesita_pasos_de_lote_y_cfd_los_lleva_al_meta(tmp_path):
    fut = write_sqx(tmp_path / "f.sqx", [long_win()])
    _, meta = read_trade_table_spec(fut, {k: v for k, v in MNQ_SPEC.items() if k not in ("lot_step", "min_lot")})
    assert "lot_step" not in meta and "min_lot" not in meta
    cfd = write_cfd(tmp_path / "c.sqx", [cfd_short(d) for d in (4, 5, 6, 7, 8)])
    _, meta = read_trade_table_spec(cfd, {**EUR_SPEC, "lot_step": 0.1, "min_lot": 0.3})
    assert (meta["lot_step"], meta["min_lot"]) == (0.1, 0.3) and meta["commission_per_lot"] == pytest.approx(5.0)


# --- errores contra la ficha: nombran los dos valores y la fuente (E11) ----------------------------------------
def test_f2_simbolo_distinto_del_de_la_ficha_es_error(tmp_path):
    p = write_sqx(tmp_path / "s.sqx", [long_win()])
    with pytest.raises(SQXFormatError, match=r"'@micro_nasdaq'.*'OTRO_x'"):
        read_trade_table_spec(p, {**MNQ_SPEC, "symbol": "OTRO_x"})


def test_f2_point_value_distinto_nombra_los_dos_valores_y_la_fuente(tmp_path):
    p = write_sqx(tmp_path / "pv.sqx", [long_win()], pv=2.0)
    spec = {**MNQ_SPEC, "point_value_usd": 5.0, "source": {**MNQ_SPEC["source"], "point_value_usd": "mt5_sheet"}}
    with pytest.raises(SQXFormatError) as ei:
        read_trade_table_spec(p, spec)
    assert str(ei.value) == "pointValue del .sqx (2.0) ≠ point_value_usd de la ficha (5.0, fuente: mt5_sheet)"
    # la tolerancia es la de siempre (PV_REL_TOL = 1e-6, relativa)
    read_trade_table_spec(p, {**MNQ_SPEC, "point_value_usd": 2.0 * (1 + 5e-7)})
    with pytest.raises(SQXFormatError, match="pointValue"):
        read_trade_table_spec(p, {**MNQ_SPEC, "point_value_usd": 2.0 * (1 + 5e-6)})


def test_f2_tick_size_de_la_ficha_solo_avisa_si_la_ficha_lo_trae(tmp_path):
    p = write_sqx(tmp_path / "t.sqx", [long_win()], tick=0.25)
    _, meta = read_trade_table_spec(p, {**MNQ_SPEC, "tick_size": None})
    assert not any("tickSize" in w for w in meta["warnings"])
    _, meta = read_trade_table_spec(p, {k: v for k, v in MNQ_SPEC.items() if k != "tick_size"})
    assert not any("tickSize" in w for w in meta["warnings"])
    _, meta = read_trade_table_spec(p, {**MNQ_SPEC, "tick_size": 0.5,
                                        "source": {**MNQ_SPEC["source"], "tick_size": "manual"}})
    assert ("tickSize del .sqx (0.25) ≠ tick_size de la ficha (0.5, fuente: manual)") in meta["warnings"]


def test_f2_comision_distinta_en_futuros_avisa_con_los_dos_valores_y_la_fuente(tmp_path):
    p = write_sqx(tmp_path / "c.sqx", [long_win(pnl=20.0 - 1.5)])                   # el backtest cobró 1,50/lote
    _, meta = read_trade_table_spec(p, {**MNQ_SPEC, "source": {**MNQ_SPEC["source"], "commission": "mt5_sheet"}})
    msg = next(w for w in meta["warnings"] if "per_lot_rt" in w)
    assert "1.5000" in msg and "1.1200" in msg and "fuente: mt5_sheet" in msg


def test_f2_comision_pct_distinta_es_error_con_los_dos_valores_y_la_fuente(tmp_path):
    tr = [gold_long(d, px, comm_lot=5.84) for d, px in zip((4, 5, 6, 7, 8), (1200.0, 1800.0, 2400.0, 3000.0, 4096.0))]
    p = write_gold(tmp_path / "k.sqx", tr)
    with pytest.raises(SQXFormatError, match="pct_notional_rt de la ficha") as ei:
        read_trade_table_spec(p, {**GOLD_SPEC, "source": {**GOLD_SPEC["source"], "commission": "manual"}})
    msg = str(ei.value)
    assert "fuente: manual" in msg and "USD/lote" in msg and "5.8400" in msg    # lo que cobró el .sqx
    assert "0.0014%" in msg                                                     # lo que dice la ficha
    # el mensaje de siempre contra el YAML no cambia
    with pytest.raises(SQXFormatError, match="commission_pct"):
        read_trade_table(p, costs_path=DEFAULT_CFD_COSTS_PATH)


def test_f2_pocos_trades_sin_rollover_citan_la_ficha_y_el_yaml_sigue_igual(tmp_path):
    tr = [cfd_short(4), cfd_short(5, nights=1, swap=0.025)]
    p = write_cfd(tmp_path / "p.sqx", tr)
    _, meta_s = read_trade_table_spec(p, {**EUR_SPEC, "source": {**EUR_SPEC["source"], "commission": "manual"}})
    assert any("tomada de la ficha (5.0, fuente: manual)" in w for w in meta_s["warnings"])
    _, meta_y = read_trade_table(p, costs_path=DEFAULT_CFD_COSTS_PATH)
    assert any("tomada del YAML (5.0)" in w for w in meta_y["warnings"])
    gp = write_gold(tmp_path / "gp.sqx", [gold_long(4, 2048.0), gold_long(5, 2048.0, nights=1, swap=-0.1)])
    _, mg = read_trade_table_spec(gp, GOLD_SPEC)
    assert any("comisión proporcional de la ficha (0.0014%, fuente: official)" in w for w in mg["warnings"])


@pytest.mark.parametrize("patch, needle", [
    ({"symbol": None}, "sin 'symbol'"),
    ({"asset_class": "fx"}, "asset_class desconocido"),
    ({"commission": {"per_lot_rt": 1.0, "pct_notional_rt": 0.1}}, "exactamente una"),
    ({"commission": {}}, "exactamente una"),
    ({"commission": {"otra": 1.0}}, "exactamente una"),
    ({"commission": {"pct_notional_rt": 0.1}}, "pct_notional_rt solo vale en CFD"),
    ({"point_value_usd": None}, "falta la clave 'point_value_usd'"),
    ({"point_value_usd": "x"}, "no numérico"),
    ({"point_value_usd": float("inf")}, "no finito"),
    ({"point_value_usd": 0.0}, "point_value_usd <= 0"),
])
def test_f2_spec_invalido_falla_con_un_error_claro(tmp_path, patch, needle):
    p = write_sqx(tmp_path / "i.sqx", [long_win()])
    with pytest.raises(SQXFutureLegError, match=needle):
        read_trade_table_spec(p, {**MNQ_SPEC, **patch})


def test_f2_spec_cfd_sin_pasos_de_lote_o_con_ellos_no_positivos_falla(tmp_path):
    p = write_cfd(tmp_path / "c.sqx", [cfd_short(d) for d in (4, 5, 6, 7, 8)])
    with pytest.raises(SQXFutureLegError, match="lot_step"):
        read_trade_table_spec(p, {k: v for k, v in EUR_SPEC.items() if k != "lot_step"})
    with pytest.raises(SQXFutureLegError, match="lot_step/min_lot <= 0"):
        read_trade_table_spec(p, {**EUR_SPEC, "min_lot": 0.0})


# --- rechazo de las salidas parciales (ATM), enmienda E8 --------------------------------------------------------
ATM_MSG = "salidas parciales (ATM) activadas en la estrategia: no soportadas en la v1"


@pytest.mark.parametrize("xml", [
    '<Strategy><ATMs source="x" enable="true"/></Strategy>',
    '<Strategy><Rules><ATMs enable="true" source="none" /></Rules></Strategy>',
    '<Strategy><ATMs source="x" enable="TRUE"/></Strategy>',
    '<Strategy><ATMs enable="True"><Level/></ATMs></Strategy>',
    '<s:Strategy xmlns:s="urn:sq"><s:ATMs enable="true"/></s:Strategy>',
    '<Strategy><ATMs enable="false"/><ATMs enable="true"/></Strategy>',     # basta con uno activado
])
def test_f2_atm_activado_se_rechaza_en_las_dos_lecturas(tmp_path, xml):
    p = write_sqx(tmp_path / "atm.sqx", [long_win()], portfolio_xml=xml)
    for read in (lambda: read_trade_table(p), lambda: read_trade_table_spec(p, MNQ_SPEC)):
        with pytest.raises(SQXFormatError) as ei:
            read()
        assert str(ei.value) == ATM_MSG


@pytest.mark.parametrize("xml", [
    '<Strategy><ATMs source="none" enable="false" /></Strategy>',            # la etiqueta real de SQX sin ATM
    '<Strategy><ATMs source="x"/></Strategy>',                                # sin atributo enable
    '<Strategy><ATMs enable="no"/></Strategy>',
    '<Strategy><ATMsExtra enable="true"/></Strategy>',                        # otro elemento
    '<Strategy><Other ATMs="true" enable="true"/></Strategy>',
    "<Strategy><ATMs enable='true'",                                          # no parsea: no se rechaza por esto
    "esto no es xml",
])
def test_f2_atm_desactivado_o_xml_que_no_parsea_no_se_rechaza(tmp_path, xml):
    p = write_sqx(tmp_path / "ok.sqx", [long_win()], portfolio_xml=xml)
    df, _ = read_trade_table(p)
    assert len(df) == 1
    assert len(read_trade_table_spec(p, MNQ_SPEC)[0]) == 1


# --- rechazo de órdenes que comparten colocación y ejecución -------------------------------------------------------
def _shared_msg(n: int) -> str:
    return f"{n} órdenes comparten hora de colocación y de ejecución (¿salidas parciales?): no soportado en la v1"


def test_f2_ordenes_con_la_misma_colocacion_y_ejecucion_se_rechazan(tmp_path):
    a = long_win(open=datetime(2024, 3, 4, 10, 0), close=datetime(2024, 3, 4, 11, 0))
    b = long_win(open=datetime(2024, 3, 4, 10, 0), close=datetime(2024, 3, 4, 12, 0))      # mismas horas de entrada
    p = write_sqx(tmp_path / "dup.sqx", [a, b])
    for read in (lambda: read_trade_table(p), lambda: read_trade_table_spec(p, MNQ_SPEC)):
        with pytest.raises(SQXFormatError) as ei:
            read()
        assert str(ei.value) == _shared_msg(2)
    assert _shared_msg(2) != ATM_MSG                                                         # mensajes distintos
    c = long_win(open=datetime(2024, 3, 4, 10, 0), close=datetime(2024, 3, 4, 13, 0))
    with pytest.raises(SQXFormatError) as ei:
        read_trade_table(write_sqx(tmp_path / "dup3.sqx", [a, b, c]))
    assert str(ei.value) == _shared_msg(3)
    d = long_win(open=datetime(2024, 3, 5, 10, 0), close=datetime(2024, 3, 5, 11, 0))
    e = long_win(open=datetime(2024, 3, 5, 10, 0), close=datetime(2024, 3, 5, 12, 0))
    with pytest.raises(SQXFormatError) as ei:
        read_trade_table(write_sqx(tmp_path / "dup22.sqx", [a, b, d, e]))                    # dos grupos de dos
    assert str(ei.value) == _shared_msg(4)


def test_f2_misma_ejecucion_con_otra_colocacion_o_al_reves_no_se_rechaza(tmp_path):
    base = dict(close=datetime(2024, 3, 4, 12, 0))
    a = long_win(order=datetime(2024, 3, 4, 8, 0), open=datetime(2024, 3, 4, 10, 0), **base)
    b = long_win(order=datetime(2024, 3, 4, 9, 0), open=datetime(2024, 3, 4, 10, 0), **base)   # otra colocación
    c = long_win(order=datetime(2024, 3, 4, 8, 0), open=datetime(2024, 3, 4, 10, 30), **base)  # otra ejecución
    df, _ = read_trade_table(write_sqx(tmp_path / "ok.sqx", [a, b, c]))
    assert len(df) == 3


def test_f2_registros_de_otro_test_con_las_mismas_horas_no_cuentan(tmp_path):
    a = long_win()
    x = crosscheck(open=a["open"], close=a["close"], lots=1.0, sl_px=17990.0, pnl=18.875)
    y = dict(x)
    df, _ = read_trade_table(write_sqx(tmp_path / "cc.sqx", [a, x, y]))                  # solo cuenta el test Main
    assert len(df) == 1


@pytest.mark.skipif(not REAL_BANK.exists(), reason="banco real MNQ M30 no disponible")
def test_f2_los_rechazos_no_cambian_nada_en_un_banco_real_de_65_sqx():
    files = sorted(REAL_BANK.glob("*.sqx"))
    assert len(files) == 65
    for f in files:                                    # ninguno rechazado por ATM ni por órdenes compartidas
        read_trade_table_spec(f, MNQ_SPEC)


# --- MM del backtest (lastSettings.xml) -----------------------------------------------------------------------------
LAST_FIXED_AMOUNT = """<root>
  <Methods>
    <Method type="FixedSize" use="false"><Params><Param key="#Size#" value="0.1">0.1</Param></Params></Method>
    <Method type="FixedAmount" use="true"><Params>
      <Param key="#RiskedMoney#" value="57.5">57.5</Param>
      <Param key="#Decimals#" value="2">2</Param>
      <Param key="#LotsIfNoMM#">0.01</Param>
      <Param key="MaxLots" className="x">10</Param>
      <Param key="#Etiqueta#" value="texto">texto</Param>
    </Params></Method>
  </Methods>
  <Commissions><Method type="SizeBased" use="true"><Params><Param key="Commission">4.5</Param></Params></Method></Commissions>
</root>"""


def test_f2_read_backtest_mm_devuelve_el_metodo_activo_y_sus_parametros(tmp_path):
    p = write_sqx(tmp_path / "mm.sqx", [long_win()], last_settings=LAST_FIXED_AMOUNT)
    assert read_backtest_mm(p) == {"method": "FixedAmount", "params": {
        "RiskedMoney": 57.5, "Decimals": 2.0, "LotsIfNoMM": 0.01, "MaxLots": 10.0}}      # clave sin #; no numérico fuera
    atr = LAST_FIXED_AMOUNT.replace('type="FixedAmount"', 'type="ATRRiskBasedSizing"')
    assert read_backtest_mm(write_sqx(tmp_path / "atr.sqx", [long_win()], last_settings=atr))["method"] == \
        "ATRRiskBasedSizing"
    assert {"FixedSize", "FixedAmount", "RiskFixedBalancePct", "RiskFixedPctOfAccount",
            "ATRRiskBasedSizing"} <= MM_CLASSES and len(MM_CLASSES) == 12


def test_f2_read_backtest_mm_sin_dato_devuelve_none_y_no_eleva(tmp_path):
    assert read_backtest_mm(write_sqx(tmp_path / "sin.sqx", [long_win()])) is None                 # sin lastSettings.xml
    assert read_backtest_mm(write_sqx(tmp_path / "x.sqx", [long_win()], last_settings="<no xml")) is None
    solo_inactivo = LAST_FIXED_AMOUNT.replace('type="FixedAmount" use="true"', 'type="FixedAmount" use="false"')
    assert read_backtest_mm(write_sqx(tmp_path / "i.sqx", [long_win()], last_settings=solo_inactivo)) is None
    # un Method activo que NO es de MM (la comisión) no cuenta
    solo_comision = ('<root><Method type="SizeBased" use="true"><Params><Param key="Commission">4.5</Param>'
                     "</Params></Method></root>")
    assert read_backtest_mm(write_sqx(tmp_path / "c.sqx", [long_win()], last_settings=solo_comision)) is None
    bad = tmp_path / "bad.sqx"
    bad.write_bytes(b"no es un zip")
    assert read_backtest_mm(bad) is None
    assert read_backtest_mm(tmp_path / "no_existe.sqx") is None


# --- InstrumentInfo (alta guiada de una ficha desde un .sqx) --------------------------------------------------------
def test_f2_read_instrument_info_lee_los_campos_y_la_comision_escapada(tmp_path):
    inner = ('<Method type="None" use="false"><Params /></Method><Method type="SizeBased" use="true"><Params>'
             '<Param key="Commission" className="SizeBased">4.5</Param></Params></Method>')
    p = patch_instrument_info(write_sqx(tmp_path / "i.sqx", [long_win()], symbol="EURUSD_axi",
                                        instrument="EURUSD_axi", pv=100000.0, tick=0.0001),
                              orderSizeStep="0.01", defaultSpread="0.5", commissions=inner)
    assert read_instrument_info(p) == {
        "symbol": "EURUSD_axi", "instrument": "EURUSD_axi", "point_value": 100000.0, "tick_size": 0.0001,
        "order_size_step": 0.01, "default_spread": 0.5, "commission": {"type": "SizeBased", "value": 4.5}}
    # sin los atributos opcionales: None (nunca inventa valores)
    q = write_sqx(tmp_path / "q.sqx", [long_win()])
    assert read_instrument_info(q) == {
        "symbol": "@micro_nasdaq", "instrument": "NQ_ftmo", "point_value": 2.0, "tick_size": 0.25,
        "order_size_step": None, "default_spread": None, "commission": None}
    # comisión de tipo distinto de SizeBased: se informa del tipo (quien llama decide)
    pct = patch_instrument_info(write_sqx(tmp_path / "p.sqx", [long_win()]), commissions=(
        '<Method type="PercentageBased" use="true"><Params><Param key="Commission">0.0014</Param></Params></Method>'))
    assert read_instrument_info(pct)["commission"] == {"type": "PercentageBased", "value": 0.0014}
    sin_activa = patch_instrument_info(write_sqx(tmp_path / "s.sqx", [long_win()]), commissions=(
        '<Method type="SizeBased" use="false"><Params><Param key="Commission">4.5</Param></Params></Method>'))
    assert read_instrument_info(sin_activa)["commission"] is None


def test_f2_read_instrument_info_ilegible_es_error_claro(tmp_path):
    bad = tmp_path / "bad.sqx"
    bad.write_bytes(b"no es un zip")
    with pytest.raises(SQXFutureLegError, match="ZIP"):
        read_instrument_info(bad)
    sin_settings = tmp_path / "ss.sqx"
    with zipfile.ZipFile(sin_settings, "w") as zf:
        zf.writestr("otro.xml", "<a/>")
    with pytest.raises(SQXFutureLegError, match="settings.xml ausente"):
        read_instrument_info(sin_settings)


# --------------------------------------------------------------------------
# F3 (guardrail entrada-4-bloques), correcciones de la revisión de código de la F2: H1 (read_instrument_info) y H2
# (tickSize del .sqx contra una ficha).
# --------------------------------------------------------------------------
def _rewrite_settings(path: Path, edit) -> Path:
    """Reescribe el ``settings.xml`` de un ``.sqx`` sintético con ``edit(texto) -> texto`` (el resto del ZIP igual)."""
    with zipfile.ZipFile(path) as zf:
        items = [(i.filename, zf.read(i.filename)) for i in zf.infolist()]
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in items:
            zf.writestr(name, edit(data.decode()).encode() if name == "settings.xml" else data)
    return path


def _sin_instrument_info(text: str) -> str:
    head, tail = text.split("<InstrumentInfo", 1)
    return head + tail.split("/>", 1)[1]


def test_f3_h1_settings_sin_instrumentinfo_es_error_claro_y_no_un_crash(tmp_path):
    """H1: un ``settings.xml`` cuyo ``SymbolInfo`` no trae ``InstrumentInfo`` da un ``SQXFutureLegError`` con un texto
    claro (el endpoint ``draft-from-sqx`` lo devuelve como 400), no un ``AttributeError``."""
    p = _rewrite_settings(write_sqx(tmp_path / "sin_instr.sqx", [long_win()]), _sin_instrument_info)
    assert "<InstrumentInfo" not in zipfile.ZipFile(p).read("settings.xml").decode()
    with pytest.raises(SQXFutureLegError, match="InstrumentInfo"):
        read_instrument_info(p)
    with pytest.raises(SQXFormatError, match="InstrumentInfo"):                    # es un SQXFormatError: los llamadores lo capturan
        read_instrument_info(p)


def test_f3_h1_los_dos_buscadores_del_symbolinfo_main_no_suponen_que_existe(tmp_path, monkeypatch):
    """H1: ``read_instrument_info`` localiza el ``SymbolInfo`` del símbolo del resultado Main con ``next(…, None)``. Aunque
    ``_parse_settings_xml`` ya valida lo mismo, la búsqueda propia no puede acabar en ``StopIteration`` ni en
    ``AttributeError``: aquí se fuerza que el símbolo no figure en ningún ``SymbolInfo`` y que el ``SymbolInfo`` no lleve
    ``InstrumentInfo``, saltándose la validación previa."""
    import sqx.sqx_trades as st
    p = write_sqx(tmp_path / "a.sqx", [long_win()])
    real = st._parse_settings_xml(zipfile.ZipFile(p).read("settings.xml"))
    monkeypatch.setattr(st, "_parse_settings_xml", lambda xml: {**real, "symbol": "NO_ESTA_EN_NINGUN_SYMBOLINFO"})
    with pytest.raises(SQXFutureLegError, match=r"sin SymbolInfo.*'NO_ESTA_EN_NINGUN_SYMBOLINFO'"):
        read_instrument_info(p)
    monkeypatch.setattr(st, "_parse_settings_xml", lambda xml: real)               # símbolo bien, pero sin InstrumentInfo
    q = _rewrite_settings(write_sqx(tmp_path / "b.sqx", [long_win()]), _sin_instrument_info)
    with pytest.raises(SQXFutureLegError, match=r"no trae InstrumentInfo"):
        read_instrument_info(q)


@pytest.mark.parametrize("tick", [0.0, -0.25, float("nan")])
def test_f3_h2_tick_size_del_sqx_no_positivo_es_error_aunque_la_ficha_no_traiga_tick(tmp_path, tick):
    """H2: la lectura por ficha rechaza un ``tickSize`` <= 0 (o no finito) del .sqx, también cuando la ficha no trae
    ``tick_size`` (caso AXI, donde la serie por minuto usa el tick del .sqx como tolerancia)."""
    p = write_sqx(tmp_path / "t.sqx", [long_win()], tick=tick)
    for spec in ({**MNQ_SPEC, "tick_size": None}, {k: v for k, v in MNQ_SPEC.items() if k != "tick_size"}, MNQ_SPEC):
        with pytest.raises(SQXFutureLegError, match=r"tickSize del \.sqx .* <= 0 .*@micro_nasdaq"):
            read_trade_table_spec(p, spec)
    # con un tick válido, la misma ficha sin tick_size lee (el aviso de tick solo existe si la ficha lo trae)
    ok = write_sqx(tmp_path / "ok.sqx", [long_win()], tick=0.25)
    assert len(read_trade_table_spec(ok, {**MNQ_SPEC, "tick_size": None})[0]) == 1


def test_f3_h2_tick_size_no_positivo_de_la_ficha_ya_no_se_omite_en_silencio(tmp_path):
    p = write_sqx(tmp_path / "t.sqx", [long_win()], tick=0.25)
    for bad in (0.0, -0.25):
        with pytest.raises(SQXFutureLegError, match=r"ficha: tick_size <= 0 para @micro_nasdaq"):
            read_trade_table_spec(p, {**MNQ_SPEC, "tick_size": bad})


def test_f3_h2_el_camino_del_yaml_no_cambia(tmp_path):
    """El camino del YAML conserva sus mensajes y su orden: un ``tickSize`` 0 del .sqx sigue siendo solo un aviso."""
    p = write_sqx(tmp_path / "t.sqx", [long_win()], tick=0.0)
    df, meta = read_trade_table(p)
    assert len(df) == 1 and any("tickSize del .sqx (0.0) != tick_size del YAML" in w for w in meta["warnings"])
