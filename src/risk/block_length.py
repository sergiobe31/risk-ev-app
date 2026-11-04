# PLAN-DESVIO: la línea "Developed by" del LICENSE.md de arch lleva dos direcciones de correo; abajo se conserva el nombre y
# la URL, sin los correos. El escaneo duro del export de la app (patrón de correo de scripts/export_shareable/manifest.yaml)
# aborta con cualquier dirección. El aviso de copyright, las tres condiciones y la exclusión de garantía, que es lo que la
# licencia obliga a conservar, van íntegros.
"""Longitud media de bloque del bootstrap estacionario: Politis y White (2004), con la corrección de Patton, Politis y
White (2009).

Port a numpy puro (sin ``arch`` como dependencia) de ``arch.bootstrap.base._single_optimal_block`` (arch 8.0.0), la
columna "stationary" de ``arch.bootstrap.optimal_block_length``. Dos capas:

* :func:`politis_white_raw`: el cálculo LITERAL de arch. Sin acotar: puede dar NaN (serie constante: 0/0) o un valor
  < 1. Es la que se compara con arch a 1e-9.
* :func:`politis_white_block` (y :func:`politis_white_info`, que además da ``opt_m`` y ``m``): la envoltura que usa la
  app. Exige n >= 25 (``ValueError`` si no), convierte un NaN o un infinito en 1 y acota a [1, b_max], porque
  :func:`src.risk.mc.stationary_block_indices` trata cualquier longitud < 1 como 1. Es una decisión de la app, no de
  arch.

Solo numpy (más stdlib). ``src.risk`` no importa ``sqx`` ni ``arch``.

Referencias:

* Politis, D. N. y White, H. (2004), "Automatic Block-Length Selection for the Dependent Bootstrap", Econometric
  Reviews 23(1), 53-70.
* Patton, A., Politis, D. N. y White, H. (2009), "Correction to 'Automatic Block-Length Selection for the Dependent
  Bootstrap' by D. Politis and H. White", Econometric Reviews 28(4), 372-375.

Aviso de copyright y licencia de arch (tipo NCSA), copiados del LICENSE.md de la distribución de arch 8.0.0: este módulo
es una obra derivada de su código y la licencia exige conservarlos.

Copyright (c) 2017 Kevin Sheppard. All rights reserved.

Developed by: Kevin Sheppard
https://www.kevinsheppard.com

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal with
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimers.

Redistributions in binary form must reproduce the above copyright notice, this
list of conditions and the following disclaimers in the documentation and/or
other materials provided with the distribution.

Neither the names of Kevin Sheppard, nor the names of its contributors may be
used to endorse or promote products derived from this Software without specific
prior written permission.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
THE SOFTWARE.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["MIN_OBS", "PolitisWhite", "politis_white_raw", "politis_white_info", "politis_white_block"]

#: Número mínimo de observaciones de la envoltura (decisión de la app: con menos, el estimador no es fiable).
MIN_OBS = 25


# PLAN-DESVIO (menor): el plan fija dos funciones (``politis_white_raw`` y ``politis_white_block``); E10 pide que ``opt_m`` y ``m``
# lleguen a ``meta.model_context.block_size``, y para eso hace falta una tercera, ``politis_white_info`` (la misma envoltura,
# que además devuelve los parámetros del estimador). ``politis_white_block(x)`` es exactamente ``politis_white_info(x).b``.
@dataclass(frozen=True)
class PolitisWhite:
    """Resultado de la envoltura para una serie.

    ``b_raw``: el valor literal de arch (puede ser NaN o < 1). ``b``: el valor de la app, acotado a [1, ``b_max``].
    ``b_max``: ``ceil(min(3·√n, n/3))``. ``opt_m``: primer ``m`` con ``k_n`` autocorrelaciones seguidas bajo la banda
    (``None`` si no hay ninguno). ``m``: la ventana usada (``2·max(opt_m, 1)`` acotada a ``m_max``, o ``m_max`` si no
    hubo ``opt_m``). ``m_max``: ``ceil(√n) + k_n``. ``n``: nº de observaciones."""

    b_raw: float
    b: float
    b_max: float
    opt_m: int | None
    m: int
    m_max: int
    n: int


def _stationary_core(x: np.ndarray) -> tuple[float, float, int | None, int, int]:
    """Port literal de ``arch.bootstrap.base._single_optimal_block`` (arch 8.0.0), solo la columna "stationary".

    Devuelve ``(b_sb, b_max, opt_m, m, m_max)``. Las líneas del algoritmo son las de arch; lo único añadido es el
    ``np.errstate`` (una serie constante da 0/0: NaN, sin avisos de numpy) y que se devuelvan ``opt_m``, ``m`` y
    ``m_max``. De la columna "circular" (``b_cb``, que arch calcula en la misma función) no se usa nada."""
    with np.errstate(invalid="ignore", divide="ignore"):
        nobs = x.shape[0]
        eps = x - x.mean(0)
        b_max = np.ceil(min(3 * np.sqrt(nobs), nobs / 3))
        kn = max(5, int(np.log10(nobs)))
        m_max = int(np.ceil(np.sqrt(nobs))) + kn
        # Find first collection of kn autocorrelations that are insignificant
        cv = 2 * np.sqrt(np.log10(nobs) / nobs)
        acv = np.zeros(m_max + 1)
        abs_acorr = np.zeros(m_max + 1)
        opt_m: int | None = None
        for i in range(m_max + 1):
            v1 = eps[i + 1 :] @ eps[i + 1 :]
            v2 = eps[: -(i + 1)] @ eps[: -(i + 1)]
            cross_prod = eps[i:] @ eps[: nobs - i]
            acv[i] = cross_prod / nobs
            abs_acorr[i] = np.abs(cross_prod) / np.sqrt(v1 * v2)
            if i >= kn:
                if np.all(abs_acorr[i - kn : i] < cv) and opt_m is None:
                    opt_m = i - kn
        m = 2 * max(opt_m, 1) if opt_m is not None else m_max
        m = min(m, m_max)

        g = 0.0
        lr_acv = acv[0]
        for k in range(1, m + 1):
            lam = 1 if k / m <= 1 / 2 else 2 * (1 - k / m)
            g += 2 * lam * k * acv[k]
            lr_acv += 2 * lam * acv[k]
        d_sb = 2 * lr_acv**2
        b_sb = ((2 * g**2) / d_sb) ** (1 / 3) * nobs ** (1 / 3)
        b_sb = min(b_sb, b_max)
    return float(b_sb), float(b_max), opt_m, int(m), int(m_max)


def _as_series(x) -> np.ndarray:
    """``x`` como vector 1-D float64 (el tipo que recibe arch por columna)."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"x debe ser una serie 1-D (recibida con {arr.ndim} dimensiones)")
    return arr


def politis_white_raw(x) -> float:
    """Longitud media óptima del bootstrap estacionario, valor LITERAL de arch: puede ser NaN (serie constante) o < 1.

    Igual a ``float(arch.bootstrap.optimal_block_length(x)["stationary"].iloc[0])`` (arch 8.0.0) para una serie 1-D.
    Sin acotar y sin exigir un mínimo de observaciones (con muy pocas, falla con el mismo error de formas que arch):
    para la app, :func:`politis_white_block`."""
    return _stationary_core(_as_series(x))[0]


def politis_white_info(x) -> PolitisWhite:
    """Envoltura para la app: el valor literal, el acotado y los parámetros del estimador (``opt_m``, ``m``).

    * ``ValueError`` si hay menos de :data:`MIN_OBS` observaciones o algún valor no es finito.
    * Un resultado NaN o infinito se convierte en 1: una serie constante tiene autocovarianzas 0 (0/0) y no tiene
      dependencia que preservar.
    * El resultado se acota a [1, ``b_max``] con ``b_max = ceil(min(3·√n, n/3))`` (arch ya aplica el tope superior; el
      inferior es de la app)."""
    arr = _as_series(x)
    n = int(arr.shape[0])
    if n < MIN_OBS:
        raise ValueError(f"Politis-White exige al menos {MIN_OBS} observaciones (recibidas {n})")
    # PLAN-DESVIO: el plan no dice qué hacer con una entrada no finita; se rechaza con ValueError en vez de devolver 1 en
    # silencio con datos corruptos (el NaN→1 del plan es para el RESULTADO, p. ej. una serie constante, no para la entrada).
    if not np.all(np.isfinite(arr)):
        raise ValueError("la serie contiene valores no finitos (NaN o infinito)")
    b_raw, b_max, opt_m, m, m_max = _stationary_core(arr)
    b = b_raw if np.isfinite(b_raw) else 1.0
    b = float(min(max(b, 1.0), b_max))
    return PolitisWhite(b_raw=b_raw, b=b, b_max=b_max, opt_m=opt_m, m=m, m_max=m_max, n=n)


def politis_white_block(x) -> float:
    """Longitud media de bloque para el bootstrap estacionario de la app: ``politis_white_info(x).b`` (en [1, b_max])."""
    return politis_white_info(x).b
