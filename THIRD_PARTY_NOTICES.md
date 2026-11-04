# Avisos de terceros

El código de este repositorio se publica bajo la licencia MIT (`LICENSE`). Incluye o deriva de estas obras de
terceros, cada una con su licencia.

## En el código Python

| Obra | Dónde | Licencia |
| --- | --- | --- |
| [arch](https://github.com/bashtage/arch) 8.0.0, Kevin Sheppard | `src/risk/block_length.py` es un port de su longitud óptima de bloque (Politis-White) | Tipo NCSA: el aviso de copyright y la licencia completos están en la cabecera del fichero |

Las dependencias de Python (`requirements.txt`) no van incluidas: las instala `pip`, cada una con su licencia.

## En el front compilado (`poc_web/web/dist/`)

| Obra | Licencia | Copyright |
| --- | --- | --- |
| [TradingView Lightweight Charts™](https://www.tradingview.com/lightweight-charts/) 5.2 | Apache-2.0 | TradingView, Inc. |
| [fancy-canvas](https://www.npmjs.com/package/fancy-canvas) 2.1 (dependencia de la anterior) | MIT | TradingView, Inc. |
| [React y React DOM](https://react.dev/) 19.2, scheduler 0.27 | MIT | Meta Platforms, Inc. and affiliates |
| [Tailwind CSS](https://tailwindcss.com/) 4.3 | MIT | Tailwind Labs, Inc. |
| [clsx](https://github.com/lukeed/clsx) 2.1 | MIT | Luke Edwards |
| [Tufte CSS](https://github.com/edwardtufte/tufte-css) (guía, `guia/tufte.css`) | MIT (`guia/LICENSE-tufte-css.txt`) | Dave Liepmann |
| [ET Book](https://github.com/edwardtufte/et-book) (fuentes de la guía, `guia/et-book/`) | MIT | Dmitry Krasny, Bonnie Scranton, Edward Tufte |

Las versiones exactas están en `poc_web/web/package-lock.json`.

**Aviso de atribución de Lightweight Charts** (lo exige su licencia):

> TradingView Lightweight Charts™
> Copyright (c) 2025 TradingView, Inc. https://www.tradingview.com/

Las gráficas de la app muestran el logo de TradingView con un enlace a <https://www.tradingview.com/>, que es como
la librería cumple el requisito del enlace.
