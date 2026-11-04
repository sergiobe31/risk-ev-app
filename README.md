# Risk App: riesgo y prop firms para carteras de StrategyQuant X

Aplicación web local que, a partir de tus estrategias exportadas de StrategyQuant X (SQX), responde a dos
preguntas:
- **Prop firms (FTMO):** ¿qué probabilidad tengo de pasar el challenge y cuánto dinero espero ganar o perder con
  cada tamaño de riesgo?
- **AXI Select:** ¿qué probabilidad tengo de pasar la etapa y de caer?

Además te dice cómo configurar el money management de tus EAs para que operen el riesgo que has simulado. Todo
corre en tu ordenador; no se envía nada a ningún servidor.

> **English:** a local web app that takes your StrategyQuant X strategies (`.sqx`) and estimates, by Monte Carlo,
> the probability of passing a prop-firm challenge (FTMO CFD and Futures) or an AXI Select stage, the expected
> value for each risk per trade, and the money-management settings your EAs need to trade that risk. Runs on
> macOS and Windows with Python 3.11. The docs are in Spanish; issues and pull requests in English are welcome.
> Unofficial tool, not affiliated with StrategyQuant, FTMO or Axi. Not financial advice.

> **Herramienta no oficial**, sin relación con StrategyQuant s.r.o., FTMO ni Axi (StrategyQuant X, FTMO y AXI
> Select son marcas de sus dueños). Las reglas de cada firma están copiadas de sus páginas oficiales en una fecha y
> pueden haber cambiado: compruébalas antes de jugar un reto. **No es asesoramiento financiero**: las cifras son
> estimaciones de un modelo sobre tu backtest, y el pasado no garantiza el futuro. Úsala bajo tu responsabilidad
> (licencia MIT, sin garantía).

> **Esta es una copia generada.** El código vive en otro repositorio y este se regenera entero con un script
> (cada commit lleva `Exported-From: <versión>`).
> - Si encuentras un fallo o quieres un cambio, abre un issue o un PR aquí (ver `CONTRIBUTING.md`): se porta a la
>   fuente y llega en el siguiente export.
> - Puedes descargarlo, clonarlo o hacer un fork y cambiar tu copia como quieras. Este repositorio solo lo
>   modifica el script de export.

## Arranque en 5 pasos

Necesitas **Python 3.11** (las versiones de `requirements.txt` están verificadas con esa). No hace falta Node ni
compilar nada: el front ya viene compilado en `poc_web/web/dist/`.

1. **Instala Python 3.11**: en Mac con `brew install python@3.11` (o desde python.org); en Windows desde python.org,
   marcando la casilla *Add python.exe to PATH*.
2. **Descarga este repositorio** (botón *Code → Download ZIP*, o `git clone`) y abre una terminal en esa carpeta.
3. **Crea un entorno virtual**.

   Mac / Linux (bash o zsh):

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

   Windows (PowerShell):

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   Si PowerShell bloquea el script, ejecuta una vez `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` y repite.
4. **Instala las dependencias**: `pip install -r requirements.txt`
5. **Arranca**: `python run_app.py`.
   - Se abre el navegador en <http://127.0.0.1:8600>.
   - Para parar, `Ctrl+C` en la terminal.
   - Opciones: `--no-browser` (no abre el navegador) y `--port 8700` (otro puerto).

Comprobación rápida de que todo funciona, sin abrir el navegador: `python run_app.py --smoke`.

## Primer uso con la demo

El repositorio trae **8 estrategias sintéticas** en `risk_app/inputs/strategies/`. Están inventadas con un
generador y no tienen relación con estrategias reales.

| Carpeta | Contenido | Pregunta que responde |
| --- | --- | --- |
| `demo_cfd` | 2 × `GBPUSD_ftmo` + 1 × `AUDUSD_ftmo` | FTMO CFD (perfil *FTMO Challenge 50k*) |
| `demo_futuros` | `@micro_nasdaq`, `@micro_oro`, `@micro_eurusd` | FTMO Futures (perfil *FTMO Futures Growth 50K*) |
| `demo_axi` | `XAUUSD_axi` + `EURUSD_axi` | AXI Select (etapa Seed, cuenta de 5.000 $) |

**Prop firm:**
- En la pestaña **Simulador**, abre la carpeta `demo_cfd` (o `demo_futuros`), pulsa **todas** y luego
  **Cargar/validar cartera**.
- Elige *¿Juego el challenge de una prop firm?*, la firma y el reto que corresponden, y pulsa **Simular EV de la
  cartera**.

**AXI:**
- Pon el tamaño de cuenta en 5000 y carga la carpeta `demo_axi`.
- Elige *¿Paso el reto AXI?* y pulsa **Simular P(pass) / P(ruin)**.

Qué vas a ver con la demo:

- **El aviso `flotante_m1`.** La demo no trae precios de 1 minuto (OHLC M1), así que el riesgo flotante dentro del
  día se acota con el peor momento de cada operación (su MAE). Es más prudente y es lo esperado.
- **En AXI, los avisos de tolerancia y de "con bloque 1".** Las estrategias de la demo son mediocres a propósito.
- **Números sintéticos.** Sirven para ver cómo funciona la herramienta, **no** como resultado de ninguna
  estrategia.

## Cómo funciona, en cuatro piezas

**Lectura.** De cada estrategia la app lee su `.sqx` y nada más. Del backtest toma, operación a operación:
- la entrada y la salida;
- el **stop loss inicial**;
- el resultado neto de costes.

El money management con el que hiciste el backtest (lote fijo, ATR, riesgo fijo…) **da igual**: la app vuelve a
calcular el lote de cada operación con su propia regla. Puedes minar con 1 lote fijo y decidir el riesgo aquí.

**Cuenta.** Cada símbolo tiene una **ficha** en su bróker: valor del punto, paso y mínimo de lote, comisión y
spread (pestaña **Símbolos**). Cada perfil de firma opera en un bróker:

| Perfiles | Bróker |
| --- | --- |
| FTMO CFD | FTMO (CFD, MT5) |
| FTMO Futures | micros CME de FTMO |
| Lente AXI | AXI Select (MT5) |

Una cartera es de **un solo bróker**. Si cargas estrategias de un bróker en el perfil de otro (por ejemplo `.sqx`
de AXI en un reto FTMO), la app lo rechaza y te pide re-testear en SQX con los datos de ese bróker: los precios, la
hora del servidor y la comisión del backtest serían los de otro.

**Dimensionado.** Hay una sola regla: **riesgo fijo hasta el stop loss inicial**. Con un riesgo *f* (por ejemplo,
1 %) y una cuenta *A*, cada operación arriesga *f × A* dólares si salta su stop:
- `lote = f·A / (distancia al SL × valor del punto)`, redondeado hacia abajo al paso del bróker;
- si no llega al lote mínimo, se abre el mínimo y esa operación arriesga más que *f*. La app lo cuenta y te lo
  enseña.

**Motor.** Monte Carlo que baraja la historia de la cartera por **episodios**: tramos entre dos momentos en que todas
las estrategias están sin posición. Aplica las reglas de la firma día a día:
- pérdida diaria y máxima;
- objetivo;
- días mínimos;
- payouts.

Los episodios se barajan en bloques de tamaño medio automático (método de Politis-White sobre el resultado de cada
episodio medido en R, que no depende de *f*). Al lado verás la cifra **con bloque 1** (episodios sueltos): es la
prudente si la dependencia entre episodios de tu backtest no se repite.

## Usar tus propias estrategias

**Qué exportar de SQX**

- Los ficheros **`.sqx`** de tus estrategias sueltas, desde el databank. No sirven los portfolios del Portfolio
  Master.
- Cada operación tiene que llevar stop loss.
- Las salidas parciales (ATM) no están soportadas.
- Opcional: el OHLC de **1 minuto** del símbolo, exportado de SQX, para calcular el riesgo flotante minuto a minuto
  en vez de con la cota por el MAE.
  - El nombre del fichero es el que diga la ficha del símbolo (campo *Fichero OHLC M1*) y va en la carpeta OHLC de
    su bróker.
  - En el bróker FTMO CFD, esa carpeta es `risk_app/inputs/ohlc/` (clave `ohlc_m1` de la ficha); en AXI,
    `risk_app/inputs/ohlc_axi/`.

**Dónde ponerlo**

| Carpeta | Qué va |
| --- | --- |
| `risk_app/inputs/strategies/` | `.sqx` (puedes crear subcarpetas: cada una es un grupo en la pantalla) |
| `risk_app/inputs/ohlc/` | OHLC M1 de FTMO (opcional) |
| `risk_app/inputs/ohlc_axi/` | OHLC M1 de AXI (opcional) |

Esas carpetas **no se suben a git** (solo la demo y los `.gitkeep`): tus datos no salen de tu ordenador.

**Si tus exportaciones ya están en otra carpeta** (por ejemplo la de SQX), no hace falta copiarlas:
- copia `config/local.example.yaml` como `config/local.yaml` y pon en `extra_input_roots` tus carpetas (con `~`
  para tu usuario);
- o, sin fichero, usa la variable de entorno `RISK_APP_EXTRA_ROOTS`, con las rutas separadas por `:` en Mac/Linux y
  por `;` en Windows.

`python run_app.py --smoke` sigue funcionando con carpetas extra (solo avisa); `--smoke --strict` exige una copia
limpia, solo con la demo.

## Fichas de símbolo (pestaña Símbolos)

**Fichas oficiales**, verificadas, de solo lectura:

| Bróker | Símbolos SQX |
| --- | --- |
| FTMO (CFD, MT5) | `EURUSD_ftmo`, `GBPUSD_ftmo`, `AUDUSD_ftmo`, `XAUUSD_ftmo(2)`, `USATECHIDXUSD_ftmo(2)` |
| FTMO Futures (micros CME) | `@micro_nasdaq`, `@micro_rusell`, `@micro_oro`, `@micro_eurusd` |
| AXI Select (MT5) | `EURUSD_axi`, `NZDUSD_axi`, `ETHUSD_axi`, `XAUUSD_axi`, `XAGUSD_axi`, `USATECHIDXUSD_axi` |

El símbolo SQX tiene que coincidir **exactamente**, carácter a carácter, con el de tu `.sqx`. Las fichas oficiales
salen de `config/ftmo_cfd_costs.yaml` y `config/ftmo_futures_costs.yaml` (los símbolos de su lista `sqx_policy`) y de
`config/broker_axi.yaml`; `config/brokers.yaml` dice qué fichero es de cada bróker.

**Si tu símbolo no tiene ficha**, la estrategia aparece en la lista con la marca **sin ficha** y un botón
**Dar de alta**:
- abre una ficha nueva rellena con lo que trae el `.sqx`;
- esos campos salen marcados **sin verificar** hasta que los confirmes;
- el lote mínimo no viene en el `.sqx`: lo tecleas tú;
- eliges el bróker, uno oficial o uno tuyo (**+ Bróker**).

**La forma más fiable de rellenar una ficha** es la hoja de tu bróker:
- ejecuta en MT5 el script de SQX *Update_SQX_Instruments_information*;
- en la pestaña Símbolos, pulsa **Importar hoja de MT5** y elige el fichero *Human Readable Exported Information
  Sheet.csv*.

La app propone una ficha por símbolo y te enseña los **conflictos** si tu hoja no coincide con la ficha que ya
existe. Las fichas oficiales no se editan: si tu cuenta es distinta, crea un bróker tuyo.

Tus fichas y brókers se guardan en `config/symbols.user.yaml` (no se sube a git). Se pueden exportar e importar para
pasárselas a un compañero.

**Límites de esta versión:**
- solo cuentas en **USD**;
- hora del servidor "Nueva York + 7 h" (la de casi todos los brókers MT5, FTMO y AXI incluidos) o reloj de Nueva York
  (futuros).

## La receta para el EA

Debajo de los resultados, la tarjeta **Receta para el EA** te dice cómo configurar el money management de cada EA
para que opere lo que has simulado:

| Parámetro | Qué poner |
| --- | --- |
| Método | `FixedAmount` |
| `RiskedMoney` | *f × A*, en USD |
| `Decimals` y `mmStep` | Del paso de lote |
| `mmMultiplier` | 1 |
| `MaxLots` | El lote máximo de la simulación |
| `LotsIfNoMM` | El lote mínimo |

**`LotsIfNoMM` = lote mínimo es una condición, no una sugerencia.** Cuando el lote calculado redondea a cero, el EA
abre `LotsIfNoMM`, y el valor por defecto de SQX (1,0) abriría un lote entero.

- Si cambias la *f*, vuelve a copiar la receta.
- Las órdenes abiertas a la vez suman riesgo.
- Confirma en MT5 que tu cuenta está en USD.

## Reglas a tener en cuenta

- Una cartera no mezcla brókers distintos (ni futuros con CFD) ni CSV con `.sqx`.
- Con menos de 5 episodios "flat-flat" (periodos con todas las patas sin posición) el cálculo da error, y con menos
  de 30 avisa de que el remuestreo es poco fiable.
- Evalúa siempre fuera de la muestra de minado si puedes (selector *Corte IS/OOS*): con la historia completa, la
  probabilidad de pasar sale inflada si elegiste las estrategias sobre ese tramo.

## Vía CSV (antigua; necesaria para Apex)

La app aún acepta el CSV de trades `orderstocsv` de SQX, que es la única vía para el perfil **Apex** y una
alternativa en la lente **AXI**. Necesita:
- el CSV, en `risk_app/inputs/trades/`;
- el `.sqx` (o `.xml`) de la estrategia, para el tamaño de posición exacto;
- un OHLC M1 del símbolo en `risk_app/inputs/ohlc/`.

La app encuentra el OHLC por la **raíz del símbolo** (lo que hay antes del primer `.` o `_`): por ejemplo,
`XAUUSD_lo-que-sea.csv` sirve para los CSV de `XAUUSD.pa` o `XAUUSD_axi`. Se usa el primer fichero que coincida, así
que no pongas dos OHLC con la misma raíz.

Los símbolos de los CSV salen de `config/broker_axi.yaml`: un CSV cuyo símbolo no esté ahí da error.

La vía CSV se retirará cuando el perfil Apex acepte `.sqx`.

## Estructura

```
run_app.py            arranque en un solo proceso (API + front compilado)
poc_web/api/          API (FastAPI): carga de carteras, simulaciones, fichas de símbolo, perfiles
poc_web/web/          front (React + Vite); dist/ es la versión compilada
src/                  motor de riesgo (Monte Carlo, prop firms, tamaño de posición) y config de brokers
sqx/                  lectores de los .sqx de SQX (trades, money management)
risk_app/             lectura de entradas y tamaño exacto; inputs/ = tus datos (no versionados)
config/               perfiles de firmas, brókers, costes y símbolos (YAML)
tests/, poc_web/api/  tests (pytest)
```

**Fronteras y destino.** La agrupación anterior es una guía, no una restricción verificada:
- el motor (`src/risk`) no debe depender de los lectores de SQX, y la API orquesta ambos;
- el destino previsto es un **paquete instalable** (`pip install`) con esas fronteras explícitas;
- hoy los imports absolutos (`src.`, `sqx.`, `risk_app.`, `poc_web.`) funcionan porque la raíz del repositorio está
  en el path. Por eso `run_app.py` y `pytest` se lanzan desde esta carpeta.

## Tests

```
pip install -r requirements-dev.txt
python -m pytest -q tests poc_web/api
```

El CI de GitHub (`.github/workflows/ci.yml`) ejecuta los tests y `python run_app.py --smoke --strict` en macOS y
Windows, y compila el front con Node 22, en cada push y en cada PR.

## Contribuir

Issues y PRs bienvenidos, en castellano o en inglés. Cómo se integran (este repo es un espejo y los PRs no se
fusionan aquí) y qué contar en un fallo: `CONTRIBUTING.md`.

## Licencia

MIT (`LICENSE`). Las obras de terceros incluidas y sus licencias, entre ellas TradingView Lightweight Charts™ para
las gráficas: `THIRD_PARTY_NOTICES.md`.
