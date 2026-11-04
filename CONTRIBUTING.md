# Cómo contribuir

> **English:** issues and pull requests in English are welcome. The docs are in Spanish; the
> process below is the same in any language.

Gracias por querer mejorar la Risk App. Antes de nada, dos cosas que conviene saber.

## Cómo se mantiene este repo

Es el **espejo público** de un repo de trabajo privado. Cada actualización llega aquí como una
instantánea (commits `Export snapshot …` con la línea `Exported-From`) que regenera el repo entero.
Con los PRs, el flujo es:

1. Abres el PR aquí y se revisa aquí.
2. Si se acepta, el cambio se integra a mano en el repo original. El PR **no se fusiona** aquí,
   porque cada instantánea regenera el repo entero desde el original.
3. La siguiente instantánea lo incluye y su commit lo cita (`PR #N: portado`). Entonces se cierra
   el PR.

Por eso, a veces el cambio llega con otra forma (mismo efecto, estilo del resto del código). No es
un rechazo.

## Issues

Para un fallo, cuenta:

- sistema (macOS o Windows, y versión) y `python --version`;
- qué hiciste en la app (lente, firma y reto, etapa AXI, rejilla de *f*) y qué esperabas;
- el mensaje de error o el aviso que salió, y la salida de la terminal donde corre `run_app.py`;
- si pasa con la demo (`risk_app/inputs/strategies/demo_*`), dilo: así se puede reproducir.

**No subas tus estrategias** (`.sqx`) ni tus CSV de operaciones: son tu trabajo y casi nunca hacen
falta para diagnosticar. Si un log lleva rutas o nombres que prefieres no enseñar, recórtalos.

## Pull requests

- Una rama por cambio, PRs pequeños y con un objetivo claro.
- `python -m pytest -q` en verde y `python run_app.py --smoke --strict` OK; la CI del PR (macOS y
  Windows) también.
- Si tocas el front (`poc_web/web/src/`), compila con `npm ci && npm run build` en `poc_web/web/` y
  cuenta en el PR qué pantalla cambia.
- Si tocas el motor o las reglas de una firma, di de dónde sale la regla (página oficial de la
  firma) y añade un test.
- Sigue el estilo del código que tocas. Los comentarios en castellano o en inglés valen.
- Nada de rutas personales, claves ni estrategias reales en el PR.

Al enviar un PR aceptas que tu contribución se publique bajo la licencia MIT del repo.

## Trato

Sé respetuoso y concreto. Las discusiones van sobre el código y los hechos, no sobre las personas.
