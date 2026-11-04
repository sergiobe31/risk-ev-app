// Semáforo de régimen ATR (producto gap-régimen-ATR, condición C1): el
// trigger compuesto NO va en la API — se compone aquí, en el frontend.
// Dispara solo si el percentil del sl vivo supera el umbral Y la pata
// quedaría en clamp hoy (sin la 2ª cláusula el aviso es cosmético).
// Veracidad, nunca `=== true` (`wouldClamp` puede venir como 0/1 del
// serializador `_jsonable`).
export function triggerCompuesto(
  pctile: number | null | undefined,
  wouldClamp: boolean | number | null | undefined,
  threshold: number,
): boolean | null {
  if (pctile == null || wouldClamp == null) return null
  return pctile >= threshold && Boolean(wouldClamp)
}
