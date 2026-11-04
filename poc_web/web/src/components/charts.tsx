// Wrapper mínimo de lightweight-charts v5.
// El eje X es un índice (0..n-1); las etiquetas reales se inyectan vía
// tickMarkFormatter / timeFormatter — así funciona igual para f (0.25%…) que
// para challenges (0..20).
import { useEffect, useRef } from 'react'
import {
  ColorType,
  CrosshairMode,
  LineSeries,
  createChart,
  type IChartApi,
  type ISeriesApi,
  type Time,
  type UTCTimestamp,
} from 'lightweight-charts'

export interface ChartSeries {
  name: string
  color: string
  data: { x: number; y: number }[]
  dashed?: boolean
  width?: 1 | 2 | 3
}

interface Props {
  series: ChartSeries[]
  xLabels: string[]
  height?: number
  yFormat?: (v: number) => string
}

export function LineChart({ series, xLabels, height = 260, yFormat }: Props) {
  const boxRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const seriesRefs = useRef<ISeriesApi<'Line'>[]>([])
  const labelsRef = useRef(xLabels)
  labelsRef.current = xLabels

  useEffect(() => {
    if (!boxRef.current) return
    const chart = createChart(boxRef.current, {
      autoSize: true,
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#a1a1aa',
        fontSize: 11,
        // La licencia de lightweight-charts exige un enlace a tradingview.com en la página: el logo lo cumple (2026-11-04).
        attributionLogo: true,
      },
      grid: {
        vertLines: { color: 'rgba(63,63,70,0.2)' },
        horzLines: { color: 'rgba(63,63,70,0.2)' },
      },
      rightPriceScale: { borderVisible: false },
      timeScale: {
        borderVisible: false,
        tickMarkFormatter: (t: Time) => labelsRef.current[Number(t)] ?? String(t),
      },
      crosshair: { mode: CrosshairMode.Normal },
      localization: {
        timeFormatter: (t: Time) => labelsRef.current[Number(t)] ?? String(t),
        ...(yFormat ? { priceFormatter: yFormat } : {}),
      },
      handleScale: { axisPressedMouseMove: false },
    })
    chartRef.current = chart
    return () => {
      chart.remove()
      chartRef.current = null
      seriesRefs.current = []
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    const chart = chartRef.current
    if (!chart) return
    for (const s of seriesRefs.current) chart.removeSeries(s)
    seriesRefs.current = series.map((s) => {
      const ser = chart.addSeries(LineSeries, {
        color: s.color,
        lineWidth: s.width ?? 2,
        lineStyle: s.dashed ? 2 : 0,
        priceLineVisible: false,
        lastValueVisible: !s.dashed,
        pointMarkersVisible: false,
      })
      ser.setData(s.data.map((d) => ({ time: d.x as UTCTimestamp, value: d.y })))
      return ser
    })
    chart.timeScale().fitContent()
  }, [series])

  return (
    <div>
      <div ref={boxRef} style={{ height }} />
      <div className="flex flex-wrap gap-4 px-4 pb-3">
        {series.map((s) => (
          <span key={s.name} className="flex items-center gap-1.5 text-[11px] text-zinc-400">
            <span className="inline-block h-0.5 w-4 rounded" style={{ backgroundColor: s.color }} />
            {s.name}
          </span>
        ))}
      </div>
    </div>
  )
}
