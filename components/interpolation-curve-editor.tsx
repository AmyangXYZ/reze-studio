"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { cn } from "@/lib/utils"

const SIZE = 160
const PAD = 16
const GX = PAD
const GY = PAD
const GW = SIZE - PAD * 2
const GH = SIZE - PAD * 2

/** Main spline stroke (cyan, readable on dark grid). */
const CURVE_CYAN = "#22d3ee"
/** Dashed handles from endpoints to P1/P2 — brighter than grid so tangents read clearly. */
const HANDLE_DASH = "rgba(255,255,255,0.42)"
const CP_RECT = 8
const CP_HALF = CP_RECT / 2
const CP_HIT = 11

export type CurvePoint = { x: number; y: number }

export const PRESETS: { label: string; p1: CurvePoint; p2: CurvePoint }[] = [
  { label: "Linear", p1: { x: 20, y: 20 }, p2: { x: 107, y: 107 } },
  { label: "In", p1: { x: 64, y: 0 }, p2: { x: 107, y: 107 } },
  { label: "Out", p1: { x: 20, y: 20 }, p2: { x: 64, y: 127 } },
  { label: "InOut", p1: { x: 64, y: 0 }, p2: { x: 64, y: 127 } },
  { label: "Slow In", p1: { x: 100, y: 0 }, p2: { x: 107, y: 107 } },
  { label: "Slow Out", p1: { x: 20, y: 20 }, p2: { x: 27, y: 127 } },
  { label: "Slow IO", p1: { x: 100, y: 0 }, p2: { x: 27, y: 127 } },
  { label: "Over", p1: { x: 0, y: 127 }, p2: { x: 127, y: 0 } },
]

function toCanvas(px: number, py: number) {
  return { x: GX + (px / 127) * GW, y: GY + GH - (py / 127) * GH }
}

function fromCanvas(cx: number, cy: number): CurvePoint {
  return {
    x: Math.round(Math.max(0, Math.min(127, ((cx - GX) / GW) * 127))),
    y: Math.round(Math.max(0, Math.min(127, ((GY + GH - cy) / GH) * 127))),
  }
}

function bezierPoint(
  t: number,
  p0x: number,
  p0y: number,
  c1x: number,
  c1y: number,
  c2x: number,
  c2y: number,
  p3x: number,
  p3y: number,
) {
  const u = 1 - t
  return {
    x: u * u * u * p0x + 3 * u * u * t * c1x + 3 * u * t * t * c2x + t * t * t * p3x,
    y: u * u * u * p0y + 3 * u * u * t * c1y + 3 * u * t * t * c2y + t * t * t * p3y,
  }
}

type InterpolationCurveEditorProps = {
  p1: CurvePoint
  p2: CurvePoint
  disabled?: boolean
  onChange: (p1: CurvePoint, p2: CurvePoint) => void
}

/** VMD-style cubic Bézier editor in 127×127 space (same as reference HTML). */
export function InterpolationCurveEditor({ p1, p2, disabled, onChange }: InterpolationCurveEditorProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const dragging = useRef<"p1" | "p2" | null>(null)
  const [dpr, setDpr] = useState(1)

  useEffect(() => {
    setDpr(typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1)
  }, [])

  const drawWith = useCallback((a: CurvePoint, b: CurvePoint) => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    ctx.setTransform(1, 0, 0, 1, 0, 0)
    ctx.scale(dpr, dpr)
    ctx.clearRect(0, 0, SIZE, SIZE)

    // Grid lines
    ctx.strokeStyle = "rgba(255,255,255,0.15)"
    ctx.lineWidth = 0.5
    for (let i = 0; i <= 4; i++) {
      const x = GX + (i / 4) * GW
      const y = GY + (i / 4) * GH
      ctx.beginPath()
      ctx.moveTo(x, GY)
      ctx.lineTo(x, GY + GH)
      ctx.stroke()
      ctx.beginPath()
      ctx.moveTo(GX, y)
      ctx.lineTo(GX + GW, y)
      ctx.stroke()
    }

    // Diagonal
    ctx.strokeStyle = "rgba(255,255,255,0.1)"
    ctx.lineWidth = 0.5
    ctx.setLineDash([3, 3])
    ctx.beginPath()
    ctx.moveTo(GX, GY + GH)
    ctx.lineTo(GX + GW, GY)
    ctx.stroke()
    ctx.setLineDash([])

    const s = toCanvas(0, 0)
    const e = toCanvas(127, 127)
    const c1 = toCanvas(a.x, a.y)
    const c2 = toCanvas(b.x, b.y)

    // Handle dashes
    ctx.strokeStyle = HANDLE_DASH
    ctx.lineWidth = 1
    ctx.setLineDash([4, 3])
    ctx.beginPath()
    ctx.moveTo(s.x, s.y)
    ctx.lineTo(c1.x, c1.y)
    ctx.stroke()
    ctx.beginPath()
    ctx.moveTo(e.x, e.y)
    ctx.lineTo(c2.x, c2.y)
    ctx.stroke()
    ctx.setLineDash([])

    // Curve
    ctx.beginPath()
    ctx.moveTo(s.x, s.y)
    for (let t = 0; t <= 1; t += 0.01) {
      const pt = bezierPoint(t, s.x, s.y, c1.x, c1.y, c2.x, c2.y, e.x, e.y)
      ctx.lineTo(pt.x, pt.y)
    }
    ctx.strokeStyle = CURVE_CYAN
    ctx.lineWidth = 1.75
    ctx.stroke()

      // Endpoints
      ;[s, e].forEach((pt) => {
        ctx.fillStyle = "rgba(255,255,255,0.4)"
        ctx.beginPath()
        ctx.arc(pt.x, pt.y, 3, 0, Math.PI * 2)
        ctx.fill()
      })

      // Control points with labels above
      ;[c1, c2].forEach((pt, i) => {
        const cpColor = i === 0 ? "#e25555" : "#44bb55"
        ctx.fillStyle = cpColor
        ctx.fillRect(pt.x - CP_HALF, pt.y - CP_HALF, CP_RECT, CP_RECT)
        ctx.strokeStyle = "rgba(255,255,255,0.45)"
        ctx.lineWidth = 1
        ctx.strokeRect(pt.x - CP_HALF, pt.y - CP_HALF, CP_RECT, CP_RECT)

        const label = i === 0 ? `(${a.x}, ${a.y})` : `(${b.x}, ${b.y})`
        ctx.font = "9px -apple-system, sans-serif"
        ctx.textAlign = "center"
        ctx.textBaseline = "bottom"
        ctx.fillStyle = cpColor
        ctx.fillText(label, pt.x, pt.y - CP_HALF - 2)
      })
  }, [dpr])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    canvas.width = SIZE * dpr
    canvas.height = SIZE * dpr
    drawWith(p1, p2)
  }, [dpr, drawWith, p1, p2])

  const getMousePos = (e: React.MouseEvent | React.PointerEvent) => {
    const canvas = canvasRef.current
    if (!canvas) return { x: 0, y: 0 }
    const rect = canvas.getBoundingClientRect()
    const scaleX = SIZE / rect.width
    const scaleY = SIZE / rect.height
    return { x: (e.clientX - rect.left) * scaleX, y: (e.clientY - rect.top) * scaleY }
  }

  const hitCp = (mx: number, my: number, cx: number, cy: number) =>
    Math.abs(mx - cx) <= CP_HIT && Math.abs(my - cy) <= CP_HIT

  /** P2 is painted after P1 — test P2 first so stacked handles prefer the top control. */
  const pickDragTarget = (m: { x: number; y: number }, a: CurvePoint, b: CurvePoint): "p1" | "p2" | null => {
    const c1 = toCanvas(a.x, a.y)
    const c2 = toCanvas(b.x, b.y)
    if (hitCp(m.x, m.y, c2.x, c2.y)) return "p2"
    if (hitCp(m.x, m.y, c1.x, c1.y)) return "p1"
    return null
  }

  const setCanvasCursor = (e: React.PointerEvent<HTMLCanvasElement> | null, draggingNow: boolean) => {
    const el = canvasRef.current
    if (!el || disabled) return
    if (draggingNow) {
      el.style.cursor = "grabbing"
      return
    }
    if (!e) {
      el.style.cursor = "default"
      return
    }
    const m = getMousePos(e)
    const t = pickDragTarget(m, p1, p2)
    el.style.cursor = t ? "grab" : "default"
  }

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (disabled) return
    const m = getMousePos(e)
    const t = pickDragTarget(m, p1, p2)
    if (!t) return
    e.preventDefault()
    dragging.current = t
    setCanvasCursor(e, true)
    e.currentTarget.setPointerCapture(e.pointerId)
    const pt = fromCanvas(m.x, m.y)
    if (t === "p1") {
      onChange(pt, p2)
      drawWith(pt, p2)
    } else {
      onChange(p1, pt)
      drawWith(p1, pt)
    }
  }

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (disabled) return
    if (dragging.current) {
      const m = getMousePos(e)
      const pt = fromCanvas(m.x, m.y)
      if (dragging.current === "p1") {
        onChange(pt, p2)
        drawWith(pt, p2)
      } else {
        onChange(p1, pt)
        drawWith(p1, pt)
      }
      setCanvasCursor(e, true)
      return
    }
    setCanvasCursor(e, false)
  }

  const onPointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    dragging.current = null
    try {
      e.currentTarget.releasePointerCapture(e.pointerId)
    } catch {
      /* */
    }
    setCanvasCursor(e, false)
  }

  const onPointerLeave = () => {
    if (!dragging.current) setCanvasCursor(null, false)
  }

  const onLostPointerCapture = () => {
    dragging.current = null
    setCanvasCursor(null, false)
  }

  return (
    <div
      className={cn(
        "shrink-0 rounded border border-border bg-background p-0.5",
        disabled && "pointer-events-none opacity-50",
      )}
      style={{ width: SIZE + 4, height: SIZE + 4 }}
    >
      <canvas
        ref={canvasRef}
        width={SIZE}
        height={SIZE}
        className="block cursor-default rounded-[3px]"
        style={{ width: SIZE, height: SIZE }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onPointerLeave={onPointerLeave}
        onLostPointerCapture={onLostPointerCapture}
      />
    </div>
  )
}
