"use client"

import { useState, useRef, useEffect, useCallback, useMemo } from "react"
import {
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
  Pause,
  Play,
  ZoomIn,
  ZoomOut,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import {
  type AnimationClip,
  type BoneKeyframe,
  C,
  FONT,
  DOPE_H,
  RULER_H,
  LABEL_W,
  DOT_R,
  DIAMOND,
  MAX_PX,
  minPxPerFrameForViewport,
  TABS,
  boneDisplayLabel,
  getChannelsForTab,
  getAxisConfig,
  bezierY,
  ALL_CHANNELS,
} from "@/lib/animation"

/** Scrub playhead 0…frameCount — track/thumb aligned with toolbar (Tailwind tokens). */
function TransportFrameSlider({
  frameCount,
  value,
  onChange,
}: {
  frameCount: number
  value: number
  onChange: (f: number) => void
}) {
  const trackRef = useRef<HTMLDivElement>(null)
  const dragging = useRef(false)

  const setFromClientX = useCallback(
    (clientX: number) => {
      const el = trackRef.current
      if (!el || frameCount <= 0) return
      const rect = el.getBoundingClientRect()
      const t = Math.max(0, Math.min(1, (clientX - rect.left) / Math.max(rect.width, 1)))
      onChange(Math.round(t * frameCount))
    },
    [frameCount, onChange],
  )

  useEffect(() => {
    const onMove = (e: PointerEvent) => {
      if (!dragging.current) return
      setFromClientX(e.clientX)
    }
    const onUp = () => {
      dragging.current = false
    }
    window.addEventListener("pointermove", onMove)
    window.addEventListener("pointerup", onUp)
    window.addEventListener("pointercancel", onUp)
    return () => {
      window.removeEventListener("pointermove", onMove)
      window.removeEventListener("pointerup", onUp)
      window.removeEventListener("pointercancel", onUp)
    }
  }, [setFromClientX])

  const disabled = frameCount <= 0
  const pct = !disabled && frameCount > 0 ? (value / frameCount) * 100 : 0

  return (
    <div className="mx-1 ml-0.5 flex shrink-0 select-none items-center">
      <div
        ref={trackRef}
        role="slider"
        aria-label="Scrub playhead"
        aria-valuemin={0}
        aria-valuemax={frameCount}
        aria-valuenow={Math.round(value)}
        aria-disabled={disabled}
        tabIndex={disabled ? -1 : 0}
        onKeyDown={(e) => {
          if (disabled) return
          if (e.key === "ArrowLeft" || e.key === "ArrowDown")
            onChange(Math.max(0, Math.round(value) - 1))
          if (e.key === "ArrowRight" || e.key === "ArrowUp")
            onChange(Math.min(frameCount, Math.round(value) + 1))
        }}
        onPointerDown={(e) => {
          if (disabled || e.button !== 0) return
          dragging.current = true
          setFromClientX(e.clientX)
          e.preventDefault()
        }}
        className={cn(
          "relative h-5 w-[88px] shrink-0 touch-none",
          disabled ? "pointer-events-none opacity-15" : "cursor-grab",
        )}
      >
        <div className="absolute left-0 right-0 top-1/2 h-0.5 -translate-y-1/2 rounded-[1px] bg-border" />
        <div
          className="pointer-events-none absolute top-1/2 size-[9px] -translate-x-1/2 -translate-y-1/2 rounded-full border-[1.5px] border-muted-foreground bg-secondary box-border"
          style={{ left: `${pct}%` }}
        />
      </div>
    </div>
  )
}

// ─── Selection type ──────────────────────────────────────────────────────
export interface SelectedKeyframe {
  bone?: string
  frame: number
  channel?: string
  type: "dope" | "curve"
}

function ZoomRuler({
  min,
  max,
  value,
  onChange,
}: {
  min: number
  max: number
  value: number
  onChange: (v: number) => void
}) {
  const trackRef = useRef<HTMLDivElement>(null)
  const dragging = useRef(false)
  const span = max - min

  const setFromClientX = useCallback(
    (clientX: number) => {
      const el = trackRef.current
      if (!el) return
      const rect = el.getBoundingClientRect()
      const s = max - min
      const t = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width))
      if (s <= 0) {
        onChange(min)
        return
      }
      const raw = min + t * s
      const snap = (v: number) => (s < 2 ? Math.round(v * 100) / 100 : Math.round(v * 2) / 2)
      onChange(Math.max(min, Math.min(max, snap(raw))))
    },
    [min, max, onChange],
  )

  useEffect(() => {
    const onMove = (e: PointerEvent) => {
      if (!dragging.current) return
      setFromClientX(e.clientX)
    }
    const onUp = () => {
      dragging.current = false
    }
    window.addEventListener("pointermove", onMove)
    window.addEventListener("pointerup", onUp)
    window.addEventListener("pointercancel", onUp)
    return () => {
      window.removeEventListener("pointermove", onMove)
      window.removeEventListener("pointerup", onUp)
      window.removeEventListener("pointercancel", onUp)
    }
  }, [setFromClientX])

  const pct = span > 0 ? ((value - min) / span) * 100 : 50
  const snapVal = (v: number) => (span < 2 ? Math.round(v * 100) / 100 : Math.round(v * 2) / 2)
  const nudgeDelta = span < 2 ? 0.05 : 0.5
  const nudge = (dir: -1 | 1) =>
    onChange(Math.max(min, Math.min(max, snapVal(value + dir * nudgeDelta))))

  return (
    <div className="flex shrink-0 select-none items-center gap-1 text-muted-foreground">
      <Button
        type="button"
        variant="ghost"
        size="icon-xs"
        aria-label="Zoom out"
        className={cn(
          "size-5 shrink-0 overflow-hidden p-0 text-muted-foreground",
          "hover:bg-transparent dark:hover:bg-transparent active:bg-muted/50",
          "focus-visible:outline-none focus-visible:ring-0",
        )}
        onClick={() => nudge(-1)}
        onPointerDown={(e) => e.stopPropagation()}
      >
        <ZoomOut size={12} strokeWidth={1.75} />
      </Button>
      <div
        ref={trackRef}
        role="slider"
        aria-label="Timeline zoom"
        aria-valuemin={min}
        aria-valuemax={max}
        aria-valuenow={value}
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "ArrowLeft" || e.key === "ArrowDown") nudge(-1)
          if (e.key === "ArrowRight" || e.key === "ArrowUp") nudge(1)
        }}
        onPointerDown={(e) => {
          if (e.button !== 0) return
          dragging.current = true
          setFromClientX(e.clientX)
          e.preventDefault()
        }}
        className="relative h-4 w-14 shrink-0 cursor-grab touch-none"
      >
        <div className="absolute left-0 right-0 top-1/2 h-0.5 -translate-y-1/2 rounded-[1px] bg-border" />
        <div
          className="pointer-events-none absolute top-1/2 size-[9px] -translate-x-1/2 -translate-y-1/2 rounded-full border-[1.5px] border-muted-foreground bg-transparent box-border"
          style={{ left: `${pct}%` }}
        />
      </div>
      <Button
        type="button"
        variant="ghost"
        size="icon-xs"
        aria-label="Zoom in"
        className={cn(
          "size-5 shrink-0 overflow-hidden p-0 text-muted-foreground",
          "hover:bg-transparent dark:hover:bg-transparent active:bg-muted/50",
          "focus-visible:outline-none focus-visible:ring-0",
        )}
        onClick={() => nudge(1)}
        onPointerDown={(e) => e.stopPropagation()}
      >
        <ZoomIn size={12} strokeWidth={1.75} />
      </Button>
    </div>
  )
}

// ─── Canvas ──────────────────────────────────────────────────────────────
interface TimelineCanvasProps {
  clip: AnimationClip
  pxPerFrame: number
  scrollX: number
  currentFrame: number
  activeBone: string | null
  visibleBones: string[]
  selectedKeyframes: SelectedKeyframe[]
  tab: string
  onSetCurrentFrame: (f: number) => void
  onSelectKeyframe: (kf: SelectedKeyframe, multi: boolean) => void
  onMoveDopeKeyframe: (from: number, to: number) => void
  onMoveCurveKeyframe: (bone: string, from: number, channel: string, toFrame: number, dv: number) => void
}

function TimelineCanvas({
  clip,
  pxPerFrame,
  scrollX,
  currentFrame,
  activeBone,
  visibleBones,
  selectedKeyframes,
  tab,
  onSetCurrentFrame,
  onSelectKeyframe,
  onMoveDopeKeyframe,
  onMoveCurveKeyframe,
}: TimelineCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const rafRef = useRef<number | null>(null)
  const sizeRef = useRef({ w: 0, h: 0, dpr: 0 })
  const drag = useRef<{
    type: string
    frame?: number
    bone?: string
    channel?: string
    startX?: number
    startY?: number
  } | null>(null)

  const getDopeFrames = useCallback(() => {
    const frames = new Map<number, number>()
    if (activeBone) {
      const track = clip.boneTracks.get(activeBone)
      if (track) for (const kf of track) frames.set(kf.frame, 1)
    } else {
      for (const name of visibleBones) {
        const track = clip.boneTracks.get(name)
        if (track) for (const kf of track) frames.set(kf.frame, (frames.get(kf.frame) || 0) + 1)
      }
    }
    return frames
  }, [clip, visibleBones, activeBone])

  const draw = useCallback(() => {
    const el = canvasRef.current
    if (!el) return
    const ctx = el.getContext("2d")
    if (!ctx) return
    const dpr = Math.min(4, Math.max(1, (window.devicePixelRatio || 1) * 1.5))
    const w = el.clientWidth,
      h = el.clientHeight
    const backingW = Math.max(1, Math.floor(w * dpr))
    const backingH = Math.max(1, Math.floor(h * dpr))
    const size = sizeRef.current
    if (size.w !== backingW || size.h !== backingH || size.dpr !== dpr) {
      el.width = backingW
      el.height = backingH
      sizeRef.current = { w: backingW, h: backingH, dpr }
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, w, h)

    const ox = LABEL_W - scrollX
    const dopeY = h - DOPE_H
    const curveTop = RULER_H
    const curveBot = dopeY - 1
    const curveH = curveBot - curveTop

    const channels = getChannelsForTab(tab)
    const ax = getAxisConfig(tab)
    const toY = (v: number) => curveTop + (1 - (v - ax.min) / (ax.max - ax.min)) * curveH
    const toX = (f: number) => ox + f * pxPerFrame

    // ── Backgrounds ──
    ctx.fillStyle = "rgba(0,0,0,0)"
    ctx.fillRect(0, 0, w, dopeY)
    ctx.fillStyle = C.dopeBg
    ctx.fillRect(0, dopeY, w, DOPE_H)
    ctx.strokeStyle = C.dopeBorder
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(0, dopeY + 0.5)
    ctx.lineTo(w, dopeY + 0.5)
    ctx.stroke()

    // ── Ruler ──
    ctx.fillStyle = C.ruler
    ctx.fillRect(0, 0, w, RULER_H)
    ctx.strokeStyle = C.border
    ctx.beginPath()
    ctx.moveTo(0, RULER_H - 0.5)
    ctx.lineTo(w, RULER_H - 0.5)
    ctx.stroke()

    const fStep = pxPerFrame >= 12 ? 1 : pxPerFrame >= 6 ? 5 : 10
    const fMajor = fStep * 10
    const rulerFontPx = 9
    ctx.font = `${rulerFontPx}px ${FONT}`
    ctx.textAlign = "center"
    ctx.textBaseline = "bottom"
    const rulerTickTop = (maj: boolean) => (maj ? 2 : RULER_H - 4)
    const minRulerLabelGapPx = 32
    let lastRulerLabelX = -1e9
    for (let f = 0; f <= clip.frameCount; f += fStep) {
      const x = ox + f * pxPerFrame
      if (x < LABEL_W - 10 || x > w + 10) continue
      const isM = f % fMajor === 0
      ctx.strokeStyle = isM ? C.rulerMajor : C.rulerTick
      ctx.beginPath()
      ctx.moveTo(Math.round(x) + 0.5, rulerTickTop(isM))
      ctx.lineTo(Math.round(x) + 0.5, RULER_H)
      ctx.stroke()
      if (isM && x - lastRulerLabelX >= minRulerLabelGapPx) {
        ctx.fillStyle = C.rulerText
        ctx.fillText(String(f), x, RULER_H - 2)
        lastRulerLabelX = x
      }
    }

    // ── Value plot: fixed left strip (doesn’t scroll with ox) + Y ticks/labels ──
    ctx.fillStyle = C.ruler
    ctx.fillRect(0, curveTop, LABEL_W, curveBot - curveTop)

    ctx.font = `9px ${FONT}`
    const isRight = false
    const isRotAxis = channels[0]?.group === "rot"
    const vSteps = Math.max(0, Math.ceil((ax.max - ax.min) / ax.subStep))
    for (let i = 0; i <= vSteps; i++) {
      const v = Math.min(ax.max, ax.min + i * ax.subStep)
      let y = toY(v)
      y = Math.max(curveTop, Math.min(curveBot, y))
      const isZero = Math.abs(v) < 0.001
      const isMajor = Math.abs(v % ax.step) < 0.001
      const stroke = isZero ? C.axisZero : isMajor ? C.axis : C.grid
      ctx.strokeStyle = stroke
      ctx.lineWidth = isZero ? 1 : 0.5
      // Tick into the fixed left gutter (value axis) so scale stays visible when scrolled
      ctx.beginPath()
      ctx.moveTo(LABEL_W - (isMajor || isZero ? 5 : 3), y)
      ctx.lineTo(LABEL_W, y)
      ctx.stroke()
      ctx.beginPath()
      ctx.moveTo(LABEL_W, y)
      ctx.lineTo(w, y)
      ctx.stroke()

      if (isMajor || isZero) {
        ctx.fillStyle = C.rulerText
        ctx.textAlign = "right"
        ctx.textBaseline = "middle"
        const label = isRotAxis
          ? `${Math.round(v)}°`
          : Math.abs(v) < 0.001
            ? "0"
            : Math.abs(v - Math.round(v)) < 0.05
              ? String(Math.round(v))
              : v.toFixed(1)
        ctx.fillText(label, LABEL_W - 6, y)
      }
    }

    // Full-height Y-axis at plot left (screen-fixed at LABEL_W when scrollX moves content)
    ctx.strokeStyle = C.axis
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(LABEL_W + 0.5, curveTop)
    ctx.lineTo(LABEL_W + 0.5, curveBot)
    ctx.stroke()

    ctx.textAlign = "left"
    ctx.textBaseline = "alphabetic"

    // Vertical frame grid (skip x = LABEL_W — Y-axis above)
    ctx.lineWidth = 0.5
    for (let f = 0; f <= clip.frameCount; f += fStep) {
      const x = toX(f)
      if (x <= LABEL_W || x > w) continue
      ctx.strokeStyle = f % fMajor === 0 ? C.axis : C.grid
      ctx.beginPath()
      ctx.moveTo(x, curveTop)
      ctx.lineTo(x, curveBot)
      ctx.stroke()
    }

    // ── Curves ──
    if (activeBone) {
      const keyframes = clip.boneTracks.get(activeBone)
      if (keyframes && keyframes.length > 0) {
        const isSingle = channels.length === 1

        for (const ch of channels) {
          const interpKey =
            ch.group === "rot"
              ? "rotation"
              : ch.key === "tx"
                ? "translationX"
                : ch.key === "ty"
                  ? "translationY"
                  : "translationZ"

          // Draw curve
          ctx.strokeStyle = ch.color
          ctx.lineWidth = isSingle ? 2 : 1.2
          ctx.beginPath()
          let started = false
          for (let i = 0; i < keyframes.length; i++) {
            const kf = keyframes[i]
            const val = ch.get(kf)
            const x = toX(kf.frame)
            if (!started) {
              ctx.moveTo(x, toY(val))
              started = true
              continue
            }
            const prev = keyframes[i - 1]
            const prevVal = ch.get(prev)
            const prevX = toX(prev.frame)
            const cp = ch.group === "rot" ? kf.interpolation.rotation : kf.interpolation[interpKey as keyof typeof kf.interpolation] as [{ x: number; y: number }, { x: number; y: number }]
            const segs = Math.max(12, Math.ceil((x - prevX) / 3))
            for (let s = 1; s <= segs; s++) {
              const t = s / segs
              const interp = bezierY(cp[0], cp[1], t)
              ctx.lineTo(prevX + (x - prevX) * t, toY(prevVal + (val - prevVal) * interp))
            }
          }
          ctx.stroke()

          // Dots
          for (const kf of keyframes) {
            const val = ch.get(kf)
            const x = toX(kf.frame)
            if (x < LABEL_W - 8 || x > w + 8) continue
            const isSel = selectedKeyframes.some(
              (s) => s.bone === activeBone && s.frame === kf.frame && s.channel === ch.key,
            )
            ctx.beginPath()
            ctx.arc(x, toY(val), isSel ? DOT_R + 1.5 : DOT_R, 0, Math.PI * 2)
            ctx.fillStyle = isSel ? C.keyDotSel : ch.color
            ctx.fill()
            if (isSel) {
              ctx.strokeStyle = ch.color
              ctx.lineWidth = 2
              ctx.stroke()
            }
          }
        }

        // Value readout at playhead
        ctx.font = `10px ${FONT}`
        ctx.textBaseline = "top"
        const readoutX = isRight ? LABEL_W + 8 : w - 8
        ctx.textAlign = isRight ? "left" : "right"
        channels.forEach((ch, i) => {
          let val = 0
          for (const k of keyframes) {
            if (k.frame <= currentFrame) val = ch.get(k)
          }
          ctx.fillStyle = ch.color
          const display = ch.group === "rot" ? `${val.toFixed(1)}°` : val.toFixed(2)
          ctx.fillText(`${ch.label}: ${display}`, readoutX, curveTop + 5 + i * 13)
        })
      } else {
        ctx.fillStyle = C.label
        ctx.font = `13px ${FONT}`
        ctx.textAlign = "center"
        ctx.textBaseline = "middle"
        ctx.fillText("No keyframes — " + boneDisplayLabel(activeBone), (w + LABEL_W) / 2, (curveTop + curveBot) / 2)
      }
    } else {
      ctx.fillStyle = C.label
      ctx.font = `13px ${FONT}`
      ctx.textAlign = "center"
      ctx.textBaseline = "middle"
      ctx.fillText("Select a bone to view curves", (w + LABEL_W) / 2, (curveTop + curveBot) / 2)
    }

    // ── Dopesheet ──
    const frames = getDopeFrames()
    const maxCount = Math.max(1, ...frames.values())
    const dopeMid = dopeY + DOPE_H * 0.4

    // Dope grid
    ctx.lineWidth = 0.3
    for (let f = 0; f <= clip.frameCount; f += fStep) {
      const x = toX(f)
      if (x < LABEL_W || x > w) continue
      ctx.strokeStyle = C.grid
      ctx.beginPath()
      ctx.moveTo(x, dopeY + 1)
      ctx.lineTo(x, h)
      ctx.stroke()
    }

    ctx.font = `10px ${FONT}`
    ctx.textAlign = "center"
    const sortedDope = Array.from(frames.entries()).sort((a, b) => a[0] - b[0])
    for (const [frame, count] of sortedDope) {
      const x = toX(frame)
      if (x < LABEL_W - DIAMOND || x > w + DIAMOND) continue
      const isSel = selectedKeyframes.some((s) => s.frame === frame && s.type === "dope")
      const intensity = activeBone ? 0.85 : 0.4 + 0.6 * (count / maxCount)

      ctx.save()
      ctx.translate(x, dopeMid)
      ctx.rotate(Math.PI / 4)
      const sz = DIAMOND + (count > 1 && !activeBone ? 1.5 : 0)
      ctx.fillStyle = isSel ? C.diamondSel : `rgba(170,170,195,${intensity})`
      ctx.fillRect(-sz / 2, -sz / 2, sz, sz)
      if (isSel) {
        ctx.strokeStyle = "rgba(156,163,175,0.35)"
        ctx.lineWidth = 1
        ctx.strokeRect(-sz / 2 - 1, -sz / 2 - 1, sz + 2, sz + 2)
      }
      ctx.restore()
    }

    // Dopesheet label
    ctx.fillStyle = C.dopeBg
    ctx.fillRect(0, dopeY, LABEL_W, DOPE_H)
    ctx.strokeStyle = C.border
    ctx.beginPath()
    ctx.moveTo(LABEL_W - 0.5, dopeY)
    ctx.lineTo(LABEL_W - 0.5, h)
    ctx.stroke()
    ctx.fillStyle = C.dopeLabel
    ctx.font = `10px ${FONT}`
    ctx.textAlign = "right"
    ctx.textBaseline = "middle"
    ctx.fillText("Keys", LABEL_W - 6, dopeMid + 2)

    // ── Playhead ──
    const px = toX(currentFrame)
    if (px >= LABEL_W && px <= w) {
      const g = ctx.createLinearGradient(px - 14, 0, px + 14, 0)
      g.addColorStop(0, "transparent")
      g.addColorStop(0.5, C.playheadGlow)
      g.addColorStop(1, "transparent")
      ctx.fillStyle = g
      ctx.fillRect(px - 14, RULER_H, 28, h - RULER_H)
      ctx.strokeStyle = C.playhead
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(px, 0)
      ctx.lineTo(px, h)
      ctx.stroke()
      ctx.fillStyle = C.playhead
      ctx.beginPath()
      ctx.moveTo(px - 5, 0)
      ctx.lineTo(px + 5, 0)
      ctx.lineTo(px, 7)
      ctx.closePath()
      ctx.fill()
    }
  }, [clip, pxPerFrame, scrollX, currentFrame, activeBone, visibleBones, selectedKeyframes, tab, getDopeFrames])

  useEffect(() => {
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current)
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = null
      draw()
    })
    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current)
        rafRef.current = null
      }
    }
  }, [draw])

  useEffect(() => {
    const el = canvasRef.current
    if (!el) return
    const obs = new ResizeObserver(() => draw())
    obs.observe(el)
    return () => obs.disconnect()
  }, [draw])

  // ── Hit testing ──
  const hitTest = useCallback(
    (e: React.MouseEvent) => {
      const el = canvasRef.current
      if (!el) return null
      const rect = el.getBoundingClientRect()
      const mx = e.clientX - rect.left,
        my = e.clientY - rect.top
      const ox = LABEL_W - scrollX
      const h = el.clientHeight
      const dopeY = h - DOPE_H
      const curveH = dopeY - 1 - RULER_H
      const ax = getAxisConfig(tab)
      const toY = (v: number) => RULER_H + (1 - (v - ax.min) / (ax.max - ax.min)) * curveH
      const toX = (f: number) => ox + f * pxPerFrame

      if (my < RULER_H) {
        const f = Math.round((mx - ox) / pxPerFrame)
        return { zone: "ruler" as const, frame: Math.max(0, Math.min(clip.frameCount, f)) }
      }

      if (my >= dopeY) {
        const frames = getDopeFrames()
        const dopeMid = dopeY + DOPE_H * 0.4
        for (const [frame] of frames) {
          const x = toX(frame)
          if (Math.abs(mx - x) < 8 && Math.abs(my - dopeMid) < 12)
            return { zone: "dope" as const, frame }
        }
        const f = Math.round((mx - ox) / pxPerFrame)
        return { zone: "ruler" as const, frame: Math.max(0, Math.min(clip.frameCount, f)) }
      }

      if (activeBone) {
        const keyframes = clip.boneTracks.get(activeBone)
        if (keyframes) {
          const channels = getChannelsForTab(tab)
          for (const ch of channels) {
            for (const kf of keyframes) {
              const x = toX(kf.frame),
                y = toY(ch.get(kf))
              if (Math.hypot(mx - x, my - y) < DOT_R + 5)
                return { zone: "curve" as const, bone: activeBone, frame: kf.frame, channel: ch.key }
            }
          }
        }
      }

      const f = Math.round((mx - ox) / pxPerFrame)
      return { zone: "ruler" as const, frame: Math.max(0, Math.min(clip.frameCount, f)) }
    },
    [clip, pxPerFrame, scrollX, activeBone, tab, getDopeFrames],
  )

  const onMouseDown = useCallback(
    (e: React.MouseEvent) => {
      const hit = hitTest(e)
      if (!hit) return
      if (hit.zone === "ruler") {
        onSetCurrentFrame(hit.frame)
        drag.current = { type: "scrub" }
      } else if (hit.zone === "dope") {
        onSelectKeyframe({ frame: hit.frame, type: "dope" }, e.shiftKey)
        drag.current = { type: "dope", frame: hit.frame, startX: e.clientX }
      } else if (hit.zone === "curve") {
        onSelectKeyframe(
          { bone: hit.bone, frame: hit.frame, channel: hit.channel, type: "curve" },
          e.shiftKey,
        )
        drag.current = {
          type: "curve",
          bone: hit.bone,
          frame: hit.frame,
          channel: hit.channel,
          startX: e.clientX,
          startY: e.clientY,
        }
      }
    },
    [hitTest, onSetCurrentFrame, onSelectKeyframe],
  )

  const onMouseMove = useCallback(
    (e: React.MouseEvent) => {
      const el = canvasRef.current
      if (!el) return
      const rect = el.getBoundingClientRect()
      const mx = e.clientX - rect.left
      const ox = LABEL_W - scrollX

      if (drag.current?.type === "scrub") {
        const f = Math.round((mx - ox) / pxPerFrame)
        onSetCurrentFrame(Math.max(0, Math.min(clip.frameCount, f)))
        return
      }
      if (drag.current?.type === "dope") {
        const dx = e.clientX - (drag.current.startX ?? 0)
        const df = Math.round(dx / pxPerFrame)
        if (df !== 0 && drag.current.frame !== undefined) {
          onMoveDopeKeyframe(drag.current.frame, drag.current.frame + df)
          drag.current.frame += df
          drag.current.startX = e.clientX
        }
        return
      }
      if (drag.current?.type === "curve") {
        const dx = e.clientX - (drag.current.startX ?? 0)
        const dy = e.clientY - (drag.current.startY ?? 0)
        const df = Math.round(dx / pxPerFrame)
        const h = el.clientHeight
        const curveH = h - DOPE_H - 1 - RULER_H
        const ax = getAxisConfig(tab)
        const dv = -(dy / curveH) * (ax.max - ax.min)
        if ((df !== 0 || Math.abs(dv) > 0.01) && drag.current.bone && drag.current.channel && drag.current.frame !== undefined) {
          const newFrame = df !== 0 ? drag.current.frame + df : drag.current.frame
          onMoveCurveKeyframe(drag.current.bone, drag.current.frame, drag.current.channel, newFrame, dv)
          if (df !== 0) {
            drag.current.frame += df
            drag.current.startX = e.clientX
          }
          drag.current.startY = e.clientY
        }
        return
      }

      const hit = hitTest(e)
      el.style.cursor =
        hit?.zone === "dope"
          ? "ew-resize"
          : hit?.zone === "curve"
            ? "grab"
            : hit?.zone === "ruler"
              ? "col-resize"
              : "default"
    },
    [hitTest, pxPerFrame, scrollX, clip, tab, onSetCurrentFrame, onMoveDopeKeyframe, onMoveCurveKeyframe],
  )

  const onMouseUp = useCallback(() => {
    drag.current = null
  }, [])

  return (
    <canvas
      ref={canvasRef}
      style={{ width: "100%", height: "100%", display: "block" }}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      onMouseLeave={() => {
        drag.current = null
      }}
    />
  )
}

// ─── Timeline (public component) ─────────────────────────────────────────
interface TimelineProps {
  clip: AnimationClip | null
  currentFrame: number
  setCurrentFrame: (f: number | ((p: number) => number)) => void
  playing: boolean
  setPlaying: (p: boolean | ((prev: boolean) => boolean)) => void
  activeBone: string | null
  visibleBones: string[]
  selectedKeyframes: SelectedKeyframe[]
  setSelectedKeyframes: (kfs: SelectedKeyframe[] | ((prev: SelectedKeyframe[]) => SelectedKeyframe[])) => void
}

export function Timeline({
  clip,
  currentFrame,
  setCurrentFrame,
  playing,
  setPlaying,
  activeBone,
  visibleBones,
  selectedKeyframes,
  setSelectedKeyframes,
}: TimelineProps) {
  const fc = clip?.frameCount ?? 0
  const [pxPerFrame, setPxPerFrame] = useState(8)
  const [scrollX, setScrollX] = useState(0)
  const [tab, setTab] = useState("allRot")
  const [, forceRedraw] = useState(0)
  const timelineAreaRef = useRef<HTMLDivElement>(null)
  const [trackWidth, setTrackWidth] = useState(0)

  const minPxPerFrame = useMemo(() => minPxPerFrameForViewport(trackWidth, fc), [trackWidth, fc])

  useEffect(() => {
    const el = timelineAreaRef.current
    if (!el) return
    const ro = new ResizeObserver(() => setTrackWidth(el.clientWidth))
    ro.observe(el)
    setTrackWidth(el.clientWidth)
    return () => ro.disconnect()
  }, [])

  useEffect(() => {
    setPxPerFrame((p) => Math.min(MAX_PX, Math.max(minPxPerFrame, p)))
  }, [minPxPerFrame])

  useEffect(() => {
    if (trackWidth <= 0) return
    const maxScroll = Math.max(0, LABEL_W + fc * pxPerFrame - trackWidth)
    setScrollX((s) => Math.min(maxScroll, Math.max(0, s)))
  }, [trackWidth, fc, pxPerFrame])

  const onWheel = useCallback(
    (e: React.WheelEvent) => {
      e.preventDefault()
      if (e.ctrlKey || e.metaKey)
        setPxPerFrame((p) => Math.max(minPxPerFrame, Math.min(MAX_PX, p - e.deltaY * 0.02)))
      else setScrollX((p) => Math.max(0, p + e.deltaX + e.deltaY))
    },
    [minPxPerFrame],
  )

  const onSelectKeyframe = useCallback(
    (kf: SelectedKeyframe, multi: boolean) => {
      setSelectedKeyframes((prev) => {
        if (multi) {
          const exists = prev.some(
            (s) => s.frame === kf.frame && s.type === kf.type && s.channel === kf.channel && s.bone === kf.bone,
          )
          return exists ? prev.filter((s) => !(s.frame === kf.frame && s.type === kf.type)) : [...prev, kf]
        }
        return [kf]
      })
    },
    [setSelectedKeyframes],
  )

  const onMoveDopeKeyframe = useCallback(
    (from: number, to: number) => {
      if (!clip) return
      const clamped = Math.max(0, Math.min(clip.frameCount, to))
      if (clamped === from) return
      const bones = activeBone ? [activeBone] : visibleBones
      for (const name of bones) {
        const track = clip.boneTracks.get(name)
        if (!track) continue
        const kf = track.find((k) => k.frame === from)
        if (kf) kf.frame = clamped
        track.sort((a, b) => a.frame - b.frame)
      }
      setSelectedKeyframes((prev) =>
        prev.map((s) => (s.frame === from && s.type === "dope" ? { ...s, frame: clamped } : s)),
      )
      forceRedraw((n) => n + 1)
    },
    [clip, activeBone, visibleBones, setSelectedKeyframes],
  )

  const onMoveCurveKeyframe = useCallback(
    (bone: string, from: number, chKey: string, toFrame: number, dv: number) => {
      if (!clip) return
      const track = clip.boneTracks.get(bone)
      if (!track) return
      const clamped = Math.max(0, Math.min(clip.frameCount, toFrame))
      const kf = track.find((k: BoneKeyframe) => k.frame === from)
      if (!kf) return
      if (clamped !== from) {
        kf.frame = clamped
        track.sort((a: BoneKeyframe, b: BoneKeyframe) => a.frame - b.frame)
      }
      if (dv) {
        const ch = ALL_CHANNELS.find((c) => c.key === chKey)
        if (ch) {
          const cur = ch.get(kf)
          ch.set(kf, cur + dv)
        }
      }
      setSelectedKeyframes((prev) =>
        prev.map((s) => (s.bone === bone && s.frame === from && s.channel === chKey ? { ...s, frame: clamped } : s)),
      )
      forceRedraw((n) => n + 1)
    },
    [clip, setSelectedKeyframes],
  )

  return (
    <div className="flex h-full w-full select-none flex-col" style={{ fontFamily: FONT }}>
      {/* Toolbar — compact controls + channel tabs; axis hues stay exact via inline `t.color` when set */}
      <div className="flex h-[26px] shrink-0 flex-nowrap items-center gap-0.5 overflow-hidden border-b border-border bg-background px-1.5">
        {/* Fixed square + Lucide icons — avoids uneven unicode box and mixed h-5 / h-[22px] misalignment */}
        {(
          [
            {
              key: "first",
              el: <ChevronsLeft className="size-3.5" strokeWidth={1.75} />,
              onClick: () => setCurrentFrame(0),
            },
            {
              key: "prev",
              el: <ChevronLeft className="size-3.5" strokeWidth={1.75} />,
              onClick: () => setCurrentFrame((p) => Math.max(0, Math.round(typeof p === "number" ? p : 0) - 1)),
            },
          ] as const
        ).map(({ key, el, onClick }) => (
          <Button
            key={key}
            type="button"
            variant="ghost"
            size="sm"
            className={cn(
              "flex size-5 shrink-0 items-center justify-center overflow-hidden p-0 text-muted-foreground",
              "hover:bg-transparent dark:hover:bg-transparent",
              "active:bg-muted/50",
              "focus-visible:outline-none focus-visible:ring-0",
            )}
            onClick={onClick}
          >
            {el}
          </Button>
        ))}
        <Button
          type="button"
          variant="secondary"
          size="sm"
          className={cn(
            "flex size-5 shrink-0 items-center justify-center overflow-hidden p-0",
            "focus-visible:outline-none focus-visible:ring-0",
            playing && "bg-[#d83838] text-[#0f0f12] hover:bg-[#d83838]/90",
            !playing && "hover:bg-secondary/90",
          )}
          onClick={() => setPlaying((p) => !p)}
        >
          {playing ? (
            <Pause className="size-3.5" strokeWidth={1.5} />
          ) : (
            <Play className="size-3.5 fill-current" strokeWidth={1.75} />
          )}
        </Button>
        {(
          [
            {
              key: "next",
              el: <ChevronRight className="size-3.5" strokeWidth={1.75} />,
              onClick: () => setCurrentFrame((p) => Math.min(fc, Math.round(typeof p === "number" ? p : 0) + 1)),
            },
            {
              key: "last",
              el: <ChevronsRight className="size-3.5" strokeWidth={1.75} />,
              onClick: () => setCurrentFrame(fc),
            },
          ] as const
        ).map(({ key, el, onClick }) => (
          <Button
            key={key}
            type="button"
            variant="ghost"
            size="sm"
            className={cn(
              "flex size-5 shrink-0 items-center justify-center overflow-hidden p-0 text-muted-foreground",
              "hover:bg-transparent dark:hover:bg-transparent",
              "active:bg-muted/50",
              "focus-visible:outline-none focus-visible:ring-0",
            )}
            onClick={onClick}
          >
            {el}
          </Button>
        ))}
        <TransportFrameSlider
          frameCount={fc}
          value={currentFrame}
          onChange={(f) => {
            setPlaying(false)
            setCurrentFrame(f)
          }}
        />
        <div className="mx-0.5 whitespace-nowrap rounded-md border border-border/50 bg-card px-1.5 py-px font-mono text-[9px] tabular-nums text-muted-foreground">
          F{String(Math.round(currentFrame)).padStart(4, "0")} / {fc}
        </div>
        <div className="mx-0.5 h-3.5 w-px shrink-0 bg-border" />
        {/* Channel tabs */}
        {TABS.map((t) => {
          if (t.sep)
            return <div key={t.key} className="mx-px h-3.5 w-px shrink-0 bg-border" />
          const active = tab === t.key
          return (
            <Button
              type="button"
              key={t.key}
              variant="ghost"
              size="sm"
              onClick={() => setTab(t.key)}
              className={cn(
                "h-5 max-h-5 min-h-5 shrink-0 overflow-hidden rounded-md px-1.5 font-mono text-[10px]",
                "focus-visible:outline-none focus-visible:ring-0",
                active
                  ? t.color
                    ? "text-[#0f0f12] hover:opacity-90"
                    : "bg-secondary text-foreground hover:bg-secondary/80"
                  : "opacity-65 hover:opacity-100 hover:bg-transparent dark:hover:bg-transparent active:bg-muted/50",
                !active && !t.color && "text-muted-foreground",
              )}
              style={
                active && t.color
                  ? { backgroundColor: t.color }
                  : !active && t.color
                    ? { color: t.color }
                    : undefined
              }
            >
              {t.label}
            </Button>
          )
        })}
        <div className="min-w-0 flex-1" />
        <ZoomRuler min={minPxPerFrame} max={MAX_PX} value={pxPerFrame} onChange={setPxPerFrame} />
      </div>
      {/* Canvas */}
      <div ref={timelineAreaRef} style={{ flex: 1, minHeight: 0 }} onWheel={onWheel}>
        {clip ? (
          <TimelineCanvas
            clip={clip}
            pxPerFrame={pxPerFrame}
            scrollX={scrollX}
            currentFrame={currentFrame}
            activeBone={activeBone}
            visibleBones={visibleBones}
            selectedKeyframes={selectedKeyframes}
            tab={tab}
            onSetCurrentFrame={(f) => {
              setPlaying(false)
              setCurrentFrame(f)
            }}
            onSelectKeyframe={onSelectKeyframe}
            onMoveDopeKeyframe={onMoveDopeKeyframe}
            onMoveCurveKeyframe={onMoveCurveKeyframe}
          />
        ) : (
          <div
            style={{
              height: "100%",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: C.label,
              fontSize: 11,
              background: C.curveBg,
            }}
          >
            Load VMD for timeline…
          </div>
        )}
      </div>
    </div>
  )
}
