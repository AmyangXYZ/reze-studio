"use client"

import {
  useState,
  useRef,
  useEffect,
  useLayoutEffect,
  useCallback,
  useMemo,
  type RefObject,
} from "react"
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
import { useStudioActions, useStudioSelector, type SelectedKeyframe } from "@/context/studio-context"
import { usePlayback } from "@/context/playback-context"
import type { AnimationClip, BoneKeyframe, MorphKeyframe } from "reze-engine"
import {
  type Channel,
  ROT_CHANNELS,
  TRA_CHANNELS,
  boneDisplayLabel,
} from "@/lib/animation"

// ─── Timeline constants ─────────────────────────────────────────────────
const DOPE_H = 34
const RULER_H = 17
const LABEL_W = 36
const DOT_R = 3.5
const DIAMOND = 5
const MIN_PX = 0.5
const MAX_PX = 40
const Y_ZOOM_MIN = 0.5
const Y_ZOOM_MAX = 8

function minPxPerFrameForViewport(trackWidthPx: number, frameCount: number): number {
  if (frameCount <= 0 || trackWidthPx <= LABEL_W + 1) return MIN_PX
  const fit = (trackWidthPx - LABEL_W) / frameCount
  return Math.max(MIN_PX, Math.min(fit, MAX_PX))
}

function bezierY(cp0: { x: number; y: number }, cp1: { x: number; y: number }, t: number) {
  const x1 = cp0.x / 127,
    y1 = cp0.y / 127,
    x2 = cp1.x / 127,
    y2 = cp1.y / 127
  let lo = 0,
    hi = 1,
    mid = 0.5
  for (let i = 0; i < 15; i++) {
    const x = 3 * (1 - mid) ** 2 * mid * x1 + 3 * (1 - mid) * mid ** 2 * x2 + mid ** 3
    if (Math.abs(x - t) < 0.0001) break
    if (x < t) lo = mid
    else hi = mid
    mid = (lo + hi) / 2
  }
  return 3 * (1 - mid) ** 2 * mid * y1 + 3 * (1 - mid) * mid ** 2 * y2 + mid ** 3
}

const C = {
  bg: "rgba(0,0,0,0)",
  curveBg: "rgba(0,0,0,0)",
  ruler: "rgba(0,0,0,0)",
  rulerText: "#9ca3af",
  rulerTick: "#2a2a34",
  rulerMajor: "#3a3a48",
  grid: "#161620",
  axis: "#222233",
  axisZero: "#2c2c44",
  playhead: "#d83838",
  playheadGlow: "rgba(216,56,56,0.18)",
  diamondSel: "#5aa0f0",
  keyDotSel: "#9ca3af",
  dopeBg: "rgba(0,0,0,0)",
  dopeBorder: "#222230",
  dopeLabel: "#9ca3af",
  dopeLabelNum: "#6b7280",
  rotX: "#e25555",
  rotY: "#44bb55",
  rotZ: "#4477dd",
  traX: "#e2a055",
  traY: "#55bba0",
  traZ: "#7755dd",
  label: "#9ca3af",
  tabBg: "rgba(0,0,0,0)",
  tabActive: "#2a2a36",
  tabText: "#9ca3af",
  tabTextActive: "#9ca3af",
  toolbarOnAccent: "#0f0f12",
  border: "border",
  frameBadge: "#1a1a22",
  frameBadgeText: "#9ca3af",
  sidebarBg: "rgba(0,0,0,0)",
  sidebarGroup: "#888898",
  sidebarBone: "#666672",
  sidebarActive: "#5aa0f0",
  sidebarGroupBg: "rgba(0,0,0,0)",
  sidebarHover: "#1e1e28",
} as const

const FONT = "'SF Mono','Cascadia Code','Fira Code','JetBrains Mono',monospace"

function padFrame4(n: number) {
  return String(Math.max(0, Math.round(n))).padStart(4, "0")
}

const ALL_CHANNELS: Channel[] = [...ROT_CHANNELS, ...TRA_CHANNELS]

function getChannelsForTab(tab: string): Channel[] {
  if (tab === "morph") return []
  if (tab === "allRot") return ROT_CHANNELS
  if (tab === "allTra") return TRA_CHANNELS
  const ch = ALL_CHANNELS.find((c) => c.key === tab)
  return ch ? [ch] : ROT_CHANNELS
}

function getAxisConfig(tab: string) {
  if (tab === "morph") {
    return { min: 0, max: 1, unit: "", side: "left" as const, step: 0.25, subStep: 0.125 }
  }
  const chans = getChannelsForTab(tab)
  const isRot = chans[0].group === "rot"
  if (isRot) {
    return { min: -90, max: 90, unit: "°", side: "left" as const, step: 30, subStep: 15 }
  } else {
    return { min: -10, max: 10, unit: "", side: "left" as const, step: 5, subStep: 2.5 }
  }
}

const MORPH_COLOR = "#c084fc"

const TABS = [
  { key: "allRot", label: "All Rot", color: null, sep: false },
  { key: "rx", label: "X", color: C.rotX, sep: false },
  { key: "ry", label: "Y", color: C.rotY, sep: false },
  { key: "rz", label: "Z", color: C.rotZ, sep: false },
  { key: "_sep1", label: "", color: null, sep: true },
  { key: "allTra", label: "All Trans", color: null, sep: false },
  { key: "tx", label: "X", color: C.traX, sep: false },
  { key: "ty", label: "Y", color: C.traY, sep: false },
  { key: "tz", label: "Z", color: C.traZ, sep: false },
  { key: "_sep2", label: "", color: null, sep: true },
  { key: "morph", label: "Morph", color: MORPH_COLOR, sep: false },
]

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
          "relative h-5 w-14 shrink-0 touch-none",
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

export type { SelectedKeyframe } from "@/context/studio-context"

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
  yZoom: number
  scrollX: number
  currentFrame: number
  selectedBone: string | null
  selectedMorph: string | null
  visibleBones: string[]
  selectedKeyframes: SelectedKeyframe[]
  tab: string
  onSetCurrentFrame: (f: number) => void
  onSelectKeyframe: (kf: SelectedKeyframe, multi: boolean) => void
  onMoveDopeKeyframe: (
    boneRefs: Array<{ bone: string; kf: BoneKeyframe }>,
    morphRefs: Array<{ morph: string; kf: MorphKeyframe }>,
    toFrame: number,
  ) => void
  onMoveCurveKeyframe: (bone: string, kfRef: BoneKeyframe, channel: string, toFrame: number, dv: number) => void
  onMoveMorphKeyframe: (morph: string, kfRef: MorphKeyframe, toFrame: number, dw: number) => void
  /** Fired on mouseup/mouseleave at the end of a keyframe drag. Parent uses
   *  this to commit the clip once (for undo/redo + engine upload) instead of
   *  per-mousemove. */
  onEndKeyframeDrag: () => void
  /** Imperative repaint hook — bumps the static-cache drag version and redraws
   *  the canvas without a React state update. Populated by TimelineCanvas. */
  dragRedrawRef?: RefObject<(() => void) | null>
  /** Imperative draw handle: playback rAF loop writes the latest frame directly
   *  via `ref.current(frame)`, bypassing React reconciliation for 60Hz playhead
   *  updates. Populated by TimelineCanvas on mount. */
  playheadDrawRef?: RefObject<((frame: number) => void) | null>
}

function TimelineCanvas({
  clip,
  pxPerFrame,
  yZoom,
  scrollX,
  currentFrame,
  selectedBone,
  selectedMorph,
  visibleBones,
  selectedKeyframes,
  tab,
  onSetCurrentFrame,
  onSelectKeyframe,
  onMoveDopeKeyframe,
  onMoveCurveKeyframe,
  onMoveMorphKeyframe,
  onEndKeyframeDrag,
  playheadDrawRef,
  dragRedrawRef,
}: TimelineCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  /** Latest frame for the draw closure to read — keeps `currentFrame` out of
   *  the `draw` useCallback deps so playback ticks don't re-run layout effects. */
  const frameRef = useRef(currentFrame)
  frameRef.current = currentFrame
  const sizeRef = useRef({ w: 0, h: 0, dpr: 0 })
  /** Offscreen cache: ruler + grid + curves + dopesheet. Repainted only when
   *  non-currentFrame deps change, so playback ticks just blit + draw the playhead. */
  const staticCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const staticCacheRef = useRef<{
    w: number
    h: number
    clip: AnimationClip | null
    pxPerFrame: number
    yZoom: number
    scrollX: number
    selectedBone: string | null
    selectedMorph: string | null
    visibleBones: readonly string[] | null
    selectedKeyframes: readonly SelectedKeyframe[] | null
    tab: string
    dragVersion: number
  }>({
    w: 0,
    h: 0,
    clip: null,
    pxPerFrame: 0,
    yZoom: 0,
    scrollX: 0,
    selectedBone: null,
    selectedMorph: null,
    visibleBones: null,
    selectedKeyframes: null,
    tab: "",
    dragVersion: 0,
  })
  /** Bumped by parent via `dragRedrawRef` while a keyframe drag is in progress.
   *  Participates in static-cache invalidation so in-place mutations repaint
   *  without needing a new clip reference (which would cascade a commit +
   *  engine clip reupload per mousemove). */
  const dragVersionRef = useRef(0)
  const drag = useRef<{
    type: string
    bone?: string
    channel?: string
    startX?: number
    startY?: number
    boneKfRef?: BoneKeyframe
    morphKfRef?: MorphKeyframe
    dopeBoneRefs?: Array<{ bone: string; kf: BoneKeyframe }>
    dopeMorphRefs?: Array<{ morph: string; kf: MorphKeyframe }>
    dopeFrame?: number
  } | null>(null)

  const getDopeFrames = useCallback(() => {
    const frames = new Map<number, number>()
    if (tab === "morph" && selectedMorph) {
      const track = clip.morphTracks.get(selectedMorph)
      if (track) for (const kf of track) frames.set(kf.frame, 1)
    } else if (selectedBone) {
      const track = clip.boneTracks.get(selectedBone)
      if (track) for (const kf of track) frames.set(kf.frame, 1)
    } else {
      for (const name of visibleBones) {
        const track = clip.boneTracks.get(name)
        if (track) for (const kf of track) frames.set(kf.frame, (frames.get(kf.frame) || 0) + 1)
      }
    }
    return frames
  }, [clip, visibleBones, selectedBone, selectedMorph, tab])

  const draw = useCallback(() => {
    const el = canvasRef.current
    if (!el) return
    const mainCtx = el.getContext("2d")
    if (!mainCtx) return
    const dpr = Math.min(4, Math.max(1, (window.devicePixelRatio || 1) * 1.5))
    const w = el.clientWidth,
      h = el.clientHeight
    const backingW = Math.max(1, Math.floor(w * dpr))
    const backingH = Math.max(1, Math.floor(h * dpr))
    const size = sizeRef.current
    let mainResized = false
    if (size.w !== backingW || size.h !== backingH || size.dpr !== dpr) {
      el.width = backingW
      el.height = backingH
      sizeRef.current = { w: backingW, h: backingH, dpr }
      mainResized = true
    }

    // ── Static cache: paint heavy layer to offscreen only when non-currentFrame deps change ──
    let off = staticCanvasRef.current
    if (!off) {
      off = document.createElement("canvas")
      staticCanvasRef.current = off
    }
    const cache = staticCacheRef.current
    const needRepaintStatic =
      mainResized ||
      cache.w !== backingW ||
      cache.h !== backingH ||
      cache.clip !== clip ||
      cache.pxPerFrame !== pxPerFrame ||
      cache.yZoom !== yZoom ||
      cache.scrollX !== scrollX ||
      cache.selectedBone !== selectedBone ||
      cache.selectedMorph !== selectedMorph ||
      cache.visibleBones !== visibleBones ||
      cache.selectedKeyframes !== selectedKeyframes ||
      cache.tab !== tab ||
      cache.dragVersion !== dragVersionRef.current

    const ox = LABEL_W - scrollX
    const dopeY = h - DOPE_H
    const curveTop = RULER_H
    const curveBot = dopeY - 1
    const curveH = curveBot - curveTop

    const channels = getChannelsForTab(tab)
    const ax = getAxisConfig(tab)
    // Y-zoom: shrink the visible value range around the axis center.
    const axCenter = (ax.min + ax.max) / 2
    const axHalf = (ax.max - ax.min) / 2 / Math.max(0.0001, yZoom)
    const vMin = axCenter - axHalf
    const vMax = axCenter + axHalf
    const toY = (v: number) => curveTop + (1 - (v - vMin) / (vMax - vMin)) * curveH
    const toX = (f: number) => ox + f * pxPerFrame

    if (needRepaintStatic) {
    if (off.width !== backingW || off.height !== backingH) {
      off.width = backingW
      off.height = backingH
    }
    const ctx = off.getContext("2d")
    if (!ctx) return
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, w, h)

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
    const isRotAxis = channels[0]?.group === "rot"
    // Snap tick iteration to multiples of subStep within the current view range.
    const firstTick = Math.ceil(vMin / ax.subStep) * ax.subStep
    const lastTick = Math.floor(vMax / ax.subStep) * ax.subStep
    const vSteps = Math.max(0, Math.round((lastTick - firstTick) / ax.subStep))
    for (let i = 0; i <= vSteps; i++) {
      const v = firstTick + i * ax.subStep
      if (v < vMin - 0.0001 || v > vMax + 0.0001) continue
      const y = toY(v)
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

    // ── Curves ── (clip to plot area so zoomed-out-of-view values don't bleed)
    ctx.save()
    ctx.beginPath()
    ctx.rect(LABEL_W, curveTop, w - LABEL_W, curveBot - curveTop)
    ctx.clip()
    const isMorphTab = tab === "morph"
    if (isMorphTab) {
      // ── Morph weight curve ──
      if (selectedMorph) {
        const morphKfs = clip.morphTracks.get(selectedMorph)
        if (morphKfs && morphKfs.length > 0) {
          // Draw linear curve
          ctx.strokeStyle = MORPH_COLOR
          ctx.lineWidth = 2
          ctx.beginPath()
          for (let i = 0; i < morphKfs.length; i++) {
            const kf = morphKfs[i]
            const x = toX(kf.frame), y = toY(kf.weight)
            if (i === 0) ctx.moveTo(x, y)
            else ctx.lineTo(x, y)
          }
          ctx.stroke()

          // Dots
          for (const kf of morphKfs) {
            const x = toX(kf.frame)
            if (x < LABEL_W - 8 || x > w + 8) continue
            const isSel = selectedKeyframes.some(
              (s) => s.morph === selectedMorph && s.frame === kf.frame,
            )
            ctx.beginPath()
            ctx.arc(x, toY(kf.weight), isSel ? DOT_R + 1.5 : DOT_R, 0, Math.PI * 2)
            ctx.fillStyle = isSel ? C.keyDotSel : MORPH_COLOR
            ctx.fill()
            if (isSel) {
              ctx.strokeStyle = MORPH_COLOR
              ctx.lineWidth = 2
              ctx.stroke()
            }
          }

          // (value readout is drawn in the per-tick overlay below)
        } else {
          ctx.fillStyle = C.label
          ctx.font = `13px ${FONT}`
          ctx.textAlign = "center"
          ctx.textBaseline = "middle"
          ctx.fillText("No keyframes — " + selectedMorph, (w + LABEL_W) / 2, (curveTop + curveBot) / 2)
        }
      } else {
        ctx.fillStyle = C.label
        ctx.font = `13px ${FONT}`
        ctx.textAlign = "center"
        ctx.textBaseline = "middle"
        ctx.fillText("Select a morph to view curve", (w + LABEL_W) / 2, (curveTop + curveBot) / 2)
      }
    } else if (selectedBone) {
      const keyframes = clip.boneTracks.get(selectedBone)
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
              (s) => s.bone === selectedBone && s.frame === kf.frame && s.channel === ch.key,
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

        // (value readout is drawn in the per-tick overlay below)
      } else {
        ctx.fillStyle = C.label
        ctx.font = `13px ${FONT}`
        ctx.textAlign = "center"
        ctx.textBaseline = "middle"
        ctx.fillText("No keyframes — " + boneDisplayLabel(selectedBone), (w + LABEL_W) / 2, (curveTop + curveBot) / 2)
      }
    } else {
      ctx.fillStyle = C.label
      ctx.font = `13px ${FONT}`
      ctx.textAlign = "center"
      ctx.textBaseline = "middle"
      ctx.fillText("Select a bone to view curves", (w + LABEL_W) / 2, (curveTop + curveBot) / 2)
    }
    ctx.restore()

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
      const intensity = selectedBone ? 0.85 : 0.4 + 0.6 * (count / maxCount)

      ctx.save()
      ctx.translate(x, dopeMid)
      ctx.rotate(Math.PI / 4)
      const sz = DIAMOND + (count > 1 && !selectedBone ? 1.5 : 0)
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

      cache.w = backingW
      cache.h = backingH
      cache.clip = clip
      cache.pxPerFrame = pxPerFrame
      cache.yZoom = yZoom
      cache.scrollX = scrollX
      cache.selectedBone = selectedBone
      cache.selectedMorph = selectedMorph
      cache.visibleBones = visibleBones
      cache.selectedKeyframes = selectedKeyframes
      cache.tab = tab
      cache.dragVersion = dragVersionRef.current
    }

    // ── Composite cached static layer onto the visible canvas ──
    mainCtx.setTransform(1, 0, 0, 1, 0, 0)
    mainCtx.clearRect(0, 0, backingW, backingH)
    mainCtx.drawImage(off, 0, 0)
    mainCtx.setTransform(dpr, 0, 0, dpr, 0, 0)

    // ── Per-tick overlay: value readout (reads frameRef so playback rAF can
    //    call `draw()` without invalidating the useCallback closure) ──
    const frame = frameRef.current
    {
      const ctx = mainCtx
      if (tab === "morph" && selectedMorph) {
        const morphKfs = clip.morphTracks.get(selectedMorph)
        if (morphKfs && morphKfs.length > 0) {
          ctx.font = `10px ${FONT}`
          ctx.textBaseline = "top"
          ctx.textAlign = "right"
          let val = 0
          for (const k of morphKfs) {
            if (k.frame <= frame) val = k.weight
          }
          ctx.fillStyle = MORPH_COLOR
          ctx.fillText(`Weight: ${val.toFixed(2)}`, w - 8, curveTop + 5)
        }
      } else if (selectedBone) {
        const keyframes = clip.boneTracks.get(selectedBone)
        if (keyframes && keyframes.length > 0) {
          ctx.font = `10px ${FONT}`
          ctx.textBaseline = "top"
          ctx.textAlign = "right"
          const isRotGroup = channels[0].group === "rot"
          const numWidth = isRotGroup ? 7 : 6
          const formatVal = (v: number) => {
            const s = isRotGroup ? `${v.toFixed(1)}°` : v.toFixed(2)
            return s.padStart(numWidth, " ")
          }
          channels.forEach((ch, i) => {
            let val = 0
            for (const k of keyframes) {
              if (k.frame <= frame) val = ch.get(k)
            }
            ctx.fillStyle = ch.color
            ctx.fillText(`${ch.label}: ${formatVal(val)}`, w - 8, curveTop + 5 + i * 13)
          })
        }
      }

      // ── Playhead ──
      const px = toX(frame)
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
    }
  }, [clip, pxPerFrame, yZoom, scrollX, selectedBone, selectedMorph, visibleBones, selectedKeyframes, tab, getDopeFrames])

  // Layout-phase paint: `useEffect`+nested rAF ran after browser paint → playhead lagged 1–2 frames behind transport.
  // `currentFrame` is in deps (not in `draw`'s deps) so scrubbing + throttled
  // playback state updates still repaint; 60Hz playback bypasses this via
  // `playheadDrawRef` below.
  useLayoutEffect(() => {
    draw()
  }, [draw, currentFrame])

  // Publish the imperative draw handle so the playback rAF loop can update the
  // playhead without going through React state.
  useEffect(() => {
    if (!playheadDrawRef) return
    playheadDrawRef.current = (frame: number) => {
      frameRef.current = frame
      draw()
    }
    return () => {
      if (playheadDrawRef.current) playheadDrawRef.current = null
    }
  }, [draw, playheadDrawRef])

  // Publish the drag-redraw handle. Parent drag callbacks mutate keyframes
  // in place then invoke this to invalidate the static cache and repaint —
  // no React state updates, no clip commits, no engine re-uploads.
  useEffect(() => {
    if (!dragRedrawRef) return
    dragRedrawRef.current = () => {
      dragVersionRef.current++
      draw()
    }
    return () => {
      if (dragRedrawRef.current) dragRedrawRef.current = null
    }
  }, [draw, dragRedrawRef])

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
      const axCenter = (ax.min + ax.max) / 2
      const axHalf = (ax.max - ax.min) / 2 / Math.max(0.0001, yZoom)
      const vMin = axCenter - axHalf
      const vMax = axCenter + axHalf
      const toY = (v: number) => RULER_H + (1 - (v - vMin) / (vMax - vMin)) * curveH
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

      if (tab === "morph" && selectedMorph) {
        const morphKfs = clip.morphTracks.get(selectedMorph)
        if (morphKfs) {
          for (const kf of morphKfs) {
            const x = toX(kf.frame), y = toY(kf.weight)
            if (Math.hypot(mx - x, my - y) < DOT_R + 5)
              return { zone: "morph-curve" as const, morph: selectedMorph, frame: kf.frame }
          }
        }
      } else if (selectedBone) {
        const keyframes = clip.boneTracks.get(selectedBone)
        if (keyframes) {
          const channels = getChannelsForTab(tab)
          for (const ch of channels) {
            for (const kf of keyframes) {
              const x = toX(kf.frame),
                y = toY(ch.get(kf))
              if (Math.hypot(mx - x, my - y) < DOT_R + 5)
                return { zone: "curve" as const, bone: selectedBone, frame: kf.frame, channel: ch.key }
            }
          }
        }
      }

      const f = Math.round((mx - ox) / pxPerFrame)
      return { zone: "ruler" as const, frame: Math.max(0, Math.min(clip.frameCount, f)) }
    },
    [clip, pxPerFrame, yZoom, scrollX, selectedBone, selectedMorph, tab, getDopeFrames],
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
        // Capture references to all keyframes (across bones/morphs) sharing this frame
        const dopeBoneRefs: Array<{ bone: string; kf: BoneKeyframe }> = []
        const dopeMorphRefs: Array<{ morph: string; kf: MorphKeyframe }> = []
        if (tab === "morph" && selectedMorph) {
          const track = clip.morphTracks.get(selectedMorph)
          const kf = track?.find((k) => k.frame === hit.frame)
          if (kf) dopeMorphRefs.push({ morph: selectedMorph, kf })
        } else {
          const bones = selectedBone ? [selectedBone] : visibleBones
          for (const name of bones) {
            const track = clip.boneTracks.get(name)
            const kf = track?.find((k) => k.frame === hit.frame)
            if (kf) dopeBoneRefs.push({ bone: name, kf })
          }
        }
        drag.current = {
          type: "dope",
          startX: e.clientX,
          dopeBoneRefs,
          dopeMorphRefs,
          dopeFrame: hit.frame,
        }
      } else if (hit.zone === "morph-curve") {
        const track = clip.morphTracks.get(hit.morph)
        const kfRef = track?.find((k) => k.frame === hit.frame)
        if (!kfRef) return
        onSelectKeyframe(
          { morph: hit.morph, frame: hit.frame, type: "curve" },
          e.shiftKey,
        )
        drag.current = {
          type: "morph-curve",
          bone: hit.morph,
          morphKfRef: kfRef,
          startX: e.clientX,
          startY: e.clientY,
        }
      } else if (hit.zone === "curve") {
        const track = clip.boneTracks.get(hit.bone)
        const kfRef = track?.find((k) => k.frame === hit.frame)
        if (!kfRef) return
        onSelectKeyframe(
          { bone: hit.bone, frame: hit.frame, channel: hit.channel, type: "curve" },
          e.shiftKey,
        )
        drag.current = {
          type: "curve",
          bone: hit.bone,
          boneKfRef: kfRef,
          channel: hit.channel,
          startX: e.clientX,
          startY: e.clientY,
        }
      }
    },
    [hitTest, onSetCurrentFrame, onSelectKeyframe, clip, tab, selectedBone, selectedMorph, visibleBones],
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
        if (df !== 0 && drag.current.dopeFrame !== undefined) {
          const newFrame = drag.current.dopeFrame + df
          onMoveDopeKeyframe(drag.current.dopeBoneRefs ?? [], drag.current.dopeMorphRefs ?? [], newFrame)
          drag.current.dopeFrame = Math.max(0, newFrame)
          drag.current.startX = e.clientX
        }
        return
      }
      if (drag.current?.type === "morph-curve") {
        const dx = e.clientX - (drag.current.startX ?? 0)
        const dy = e.clientY - (drag.current.startY ?? 0)
        const df = Math.round(dx / pxPerFrame)
        const h = el.clientHeight
        const curveH = h - DOPE_H - 1 - RULER_H
        const ax = getAxisConfig(tab)
        const dw = -(dy / curveH) * ((ax.max - ax.min) / Math.max(0.0001, yZoom))
        const ref = drag.current.morphKfRef
        if ((df !== 0 || Math.abs(dw) > 0.005) && drag.current.bone && ref) {
          const newFrame = ref.frame + df
          onMoveMorphKeyframe(drag.current.bone, ref, newFrame, dw)
          if (df !== 0) drag.current.startX = e.clientX
          drag.current.startY = e.clientY
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
        const dv = -(dy / curveH) * ((ax.max - ax.min) / Math.max(0.0001, yZoom))
        const ref = drag.current.boneKfRef
        if ((df !== 0 || Math.abs(dv) > 0.01) && drag.current.bone && drag.current.channel && ref) {
          const newFrame = ref.frame + df
          onMoveCurveKeyframe(drag.current.bone, ref, drag.current.channel, newFrame, dv)
          if (df !== 0) drag.current.startX = e.clientX
          drag.current.startY = e.clientY
        }
        return
      }

      const hit = hitTest(e)
      el.style.cursor =
        hit?.zone === "dope"
          ? "ew-resize"
          : hit?.zone === "curve" || hit?.zone === "morph-curve"
            ? "grab"
            : hit?.zone === "ruler"
              ? "col-resize"
              : "default"
    },
    [hitTest, pxPerFrame, yZoom, scrollX, clip, tab, onSetCurrentFrame, onMoveDopeKeyframe, onMoveCurveKeyframe, onMoveMorphKeyframe],
  )

  const endDrag = useCallback(() => {
    const d = drag.current
    drag.current = null
    if (d && (d.type === "dope" || d.type === "curve" || d.type === "morph-curve")) {
      onEndKeyframeDrag()
    }
  }, [onEndKeyframeDrag])

  return (
    <canvas
      ref={canvasRef}
      style={{ width: "100%", height: "100%", display: "block" }}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={endDrag}
      onMouseLeave={endDrag}
    />
  )
}

// ─── Timeline (public component) ─────────────────────────────────────────
interface TimelineProps {
  visibleBones: string[]
  /** Bumped on new clip load / reset — triggers local view state reset. */
  clipVersion: number
  /** Lifted channel tab state — synced from keyframe selection and slider interactions. */
  tab: string
  setTab: (tab: string) => void
  /** Imperative playhead draw handle — see TimelineCanvasProps. */
  playheadDrawRef?: RefObject<((frame: number) => void) | null>
}

export function Timeline({
  visibleBones,
  clipVersion,
  tab,
  setTab,
  playheadDrawRef,
}: TimelineProps) {
  const clip = useStudioSelector((s) => s.clip)
  const selectedBone = useStudioSelector((s) => s.selectedBone)
  const selectedMorph = useStudioSelector((s) => s.selectedMorph)
  const selectedKeyframes = useStudioSelector((s) => s.selectedKeyframes)
  const { commit, setSelectedKeyframes } = useStudioActions()
  const { currentFrame, setCurrentFrame, playing, setPlaying } = usePlayback()
  const fc = clip?.frameCount ?? 0
  const [endDraft, setEndDraft] = useState<string | null>(null)
  const [frameDraft, setFrameDraft] = useState<string | null>(null)
  const [pxPerFrame, setPxPerFrame] = useState(4)
  const pxRef = useRef(pxPerFrame)
  pxRef.current = pxPerFrame
  const [yZoom, setYZoom] = useState(1)
  const yZoomRef = useRef(yZoom)
  yZoomRef.current = yZoom
  const [scrollX, setScrollX] = useState(0)
  const scrollXRef = useRef(0)
  scrollXRef.current = scrollX
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

  // Reset local view state when a new clip is loaded or editor is reset.
  const clipVersionRef = useRef(clipVersion)
  useEffect(() => {
    if (clipVersionRef.current === clipVersion) return
    clipVersionRef.current = clipVersion
    setScrollX(0)
    setPxPerFrame(4)
    setYZoom(1)
    setEndDraft(null)
  }, [clipVersion])

  // Clamp scroll when viewport or clip size changes (NOT on pxPerFrame — zoom handles its own scroll)
  useEffect(() => {
    if (trackWidth <= 0) return
    const maxScroll = Math.max(0, LABEL_W + fc * pxRef.current - trackWidth)
    setScrollX((s) => Math.min(maxScroll, Math.max(0, s)))
  }, [trackWidth, fc])

  // ── Auto-scroll: page-turn when playhead leaves the visible window ──
  // During playback the rAF loop drives the playhead imperatively and does NOT
  // update `currentFrame` state — so page-turn is handled in the wrapped
  // `playheadDrawRef` below. This effect only covers scrub / paused navigation.
  useEffect(() => {
    if (trackWidth <= 0) return
    const viewable = trackWidth - LABEL_W
    if (viewable <= 0) return
    const px = pxRef.current
    const playheadX = currentFrame * px
    const maxScroll = Math.max(0, LABEL_W + fc * px - trackWidth)
    const visLeft = scrollXRef.current
    const visRight = scrollXRef.current + viewable

    if (playheadX >= visLeft && playheadX <= visRight) return

    const target = Math.max(0, Math.min(maxScroll, playheadX - viewable * 0.1))
    setScrollX(target)
  }, [currentFrame, trackWidth, fc])

  // Wrap the parent's imperative draw handle so each 60Hz tick can page-turn
  // the timeline if the playhead leaves the visible window. `setScrollX` fires
  // only on the page-turn frame (a few times per clip), not per-frame.
  const innerDrawRef = useRef<((frame: number) => void) | null>(null)
  const fcRef = useRef(fc)
  fcRef.current = fc
  const trackWidthRef = useRef(trackWidth)
  trackWidthRef.current = trackWidth
  useEffect(() => {
    if (!playheadDrawRef) return
    playheadDrawRef.current = (frame: number) => {
      const tw = trackWidthRef.current
      const viewable = tw - LABEL_W
      if (viewable > 0) {
        const px = pxRef.current
        const playheadX = frame * px
        const visLeft = scrollXRef.current
        const visRight = visLeft + viewable
        if (playheadX < visLeft || playheadX > visRight) {
          const maxScroll = Math.max(0, LABEL_W + fcRef.current * px - tw)
          const target = Math.max(0, Math.min(maxScroll, playheadX - viewable * 0.1))
          setScrollX(target)
        }
      }
      innerDrawRef.current?.(frame)
    }
    return () => {
      if (playheadDrawRef.current) playheadDrawRef.current = null
    }
  }, [playheadDrawRef])

  // Zoom anchored on the playhead: adjust scrollX so the playhead stays at the
  // same screen-relative position before and after the pxPerFrame change.
  const zoomTo = useCallback(
    (newPx: number) => {
      const clamped = Math.max(minPxPerFrame, Math.min(MAX_PX, newPx))
      const oldPx = pxRef.current
      if (clamped === oldPx) return
      const viewable = trackWidth - LABEL_W
      if (viewable > 0) {
        const playheadScreen = currentFrame * oldPx - scrollXRef.current
        const newScroll = currentFrame * clamped - playheadScreen
        const maxScroll = Math.max(0, LABEL_W + fc * clamped - trackWidth)
        setScrollX(Math.max(0, Math.min(maxScroll, newScroll)))
      }
      setPxPerFrame(clamped)
    },
    [minPxPerFrame, trackWidth, currentFrame, fc],
  )

  // Native non-passive wheel listener — React's synthetic onWheel is passive,
  // so `preventDefault()` there is ignored and the page scrolls instead of the
  // timeline handling the gesture. Attach directly to the DOM node.
  //  - ctrl/⌘ + wheel → time zoom
  //  - shift + wheel  → value zoom
  //  - plain wheel    → horizontal scroll
  useEffect(() => {
    const el = timelineAreaRef.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      if (e.ctrlKey || e.metaKey) {
        zoomTo(pxRef.current - e.deltaY * 0.02)
      } else if (e.shiftKey) {
        // macOS remaps shift+wheel vertical delta onto deltaX — take whichever is non-zero.
        const delta = e.deltaY !== 0 ? e.deltaY : e.deltaX
        const factor = Math.exp(-delta * 0.002)
        setYZoom((z) => Math.max(Y_ZOOM_MIN, Math.min(Y_ZOOM_MAX, z * factor)))
      } else {
        setScrollX((p) => Math.max(0, p + e.deltaX + e.deltaY))
      }
    }
    el.addEventListener("wheel", onWheel, { passive: false })
    return () => el.removeEventListener("wheel", onWheel)
  }, [zoomTo])

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

  // ─── Imperative drag path ───────────────────────────────────────────
  //     Drag move callbacks mutate keyframes in place and trigger a canvas
  //     repaint via `dragRedrawRef` — no React state updates, no clip
  //     commits, no engine clip reuploads. The single commit happens in
  //     `onEndKeyframeDrag` on mouseup, which cascades once:
  //       studio store → EngineBridge `loadClip` + seek → Timeline repaint.
  //     `selectedKeyframes` entries are mutated in place so the highlighted
  //     diamonds track the drag; on end, we clone the array to notify the
  //     store for downstream subscribers (inspector, etc.).
  const dragRedrawRef = useRef<(() => void) | null>(null)
  const dragTouchedRef = useRef(false)

  const onMoveDopeKeyframe = useCallback(
    (
      boneRefs: Array<{ bone: string; kf: BoneKeyframe }>,
      morphRefs: Array<{ morph: string; kf: MorphKeyframe }>,
      toFrame: number,
    ) => {
      if (!clip) return
      const clamped = Math.max(0, toFrame)
      const fromFrame = boneRefs[0]?.kf.frame ?? morphRefs[0]?.kf.frame
      if (fromFrame === undefined || clamped === fromFrame) return
      for (const { bone, kf } of boneRefs) {
        kf.frame = clamped
        clip.boneTracks.get(bone)?.sort((a, b) => a.frame - b.frame)
      }
      for (const { morph, kf } of morphRefs) {
        kf.frame = clamped
        clip.morphTracks.get(morph)?.sort((a, b) => a.frame - b.frame)
      }
      for (const s of selectedKeyframes) {
        if (s.type === "dope" && s.frame === fromFrame) (s as { frame: number }).frame = clamped
      }
      dragTouchedRef.current = true
      dragRedrawRef.current?.()
    },
    [clip, selectedKeyframes],
  )

  const onMoveCurveKeyframe = useCallback(
    (bone: string, kfRef: BoneKeyframe, chKey: string, toFrame: number, dv: number) => {
      if (!clip) return
      const track = clip.boneTracks.get(bone)
      if (!track || !track.includes(kfRef)) return
      const clamped = Math.max(0, toFrame)
      const fromFrame = kfRef.frame
      if (clamped !== fromFrame) {
        kfRef.frame = clamped
        track.sort((a: BoneKeyframe, b: BoneKeyframe) => a.frame - b.frame)
      }
      if (dv) {
        const ch = ALL_CHANNELS.find((c) => c.key === chKey)
        if (ch) ch.set(kfRef, ch.get(kfRef) + dv)
      }
      for (const s of selectedKeyframes) {
        if (s.bone === bone && s.channel === chKey && s.frame === fromFrame) {
          ;(s as { frame: number }).frame = clamped
        }
      }
      dragTouchedRef.current = true
      dragRedrawRef.current?.()
    },
    [clip, selectedKeyframes],
  )

  const onMoveMorphKeyframe = useCallback(
    (morph: string, kfRef: MorphKeyframe, toFrame: number, dw: number) => {
      if (!clip) return
      const track = clip.morphTracks.get(morph)
      if (!track || !track.includes(kfRef)) return
      const clamped = Math.max(0, toFrame)
      const fromFrame = kfRef.frame
      if (clamped !== fromFrame) {
        kfRef.frame = clamped
        track.sort((a, b) => a.frame - b.frame)
      }
      if (dw) kfRef.weight = Math.max(0, Math.min(1, kfRef.weight + dw))
      for (const s of selectedKeyframes) {
        if (s.morph === morph && s.frame === fromFrame) (s as { frame: number }).frame = clamped
      }
      dragTouchedRef.current = true
      dragRedrawRef.current?.()
    },
    [clip, selectedKeyframes],
  )

  const onEndKeyframeDrag = useCallback(() => {
    if (!dragTouchedRef.current) return
    dragTouchedRef.current = false
    // Single commit for the whole drag — this is what lands in undo/redo and
    // triggers the engine reupload via EngineBridge.
    commit((c) =>
      c ? { ...c, boneTracks: new Map(c.boneTracks), morphTracks: new Map(c.morphTracks) } : null,
    )
    // Notify selection subscribers with a new array reference (entries were
    // already mutated in place during the drag).
    setSelectedKeyframes((prev) => [...prev])
  }, [commit, setSelectedKeyframes])

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
            "bg-transparent"
          )}
          onClick={() => setPlaying((p) => !p)}
        >
          {playing ? (
            <Pause className="size-3.5 fill-current" strokeWidth={1.5} />
          ) : (
            <Play className="size-3.5 fill-current" strokeWidth={1.5} />
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
        <div className="mx-0.5 flex min-w-0 items-center gap-0.5 whitespace-nowrap rounded-md border border-border/50 bg-card px-1 py-px font-mono text-[9px] tabular-nums text-muted-foreground">
          <span className="opacity-60">F</span>
          <input
            type="text"
            inputMode="numeric"
            aria-label="Current frame"
            disabled={!clip}
            value={frameDraft ?? padFrame4(currentFrame)}
            onFocus={() => setFrameDraft(padFrame4(currentFrame))}
            onChange={(e) => setFrameDraft(e.target.value)}
            onBlur={() => {
              const raw = frameDraft ?? ""
              setFrameDraft(null)
              const v = parseInt(raw.replace(/\s/g, ""), 10)
              if (!Number.isFinite(v) || !clip) return
              setPlaying(false)
              setCurrentFrame(Math.max(0, Math.min(fc, v)))
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter") (e.target as HTMLInputElement).blur()
            }}
            className={cn(
              "h-4 w-8 min-w-0 rounded border border-transparent bg-transparent px-0.5 text-right text-[9px] tabular-nums outline-none",
              "focus-visible:border-ring focus-visible:ring-1 focus-visible:ring-ring/30",
              !clip && "pointer-events-none opacity-40",
            )}
          />
          <span className="opacity-40">/</span>
          <input
            type="text"
            inputMode="numeric"
            aria-label="Clip end frame"
            disabled={!clip}
            value={endDraft ?? padFrame4(fc)}
            onFocus={() => setEndDraft(padFrame4(fc))}
            onChange={(e) => setEndDraft(e.target.value)}
            onBlur={() => {
              const raw = endDraft ?? ""
              setEndDraft(null)
              const v = parseInt(raw.replace(/\s/g, ""), 10)
              if (!Number.isFinite(v) || !clip) return
              commit({ ...clip, frameCount: v })
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter") (e.target as HTMLInputElement).blur()
            }}
            className={cn(
              "h-4 w-8 min-w-0 rounded border border-transparent bg-transparent px-0.5 text-right text-[9px] tabular-nums outline-none",
              "focus-visible:border-ring focus-visible:ring-1 focus-visible:ring-ring/30",
              !clip && "pointer-events-none opacity-40",
            )}
          />
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
        <span className="shrink-0 px-1 text-[10px] uppercase tracking-wide text-muted-foreground">Time</span>
        <ZoomRuler min={minPxPerFrame} max={MAX_PX} value={pxPerFrame} onChange={zoomTo} />
        <span className="shrink-0 px-1 pl-2 text-[10px] uppercase tracking-wide text-muted-foreground">Value</span>
        <ZoomRuler min={Y_ZOOM_MIN} max={Y_ZOOM_MAX} value={yZoom} onChange={setYZoom} />
      </div>
      {/* Canvas */}
      <div ref={timelineAreaRef} style={{ flex: 1, minHeight: 0 }}>
        {clip ? (
          <TimelineCanvas
            clip={clip}
            pxPerFrame={pxPerFrame}
            yZoom={yZoom}
            scrollX={scrollX}
            currentFrame={currentFrame}
            selectedBone={selectedBone}
            selectedMorph={selectedMorph}
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
            onMoveMorphKeyframe={onMoveMorphKeyframe}
            onEndKeyframeDrag={onEndKeyframeDrag}
            playheadDrawRef={innerDrawRef}
            dragRedrawRef={dragRedrawRef}
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
