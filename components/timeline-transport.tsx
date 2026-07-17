"use client"

/** Transport chrome shared by the bone-mode Timeline and the clip-mode track
 *  timeline: ⏮ ◀ ▶ ▶ ⏭ buttons, scrub slider, F 0000/0000 counter, and the
 *  zoom slider. Extracted from timeline.tsx so both modes render the exact
 *  same controls. */

import { useCallback, useEffect, useRef, useState } from "react"
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
import { usePlayback } from "@/context/playback-context"

export const TIMELINE_FONT = "'SF Mono','Cascadia Code','Fira Code','JetBrains Mono',monospace"

export function padFrame4(n: number) {
  return String(Math.max(0, Math.round(n))).padStart(4, "0")
}

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

export function ZoomRuler({
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

interface TransportControlsProps {
  frameCount: number
  hasClip: boolean
  /** Bone mode: editing the end field commits a new clip frameCount. Omit to
   *  make the end field read-only (clip mode — length comes from tracks). */
  onCommitEndFrame?: (v: number) => void
}

/** ⏮ ◀ ▶(⏸) ▶ ⏭ + scrub slider + F 0000/0000 counter. */
export function TransportControls({ frameCount, hasClip, onCommitEndFrame }: TransportControlsProps) {
  const { currentFrame, setCurrentFrame, playing, setPlaying } = usePlayback()
  const [endDraft, setEndDraft] = useState<string | null>(null)
  const [frameDraft, setFrameDraft] = useState<string | null>(null)
  const fc = frameCount

  return (
    <>
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
          "bg-transparent",
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
          disabled={!hasClip}
          value={frameDraft ?? padFrame4(currentFrame)}
          onFocus={() => setFrameDraft(padFrame4(currentFrame))}
          onChange={(e) => setFrameDraft(e.target.value)}
          onBlur={() => {
            const raw = frameDraft ?? ""
            setFrameDraft(null)
            const v = parseInt(raw.replace(/\s/g, ""), 10)
            if (!Number.isFinite(v) || !hasClip) return
            setPlaying(false)
            setCurrentFrame(Math.max(0, Math.min(fc, v)))
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter") (e.target as HTMLInputElement).blur()
          }}
          className={cn(
            "h-4 w-8 min-w-0 rounded border border-transparent bg-transparent px-0.5 text-right text-[9px] tabular-nums outline-none",
            "focus-visible:border-ring focus-visible:ring-1 focus-visible:ring-ring/30",
            !hasClip && "pointer-events-none opacity-40",
          )}
        />
        <span className="opacity-40">/</span>
        {onCommitEndFrame ? (
          <input
            type="text"
            inputMode="numeric"
            aria-label="Clip end frame"
            disabled={!hasClip}
            value={endDraft ?? padFrame4(fc)}
            onFocus={() => setEndDraft(padFrame4(fc))}
            onChange={(e) => setEndDraft(e.target.value)}
            onBlur={() => {
              const raw = endDraft ?? ""
              setEndDraft(null)
              const v = parseInt(raw.replace(/\s/g, ""), 10)
              if (!Number.isFinite(v) || !hasClip) return
              onCommitEndFrame(v)
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter") (e.target as HTMLInputElement).blur()
            }}
            className={cn(
              "h-4 w-8 min-w-0 rounded border border-transparent bg-transparent px-0.5 text-right text-[9px] tabular-nums outline-none",
              "focus-visible:border-ring focus-visible:ring-1 focus-visible:ring-ring/30",
              !hasClip && "pointer-events-none opacity-40",
            )}
          />
        ) : (
          <span
            aria-label="Arrangement end frame"
            className="h-4 w-8 min-w-0 px-0.5 text-right text-[9px] leading-4 tabular-nums"
            title="Arrangement length follows the clips on tracks"
          >
            {padFrame4(fc)}
          </span>
        )}
      </div>
    </>
  )
}
