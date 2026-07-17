"use client"

/** Clip-arrangement mode UI: library panel (left sidebar tab), track-lane
 *  timeline (replaces the dopesheet while in clip mode), and placement
 *  properties (right sidebar). All arrangement state lives in the project
 *  store; drags preview via direct style writes and commit to the store on
 *  release, matching the studio's hot-path discipline. */

import {
  memo,
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
  type RefObject,
} from "react"
import { Copy, FileMusic, Pencil, Plus, Trash2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { usePlayback, usePlaybackFrameRef, usePlaybackSelector } from "@/context/playback-context"
import { useProjectActions, useProjectSelector } from "@/context/project-context"
import {
  arrangementEndFrame,
  findPlacement,
  placementFrames,
  scaledClipFrames,
  type ClipPlacement,
  type LibraryClip,
} from "@/lib/project"
import { cn } from "@/lib/utils"

const LANE_HEIGHT = 48
const RULER_HEIGHT = 24
const MIN_PPF = 0.25
const MAX_PPF = 12

function clipKeyCount(clip: LibraryClip["clip"]): number {
  let n = 0
  for (const t of clip.boneTracks.values()) n += t.length
  for (const t of clip.morphTracks.values()) n += t.length
  return n
}

// ─── Library panel (left sidebar "Clip" tab) ─────────────────────────────

interface ClipPanelProps {
  /** Opens the Import VMD file picker (input element lives in StudioPage). */
  onImportVmd: () => void
  /** Make this library clip the bone-mode editing target. */
  onActivateClip: (id: string) => void
}

export const ClipPanel = memo(function ClipPanel({ onImportVmd, onActivateClip }: ClipPanelProps) {
  const library = useProjectSelector((s) => s.library)
  const activeClipId = useProjectSelector((s) => s.activeClipId)
  const firstTrackId = useProjectSelector((s) => s.tracks[0]?.id ?? null)
  const { placeClip, renameClip, removeClip } = useProjectActions()
  const playbackFrameRef = usePlaybackFrameRef()
  const [renamingId, setRenamingId] = useState<string | null>(null)

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between px-3 py-1">
        <span className="text-[10.5px] font-medium uppercase leading-tight tracking-widest text-muted-foreground">
          Library
        </span>
        <Button
          variant="ghost"
          size="sm"
          className="h-5 gap-1 px-1.5 text-[10px] text-muted-foreground"
          onClick={onImportVmd}
        >
          <FileMusic className="size-3" />
          Import
        </Button>
      </div>
      {library.length === 0 ? (
        <div className="px-3 py-1.5 text-[11px] leading-relaxed text-muted-foreground">
          Import a VMD to build a clip library, then place clips on tracks below.
        </div>
      ) : (
        <div className="min-h-0 flex-1 overflow-y-auto [scrollbar-width:thin]">
          {library.map((c) => {
            const active = c.id === activeClipId
            return (
              <div
                key={c.id}
                className={cn(
                  "group flex items-center gap-1.5 px-3 py-1.5 text-[11px]",
                  active ? "bg-accent/60 text-foreground" : "text-muted-foreground hover:bg-accent/30",
                )}
              >
                <span
                  className={cn("size-1.5 shrink-0 rounded-full", active ? "bg-primary" : "bg-muted-foreground/30")}
                  title={active ? "Editing in Bone/Morph mode" : undefined}
                />
                {renamingId === c.id ? (
                  <input
                    autoFocus
                    defaultValue={c.name}
                    className="min-w-0 flex-1 rounded border border-border bg-background px-1 py-0 text-[11px]"
                    onBlur={(e) => {
                      renameClip(c.id, e.target.value)
                      setRenamingId(null)
                    }}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") (e.target as HTMLInputElement).blur()
                      if (e.key === "Escape") setRenamingId(null)
                    }}
                  />
                ) : (
                  <span
                    className="min-w-0 flex-1 truncate"
                    title={`${c.name} — ${c.clip.frameCount}f, ${clipKeyCount(c.clip)} keys (double-click to rename)`}
                    onDoubleClick={() => setRenamingId(c.id)}
                  >
                    {c.name}
                  </span>
                )}
                <span className="shrink-0 text-[9px] tabular-nums text-muted-foreground/60">{c.clip.frameCount}f</span>
                <span className="hidden shrink-0 items-center gap-0.5 group-hover:flex">
                  <button
                    type="button"
                    className="rounded p-0.5 hover:bg-accent hover:text-foreground"
                    title="Edit keyframes (Bone/Morph mode)"
                    onClick={() => onActivateClip(c.id)}
                  >
                    <Pencil className="size-3" />
                  </button>
                  <button
                    type="button"
                    className="rounded p-0.5 hover:bg-accent hover:text-foreground"
                    title="Place on Track 1 at playhead"
                    onClick={() => {
                      if (firstTrackId) placeClip(c.id, firstTrackId, Math.round(playbackFrameRef.current))
                    }}
                  >
                    <Plus className="size-3" />
                  </button>
                  <button
                    type="button"
                    className="rounded p-0.5 hover:bg-accent hover:text-red-400"
                    title="Remove from library (and all tracks)"
                    onClick={() => {
                      if (window.confirm(`Remove "${c.name}" from the library and all tracks?`)) removeClip(c.id)
                    }}
                  >
                    <Trash2 className="size-3" />
                  </button>
                </span>
              </div>
            )
          })}
        </div>
      )}
      <div className="border-t border-border px-3 py-1.5 text-[10px] leading-relaxed text-muted-foreground/70">
        Tracks compose top-down: for each bone, the topmost track with keys for it wins — face-only and body-only
        clips blend automatically.
      </div>
    </div>
  )
})

// ─── Track-lane timeline ─────────────────────────────────────────────────

interface ClipModeTimelineProps {
  /** Shared with the playback rAF loop — registers a playhead redraw fn while
   *  mounted (the bone-mode Timeline registers its own when it remounts). */
  playheadDrawRef: RefObject<((frame: number) => void) | null>
}

type DragState = {
  kind: "move" | "trim"
  placementId: string
  el: HTMLDivElement
  startX: number
  startY: number
  origStart: number
  origLen: number
  fullLen: number
  origLaneIndex: number
  laneCount: number
  /** Live preview values, committed on pointerup. */
  nextStart: number
  nextLen: number
  nextLaneIndex: number
  moved: boolean
}

export const ClipModeTimeline = memo(function ClipModeTimeline({ playheadDrawRef }: ClipModeTimelineProps) {
  const tracks = useProjectSelector((s) => s.tracks)
  const library = useProjectSelector((s) => s.library)
  const selectedPlacementId = useProjectSelector((s) => s.selectedPlacementId)
  const { addTrack, removeTrack, toggleTrackMute, toggleTrackSolo, updatePlacement, setSelectedPlacement } =
    useProjectActions()
  const { setCurrentFrame } = usePlayback()
  const currentFrame = usePlaybackSelector((s) => s.currentFrame)
  const playbackFrameRef = usePlaybackFrameRef()

  const [ppf, setPpf] = useState(2)
  const ppfRef = useRef(ppf)
  ppfRef.current = ppf

  const scrollRef = useRef<HTMLDivElement>(null)
  const contentRef = useRef<HTMLDivElement>(null)
  const playheadElRef = useRef<HTMLDivElement>(null)
  const dragRef = useRef<DragState | null>(null)

  /** Latest tracks for the pointerup commit (lane index → track id). */
  const tracksRef = useRef(tracks)
  tracksRef.current = tracks

  const clipById = new Map(library.map((c) => [c.id, c]))
  const endFrame = arrangementEndFrame(tracks, library)
  const rulerFrames = Math.max(300, endFrame + 90)

  // ── Playhead: registered with the shared draw ref so the playback rAF
  //    loop moves it imperatively; paused scrubs redraw via React state.
  useLayoutEffect(() => {
    const draw = (frame: number) => {
      const el = playheadElRef.current
      if (el) el.style.transform = `translateX(${frame * ppfRef.current}px)`
    }
    playheadDrawRef.current = draw
    draw(playbackFrameRef.current)
    return () => {
      if (playheadDrawRef.current === draw) playheadDrawRef.current = null
    }
  }, [playheadDrawRef, playbackFrameRef])

  useLayoutEffect(() => {
    playheadDrawRef.current?.(Math.max(0, currentFrame))
  }, [currentFrame, ppf, playheadDrawRef])

  // ── Ctrl/⌘ + wheel zoom (native listener — React's onWheel is passive).
  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      if (!e.ctrlKey && !e.metaKey) return
      e.preventDefault()
      setPpf((prev) => Math.min(MAX_PPF, Math.max(MIN_PPF, prev * (e.deltaY < 0 ? 1.2 : 1 / 1.2))))
    }
    el.addEventListener("wheel", onWheel, { passive: false })
    return () => el.removeEventListener("wheel", onWheel)
  }, [])

  // ── Ruler scrub ──
  const scrubToClientX = useCallback(
    (clientX: number) => {
      const content = contentRef.current
      if (!content) return
      const x = clientX - content.getBoundingClientRect().left
      setCurrentFrame(Math.max(0, Math.round(x / ppfRef.current)))
    },
    [setCurrentFrame],
  )

  const onRulerPointerDown = useCallback(
    (e: ReactPointerEvent<HTMLDivElement>) => {
      e.currentTarget.setPointerCapture(e.pointerId)
      scrubToClientX(e.clientX)
    },
    [scrubToClientX],
  )

  const onRulerPointerMove = useCallback(
    (e: ReactPointerEvent<HTMLDivElement>) => {
      if (e.buttons & 1) scrubToClientX(e.clientX)
    },
    [scrubToClientX],
  )

  // ── Placement drag (move / trim). Preview by mutating the block's style;
  //    commit one store update on release. ──
  const beginDrag = useCallback(
    (e: ReactPointerEvent<HTMLElement>, placement: ClipPlacement, laneIndex: number, kind: DragState["kind"]) => {
      const lib = libraryRef.current.find((c) => c.id === placement.clipId)
      if (!lib) return
      e.stopPropagation()
      setSelectedPlacement(placement.id)
      const el = (e.target as HTMLElement).closest("[data-placement]") as HTMLDivElement | null
      if (!el) return
      el.setPointerCapture?.(e.pointerId)
      const fullLen = scaledClipFrames(lib.clip, placement.timeScale)
      const origLen = placementFrames(placement, lib.clip)
      dragRef.current = {
        kind,
        placementId: placement.id,
        el,
        startX: e.clientX,
        startY: e.clientY,
        origStart: placement.startFrame,
        origLen,
        fullLen,
        origLaneIndex: laneIndex,
        laneCount: tracksRef.current.length,
        nextStart: placement.startFrame,
        nextLen: origLen,
        nextLaneIndex: laneIndex,
        moved: false,
      }
    },
    [setSelectedPlacement],
  )

  /** Latest library for gesture handlers (avoids stale closures without
   *  re-creating callbacks per arrangement change). */
  const libraryRef = useRef(library)
  libraryRef.current = library

  const onBlockPointerMove = useCallback((e: ReactPointerEvent<HTMLDivElement>) => {
    const d = dragRef.current
    if (!d) return
    const dxFrames = Math.round((e.clientX - d.startX) / ppfRef.current)
    if (d.kind === "move") {
      d.nextStart = Math.max(0, d.origStart + dxFrames)
      const dLane = Math.round((e.clientY - d.startY) / LANE_HEIGHT)
      d.nextLaneIndex = Math.min(d.laneCount - 1, Math.max(0, d.origLaneIndex + dLane))
      d.el.style.left = `${d.nextStart * ppfRef.current}px`
      d.el.style.transform = `translateY(${(d.nextLaneIndex - d.origLaneIndex) * LANE_HEIGHT}px)`
      d.el.style.zIndex = "10"
    } else {
      d.nextLen = Math.min(d.fullLen, Math.max(1, d.origLen + dxFrames))
      d.el.style.width = `${d.nextLen * ppfRef.current}px`
    }
    if (dxFrames !== 0 || d.nextLaneIndex !== d.origLaneIndex) d.moved = true
  }, [])

  const onBlockPointerUp = useCallback(() => {
    const d = dragRef.current
    if (!d) return
    dragRef.current = null
    d.el.style.transform = ""
    d.el.style.zIndex = ""
    if (!d.moved) return
    if (d.kind === "move") {
      updatePlacement(d.placementId, {
        startFrame: d.nextStart,
        ...(d.nextLaneIndex !== d.origLaneIndex ? { trackId: tracksRef.current[d.nextLaneIndex]?.id } : {}),
      })
    } else {
      updatePlacement(d.placementId, { length: d.nextLen >= d.fullLen ? undefined : d.nextLen })
    }
  }, [updatePlacement])

  // Frame ticks: pick a step that keeps labels ≥ ~56px apart.
  const tickStep = ppf >= 4 ? 15 : ppf >= 2 ? 30 : ppf >= 1 ? 60 : ppf >= 0.5 ? 120 : 300
  const ticks: number[] = []
  for (let f = 0; f <= rulerFrames; f += tickStep) ticks.push(f)

  const anySolo = tracks.some((t) => t.solo)

  return (
    <div className="flex h-full select-none text-xs">
      {/* Track headers */}
      <div className="flex w-32 shrink-0 flex-col border-r border-border">
        <div
          className="flex shrink-0 items-center justify-between border-b border-border px-2"
          style={{ height: RULER_HEIGHT }}
        >
          <span className="text-[10px] font-medium uppercase tracking-widest text-muted-foreground">Tracks</span>
          <button
            type="button"
            className="rounded p-0.5 text-muted-foreground hover:bg-accent hover:text-foreground"
            title="Add track"
            onClick={addTrack}
          >
            <Plus className="size-3" />
          </button>
        </div>
        <div className="min-h-0 flex-1 overflow-hidden">
          {tracks.map((t) => {
            const inactive = anySolo ? !t.solo : t.mute
            return (
              <div
                key={t.id}
                className="group flex items-center gap-1 border-b border-border/60 px-2"
                style={{ height: LANE_HEIGHT }}
              >
                <span
                  className={cn(
                    "min-w-0 flex-1 truncate text-[11px]",
                    inactive ? "text-muted-foreground/40" : "text-muted-foreground",
                  )}
                >
                  {t.name}
                </span>
                <button
                  type="button"
                  className={cn(
                    "rounded px-1 text-[9px] font-semibold",
                    t.mute ? "bg-red-500/30 text-red-300" : "text-muted-foreground/50 hover:bg-accent",
                  )}
                  title="Mute"
                  onClick={() => toggleTrackMute(t.id)}
                >
                  M
                </button>
                <button
                  type="button"
                  className={cn(
                    "rounded px-1 text-[9px] font-semibold",
                    t.solo ? "bg-yellow-500/30 text-yellow-300" : "text-muted-foreground/50 hover:bg-accent",
                  )}
                  title="Solo"
                  onClick={() => toggleTrackSolo(t.id)}
                >
                  S
                </button>
                <button
                  type="button"
                  className="hidden rounded p-0.5 text-muted-foreground/60 hover:bg-accent hover:text-red-400 group-hover:block disabled:opacity-30"
                  title="Remove track"
                  disabled={tracks.length <= 1}
                  onClick={() => {
                    if (
                      t.placements.length === 0 ||
                      window.confirm(`Remove ${t.name} and its ${t.placements.length} clip block(s)?`)
                    )
                      removeTrack(t.id)
                  }}
                >
                  <Trash2 className="size-3" />
                </button>
              </div>
            )
          })}
        </div>
      </div>

      {/* Lanes */}
      <div ref={scrollRef} className="relative min-w-0 flex-1 overflow-auto [scrollbar-width:thin]">
        <div ref={contentRef} className="relative" style={{ width: rulerFrames * ppf, minWidth: "100%" }}>
          {/* Ruler */}
          <div
            className="sticky top-0 z-20 cursor-crosshair border-b border-border bg-background"
            style={{ height: RULER_HEIGHT }}
            onPointerDown={onRulerPointerDown}
            onPointerMove={onRulerPointerMove}
          >
            {ticks.map((f) => (
              <span
                key={f}
                className="absolute top-1 border-l border-border pl-1 text-[9px] tabular-nums text-muted-foreground/60"
                style={{ left: f * ppf, height: RULER_HEIGHT - 8 }}
              >
                {f}
              </span>
            ))}
          </div>
          {/* Track lanes */}
          {tracks.map((t, laneIndex) => {
            const inactive = anySolo ? !t.solo : t.mute
            return (
              <div
                key={t.id}
                className={cn("relative border-b border-border/60", inactive && "opacity-40")}
                style={{ height: LANE_HEIGHT }}
                onPointerDown={() => setSelectedPlacement(null)}
              >
                {t.placements.map((p) => {
                  const lib = clipById.get(p.clipId)
                  if (!lib) return null
                  const len = placementFrames(p, lib.clip)
                  const selected = p.id === selectedPlacementId
                  const trimmed = p.length != null && p.length < scaledClipFrames(lib.clip, p.timeScale)
                  return (
                    <div
                      key={p.id}
                      data-placement={p.id}
                      className={cn(
                        "absolute top-1.5 bottom-1.5 flex cursor-grab items-center overflow-hidden rounded border px-1.5 active:cursor-grabbing",
                        selected
                          ? "border-primary bg-primary/40 text-foreground"
                          : "border-primary/40 bg-primary/15 text-muted-foreground hover:bg-primary/25",
                      )}
                      style={{ left: p.startFrame * ppf, width: Math.max(4, len * ppf) }}
                      onPointerDown={(e) => beginDrag(e, p, laneIndex, "move")}
                      onPointerMove={onBlockPointerMove}
                      onPointerUp={onBlockPointerUp}
                      onPointerCancel={onBlockPointerUp}
                    >
                      <span className="pointer-events-none min-w-0 flex-1 truncate text-[10px]">
                        {lib.name}
                        {p.timeScale != null ? ` ×${p.timeScale}` : ""}
                        {trimmed ? " ✂" : ""}
                      </span>
                      <div
                        className="absolute top-0 bottom-0 right-0 w-1.5 cursor-ew-resize bg-primary/30 hover:bg-primary/70"
                        title="Trim length"
                        onPointerDown={(e) => beginDrag(e, p, laneIndex, "trim")}
                      />
                    </div>
                  )
                })}
              </div>
            )
          })}
          {/* Playhead */}
          <div
            ref={playheadElRef}
            className="pointer-events-none absolute top-0 bottom-0 z-10 w-px bg-red-500"
            style={{ left: 0 }}
          />
        </div>
      </div>
    </div>
  )
})

// ─── Placement properties (right sidebar while in clip mode) ─────────────

export const ClipModeProperties = memo(function ClipModeProperties() {
  const tracks = useProjectSelector((s) => s.tracks)
  const library = useProjectSelector((s) => s.library)
  const selectedPlacementId = useProjectSelector((s) => s.selectedPlacementId)
  const { updatePlacement, duplicatePlacement, deletePlacement } = useProjectActions()

  const found = selectedPlacementId != null ? findPlacement(tracks, selectedPlacementId) : null
  const lib = found ? library.find((c) => c.id === found.placement.clipId) : null

  if (!found || !lib) {
    return (
      <div className="flex flex-col gap-2 text-[11px] text-muted-foreground">
        <div className="text-[10.5px] font-medium uppercase tracking-widest text-muted-foreground/70">Clip</div>
        <div className="leading-relaxed">
          Select a clip block on a track to edit its placement, or use the Library tab to import and place clips.
        </div>
      </div>
    )
  }

  const p = found.placement
  const full = scaledClipFrames(lib.clip, p.timeScale)
  const len = placementFrames(p, lib.clip)

  return (
    <div className="flex flex-col gap-2 text-[11px]">
      <div className="text-[10.5px] font-medium uppercase tracking-widest text-muted-foreground/70">Clip</div>
      <div className="truncate font-medium text-foreground" title={lib.name}>
        {lib.name}
      </div>
      <div className="text-[10px] text-muted-foreground">
        {found.track.name} · source {lib.clip.frameCount}f
      </div>

      <label className="mt-1 flex items-center justify-between gap-2 text-muted-foreground">
        Start
        <NumberField value={p.startFrame} min={0} onCommit={(v) => updatePlacement(p.id, { startFrame: v })} />
      </label>
      <label className="flex items-center justify-between gap-2 text-muted-foreground">
        Length
        <span className="flex items-center gap-1">
          <NumberField
            value={len}
            min={1}
            max={full}
            onCommit={(v) => updatePlacement(p.id, { length: v >= full ? undefined : v })}
          />
          {p.length != null ? (
            <button
              type="button"
              className="rounded border border-border px-1 py-0.5 text-[9px] text-muted-foreground hover:bg-accent"
              onClick={() => updatePlacement(p.id, { length: undefined })}
            >
              Full
            </button>
          ) : null}
        </span>
      </label>
      <label className="flex items-center justify-between gap-2 text-muted-foreground">
        Time scale
        <NumberField
          value={p.timeScale ?? 1}
          min={0.1}
          max={8}
          step={0.1}
          onCommit={(v) => updatePlacement(p.id, { timeScale: v === 1 ? undefined : v, length: undefined })}
        />
      </label>

      <div className="mt-2 flex gap-1.5">
        <Button
          variant="secondary"
          size="sm"
          className="h-6 flex-1 gap-1 text-[10px]"
          onClick={() => duplicatePlacement(p.id)}
        >
          <Copy className="size-3" />
          Duplicate
        </Button>
        <Button
          variant="secondary"
          size="sm"
          className="h-6 flex-1 gap-1 text-[10px] hover:text-red-400"
          onClick={() => deletePlacement(p.id)}
        >
          <Trash2 className="size-3" />
          Remove
        </Button>
      </div>
      <div className="mt-1 text-[9.5px] leading-relaxed text-muted-foreground/60">
        Removing a block only takes it off the track — the clip stays in the library.
      </div>
    </div>
  )
})

/** Number input that commits on blur/Enter (store updates rebake the whole
 *  arrangement, so per-keystroke commits would be wasteful). */
function NumberField({
  value,
  min,
  max,
  step,
  onCommit,
}: {
  value: number
  min?: number
  max?: number
  step?: number
  onCommit: (v: number) => void
}) {
  const [text, setText] = useState(String(value))
  const editingRef = useRef(false)
  useEffect(() => {
    if (!editingRef.current) setText(String(value))
  }, [value])
  const commit = () => {
    editingRef.current = false
    let v = Number(text)
    if (!Number.isFinite(v)) {
      setText(String(value))
      return
    }
    if (min != null) v = Math.max(min, v)
    if (max != null) v = Math.min(max, v)
    v = step != null ? Math.round(v / step) * step : Math.round(v)
    v = Number(v.toFixed(2))
    setText(String(v))
    if (v !== value) onCommit(v)
  }
  return (
    <input
      type="number"
      value={text}
      min={min}
      max={max}
      step={step ?? 1}
      className="w-16 rounded border border-border bg-background px-1 py-0.5 text-right text-[11px] tabular-nums"
      onFocus={() => {
        editingRef.current = true
      }}
      onChange={(e) => setText(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === "Enter") (e.target as HTMLInputElement).blur()
      }}
    />
  )
}
