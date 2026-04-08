"use client"

/** Headless component that owns every engine-coupled effect: initialization,
 *  clip upload, scrub/seek, play/pause, end-of-clip handling, and the 60Hz
 *  playback rAF loop that imperatively drives the timeline playhead.
 *
 *  StudioPage mounts this once (with refs + chrome setters) and otherwise has
 *  no engine logic in its render body. EngineBridge returns null. */

import {
  useEffect,
  useLayoutEffect,
  useRef,
  type Dispatch,
  type RefObject,
  type SetStateAction,
} from "react"
import { Engine, Model, Vec3 } from "reze-engine"
import { useStudioActions, useStudioSelector } from "@/context/studio-context"
import { usePlayback } from "@/context/playback-context"
import { useStudioStatusActions } from "@/components/studio-status"

// ─── Constants shared with StudioPage file handlers ──────────────────────
export const MODEL_PATH = "/models/reze/reze.pmx"
export const VMD_PATH = "/animations/miku.vmd"
export const STUDIO_ANIM_NAME = "studio"
export const BUNDLED_PMX_FILENAME = MODEL_PATH.replace(/^.*\//, "") || "model.pmx"

// ─── Filename helpers — used by EngineBridge (initial VMD load) and by
//     StudioPage (file menu / export). Kept here so both can import without
//     a circular dependency. ──────────────────────────────────────────────
export function fileStem(pathOrName: string): string {
  const base = pathOrName.replace(/^.*[/\\]/, "")
  const i = base.lastIndexOf(".")
  return (i > 0 ? base.slice(0, i) : base).trim() || "clip"
}

export function sanitizeClipFilenameBase(name: string): string {
  const s = name.trim() || "clip"
  const cleaned = s.replace(/[/\\<>:"|?*\x00-\x1f]/g, "-").replace(/-+/g, "-")
  return cleaned.slice(0, 120).replace(/^-|-$/g, "") || "clip"
}

interface EngineBridgeProps {
  canvasRef: RefObject<HTMLCanvasElement | null>
  engineRef: RefObject<Engine | null>
  modelRef: RefObject<Model | null>
  currentFrameRef: RefObject<number>
  playheadDrawRef: RefObject<((frame: number) => void) | null>
  documentDirtyRef: RefObject<boolean>
  suppressClipDirtyRef: RefObject<number>
  setPmxBoneNames: Dispatch<SetStateAction<ReadonlySet<string>>>
  setModelBoneOrder: Dispatch<SetStateAction<string[]>>
  setMorphNames: Dispatch<SetStateAction<string[]>>
  setEngineError: Dispatch<SetStateAction<string | null>>
  setStudioReady: Dispatch<SetStateAction<boolean>>
}

export function EngineBridge({
  canvasRef,
  engineRef,
  modelRef,
  currentFrameRef,
  playheadDrawRef,
  documentDirtyRef,
  suppressClipDirtyRef,
  setPmxBoneNames,
  setModelBoneOrder,
  setMorphNames,
  setEngineError,
  setStudioReady,
}: EngineBridgeProps) {
  const clip = useStudioSelector((s) => s.clip)
  const { commit, setClipDisplayName, setSelectedMorph } = useStudioActions()
  const { currentFrame, setCurrentFrame, playing, setPlaying } = usePlayback()
  const { setPmxFileName: setStatusPmxFileName, setFps: setStatusFps } = useStudioStatusActions()
  const frameCount = clip?.frameCount ?? 0

  const playRef = useRef(false)
  const lastFpsRef = useRef<number | null>(null)

  // ─── Engine init + initial model/VMD load ───────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const el = canvas
    let disposed = false

    async function initEngine() {
      try {
        const engine = new Engine(el, {
          ambientColor: new Vec3(0.86, 0.84, 0.88),
          cameraDistance: 31.5,
          cameraTarget: new Vec3(0, 11.5, 0),
        })
        await engine.init()
        if (disposed) return

        try {
          const model = await engine.loadModel("reze", MODEL_PATH)
          if (disposed) return
          modelRef.current = model
          const sk = model.getSkeleton().bones.map((b) => b.name)
          setPmxBoneNames(new Set(sk))
          setModelBoneOrder(sk)
          setMorphNames(model.getMorphing().morphs.map((m) => m.name))
          setStatusPmxFileName(BUNDLED_PMX_FILENAME)
          model.setMorphWeight("抗穿模", 0.5)
          engine.addGround({ diffuseColor: new Vec3(0.14, 0.12, 0.16) })
        } catch {
          setEngineError(`Add model at public${MODEL_PATH}`)
        }

        lastFpsRef.current = null
        engine.runRenderLoop(() => {
          const fps = engine.getStats().fps
          if (fps === lastFpsRef.current) return
          lastFpsRef.current = fps
          setStatusFps(fps > 0 ? fps : null)
        })

        try {
          await modelRef.current?.loadVmd(STUDIO_ANIM_NAME, VMD_PATH)
          if (disposed) return
          const c = modelRef.current?.getClip(STUDIO_ANIM_NAME)
          if (c) {
            suppressClipDirtyRef.current += 1
            commit(c)
            documentDirtyRef.current = false
            setClipDisplayName(sanitizeClipFilenameBase(fileStem(VMD_PATH)))
            modelRef.current?.show(STUDIO_ANIM_NAME)
            modelRef.current?.seek(0)
            if (modelRef.current?.name === "reze") modelRef.current?.setMorphWeight("抗穿模", 0.5)
          }
        } catch (e) {
          console.warn(`VMD load failed — add file at public${VMD_PATH}`, e)
        }
        setStudioReady(true)
        engineRef.current = engine
      } catch (e) {
        console.error(e)
        setEngineError(e instanceof Error ? e.message : String(e))
      }
    }

    void initEngine()

    return () => {
      disposed = true
      setStudioReady(false)
      setModelBoneOrder([])
      setPmxBoneNames(new Set())
      setMorphNames([])
      setSelectedMorph(null)
      setStatusPmxFileName("—")
      setStatusFps(null)
      lastFpsRef.current = null
      modelRef.current = null
      engineRef.current?.stopRenderLoop()
      engineRef.current?.dispose()
      engineRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // ─── Upload clip to engine ONLY on edits (not on playhead movement).
  //     After upload, re-seek to the current React frame so a commit during
  //     pause doesn't snap the viewport back to frame 0. ────────────────
  useEffect(() => {
    const model = modelRef.current
    if (!model || !clip) return
    model.loadClip(STUDIO_ANIM_NAME, clip)
    model.seek(Math.max(0, currentFrameRef.current) / 30)
  }, [clip, currentFrameRef, modelRef])

  // ─── Scrub: when paused, React owns the playhead and pushes seeks into
  //     the engine. When playing, the engine owns the playhead; the rAF
  //     loop below reads from it — do NOT seek here. ────────────────────
  useLayoutEffect(() => {
    const model = modelRef.current
    if (!model || !clip) return
    if (!playing) {
      model.seek(Math.max(0, currentFrame) / 30)
    }
  }, [currentFrame, clip, playing, modelRef])

  // ─── Play / pause ───────────────────────────────────────────────────
  useEffect(() => {
    const model = modelRef.current
    if (!model || !clip) return
    if (playing) {
      // If the user pressed play at the end, rewind to 0 first and mirror.
      let startFrame = currentFrameRef.current
      if (startFrame >= frameCount) {
        startFrame = 0
        setCurrentFrame(0)
      }
      model.seek(Math.max(0, startFrame) / 30)
      model.play()
      if (model.name === "reze") model.setMorphWeight("抗穿模", 0.5)
    } else {
      model.pause()
    }
  }, [playing, clip, frameCount, setCurrentFrame, currentFrameRef, modelRef])

  // Clamp currentFrame to [0, frameCount] whenever the clip shrinks.
  useEffect(() => {
    setCurrentFrame((c) => Math.min(c, frameCount))
  }, [frameCount, setCurrentFrame])

  // ─── Playback rAF loop ──────────────────────────────────────────────
  //     Engine owns the clock during playback; React's job is to mirror it
  //     imperatively into the timeline playhead via `playheadDrawRef`. No
  //     `setCurrentFrame` per-tick — zero reconciliation cost at 60Hz.
  useEffect(() => {
    playRef.current = playing
    if (!playing) return
    if (frameCount <= 0) return
    const model = modelRef.current
    if (!model) return
    let raf: number
    const tick = () => {
      if (!playRef.current) return
      const m = modelRef.current
      if (!m) return
      const progress = m.getAnimationProgress()
      const frame = progress.current * 30
      if (frame >= frameCount) {
        setCurrentFrame(frameCount)
        setPlaying(false)
        return
      }
      currentFrameRef.current = frame
      playheadDrawRef.current?.(frame)
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => {
      cancelAnimationFrame(raf)
      // Flush the final frame into React state so the paused view matches
      // what the playhead was last showing.
      setCurrentFrame(currentFrameRef.current)
    }
  }, [playing, frameCount, setCurrentFrame, setPlaying, currentFrameRef, modelRef, playheadDrawRef])

  return null
}
