"use client"

/** Headless component that owns every engine-coupled effect: initialization,
 *  clip upload, scrub/seek, play/pause, end-of-clip handling, and the 60Hz
 *  playback rAF loop that imperatively drives the timeline playhead.
 *
 *  StudioPage mounts this once (with refs + chrome setters) and otherwise has
 *  no engine logic in its render body. EngineBridge returns null. */

import {
  useCallback,
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
export const MODEL_PATH = "/models/塞尔凯特/塞尔凯特.pmx"
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

  // ─── Physics reset after animation-time jumps ───────────────────────
  //     `model.seek` retargets the animation; rigid bodies only catch up
  //     on the engine's next tick, so resetting in the same call zeroes
  //     velocities against the *old* pose and things explode. One rAF
  //     of delay lets the engine propagate the new pose to physics, then
  //     `resetPhysics` stabilizes velocities at the new rest state.
  //
  //     Small frame-to-frame deltas (smooth scrub drag) don't need a
  //     reset — physics can integrate continuously between neighboring
  //     poses without blowing up. Only jumps beyond `RESET_PHYSICS_FRAME_THRESHOLD`
  //     trigger the next-frame reset. Bursts of qualifying seeks collapse
  //     into one reset via rAF cancellation.
  const RESET_PHYSICS_FRAME_THRESHOLD = 2
  const physicsResetRafRef = useRef<number | null>(null)
  const lastSeekFrameRef = useRef<number | null>(null)

  const maybeResetPhysicsAfterSeek = useCallback(
    (frame: number) => {
      const prev = lastSeekFrameRef.current
      lastSeekFrameRef.current = frame
      if (prev !== null && Math.abs(frame - prev) <= RESET_PHYSICS_FRAME_THRESHOLD) return
      if (physicsResetRafRef.current !== null) cancelAnimationFrame(physicsResetRafRef.current)
      physicsResetRafRef.current = requestAnimationFrame(() => {
        physicsResetRafRef.current = null
        engineRef.current?.resetPhysics()
      })
    },
    [engineRef],
  )

  useEffect(() => {
    return () => {
      if (physicsResetRafRef.current !== null) cancelAnimationFrame(physicsResetRafRef.current)
      physicsResetRafRef.current = null
    }
  }, [])

  // ─── Engine init + initial model/VMD load ───────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const el = canvas
    let disposed = false

    async function initEngine() {
      try {
        const engine = new Engine(el, {
          camera: {
            distance: 31.5,
            target: new Vec3(0, 11.5, 0),
            
          },
        bloom:{color: new Vec3(1, 0.1, 0.88)},
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

          engine.setMaterialPresets("reze", {
            eye: ["眼睛", "眼白", "目白", "右瞳","左瞳","眉毛"],
            face: ["脸", "face01"],
            body: ["皮肤", "skin"],
            hair: ["头发", "hair_f"],
            cloth_smooth: [
              "衣服",
              "裙子",
              "裙带",
              "裙布",
              "外套",
              "外套饰",
              "裤子",
              "裤子0",
              "腿环",
              "发饰",
              "鞋子",
              "鞋子饰",
              "shirt",
              "shoes",
              "shorts",
              "trigger",
              "dress",
              "hair_accessory",
              "cloth01_shoes"
            ],
            stockings: ["袜子", "stockings"],
            metal: ["metal01","earring"],
          })

          engine.addGround({ diffuseColor: new Vec3(0.05, 0.04, 0.06) })
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
            lastSeekFrameRef.current = 0
            requestAnimationFrame(() => engine.resetPhysics())
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
    const f = Math.max(0, currentFrameRef.current)
    model.seek(f / 30)
    maybeResetPhysicsAfterSeek(f)
  }, [clip, currentFrameRef, modelRef, maybeResetPhysicsAfterSeek])

  // ─── Scrub: when paused, React owns the playhead and pushes seeks into
  //     the engine. When playing, the engine owns the playhead; the rAF
  //     loop below reads from it — do NOT seek here. ────────────────────
  useLayoutEffect(() => {
    const model = modelRef.current
    if (!model || !clip) return
    if (!playing) {
      const f = Math.max(0, currentFrame)
      model.seek(f / 30)
      maybeResetPhysicsAfterSeek(f)
    }
  }, [currentFrame, clip, playing, modelRef, maybeResetPhysicsAfterSeek])

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
      const f = Math.max(0, startFrame)
      model.seek(f / 30)
      maybeResetPhysicsAfterSeek(f)
      model.play()
      if (model.name === "reze") model.setMorphWeight("抗穿模", 0.5)
    } else {
      model.pause()
    }
  }, [playing, clip, frameCount, setCurrentFrame, currentFrameRef, modelRef, maybeResetPhysicsAfterSeek])

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
