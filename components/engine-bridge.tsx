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
import type { AnimationClip, GizmoDragEvent } from "reze-engine"
import { useStudioActions, useStudioSelector } from "@/context/studio-context"
import { usePlayback, usePlaybackFrameRef } from "@/context/playback-context"
import { useStudioStatusActions } from "@/components/studio-status"
import { autoClassifyMaterials } from "@/lib/materials"
import { interpolationTemplateForFrame, readLocalPoseAfterSeek } from "@/lib/utils"

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
  /** Current engine model key — "reze" at boot, replaced on PMX folder upload.
   *  EngineBridge needs this to push selectedBone / selectedMaterial to the
   *  right model (the engine keys selection state per model name). */
  loadedModelNameRef: RefObject<string>
  /** Parent's imperative "scroll the bone list to this bone" hook. Called on
   *  raycast hit so a bone picked in the viewport auto-centers in the list. */
  revealBoneInListRef: RefObject<((bone: string) => void) | null>
  currentFrameRef: RefObject<number>
  playheadDrawRef: RefObject<((frame: number) => void) | null>
  documentDirtyRef: RefObject<boolean>
  suppressClipDirtyRef: RefObject<number>
  setPmxBoneNames: Dispatch<SetStateAction<ReadonlySet<string>>>
  setModelBoneOrder: Dispatch<SetStateAction<string[]>>
  setMorphNames: Dispatch<SetStateAction<string[]>>
  setMaterialNames: Dispatch<SetStateAction<string[]>>
  setEngineError: Dispatch<SetStateAction<string | null>>
  setStudioReady: Dispatch<SetStateAction<boolean>>
}

export function EngineBridge({
  canvasRef,
  engineRef,
  modelRef,
  loadedModelNameRef,
  revealBoneInListRef,
  currentFrameRef,
  playheadDrawRef,
  documentDirtyRef,
  suppressClipDirtyRef,
  setPmxBoneNames,
  setModelBoneOrder,
  setMorphNames,
  setMaterialNames,
  setEngineError,
  setStudioReady,
}: EngineBridgeProps) {
  const clip = useStudioSelector((s) => s.clip)
  const selectedBone = useStudioSelector((s) => s.selectedBone)
  const selectedMaterial = useStudioSelector((s) => s.selectedMaterial)
  const gizmoVisible = useStudioSelector((s) => s.gizmoVisible)
  const {
    commit,
    replaceClip,
    setClipDisplayName,
    setSelectedBone,
    setSelectedMorph,
    setSelectedMaterial,
    setGizmoVisible,
    setSelectedKeyframes,
  } = useStudioActions()
  const { currentFrame, setCurrentFrame, playing, setPlaying } = usePlayback()
  const playbackFrameRef = usePlaybackFrameRef()
  const { setPmxFileName: setStatusPmxFileName, setFps: setStatusFps } = useStudioStatusActions()
  const frameCount = clip?.frameCount ?? 0

  // ─── Refs for the engine-supplied callbacks ──────────────────────────
  //     The Engine constructor takes `onRaycast` / `onGizmoDrag` once at
  //     startup and there's no setter — so we hand it stable thunks that
  //     read the latest handler from a ref. The handlers themselves close
  //     over refs (clipRef, playbackFrameRef, modelRef) so they always see
  //     current values without needing re-registration.
  const clipRef = useRef<AnimationClip | null>(clip)
  useEffect(() => {
    clipRef.current = clip
  }, [clip])
  const dragDirtyRef = useRef(false)

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

  // ─── Viewport raycast (dblclick on model) ───────────────────────────
  //     Engine resolves bone + material for the hit triangle; studio only
  //     consumes the bone (material lives in the panel, per UX rule:
  //     "material picks happen in the material list, not the viewport").
  //     Null modelName means the click missed the mesh — deselect.
  const handleRaycast = useCallback(
    (modelName: string, _material: string | null, bone: string | null, _screenX: number, _screenY: number) => {
      if (!modelName) {
        // Miss (dblclick on empty space) → hide the gizmo in the viewport
        // without touching studio selection. Bone-list highlight, Properties
        // inspector, and timeline state all stay intact — the flag is the
        // only thing that changes, and re-selecting the bone brings it back.
        setGizmoVisible(false)
        return
      }
      // Hit → select the bone. Mirrors `handleSelectBone` in studio.tsx so the
      // mutual-exclusion contract holds whether picks come from viewport or
      // from the bone list.
      setSelectedBone(bone)
      setSelectedMorph(null)
      setSelectedMaterial(null)
      setSelectedKeyframes([])
      // Scroll the bone list so the pick lands in view. Only for raycasts —
      // bone-list clicks don't need this (the row is already where the user
      // pointed).
      if (bone) revealBoneInListRef.current?.(bone)
    },
    [setSelectedBone, setSelectedMorph, setSelectedMaterial, setGizmoVisible, setSelectedKeyframes, revealBoneInListRef],
  )

  // ─── Gizmo drag → keyframe edit (undoable) ──────────────────────────
  //     Mirrors the preview/commit pattern in properties-inspector.tsx:
  //     mutate the keyframe in place during drag moves (no React churn),
  //     commit a new clip ref on drag end so history records one entry
  //     per gesture. `dragDirtyRef` suppresses the commit for no-op drags
  //     (gizmo click without movement).
  const handleGizmoDrag = useCallback(
    (e: GizmoDragEvent) => {
      const model = modelRef.current
      const clip = clipRef.current
      if (!model || !clip) return

      if (e.phase === "start") {
        dragDirtyRef.current = false
        // Sidebar + Properties should track whatever the user is dragging.
        // Same mutual-exclusion contract as a raycast pick.
        setSelectedBone(e.boneName)
        setSelectedMorph(null)
        setSelectedMaterial(null)
        setSelectedKeyframes([])
        return
      }

      const frame = Math.round(Math.max(0, Math.min(clip.frameCount, playbackFrameRef.current)))
      const bone = e.boneName
      const track = clip.boneTracks.get(bone) ?? []
      const atKey = track.find((k) => k.frame === frame)

      if (atKey) {
        if (e.kind === "rotate") atKey.rotation = e.localRotation
        else atKey.translation = e.localTranslation
      } else {
        // No key at this frame yet — pull the untouched channel from the
        // interpolated pose so the new key preserves whatever's currently
        // displayed on the channel the user isn't dragging.
        model.loadClip(STUDIO_ANIM_NAME, clip)
        model.seek(frame / 30)
        const pose = readLocalPoseAfterSeek(model, bone)
        if (!pose) return
        const rotation = e.kind === "rotate" ? e.localRotation : pose.rotation
        const translation = e.kind === "translate" ? e.localTranslation : pose.translation
        if (!clip.boneTracks.has(bone)) clip.boneTracks.set(bone, track)
        track.push({
          boneName: bone,
          frame,
          rotation,
          translation,
          interpolation: interpolationTemplateForFrame(track, frame),
        })
        track.sort((a, b) => a.frame - b.frame)
      }

      model.loadClip(STUDIO_ANIM_NAME, clip)
      model.seek(frame / 30)

      if (e.phase === "end") {
        if (dragDirtyRef.current) {
          commit({ ...clip, boneTracks: new Map(clip.boneTracks) })
        }
        dragDirtyRef.current = false
      } else {
        dragDirtyRef.current = true
      }
    },
    [commit, modelRef, playbackFrameRef, setSelectedBone, setSelectedMorph, setSelectedMaterial, setSelectedKeyframes],
  )

  // Stable thunks that read the latest handlers via ref — re-registration
  // would require recreating the Engine.
  const handleRaycastRef = useRef(handleRaycast)
  const handleGizmoDragRef = useRef(handleGizmoDrag)
  useEffect(() => {
    handleRaycastRef.current = handleRaycast
  }, [handleRaycast])
  useEffect(() => {
    handleGizmoDragRef.current = handleGizmoDrag
  }, [handleGizmoDrag])

  // ─── Mirror React selection → engine gizmo/outline ──────────────────
  //     The engine keys selection per model name, so every write uses the
  //     live `loadedModelNameRef.current` (swaps on PMX folder upload).
  useEffect(() => {
    const engine = engineRef.current
    if (!engine) return
    engine.setSelectedBone(loadedModelNameRef.current, gizmoVisible ? selectedBone : null)
  }, [selectedBone, gizmoVisible, engineRef, loadedModelNameRef])

  useEffect(() => {
    const engine = engineRef.current
    if (!engine) return
    engine.setSelectedMaterial(loadedModelNameRef.current, selectedMaterial)
  }, [selectedMaterial, engineRef, loadedModelNameRef])

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
          bloom: { color: new Vec3(1, 0.1, 0.88) },
          onRaycast: (modelName, material, bone, screenX, screenY) =>
            handleRaycastRef.current(modelName, material, bone, screenX, screenY),
          onGizmoDrag: (event) => handleGizmoDragRef.current(event),
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
          const materialNames = model.getMaterials().map((m) => m.name)
          setMaterialNames(materialNames)
          setStatusPmxFileName(BUNDLED_PMX_FILENAME)
          model.setMorphWeight("抗穿模", 0.5)

          // Keep boot deterministic: classify before the render loop starts so
          // the first frame uses the correct NPR buckets instead of falling
          // through to the default Principled BSDF path. StudioPage's materials
          // effect will mirror the same map into React state (idempotent).
          engine.setMaterialPresets("reze", autoClassifyMaterials(materialNames))

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
            replaceClip(c)
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
      setMaterialNames([])
      setSelectedBone(null)
      setSelectedMorph(null)
      setSelectedMaterial(null)
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
        // Natural end isn't a jump — physics integrated continuously through
        // the last frame. Sync seek tracking BEFORE setState so the scrub
        // useLayoutEffect (which runs on the resulting commit, ahead of this
        // effect's cleanup) sees delta=0 and skips the physics reset.
        currentFrameRef.current = frameCount
        lastSeekFrameRef.current = frameCount
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
      // what the playhead was last showing. Sync `lastSeekFrameRef` first so
      // the scrub effect this flush triggers sees delta=0 and skips the
      // physics reset — pausing from playback isn't a jump, physics is
      // already in a valid state.
      lastSeekFrameRef.current = currentFrameRef.current
      setCurrentFrame(currentFrameRef.current)
    }
  }, [playing, frameCount, setCurrentFrame, setPlaying, currentFrameRef, modelRef, playheadDrawRef])

  return null
}
