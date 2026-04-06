"use client"

import {
  useEffect,
  useRef,
  useState,
  useMemo,
  useCallback,
  type ChangeEvent,
  type SetStateAction,
} from "react"
import { Engine, Model, Vec3, parsePmxFolderInput, pmxFileAtRelativePath } from "reze-engine"
import { Button } from "@/components/ui/button"
import { EditorLeftPanel, EditorStatusFooter, EditorViewport } from "@/components/editor-chrome"
import { PropertiesInspector } from "@/components/properties-inspector"
import { Timeline, type SelectedKeyframe } from "@/components/timeline"
import { BONE_GROUPS, quatToEuler } from "@/lib/animation"
import { interpolationTemplateForFrame, readLocalPoseAfterSeek } from "@/lib/keyframe-insert"
import type { AnimationClip, BoneKeyframe, MorphKeyframe } from "reze-engine"
import {
  saveMeta,
  loadMeta,
  saveClip as idbSaveClip,
  loadClip as idbLoadClip,
  clearClip as idbClearClip,
} from "@/lib/editor-persist"
import { clipAfterKeyframeEdit, DEFAULT_STUDIO_CLIP_FRAMES } from "@/lib/clip-duration"
import packageJson from "../package.json"

const MODEL_PATH = "/models/reze/reze.pmx"
const APP_VERSION = packageJson.version
const REPO_URL = "https://github.com/AmyangXYZ/reze-studio"
const DOCS_README_URL = `${REPO_URL}/blob/main/README.md`
const VMD_PATH = "/animations/miku.vmd"
const STUDIO_ANIM_NAME = "studio"
/** Autosave cadence — VMD export + IDB are heavy; keep off the hot path vs 5s. */
const PERSIST_INTERVAL_MS = 20_000
/** Ensures idle clip backup runs even under load (still after meta, which is cheap). */
const IDLE_CLIP_TIMEOUT_MS = 10_000
/** Basename for status bar when using bundled `MODEL_PATH` PMX. */
const BUNDLED_PMX_FILENAME = MODEL_PATH.replace(/^.*\//, "") || "model.pmx"

function emptyStudioClip(): AnimationClip {
  return { boneTracks: new Map(), morphTracks: new Map(), frameCount: DEFAULT_STUDIO_CLIP_FRAMES }
}

/** Keep only tracks whose bones/morphs exist on the new model. */
function clipRetainedForModel(
  clip: AnimationClip,
  boneNames: ReadonlySet<string>,
  morphNames: ReadonlySet<string>,
): AnimationClip {
  const boneTracks = new Map<string, BoneKeyframe[]>()
  for (const [name, track] of clip.boneTracks) {
    if (!boneNames.has(name) || !track?.length) continue
    boneTracks.set(name, track.map((kf) => ({ ...kf })))
  }
  const morphTracks = new Map<string, MorphKeyframe[]>()
  for (const [name, track] of clip.morphTracks) {
    if (!morphNames.has(name) || !track?.length) continue
    morphTracks.set(name, track.map((kf) => ({ ...kf })))
  }
  let inferred = 0
  for (const t of boneTracks.values()) for (const k of t) inferred = Math.max(inferred, k.frame)
  for (const t of morphTracks.values()) for (const k of t) inferred = Math.max(inferred, k.frame)
  const empty = boneTracks.size === 0 && morphTracks.size === 0
  const end = empty
    ? Math.max(clip.frameCount, DEFAULT_STUDIO_CLIP_FRAMES)
    : Math.max(clip.frameCount, inferred)
  return { boneTracks, morphTracks, frameCount: end }
}

function downloadBlob(blob: Blob, filename: string) {
  const a = document.createElement("a")
  const url = URL.createObjectURL(blob)
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

/** Stem of a path or file name (`/animations/miku.vmd` → `miku`). */
function fileStem(pathOrName: string): string {
  const base = pathOrName.replace(/^.*[/\\]/, "")
  const i = base.lastIndexOf(".")
  return (i > 0 ? base.slice(0, i) : base).trim() || "clip"
}

/** One safe path segment for downloads (no slashes or reserved characters). */
function sanitizeClipFilenameBase(name: string): string {
  const s = name.trim() || "clip"
  const cleaned = s.replace(/[/\\<>:"|?*\x00-\x1f]/g, "-").replace(/-+/g, "-")
  return cleaned.slice(0, 120).replace(/^-|-$/g, "") || "clip"
}

/** Reuse `livePose` object when floats haven’t moved — keeps memo’d Properties from reconciling every RAF. */
function poseNearEqual(
  a: { euler: { x: number; y: number; z: number }; translation: Vec3 },
  b: typeof a,
  eps = 1e-5,
) {
  return (
    Math.abs(a.euler.x - b.euler.x) < eps &&
    Math.abs(a.euler.y - b.euler.y) < eps &&
    Math.abs(a.euler.z - b.euler.z) < eps &&
    Math.abs(a.translation.x - b.translation.x) < eps &&
    Math.abs(a.translation.y - b.translation.y) < eps &&
    Math.abs(a.translation.z - b.translation.z) < eps
  )
}

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const engineRef = useRef<Engine | null>(null)
  const modelRef = useRef<Model | null>(null)
  const [engineError, setEngineError] = useState<string | null>(null)

  // ─── Persisted meta (deferred to useEffect to avoid SSR hydration mismatch) ──
  const persistedMeta = useRef<ReturnType<typeof loadMeta> | null>(null)

  // ─── Clip synced with engine via loadClip(STUDIO_ANIM_NAME) / getClip ──
  const [clip, setClipState] = useState<AnimationClip | null>(null)
  /** Normalize duration after key edits / loads so end ≥ last key and transport never stays at 0. */
  const setClip = useCallback((action: SetStateAction<AnimationClip | null>) => {
    setClipState((prev) => {
      const next = typeof action === "function" ? action(prev) : action
      if (next == null) return null
      return clipAfterKeyframeEdit(next)
    })
  }, [])
  /** Model finished loading (file menu + export need a live Model instance). */
  const [studioReady, setStudioReady] = useState(false)
  /** User-facing clip label for default save names (`{clipDisplayName}-export.vmd` / `.json`). */
  const [clipDisplayName, setClipDisplayName] = useState("clip")

  const vmdInputRef = useRef<HTMLInputElement>(null)
  const pmxFolderInputRef = useRef<HTMLInputElement>(null)
  /** Matches `engine.loadModel` name so `removeModel` can swap uploads without patching the engine. */
  const loadedModelNameRef = useRef("reze")
  /** Folder files from the last pick — kept for multi-PMX selection flow. */
  const pmxFolderFilesRef = useRef<File[] | null>(null)
  const frameCount = clip?.frameCount ?? 0
  /** PMX skeleton bone names; used to hide VMD tracks that do not exist on the loaded model. */
  const [pmxBoneNames, setPmxBoneNames] = useState<ReadonlySet<string>>(new Set())
  /** PMX bone order (skeleton array) — remainder list after clip bones in the sidebar. */
  const [modelBoneOrder, setModelBoneOrder] = useState<string[]>([])
  /** From `model.getMorphing().morphs` (engine has no `getMorphs()` alias yet). */
  const [morphNames, setMorphNames] = useState<string[]>([])
  const [activeMorph, setActiveMorph] = useState<string | null>(null)
  const [morphWeightReadout, setMorphWeightReadout] = useState<number | null>(null)

  /** Bones with tracks in the current clip (and on the model) — timeline rows + keying. */
  const clipBones = useMemo(() => {
    if (!clip) return []
    const keys = Array.from(clip.boneTracks.keys())
    if (pmxBoneNames.size === 0) return keys
    return keys.filter((k) => pmxBoneNames.has(k))
  }, [clip, pmxBoneNames])

  /** Sidebar list: strict PMX skeleton order (same for new clips and edits). */
  const sidebarBones = modelBoneOrder

  const [currentFrame, setCurrentFrame] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [activeBone, setActiveBone] = useState<string | null>(null)
  const [selectedGroup, setSelectedGroup] = useState("All Bones")
  const [selectedKeyframes, setSelectedKeyframes] = useState<SelectedKeyframe[]>([])
  /** Bumped on new clip load / reset so Timeline can reset its local view state. */
  const [clipVersion, setClipVersion] = useState(0)
  /** Lifted from Timeline so PropertiesInspector sliders + keyframe selection can sync it. */
  const [timelineTab, setTimelineTab] = useState("allRot")

  /** Folder upload contained multiple `.pmx`; user picks one then clicks Load. */
  const [pmxPickFiles, setPmxPickFiles] = useState<File[] | null>(null)
  const [pmxPickPaths, setPmxPickPaths] = useState<string[]>([])
  const [pmxPickSelected, setPmxPickSelected] = useState("")
  /** Radix menubar: which submenu is open (`""` = all closed). */
  const [menubarValue, setMenubarValue] = useState("")
  /** Bundled or uploaded `.pmx` file name for the status bar. */
  const [statusPmxFileName, setStatusPmxFileName] = useState("—")
  /** VS Code–style transient line (save feedback, errors, hints) — set from chrome later. */
  const [statusMessage, setStatusMessage] = useState("")
  /** Render FPS from `Engine.getStats()` (updated inside the engine render path, ~1s window). */
  const [statusFps, setStatusFps] = useState<number | null>(null)
  const lastReportedEngineFpsRef = useRef<number | null>(null)

  const playRef = useRef(false)
  const lastT = useRef<number | null>(null)
  /** Snapshotted before async PMX swap so clip/playhead survive `await loadModel`. */
  const clipRef = useRef<AnimationClip | null>(null)
  const currentFrameRef = useRef(0)
  const clipDisplayNameRef = useRef("clip")
  const livePoseStableRef = useRef<{
    euler: { x: number; y: number; z: number }
    translation: Vec3
  } | null>(null)

  const visibleBones = useMemo(() => {
    const g = BONE_GROUPS[selectedGroup]
    if (!g) return clipBones
    return g.filter((name) => clipBones.includes(name))
  }, [selectedGroup, clipBones])

  useEffect(() => {
    clipRef.current = clip
  }, [clip])
  useEffect(() => {
    currentFrameRef.current = currentFrame
  }, [currentFrame])
  useEffect(() => {
    clipDisplayNameRef.current = clipDisplayName
  }, [clipDisplayName])

  // ─── Persist editor state (interval + beforeunload) ──────────────────
  /** Last clip reference that was written to IndexedDB — skip re-serializing the same object. */
  const lastSavedClipRef = useRef<AnimationClip | null>(null)
  /** Coalesce deferred clip writes so rapid timers don’t stack export work. */
  const idleClipHandleRef = useRef<ReturnType<typeof requestIdleCallback> | number | null>(null)

  const cancelScheduledClipPersist = useCallback(() => {
    const h = idleClipHandleRef.current
    if (h == null) return
    if (typeof cancelIdleCallback !== "undefined") cancelIdleCallback(h as number)
    else clearTimeout(h as number)
    idleClipHandleRef.current = null
  }, [])

  /** Refs only — safe inside requestIdleCallback (latest clip vs stale closure). */
  const persistClipToIdbSync = useCallback(() => {
    const c = clipRef.current
    if (playRef.current || c === lastSavedClipRef.current) return
    const model = modelRef.current
    if (c && model) {
      try {
        model.loadClip(STUDIO_ANIM_NAME, c)
        const buf = model.exportVmd(STUDIO_ANIM_NAME)
        void idbSaveClip(buf)
      } catch { /* export can fail on empty clips — ignore */ }
    } else {
      void idbClearClip()
    }
    lastSavedClipRef.current = c
  }, [])

  const persistState = useCallback(
    (opts?: { syncClip?: boolean }) => {
      saveMeta({
        activeBone,
        activeMorph,
        selectedGroup,
        currentFrame,
        clipDisplayName,
        hasClip: clip != null,
      })
      if (opts?.syncClip) {
        cancelScheduledClipPersist()
        persistClipToIdbSync()
        return
      }
      cancelScheduledClipPersist()
      if (typeof requestIdleCallback !== "undefined") {
        idleClipHandleRef.current = requestIdleCallback(
          () => {
            idleClipHandleRef.current = null
            persistClipToIdbSync()
          },
          { timeout: IDLE_CLIP_TIMEOUT_MS },
        )
      } else {
        idleClipHandleRef.current = window.setTimeout(() => {
          idleClipHandleRef.current = null
          persistClipToIdbSync()
        }, 0)
      }
    },
    [
      activeBone,
      activeMorph,
      selectedGroup,
      currentFrame,
      clipDisplayName,
      clip,
      cancelScheduledClipPersist,
      persistClipToIdbSync,
    ],
  )

  const persistRef = useRef(persistState)
  useEffect(() => {
    persistRef.current = persistState
  }, [persistState])

  useEffect(() => {
    const iv = setInterval(() => persistRef.current(), PERSIST_INTERVAL_MS)
    const onUnload = () => persistRef.current({ syncClip: true })
    window.addEventListener("beforeunload", onUnload)
    return () => {
      clearInterval(iv)
      window.removeEventListener("beforeunload", onUnload)
      persistRef.current({ syncClip: true })
    }
  }, [])

  // ─── Playback loop ───────────────────────────────────────────────────
  useEffect(() => {
    playRef.current = playing
    if (!playing) {
      lastT.current = null
      return
    }
    let raf: number
    const tick = (ts: number) => {
      if (!playRef.current) return
      if (lastT.current !== null)
        setCurrentFrame((p) => {
          const n = p + ((ts - (lastT.current ?? ts)) / 1000) * 30
          if (n >= frameCount) {
            setPlaying(false)
            return frameCount
          }
          return n
        })
      lastT.current = ts
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [playing, frameCount])

  // ─── Keyboard shortcuts ──────────────────────────────────────────────
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code === "Space") {
        e.preventDefault()
        setPlaying((p) => !p)
      }
      if (e.code === "ArrowLeft") setCurrentFrame((p) => Math.max(0, Math.round(p) - 1))
      if (e.code === "ArrowRight") setCurrentFrame((p) => Math.min(frameCount, Math.round(p) + 1))
      if (e.code === "Home") setCurrentFrame(0)
      if (e.code === "End") setCurrentFrame(frameCount)
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [frameCount])

  // ─── Bone selection handlers ─────────────────────────────────────────
  const handleSelectGroup = useCallback((g: string) => {
    setSelectedGroup((prev) => (prev === g ? "" : g))
    setActiveBone(null)
    setActiveMorph(null)
    setSelectedKeyframes([])
  }, [])

  const handleSelectBone = useCallback((b: string) => {
    setActiveMorph(null)
    setActiveBone(b)
    setSelectedKeyframes([])
  }, [])

  const handleSelectMorph = useCallback((name: string) => {
    setActiveBone(null)
    setActiveMorph(name)
    setSelectedKeyframes([])
  }, [])

  useEffect(() => {
    if (activeBone && !pmxBoneNames.has(activeBone)) setActiveBone(null)
  }, [activeBone, pmxBoneNames])

  useEffect(() => {
    if (activeMorph && !morphNames.includes(activeMorph)) setActiveMorph(null)
  }, [activeMorph, morphNames])

  useEffect(() => {
    setSelectedKeyframes((prev) =>
      prev.filter((s) => s.type !== "curve" || !s.bone || pmxBoneNames.has(s.bone)),
    )
  }, [pmxBoneNames])

  // ─── Engine init ─────────────────────────────────────────────────────
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
          engine.addGround({
            diffuseColor: new Vec3(0.14, 0.12, 0.16),
          })
        } catch {
          setEngineError(`Add model at public${MODEL_PATH}`)
        }

        lastReportedEngineFpsRef.current = null
        engine.runRenderLoop(() => {
          const fps = engine.getStats().fps
          if (fps === lastReportedEngineFpsRef.current) return
          lastReportedEngineFpsRef.current = fps
          setStatusFps(fps > 0 ? fps : null)
        })

        // Hydrate persisted meta (client-only — safe from SSR mismatch)
        const meta = loadMeta()
        persistedMeta.current = meta

        // Restore persisted clip from IndexedDB, or fall back to default VMD
        let restored = false
        if (meta.hasClip && modelRef.current) {
          try {
            const buf = await idbLoadClip()
            if (buf && !disposed) {
              const blob = new Blob([buf], { type: "application/octet-stream" })
              const url = URL.createObjectURL(blob)
              try {
                await modelRef.current.loadVmd(STUDIO_ANIM_NAME, url)
                const c = modelRef.current.getClip(STUDIO_ANIM_NAME)
                if (c) {
                  setClip(c)
                  setClipDisplayName(meta.clipDisplayName)
                  setCurrentFrame(meta.currentFrame)
                  setActiveBone(meta.activeBone)
                  setActiveMorph(meta.activeMorph)
                  setSelectedGroup(meta.selectedGroup)
                  modelRef.current.show(STUDIO_ANIM_NAME)
                  modelRef.current.seek(Math.max(0, meta.currentFrame) / 30)
                  if (modelRef.current.name === "reze") modelRef.current.setMorphWeight("抗穿模", 0.5)
                  restored = true
                }
              } finally {
                URL.revokeObjectURL(url)
              }
            }
          } catch (e) {
            console.warn("Failed to restore persisted clip:", e)
          }
        }
        if (!restored) {
          try {
            await modelRef.current?.loadVmd(STUDIO_ANIM_NAME, VMD_PATH)
            if (disposed) return
            const c = modelRef.current?.getClip(STUDIO_ANIM_NAME)
            if (c) {
              setClip(c)
              setClipDisplayName(sanitizeClipFilenameBase(fileStem(VMD_PATH)))
              modelRef.current?.show(STUDIO_ANIM_NAME)
              modelRef.current?.seek(0)
              if (modelRef.current?.name === "reze") modelRef.current?.setMorphWeight("抗穿模", 0.5)
            }
          } catch (e) {
            console.warn(`VMD load failed — add file at public${VMD_PATH}`, e)
          }
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
      setActiveMorph(null)
      setMorphWeightReadout(null)
      setStatusPmxFileName("—")
      setStatusFps(null)
      lastReportedEngineFpsRef.current = null
      modelRef.current = null
      engineRef.current?.stopRenderLoop()
      engineRef.current?.dispose()
      engineRef.current = null
    }
  }, [])

  // Keep model pose locked to timeline frame; refresh morph weight readout when a morph is selected.
  useEffect(() => {
    const model = modelRef.current
    if (!model || !clip) return
    model.loadClip(STUDIO_ANIM_NAME, clip)
    model.seek(Math.max(0, currentFrame) / 30)
    if (!activeMorph) {
      setMorphWeightReadout(null)
      return
    }
    const morphing = model.getMorphing()
    const idx = morphing.morphs.findIndex((m) => m.name === activeMorph)
    if (idx < 0) {
      setMorphWeightReadout(null)
      return
    }
    const w = model.getMorphWeights()[idx]
    setMorphWeightReadout((prev) => (prev === w ? prev : w))
  }, [currentFrame, clip, activeMorph])

  useEffect(() => {
    const model = modelRef.current
    if (!model || !clip) return
    if (playing) {
      model.play()
      if (model.name === "reze") {
        model.setMorphWeight("抗穿模", 0.5)
      }
    }
    else model.pause()
  }, [playing, clip])

  useEffect(() => {
    if (!playing || frameCount <= 0) return
    if (currentFrame >= frameCount) setCurrentFrame(0)
  }, [playing, currentFrame, frameCount])

  useEffect(() => {
    setCurrentFrame((c) => Math.min(c, frameCount))
  }, [frameCount])

  // Timeline key click: jump playhead; curve keys also focus the bone on the list
  // and auto-switch the timeline channel tab to match the selected channel.
  useEffect(() => {
    if (selectedKeyframes.length !== 1) return
    const s = selectedKeyframes[0]
    setActiveMorph(null)
    if (s.type === "curve" && s.bone) setActiveBone(s.bone)
    setCurrentFrame(s.frame)
    if (s.channel && ["rx", "ry", "rz", "tx", "ty", "tz"].includes(s.channel)) {
      setTimelineTab(s.channel)
    }
  }, [selectedKeyframes])

  const deleteSelectedKeyframes = useCallback(() => {
    if (!clip || selectedKeyframes.length !== 1) return
    const sel = selectedKeyframes[0]
    if (sel.type === "curve" && sel.bone) {
      const track = clip.boneTracks.get(sel.bone)
      if (!track) return
      const i = track.findIndex((k) => k.frame === sel.frame)
      if (i < 0) return
      track.splice(i, 1)
      if (track.length === 0) clip.boneTracks.delete(sel.bone)
    } else if (sel.type === "dope") {
      const f = sel.frame
      const dropBones: string[] = []
      for (const [name, track] of clip.boneTracks.entries()) {
        const i = track.findIndex((k) => k.frame === f)
        if (i >= 0) {
          track.splice(i, 1)
          if (track.length === 0) dropBones.push(name)
        }
      }
      for (const name of dropBones) clip.boneTracks.delete(name)
    } else return

    setSelectedKeyframes([])
    setClip({ ...clip, boneTracks: new Map(clip.boneTracks) })
  }, [clip, selectedKeyframes])

  const livePose = useMemo(() => {
    const model = modelRef.current
    if (!model || !activeBone || !clip) {
      livePoseStableRef.current = null
      return null
    }
    // React clip can fork from the engine’s internal clip; push state back before seek/read so sliders stay in sync
    model.loadClip(STUDIO_ANIM_NAME, clip)
    model.seek(Math.max(0, currentFrame) / 30)
    const p = readLocalPoseAfterSeek(model, activeBone)
    if (!p) {
      livePoseStableRef.current = null
      return null
    }
    const next = {
      euler: quatToEuler(p.rotation),
      translation: p.translation,
    }
    const prev = livePoseStableRef.current
    if (prev && poseNearEqual(prev, next)) return prev
    livePoseStableRef.current = next
    return next
  }, [currentFrame, clip, activeBone])

  const insertKeyframeAtPlayhead = useCallback(() => {
    const model = modelRef.current
    if (!clip || !activeBone || activeMorph || !model) return
    const frame = Math.round(Math.max(0, currentFrame))
    model.loadClip(STUDIO_ANIM_NAME, clip)
    model.seek(Math.max(0, currentFrame) / 30)
    const pose = readLocalPoseAfterSeek(model, activeBone)
    if (!pose) return

    const prevTrack = clip.boneTracks.get(activeBone)
    const ip = interpolationTemplateForFrame(prevTrack, frame)
    const nextTrack = [...(prevTrack ?? [])].filter((k) => k.frame !== frame)
    nextTrack.push({
      boneName: activeBone,
      frame,
      rotation: pose.rotation,
      translation: pose.translation,
      interpolation: ip,
    })
    nextTrack.sort((a, b) => a.frame - b.frame)
    const boneTracks = new Map(clip.boneTracks)
    boneTracks.set(activeBone, nextTrack)
    setClip({ ...clip, boneTracks })
    // Focus properties (sliders + curves) like selecting a key on the graph
    setSelectedKeyframes([{ type: "curve", bone: activeBone, frame, channel: "rx" }])
  }, [clip, activeBone, activeMorph, currentFrame])

  const syncStudioAfterNewClip = useCallback((model: Model) => {
    setCurrentFrame(0)
    setPlaying(false)
    setSelectedKeyframes([])
    setTimelineTab("allRot")
    setClipVersion((v) => v + 1)
    model.show(STUDIO_ANIM_NAME)
    model.seek(0)
    if (model.name === "reze") model.setMorphWeight("抗穿模", 0.5)
  }, [])

  const applyLoadedPmxModel = useCallback(
    (
      model: Model,
      engineInstanceKey: string,
      displayStem: string,
      pmxFileName: string,
      animationSnapshot: {
        clip: AnimationClip | null
        currentFrame: number
        playing: boolean
        clipDisplayName: string
      },
    ) => {
      modelRef.current = model
      loadedModelNameRef.current = engineInstanceKey
      const sk = model.getSkeleton().bones.map((b) => b.name)
      const boneSet = new Set(sk)
      const morphNamesList = model.getMorphing().morphs.map((m) => m.name)
      const morphSet = new Set(morphNamesList)
      setPmxBoneNames(boneSet)
      setModelBoneOrder(sk)
      setMorphNames(morphNamesList)
      setStatusPmxFileName(pmxFileName.trim() || `${displayStem}.pmx`)
      setActiveBone((prev) => (prev && boneSet.has(prev) ? prev : null))
      setActiveMorph((prev) => (prev && morphSet.has(prev) ? prev : null))
      setSelectedKeyframes((prev) =>
        prev.filter((s) => s.type !== "curve" || !s.bone || boneSet.has(s.bone)),
      )

      const prev = animationSnapshot.clip
      const hasPrevTimeline =
        prev != null &&
        (prev.boneTracks.size > 0 || prev.morphTracks.size > 0 || prev.frameCount > 0)

      let nextClip: AnimationClip
      let nextDisplay: string
      let nextFrame: number
      let nextPlaying: boolean

      if (hasPrevTimeline) {
        nextClip = clipRetainedForModel(prev, boneSet, morphSet)
        nextDisplay = animationSnapshot.clipDisplayName
        nextFrame = Math.min(
          Math.max(0, animationSnapshot.currentFrame),
          Math.max(0, nextClip.frameCount),
        )
        nextPlaying = animationSnapshot.playing
      } else {
        nextClip = emptyStudioClip()
        nextDisplay = sanitizeClipFilenameBase(displayStem)
        nextFrame = 0
        nextPlaying = false
      }

      model.loadClip(STUDIO_ANIM_NAME, nextClip)
      setClip(nextClip)
      setClipDisplayName(nextDisplay)
      setCurrentFrame(nextFrame)
      setPlaying(nextPlaying)
      model.show(STUDIO_ANIM_NAME)
      model.seek(nextFrame / 30)
      if (nextPlaying) model.play()
      else model.pause()
      setEngineError(null)
    },
    [],
  )

  const loadPmxFromFolder = useCallback(
    async (files: File[], pmxFile: File) => {
      const engine = engineRef.current
      if (!engine) {
        window.alert("Viewport is not ready yet. Wait for the model to load, then try again.")
        return
      }
      const stem = fileStem(pmxFile.name)
      const instanceKey = `u_${crypto.randomUUID().replace(/-/g, "").slice(0, 16)}`
      try {
        engine.removeModel(loadedModelNameRef.current)
      } catch {
        /* removeModel is a no-op if the name is stale */
      }
      try {
        const model = await engine.loadModel(instanceKey, { files, pmxFile })
        await new Promise(resolve => requestAnimationFrame(resolve))
        model.setName(sanitizeClipFilenameBase(stem))
        applyLoadedPmxModel(model, instanceKey, stem, pmxFile.name, {
          clip: clipRef.current,
          currentFrame: currentFrameRef.current,
          playing: playRef.current,
          clipDisplayName: clipDisplayNameRef.current,
        })
      } catch (e) {
        console.error("[pmx-upload] loadModel failed:", e)
        window.alert(e instanceof Error ? e.message : String(e))
      }
    },
    [applyLoadedPmxModel],
  )

  const onPickPmxFolder = useCallback(
    async (e: ChangeEvent<HTMLInputElement>) => {
      try {
        const picked = parsePmxFolderInput(e.target.files)
        e.target.value = ""

        if (picked.status === "empty") return
        if (picked.status === "not_directory") {
          window.alert("Please select a folder, not individual files.")
          return
        }
        if (picked.status === "no_pmx") {
          window.alert("No .pmx file in the selected folder.")
          return
        }

        setPmxPickFiles(null)
        setPmxPickPaths([])
        setPmxPickSelected("")

        if (picked.status === "single") {
          await loadPmxFromFolder(picked.files, picked.pmxFile)
        } else {
          pmxFolderFilesRef.current = picked.files
          setPmxPickFiles(picked.files)
          setPmxPickPaths(picked.pmxRelativePaths)
          setPmxPickSelected(picked.pmxRelativePaths[0] ?? "")
        }
      } finally {
        setMenubarValue("")
      }
    },
    [loadPmxFromFolder],
  )

  const onConfirmPmxPick = useCallback(async () => {
    const files = pmxPickFiles
    const path = pmxPickSelected
    if (!files || !path) return
    const pmxFile = pmxFileAtRelativePath(files, path)
    if (!pmxFile) {
      window.alert("Could not find the selected PMX file.")
      return
    }
    await loadPmxFromFolder(files, pmxFile)
    setPmxPickFiles(null)
    setPmxPickPaths([])
    setPmxPickSelected("")
  }, [loadPmxFromFolder, pmxPickFiles, pmxPickSelected])

  const onPickVmdFile = useCallback(
    async (e: ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      e.target.value = ""
      const model = modelRef.current
      if (!file || !model) return
      const url = URL.createObjectURL(file)
      try {
        await model.loadVmd(STUDIO_ANIM_NAME, url)
        const c = model.getClip(STUDIO_ANIM_NAME)
        if (c) {
          setClip(c)
          setClipDisplayName(sanitizeClipFilenameBase(fileStem(file.name)))
          syncStudioAfterNewClip(model)
        }
      } catch (err) {
        window.alert(err instanceof Error ? err.message : String(err))
      } finally {
        URL.revokeObjectURL(url)
      }
    },
    [syncStudioAfterNewClip],
  )

  const exportClipVmd = useCallback(() => {
    const model = modelRef.current
    if (!model || !clip) return
    const base = sanitizeClipFilenameBase(clipDisplayName)
    try {
      model.loadClip(STUDIO_ANIM_NAME, clip)
      const buf = model.exportVmd(STUDIO_ANIM_NAME)
      downloadBlob(new Blob([buf], { type: "application/octet-stream" }), `${base}-export.vmd`)
    } catch (err) {
      window.alert(err instanceof Error ? err.message : String(err))
    }
  }, [clip, clipDisplayName])

  const resetEditorState = useCallback(() => {
    const model = modelRef.current
    if (!model) return
    const fresh = emptyStudioClip()
    model.loadClip(STUDIO_ANIM_NAME, fresh)
    setClip(fresh)
    setClipDisplayName("clip")
    setCurrentFrame(0)
    setPlaying(false)
    setActiveBone(null)
    setActiveMorph(null)
    setSelectedKeyframes([])
    setTimelineTab("allRot")
    // Bump after clearing selections so downstream effects don't see stale keyframes.
    setClipVersion((v) => v + 1)
    model.show(STUDIO_ANIM_NAME)
    model.seek(0)
    void idbClearClip()
    saveMeta({ hasClip: false })
    lastSavedClipRef.current = null
  }, [])

  return (
    <div className="flex h-screen w-full flex-col overflow-hidden text-foreground">
      <div className="flex min-h-0 flex-1">
        <EditorLeftPanel
          vmdInputRef={vmdInputRef}
          pmxFolderInputRef={pmxFolderInputRef}
          onPickVmdFile={onPickVmdFile}
          onPickPmxFolder={onPickPmxFolder}
          menubarValue={menubarValue}
          onMenubarValueChange={setMenubarValue}
          studioReady={studioReady}
          resetEditorState={resetEditorState}
          exportClipVmd={exportClipVmd}
          hasClip={clip != null}
          pmxPickFiles={pmxPickFiles}
          pmxPickPaths={pmxPickPaths}
          pmxPickSelected={pmxPickSelected}
          onPmxPickSelectedChange={setPmxPickSelected}
          onConfirmPmxPick={onConfirmPmxPick}
          modelBones={sidebarBones}
          clip={clip}
          selectedGroup={selectedGroup}
          activeBone={activeBone}
          onSelectGroup={handleSelectGroup}
          onSelectBone={handleSelectBone}
          morphNames={morphNames}
          activeMorph={activeMorph}
          onSelectMorph={handleSelectMorph}
          docsReadmeUrl={DOCS_README_URL}
          repoUrl={REPO_URL}
          appVersion={APP_VERSION}
        />

        {/* Center: viewport + timeline */}
        <div className="flex min-h-0 min-w-0 flex-1 flex-col">
          <EditorViewport ref={canvasRef} engineError={engineError} />
          {/* Timeline with dopesheet + value graph */}
          <div className="h-[220px] shrink-0 border-t border-border">
            <Timeline
              clip={clip}
              setClip={setClip}
              currentFrame={currentFrame}
              setCurrentFrame={setCurrentFrame}
              playing={playing}
              setPlaying={setPlaying}
              activeBone={activeBone}
              visibleBones={visibleBones}
              selectedKeyframes={selectedKeyframes}
              setSelectedKeyframes={setSelectedKeyframes}
              clipVersion={clipVersion}
              tab={timelineTab}
              setTab={setTimelineTab}
            />
          </div>
        </div>

        {/* Right sidebar — properties for active bone / morph / keyframe context */}
        <aside className="flex w-[256px] shrink-0 flex-col border-l border-sidebar-border text-sidebar-foreground">
          <div className="flex min-h-9 shrink-0 items-center border-b border-sidebar-border px-3 py-2">
            <span className="text-[11px] font-medium uppercase tracking-widest text-sidebar-foreground/70">
              Properties
            </span>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden px-3 py-2 text-[11px] [scrollbar-width:thin]">
            <PropertiesInspector
              clip={clip}
              currentFrame={currentFrame}
              activeBone={activeBone}
              activeMorph={activeMorph}
              morphWeight={morphWeightReadout}
              selectedKeyframes={selectedKeyframes}
              modelRef={modelRef}
              setClip={setClip}
              livePose={livePose}
              onInsertKeyframeAtPlayhead={insertKeyframeAtPlayhead}
              onDeleteSelectedKeyframes={deleteSelectedKeyframes}
              timelineTab={timelineTab}
              setTimelineTab={setTimelineTab}
              clipVersion={clipVersion}
            />
          </div>
        </aside>
      </div>

      <EditorStatusFooter
        statusPmxFileName={statusPmxFileName}
        clipDisplayName={clipDisplayName}
        hasClip={clip != null}
        statusMessage={statusMessage}
        statusFps={statusFps}
        appVersion={APP_VERSION}
      />
    </div>
  )
}
