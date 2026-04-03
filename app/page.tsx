"use client"

import {
  useEffect,
  useRef,
  useState,
  useMemo,
  useCallback,
  type ChangeEvent,
  type InputHTMLAttributes,
} from "react"
import { Engine, Model, Vec3, parsePmxFolderInput, pmxFileAtRelativePath } from "reze-engine"
import { Button } from "@/components/ui/button"
import {
  Menubar,
  MenubarContent,
  MenubarGroup,
  MenubarItem,
  MenubarMenu,
  MenubarSeparator,
  MenubarTrigger,
} from "@/components/ui/menubar"
import Link from "next/link"
import Image from "next/image"
import { FilePlus2, FolderOpen, FileMusic, FileDown } from "lucide-react"
import { BoneList } from "@/components/bone-list"
import { MorphList } from "@/components/morph-list"
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
import packageJson from "../package.json"

const MODEL_PATH = "/models/reze/reze.pmx"
const APP_VERSION = packageJson.version
const REPO_URL = "https://github.com/AmyangXYZ/reze-studio"
const DOCS_README_URL = `${REPO_URL}/blob/main/README.md`
const VMD_PATH = "/animations/miku.vmd"
const STUDIO_ANIM_NAME = "studio"
/** Basename for status bar when using bundled `MODEL_PATH` PMX. */
const BUNDLED_PMX_FILENAME = MODEL_PATH.replace(/^.*\//, "") || "model.pmx"

function emptyStudioClip(): AnimationClip {
  return { boneTracks: new Map(), morphTracks: new Map(), frameCount: 0 }
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
  return { boneTracks, morphTracks, frameCount: empty ? 0 : Math.max(clip.frameCount, inferred) }
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

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const engineRef = useRef<Engine | null>(null)
  const modelRef = useRef<Model | null>(null)
  const [engineError, setEngineError] = useState<string | null>(null)

  // ─── Persisted meta (deferred to useEffect to avoid SSR hydration mismatch) ──
  const persistedMeta = useRef<ReturnType<typeof loadMeta> | null>(null)

  // ─── Clip synced with engine via loadClip(STUDIO_ANIM_NAME) / getClip ──
  const [clip, setClip] = useState<AnimationClip | null>(null)
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
  /** Smoothed repaint FPS (same clock as the compositor; good enough for a readout). */
  const [statusFps, setStatusFps] = useState<number | null>(null)

  const playRef = useRef(false)
  const lastT = useRef<number | null>(null)
  /** Snapshotted before async PMX swap so clip/playhead survive `await loadModel`. */
  const clipRef = useRef<AnimationClip | null>(null)
  const currentFrameRef = useRef(0)
  const clipDisplayNameRef = useRef("clip")

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

  const persistState = useCallback(() => {
    saveMeta({
      activeBone,
      activeMorph,
      selectedGroup,
      currentFrame,
      clipDisplayName,
      hasClip: clip != null,
    })
    // Skip during playback (clip doesn't change) and when clip hasn't changed since last save.
    if (playing || clip === lastSavedClipRef.current) return
    const model = modelRef.current
    if (clip && model) {
      try {
        model.loadClip(STUDIO_ANIM_NAME, clip)
        const buf = model.exportVmd(STUDIO_ANIM_NAME)
        void idbSaveClip(buf)
      } catch { /* export can fail on empty clips — ignore */ }
    } else {
      void idbClearClip()
    }
    lastSavedClipRef.current = clip
  }, [activeBone, activeMorph, selectedGroup, currentFrame, clipDisplayName, clip, playing])

  const persistRef = useRef(persistState)
  useEffect(() => {
    persistRef.current = persistState
  }, [persistState])

  useEffect(() => {
    const iv = setInterval(() => persistRef.current(), 5000)
    const onUnload = () => persistRef.current()
    window.addEventListener("beforeunload", onUnload)
    return () => {
      clearInterval(iv)
      window.removeEventListener("beforeunload", onUnload)
      persistRef.current()
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

  // ─── Status bar: compositor FPS (cheap rolling average) ───────────────
  useEffect(() => {
    let raf = 0
    let frames = 0
    let lastReport = performance.now()
    const loop = (now: number) => {
      frames++
      if (now - lastReport >= 400) {
        const s = (now - lastReport) / 1000
        setStatusFps(s > 0 ? Math.round(frames / s) : null)
        frames = 0
        lastReport = now
      }
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
  }, [])

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

        engine.runRenderLoop()

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
    setMorphWeightReadout(model.getMorphWeights()[idx])
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

  // Timeline key click: jump playhead; curve keys also focus the bone on the list.
  useEffect(() => {
    if (selectedKeyframes.length !== 1) return
    const s = selectedKeyframes[0]
    setActiveMorph(null)
    if (s.type === "curve" && s.bone) setActiveBone(s.bone)
    setCurrentFrame(s.frame)
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
    if (!model || !activeBone || !clip) return null
    // React clip can fork from the engine’s internal clip; push state back before seek/read so sliders stay in sync
    model.loadClip(STUDIO_ANIM_NAME, clip)
    model.seek(Math.max(0, currentFrame) / 30)
    const p = readLocalPoseAfterSeek(model, activeBone)
    if (!p) return null
    return {
      euler: quatToEuler(p.rotation),
      translation: p.translation,
    }
  }, [currentFrame, clip, activeBone])

  const insertKeyframeAtPlayhead = useCallback(() => {
    const model = modelRef.current
    if (!clip || !activeBone || activeMorph || !model) return
    const frame = Math.round(Math.max(0, Math.min(clip.frameCount, currentFrame)))
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
    model.show(STUDIO_ANIM_NAME)
    model.seek(0)
    void idbClearClip()
    saveMeta({ hasClip: false })
    lastSavedClipRef.current = null
  }, [])

  return (
    <div className="flex h-screen w-full flex-col overflow-hidden text-foreground">
      <div className="flex min-h-0 flex-1">
        {/* Left sidebar */}
        <aside className="flex w-[225px] shrink-0 flex-col border-r border-border">
          <div className="shrink-0 border-b">
            <div className="pl-2 pt-0 flex items-center justify-between pb-1">
              <h1 className="scroll-m-20 max-w-[11rem] text-md font-extrabold leading-tight tracking-tight text-balance">
                REZE STUDIO
              </h1>
              <div className="flex shrink-0 items-center gap-0.5">
                <Button variant="ghost" size="sm" asChild className="hover:bg-black hover:text-white rounded-full">
                  <Link href="https://github.com/AmyangXYZ/reze-studio" target="_blank">
                    <Image src="/github-mark-white.svg" alt="GitHub" width={16} height={16} />
                  </Link>
                </Button>
              </div>
            </div>

            <div className="px-3 pb-2">
              <input
                ref={vmdInputRef}
                type="file"
                accept=".vmd"
                className="hidden"
                tabIndex={-1}
                aria-hidden
                onChange={onPickVmdFile}
              />
              {/* Off-screen, not `hidden`/`display:none` — some browsers ignore .click() on those. */}
              <input
                ref={pmxFolderInputRef}
                type="file"
                className="fixed left-0 top-0 -z-10 h-px w-px opacity-0"
                multiple
                {...({ webkitdirectory: "", mozdirectory: "" } as InputHTMLAttributes<HTMLInputElement>)}
                onChange={onPickPmxFolder}
              />
              <Menubar
                value={menubarValue}
                onValueChange={setMenubarValue}
                className="h-4 gap-0 rounded-none border-0 bg-transparent p-0 shadow-none"
              >
                <MenubarMenu value="file">
                  <MenubarTrigger className="h-4 rounded-sm px-1.5 py-0 text-xs font-normal text-muted-foreground">
                    File
                  </MenubarTrigger>
                  <MenubarContent
                    sideOffset={4}
                    className="min-w-[10.5rem] p-0.5 text-xs"
                  >
                    <MenubarGroup>
                      <MenubarItem
                        className="gap-2 py-1 pl-2 pr-1.5 text-[11px] text-muted-foreground"
                        disabled={!studioReady}
                        onSelect={resetEditorState}
                      >
                        <FilePlus2 className="size-3.5" />
                        New
                      </MenubarItem>
                      <MenubarSeparator className="my-0.5" />
                      <MenubarItem
                        className="gap-2 py-1 pl-2 pr-1.5 text-[11px] text-muted-foreground"
                        onSelect={(e) => {
                          e.preventDefault()
                          pmxFolderInputRef.current?.click()
                        }}
                      >
                        <FolderOpen className="size-3.5" />
                        Load PMX folder…
                      </MenubarItem>
                      <MenubarItem
                        className="gap-2 py-1 pl-2 pr-1.5 text-[11px] text-muted-foreground"
                        disabled={!studioReady}
                        onSelect={() => vmdInputRef.current?.click()}
                      >
                        <FileMusic className="size-3.5" />
                        Load VMD…
                      </MenubarItem>
                    </MenubarGroup>
                    <MenubarSeparator className="my-0.5" />
                    <MenubarGroup>
                      <MenubarItem
                        className="gap-2 py-1 pl-2 pr-1.5 text-[11px] text-muted-foreground"
                        disabled={!studioReady || !clip}
                        onSelect={exportClipVmd}
                      >
                        <FileDown className="size-3.5" />
                        Export VMD…
                      </MenubarItem>
                    </MenubarGroup>
                  </MenubarContent>
                </MenubarMenu>
                <MenubarMenu value="help">
                  <MenubarTrigger className="h-4 rounded-sm px-1.5 py-0 text-xs font-normal text-muted-foreground">
                    Help
                  </MenubarTrigger>
                  <MenubarContent sideOffset={4} className="min-w-[11rem] p-0.5 text-xs">
                    <MenubarGroup>
                      <MenubarItem
                        className="gap-2 py-1 pl-2 pr-1.5 text-[11px] text-muted-foreground"
                        asChild
                      >
                        <Link href={DOCS_README_URL} target="_blank" rel="noreferrer">
                          Tutorial (README)
                        </Link>
                      </MenubarItem>
                      <MenubarItem
                        className="gap-2 py-1 pl-2 pr-1.5 text-[11px] text-muted-foreground"
                        disabled
                      >
                        Keyboard shortcuts…
                      </MenubarItem>
                      <MenubarSeparator className="my-0.5" />
                      <MenubarItem
                        className="gap-2 py-1 pl-2 pr-1.5 text-[11px] text-muted-foreground"
                        onSelect={() => {
                          window.alert(`Reze Studio ${APP_VERSION}\nWebGPU MMD editor — ${REPO_URL}`)
                        }}
                      >
                        About Reze Studio
                      </MenubarItem>
                      <MenubarItem
                        className="gap-2 py-1 pl-2 pr-1.5 text-[11px] text-muted-foreground"
                        asChild
                      >
                        <Link href={`${REPO_URL}/issues`} target="_blank" rel="noreferrer">
                          Report an issue
                        </Link>
                      </MenubarItem>
                    </MenubarGroup>
                  </MenubarContent>
                </MenubarMenu>
                <MenubarMenu value="settings">
                  <MenubarTrigger className="h-4 rounded-sm px-1.5 py-0 text-xs font-normal text-muted-foreground">
                    Settings
                  </MenubarTrigger>
                  <MenubarContent sideOffset={4} className="min-w-[10rem] p-0.5 text-xs">
                    <MenubarGroup>
                      <MenubarItem
                        className="gap-2 py-1 pl-2 pr-1.5 text-[11px] text-muted-foreground"
                        disabled
                      >
                        Theme…
                      </MenubarItem>
                    </MenubarGroup>
                  </MenubarContent>
                </MenubarMenu>
              </Menubar>
              {pmxPickFiles && pmxPickPaths.length > 1 ? (
                <div className="mt-2 flex flex-col gap-1.5 rounded border border-border bg-muted/30 p-2 text-[10px]">
                  <span className="text-muted-foreground">Multiple .pmx files — choose one:</span>
                  <select
                    className="w-full rounded border border-border bg-background px-1 py-0.5 text-[11px]"
                    value={pmxPickSelected}
                    onChange={(e) => setPmxPickSelected(e.target.value)}
                  >
                    {pmxPickPaths.map((p) => (
                      <option key={p} value={p}>
                        {p}
                      </option>
                    ))}
                  </select>
                  <Button
                    type="button"
                    variant="secondary"
                    size="sm"
                    className="h-7 text-[11px]"
                    onClick={() => void onConfirmPmxPick()}
                  >
                    Load selected PMX
                  </Button>
                </div>
              ) : null}
            </div>
          </div>
          <div className="flex min-h-0 flex-1 flex-col">
            <div className="min-h-0 flex-1 overflow-hidden">
              <BoneList
                modelBones={sidebarBones}
                clip={clip}
                selectedGroup={selectedGroup}
                activeBone={activeBone}
                onSelectGroup={handleSelectGroup}
                onSelectBone={handleSelectBone}
              />
            </div>
            <div className="flex max-h-[168px] shrink-0 flex-col border-t border-border">
              <div className="shrink-0 px-3 pb-1 pt-2 text-[11px] font-medium uppercase tracking-widest text-muted-foreground">
                Morphs
              </div>
              <div className="min-h-0 flex-1 overflow-hidden">
                <MorphList
                  morphNames={morphNames}
                  activeMorph={activeMorph}
                  onSelectMorph={handleSelectMorph}
                />
              </div>
            </div>
          </div>
        </aside>

        {/* Center: viewport + timeline */}
        <div className="flex min-h-0 min-w-0 flex-1 flex-col">
          <div className="relative min-h-0 flex-1 overflow-hidden">
            <canvas ref={canvasRef} className="block h-full w-full touch-none" />
            {engineError ? (
              <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-background/70 p-4 text-center text-sm text-muted-foreground">
                {engineError}
              </div>
            ) : null}
          </div>
          {/* Timeline with dopesheet + value graph */}
          <div className="h-[220px] shrink-0 border-t border-border">
            <Timeline
              clip={clip}
              currentFrame={currentFrame}
              setCurrentFrame={setCurrentFrame}
              playing={playing}
              setPlaying={setPlaying}
              activeBone={activeBone}
              visibleBones={visibleBones}
              selectedKeyframes={selectedKeyframes}
              setSelectedKeyframes={setSelectedKeyframes}
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
            />
          </div>
        </aside>
      </div>

      <footer
        className="flex h-6 shrink-0 items-center gap-2 border-t border-border px-2 text-[10.5px] text-muted-foreground"
        role="status"
        aria-live="polite"
      >
        <div className="flex min-w-0 shrink-0 items-center gap-x-2 [overflow-wrap:anywhere]">
          <span>
            Model:{" "}
            <span className="font-medium text-foreground" title={statusPmxFileName}>
              {statusPmxFileName}
            </span>
          </span>
          <span className="text-border" aria-hidden>
            ·
          </span>
          <span>
            Animation:{" "}
            <span
              className="font-medium text-foreground"
              title={clip ? `${clipDisplayName}.vmd` : undefined}
            >
              {clip ? `${clipDisplayName}.vmd` : "—"}
            </span>
          </span>
        </div>
        <div className="min-w-0 flex-1 truncate px-2 text-left text-[10px] text-muted-foreground/90">
          {statusMessage}
        </div>
        <div className="flex shrink-0 items-center gap-x-2 tabular-nums">
          <span title="Main-thread / compositor frame rate">
            {statusFps != null ? `${statusFps} FPS` : "— FPS"}
          </span>
          <span className="text-border" aria-hidden>
            ·
          </span>
          <span>v{APP_VERSION}</span>
        </div>
      </footer>
    </div>
  )
}
