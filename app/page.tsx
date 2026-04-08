"use client"

import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  useMemo,
  useCallback,
  memo,
  forwardRef,
  type ChangeEvent,
  type InputHTMLAttributes,
  type RefObject,
} from "react"
import Link from "next/link"
import Image from "next/image"
import { FilePlus2, FolderOpen, FileMusic, FileDown } from "lucide-react"
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
import { BoneList } from "@/components/bone-list"
import { MorphList } from "@/components/morph-list"
import { PropertiesInspector } from "@/components/properties-inspector"
import { Timeline } from "@/components/timeline"
import { BONE_GROUPS, quatToEuler } from "@/lib/animation"
import type { AnimationClip, BoneKeyframe, MorphKeyframe } from "reze-engine"
import { Studio, useStudioActions, useStudioSelector } from "@/context/studio-context"
import { Playback, usePlayback } from "@/context/playback-context"
import {
  DEFAULT_STUDIO_CLIP_FRAMES,
  interpolationTemplateForFrame,
  readLocalPoseAfterSeek,
  upsertMorphKeyframeAtFrame,
} from "@/lib/utils"
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
    boneTracks.set(
      name,
      track.map((kf) => ({ ...kf })),
    )
  }
  const morphTracks = new Map<string, MorphKeyframe[]>()
  for (const [name, track] of clip.morphTracks) {
    if (!morphNames.has(name) || !track?.length) continue
    morphTracks.set(
      name,
      track.map((kf) => ({ ...kf })),
    )
  }
  let inferred = 0
  for (const t of boneTracks.values()) for (const k of t) inferred = Math.max(inferred, k.frame)
  for (const t of morphTracks.values()) for (const k of t) inferred = Math.max(inferred, k.frame)
  const empty = boneTracks.size === 0 && morphTracks.size === 0
  const end = empty ? Math.max(clip.frameCount, DEFAULT_STUDIO_CLIP_FRAMES) : Math.max(clip.frameCount, inferred)
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
function poseNearEqual(a: { euler: { x: number; y: number; z: number }; translation: Vec3 }, b: typeof a, eps = 1e-5) {
  return (
    Math.abs(a.euler.x - b.euler.x) < eps &&
    Math.abs(a.euler.y - b.euler.y) < eps &&
    Math.abs(a.euler.z - b.euler.z) < eps &&
    Math.abs(a.translation.x - b.translation.x) < eps &&
    Math.abs(a.translation.y - b.translation.y) < eps &&
    Math.abs(a.translation.z - b.translation.z) < eps
  )
}

/** Canvas + error overlay — playhead updates won’t reconcile this subtree. */
const StudioViewport = memo(
  forwardRef<HTMLCanvasElement, { engineError: string | null }>(function StudioViewport({ engineError }, ref) {
    return (
      <div className="relative min-h-0 flex-1 overflow-hidden">
        <canvas ref={ref} className="block h-full w-full touch-none" />
        {engineError ? (
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-background/70 p-4 text-center text-sm text-muted-foreground">
            {engineError}
          </div>
        ) : null}
      </div>
    )
  }),
)

type StudioLeftPanelProps = {
  vmdInputRef: RefObject<HTMLInputElement | null>
  pmxFolderInputRef: RefObject<HTMLInputElement | null>
  onPickVmdFile: (e: ChangeEvent<HTMLInputElement>) => void
  onPickPmxFolder: (e: ChangeEvent<HTMLInputElement>) => void
  menubarValue: string
  onMenubarValueChange: (v: string) => void
  studioReady: boolean
  resetStudioDocument: () => void
  exportClipVmd: () => void
  pmxPickFiles: File[] | null
  pmxPickPaths: string[]
  pmxPickSelected: string
  onPmxPickSelectedChange: (path: string) => void
  onConfirmPmxPick: () => void
  modelBones: string[]
  selectedGroup: string
  selectedBone: string | null
  onSelectGroup: (g: string) => void
  onSelectBone: (b: string) => void
  morphNames: string[]
  selectedMorph: string | null
  onSelectMorph: (name: string) => void
  docsReadmeUrl: string
  repoUrl: string
  appVersion: string
}

/** File menu + bone/morph lists — lives in page so the shell isn’t a separate layout file. */
const StudioLeftPanel = memo(function StudioLeftPanel({
  vmdInputRef,
  pmxFolderInputRef,
  onPickVmdFile,
  onPickPmxFolder,
  menubarValue,
  onMenubarValueChange,
  studioReady,
  resetStudioDocument,
  exportClipVmd,
  pmxPickFiles,
  pmxPickPaths,
  pmxPickSelected,
  onPmxPickSelectedChange,
  onConfirmPmxPick,
  modelBones,
  selectedGroup,
  selectedBone,
  onSelectGroup,
  onSelectBone,
  morphNames,
  selectedMorph,
  onSelectMorph,
  docsReadmeUrl,
  repoUrl,
  appVersion,
}: StudioLeftPanelProps) {
  const clip = useStudioSelector((s) => s.clip)
  const hasClip = clip != null
  return (
    <aside className="flex w-56 shrink-0 flex-col border-r border-border">
      <div className="shrink-0 border-b">
        <div className="pl-2 pt-0 flex items-center justify-between pb-1">
          <h1 className="scroll-m-20 max-w-28 text-md font-extrabold leading-tight tracking-tight text-balance">
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
            onValueChange={onMenubarValueChange}
            className="h-4 gap-0 rounded-none border-0 bg-transparent p-0 shadow-none"
          >
            <MenubarMenu value="file">
              <MenubarTrigger className="h-4 rounded-sm px-1.5 py-0 text-xs font-normal text-muted-foreground">
                File
              </MenubarTrigger>
              <MenubarContent sideOffset={4} className="min-w-32 p-0.5 text-xs">
                <MenubarGroup>
                  <MenubarItem
                    className="gap-2 py-1 pl-2 pr-1.5 text-[11px] text-muted-foreground"
                    disabled={!studioReady}
                    onSelect={resetStudioDocument}
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
                    disabled={!studioReady || !hasClip}
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
              <MenubarContent sideOffset={4} className="min-w-32 p-0.5 text-xs">
                <MenubarGroup>
                  <MenubarItem className="gap-2 py-1 pl-2 pr-1.5 text-[11px] text-muted-foreground" asChild>
                    <Link href={docsReadmeUrl} target="_blank" rel="noreferrer">
                      Tutorial (README)
                    </Link>
                  </MenubarItem>
                  <MenubarItem className="gap-2 py-1 pl-2 pr-1.5 text-[11px] text-muted-foreground" disabled>
                    Keyboard shortcuts…
                  </MenubarItem>
                  <MenubarSeparator className="my-0.5" />
                  <MenubarItem
                    className="gap-2 py-1 pl-2 pr-1.5 text-[11px] text-muted-foreground"
                    onSelect={() => {
                      window.alert(`Reze Studio ${appVersion}\nWebGPU MMD editor — ${repoUrl}`)
                    }}
                  >
                    About Reze Studio
                  </MenubarItem>
                  <MenubarItem className="gap-2 py-1 pl-2 pr-1.5 text-[11px] text-muted-foreground" asChild>
                    <Link href={`${repoUrl}/issues`} target="_blank" rel="noreferrer">
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
              <MenubarContent sideOffset={4} className="min-w-32 p-0.5 text-xs">
                <MenubarGroup>
                  <MenubarItem className="gap-2 py-1 pl-2 pr-1.5 text-[11px] text-muted-foreground" disabled>
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
                onChange={(e) => onPmxPickSelectedChange(e.target.value)}
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
            modelBones={modelBones}
            clip={clip}
            selectedGroup={selectedGroup}
            selectedBone={selectedBone}
            onSelectGroup={onSelectGroup}
            onSelectBone={onSelectBone}
          />
        </div>
        <div className="flex max-h-[196px] shrink-0 flex-col border-t border-border">
          <div className="shrink-0 px-3 pb-1 pt-2 text-[11px] font-medium uppercase tracking-widest text-muted-foreground">
            Morphs
          </div>
          <div className="min-h-0 flex-1 overflow-hidden">
            <MorphList
              morphNames={morphNames}
              clip={clip}
              selectedMorph={selectedMorph}
              onSelectMorph={onSelectMorph}
            />
          </div>
        </div>
      </div>
    </aside>
  )
})

const StudioStatusFooter = memo(function StudioStatusFooter({
  statusPmxFileName,
  clipDisplayName,
  hasClip,
  statusMessage,
  statusFps,
  appVersion,
}: {
  statusPmxFileName: string
  clipDisplayName: string
  hasClip: boolean
  statusMessage: string
  statusFps: number | null
  appVersion: string
}) {
  return (
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
          <span className="font-medium text-foreground" title={hasClip ? `${clipDisplayName}.vmd` : undefined}>
            {hasClip ? `${clipDisplayName}.vmd` : "—"}
          </span>
        </span>
      </div>
      <div className="min-w-0 flex-1 truncate px-2 text-left text-[10px] text-muted-foreground/90">{statusMessage}</div>
      <div className="flex shrink-0 items-center gap-x-2 tabular-nums">
        <span title="Main-thread / compositor frame rate">{statusFps != null ? `${statusFps} FPS` : "— FPS"}</span>
        <span className="text-border" aria-hidden>
          ·
        </span>
        <span>v{appVersion}</span>
      </div>
    </footer>
  )
})

function StudioPage() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const engineRef = useRef<Engine | null>(null)
  const modelRef = useRef<Model | null>(null)
  const [engineError, setEngineError] = useState<string | null>(null)

  // ─── Document + selection live in `<Studio>`; page wires engine + chrome only ──
  // Slice subscriptions so unrelated state changes don't re-render this page.
  const clip = useStudioSelector((s) => s.clip)
  const clipDisplayName = useStudioSelector((s) => s.clipDisplayName)
  const selectedBone = useStudioSelector((s) => s.selectedBone)
  const selectedMorph = useStudioSelector((s) => s.selectedMorph)
  const selectedKeyframes = useStudioSelector((s) => s.selectedKeyframes)
  const { commit, setClipDisplayName, setSelectedBone, setSelectedMorph, setSelectedKeyframes } =
    useStudioActions()
  const { currentFrame, setCurrentFrame, playing, setPlaying } = usePlayback()
  /** Model finished loading (file menu + export need a live Model instance). */
  const [studioReady, setStudioReady] = useState(false)

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

  const [selectedGroup, setSelectedGroup] = useState("All Bones")
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
  /** Imperative handle into the timeline canvas — the playback rAF loop uses
   *  this to repaint the playhead at 60Hz without re-rendering React. */
  const playheadDrawRef = useRef<((frame: number) => void) | null>(null)
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

  /** Unsaved clip edits — browser `beforeunload` only reads refs (stable listener). */
  const documentDirtyRef = useRef(false)
  /** Skip marking dirty for the next `clip` update (loads / reset / export handoff). */
  const suppressClipDirtyRef = useRef(0)

  useEffect(() => {
    if (clip == null) return
    if (suppressClipDirtyRef.current > 0) {
      suppressClipDirtyRef.current -= 1
      return
    }
    documentDirtyRef.current = true
  }, [clip])

  useEffect(() => {
    const onBeforeUnload = (e: BeforeUnloadEvent) => {
      if (!documentDirtyRef.current) return
      e.preventDefault()
      e.returnValue = ""
    }
    window.addEventListener("beforeunload", onBeforeUnload)
    return () => window.removeEventListener("beforeunload", onBeforeUnload)
  }, [])

  // ─── Playback loop ───────────────────────────────────────────────────
  // While playing, the engine owns the timeline (advances pose every render).
  // React's job is to MIRROR the engine's clock into `currentFrame` so the UI
  // (timeline playhead, readouts) stays in sync — not to drive it. No dt math,
  // no double-seek.
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
      // Pure imperative path — no React state updates during playback.
      // `currentFrameRef` + `playheadDrawRef(frame)` drive the visual playhead
      // at monitor refresh rate with zero reconciliation cost.
      currentFrameRef.current = frame
      playheadDrawRef.current?.(frame)
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => {
      cancelAnimationFrame(raf)
      // Flush the last drawn frame into React state so the paused view matches
      // what the playhead was last showing (throttled sync may lag by up to 100ms).
      setCurrentFrame(currentFrameRef.current)
    }
  }, [playing, frameCount, setCurrentFrame, setPlaying])

  // ─── Keyboard shortcuts ──────────────────────────────────────────────
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // Don't hijack keys while the user is typing into an input/textarea/contenteditable.
      const t = e.target as HTMLElement | null
      if (t) {
        const tag = t.tagName
        if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || t.isContentEditable) return
      }
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
  }, [frameCount, setCurrentFrame, setPlaying])

  // ─── Bone selection handlers ─────────────────────────────────────────
  const handleSelectGroup = useCallback(
    (g: string) => {
      setSelectedGroup((prev) => (prev === g ? "" : g))
      setSelectedBone(null)
      setSelectedMorph(null)
      setSelectedKeyframes([])
    },
    [setSelectedBone, setSelectedMorph, setSelectedKeyframes],
  )

  const handleSelectBone = useCallback(
    (b: string) => {
      setSelectedMorph(null)
      setSelectedBone(b)
      setSelectedKeyframes([])
    },
    [setSelectedBone, setSelectedMorph, setSelectedKeyframes],
  )

  const handleSelectMorph = useCallback(
    (name: string) => {
      setSelectedBone(null)
      setSelectedMorph(name)
      setSelectedKeyframes([])
    },
    [setSelectedBone, setSelectedMorph, setSelectedKeyframes],
  )

  useEffect(() => {
    if (selectedBone && !pmxBoneNames.has(selectedBone)) setSelectedBone(null)
  }, [selectedBone, pmxBoneNames, setSelectedBone])

  useEffect(() => {
    if (selectedMorph && !morphNames.includes(selectedMorph)) setSelectedMorph(null)
  }, [selectedMorph, morphNames, setSelectedMorph])

  useEffect(() => {
    setSelectedKeyframes((prev) => prev.filter((s) => s.type !== "curve" || !s.bone || pmxBoneNames.has(s.bone)))
  }, [pmxBoneNames, setSelectedKeyframes])

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
      setMorphWeightReadout(null)
      setStatusPmxFileName("—")
      setStatusFps(null)
      lastReportedEngineFpsRef.current = null
      modelRef.current = null
      engineRef.current?.stopRenderLoop()
      engineRef.current?.dispose()
      engineRef.current = null
    }
  }, [commit, setClipDisplayName, setSelectedMorph])

  // Upload clip to engine ONLY on edits — never on playhead movement. After upload
  // we re-seek the engine to the current React frame so a commit during pause
  // doesn't snap the viewport back to frame 0.
  useEffect(() => {
    const model = modelRef.current
    if (!model || !clip) return
    model.loadClip(STUDIO_ANIM_NAME, clip)
    model.seek(Math.max(0, currentFrameRef.current) / 30)
  }, [clip])

  // Scrub: when paused, React owns the playhead and pushes seeks into the engine.
  // When playing, the engine owns the playhead — we read FROM it via the rAF loop
  // below and must NOT seek here (would fight the engine's own timeline).
  useLayoutEffect(() => {
    const model = modelRef.current
    if (!model || !clip) return
    if (!playing) {
      model.seek(Math.max(0, currentFrame) / 30)
    }
    if (!selectedMorph) {
      setMorphWeightReadout(null)
      return
    }
    const morphing = model.getMorphing()
    const idx = morphing.morphs.findIndex((m) => m.name === selectedMorph)
    if (idx < 0) {
      setMorphWeightReadout(null)
      return
    }
    const w = model.getMorphWeights()[idx]
    setMorphWeightReadout((prev) => (prev === w ? prev : w))
  }, [currentFrame, clip, selectedMorph, playing])

  useEffect(() => {
    const model = modelRef.current
    if (!model || !clip) return
    if (playing) {
      // Make sure the engine starts from the React playhead. If the user pressed
      // play at the end, rewind to 0 first (and mirror that into React state).
      let startFrame = currentFrameRef.current
      if (startFrame >= frameCount) {
        startFrame = 0
        setCurrentFrame(0)
      }
      model.seek(Math.max(0, startFrame) / 30)
      model.play()
      if (model.name === "reze") {
        model.setMorphWeight("抗穿模", 0.5)
      }
    } else model.pause()
  }, [playing, clip, frameCount, setCurrentFrame])

  useEffect(() => {
    setCurrentFrame((c) => Math.min(c, frameCount))
  }, [frameCount, setCurrentFrame])

  // Timeline key click: jump playhead; curve keys focus bone/morph on the list — tab stays user-controlled.
  useEffect(() => {
    if (selectedKeyframes.length !== 1) return
    const s = selectedKeyframes[0]
    if (s.morph) {
      setSelectedBone(null)
      setSelectedMorph(s.morph)
    } else {
      setSelectedMorph(null)
      if (s.type === "curve" && s.bone) setSelectedBone(s.bone)
    }
    setCurrentFrame(s.frame)
  }, [selectedKeyframes, setSelectedBone, setSelectedMorph, setCurrentFrame])

  const deleteSelectedKeyframes = useCallback(() => {
    if (!clip || selectedKeyframes.length !== 1) return
    const sel = selectedKeyframes[0]
    setSelectedKeyframes([])

    if (sel.type === "curve" && sel.morph) {
      commit((prev) => {
        if (!prev) return prev
        const track = prev.morphTracks.get(sel.morph!)
        if (!track) return prev
        const i = track.findIndex((k) => k.frame === sel.frame)
        if (i < 0) return prev
        const morphTracks = new Map(prev.morphTracks)
        const next = track.filter((_, j) => j !== i)
        if (next.length === 0) morphTracks.delete(sel.morph!)
        else morphTracks.set(sel.morph!, next)
        return { ...prev, morphTracks }
      })
      return
    }
    if (sel.type === "curve" && sel.bone) {
      commit((prev) => {
        if (!prev) return prev
        const track = prev.boneTracks.get(sel.bone!)
        if (!track) return prev
        const i = track.findIndex((k) => k.frame === sel.frame)
        if (i < 0) return prev
        const boneTracks = new Map(prev.boneTracks)
        const next = track.filter((_, j) => j !== i)
        if (next.length === 0) boneTracks.delete(sel.bone!)
        else boneTracks.set(sel.bone!, next)
        return { ...prev, boneTracks }
      })
      return
    }
    if (sel.type !== "dope") return
    const f = sel.frame
    if (timelineTab === "morph" && selectedMorph) {
      commit((prev) => {
        if (!prev) return prev
        const track = prev.morphTracks.get(selectedMorph)
        if (!track) return prev
        const i = track.findIndex((k) => k.frame === f)
        if (i < 0) return prev
        const morphTracks = new Map(prev.morphTracks)
        const next = track.filter((_, j) => j !== i)
        if (next.length === 0) morphTracks.delete(selectedMorph)
        else morphTracks.set(selectedMorph, next)
        return { ...prev, morphTracks }
      })
      return
    }
    commit((prev) => {
      if (!prev) return prev
      const boneTracks = new Map(prev.boneTracks)
      for (const [name, track] of prev.boneTracks) {
        const i = track.findIndex((k) => k.frame === f)
        if (i < 0) continue
        const next = track.filter((_, j) => j !== i)
        if (next.length === 0) boneTracks.delete(name)
        else boneTracks.set(name, next)
      }
      return { ...prev, boneTracks }
    })
  }, [clip, selectedKeyframes, timelineTab, selectedMorph, commit, setSelectedKeyframes])

  const livePose = useMemo(() => {
    const model = modelRef.current
    if (!model || !selectedBone || !clip) {
      livePoseStableRef.current = null
      return null
    }
    // Engine is already at the right time: while playing it advances itself,
    // while paused the scrub effect above has seeked it. No loadClip — that's
    // edit-only now (see useEffect([clip])).
    if (!playing) {
      model.seek(Math.max(0, currentFrame) / 30)
    }
    const p = readLocalPoseAfterSeek(model, selectedBone)
    if (!p) {
      livePoseStableRef.current = null
      return null
    }
    // Prefer the stored keyframe value at the current frame when one exists:
    // the runtime skeleton returns the post-IK / post-constraint rotation, so
    // bones under an IK chain would otherwise display a different value than
    // what's actually stored in the keyframe (and what the timeline shows).
    const frameInt = Math.round(Math.max(0, currentFrame))
    const boneTrack = clip.boneTracks.get(selectedBone)
    const kfAt = boneTrack?.find((k) => k.frame === frameInt)
    const next = kfAt
      ? {
          euler: quatToEuler(kfAt.rotation),
          translation: kfAt.translation,
        }
      : {
          euler: quatToEuler(p.rotation),
          translation: p.translation,
        }
    const prev = livePoseStableRef.current
    if (prev && poseNearEqual(prev, next)) return prev
    livePoseStableRef.current = next
    return next
  }, [currentFrame, clip, selectedBone, playing])

  const insertKeyframeAtPlayhead = useCallback(() => {
    const model = modelRef.current
    if (!clip || !model) return
    const frame = Math.round(Math.max(0, currentFrame))

    if (selectedMorph && !selectedBone) {
      const w = morphWeightReadout ?? 0
      commit(upsertMorphKeyframeAtFrame(clip, selectedMorph, frame, w))
      setSelectedKeyframes([{ type: "curve", morph: selectedMorph, frame }])
      return
    }

    if (!selectedBone) return
    model.loadClip(STUDIO_ANIM_NAME, clip)
    model.seek(Math.max(0, currentFrame) / 30)
    const pose = readLocalPoseAfterSeek(model, selectedBone)
    if (!pose) return

    const prevTrack = clip.boneTracks.get(selectedBone)
    const ip = interpolationTemplateForFrame(prevTrack, frame)
    const nextTrack = [...(prevTrack ?? [])].filter((k) => k.frame !== frame)
    nextTrack.push({
      boneName: selectedBone,
      frame,
      rotation: pose.rotation,
      translation: pose.translation,
      interpolation: ip,
    })
    nextTrack.sort((a, b) => a.frame - b.frame)
    const boneTracks = new Map(clip.boneTracks)
    boneTracks.set(selectedBone, nextTrack)
    commit({ ...clip, boneTracks })
    setSelectedKeyframes([{ type: "curve", bone: selectedBone, frame, channel: "rx" }])
  }, [clip, selectedBone, selectedMorph, currentFrame, morphWeightReadout, commit, setSelectedKeyframes])

  const syncStudioAfterNewClip = useCallback(
    (model: Model) => {
      setCurrentFrame(0)
      setPlaying(false)
      setSelectedKeyframes([])
      setClipVersion((v) => v + 1)
      model.show(STUDIO_ANIM_NAME)
      model.seek(0)
      if (model.name === "reze") model.setMorphWeight("抗穿模", 0.5)
    },
    [setSelectedKeyframes, setCurrentFrame, setPlaying],
  )

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
      setSelectedBone((prev) => (prev && boneSet.has(prev) ? prev : null))
      setSelectedMorph((prev) => (prev && morphSet.has(prev) ? prev : null))
      setSelectedKeyframes((prev) => prev.filter((s) => s.type !== "curve" || !s.bone || boneSet.has(s.bone)))

      const prev = animationSnapshot.clip
      const hasPrevTimeline =
        prev != null && (prev.boneTracks.size > 0 || prev.morphTracks.size > 0 || prev.frameCount > 0)

      let nextClip: AnimationClip
      let nextDisplay: string
      let nextFrame: number
      let nextPlaying: boolean

      if (hasPrevTimeline) {
        nextClip = clipRetainedForModel(prev, boneSet, morphSet)
        nextDisplay = animationSnapshot.clipDisplayName
        nextFrame = Math.min(Math.max(0, animationSnapshot.currentFrame), Math.max(0, nextClip.frameCount))
        nextPlaying = animationSnapshot.playing
      } else {
        nextClip = emptyStudioClip()
        nextDisplay = sanitizeClipFilenameBase(displayStem)
        nextFrame = 0
        nextPlaying = false
      }

      model.loadClip(STUDIO_ANIM_NAME, nextClip)
      suppressClipDirtyRef.current += 1
      commit(nextClip)
      // Retained motion is still only in memory until export — keep warning if tracks exist.
      documentDirtyRef.current = hasPrevTimeline
      setClipDisplayName(nextDisplay)
      setCurrentFrame(nextFrame)
      setPlaying(nextPlaying)
      model.show(STUDIO_ANIM_NAME)
      model.seek(nextFrame / 30)
      if (nextPlaying) model.play()
      else model.pause()
      setEngineError(null)
    },
    [commit, setSelectedBone, setSelectedMorph, setSelectedKeyframes, setClipDisplayName, setCurrentFrame, setPlaying],
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
        await new Promise((resolve) => requestAnimationFrame(resolve))
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
          suppressClipDirtyRef.current += 1
          commit(c)
          documentDirtyRef.current = false
          setClipDisplayName(sanitizeClipFilenameBase(fileStem(file.name)))
          syncStudioAfterNewClip(model)
        }
      } catch (err) {
        window.alert(err instanceof Error ? err.message : String(err))
      } finally {
        URL.revokeObjectURL(url)
      }
    },
    [syncStudioAfterNewClip, commit, setClipDisplayName],
  )

  const exportClipVmd = useCallback(() => {
    const model = modelRef.current
    if (!model || !clip) return
    const base = sanitizeClipFilenameBase(clipDisplayName)
    try {
      model.loadClip(STUDIO_ANIM_NAME, clip)
      const buf = model.exportVmd(STUDIO_ANIM_NAME)
      downloadBlob(new Blob([buf], { type: "application/octet-stream" }), `${base}-export.vmd`)
      documentDirtyRef.current = false
    } catch (err) {
      window.alert(err instanceof Error ? err.message : String(err))
    }
  }, [clip, clipDisplayName])

  const resetStudioDocument = useCallback(() => {
    const model = modelRef.current
    if (!model) return
    const fresh = emptyStudioClip()
    model.loadClip(STUDIO_ANIM_NAME, fresh)
    suppressClipDirtyRef.current += 1
    commit(fresh)
    documentDirtyRef.current = false
    setClipDisplayName("clip")
    setCurrentFrame(0)
    setPlaying(false)
    setSelectedBone(null)
    setSelectedMorph(null)
    setSelectedKeyframes([])
    // Bump after clearing selections so downstream effects don't see stale keyframes.
    setClipVersion((v) => v + 1)
    model.show(STUDIO_ANIM_NAME)
    model.seek(0)
  }, [commit, setClipDisplayName, setSelectedBone, setSelectedMorph, setSelectedKeyframes, setCurrentFrame, setPlaying])

  return (
    <div className="flex h-screen w-full flex-col overflow-hidden text-foreground">
      <div className="flex min-h-0 flex-1">
        <StudioLeftPanel
          vmdInputRef={vmdInputRef}
          pmxFolderInputRef={pmxFolderInputRef}
          onPickVmdFile={onPickVmdFile}
          onPickPmxFolder={onPickPmxFolder}
          menubarValue={menubarValue}
          onMenubarValueChange={setMenubarValue}
          studioReady={studioReady}
          resetStudioDocument={resetStudioDocument}
          exportClipVmd={exportClipVmd}
          pmxPickFiles={pmxPickFiles}
          pmxPickPaths={pmxPickPaths}
          pmxPickSelected={pmxPickSelected}
          onPmxPickSelectedChange={setPmxPickSelected}
          onConfirmPmxPick={onConfirmPmxPick}
          modelBones={sidebarBones}
          selectedGroup={selectedGroup}
          selectedBone={selectedBone}
          onSelectGroup={handleSelectGroup}
          onSelectBone={handleSelectBone}
          morphNames={morphNames}
          selectedMorph={selectedMorph}
          onSelectMorph={handleSelectMorph}
          docsReadmeUrl={DOCS_README_URL}
          repoUrl={REPO_URL}
          appVersion={APP_VERSION}
        />

        {/* Center: viewport + timeline */}
        <div className="flex min-h-0 min-w-0 flex-1 flex-col">
          <StudioViewport ref={canvasRef} engineError={engineError} />
          {/* Timeline with dopesheet + value graph */}
          <div className="h-[220px] shrink-0 border-t border-border">
            <Timeline visibleBones={visibleBones} clipVersion={clipVersion} tab={timelineTab} setTab={setTimelineTab} playheadDrawRef={playheadDrawRef} />
          </div>
        </div>

        {/* Right sidebar — properties for selected bone / morph / keyframe context */}
        <aside className="flex w-64 shrink-0 flex-col border-l border-sidebar-border text-sidebar-foreground">
          <div className="flex min-h-9 shrink-0 items-center border-b border-sidebar-border px-3 py-2">
            <span className="text-[11px] font-medium uppercase tracking-widest text-sidebar-foreground/70">
              Properties
            </span>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden px-3 py-2 text-[11px] [scrollbar-width:thin]">
            <PropertiesInspector
              morphWeight={morphWeightReadout}
              modelRef={modelRef}
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

      <StudioStatusFooter
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

export default function Home() {
  return (
    <Studio>
      <Playback>
        <StudioPage />
      </Playback>
    </Studio>
  )
}
