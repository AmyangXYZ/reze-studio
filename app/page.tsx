"use client"

import { useEffect, useRef, useState, useMemo, useCallback } from "react"
import { Engine, Model, Vec3 } from "reze-engine"
import { Button } from "@/components/ui/button"
import {
  Menubar,
  MenubarContent,
  MenubarGroup,
  MenubarItem,
  MenubarMenu,
  MenubarSeparator,
  MenubarShortcut,
  MenubarTrigger,
} from "@/components/ui/menubar"
import Link from "next/link"
import Image from "next/image"
import { BoneList } from "@/components/bone-list"
import { Timeline, type SelectedKeyframe } from "@/components/timeline"
import { BONE_GROUPS } from "@/lib/animation"
import type { AnimationClip } from "reze-engine"

const MODEL_PATH = "/models/reze/reze.pmx"
const VMD_PATH = "/animations/miku.vmd"
const STUDIO_ANIM_NAME = "studio"

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const engineRef = useRef<Engine | null>(null)
  const modelRef = useRef<Model | null>(null)
  const [engineError, setEngineError] = useState<string | null>(null)

  // ─── Clip from VMD (via engine loadAnimation → getAnimationClip) ─────
  const [clip, setClip] = useState<AnimationClip | null>(null)
  const frameCount = clip?.frameCount ?? 0
  /** PMX skeleton bone names; used to hide VMD tracks that do not exist on the loaded model. */
  const [pmxBoneNames, setPmxBoneNames] = useState<ReadonlySet<string>>(new Set())

  const allBones = useMemo(() => {
    if (!clip) return []
    const keys = Array.from(clip.boneTracks.keys())
    if (pmxBoneNames.size === 0) return keys
    return keys.filter((k) => pmxBoneNames.has(k))
  }, [clip, pmxBoneNames])

  const [currentFrame, setCurrentFrame] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [activeBone, setActiveBone] = useState<string | null>(null)
  const [selectedGroup, setSelectedGroup] = useState("All Bones")
  const [selectedKeyframes, setSelectedKeyframes] = useState<SelectedKeyframe[]>([])

  const playRef = useRef(false)
  const lastT = useRef<number | null>(null)

  const visibleBones = useMemo(() => {
    const g = BONE_GROUPS[selectedGroup]
    if (!g) return allBones
    return g.filter((name) => allBones.includes(name))
  }, [selectedGroup, allBones])

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
  }, [])

  const handleSelectBone = useCallback((b: string) => {
    setActiveBone(b)
  }, [])

  useEffect(() => {
    if (activeBone && !allBones.includes(activeBone)) setActiveBone(null)
  }, [activeBone, allBones])

  useEffect(() => {
    setSelectedKeyframes((prev) =>
      prev.filter((s) => s.type !== "curve" || !s.bone || allBones.includes(s.bone)),
    )
  }, [allBones])

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

        engine.setPhysicsEnabled(false)
        engine.addGround({
          diffuseColor: new Vec3(0.14, 0.12, 0.16),
        })

        try {
          const model = await engine.loadModel("reze", MODEL_PATH)
          if (disposed) return
          modelRef.current = model
          setPmxBoneNames(new Set(model.getSkeleton().bones.map((b) => b.name)))
          model.setMorphWeight("抗穿模", 0.5)
          try {
            await model.loadAnimation(STUDIO_ANIM_NAME, VMD_PATH)
            if (disposed) return
            const c = model.getAnimationClip(STUDIO_ANIM_NAME)
            if (c) {
              setClip(c)
              model.play(STUDIO_ANIM_NAME, { loop: false })
              model.pause()
              model.seek(0)
            }
          } catch (e) {
            console.warn(`VMD load failed — add file at public${VMD_PATH}`, e)
          }
        } catch {
          setEngineError(`Add model at public${MODEL_PATH}`)
        }

        engine.runRenderLoop()
        engineRef.current = engine
      } catch (e) {
        console.error(e)
        setEngineError(e instanceof Error ? e.message : String(e))
      }
    }

    void initEngine()

    return () => {
      disposed = true
      setPmxBoneNames(new Set())
      modelRef.current = null
      engineRef.current?.stopRenderLoop()
      engineRef.current?.dispose()
      engineRef.current = null
    }
  }, [])

  // Keep model pose locked to timeline frame.
  useEffect(() => {
    const model = modelRef.current
    if (!model || !clip) return
    model.seek(Math.max(0, currentFrame) / 30)
  }, [currentFrame, clip])

  useEffect(() => {
    const model = modelRef.current
    if (!model || !clip) return
    if (playing) model.play()
    else model.pause()
  }, [playing, clip])

  useEffect(() => {
    if (!playing || frameCount <= 0) return
    if (currentFrame >= frameCount) setCurrentFrame(0)
  }, [playing, currentFrame, frameCount])

  return (
    <div className="flex h-screen w-full flex-col overflow-hidden text-foreground">
      <div className="flex min-h-0 flex-1">
        {/* Left sidebar */}
        <aside className="flex w-[240px] shrink-0 flex-col border-r border-border">
          <div className="shrink-0 border-b">
            <div className="pl-2 pt-0 flex items-center justify-between pb-1">
              <h1 className="scroll-m-20 max-w-[11rem] text-md font-extrabold leading-tight tracking-tight text-balance">
                REZE STUDIO
              </h1>
              <Button variant="ghost" size="sm" asChild className="hover:bg-black hover:text-white rounded-full">
                <Link href="https://github.com/AmyangXYZ/reze-studio" target="_blank">
                  <Image src="/github-mark-white.svg" alt="GitHub" width={16} height={16} />
                </Link>
              </Button>
            </div>

            <div className="px-3 pb-2">
              <Menubar className="h-4 gap-0 rounded-none border-0 bg-transparent p-0 shadow-none">
                <MenubarMenu>
                  <MenubarTrigger className="h-4 rounded-sm px-1.5 py-0 text-xs font-normal text-muted-foreground">
                    File
                  </MenubarTrigger>
                  <MenubarContent
                    sideOffset={4}
                    className="min-w-[10.5rem] p-0.5 text-xs"
                  >
                    <MenubarGroup>
                      <MenubarItem className="gap-2 py-1 pl-2 pr-1.5 text-xs">
                        Load model…
                      </MenubarItem>
                      <MenubarItem className="gap-2 py-1 pl-2 pr-1.5 text-xs">
                        Load animation…
                      </MenubarItem>
                    </MenubarGroup>
                    <MenubarSeparator className="my-0.5" />
                    <MenubarGroup>
                      <MenubarItem className="gap-2 py-1 pl-2 pr-1.5 text-xs">
                        Export…
                      </MenubarItem>
                    </MenubarGroup>
                  </MenubarContent>
                </MenubarMenu>
                <MenubarMenu>
                  <MenubarTrigger className="h-4 rounded-sm px-1.5 py-0 text-xs font-normal text-muted-foreground">
                    Edit
                  </MenubarTrigger>
                  <MenubarContent sideOffset={4} className="min-w-[9rem] p-0.5 text-xs">
                    <MenubarGroup>
                      <MenubarItem className="gap-2 py-1 pl-2 pr-1.5 text-xs" disabled>
                        Undo
                        <MenubarShortcut className="text-[10px] tracking-wide">⌘Z</MenubarShortcut>
                      </MenubarItem>
                      <MenubarItem className="gap-2 py-1 pl-2 pr-1.5 text-xs" disabled>
                        Redo
                        <MenubarShortcut className="text-[10px] tracking-wide">⇧⌘Z</MenubarShortcut>
                      </MenubarItem>
                    </MenubarGroup>
                  </MenubarContent>
                </MenubarMenu>
                <MenubarMenu>
                  <MenubarTrigger className="h-4 rounded-sm px-1.5 py-0 text-xs font-normal text-muted-foreground">
                    Preferences
                  </MenubarTrigger>
                  <MenubarContent sideOffset={4} className="min-w-[10rem] p-0.5 text-xs">
                    <MenubarGroup>
                      <MenubarItem className="py-1 pl-2 pr-1.5 text-xs" disabled>
                        Theme…
                      </MenubarItem>
                      <MenubarItem className="py-1 pl-2 pr-1.5 text-xs" disabled>
                        Keyboard shortcuts…
                      </MenubarItem>
                    </MenubarGroup>
                  </MenubarContent>
                </MenubarMenu>
              </Menubar>
            </div>
          </div>
          {/* Bone list */}
          <div className="min-h-0 flex-1 overflow-hidden">
            <BoneList
              allBones={allBones}
              selectedGroup={selectedGroup}
              activeBone={activeBone}
              onSelectGroup={handleSelectGroup}
              onSelectBone={handleSelectBone}
            />
          </div>
          <div className="shrink-0 space-y-2 border-t border-border px-3 py-2">
            <div className="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
              Morphs
            </div>
            <div className="h-24 rounded-md border border-dashed border-border bg-muted/30 p-2.5 text-[10px] text-muted-foreground">
              Sliders area
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

        {/* Right sidebar */}
        <aside className="flex w-[240px] shrink-0 flex-col border-l border-border">
          <div className="flex min-h-9 shrink-0 items-center border-b border-border px-3 py-2 text-xs font-medium text-muted-foreground">
            Selection / props
          </div>
          <div className="min-h-0 flex-1 space-y-2 overflow-auto px-3 py-2">
            <div className="rounded-md border border-dashed border-border bg-muted/30 p-2.5 text-[10px] text-muted-foreground">
              Rotation / position
            </div>
            <div className="rounded-md border border-dashed border-border bg-muted/30 p-2.5 text-[10px] text-muted-foreground">
              Interpolation
            </div>
            <div className="rounded-md border border-dashed border-border bg-muted/30 p-2.5 text-[10px] text-muted-foreground">
              Actions
            </div>
          </div>
        </aside>
      </div>
    </div>
  )
}
