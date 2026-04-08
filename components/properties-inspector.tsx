"use client"

import type { RefObject } from "react"
import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react"
import type { AnimationClip, BoneInterpolation, BoneKeyframe, Model } from "reze-engine"
import { Quat, Vec3 } from "reze-engine"
import { Button } from "@/components/ui/button"
import {
  boneTitleSubtitle,
  eulerToQuat,
  quatToEuler,
  ROT_CHANNELS,
  TRA_CHANNELS,
} from "@/lib/animation"
import { AxisSliderRow } from "@/components/axis-slider-row"
import { InterpolationCurveEditor, PRESETS, type CurvePoint } from "@/components/interpolation-curve-editor"
import {
  cloneBoneInterpolation,
  cn,
  interpolationTemplateForFrame,
  readLocalPoseAfterSeek,
  VMD_LINEAR_DEFAULT_IP,
} from "@/lib/utils"
import { useStudioActions, useStudioSelector } from "@/context/studio-context"
import { usePlaybackFrameRef, usePlaybackSelector } from "@/context/playback-context"

/** Must match `loadClip` name in app/page (engine clip vs React state). */
const STUDIO_ANIM_NAME = "studio"

/** Curve tabs that show rotation channels — keys must match `components/timeline.tsx` TABS. */
const ROT_TAB_KEYS = new Set<string>(["allRot", "rx", "ry", "rz"])
const TRA_TAB_KEYS = new Set<string>(["allTra", "tx", "ty", "tz"])
const ROT_AXIS_KEYS = ["rx", "ry", "rz"] as const
const TRA_AXIS_KEYS = ["tx", "ty", "tz"] as const

/** Off rotation group → All Rot; on RY/RZ but dragging X → RX (same for translation / All Trans). */
function syncTimelineTabForRotationDrag(
  currentTab: string,
  axisIdx: 0 | 1 | 2,
  setTimelineTab: (t: string) => void,
) {
  if (!ROT_TAB_KEYS.has(currentTab)) {
    setTimelineTab("allRot")
    return
  }
  if (currentTab === "allRot") return
  const want = ROT_AXIS_KEYS[axisIdx]
  if (currentTab !== want) setTimelineTab(want)
}

function syncTimelineTabForTranslationDrag(
  currentTab: string,
  axisIdx: 0 | 1 | 2,
  setTimelineTab: (t: string) => void,
) {
  if (!TRA_TAB_KEYS.has(currentTab)) {
    setTimelineTab("allTra")
    return
  }
  if (currentTab === "allTra") return
  const want = TRA_AXIS_KEYS[axisIdx]
  if (currentTab !== want) setTimelineTab(want)
}

function syncTimelineTabForMorphDrag(currentTab: string, setTimelineTab: (t: string) => void) {
  if (currentTab !== "morph") setTimelineTab("morph")
}

function sampleBoneKeyframe(clip: AnimationClip | null, boneName: string, frame: number) {
  if (!clip) return null
  const track = clip.boneTracks.get(boneName)
  if (!track?.length) return null
  const f = Math.round(frame)
  let kf = track[0]
  for (const k of track) {
    if (k.frame <= f) kf = k
    else break
  }
  return kf
}

function findKeyframeAt(clip: AnimationClip, bone: string, frame: number): BoneKeyframe | null {
  return clip.boneTracks.get(bone)?.find((k) => k.frame === frame) ?? null
}

type IpTab = "rot" | "tx" | "ty" | "tz"

function interpolationPairFromTab(kf: BoneKeyframe, tab: IpTab): [CurvePoint, CurvePoint] | null {
  let row: { x: number; y: number }[] | undefined
  if (tab === "rot") row = kf.interpolation.rotation
  else if (tab === "tx") row = kf.interpolation.translationX
  else if (tab === "ty") row = kf.interpolation.translationY
  else row = kf.interpolation.translationZ
  if (!row || row.length < 2) return null
  return [{ x: row[0].x, y: row[0].y }, { x: row[1].x, y: row[1].y }]
}

function mergeInterpolation(kf: BoneKeyframe, tab: IpTab, p1: CurvePoint, p2: CurvePoint): BoneInterpolation {
  const ip = cloneBoneInterpolation(kf.interpolation)
  const pair = [
    { x: p1.x, y: p1.y },
    { x: p2.x, y: p2.y },
  ]
  if (tab === "rot") ip.rotation = pair
  else if (tab === "tx") ip.translationX = pair
  else if (tab === "ty") ip.translationY = pair
  else ip.translationZ = pair
  return ip
}

/** Mutate the keyframe in the shared track (engine clip shares this array) then shallow-copy clip for React. */
function patchKeyframeAt(
  clip: AnimationClip,
  bone: string,
  keyFrame: number,
  patch: (kf: BoneKeyframe) => void,
): AnimationClip {
  const track = clip.boneTracks.get(bone)
  if (!track) return clip
  const i = track.findIndex((k) => k.frame === keyFrame)
  if (i < 0) return clip
  patch(track[i])
  return { ...clip, boneTracks: new Map(clip.boneTracks) }
}

function interpolationTemplateForChannel(tab: IpTab): [CurvePoint, CurvePoint] {
  const ip = VMD_LINEAR_DEFAULT_IP
  if (tab === "rot") return [{ x: ip.rotation[0].x, y: ip.rotation[0].y }, { x: ip.rotation[1].x, y: ip.rotation[1].y }]
  if (tab === "tx")
    return [{ x: ip.translationX[0].x, y: ip.translationX[0].y }, { x: ip.translationX[1].x, y: ip.translationX[1].y }]
  if (tab === "ty")
    return [{ x: ip.translationY[0].x, y: ip.translationY[0].y }, { x: ip.translationY[1].x, y: ip.translationY[1].y }]
  return [{ x: ip.translationZ[0].x, y: ip.translationZ[0].y }, { x: ip.translationZ[1].x, y: ip.translationZ[1].y }]
}

interface PropertiesInspectorProps {
  morphWeight: number | null
  modelRef: RefObject<Model | null>
  livePose: {
    euler: { x: number; y: number; z: number }
    translation: Vec3
  } | null
  onInsertKeyframeAtPlayhead: () => void
  onDeleteSelectedKeyframes: () => void
  timelineTab: string
  setTimelineTab: (tab: string) => void
  clipVersion: number
}

export const PropertiesInspector = memo(function PropertiesInspector({
  morphWeight,
  modelRef,
  livePose,
  onInsertKeyframeAtPlayhead,
  onDeleteSelectedKeyframes,
  timelineTab,
  setTimelineTab,
  clipVersion,
}: PropertiesInspectorProps) {
  const clip = useStudioSelector((s) => s.clip)
  const selectedBone = useStudioSelector((s) => s.selectedBone)
  const selectedMorph = useStudioSelector((s) => s.selectedMorph)
  const selectedKeyframes = useStudioSelector((s) => s.selectedKeyframes)
  const { commit } = useStudioActions()
  /** Read-only ref to the playhead. Subscribing here would re-render Properties
   *  every rAF tick during playback; instead we read .current inside callbacks
   *  and let the small <PlayheadFrameLabel/> + <InterpolationSection/> children
   *  subscribe for the handful of visible bits that actually need to update. */
  const playbackFrameRef = usePlaybackFrameRef()
  const singleSel = selectedKeyframes.length === 1 ? selectedKeyframes[0] : null
  const multiSel = selectedKeyframes.length > 1

  const canDelete = clip && singleSel !== null
  const canInsert = !!(clip && (selectedBone || selectedMorph))

  const showBoneStats = !!(selectedBone && clip && !selectedMorph && !multiSel)

  const ROT_RANGE = { min: -180, max: 180 }
  const TRA_RANGE = { min: -10, max: 10 }

  // ─── Slider preview / commit split ──────────────────────────────────
  //     `*Preview` fires every drag tick: mutates the clip's keyframe in
  //     place (the engine shares the same track arrays), reloads + seeks so
  //     the 3D viewport reflects the new pose, and skips `commit()` so we
  //     don't re-render Timeline / Properties / invalidate caches per frame.
  //     `*Commit` fires once on pointer-up: commits a new clip reference,
  //     which cascades through the studio store, landing the change in
  //     undo/redo and causing EngineBridge to reupload the clip once.
  //     A ref tracks whether the current drag has actually touched the clip
  //     so the commit is a no-op when a user just clicks the thumb.
  const dragDirtyRef = useRef(false)

  const applyRotationAxis = useCallback(
    (axisIdx: 0 | 1 | 2, v: number, mode: "preview" | "commit") => {
      const model = modelRef.current
      if (!selectedBone || !clip || !model) return
      const cf = playbackFrameRef.current
      const frame = Math.round(Math.max(0, Math.min(clip.frameCount, cf)))
      const atKey = findKeyframeAt(clip, selectedBone, frame)
      let q: Quat
      if (atKey) {
        const e = quatToEuler(atKey.rotation)
        const next = axisIdx === 0 ? { ...e, x: v } : axisIdx === 1 ? { ...e, y: v } : { ...e, z: v }
        q = eulerToQuat(next.x, next.y, next.z)
        // Mutate in place — engine's clip shares the same keyframe objects.
        atKey.rotation = q
      } else {
        // Need pose to create the new keyframe — seek first.
        model.loadClip(STUDIO_ANIM_NAME, clip)
        model.seek(Math.max(0, cf) / 30)
        const pose = readLocalPoseAfterSeek(model, selectedBone)
        if (!pose) return
        const e = quatToEuler(pose.rotation)
        const next = axisIdx === 0 ? { ...e, x: v } : axisIdx === 1 ? { ...e, y: v } : { ...e, z: v }
        q = eulerToQuat(next.x, next.y, next.z)
        // Insert by mutating the track array in place.
        const track = clip.boneTracks.get(selectedBone) ?? []
        if (!clip.boneTracks.has(selectedBone)) clip.boneTracks.set(selectedBone, track)
        track.push({
          boneName: selectedBone,
          frame,
          rotation: q,
          translation: pose.translation,
          interpolation: interpolationTemplateForFrame(track, frame),
        })
        track.sort((a, b) => a.frame - b.frame)
      }
      // Push to engine for viewport update.
      model.loadClip(STUDIO_ANIM_NAME, clip)
      model.seek(Math.max(0, cf) / 30)
      if (mode === "preview") {
        dragDirtyRef.current = true
      } else {
        // Clone for React notification + undo/redo snapshot.
        commit({ ...clip, boneTracks: new Map(clip.boneTracks) })
        dragDirtyRef.current = false
      }
    },
    [selectedBone, clip, commit, playbackFrameRef, modelRef],
  )

  const applyTranslationAxis = useCallback(
    (axisIdx: 0 | 1 | 2, v: number, mode: "preview" | "commit") => {
      const model = modelRef.current
      if (!selectedBone || !clip || !model) return
      const cf = playbackFrameRef.current
      const frame = Math.round(Math.max(0, Math.min(clip.frameCount, cf)))
      const atKey = findKeyframeAt(clip, selectedBone, frame)
      if (atKey) {
        const t = atKey.translation
        atKey.translation =
          axisIdx === 0 ? new Vec3(v, t.y, t.z) : axisIdx === 1 ? new Vec3(t.x, v, t.z) : new Vec3(t.x, t.y, v)
      } else {
        model.loadClip(STUDIO_ANIM_NAME, clip)
        model.seek(Math.max(0, cf) / 30)
        const pose = readLocalPoseAfterSeek(model, selectedBone)
        if (!pose) return
        const t = pose.translation
        const nextT =
          axisIdx === 0 ? new Vec3(v, t.y, t.z) : axisIdx === 1 ? new Vec3(t.x, v, t.z) : new Vec3(t.x, t.y, v)
        const track = clip.boneTracks.get(selectedBone) ?? []
        if (!clip.boneTracks.has(selectedBone)) clip.boneTracks.set(selectedBone, track)
        track.push({
          boneName: selectedBone,
          frame,
          rotation: pose.rotation,
          translation: nextT,
          interpolation: interpolationTemplateForFrame(track, frame),
        })
        track.sort((a, b) => a.frame - b.frame)
      }
      model.loadClip(STUDIO_ANIM_NAME, clip)
      model.seek(Math.max(0, cf) / 30)
      if (mode === "preview") {
        dragDirtyRef.current = true
      } else {
        commit({ ...clip, boneTracks: new Map(clip.boneTracks) })
        dragDirtyRef.current = false
      }
    },
    [selectedBone, clip, commit, playbackFrameRef, modelRef],
  )

  const applyMorphWeight = useCallback(
    (w: number, mode: "preview" | "commit") => {
      if (!selectedMorph || !clip) return
      const frame = Math.round(Math.max(0, Math.min(clip.frameCount, playbackFrameRef.current)))
      const track = clip.morphTracks.get(selectedMorph) ?? []
      if (!clip.morphTracks.has(selectedMorph)) clip.morphTracks.set(selectedMorph, track)
      const existing = track.find((k) => k.frame === frame)
      if (existing) {
        existing.weight = w
      } else {
        track.push({ morphName: selectedMorph, frame, weight: w })
        track.sort((a, b) => a.frame - b.frame)
      }
      const model = modelRef.current
      if (model) {
        model.loadClip(STUDIO_ANIM_NAME, clip)
        model.seek(Math.max(0, playbackFrameRef.current) / 30)
      }
      syncTimelineTabForMorphDrag(timelineTab, setTimelineTab)
      if (mode === "preview") {
        dragDirtyRef.current = true
      } else {
        commit({ ...clip, morphTracks: new Map(clip.morphTracks) })
        dragDirtyRef.current = false
      }
    },
    [selectedMorph, clip, commit, timelineTab, setTimelineTab, playbackFrameRef, modelRef],
  )

  return (
    <div className="space-y-0 text-[11px] leading-relaxed text-inherit">

      {/* ─── Bone: sliders always; clip write updates key at playhead or inserts one ─── */}
      {showBoneStats && selectedBone ? (
        <section className="border-b border-border pb-3">
          <div className="mb-2 flex items-start justify-between gap-2">
            <div>
              {(() => {
                const { title, subtitle } = boneTitleSubtitle(selectedBone)
                return (
                  <>
                    <div className="text-xs font-semibold text-inherit">{title}</div>
                    {subtitle ? <div className="text-[10px] text-muted-foreground">{subtitle}</div> : null}
                  </>
                )
              })()}
            </div>
            <PlayheadFrameLabel frameCount={clip?.frameCount ?? null} />
          </div>

          <div className="mb-2 text-[10px] font-medium uppercase tracking-[0.06em] text-muted-foreground">Rotation (°)</div>
          {livePose ? (
            ROT_CHANNELS.map((ch, i) => (
              <AxisSliderRow
                key={ch.key}
                axis={["X", "Y", "Z"][i] as string}
                color={ch.color}
                value={[livePose.euler.x, livePose.euler.y, livePose.euler.z][i]}
                min={ROT_RANGE.min}
                max={ROT_RANGE.max}
                decimals={2}
                disabled={!clip}
                onChange={(v) => {
                  syncTimelineTabForRotationDrag(timelineTab, i as 0 | 1 | 2, setTimelineTab)
                  applyRotationAxis(i as 0 | 1 | 2, v, "preview")
                }}
                onCommit={(v) => applyRotationAxis(i as 0 | 1 | 2, v, "commit")}
              />
            ))
          ) : (
            <div className="text-[11px] text-muted-foreground">—</div>
          )}

          <div className="mb-2 mt-3 text-[10px] font-medium uppercase tracking-[0.06em] text-muted-foreground">
            Translation
          </div>
          {livePose ? (
            TRA_CHANNELS.map((ch, i) => (
              <AxisSliderRow
                key={ch.key}
                axis={["X", "Y", "Z"][i] as string}
                color={ch.color}
                value={[livePose.translation.x, livePose.translation.y, livePose.translation.z][i]}
                min={TRA_RANGE.min}
                max={TRA_RANGE.max}
                decimals={3}
                disabled={!clip}
                onChange={(v) => {
                  syncTimelineTabForTranslationDrag(timelineTab, i as 0 | 1 | 2, setTimelineTab)
                  applyTranslationAxis(i as 0 | 1 | 2, v, "preview")
                }}
                onCommit={(v) => applyTranslationAxis(i as 0 | 1 | 2, v, "commit")}
              />
            ))
          ) : (
            <div className="text-[11px] text-muted-foreground">—</div>
          )}

          <InterpolationSection
            clip={clip}
            selectedBone={selectedBone}
            commit={commit}
            clipVersion={clipVersion}
          />
        </section>
      ) : null}

      {selectedMorph && clip && !multiSel ? (
        <section className="border-b border-border pb-3">
          <div className="mb-2 flex items-start justify-between gap-2">
            <div>
              <div className="text-xs font-semibold text-inherit">{selectedMorph}</div>
            </div>
            <PlayheadFrameLabel frameCount={clip?.frameCount ?? null} />
          </div>
          <div className="mb-2 text-[10px] font-medium uppercase tracking-[0.06em] text-muted-foreground">Weight</div>
          <AxisSliderRow
            axis="W"
            color="#c084fc"
            value={morphWeight ?? 0}
            min={0}
            max={1}
            decimals={2}
            disabled={!clip}
            onChange={(v) => applyMorphWeight(v, "preview")}
            onCommit={(v) => applyMorphWeight(v, "commit")}
          />
        </section>
      ) : null}

      <section className="space-y-2 pt-2.5">
        <div className="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">Operations</div>
        <div className="flex flex-wrap gap-1.5">
          <Button
            type="button"
            variant="secondary"
            size="xs"
            className="h-7 px-2 text-[11px]"
            disabled={!canInsert}
            onClick={onInsertKeyframeAtPlayhead}
          >
            Insert key
          </Button>
          <Button
            type="button"
            variant="secondary"
            size="xs"
            className="h-7 px-2 text-[11px]"
            disabled={!canDelete}
            onClick={onDeleteSelectedKeyframes}
          >
            Delete key
          </Button>
        </div>
      </section>
    </div>
  )
})

/** Subscribes to the playhead so the parent <PropertiesInspector/> doesn't have to. */
function PlayheadFrameLabel({ frameCount }: { frameCount: number | null }) {
  const currentFrame = usePlaybackSelector((s) => s.currentFrame)
  return (
    <div className="shrink-0 font-mono text-[10px] tabular-nums text-muted-foreground">
      F {Math.round(currentFrame)}
      {frameCount != null ? ` / ${frameCount}` : ""}
    </div>
  )
}

/** Owns the interpolation tab + curve preview. Subscribes to currentFrame
 *  internally so the parent inspector (and its sliders) don't re-render every
 *  rAF tick during playback. */
function InterpolationSection({
  clip,
  selectedBone,
  commit,
  clipVersion,
}: {
  clip: AnimationClip | null
  selectedBone: string | null
  commit: ReturnType<typeof useStudioActions>["commit"]
  clipVersion: number
}) {
  const currentFrame = usePlaybackSelector((s) => s.currentFrame)
  const [ipTab, setIpTab] = useState<IpTab>("rot")

  // Reset interpolation tab when a new clip is loaded.
  const clipVersionRef = useRef(clipVersion)
  useEffect(() => {
    if (clipVersionRef.current === clipVersion) return
    clipVersionRef.current = clipVersion
    setIpTab("rot")
  }, [clipVersion])

  const kfSample = clip && selectedBone ? sampleBoneKeyframe(clip, selectedBone, currentFrame) : null
  const canEditIp = !!(clip && selectedBone && kfSample)

  const ipPair = useMemo(() => {
    if (kfSample) {
      const p = interpolationPairFromTab(kfSample, ipTab)
      if (p) return p
    }
    return interpolationTemplateForChannel(ipTab)
  }, [kfSample, ipTab])

  const applyInterpolation = useCallback(
    (p1: CurvePoint, p2: CurvePoint) => {
      if (!clip || !selectedBone || !kfSample) return
      const keyFrame = kfSample.frame
      commit(
        patchKeyframeAt(clip, selectedBone, keyFrame, (kf) => {
          kf.interpolation = mergeInterpolation(kf, ipTab, p1, p2)
        }),
      )
    },
    [clip, selectedBone, ipTab, kfSample, commit],
  )

  return (
    <>
      <div className="mb-2 mt-3 text-[10px] font-medium uppercase tracking-[0.06em] text-muted-foreground">
        Interpolation
      </div>
      <div className="mb-1.5 flex flex-wrap gap-0.5">
        {(
          [
            ["rot", "Rotation"],
            ["tx", "Trans X"],
            ["ty", "Trans Y"],
            ["tz", "Trans Z"],
          ] as const
        ).map(([key, label]) => (
          <Button
            key={key}
            type="button"
            variant={ipTab === key ? "secondary" : "ghost"}
            size="xs"
            disabled={!canEditIp}
            onClick={() => setIpTab(key)}
            className="h-6 px-2 text-[9px] font-medium"
          >
            {label}
          </Button>
        ))}
      </div>
      <div className="flex items-stretch gap-1.5" style={{ height: 164 }}>
        <InterpolationCurveEditor
          p1={ipPair[0]}
          p2={ipPair[1]}
          disabled={!canEditIp}
          onChange={applyInterpolation}
        />
        <div className="flex min-w-0 flex-1 flex-col gap-1">
          {PRESETS.map((pr) => {
            const active =
              pr.p1.x === ipPair[0].x &&
              pr.p1.y === ipPair[0].y &&
              pr.p2.x === ipPair[1].x &&
              pr.p2.y === ipPair[1].y
            return (
              <Button
                key={pr.label}
                type="button"
                variant={active ? "secondary" : "outline"}
                size="xs"
                disabled={!canEditIp}
                onClick={() => applyInterpolation(pr.p1, pr.p2)}
                className={cn(
                  "h-auto min-h-0 flex-1 truncate px-1 py-0.5 text-center text-[9.5px] font-medium leading-tight",
                  active
                    ? "border-primary/30 text-primary"
                    : "text-muted-foreground hover:border-primary/25 hover:text-accent-foreground",
                )}
              >
                {pr.label}
              </Button>
            )
          })}
        </div>
      </div>
    </>
  )
}
