"use client"

import type { Dispatch, RefObject, SetStateAction } from "react"
import { useCallback, useMemo, useState } from "react"
import type { AnimationClip, BoneInterpolation, BoneKeyframe, Model } from "reze-engine"
import { Quat, Vec3 } from "reze-engine"
import { Button } from "@/components/ui/button"
import type { SelectedKeyframe } from "@/components/timeline"
import {
  boneTitleSubtitle,
  eulerToQuat,
  quatToEuler,
  ROT_CHANNELS,
  TRA_CHANNELS,
} from "@/lib/animation"
import {
  cloneBoneInterpolation,
  readLocalPoseAfterSeek,
  upsertBoneKeyframeAtFrame,
  VMD_LINEAR_DEFAULT_IP,
} from "@/lib/keyframe-insert"
import { AxisSliderRow } from "@/components/axis-slider-row"
import { InterpolationCurveEditor, PRESETS, type CurvePoint } from "@/components/interpolation-curve-editor"
import { cn } from "@/lib/utils"

/** Must match `loadAnimation` name in app/page (engine clip vs React state). */
const STUDIO_ANIM_NAME = "studio"

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
  clip: AnimationClip | null
  currentFrame: number
  activeBone: string | null
  activeMorph: string | null
  morphWeight: number | null
  selectedKeyframes: SelectedKeyframe[]
  modelRef: RefObject<Model | null>
  setClip: Dispatch<SetStateAction<AnimationClip | null>>
  livePose: {
    euler: { x: number; y: number; z: number }
    translation: Vec3
  } | null
  onInsertKeyframeAtPlayhead: () => void
  onDeleteSelectedKeyframes: () => void
}

export function PropertiesInspector({
  clip,
  currentFrame,
  activeBone,
  activeMorph,
  morphWeight,
  selectedKeyframes,
  modelRef,
  setClip,
  livePose,
  onInsertKeyframeAtPlayhead,
  onDeleteSelectedKeyframes,
}: PropertiesInspectorProps) {
  const fPlay = Math.round(currentFrame)
  const singleSel = selectedKeyframes.length === 1 ? selectedKeyframes[0] : null
  const multiSel = selectedKeyframes.length > 1

  const canDelete = clip && singleSel !== null
  const canInsert = !!(clip && activeBone && !activeMorph)

  const [ipTab, setIpTab] = useState<IpTab>("rot")

  /** Last key at or before playhead — owns outgoing handles to the next key. */
  const kfSample = clip && activeBone ? sampleBoneKeyframe(clip, activeBone, currentFrame) : null

  const ipPair = useMemo(() => {
    if (kfSample) {
      const p = interpolationPairFromTab(kfSample, ipTab)
      if (p) return p
    }
    return interpolationTemplateForChannel(ipTab)
  }, [clip, kfSample, ipTab])

  const applyInterpolation = useCallback(
    (p1: CurvePoint, p2: CurvePoint) => {
      if (!clip || !activeBone || !kfSample) return
      const keyFrame = kfSample.frame
      setClip(
        patchKeyframeAt(clip, activeBone, keyFrame, (kf) => {
          kf.interpolation = mergeInterpolation(kf, ipTab, p1, p2)
        }),
      )
    },
    [clip, activeBone, ipTab, kfSample, setClip],
  )

  const showBoneStats = !!(activeBone && clip && !activeMorph && !multiSel)
  const canEditIp = !!(clip && activeBone && kfSample)

  const ROT_RANGE = { min: -180, max: 180 }
  const TRA_RANGE = { min: -5, max: 5 }

  const setRotationAxis = useCallback(
    (axisIdx: 0 | 1 | 2, v: number) => {
      const model = modelRef.current
      if (!activeBone || !clip || !model) return
      const frame = Math.round(Math.max(0, Math.min(clip.frameCount, currentFrame)))
      const atKey = findKeyframeAt(clip, activeBone, frame)
      model.loadAnimation(STUDIO_ANIM_NAME, clip)
      model.seek(Math.max(0, currentFrame) / 30)
      let q: Quat
      if (atKey) {
        // Other euler components from stored key, not blended pose (avoids sibling axes drifting)
        const e = quatToEuler(atKey.rotation)
        const next = axisIdx === 0 ? { ...e, x: v } : axisIdx === 1 ? { ...e, y: v } : { ...e, z: v }
        q = eulerToQuat(next.x, next.y, next.z)
        setClip(patchKeyframeAt(clip, activeBone, frame, (kf) => { kf.rotation = q }))
      } else {
        const pose = readLocalPoseAfterSeek(model, activeBone)
        if (!pose) return
        const e = quatToEuler(pose.rotation)
        const next = axisIdx === 0 ? { ...e, x: v } : axisIdx === 1 ? { ...e, y: v } : { ...e, z: v }
        q = eulerToQuat(next.x, next.y, next.z)
        setClip(upsertBoneKeyframeAtFrame(clip, activeBone, frame, q, pose.translation))
      }
    },
    [activeBone, clip, currentFrame, setClip],
  )

  const setTranslationAxis = useCallback(
    (axisIdx: 0 | 1 | 2, v: number) => {
      const model = modelRef.current
      if (!activeBone || !clip || !model) return
      const frame = Math.round(Math.max(0, Math.min(clip.frameCount, currentFrame)))
      const atKey = findKeyframeAt(clip, activeBone, frame)
      model.loadAnimation(STUDIO_ANIM_NAME, clip)
      model.seek(Math.max(0, currentFrame) / 30)
      if (atKey) {
        const t = atKey.translation
        const next =
          axisIdx === 0 ? new Vec3(v, t.y, t.z) : axisIdx === 1 ? new Vec3(t.x, v, t.z) : new Vec3(t.x, t.y, v)
        setClip(patchKeyframeAt(clip, activeBone, frame, (kf) => { kf.translation = next }))
      } else {
        const pose = readLocalPoseAfterSeek(model, activeBone)
        if (!pose) return
        const t = pose.translation
        const next =
          axisIdx === 0 ? new Vec3(v, t.y, t.z) : axisIdx === 1 ? new Vec3(t.x, v, t.z) : new Vec3(t.x, t.y, v)
        setClip(upsertBoneKeyframeAtFrame(clip, activeBone, frame, pose.rotation, next))
      }
    },
    [activeBone, clip, currentFrame, setClip],
  )

  return (
    <div className="space-y-0 text-[11px] leading-relaxed text-inherit">

      {/* ─── Bone: sliders always; clip write updates key at playhead or inserts one ─── */}
      {showBoneStats && activeBone ? (
        <section className="border-b border-border pb-3">
          <div className="mb-2 flex items-start justify-between gap-2">
            <div>
              {(() => {
                const { title, subtitle } = boneTitleSubtitle(activeBone)
                return (
                  <>
                    <div className="text-xs font-semibold text-inherit">{title}</div>
                    {subtitle ? <div className="text-[10px] text-muted-foreground">{subtitle}</div> : null}
                  </>
                )
              })()}
            </div>
            <div className="shrink-0 font-mono text-[10px] tabular-nums text-muted-foreground">
              F {fPlay}
              {clip ? ` / ${clip.frameCount}` : ""}
            </div>
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
                onChange={(v) => setRotationAxis(i as 0 | 1 | 2, v)}
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
                onChange={(v) => setTranslationAxis(i as 0 | 1 | 2, v)}
              />
            ))
          ) : (
            <div className="text-[11px] text-muted-foreground">—</div>
          )}

          <div className="mb-2 mt-3 text-[10px] font-medium uppercase tracking-[0.06em] text-muted-foreground">
            Interpolation
          </div>
          <div className="mb-1.5 flex flex-wrap gap-0.5">
            {(
              [
                ["rot", "Rot"],
                ["tx", "Tra X"],
                ["ty", "Tra Y"],
                ["tz", "Tra Z"],
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
        </section>
      ) : null}

      {activeMorph && clip && !multiSel ? (
        <section className="space-y-2.5 border-b border-border pb-3">
          <div className="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">Playhead · morph</div>
          <div className="font-mono text-[11px] text-primary">{activeMorph}</div>
          <div className="font-mono tabular-nums text-[11px] text-inherit">
            {morphWeight !== null ? morphWeight.toFixed(2) : "—"}
          </div>
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
}
