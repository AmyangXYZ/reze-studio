import type { AnimationClip, BoneKeyframe, ControlPoint, MorphKeyframe } from "reze-engine"
import { Quat, Vec3 } from "reze-engine"
import { cloneBoneInterpolation, DEFAULT_STUDIO_CLIP_FRAMES } from "@/lib/utils"

// ─── Clip arrangement data model ─────────────────────────────────────────
// A Project arranges library clips on tracks. The engine never sees tracks:
// the studio bakes the arrangement into one flat AnimationClip per change
// and uploads it via the existing `model.loadClip` path.

export type ClipId = string

export interface LibraryClip {
  id: ClipId
  name: string
  clip: AnimationClip
}

export interface ClipPlacement {
  id: string
  clipId: ClipId
  startFrame: number
  /** Output-frames-per-source-frame (2 = twice as long / half speed). */
  timeScale?: number
  /** Trimmed visible length in output frames; undefined = full scaled clip. */
  length?: number
}

export interface Track {
  id: string
  name: string
  mute: boolean
  solo: boolean
  placements: ClipPlacement[]
}

export function newId(): string {
  return crypto.randomUUID()
}

export function emptyTrack(name: string): Track {
  return { id: newId(), name, mute: false, solo: false, placements: [] }
}

/** Full scaled length of a clip in output frames (before trim). */
export function scaledClipFrames(clip: AnimationClip, timeScale: number | undefined): number {
  return Math.max(1, Math.round(clip.frameCount * (timeScale ?? 1)))
}

/** Visible length of a placement in output frames (trim applied). */
export function placementFrames(p: ClipPlacement, clip: AnimationClip): number {
  const full = scaledClipFrames(clip, p.timeScale)
  return p.length != null ? Math.max(1, Math.min(p.length, full)) : full
}

export function findPlacement(
  tracks: Track[],
  placementId: string,
): { track: Track; placement: ClipPlacement } | null {
  for (const track of tracks) {
    const placement = track.placements.find((p) => p.id === placementId)
    if (placement) return { track, placement }
  }
  return null
}

/** Last frame covered by any placement on any track (0 when empty). */
export function arrangementEndFrame(tracks: Track[], library: LibraryClip[]): number {
  const byId = new Map(library.map((c) => [c.id, c]))
  let end = 0
  for (const track of tracks) {
    for (const p of track.placements) {
      const lib = byId.get(p.clipId)
      if (!lib) continue
      end = Math.max(end, p.startFrame + placementFrames(p, lib.clip))
    }
  }
  return end
}

// ─── Bake: arrangement → flat clip ───────────────────────────────────────
// Implicit-by-keyframe-presence masking: for each bone (and morph), walk
// tracks top-to-bottom and the first active track whose placed clips contain
// keyframes for that name owns it entirely — lower tracks' data for the same
// name is ignored. This is why face-only and body-only VMDs compose without
// any mask UI. Within the owning track, placements merge by output frame
// (later placements in the array win on collisions).
export function bakeTracks(tracks: Track[], library: LibraryClip[]): AnimationClip {
  const byId = new Map(library.map((c) => [c.id, c]))
  const anySolo = tracks.some((t) => t.solo)
  const active = tracks.filter((t) => (anySolo ? t.solo : !t.mute))

  const boneOwner = new Map<string, Track>()
  const morphOwner = new Map<string, Track>()
  for (const track of active) {
    for (const p of track.placements) {
      const lib = byId.get(p.clipId)
      if (!lib) continue
      for (const [bone, kfs] of lib.clip.boneTracks) {
        if (kfs.length && !boneOwner.has(bone)) boneOwner.set(bone, track)
      }
      for (const [morph, kfs] of lib.clip.morphTracks) {
        if (kfs.length && !morphOwner.has(morph)) morphOwner.set(morph, track)
      }
    }
  }

  const boneTracks = new Map<string, BoneKeyframe[]>()
  const morphTracks = new Map<string, MorphKeyframe[]>()
  let frameCount = 0

  for (const track of active) {
    for (const p of track.placements) {
      const lib = byId.get(p.clipId)
      if (!lib) continue
      const scale = p.timeScale ?? 1
      const limit = placementFrames(p, lib.clip)
      frameCount = Math.max(frameCount, p.startFrame + limit)
      for (const [bone, kfs] of lib.clip.boneTracks) {
        if (boneOwner.get(bone) !== track) continue
        let out = boneTracks.get(bone)
        if (!out) {
          out = []
          boneTracks.set(bone, out)
        }
        for (const k of kfs) {
          const local = Math.round(k.frame * scale)
          if (local > limit) continue
          out.push({
            boneName: k.boneName,
            frame: p.startFrame + local,
            rotation: k.rotation.clone(),
            translation: new Vec3(k.translation.x, k.translation.y, k.translation.z),
            interpolation: cloneBoneInterpolation(k.interpolation),
          })
        }
      }
      for (const [morph, kfs] of lib.clip.morphTracks) {
        if (morphOwner.get(morph) !== track) continue
        let out = morphTracks.get(morph)
        if (!out) {
          out = []
          morphTracks.set(morph, out)
        }
        for (const k of kfs) {
          const local = Math.round(k.frame * scale)
          if (local > limit) continue
          out.push({ morphName: k.morphName, frame: p.startFrame + local, weight: k.weight })
        }
      }
    }
  }

  // Sort by frame; on collisions keep the later placement's key (Array.sort
  // is stable, so push order — placement order within the track — survives).
  for (const [bone, arr] of boneTracks) boneTracks.set(bone, dedupeKeepLast(arr))
  for (const [morph, arr] of morphTracks) morphTracks.set(morph, dedupeKeepLast(arr))

  const empty = boneTracks.size === 0 && morphTracks.size === 0
  return {
    boneTracks,
    morphTracks,
    frameCount: empty ? Math.max(frameCount, DEFAULT_STUDIO_CLIP_FRAMES) : Math.max(1, frameCount),
  }
}

function dedupeKeepLast<T extends { frame: number }>(arr: T[]): T[] {
  arr.sort((a, b) => a.frame - b.frame)
  const out: T[] = []
  for (const k of arr) {
    if (out.length && out[out.length - 1].frame === k.frame) out[out.length - 1] = k
    else out.push(k)
  }
  return out
}

// ─── Project (de)serialization — .rsproj JSON ───────────────────────────
// Quat/Vec3 are class instances and clip tracks are Maps, so clips need an
// explicit encoding. The PMX itself is too large to embed — the project
// stores the model by name only and the user re-picks it if missing.

type EncodedIp = [number, number, number, number]
interface EncodedBoneKey {
  f: number
  r: [number, number, number, number]
  t: [number, number, number]
  i: { r: EncodedIp; x: EncodedIp; y: EncodedIp; z: EncodedIp }
}
interface EncodedMorphKey {
  f: number
  w: number
}
interface EncodedClip {
  frameCount: number
  boneTracks: [string, EncodedBoneKey[]][]
  morphTracks: [string, EncodedMorphKey[]][]
}
interface EncodedLibraryClip {
  id: ClipId
  name: string
  clip: EncodedClip
}

export interface ProjectFile {
  app: "reze-studio"
  type: "project"
  version: 1
  name: string
  modelRef: { name: string }
  library: EncodedLibraryClip[]
  tracks: Track[]
  viewState?: { activeClipId: ClipId | null }
}

export interface ProjectSnapshot {
  name: string
  modelRef: { name: string }
  library: LibraryClip[]
  tracks: Track[]
  activeClipId: ClipId | null
}

function encodeIp(cp: ControlPoint[]): EncodedIp {
  return [cp[0]?.x ?? 20, cp[0]?.y ?? 20, cp[1]?.x ?? 107, cp[1]?.y ?? 107]
}

function decodeIp(e: EncodedIp): ControlPoint[] {
  return [
    { x: e[0], y: e[1] },
    { x: e[2], y: e[3] },
  ]
}

export function encodeClip(clip: AnimationClip): EncodedClip {
  const boneTracks: [string, EncodedBoneKey[]][] = []
  for (const [name, track] of clip.boneTracks) {
    boneTracks.push([
      name,
      track.map((k) => ({
        f: k.frame,
        r: [k.rotation.x, k.rotation.y, k.rotation.z, k.rotation.w],
        t: [k.translation.x, k.translation.y, k.translation.z],
        i: {
          r: encodeIp(k.interpolation.rotation),
          x: encodeIp(k.interpolation.translationX),
          y: encodeIp(k.interpolation.translationY),
          z: encodeIp(k.interpolation.translationZ),
        },
      })),
    ])
  }
  const morphTracks: [string, EncodedMorphKey[]][] = []
  for (const [name, track] of clip.morphTracks) {
    morphTracks.push([name, track.map((k) => ({ f: k.frame, w: k.weight }))])
  }
  return { frameCount: clip.frameCount, boneTracks, morphTracks }
}

export function decodeClip(e: EncodedClip): AnimationClip {
  const boneTracks = new Map<string, BoneKeyframe[]>()
  for (const [name, track] of e.boneTracks) {
    boneTracks.set(
      name,
      track.map((k) => ({
        boneName: name,
        frame: k.f,
        rotation: new Quat(k.r[0], k.r[1], k.r[2], k.r[3]),
        translation: new Vec3(k.t[0], k.t[1], k.t[2]),
        interpolation: {
          rotation: decodeIp(k.i.r),
          translationX: decodeIp(k.i.x),
          translationY: decodeIp(k.i.y),
          translationZ: decodeIp(k.i.z),
        },
      })),
    )
  }
  const morphTracks = new Map<string, MorphKeyframe[]>()
  for (const [name, track] of e.morphTracks) {
    morphTracks.set(
      name,
      track.map((k) => ({ morphName: name, frame: k.f, weight: k.w })),
    )
  }
  return { boneTracks, morphTracks, frameCount: Math.max(1, e.frameCount) }
}

export function encodeProject(snapshot: ProjectSnapshot): ProjectFile {
  return {
    app: "reze-studio",
    type: "project",
    version: 1,
    name: snapshot.name,
    modelRef: snapshot.modelRef,
    library: snapshot.library.map((c) => ({ id: c.id, name: c.name, clip: encodeClip(c.clip) })),
    tracks: snapshot.tracks.map((t) => ({
      id: t.id,
      name: t.name,
      mute: t.mute,
      solo: t.solo,
      placements: t.placements.map((p) => ({ ...p })),
    })),
    viewState: { activeClipId: snapshot.activeClipId },
  }
}

export function decodeProject(json: unknown): ProjectSnapshot {
  const f = json as Partial<ProjectFile> | null
  if (!f || f.app !== "reze-studio" || f.type !== "project" || !Array.isArray(f.library) || !Array.isArray(f.tracks)) {
    throw new Error("Not a Reze Studio project file.")
  }
  if (f.version !== 1) throw new Error(`Unsupported project version ${String(f.version)}.`)
  const library: LibraryClip[] = f.library.map((c) => ({
    id: String(c.id),
    name: String(c.name),
    clip: decodeClip(c.clip),
  }))
  const clipIds = new Set(library.map((c) => c.id))
  const tracks: Track[] = f.tracks.map((t, i) => ({
    id: String(t.id ?? newId()),
    name: String(t.name ?? `Track ${i + 1}`),
    mute: Boolean(t.mute),
    solo: Boolean(t.solo),
    placements: (Array.isArray(t.placements) ? t.placements : [])
      .filter((p) => clipIds.has(p.clipId))
      .map((p) => ({
        id: String(p.id ?? newId()),
        clipId: p.clipId,
        startFrame: Math.max(0, Math.round(Number(p.startFrame) || 0)),
        ...(p.timeScale != null ? { timeScale: Number(p.timeScale) } : {}),
        ...(p.length != null ? { length: Math.max(1, Math.round(Number(p.length))) } : {}),
      })),
  }))
  const activeClipId = f.viewState?.activeClipId != null && clipIds.has(f.viewState.activeClipId)
    ? f.viewState.activeClipId
    : (library[0]?.id ?? null)
  return {
    name: String(f.name ?? "project"),
    modelRef: { name: String(f.modelRef?.name ?? "") },
    library,
    tracks: tracks.length > 0 ? tracks : [emptyTrack("Track 1")],
    activeClipId,
  }
}
