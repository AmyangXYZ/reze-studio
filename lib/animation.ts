import { Vec3, Quat } from "reze-engine"
import type { AnimationClip, BoneKeyframe, MorphKeyframe } from "reze-engine"

export { Vec3, Quat }
export type { AnimationClip, BoneKeyframe, MorphKeyframe }

// Quat → Euler YXZ (degrees) — matches MMD convention
export function quatToEuler(q: Quat) {
  const sinX = 2 * (q.w * q.x - q.y * q.z)
  const clamped = Math.max(-1, Math.min(1, sinX))
  const x = Math.asin(clamped)
  const cosX = Math.cos(x)
  let y: number, z: number
  if (Math.abs(cosX) > 0.0001) {
    y = Math.atan2(2 * (q.w * q.y + q.x * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))
    z = Math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.x * q.x + q.z * q.z))
  } else {
    y = Math.atan2(-2 * (q.x * q.z - q.w * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    z = 0
  }
  const DEG = 180 / Math.PI
  return { x: x * DEG, y: y * DEG, z: z * DEG }
}

export function eulerToQuat(ex: number, ey: number, ez: number): Quat {
  const RAD = Math.PI / 180
  const hx = ex * RAD * 0.5,
    hy = ey * RAD * 0.5,
    hz = ez * RAD * 0.5
  const cx = Math.cos(hx),
    sx = Math.sin(hx)
  const cy = Math.cos(hy),
    sy = Math.sin(hy)
  const cz = Math.cos(hz),
    sz = Math.sin(hz)
  // YXZ order
  return new Quat(
    cy * sx * cz + sy * cx * sz,
    sy * cx * cz - cy * sx * sz,
    cy * cx * sz - sy * sx * cz,
    cy * cx * cz + sy * sx * sz,
  )
}

export function bezierY(cp0: { x: number; y: number }, cp1: { x: number; y: number }, t: number) {
  const x1 = cp0.x / 127,
    y1 = cp0.y / 127,
    x2 = cp1.x / 127,
    y2 = cp1.y / 127
  let lo = 0,
    hi = 1,
    mid = 0.5
  for (let i = 0; i < 15; i++) {
    const x = 3 * (1 - mid) ** 2 * mid * x1 + 3 * (1 - mid) * mid ** 2 * x2 + mid ** 3
    if (Math.abs(x - t) < 0.0001) break
    if (x < t) lo = mid
    else hi = mid
    mid = (lo + hi) / 2
  }
  return 3 * (1 - mid) ** 2 * mid * y1 + 3 * (1 - mid) * mid ** 2 * y2 + mid ** 3
}

// ─── Constants ───────────────────────────────────────────────────────────
export const DOPE_H = 34
/** Frame ruler height (px); keep in sync with tick/label layout in timeline canvas draw. */
export const RULER_H = 17
export const LABEL_W = 36
export const DOT_R = 3.5
export const DIAMOND = 5
export const MIN_PX = 1
export const MAX_PX = 20

/** Max zoom-out (min px/frame) for a given track width: show 0…frameCount inside [LABEL_W, width].
 *  The floor is clamped so that at minimum zoom the frame ruler ticks remain readable
 *  (at least ~1 px/frame, with label-gap logic in the canvas preventing overlap). */
export function minPxPerFrameForViewport(trackWidthPx: number, frameCount: number): number {
  if (frameCount <= 0 || trackWidthPx <= LABEL_W + 1) return MIN_PX
  const fit = (trackWidthPx - LABEL_W) / frameCount
  return Math.max(MIN_PX, Math.min(fit, MAX_PX))
}

export const C = {
  bg: "rgba(0,0,0,0)",
  curveBg: "rgba(0,0,0,0)",
  ruler: "rgba(0,0,0,0)",
  // Align with shadcn `muted-foreground` (~ oklch(0.708)) — no near-white UI text.
  rulerText: "#9ca3af",
  rulerTick: "#2a2a34",
  rulerMajor: "#3a3a48",
  grid: "#161620",
  axis: "#222233",
  axisZero: "#2c2c44",
  playhead: "#d83838",
  playheadGlow: "rgba(216,56,56,0.18)",
  diamondSel: "#5aa0f0",
  keyDotSel: "#9ca3af",
  dopeBg: "rgba(0,0,0,0)",
  dopeBorder: "#222230",
  dopeLabel: "#9ca3af",
  dopeLabelNum: "#6b7280",
  rotX: "#e25555",
  rotY: "#44bb55",
  rotZ: "#4477dd",
  traX: "#e2a055",
  traY: "#55bba0",
  traZ: "#7755dd",
  label: "#9ca3af",
  tabBg: "rgba(0,0,0,0)",
  tabActive: "#2a2a36",
  tabText: "#9ca3af",
  tabTextActive: "#9ca3af",
  toolbarOnAccent: "#0f0f12",
  border: "border",
  frameBadge: "#1a1a22",
  frameBadgeText: "#9ca3af",
  sidebarBg: "rgba(0,0,0,0)",
  sidebarGroup: "#888898",
  sidebarBone: "#666672",
  sidebarActive: "#5aa0f0",
  sidebarGroupBg: "rgba(0,0,0,0)",
  sidebarHover: "#1e1e28",
} as const

export const FONT = "'SF Mono','Cascadia Code','Fira Code','JetBrains Mono',monospace"

// ─── Bone groups ─────────────────────────────────────────────────────────
export const BONE_GROUPS: Record<string, string[] | null> = {
  "All Bones": null,
  "Upper Body": ["頭", "首", "上半身", "上半身2", "胸", "首根元", "腰"],
  "Left Arm": [
    "左肩",
    "左腕",
    "左ひじ",
    "左手首",
    "左腕捩",
    "左手捩",
    "左肩P",
    "左腕P",
    "左ひじP",
  ],
  "Right Arm": [
    "右肩",
    "右腕",
    "右ひじ",
    "右手首",
    "右腕捩",
    "右手捩",
    "右肩P",
    "右腕P",
    "右ひじP",
  ],
  "Left Hand": [
    "左手首",
    "左親指０",
    "左親指１",
    "左親指２",
    "左親指３",
    "左親指0",
    "左親指1",
    "左親指2",
    "左親指3",
    "左人指１",
    "左人指２",
    "左人指３",
    "左人指1",
    "左人指2",
    "左人指3",
    "左人差し指１",
    "左人差し指２",
    "左人差し指３",
    "左中指１",
    "左中指２",
    "左中指３",
    "左中指1",
    "左中指2",
    "左中指3",
    "左薬指１",
    "左薬指２",
    "左薬指３",
    "左薬指1",
    "左薬指2",
    "左薬指3",
    "左小指１",
    "左小指２",
    "左小指３",
    "左小指1",
    "左小指2",
    "左小指3",
    "左手先",
    "左親指先",
    "左人指先",
    "左中指先",
    "左薬指先",
    "左小指先",
  ],
  "Right Hand": [
    "右手首",
    "右親指０",
    "右親指１",
    "右親指２",
    "右親指３",
    "右親指0",
    "右親指1",
    "右親指2",
    "右親指3",
    "右人指１",
    "右人指２",
    "右人指３",
    "右人指1",
    "右人指2",
    "右人指3",
    "右人差し指１",
    "右人差し指２",
    "右人差し指３",
    "右中指１",
    "右中指２",
    "右中指３",
    "右中指1",
    "右中指2",
    "右中指3",
    "右薬指１",
    "右薬指２",
    "右薬指３",
    "右薬指1",
    "右薬指2",
    "右薬指3",
    "右小指１",
    "右小指２",
    "右小指３",
    "右小指1",
    "右小指2",
    "右小指3",
    "右手先",
    "右親指先",
    "右人指先",
    "右中指先",
    "右薬指先",
    "右小指先",
  ],
  "Lower Body": [
    "下半身",
    "左足",
    "右足",
    "左ひざ",
    "右ひざ",
    "左足首",
    "右足首",
    "左つま先",
    "右つま先",
    "センター",
    "グルーブ",
    "左足ＩＫ",
    "右足ＩＫ",
  ],
}

/** English labels for known bone names (JP + common aliases); used for `首 (Neck)` style UI. */
export const BONE_NAME_EN: Record<string, string> = {
  頭: "Head",
  首: "Neck",
  上半身: "Upper body",
  上半身2: "Upper body 2",
  胸: "Chest",
  首根元: "Neck root",
  head: "Head",
  neck: "Neck",
  "upper body": "Upper body",
  左肩: "Left shoulder",
  左腕: "Left upper arm",
  左ひじ: "Left elbow",
  左手首: "Left wrist",
  左腕捩: "Left arm twist",
  左手捩: "Left wrist twist",
  左肩P: "Left shoulder (P)",
  左腕P: "Left upper arm (P)",
  左ひじP: "Left elbow (P)",
  右肩: "Right shoulder",
  右腕: "Right upper arm",
  右ひじ: "Right elbow",
  右手首: "Right wrist",
  右腕捩: "Right arm twist",
  右手捩: "Right wrist twist",
  右肩P: "Right shoulder (P)",
  右腕P: "Right upper arm (P)",
  右ひじP: "Right elbow (P)",
  下半身: "Lower body",
  左足: "Left leg",
  右足: "Right leg",
  左ひざ: "Left knee",
  右ひざ: "Right knee",
  左足首: "Left ankle",
  右足首: "Right ankle",
  左つま先: "Left toe",
  右つま先: "Right toe",
  センター: "Center",
  グルーブ: "Groove",
  左足ＩＫ: "Left leg IK",
  右足ＩＫ: "Right leg IK",
  左足IK: "Left leg IK",
  右足IK: "Right leg IK",

  左目: "Left eye",
  右目: "Right eye",
  両目: "Both eyes",

  左親指０: "Left thumb 0",
  左親指１: "Left thumb 1",
  左親指２: "Left thumb 2",
  左親指３: "Left thumb 3",
  左人差し指１: "Left index 1",
  左人差し指２: "Left index 2",
  左人差し指３: "Left index 3",
  左人指１: "Left index 1",
  左人指２: "Left index 2",
  左人指３: "Left index 3",
  左中指１: "Left middle 1",
  左中指２: "Left middle 2",
  左中指３: "Left middle 3",
  左薬指１: "Left ring 1",
  左薬指２: "Left ring 2",
  左薬指３: "Left ring 3",
  左小指１: "Left little 1",
  左小指２: "Left little 2",
  左小指３: "Left little 3",
  右親指０: "Right thumb 0",
  右親指１: "Right thumb 1",
  右親指２: "Right thumb 2",
  右親指３: "Right thumb 3",
  右人差し指１: "Right index 1",
  右人差し指２: "Right index 2",
  右人差し指３: "Right index 3",
  右人指１: "Right index 1",
  右人指２: "Right index 2",
  右人指３: "Right index 3",
  右中指１: "Right middle 1",
  右中指２: "Right middle 2",
  右中指３: "Right middle 3",
  右薬指１: "Right ring 1",
  右薬指２: "Right ring 2",
  右薬指３: "Right ring 3",
  右小指１: "Right pinky 1",
  右小指２: "Right pinky 2",
  右小指３: "Right pinky 3",

  左腕捩１: "Left arm twist 1",
  左腕捩２: "Left arm twist 2",
  左腕捩３: "Left arm twist 3",
  右腕捩１: "Right arm twist 1",
  右腕捩２: "Right arm twist 2",
  右腕捩３: "Right arm twist 3",

  左つま先ＩＫ: "Left toe IK",
  右つま先ＩＫ: "Right toe IK",
  左つま先IK: "Left toe IK",
  右つま先IK: "Right toe IK",

  左手先: "Left hand tip",
  右手先: "Right hand tip",
  左親指先: "Left thumb tip",
  右親指先: "Right thumb tip",
  左人指先: "Left index tip",
  右人指先: "Right index tip",
  左中指先: "Left middle tip",
  右中指先: "Right middle tip",
  左薬指先: "Left ring tip",
  右薬指先: "Right ring tip",
  左小指先: "Left pinky tip",
  右小指先: "Right pinky tip",
  両目先: "Both eyes target",
}

function boneNameLookupVariants(name: string): string[] {
  const v: string[] = [name]
  const fw = name.replace(/[0-9]/g, (c) => String.fromCharCode(c.charCodeAt(0) - 0x30 + 0xff10))
  if (fw !== name) v.push(fw)
  const hw = name.replace(/[\uFF10-\uFF19]/g, (c) => String.fromCharCode(c.charCodeAt(0) - 0xff10 + 0x30))
  if (hw !== name) v.push(hw)
  if (name.includes("ＩＫ")) v.push(name.replace(/ＩＫ/g, "IK"))
  if (name.includes("IK")) v.push(name.replace(/IK/g, "ＩＫ"))
  return v
}

function boneEnglishLookup(boneName: string): string | undefined {
  for (const key of boneNameLookupVariants(boneName)) {
    const hit = BONE_NAME_EN[key]
    if (hit) return hit
  }
  const lower = boneName.toLowerCase()
  for (const [k, v] of Object.entries(BONE_NAME_EN)) {
    if (k.toLowerCase() === lower) return v
  }
  return undefined
}

export function boneDisplayLabel(boneName: string): string {
  const en = boneEnglishLookup(boneName)
  if (!en) return boneName
  const hasJp = /[\u3040-\u30ff\u4e00-\u9fff]/.test(boneName)
  if (!hasJp && boneName.toLowerCase().trim() === en.toLowerCase().trim()) return boneName
  return `${boneName} (${en})`
}

/** Sidebar-style two lines: English title + JP bone name when they differ (matches reference layout). */
export function boneTitleSubtitle(boneName: string): { title: string; subtitle: string | null } {
  const en = boneEnglishLookup(boneName)
  const hasJp = /[\u3040-\u30ff\u4e00-\u9fff]/.test(boneName)
  if (en && hasJp) return { title: boneName, subtitle: en }
  if (en && boneName.toLowerCase().trim() !== en.toLowerCase().trim()) return { title: boneName, subtitle: en }
  return { title: boneName, subtitle: null }
}

// ─── Channel definitions ─────────────────────────────────────────────────
export interface Channel {
  key: string
  label: string
  color: string
  group: "rot" | "tra"
  get: (kf: BoneKeyframe) => number
  set: (kf: BoneKeyframe, v: number) => void
}

export const ROT_CHANNELS: Channel[] = [
  {
    key: "rx",
    label: "Rot.X",
    color: C.rotX,
    group: "rot",
    get: (kf) => quatToEuler(kf.rotation).x,
    set: (kf, v) => {
      const e = quatToEuler(kf.rotation)
      kf.rotation = eulerToQuat(v, e.y, e.z)
    },
  },
  {
    key: "ry",
    label: "Rot.Y",
    color: C.rotY,
    group: "rot",
    get: (kf) => quatToEuler(kf.rotation).y,
    set: (kf, v) => {
      const e = quatToEuler(kf.rotation)
      kf.rotation = eulerToQuat(e.x, v, e.z)
    },
  },
  {
    key: "rz",
    label: "Rot.Z",
    color: C.rotZ,
    group: "rot",
    get: (kf) => quatToEuler(kf.rotation).z,
    set: (kf, v) => {
      const e = quatToEuler(kf.rotation)
      kf.rotation = eulerToQuat(e.x, e.y, v)
    },
  },
]

export const TRA_CHANNELS: Channel[] = [
  {
    key: "tx",
    label: "Tra.X",
    color: C.traX,
    group: "tra",
    get: (kf) => kf.translation.x,
    set: (kf, v) => {
      kf.translation = new Vec3(v, kf.translation.y, kf.translation.z)
    },
  },
  {
    key: "ty",
    label: "Tra.Y",
    color: C.traY,
    group: "tra",
    get: (kf) => kf.translation.y,
    set: (kf, v) => {
      kf.translation = new Vec3(kf.translation.x, v, kf.translation.z)
    },
  },
  {
    key: "tz",
    label: "Tra.Z",
    color: C.traZ,
    group: "tra",
    get: (kf) => kf.translation.z,
    set: (kf, v) => {
      kf.translation = new Vec3(kf.translation.x, kf.translation.y, v)
    },
  },
]

export const ALL_CHANNELS: Channel[] = [...ROT_CHANNELS, ...TRA_CHANNELS]

export function getChannelsForTab(tab: string): Channel[] {
  if (tab === "allRot") return ROT_CHANNELS
  if (tab === "allTra") return TRA_CHANNELS
  const ch = ALL_CHANNELS.find((c) => c.key === tab)
  return ch ? [ch] : ROT_CHANNELS
}

export function getAxisConfig(tab: string) {
  const chans = getChannelsForTab(tab)
  const isRot = chans[0].group === "rot"
  if (isRot) {
    return { min: -90, max: 90, unit: "°", side: "left" as const, step: 30, subStep: 15 }
  } else {
    return { min: -5, max: 20, unit: "", side: "left" as const, step: 5, subStep: 2.5 }
  }
}

export const TABS = [
  { key: "allRot", label: "All Rot", color: null, sep: false },
  { key: "rx", label: "X", color: C.rotX, sep: false },
  { key: "ry", label: "Y", color: C.rotY, sep: false },
  { key: "rz", label: "Z", color: C.rotZ, sep: false },
  { key: "_sep", label: "", color: null, sep: true },
  { key: "allTra", label: "All Tra", color: null, sep: false },
  { key: "tx", label: "X", color: C.traX, sep: false },
  { key: "ty", label: "Y", color: C.traY, sep: false },
  { key: "tz", label: "Z", color: C.traZ, sep: false },
]

// ─── Mock data ───────────────────────────────────────────────────────────
export function makeMockClip(): AnimationClip {
  const boneTracks = new Map<string, BoneKeyframe[]>()
  const bones = [
    { name: "首", kf: [0, 8, 20, 35, 50, 68, 80, 95, 110, 120] },
    { name: "頭", kf: [0, 12, 30, 45, 60, 75, 90, 105, 120] },
    { name: "上半身", kf: [0, 15, 35, 55, 70, 90, 110, 120] },
    { name: "左ひざ", kf: [0, 6, 15, 24, 36, 48, 60, 72, 84, 96, 108, 120] },
    { name: "右ひざ", kf: [0, 10, 25, 40, 55, 70, 85, 100, 115, 120] },
    { name: "下半身", kf: [0, 20, 40, 60, 80, 100, 120] },
  ]

  for (const { name, kf } of bones) {
    const i = bones.findIndex((b) => b.name === name)
    boneTracks.set(
      name,
      kf.map((f) => ({
        boneName: name,
        frame: f,
        rotation: eulerToQuat(
          Math.sin(f * 0.08 + i) * 25,
          Math.cos(f * 0.06 + i) * 15,
          Math.sin(f * 0.04 + i) * 10,
        ),
        translation: new Vec3(
          Math.sin(f * 0.04) * 2.5,
          Math.cos(f * 0.06) * 4 + 12,
          Math.sin(f * 0.05) * 1.5,
        ),
        interpolation: {
          rotation: [
            { x: 20 + Math.floor(Math.sin(i * 1.1) * 35), y: 20 + Math.floor(Math.cos(i * 1.3) * 35) },
            { x: 107 - Math.floor(Math.sin(i * 1.1) * 35), y: 107 - Math.floor(Math.cos(i * 1.3) * 35) },
          ],
          translationX: [
            { x: 20, y: 20 },
            { x: 107, y: 107 },
          ],
          translationY: [
            { x: 20, y: 20 },
            { x: 107, y: 107 },
          ],
          translationZ: [
            { x: 20, y: 20 },
            { x: 107, y: 107 },
          ],
        },
      })),
    )
  }
  return { boneTracks, morphTracks: new Map(), frameCount: 120 }
}
