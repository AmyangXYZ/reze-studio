"use client"

// The whole i18n layer in one file, the same shape reze-design uses: one `en`
// object that IS the type, a `zh` that has to match it, and a tiny context on
// top. No key strings, no lookup misses — `t.menu.exportVmd` either exists in
// both dictionaries or the build fails.
//
// Bone and morph names are deliberately absent. Those are the model's own data
// in Japanese, they are what a VMD keys against, and translating them would
// make the editor disagree with the file it is editing. `boneDisplayLabel` in
// lib/animation.ts already puts an English gloss beside the Japanese where one
// exists, which is the right treatment: an annotation, not a replacement.

import { createContext, useCallback, useContext, useEffect, useState } from "react"
import { storageKey } from "@/lib/storage"

export const LOCALES = ["en", "zh"] as const
export type Locale = (typeof LOCALES)[number]

/** Native names for the switcher — each shown in its own script. */
export const LOCALE_LABELS: Record<Locale, string> = {
  en: "English",
  zh: "中文",
}

// No `as const`: leaves the leaves inferred as `string`, so `Dictionary` is
// this shape with every string translatable.
const en = {
  menu: {
    file: "File",
    groupMotion: "Motion",
    groupMorphs: "Morphs",
    groupCamera: "Camera",
    groupMusic: "Music",
    groupModel: "Model & reference",
    help: "Help",
    settings: "Settings",
    newProject: "New project",
    resetProject: "Reset project…",
    loadPmx: "Load PMX folder…",
    openVmd: "Open VMD…",
    importVmd: "Import VMD…",
    importMorphVmd: "Import morph VMD…",
    importCameraVmd: "Import camera VMD…",
    importMusic: "Import music…",
    importVideo: "Import reference video…",
    exportVmd: "Export VMD…",
    exportMotion: "Export motion only…",
    exportMorphs: "Export morphs only…",
    exportCamera: "Export camera…",
    clearMotion: "Clear motion",
    clearMorphs: "Clear morphs",
    clearCamera: "Clear camera",
    clearMusic: "Clear music",
    tutorial: "Tutorial (README)",
    renderScene: "Render a scene — Reze Design",
    shortcuts: "Keyboard shortcuts…",
    about: "About Reze Studio",
    reportIssue: "Report an issue",
    ikEnabled: "IK enabled",
    showSkeleton: "Show skeleton",
    theme: "Theme…",
    language: "Language",
  },
  panel: {
    clips: "Clips",
    bones: "Bones",
    morphs: "Morphs",
    camera: "Camera",
    properties: "Properties",
    materials: "Materials",
    clipsTitle: "Every clip in this project. Click one to edit it.",
    bonesTitle: "The model's bones, grouped. The number beside a bone is its keyframe count.",
    morphsTitle: "The model's morphs. The number beside one is its keyframe count.",
    cameraNone: "No camera motion yet — File › Load camera VMD…",
    cameraEdit: "Edit the camera shot",
    noMorphs: "No morphs",
    resize: (name: string) => `Resize ${name}`,
  },
  clips: {
    empty: "No clips yet",
    import: "Import VMD…",
    rowTitle: (name: string, frames: number) => `${name} — ${frames} frames`,
    dragHint: (name: string, frames: number) => `${name} — ${frames} frames. Click to edit, drag onto a lane.`,
    removeHint: (name: string) => `Remove ${name} from the project`,
    removeTitle: (name: string) => `Remove ${name}?`,
    removeBlurb: "Its keyframes leave the project. Undo does not cover this — export it first if you want it back.",
    cancel: "Cancel",
    remove: "Remove",
    importedOne: (file: string) => `Imported ${file} — it is in Clips`,
    importedMany: (n: number) => `Imported ${n} clips`,
  },
  arrange: {
    arrange: "Arrange",
    keys: "Keys",
    arrangeTitle: "Lay clips out on tracks",
    keysTitle: "Edit the active clip's keyframes",
    topLane: "The top lane wins where two lanes key the same bone",
    dropHere: "Drop to place the clip here",
    mute: "Stop this lane driving the model",
    unmute: "Let this lane drive the model again",
    split: "Split",
    splitTitle: "Cut the selected clips at the playhead",
    copy: "Copy",
    cut: "Cut",
    paste: "Paste",
    editing: (name: string) => `Editing ${name}`,
  },
  timeline: {
    time: "Time",
    value: "Value",
    keys: "Keys",
    music: "Music",
    currentFrame: "Current frame",
    endFrame: "Clip end frame",
    scrub: "Scrub playhead",
    zoom: "Timeline zoom",
    zoomIn: "Zoom in",
    zoomOut: "Zoom out",
    camera: "Camera",
    selectBone: "Select a bone to view curves",
    selectMorph: "Select a morph to view curve",
    noKeyframes: (what: string) => `No keyframes — ${what}`,
    empty: "Load VMD for timeline…",
    rotation: "Rotation",
    translation: "Translation",
    target: "Target",
    weight: "Weight",
    distance: "Distance",
    fov: "FOV",
  },
  inspector: {
    rotationDeg: "Rotation (°)",
    translation: "Translation",
    weight: "Weight",
    interpolation: "Interpolation",
    operations: "Operations",
    groupKey: "Key",
    groupSel: "Sel",
    groupTrack: "Track",
    groupGizmo: "Gizmo",
    insert: "Insert",
    delete: "Delete",
    copy: "Copy",
    cut: "Cut",
    paste: "Paste",
    simplify: "Simplify",
    clear: "Clear",
    visible: "Visible",
    hidden: "Hidden",
    simplifyTitle: "Reduce redundant keyframes on the selected bone track",
    clearTitle: "Remove all keyframes on the selected bone or morph track",
    gizmoShow: "Show the transform gizmo on the selected bone (or dblclick the bone in the viewport)",
    gizmoHide: "Hide the transform gizmo (or dblclick empty space in the viewport)",
    none: "—",
  },
  camera: {
    title: "Camera",
    noKeys: "No keys — move a slider to key one",
    editingAt: (count: number, frame: number) => `${count} keys · editing key @ ${frame}`,
    noneAt: (count: number, frame: number) => `${count} keys · none @ ${frame}`,
    rotation: "Rotation",
    target: "Target",
    insertTitle: "Key the camera's current pose at the playhead",
    deleteTitle: "Remove the camera keyframe at the playhead",
    simplifyDisabled: "Simplify applies to dense bone tracks — a camera's keys are its cuts",
    clearTitle: "Remove every camera keyframe",
  },
  materials: {
    empty: "No materials",
    material: "Material",
    styleGroup: "Style Group",
    visible: "Visible",
  },
  footer: {
    ik: "IK",
    off: "off",
    fpsTitle: "Main-thread / compositor frame rate",
  },
  viewport: {
    following: "Following the camera track — click to orbit freely",
    orbiting: "Free orbit — click to follow the camera track",
    hidePanels: "Hide the side panels (\\)",
    showPanels: "Show the side panels (\\)",
  },
  pmx: {
    title: "Multiple .pmx files found",
    blurb: "Choose one to load.",
  },
  boneGroups: {
    "All Bones": "All bones",
    "Upper Body": "Upper body",
    "Left Arm": "Left arm",
    "Right Arm": "Right arm",
    "Left Hand": "Left hand",
    "Right Hand": "Right hand",
    "Lower Body": "Lower body",
  } as Record<string, string>,
  notice: {
    title: "This is not the official Reze Studio",
    body: "This copy was repackaged by a third party. Reze Studio runs only as the web app at reze.studio; wrapped builds are not supported and may be modified or out of date.",
    source: "The source is open at ",
    sourceEnd: ".",
    open: "Open",
  },
}

export type Dictionary = typeof en

const zh: Dictionary = {
  menu: {
    file: "文件",
    groupMotion: "动作",
    groupMorphs: "表情",
    groupCamera: "镜头",
    groupMusic: "音乐",
    groupModel: "模型与参考",
    help: "帮助",
    settings: "设置",
    newProject: "新建项目",
    resetProject: "重置项目…",
    loadPmx: "载入 PMX 文件夹…",
    openVmd: "打开 VMD…",
    importVmd: "导入 VMD…",
    importMorphVmd: "导入表情 VMD…",
    importCameraVmd: "导入镜头 VMD…",
    importMusic: "导入音乐…",
    importVideo: "导入参考视频…",
    exportVmd: "导出 VMD…",
    exportMotion: "仅导出动作…",
    exportMorphs: "仅导出表情…",
    exportCamera: "导出镜头…",
    clearMotion: "清除动作",
    clearMorphs: "清除表情",
    clearCamera: "清除镜头",
    clearMusic: "清除音乐",
    tutorial: "使用说明（README）",
    renderScene: "渲染场景 — Reze Design",
    shortcuts: "快捷键…",
    about: "关于 Reze Studio",
    reportIssue: "反馈问题",
    ikEnabled: "启用 IK",
    showSkeleton: "显示骨架",
    theme: "主题…",
    language: "语言",
  },
  panel: {
    clips: "片段",
    bones: "骨骼",
    morphs: "表情",
    camera: "镜头",
    properties: "属性",
    materials: "材质",
    clipsTitle: "本项目的全部片段。点击其中一个即可编辑。",
    bonesTitle: "模型的骨骼，按组排列。骨骼旁的数字是关键帧数量。",
    morphsTitle: "模型的表情。名称旁的数字是关键帧数量。",
    cameraNone: "尚无镜头动作 — 文件 › 载入镜头 VMD…",
    cameraEdit: "编辑镜头",
    noMorphs: "无表情",
    resize: (name: string) => `调整${name}高度`,
  },
  clips: {
    empty: "暂无片段",
    import: "导入 VMD…",
    rowTitle: (name: string, frames: number) => `${name} — ${frames} 帧`,
    dragHint: (name: string, frames: number) => `${name} — ${frames} 帧。点击编辑，拖到轨道上放置。`,
    removeHint: (name: string) => `从项目中移除 ${name}`,
    removeTitle: (name: string) => `移除 ${name}？`,
    removeBlurb: "它的关键帧将离开项目。撤销无法恢复 — 若还需要，请先导出。",
    cancel: "取消",
    remove: "移除",
    importedOne: (file: string) => `已导入 ${file} — 见「片段」`,
    importedMany: (n: number) => `已导入 ${n} 个片段`,
  },
  arrange: {
    arrange: "编排",
    keys: "关键帧",
    arrangeTitle: "在轨道上编排片段",
    keysTitle: "编辑当前片段的关键帧",
    topLane: "两条轨道驱动同一骨骼时，上层轨道优先",
    dropHere: "松开即可把片段放在这里",
    mute: "停用此轨道",
    unmute: "启用此轨道",
    split: "分割",
    splitTitle: "在播放头处分割所选片段",
    copy: "复制",
    cut: "剪切",
    paste: "粘贴",
    editing: (name: string) => `正在编辑 ${name}`,
  },
  timeline: {
    time: "时间",
    value: "数值",
    keys: "关键帧",
    music: "音乐",
    currentFrame: "当前帧",
    endFrame: "片段结束帧",
    scrub: "拖动播放头",
    zoom: "时间轴缩放",
    zoomIn: "放大",
    zoomOut: "缩小",
    camera: "镜头",
    selectBone: "选择一根骨骼以查看曲线",
    selectMorph: "选择一个表情以查看曲线",
    noKeyframes: (what: string) => `无关键帧 — ${what}`,
    empty: "载入 VMD 以显示时间轴…",
    rotation: "旋转",
    translation: "位移",
    target: "目标点",
    weight: "权重",
    distance: "距离",
    fov: "视角",
  },
  inspector: {
    rotationDeg: "旋转（°）",
    translation: "位移",
    weight: "权重",
    interpolation: "插值",
    operations: "操作",
    groupKey: "关键帧",
    groupSel: "选中",
    groupTrack: "轨道",
    groupGizmo: "控制器",
    insert: "插入",
    delete: "删除",
    copy: "复制",
    cut: "剪切",
    paste: "粘贴",
    simplify: "精简",
    clear: "清空",
    visible: "显示",
    hidden: "隐藏",
    simplifyTitle: "移除所选骨骼轨道上多余的关键帧",
    clearTitle: "清除所选骨骼或表情轨道上的全部关键帧",
    gizmoShow: "在所选骨骼上显示变换控制器（或在视口中双击该骨骼）",
    gizmoHide: "隐藏变换控制器（或在视口空白处双击）",
    none: "—",
  },
  camera: {
    title: "镜头",
    noKeys: "尚无关键帧 — 拖动滑块即可打上一帧",
    editingAt: (count: number, frame: number) => `${count} 帧 · 正在编辑第 ${frame} 帧`,
    noneAt: (count: number, frame: number) => `${count} 帧 · 第 ${frame} 帧无关键帧`,
    rotation: "旋转",
    target: "目标点",
    insertTitle: "在播放头处记录镜头当前姿态",
    deleteTitle: "删除播放头处的镜头关键帧",
    simplifyDisabled: "精简用于关键帧密集的骨骼轨道 — 镜头的关键帧就是分镜",
    clearTitle: "删除全部镜头关键帧",
  },
  materials: {
    empty: "无材质",
    material: "材质",
    styleGroup: "样式组",
    visible: "可见",
  },
  footer: {
    ik: "IK",
    off: "已关闭",
    fpsTitle: "主线程 / 合成器帧率",
  },
  viewport: {
    following: "正跟随镜头轨道 — 点击可自由环绕",
    orbiting: "自由环绕 — 点击可跟随镜头轨道",
    hidePanels: "隐藏侧边栏（\\）",
    showPanels: "显示侧边栏（\\）",
  },
  pmx: {
    title: "发现多个 .pmx 文件",
    blurb: "请选择要载入的一个。",
  },
  boneGroups: {
    "All Bones": "全部骨骼",
    "Upper Body": "上半身",
    "Left Arm": "左臂",
    "Right Arm": "右臂",
    "Left Hand": "左手",
    "Right Hand": "右手",
    "Lower Body": "下半身",
  },
  notice: {
    title: "这不是官方的 Reze Studio",
    body: "此版本由第三方封装。Reze Studio 只以 reze.studio 网页版的形式提供；封装版不受支持，且可能被篡改或过时。",
    source: "源码公开于 ",
    sourceEnd: "。",
    open: "打开",
  },
}

export const dictionaries: Record<Locale, Dictionary> = { en, zh }

// ── The tiny reactive layer ──
const STORAGE_KEY = storageKey("locale")
const isLocale = (v: string): v is Locale => (LOCALES as readonly string[]).includes(v)

/** Saved choice wins; otherwise map the browser language onto a supported locale. */
function detectLocale(): Locale {
  if (typeof window === "undefined") return "en"
  try {
    const saved = window.localStorage.getItem(STORAGE_KEY)
    if (saved && isLocale(saved)) return saved
  } catch {
    // localStorage can throw in private mode — fall through to the browser's.
  }
  return navigator.language.toLowerCase().startsWith("zh") ? "zh" : "en"
}

type I18nContextValue = {
  locale: Locale
  setLocale: (locale: Locale) => void
  /** The active dictionary — read keys directly, e.g. `t.menu.exportVmd`. */
  t: Dictionary
}

const I18nContext = createContext<I18nContextValue | null>(null)

export function I18nProvider({ children }: { children: React.ReactNode }) {
  // First render is "en" to match `<html lang="en">`; the detected locale
  // arrives in an effect, so there is no hydration mismatch to flash through.
  const [locale, setLocaleState] = useState<Locale>("en")

  useEffect(() => {
    setLocaleState(detectLocale())
  }, [])

  const setLocale = useCallback((next: Locale) => {
    setLocaleState(next)
    try {
      window.localStorage.setItem(STORAGE_KEY, next)
    } catch {
      // Non-fatal: the in-memory choice still applies for this session.
    }
  }, [])

  return <I18nContext.Provider value={{ locale, setLocale, t: dictionaries[locale] }}>{children}</I18nContext.Provider>
}

export function useI18n(): I18nContextValue {
  const ctx = useContext(I18nContext)
  if (!ctx) throw new Error("useI18n must be used within <I18nProvider>")
  return ctx
}

/** Shorthand for components that only read strings: `const t = useT()`. */
export function useT(): Dictionary {
  return useI18n().t
}
