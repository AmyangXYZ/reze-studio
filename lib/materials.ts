import type { MaterialPreset, MaterialPresetMap } from "reze-engine"

/** Preset order shown in the Materials panel Select. `default` is the fallback
 *  (Principled BSDF in the engine — anything not matched lands here). */
export const MATERIAL_PRESETS: MaterialPreset[] = [
  "default",
  "face",
  "body",
  "eye",
  "hair",
  "cloth_smooth",
  "cloth_rough",
  "stockings",
  "metal",
]

export const MATERIAL_PRESET_LABEL: Record<MaterialPreset, string> = {
  default: "Default",
  face: "Face",
  body: "Body",
  eye: "Eye",
  hair: "Hair",
  cloth_smooth: "Smooth Cloth",
  cloth_rough: "Rough Cloth",
  stockings: "Stockings",
  metal: "Metal",
}

/** Substring keywords for each preset. First-match wins in preset list order,
 *  so put more specific presets (eye, face) ahead of generic ones (body, cloth).
 *  Exact reze PMX material names live here too so the bundled model lands in
 *  the same buckets the engine used to hardcode. */
const CLASSIFY_KEYWORDS: Partial<Record<MaterialPreset, string[]>> = {
  eye: ["眼睛", "眼白", "目白", "右瞳", "左瞳", "眉毛", "eye", "iris", "pupil", "brow"],
  face: ["脸", "顔", "face"],
  hair: ["头发", "髪", "hair"],
  stockings: ["袜子", "stocking", "tights", "pantyhose"],
  metal: ["earring", "metal", "金属"],
  cloth_smooth: [
    "衣服",
    "裙子",
    "裙带",
    "裙布",
    "外套",
    "外套饰",
    "裤子",
    "裤子0",
    "腿环",
    "发饰",
    "鞋子",
    "鞋子饰",
    "shirt",
    "shoes",
    "shorts",
    "trigger",
    "dress",
    "hair_accessory",
    "cloth01_shoes",
    "cloth",
    "skirt",
    "jacket",
    "pants",
  ],
  body: ["皮肤", "skin", "body"],
}

/** Classification order — presets probed top-down; first keyword hit wins. */
const CLASSIFY_ORDER: MaterialPreset[] = ["eye", "face", "hair", "stockings", "metal", "cloth_smooth", "body"]

/** Heuristic: substring-match each material against the keyword corpus and
 *  bucket it into a MaterialPresetMap the engine can consume. Unmatched names
 *  stay out of the map (the engine treats absent entries as "default"). */
export function autoClassifyMaterials(materialNames: readonly string[]): MaterialPresetMap {
  const next: MaterialPresetMap = {}
  for (const raw of materialNames) {
    if (!raw) continue
    const lower = raw.toLowerCase()
    for (const preset of CLASSIFY_ORDER) {
      const kws = CLASSIFY_KEYWORDS[preset]
      if (!kws) continue
      if (!kws.some((k) => lower.includes(k.toLowerCase()))) continue
      const bucket = (next[preset] ??= [])
      bucket.push(raw)
      break
    }
  }
  return next
}

/** Reverse lookup: which preset does this material currently sit in? Returns
 *  "default" when the name is missing from the map (engine's own fallback). */
export function presetForMaterial(materialName: string, map: MaterialPresetMap): MaterialPreset {
  for (const preset of MATERIAL_PRESETS) {
    if (preset === "default") continue
    if (map[preset]?.includes(materialName)) return preset
  }
  return "default"
}

/** Move a material into the given preset (removing it from any other bucket).
 *  "default" = drop from the map entirely. Returns a new map; original unchanged. */
export function setMaterialPreset(
  map: MaterialPresetMap,
  materialName: string,
  preset: MaterialPreset,
): MaterialPresetMap {
  const next: MaterialPresetMap = {}
  for (const key of MATERIAL_PRESETS) {
    if (key === "default") continue
    const names = map[key]
    if (!names) continue
    const kept = names.filter((n) => n !== materialName)
    if (kept.length > 0) next[key] = kept
  }
  if (preset !== "default") {
    const bucket = (next[preset] ??= [])
    if (!bucket.includes(materialName)) bucket.push(materialName)
  }
  return next
}
