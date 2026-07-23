import type { AlphaMode, MaterialPreset, MaterialPresetMap, RenderClass, ShaderGraph, StyleGroup } from "reze-engine"
import {
  BODY_GRAPH,
  CLOTH_ROUGH_GRAPH,
  CLOTH_SMOOTH_GRAPH,
  EYE_GRAPH,
  FACE_GRAPH,
  HAIR_GRAPH,
  METAL_GRAPH,
  STOCKINGS_GRAPH,
} from "reze-engine"

/** Style-group order shown in the Materials panel Select. `default` is the
 *  fallback: a material in no group renders the engine's neutral DEFAULT_GRAPH
 *  (Principled BSDF), so we never emit a group for it. */
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

/** Built-in shader graph backing each style category. `default` is intentionally
 *  absent — ungrouped materials fall through to the engine's neutral DEFAULT_GRAPH. */
const GRAPH_FOR_PRESET: Record<Exclude<MaterialPreset, "default">, ShaderGraph> = {
  face: FACE_GRAPH,
  body: BODY_GRAPH,
  eye: EYE_GRAPH,
  hair: HAIR_GRAPH,
  cloth_smooth: CLOTH_SMOOTH_GRAPH,
  cloth_rough: CLOTH_ROUGH_GRAPH,
  stockings: STOCKINGS_GRAPH,
  metal: METAL_GRAPH,
}

/** Pass-integration class per category. The graph is pure shading; the render
 *  class carries the built-in effects (hair's over-eyes stencil, the eye
 *  see-through stamp). Everything else stays "auto". */
const RENDER_CLASS_FOR_PRESET: Partial<Record<MaterialPreset, RenderClass>> = {
  eye: "eye",
  hair: "hair",
}

/** Alpha axis (orthogonal to render class). Stockings use Wyman & McGuire
 *  object-space hashed alpha; everything else is opaque. */
const ALPHA_MODE_FOR_PRESET: Partial<Record<MaterialPreset, AlphaMode>> = {
  stockings: "hashed",
}

/** Convert a category → material-names map into the engine's StyleGroup[] for
 *  `engine.applyStyleGroups`. One group per non-empty, non-`default` category;
 *  the category name is the stable group id, backed by its built-in graph plus
 *  the matching render class + alpha mode. Empty categories and `default` are
 *  omitted so their materials revert to the neutral ungrouped path. */
export function buildStyleGroups(map: MaterialPresetMap): StyleGroup[] {
  const groups: StyleGroup[] = []
  for (const preset of MATERIAL_PRESETS) {
    if (preset === "default") continue
    const materials = map[preset]
    if (!materials || materials.length === 0) continue
    groups.push({
      id: preset,
      label: MATERIAL_PRESET_LABEL[preset],
      materials: [...materials],
      graph: GRAPH_FOR_PRESET[preset],
      renderClass: RENDER_CLASS_FOR_PRESET[preset] ?? "auto",
      alphaMode: ALPHA_MODE_FOR_PRESET[preset] ?? "opaque",
    })
  }
  return groups
}

const KNOWN_PRESETS = new Set<string>(MATERIAL_PRESETS)

/** Reverse of {@link buildStyleGroups}: read the engine's installed StyleGroup[]
 *  (e.g. what `engine.autoStyleGroups` produced) back into the panel's category
 *  map, so the Select reflects exactly what's rendering — including groups the
 *  engine's name hints inferred that our local keyword pass missed. autoStyleGroups
 *  keys groups by category id, so this is a direct read; any group whose id isn't
 *  one of our categories (e.g. a future custom group) is ignored. */
export function styleGroupsToPresetMap(groups: readonly StyleGroup[]): MaterialPresetMap {
  const map: MaterialPresetMap = {}
  for (const g of groups) {
    if (g.id === "default" || !KNOWN_PRESETS.has(g.id) || g.materials.length === 0) continue
    map[g.id as MaterialPreset] = [...g.materials]
  }
  return map
}
