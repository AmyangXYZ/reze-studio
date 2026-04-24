"use client"

import { memo, useMemo } from "react"
import { EyeIcon } from "lucide-react"
import type { MaterialPreset, MaterialPresetMap } from "reze-engine"
import { Checkbox } from "@/components/ui/checkbox"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { MATERIAL_PRESETS, MATERIAL_PRESET_LABEL, presetForMaterial, setMaterialPreset } from "@/lib/materials"

interface MaterialListProps {
  materialNames: string[]
  presets: MaterialPresetMap
  hiddenMaterials: ReadonlySet<string>
  selectedMaterial: string | null
  onChangePresets: (next: MaterialPresetMap) => void
  onChangeVisible: (materialName: string, visible: boolean) => void
  onToggleSelect: (materialName: string) => void
  onDeselect: () => void
}

export const MaterialList = memo(function MaterialList({
  materialNames,
  presets,
  hiddenMaterials,
  selectedMaterial,
  onChangePresets,
  onChangeVisible,
  onToggleSelect,
  onDeselect,
}: MaterialListProps) {
  const rows = useMemo(
    () =>
      materialNames.map((name) => ({
        name,
        preset: presetForMaterial(name, presets),
        visible: !hiddenMaterials.has(name),
      })),
    [materialNames, presets, hiddenMaterials],
  )

  if (materialNames.length === 0) {
    return <div className="px-3 py-2 text-[11px] text-muted-foreground">No materials</div>
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex shrink-0 items-center gap-2 border-sidebar-border px-4 py-1.5 text-[10px] font-medium tracking-widest text-muted-foreground">
        <EyeIcon className="size-3 shrink-0" aria-label="Visible" />
        <span className="min-w-0 flex-1 truncate">Material</span>
        <span className="w-28 shrink-0">Shader Preset</span>
      </div>
      {/* Click-on-blank-area deselects — fires only when the click lands on
          the flex container itself (empty space below the last row), not on
          a row or any of its interactive children. */}
      <ScrollArea className="min-h-0 flex-1">
        <div
          className="flex flex-col gap-1 py-1.5"
          onClick={(e) => {
            if (e.target === e.currentTarget) onDeselect()
          }}
        >
          {rows.map(({ name, preset, visible }) => {
            const isSelected = selectedMaterial === name
            return (
              <div
                key={name}
                className="flex items-center gap-2 px-4"
              >
                <Checkbox
                  checked={visible}
                  onCheckedChange={(c) => onChangeVisible(name, c === true)}
                  aria-label={`Toggle visibility for ${name}`}
                />
                <button
                  type="button"
                  onClick={() => onToggleSelect(name)}
                  className={`min-w-0 flex-1 truncate text-left font-mono text-[11px] underline-offset-2 hover:underline ${
                    isSelected
                      ? "text-blue-400 underline"
                      : visible
                        ? "text-muted-foreground"
                        : "text-muted-foreground/40 line-through"
                  }`}
                  title={name}
                >
                  {name}
                </button>
                <Select
                  value={preset}
                  onValueChange={(next) => onChangePresets(setMaterialPreset(presets, name, next as MaterialPreset))}
                >
                  <SelectTrigger className="h-6 w-28 shrink-0">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {MATERIAL_PRESETS.map((p) => (
                      <SelectItem key={p} value={p}>
                        {MATERIAL_PRESET_LABEL[p]}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            )
          })}
        </div>
      </ScrollArea>
    </div>
  )
})
