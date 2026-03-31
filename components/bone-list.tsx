"use client"

import { ScrollArea } from "@/components/ui/scroll-area"
import { BONE_GROUPS, boneDisplayLabel } from "@/lib/animation"
import { cn } from "@/lib/utils"

interface BoneListProps {
  allBones: string[]
  selectedGroup: string
  activeBone: string | null
  onSelectGroup: (group: string) => void
  onSelectBone: (bone: string) => void
}

export function BoneList({
  allBones,
  selectedGroup,
  activeBone,
  onSelectGroup,
  onSelectBone,
}: BoneListProps) {
  return (
    <ScrollArea className="h-full">
      <div className="py-1">
        <div className="px-3 pb-1 pt-0.5 text-[11px] font-medium uppercase tracking-widest text-muted-foreground">
          Bone Groups
        </div>
        {Object.entries(BONE_GROUPS).map(([name, bones]) => {
          const isSelected = selectedGroup === name
          const groupBones = bones ? bones.filter((b) => allBones.includes(b)) : allBones
          return (
            <div key={name}>
              <button
                onClick={() => onSelectGroup(name)}
                className={cn(
                  "flex w-full items-center px-3 py-1 text-left text-[11px] font-medium text-muted-foreground transition-colors",
                  isSelected
                    ? "border-l-2 border-blue-400 bg-white/[0.03] text-blue-400"
                    : "border-l-2 border-transparent hover:bg-white/[0.03]",
                )}
              >
                <span
                  className={cn(
                    "mr-1 text-[9px] transition-transform",
                    isSelected ? "rotate-90 text-blue-400" : "text-muted-foreground",
                  )}
                >
                  ▶
                </span>
                {name}
              </button>
              {isSelected &&
                groupBones.map((b) => (
                  <button
                    key={b}
                    onClick={(e) => {
                      e.stopPropagation()
                      onSelectBone(b)
                    }}
                    className={cn(
                      "flex w-full items-center py-0.5 pl-6 pr-3 text-left text-[11px] font-mono leading-snug text-muted-foreground transition-colors",
                      activeBone === b
                        ? "bg-blue-400/[0.08] text-blue-400"
                        : "hover:bg-white/[0.03]",
                    )}
                  >
                    <span className="mr-1 w-2 text-[8px]">
                      {activeBone === b ? "●" : ""}
                    </span>
                    {boneDisplayLabel(b)}
                  </button>
                ))}
            </div>
          )
        })}
      </div>
    </ScrollArea>
  )
}
