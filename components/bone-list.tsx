"use client"

import { Button } from "@/components/ui/button"
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
      {/* flex-col + gap-0: block-level stack — inline-flex buttons in a block leave line-box gaps between rows */}
      <div className="flex flex-col gap-1.5 py-0.5 leading-none">
        <div className="px-3 pb-1 pt-1 text-[11px] font-medium uppercase leading-tight tracking-widest text-muted-foreground">
          Bones
        </div>
        {Object.entries(BONE_GROUPS).map(([name, bones]) => {
          const isSelected = selectedGroup === name
          const groupBones = bones ? bones.filter((b) => allBones.includes(b)) : allBones
          return (
            <div key={name} className="flex flex-col gap-0">
              <Button
                type="button"
                variant="ghost"
                size="xs"
                onClick={() => onSelectGroup(name)}
                className={cn(
                  "flex h-auto min-h-0 w-full min-w-0 shrink justify-start gap-0 rounded-none border-l-2 px-3 pb-1 text-left text-[11px] font-medium leading-none text-muted-foreground [&_svg]:size-3",
                  isSelected
                    ? "border-blue-400 bg-white/[0.03] text-blue-400 hover:bg-white/[0.05] hover:text-blue-400"
                    : "border-transparent hover:bg-white/[0.03] hover:text-muted-foreground",
                )}
              >
                {/* Fixed box so rotate doesn’t change row height / flex metrics */}
                <span className="mr-1 inline-flex size-3 shrink-0 items-center justify-center text-[9px] leading-none">
                  <span
                    className={cn(
                      "transition-transform",
                      isSelected ? "rotate-90 text-blue-400" : "text-muted-foreground",
                    )}
                  >
                    ▶
                  </span>
                </span>
                <span className="min-w-0 flex-1 truncate text-left">
                  {name}{" "}
                  <span className="tabular-nums opacity-70">({groupBones.length})</span>
                </span>
              </Button>
              {isSelected &&
                groupBones.map((b) => (
                  <Button
                    type="button"
                    variant="ghost"
                    size="xs"
                    key={b}
                    onClick={(e) => {
                      e.stopPropagation()
                      onSelectBone(b)
                    }}
                    className={cn(
                      "flex h-auto min-h-0 w-full shrink justify-start gap-1 rounded-none py-1 pl-6 pr-3 text-left text-[11px] font-mono font-normal leading-none text-muted-foreground [&_svg]:size-3",
                      activeBone === b
                        ? "bg-blue-400/[0.08] text-blue-400 hover:bg-blue-400/12 hover:text-blue-400"
                        : "hover:bg-white/[0.03] hover:text-muted-foreground",
                    )}
                  >
                    <span
                      className="inline-flex w-2 shrink-0 items-center justify-center text-[9px] leading-none"
                      aria-hidden
                    >
                      <span className={activeBone === b ? "text-blue-400" : "text-transparent"}>●</span>
                    </span>
                    <span className="min-w-0 flex-1 truncate">{boneDisplayLabel(b)}</span>
                  </Button>
                ))}
            </div>
          )
        })}
      </div>
    </ScrollArea>
  )
}
