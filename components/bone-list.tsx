"use client"

import { useRef, useState, useEffect, useCallback, useMemo, memo } from "react"
import { BONE_GROUPS } from "@/lib/animation"
import { cn } from "@/lib/utils"
import type { AnimationClip } from "reze-engine"

const GROUP_H = 24
const BONE_H = 20
const OVERSCAN = 8

interface BoneListProps {
  modelBones: string[]
  clip: AnimationClip | null
  selectedGroup: string
  activeBone: string | null
  onSelectGroup: (group: string) => void
  onSelectBone: (bone: string) => void
}

type Row =
  | { type: "group"; name: string; boneCount: number; isSelected: boolean }
  | { type: "bone"; name: string; kfCount: number; isActive: boolean }

const GroupRow = memo(function GroupRow({
  name,
  boneCount,
  isSelected,
  onClick,
}: {
  name: string
  boneCount: number
  isSelected: boolean
  onClick: () => void
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex h-full w-full items-center gap-0 border-l-2 px-3 text-left text-[11px] font-medium leading-none text-muted-foreground",
        isSelected
          ? "border-blue-400 bg-white/[0.03] text-blue-400 hover:bg-white/[0.05]"
          : "border-transparent hover:bg-white/[0.03]",
      )}
    >
      <span className="mr-1 inline-flex size-3 shrink-0 items-center justify-center text-[9px] leading-none">
        <span className={cn("transition-transform", isSelected ? "rotate-90 text-blue-400" : "text-muted-foreground")}>
          ▶
        </span>
      </span>
      <span className="min-w-0 flex-1 truncate py-[1px] ">
        {name}{" "}
        <span className="tabular-nums opacity-70">
          ({boneCount})
        </span>
      </span>
    </button>
  )
})

const BoneRow = memo(function BoneRow({
  name,
  kfCount,
  isActive,
  onClick,
}: {
  name: string
  kfCount: number
  isActive: boolean
  onClick: () => void
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex h-full w-full items-center gap-1 pl-6 pr-3 text-left font-mono text-[11px] font-normal leading-none",
        isActive
          ? "bg-blue-400/[0.08] text-blue-400 hover:bg-blue-400/12"
          : kfCount > 0
            ? "text-muted-foreground hover:bg-white/[0.03]"
            : "text-muted-foreground/65 hover:bg-white/[0.03]",
      )}
    >
      <span className="inline-flex w-1.5 shrink-0 text-[7px] leading-none" aria-hidden>
        {isActive ? "●" : ""}
      </span>
      <span className="min-w-0 flex-1 truncate">{name}</span>
      {kfCount > 0 && (
        <span className={cn("shrink-0 pr-1 tabular-nums text-[10px]", isActive ? "text-blue-400/80" : "opacity-55")}>
          [{kfCount}]
        </span>
      )}
    </button>
  )
})

export const BoneList = memo(function BoneList({
  modelBones,
  clip,
  selectedGroup,
  activeBone,
  onSelectGroup,
  onSelectBone,
}: BoneListProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [scrollTop, setScrollTop] = useState(0)
  const [viewH, setViewH] = useState(0)

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const ro = new ResizeObserver(() => setViewH(el.clientHeight))
    ro.observe(el)
    setViewH(el.clientHeight)
    return () => ro.disconnect()
  }, [])

  const onScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
    setScrollTop(e.currentTarget.scrollTop)
  }, [])

  const rows = useMemo<Row[]>(() => {
    const out: Row[] = []
    for (const [name, groupDef] of Object.entries(BONE_GROUPS)) {
      const groupBones = groupDef ? modelBones.filter((b) => groupDef.includes(b)) : modelBones
      const isSelected = selectedGroup === name
      out.push({ type: "group", name, boneCount: groupBones.length, isSelected })
      if (isSelected) {
        for (const b of groupBones) {
          out.push({
            type: "bone",
            name: b,
            kfCount: clip?.boneTracks.get(b)?.length ?? 0,
            isActive: activeBone === b,
          })
        }
      }
    }
    return out
  }, [modelBones, clip, selectedGroup, activeBone])

  // Precompute offsets
  const { offsets, total } = useMemo(() => {
    const offs: number[] = []
    let t = 0
    for (const r of rows) {
      offs.push(t)
      t += r.type === "group" ? GROUP_H : BONE_H
    }
    return { offsets: offs, total: t }
  }, [rows])

  // Visible window
  const startY = scrollTop - OVERSCAN * BONE_H
  const endY = scrollTop + viewH + OVERSCAN * BONE_H
  let startIdx = 0
  let endIdx = rows.length
  // Binary search for start
  let lo = 0, hi = rows.length - 1
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    const h = rows[mid].type === "group" ? GROUP_H : BONE_H
    if (offsets[mid] + h < startY) { startIdx = mid + 1; lo = mid + 1 }
    else hi = mid - 1
  }
  // Binary search for end
  lo = startIdx; hi = rows.length - 1
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    if (offsets[mid] > endY) { endIdx = mid; hi = mid - 1 }
    else lo = mid + 1
  }

  return (
    <div ref={containerRef} className="h-full overflow-y-auto [scrollbar-width:none] [&::-webkit-scrollbar]:hidden" onScroll={onScroll}>
      <div className="px-3 py-1 text-[11px] font-medium uppercase leading-tight tracking-widest text-muted-foreground">
        Bones
      </div>
      <div style={{ position: "relative", height: total }}>
        {rows.slice(startIdx, endIdx).map((row, i) => {
          const idx = startIdx + i
          const top = offsets[idx]
          const h = row.type === "group" ? GROUP_H : BONE_H
          return (
            <div
              key={row.type === "group" ? `g:${row.name}` : `b:${row.name}`}
              style={{ position: "absolute", top, left: 0, right: 0, height: h }}
            >
              {row.type === "group" ? (
                <GroupRow
                  name={row.name}
                  boneCount={row.boneCount}
                  isSelected={row.isSelected}
                  onClick={() => onSelectGroup(row.name)}
                />
              ) : (
                <BoneRow
                  name={row.name}
                  kfCount={row.kfCount}
                  isActive={row.isActive}
                  onClick={() => onSelectBone(row.name)}
                />
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
})
