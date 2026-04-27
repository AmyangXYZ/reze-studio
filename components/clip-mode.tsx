"use client"

export function ClipPanel() {
  return (
    <div className="flex h-full flex-col gap-2 px-3 py-3 text-sm text-muted-foreground">
      <div className="text-[11px] font-medium uppercase tracking-widest text-muted-foreground/70">
        Library
      </div>
      <div className="text-[11px] leading-relaxed">
        Imported and created clips will appear here. Drag a clip onto the
        timeline to add it to a track.
      </div>
    </div>
  )
}

export function ClipModeTimeline() {
  return (
    <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
      Clip-mode timeline (coming soon)
    </div>
  )
}

export function ClipModeProperties() {
  return (
    <div className="flex h-full flex-col items-start gap-2 text-sm text-muted-foreground">
      <div className="text-xs font-medium uppercase tracking-widest text-muted-foreground/70">
        Clip
      </div>
      <div className="leading-relaxed">
        Select a clip to see its options.
      </div>
    </div>
  )
}
