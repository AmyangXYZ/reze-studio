"use client"

/** Status bar — self-contained external store + footer component.
 *
 *  Extracted from StudioPage so high-frequency chrome updates (FPS ticks, PMX
 *  swap feedback) don't re-render the page shell. The footer subscribes to its
 *  own slices via `useStudioStatusSelector`; producers push via
 *  `useStudioStatusActions()` without causing any parent re-render. */

import {
  createContext,
  createElement,
  memo,
  useContext,
  useRef,
  useSyncExternalStore,
  type ReactNode,
} from "react"

export type StudioStatusState = {
  pmxFileName: string
  fps: number | null
  /** Reserved for save feedback / transient hints. */
  message: string
}

export type StudioStatusActions = {
  setPmxFileName: (name: string) => void
  setFps: (fps: number | null) => void
  setMessage: (msg: string) => void
}

const INITIAL_STATE: StudioStatusState = {
  pmxFileName: "—",
  fps: null,
  message: "",
}

type StudioStatusStore = {
  getState: () => StudioStatusState
  subscribe: (l: () => void) => () => void
  actions: StudioStatusActions
}

function createStore(): StudioStatusStore {
  let state = INITIAL_STATE
  const listeners = new Set<() => void>()
  const set = (next: StudioStatusState) => {
    if (next === state) return
    state = next
    listeners.forEach((l) => l())
  }
  const update = <K extends keyof StudioStatusState>(key: K, value: StudioStatusState[K]) => {
    if (state[key] === value) return
    set({ ...state, [key]: value })
  }
  const actions: StudioStatusActions = {
    setPmxFileName: (name) => update("pmxFileName", name),
    setFps: (fps) => update("fps", fps),
    setMessage: (msg) => update("message", msg),
  }
  return {
    getState: () => state,
    subscribe: (l) => {
      listeners.add(l)
      return () => {
        listeners.delete(l)
      }
    },
    actions,
  }
}

const Ctx = createContext<StudioStatusStore | null>(null)

export function StudioStatusProvider({ children }: { children: ReactNode }) {
  const ref = useRef<StudioStatusStore | null>(null)
  if (ref.current == null) ref.current = createStore()
  return createElement(Ctx.Provider, { value: ref.current }, children)
}

function useStore(): StudioStatusStore {
  const s = useContext(Ctx)
  if (s == null) throw new Error("useStudioStatus* must be used within <StudioStatusProvider>")
  return s
}

export function useStudioStatusSelector<T>(selector: (s: StudioStatusState) => T): T {
  const store = useStore()
  const getSnapshot = () => selector(store.getState())
  return useSyncExternalStore(store.subscribe, getSnapshot, getSnapshot)
}

export function useStudioStatusActions(): StudioStatusActions {
  return useStore().actions
}

/** Footer — subscribes to its own slices so FPS ticks don't touch the page. */
export const StudioStatusFooter = memo(function StudioStatusFooter({
  clipDisplayName,
  hasClip,
  appVersion,
}: {
  clipDisplayName: string
  hasClip: boolean
  appVersion: string
}) {
  const pmxFileName = useStudioStatusSelector((s) => s.pmxFileName)
  const fps = useStudioStatusSelector((s) => s.fps)
  const message = useStudioStatusSelector((s) => s.message)
  return (
    <footer
      className="flex h-6 shrink-0 items-center gap-2 border-t border-border px-2 text-[10.5px] text-muted-foreground"
      role="status"
      aria-live="polite"
    >
      <div className="flex min-w-0 shrink-0 items-center gap-x-2 [overflow-wrap:anywhere]">
        <span>
          Model:{" "}
          <span className="font-medium text-foreground" title={pmxFileName}>
            {pmxFileName}
          </span>
        </span>
        <span className="text-border" aria-hidden>
          ·
        </span>
        <span>
          Animation:{" "}
          <span className="font-medium text-foreground" title={hasClip ? `${clipDisplayName}.vmd` : undefined}>
            {hasClip ? `${clipDisplayName}.vmd` : "—"}
          </span>
        </span>
      </div>
      <div className="min-w-0 flex-1 truncate px-2 text-left text-[10px] text-muted-foreground/90">{message}</div>
      <div className="flex shrink-0 items-center gap-x-2 tabular-nums">
        <span title="Main-thread / compositor frame rate">{fps != null ? `${fps} FPS` : "— FPS"}</span>
        <span className="text-border" aria-hidden>
          ·
        </span>
        <span>v{appVersion}</span>
      </div>
    </footer>
  )
})
