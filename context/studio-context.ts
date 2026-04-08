"use client"

/** Editable document + selection — the undo/redo target.
 *  External store so consumers can subscribe to slices via `useStudioSelector`
 *  without re-rendering on unrelated changes. Transport (playhead, play/pause)
 *  lives in <Playback>; playback ticks never touch this store. */
import {
  createContext,
  createElement,
  useContext,
  useRef,
  useSyncExternalStore,
  type Dispatch,
  type ReactNode,
  type SetStateAction,
} from "react"
import type { AnimationClip } from "reze-engine"
import { clipAfterKeyframeEdit } from "@/lib/utils"

/** Dopesheet diamond vs curve-editor handle — shared by timeline hit-testing. */
export interface SelectedKeyframe {
  bone?: string
  morph?: string
  frame: number
  channel?: string
  type: "dope" | "curve"
}

export type StudioState = {
  clip: AnimationClip | null
  clipDisplayName: string
  selectedBone: string | null
  selectedMorph: string | null
  selectedKeyframes: SelectedKeyframe[]
}

export type StudioClipCommit = Dispatch<SetStateAction<AnimationClip | null>>
export type StudioKeyframesSetter = Dispatch<SetStateAction<SelectedKeyframe[]>>

export type StudioActions = {
  commit: StudioClipCommit
  setClipDisplayName: (name: string) => void
  setSelectedBone: Dispatch<SetStateAction<string | null>>
  setSelectedMorph: Dispatch<SetStateAction<string | null>>
  setSelectedKeyframes: StudioKeyframesSetter
}

const INITIAL_STATE: StudioState = {
  clip: null,
  clipDisplayName: "clip",
  selectedBone: null,
  selectedMorph: null,
  selectedKeyframes: [],
}

/** Resolve a `SetStateAction<T>` against the current value. */
function resolve<T>(action: SetStateAction<T>, prev: T): T {
  return typeof action === "function" ? (action as (p: T) => T)(prev) : action
}

type StudioStore = {
  getState: () => StudioState
  subscribe: (listener: () => void) => () => void
  actions: StudioActions
}

function createStudioStore(): StudioStore {
  let state = INITIAL_STATE
  const listeners = new Set<() => void>()

  /** Replace state and notify — no-op if nothing changed. */
  const set = (next: StudioState) => {
    if (next === state) return
    state = next
    listeners.forEach((l) => l())
  }

  /** Update a single field, bailing if the resolved value is identical. */
  const update = <K extends keyof StudioState>(key: K, action: SetStateAction<StudioState[K]>) => {
    const next = resolve(action, state[key])
    if (next === state[key]) return
    set({ ...state, [key]: next })
  }

  const actions: StudioActions = {
    commit: (payload) => {
      const next = resolve(payload, state.clip)
      if (next == null) {
        set({ ...state, clip: null, selectedBone: null, selectedMorph: null, selectedKeyframes: [] })
      } else {
        set({ ...state, clip: clipAfterKeyframeEdit(next) })
      }
    },
    setClipDisplayName: (name) => update("clipDisplayName", name),
    setSelectedBone: (payload) => update("selectedBone", payload),
    setSelectedMorph: (payload) => update("selectedMorph", payload),
    setSelectedKeyframes: (payload) => update("selectedKeyframes", payload),
  }

  return {
    getState: () => state,
    subscribe: (listener) => {
      listeners.add(listener)
      return () => {
        listeners.delete(listener)
      }
    },
    actions,
  }
}

const StudioStoreContext = createContext<StudioStore | null>(null)

export function Studio({ children }: { children: ReactNode }) {
  const storeRef = useRef<StudioStore | null>(null)
  if (storeRef.current == null) storeRef.current = createStudioStore()
  return createElement(StudioStoreContext.Provider, { value: storeRef.current }, children)
}

function useStudioStore(): StudioStore {
  const store = useContext(StudioStoreContext)
  if (store == null) throw new Error("useStudio* must be used within <Studio>")
  return store
}

/** Subscribe to a slice of studio state. Component re-renders only when the
 *  selected value changes (Object.is compare). Selectors should return a
 *  reference-stable value from state — prefer top-level fields. */
export function useStudioSelector<T>(selector: (state: StudioState) => T): T {
  const store = useStudioStore()
  const getSnapshot = () => selector(store.getState())
  return useSyncExternalStore(store.subscribe, getSnapshot, getSnapshot)
}

/** Stable actions bag — never causes a re-render. Use this in components that
 *  only dispatch without reading state. */
export function useStudioActions(): StudioActions {
  return useStudioStore().actions
}

