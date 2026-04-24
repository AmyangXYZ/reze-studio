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
import { clipAfterKeyframeEdit, cloneAnimationClip } from "@/lib/utils"

const HISTORY_LIMIT = 100

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
  /** Engine shows an orange outline on this material. Mutually exclusive with
   *  bone/morph — selecting either clears this, and selecting a material
   *  clears bone/morph. Does not belong to the clip — not in undo history. */
  selectedMaterial: string | null
  /** Whether the viewport gizmo is currently shown. Decoupled from
   *  `selectedBone` so the user can hide the gizmo during playback (dblclick
   *  empty viewport) without losing inspector/bone-list context. Any
   *  `setSelectedBone` call re-shows the gizmo, including re-selecting the
   *  same bone — so the user never gets stuck with a hidden gizmo. */
  gizmoVisible: boolean
  selectedKeyframes: SelectedKeyframe[]
  /** Immutable clone of `clip` taken at the last commit / undo / redo. Lets
   *  us push a *clean* snapshot onto history even though slider preview
   *  mutates `clip`'s keyframes in place between commits. */
  clipSnapshot: AnimationClip | null
  past: AnimationClip[]
  future: AnimationClip[]
}

export type StudioClipCommit = Dispatch<SetStateAction<AnimationClip | null>>
export type StudioKeyframesSetter = Dispatch<SetStateAction<SelectedKeyframe[]>>

export type StudioActions = {
  commit: StudioClipCommit
  /** Load a clip without recording history — for VMD imports, PMX swaps,
   *  document reset. Clears past/future. Editing actions go through `commit`. */
  replaceClip: (next: AnimationClip | null) => void
  setClipDisplayName: (name: string) => void
  setSelectedBone: Dispatch<SetStateAction<string | null>>
  setSelectedMorph: Dispatch<SetStateAction<string | null>>
  setSelectedMaterial: Dispatch<SetStateAction<string | null>>
  setGizmoVisible: Dispatch<SetStateAction<boolean>>
  setSelectedKeyframes: StudioKeyframesSetter
  undo: () => void
  redo: () => void
}

const INITIAL_STATE: StudioState = {
  clip: null,
  clipDisplayName: "clip",
  selectedBone: null,
  selectedMorph: null,
  selectedMaterial: null,
  gizmoVisible: true,
  selectedKeyframes: [],
  clipSnapshot: null,
  past: [],
  future: [],
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

  /** Append snapshot to `past`, capping at HISTORY_LIMIT (drop oldest). */
  const pushPast = (past: AnimationClip[], snap: AnimationClip | null): AnimationClip[] => {
    if (snap == null) return past
    const next = past.length >= HISTORY_LIMIT ? past.slice(past.length - HISTORY_LIMIT + 1) : past.slice()
    next.push(snap)
    return next
  }

  const actions: StudioActions = {
    commit: (payload) => {
      const next = resolve(payload, state.clip)
      if (next == null) {
        set({
          ...state,
          clip: null,
          clipSnapshot: null,
          past: pushPast(state.past, state.clipSnapshot),
          future: [],
          selectedBone: null,
          selectedMorph: null,
          selectedKeyframes: [],
        })
        return
      }
      const finalNext = clipAfterKeyframeEdit(next)
      set({
        ...state,
        clip: finalNext,
        clipSnapshot: cloneAnimationClip(finalNext),
        past: pushPast(state.past, state.clipSnapshot),
        future: [],
      })
    },
    replaceClip: (next) => {
      if (next == null) {
        set({
          ...state,
          clip: null,
          clipSnapshot: null,
          past: [],
          future: [],
          selectedBone: null,
          selectedMorph: null,
          selectedKeyframes: [],
        })
        return
      }
      const finalNext = clipAfterKeyframeEdit(next)
      set({
        ...state,
        clip: finalNext,
        clipSnapshot: cloneAnimationClip(finalNext),
        past: [],
        future: [],
      })
    },
    setClipDisplayName: (name) => update("clipDisplayName", name),
    // Every bone-select event (even re-selecting the same bone) re-shows the
    // gizmo, so a user who dblclick-empty'd to hide it can bring it back by
    // clicking the already-highlighted bone in the list. Otherwise the set
    // bails at the Object.is equality guard in `update()` and the mirror
    // effect in EngineBridge never re-runs.
    setSelectedBone: (payload) => {
      const next = resolve(payload, state.selectedBone)
      const nextGizmoVisible = next != null ? true : state.gizmoVisible
      if (next === state.selectedBone && nextGizmoVisible === state.gizmoVisible) return
      set({ ...state, selectedBone: next, gizmoVisible: nextGizmoVisible })
    },
    setSelectedMorph: (payload) => update("selectedMorph", payload),
    setSelectedMaterial: (payload) => update("selectedMaterial", payload),
    setGizmoVisible: (payload) => update("gizmoVisible", payload),
    setSelectedKeyframes: (payload) => update("selectedKeyframes", payload),
    undo: () => {
      if (state.past.length === 0) return
      const popped = state.past[state.past.length - 1]
      const past = state.past.slice(0, -1)
      const future = state.clipSnapshot != null ? [state.clipSnapshot, ...state.future] : state.future
      // popped is immutable; clone it so preview-time mutation can't poison history.
      set({
        ...state,
        clip: cloneAnimationClip(popped),
        clipSnapshot: popped,
        past,
        future,
      })
    },
    redo: () => {
      if (state.future.length === 0) return
      const popped = state.future[0]
      const future = state.future.slice(1)
      const past = pushPast(state.past, state.clipSnapshot)
      set({
        ...state,
        clip: cloneAnimationClip(popped),
        clipSnapshot: popped,
        past,
        future,
      })
    },
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

