"use client"

/** Transport state (playhead + play/pause). Split from <Studio> so playback ticks
 *  don't invalidate the document/selection store. Not part of undo/redo.
 *
 *  External store (`useSyncExternalStore`) so consumers can subscribe to slices
 *  via `usePlaybackSelector` and only re-render on their own field. */
import {
  createContext,
  createElement,
  useContext,
  useRef,
  useSyncExternalStore,
  type Dispatch,
  type ReactNode,
  type RefObject,
  type SetStateAction,
} from "react"

export type PlaybackState = {
  /** Transport playhead in clip frames (fractional allowed while scrubbing / playing). */
  currentFrame: number
  playing: boolean
}

export type PlaybackActions = {
  setCurrentFrame: Dispatch<SetStateAction<number>>
  setPlaying: Dispatch<SetStateAction<boolean>>
}

const INITIAL_STATE: PlaybackState = {
  currentFrame: 0,
  playing: false,
}

function resolve<T>(action: SetStateAction<T>, prev: T): T {
  return typeof action === "function" ? (action as (p: T) => T)(prev) : action
}

type PlaybackStore = {
  getState: () => PlaybackState
  subscribe: (listener: () => void) => () => void
  actions: PlaybackActions
  currentFrameRef: RefObject<number>
}

function createPlaybackStore(): PlaybackStore {
  let state = INITIAL_STATE
  const listeners = new Set<() => void>()
  const currentFrameRef: RefObject<number> = { current: 0 }

  const set = (next: PlaybackState) => {
    if (next === state) return
    state = next
    currentFrameRef.current = next.currentFrame
    listeners.forEach((l) => l())
  }

  const update = <K extends keyof PlaybackState>(key: K, action: SetStateAction<PlaybackState[K]>) => {
    const next = resolve(action, state[key])
    if (next === state[key]) return
    set({ ...state, [key]: next })
  }

  const actions: PlaybackActions = {
    setCurrentFrame: (payload) => update("currentFrame", payload),
    setPlaying: (payload) => update("playing", payload),
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
    currentFrameRef,
  }
}

const PlaybackStoreContext = createContext<PlaybackStore | null>(null)

export function Playback({ children }: { children: ReactNode }) {
  const storeRef = useRef<PlaybackStore | null>(null)
  if (storeRef.current == null) storeRef.current = createPlaybackStore()
  return createElement(PlaybackStoreContext.Provider, { value: storeRef.current }, children)
}

function usePlaybackStore(): PlaybackStore {
  const store = useContext(PlaybackStoreContext)
  if (store == null) throw new Error("usePlayback* must be used within <Playback>")
  return store
}

/** Subscribe to a slice of playback state. */
export function usePlaybackSelector<T>(selector: (state: PlaybackState) => T): T {
  const store = usePlaybackStore()
  const getSnapshot = () => selector(store.getState())
  return useSyncExternalStore(store.subscribe, getSnapshot, getSnapshot)
}

/** Stable actions bag — never causes a re-render. */
export function usePlaybackActions(): PlaybackActions {
  return usePlaybackStore().actions
}

/** Convenience: subscribe to everything. Prefer `usePlaybackSelector` when you
 *  only need one field. */
export function usePlayback(): PlaybackState & PlaybackActions {
  const currentFrame = usePlaybackSelector((s) => s.currentFrame)
  const playing = usePlaybackSelector((s) => s.playing)
  const actions = usePlaybackActions()
  return { currentFrame, playing, ...actions }
}

/** Read-only, non-subscribing access to the latest playhead. The returned ref
 *  identity is stable for the lifetime of <Playback>, so consuming this hook
 *  will NOT cause a re-render when the playhead moves. */
export function usePlaybackFrameRef(): RefObject<number> {
  return usePlaybackStore().currentFrameRef
}
