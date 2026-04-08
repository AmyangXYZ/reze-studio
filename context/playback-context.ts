"use client"

/** Transport state (playhead + play/pause). Split from <Studio> so playback ticks
 *  don't invalidate the document/selection context every rAF. Not part of undo/redo. */
import {
  createContext,
  createElement,
  useCallback,
  useContext,
  useMemo,
  useReducer,
  type Dispatch,
  type ReactNode,
  type SetStateAction,
} from "react"

export type PlaybackState = {
  /** Transport playhead in clip frames (fractional allowed while scrubbing / playing). */
  currentFrame: number
  playing: boolean
}

type PlaybackAction =
  | { type: "SET_CURRENT_FRAME"; payload: SetStateAction<number> }
  | { type: "SET_PLAYING"; payload: SetStateAction<boolean> }

function playbackReducer(state: PlaybackState, action: PlaybackAction): PlaybackState {
  switch (action.type) {
    case "SET_CURRENT_FRAME": {
      const next =
        typeof action.payload === "function"
          ? (action.payload as (prev: number) => number)(state.currentFrame)
          : action.payload
      if (next === state.currentFrame) return state
      return { ...state, currentFrame: next }
    }
    case "SET_PLAYING": {
      const next =
        typeof action.payload === "function"
          ? (action.payload as (prev: boolean) => boolean)(state.playing)
          : action.payload
      if (next === state.playing) return state
      return { ...state, playing: next }
    }
    default:
      return state
  }
}

const initialPlaybackState: PlaybackState = {
  currentFrame: 0,
  playing: false,
}

type PlaybackContextValue = {
  currentFrame: number
  setCurrentFrame: Dispatch<SetStateAction<number>>
  playing: boolean
  setPlaying: Dispatch<SetStateAction<boolean>>
}

const PlaybackContext = createContext<PlaybackContextValue | null>(null)

export function Playback({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(playbackReducer, initialPlaybackState)

  const setCurrentFrame = useCallback((payload: SetStateAction<number>) => {
    dispatch({ type: "SET_CURRENT_FRAME", payload })
  }, [])

  const setPlaying = useCallback((payload: SetStateAction<boolean>) => {
    dispatch({ type: "SET_PLAYING", payload })
  }, [])

  const value = useMemo(
    (): PlaybackContextValue => ({
      currentFrame: state.currentFrame,
      setCurrentFrame,
      playing: state.playing,
      setPlaying,
    }),
    [state.currentFrame, state.playing, setCurrentFrame, setPlaying],
  )

  return createElement(PlaybackContext.Provider, { value }, children)
}

export function usePlayback(): PlaybackContextValue {
  const ctx = useContext(PlaybackContext)
  if (ctx == null) throw new Error("usePlayback must be used within <Playback>")
  return ctx
}
