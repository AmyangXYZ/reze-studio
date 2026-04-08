"use client"

/** Editable document + selection. Wraps the reducer that becomes the undo/redo target.
 *  Transport (playhead + play/pause) lives in <Playback> so playback ticks don't
 *  invalidate this context every rAF. */
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
import type { AnimationClip } from "reze-engine"
import { clipAfterKeyframeEdit } from "@/lib/utils"

/** Dopesheet diamond vs curve-editor handle — shared by timeline hit-testing and reducer state. */
export interface SelectedKeyframe {
  bone?: string
  morph?: string
  frame: number
  channel?: string
  type: "dope" | "curve"
}

export type StudioClipCommit = Dispatch<SetStateAction<AnimationClip | null>>

export type StudioState = {
  clip: AnimationClip | null
  clipDisplayName: string
  selectedBone: string | null
  selectedMorph: string | null
  selectedKeyframes: SelectedKeyframe[]
}

type StudioReducerAction =
  | { type: "COMMIT"; payload: SetStateAction<AnimationClip | null> }
  | { type: "SET_CLIP_DISPLAY_NAME"; payload: string }
  | { type: "SET_SELECTED_BONE"; payload: SetStateAction<string | null> }
  | { type: "SET_SELECTED_MORPH"; payload: SetStateAction<string | null> }
  | { type: "SET_SELECTED_KEYFRAMES"; payload: SetStateAction<SelectedKeyframe[]> }

function studioReducer(state: StudioState, action: StudioReducerAction): StudioState {
  switch (action.type) {
    case "COMMIT": {
      const next =
        typeof action.payload === "function"
          ? (action.payload as (prev: AnimationClip | null) => AnimationClip | null)(state.clip)
          : action.payload
      if (next == null) {
        return {
          ...state,
          clip: null,
          selectedBone: null,
          selectedMorph: null,
          selectedKeyframes: [],
        }
      }
      return { ...state, clip: clipAfterKeyframeEdit(next) }
    }
    case "SET_CLIP_DISPLAY_NAME":
      return { ...state, clipDisplayName: action.payload }
    case "SET_SELECTED_BONE": {
      const next =
        typeof action.payload === "function"
          ? (action.payload as (prev: string | null) => string | null)(state.selectedBone)
          : action.payload
      return { ...state, selectedBone: next }
    }
    case "SET_SELECTED_MORPH": {
      const next =
        typeof action.payload === "function"
          ? (action.payload as (prev: string | null) => string | null)(state.selectedMorph)
          : action.payload
      return { ...state, selectedMorph: next }
    }
    case "SET_SELECTED_KEYFRAMES": {
      const next =
        typeof action.payload === "function"
          ? (action.payload as (prev: SelectedKeyframe[]) => SelectedKeyframe[])(state.selectedKeyframes)
          : action.payload
      return { ...state, selectedKeyframes: next }
    }
    default:
      return state
  }
}

const initialStudioState: StudioState = {
  clip: null,
  clipDisplayName: "clip",
  selectedBone: null,
  selectedMorph: null,
  selectedKeyframes: [],
}

export type StudioKeyframesSetter = Dispatch<SetStateAction<SelectedKeyframe[]>>

type StudioContextValue = {
  clip: AnimationClip | null
  commit: StudioClipCommit
  clipDisplayName: string
  setClipDisplayName: (name: string) => void
  selectedBone: string | null
  setSelectedBone: Dispatch<SetStateAction<string | null>>
  selectedMorph: string | null
  setSelectedMorph: Dispatch<SetStateAction<string | null>>
  selectedKeyframes: SelectedKeyframe[]
  setSelectedKeyframes: StudioKeyframesSetter
}

const StudioContext = createContext<StudioContextValue | null>(null)

export function Studio({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(studioReducer, initialStudioState)

  const commit = useCallback((payload: SetStateAction<AnimationClip | null>) => {
    dispatch({ type: "COMMIT", payload })
  }, [])

  const setClipDisplayName = useCallback((name: string) => {
    dispatch({ type: "SET_CLIP_DISPLAY_NAME", payload: name })
  }, [])

  const setSelectedBone = useCallback((payload: SetStateAction<string | null>) => {
    dispatch({ type: "SET_SELECTED_BONE", payload })
  }, [])

  const setSelectedMorph = useCallback((payload: SetStateAction<string | null>) => {
    dispatch({ type: "SET_SELECTED_MORPH", payload })
  }, [])

  const setSelectedKeyframes = useCallback((payload: SetStateAction<SelectedKeyframe[]>) => {
    dispatch({ type: "SET_SELECTED_KEYFRAMES", payload })
  }, [])

  const value = useMemo(
    (): StudioContextValue => ({
      clip: state.clip,
      commit,
      clipDisplayName: state.clipDisplayName,
      setClipDisplayName,
      selectedBone: state.selectedBone,
      setSelectedBone,
      selectedMorph: state.selectedMorph,
      setSelectedMorph,
      selectedKeyframes: state.selectedKeyframes,
      setSelectedKeyframes,
    }),
    [
      state.clip,
      state.clipDisplayName,
      state.selectedBone,
      state.selectedMorph,
      state.selectedKeyframes,
      commit,
      setClipDisplayName,
      setSelectedBone,
      setSelectedMorph,
      setSelectedKeyframes,
    ],
  )

  return createElement(StudioContext.Provider, { value }, children)
}

export function useStudio(): StudioContextValue {
  const ctx = useContext(StudioContext)
  if (ctx == null) throw new Error("useStudio must be used within <Studio>")
  return ctx
}
