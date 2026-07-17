"use client"

/** Clip-arrangement project state — library, tracks, editor mode. External
 *  store following the studio-context pattern so consumers subscribe to
 *  slices. The studio store stays the undo target for keyframe edits; this
 *  store owns the arrangement (no undo history for arrangement ops yet).
 *
 *  Invariant: the studio store's `clip` is the working copy of the library
 *  entry `activeClipId` points at. StudioPage mirrors every commit back via
 *  `updateActiveClipData`, so baking always reads the latest edits. */

import {
  createContext,
  createElement,
  useContext,
  useMemo,
  useRef,
  useSyncExternalStore,
  type ReactNode,
} from "react"
import type { AnimationClip } from "reze-engine"
import {
  bakeTracks,
  emptyTrack,
  newId,
  type ClipId,
  type LibraryClip,
  type ProjectSnapshot,
  type Track,
} from "@/lib/project"

export type StudioMode = "bone" | "clip"

export type ProjectState = {
  mode: StudioMode
  library: LibraryClip[]
  tracks: Track[]
  activeClipId: ClipId | null
  selectedPlacementId: string | null
}

export type ProjectActions = {
  setMode: (mode: StudioMode) => void
  /** Boot-time: if the library is empty, register the sample clip and place
   *  it on Track 1 so first launch lands in the documented default state.
   *  Idempotent — a second call (strict-mode remount) is a no-op. */
  seedLibrary: (name: string, clip: AnimationClip) => void
  /** Add a clip to the library without touching the active edit. Returns id. */
  importClip: (name: string, clip: AnimationClip) => ClipId
  /** "Open VMD" semantics: replace the active library clip's data + name
   *  (placements stay). Creates an entry if there is no active clip. */
  openClipIntoActive: (name: string, clip: AnimationClip) => ClipId
  /** Mirror a studio-store commit into the active library entry. */
  updateActiveClipData: (clip: AnimationClip) => void
  setActiveClip: (id: ClipId) => void
  renameClip: (id: ClipId, name: string) => void
  /** Remove from library + all placements. Active falls back to null. */
  removeClip: (id: ClipId) => void
  addTrack: () => void
  removeTrack: (id: string) => void
  toggleTrackMute: (id: string) => void
  toggleTrackSolo: (id: string) => void
  placeClip: (clipId: ClipId, trackId: string, startFrame: number) => void
  updatePlacement: (
    id: string,
    patch: Partial<{ startFrame: number; length: number | undefined; timeScale: number | undefined; trackId: string }>,
  ) => void
  duplicatePlacement: (id: string) => void
  deletePlacement: (id: string) => void
  setSelectedPlacement: (id: string | null) => void
  /** Replace the whole arrangement (project import / autosave restore). */
  loadProject: (snapshot: Pick<ProjectSnapshot, "library" | "tracks" | "activeClipId">) => void
  /** New Project: one empty track, one empty active clip (the same object the
   *  caller just put in the studio store, so the mirror effect no-ops). */
  resetProject: (activeClip: AnimationClip, activeName: string) => ClipId
}

const INITIAL_STATE: ProjectState = {
  mode: "bone",
  library: [],
  tracks: [emptyTrack("Track 1")],
  activeClipId: null,
  selectedPlacementId: null,
}

type ProjectStore = {
  getState: () => ProjectState
  subscribe: (listener: () => void) => () => void
  actions: ProjectActions
}

function createProjectStore(): ProjectStore {
  let state = INITIAL_STATE
  const listeners = new Set<() => void>()

  const set = (next: ProjectState) => {
    if (next === state) return
    state = next
    listeners.forEach((l) => l())
  }

  const updateTracks = (fn: (tracks: Track[]) => Track[]) => {
    set({ ...state, tracks: fn(state.tracks) })
  }

  const actions: ProjectActions = {
    setMode: (mode) => {
      if (mode === state.mode) return
      set({ ...state, mode })
    },
    seedLibrary: (name, clip) => {
      if (state.library.length > 0) return
      const id = newId()
      const entry: LibraryClip = { id, name, clip }
      const tracks = state.tracks.length > 0 ? state.tracks : [emptyTrack("Track 1")]
      const first = tracks[0]
      set({
        ...state,
        library: [entry],
        tracks: [
          { ...first, placements: [{ id: newId(), clipId: id, startFrame: 0 }, ...first.placements] },
          ...tracks.slice(1),
        ],
        activeClipId: id,
      })
    },
    importClip: (name, clip) => {
      const id = newId()
      set({ ...state, library: [...state.library, { id, name, clip }] })
      return id
    },
    openClipIntoActive: (name, clip) => {
      const activeId = state.activeClipId
      if (activeId != null && state.library.some((c) => c.id === activeId)) {
        set({
          ...state,
          library: state.library.map((c) => (c.id === activeId ? { ...c, name, clip } : c)),
        })
        return activeId
      }
      const id = newId()
      set({ ...state, library: [...state.library, { id, name, clip }], activeClipId: id })
      return id
    },
    updateActiveClipData: (clip) => {
      const activeId = state.activeClipId
      if (activeId == null) return
      const entry = state.library.find((c) => c.id === activeId)
      if (!entry || entry.clip === clip) return
      set({
        ...state,
        library: state.library.map((c) => (c.id === activeId ? { ...c, clip } : c)),
      })
    },
    setActiveClip: (id) => {
      if (id === state.activeClipId || !state.library.some((c) => c.id === id)) return
      set({ ...state, activeClipId: id })
    },
    renameClip: (id, name) => {
      const trimmed = name.trim()
      if (!trimmed) return
      set({ ...state, library: state.library.map((c) => (c.id === id ? { ...c, name: trimmed } : c)) })
    },
    removeClip: (id) => {
      set({
        ...state,
        library: state.library.filter((c) => c.id !== id),
        tracks: state.tracks.map((t) => ({
          ...t,
          placements: t.placements.filter((p) => p.clipId !== id),
        })),
        activeClipId: state.activeClipId === id ? null : state.activeClipId,
        selectedPlacementId: null,
      })
    },
    addTrack: () => {
      updateTracks((tracks) => [...tracks, emptyTrack(`Track ${tracks.length + 1}`)])
    },
    removeTrack: (id) => {
      if (state.tracks.length <= 1) return
      set({
        ...state,
        tracks: state.tracks.filter((t) => t.id !== id),
        selectedPlacementId: null,
      })
    },
    toggleTrackMute: (id) => {
      updateTracks((tracks) => tracks.map((t) => (t.id === id ? { ...t, mute: !t.mute } : t)))
    },
    toggleTrackSolo: (id) => {
      updateTracks((tracks) => tracks.map((t) => (t.id === id ? { ...t, solo: !t.solo } : t)))
    },
    placeClip: (clipId, trackId, startFrame) => {
      if (!state.library.some((c) => c.id === clipId)) return
      const placement = { id: newId(), clipId, startFrame: Math.max(0, Math.round(startFrame)) }
      set({
        ...state,
        tracks: state.tracks.map((t) =>
          t.id === trackId ? { ...t, placements: [...t.placements, placement] } : t,
        ),
        selectedPlacementId: placement.id,
      })
    },
    updatePlacement: (id, patch) => {
      const tracks = state.tracks
      const fromTrack = tracks.find((t) => t.placements.some((p) => p.id === id))
      if (!fromTrack) return
      const prev = fromTrack.placements.find((p) => p.id === id)!
      const next = { ...prev }
      if (patch.startFrame != null) next.startFrame = Math.max(0, Math.round(patch.startFrame))
      if ("length" in patch) {
        if (patch.length == null) delete next.length
        else next.length = Math.max(1, Math.round(patch.length))
      }
      if ("timeScale" in patch) {
        if (patch.timeScale == null || patch.timeScale === 1) delete next.timeScale
        else next.timeScale = Math.min(8, Math.max(0.1, patch.timeScale))
      }
      const targetTrackId = patch.trackId ?? fromTrack.id
      updateTracks((ts) =>
        ts.map((t) => {
          const without = t.placements.filter((p) => p.id !== id)
          if (t.id === targetTrackId) return { ...t, placements: [...without, next] }
          return without.length === t.placements.length ? t : { ...t, placements: without }
        }),
      )
    },
    duplicatePlacement: (id) => {
      const tracks = state.tracks
      const track = tracks.find((t) => t.placements.some((p) => p.id === id))
      if (!track) return
      const src = track.placements.find((p) => p.id === id)!
      const lib = state.library.find((c) => c.id === src.clipId)
      const span = lib ? Math.max(1, Math.round(lib.clip.frameCount * (src.timeScale ?? 1))) : 1
      const copy = { ...src, id: newId(), startFrame: src.startFrame + (src.length ?? span) }
      set({
        ...state,
        tracks: tracks.map((t) =>
          t.id === track.id ? { ...t, placements: [...t.placements, copy] } : t,
        ),
        selectedPlacementId: copy.id,
      })
    },
    deletePlacement: (id) => {
      updateTracks((ts) =>
        ts.map((t) =>
          t.placements.some((p) => p.id === id)
            ? { ...t, placements: t.placements.filter((p) => p.id !== id) }
            : t,
        ),
      )
      if (state.selectedPlacementId === id) set({ ...state, selectedPlacementId: null })
    },
    setSelectedPlacement: (id) => {
      if (id === state.selectedPlacementId) return
      set({ ...state, selectedPlacementId: id })
    },
    loadProject: (snapshot) => {
      set({
        ...state,
        library: snapshot.library,
        tracks: snapshot.tracks.length > 0 ? snapshot.tracks : [emptyTrack("Track 1")],
        activeClipId: snapshot.activeClipId,
        selectedPlacementId: null,
      })
    },
    resetProject: (activeClip, activeName) => {
      const id = newId()
      set({
        ...state,
        library: [{ id, name: activeName, clip: activeClip }],
        tracks: [emptyTrack("Track 1")],
        activeClipId: id,
        selectedPlacementId: null,
      })
      return id
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

const ProjectStoreContext = createContext<ProjectStore | null>(null)

export function Project({ children }: { children: ReactNode }) {
  const storeRef = useRef<ProjectStore | null>(null)
  if (storeRef.current == null) storeRef.current = createProjectStore()
  return createElement(ProjectStoreContext.Provider, { value: storeRef.current }, children)
}

function useProjectStore(): ProjectStore {
  const store = useContext(ProjectStoreContext)
  if (store == null) throw new Error("useProject* must be used within <Project>")
  return store
}

export function useProjectSelector<T>(selector: (state: ProjectState) => T): T {
  const store = useProjectStore()
  const getSnapshot = () => selector(store.getState())
  return useSyncExternalStore(store.subscribe, getSnapshot, getSnapshot)
}

export function useProjectActions(): ProjectActions {
  return useProjectStore().actions
}

/** Flat clip for the engine while in clip mode; null in bone mode. Memoized
 *  per arrangement change — every store mutation replaces `tracks`/`library`
 *  references, so the deps are exact. */
export function useBakedArrangement(): AnimationClip | null {
  const mode = useProjectSelector((s) => s.mode)
  const tracks = useProjectSelector((s) => s.tracks)
  const library = useProjectSelector((s) => s.library)
  return useMemo(() => (mode === "clip" ? bakeTracks(tracks, library) : null), [mode, tracks, library])
}
