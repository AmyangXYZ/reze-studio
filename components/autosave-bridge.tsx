"use client"

/** Headless autosave: restores the last project from IndexedDB once the
 *  model is ready, then debounce-saves the arrangement on every change.
 *  Keyframe edits reach this through the active-clip mirror (each commit
 *  replaces the active library entry, changing `library`). */

import { useEffect, useRef } from "react"
import { useProjectSelector } from "@/context/project-context"
import { useStudioStatusActions } from "@/components/studio-status"
import { loadAutosave, saveAutosave } from "@/lib/autosave"
import { decodeProject, encodeProject, type ProjectSnapshot } from "@/lib/project"

const SAVE_DEBOUNCE_MS = 2000

interface AutosaveBridgeProps {
  studioReady: boolean
  /** Same routine Project import uses — replaces arrangement + active clip. */
  applyProjectSnapshot: (snapshot: ProjectSnapshot, fallbackName: string) => void
}

export function AutosaveBridge({ studioReady, applyProjectSnapshot }: AutosaveBridgeProps) {
  const library = useProjectSelector((s) => s.library)
  const tracks = useProjectSelector((s) => s.tracks)
  const activeClipId = useProjectSelector((s) => s.activeClipId)
  const { setMessage } = useStudioStatusActions()

  /** Gate saves until the restore attempt finishes, so the boot-seeded
   *  sample state can't clobber a real saved project. */
  const restoreDoneRef = useRef(false)
  const restoreStartedRef = useRef(false)

  useEffect(() => {
    if (!studioReady || restoreStartedRef.current) return
    restoreStartedRef.current = true
    void (async () => {
      try {
        const record = await loadAutosave()
        if (record) {
          const snapshot = decodeProject(record.project)
          // Empty projects aren't worth restoring over the sample-clip boot.
          if (snapshot.library.some((c) => c.clip.boneTracks.size > 0 || c.clip.morphTracks.size > 0)) {
            applyProjectSnapshot(snapshot, snapshot.name)
            setMessage("Restored last session")
          }
        }
      } catch (e) {
        console.warn("[autosave] restore failed:", e)
      } finally {
        restoreDoneRef.current = true
      }
    })()
  }, [studioReady, applyProjectSnapshot, setMessage])

  useEffect(() => {
    if (!restoreDoneRef.current || library.length === 0) return
    const timer = setTimeout(() => {
      try {
        const encoded = encodeProject({
          name: "autosave",
          modelRef: { name: "" },
          library,
          tracks,
          activeClipId,
        })
        void saveAutosave(encoded)
          .then(() => {
            const t = new Date()
            const hh = String(t.getHours()).padStart(2, "0")
            const mm = String(t.getMinutes()).padStart(2, "0")
            setMessage(`Autosaved ${hh}:${mm}`)
          })
          .catch((e) => console.warn("[autosave] save failed:", e))
      } catch (e) {
        console.warn("[autosave] encode failed:", e)
      }
    }, SAVE_DEBOUNCE_MS)
    return () => clearTimeout(timer)
  }, [library, tracks, activeClipId, setMessage])

  return null
}
