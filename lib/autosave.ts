/** IndexedDB persistence for the project autosave — one record, overwritten
 *  on every save. Kept promise-based and dependency-free. */

import type { ProjectFile } from "@/lib/project"

const DB_NAME = "reze-studio"
const STORE = "autosave"
const KEY = "project"

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1)
    req.onupgradeneeded = () => {
      if (!req.result.objectStoreNames.contains(STORE)) req.result.createObjectStore(STORE)
    }
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error ?? new Error("IndexedDB open failed"))
  })
}

async function withStore<T>(mode: IDBTransactionMode, fn: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  const db = await openDb()
  try {
    return await new Promise<T>((resolve, reject) => {
      const tx = db.transaction(STORE, mode)
      const req = fn(tx.objectStore(STORE))
      req.onsuccess = () => resolve(req.result)
      req.onerror = () => reject(req.error ?? new Error("IndexedDB request failed"))
    })
  } finally {
    db.close()
  }
}

export interface AutosaveRecord {
  savedAt: number
  project: ProjectFile
}

export function saveAutosave(project: ProjectFile): Promise<IDBValidKey> {
  const record: AutosaveRecord = { savedAt: Date.now(), project }
  return withStore("readwrite", (store) => store.put(record, KEY))
}

export async function loadAutosave(): Promise<AutosaveRecord | null> {
  const result = await withStore<AutosaveRecord | undefined>("readonly", (store) => store.get(KEY))
  return result && result.project ? result : null
}

export function clearAutosave(): Promise<undefined> {
  return withStore("readwrite", (store) => store.delete(KEY))
}
