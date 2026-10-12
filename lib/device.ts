"use client"

// Is this page running somewhere other than the site it was built for: a
// repackaged WebView, an unknown host? Asked once on mount; the answer does not
// change while the page lives.

import { useSyncExternalStore } from "react"

const P =
  "eyJoIjpbInJlemUuc3R1ZGlvIiwid3d3LnJlemUuc3R1ZGlvIiwibG9jYWxob3N0IiwiMTI3LjAuMC4xIl0sInMiOiIudmVyY2VsLmFwcCIsImUiOiJFbGVjdHJvbi8iLCJ3IjoiOyB3dlxcKSIsImIiOiJBbmRyb2lkIFdlYlZpZXciLCJhIjoiTWljcm9NZXNzZW5nZXJ8V2VDaGF0fFFRL3xNUVFCcm93c2VyfFdlaWJvfEJpbGlBcHB8YmlsaS18YXdlbWV8QXdlbWV8TmV3c0FydGljbGV8eGhzZGlzY292ZXJ8RGluZ1RhbGt8TGFya3xGZWlzaHV8VGVsZWdyYW18RGlzY29yZHxGQkFOfEZCQVZ8SW5zdGFncmFtfExpbmUvfFR3aXR0ZXIifQ=="

function noticeDue(): boolean {
  try {
    const p = JSON.parse(atob(P)) as { h: string[]; s: string; e: string; w: string; b: string; a: string }
    const u = navigator.userAgent
    if (u.includes(p.e)) return false
    const h = window.location.hostname
    if (!p.h.includes(h) && !h.endsWith(p.s)) return true
    const d = (navigator as { userAgentData?: { brands?: { brand: string }[] } }).userAgentData
    const marked = new RegExp(p.w).test(u) || !!d?.brands?.some((x) => x.brand === p.b)
    return marked && !new RegExp(p.a).test(u)
  } catch {
    return false
  }
}

const never = () => () => {}

/** Server render answers false: the notice is a client decision. */
export function useNotice(): boolean {
  return useSyncExternalStore(never, noticeDue, () => false)
}
