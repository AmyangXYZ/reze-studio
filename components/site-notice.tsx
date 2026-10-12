"use client"

import { useNotice } from "@/lib/device"
import { useT } from "@/lib/i18n"

const HOME = "https://reze.studio"
const REPO = "https://github.com/AmyangXYZ/reze-studio"
const SITE = HOME.replace("https://", "")

/** A repackaged build or an unknown host: the page stays covered, with the
 *  way to the official site. Nothing underneath is reachable. */
export function SiteNotice() {
  const t = useT()
  if (!useNotice()) return null
  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 p-4 text-white"
      role="alertdialog"
      aria-modal="true"
      aria-labelledby="site-notice-title"
    >
      <div className="w-full max-w-sm rounded-lg border border-white/10 bg-neutral-900 p-5 text-left shadow-xl">
        <h2 id="site-notice-title" className="text-sm font-semibold">
          {t.notice.title}
        </h2>
        <div className="mt-2 space-y-2 text-xs leading-relaxed text-neutral-300">
          <p>{t.notice.body}</p>
          <p>
            {t.notice.source}
            <a href={REPO} target="_blank" rel="noreferrer" className="text-blue-400 underline underline-offset-2">
              {REPO.replace("https://", "")}
            </a>
            {t.notice.sourceEnd}
          </p>
        </div>
        <div className="mt-4 flex justify-end">
          <a
            href={HOME}
            target="_blank"
            rel="noreferrer"
            className="rounded-md bg-blue-400 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-400/90"
          >
            {t.notice.open} {SITE}
          </a>
        </div>
      </div>
    </div>
  )
}
