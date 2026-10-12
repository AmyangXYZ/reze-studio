import type { Metadata } from "next"
import { Instrument_Sans, JetBrains_Mono } from "next/font/google"
import "./globals.css"
import { Analytics } from "@vercel/analytics/next"
import { SiteNotice } from "@/components/site-notice"
import { CrashLogCapture } from "@/components/crash-log-capture"
import { NoStickyFocus } from "@/components/no-sticky-focus"
import { NoNativeContextMenu } from "@/components/no-native-context-menu"

// Instrument Sans for chrome, JetBrains Mono for the timeline, numerals and
// file names. The mono face carries most of this app's information — frame
// numbers, bone names, keyframe counts — so it is chosen for a tall x-height
// and unambiguous digits rather than as a decorative pair with the sans.
const instrumentSans = Instrument_Sans({
  variable: "--font-instrument-sans",
  subsets: ["latin"],
})

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-jetbrains-mono",
  subsets: ["latin"],
})

export const metadata: Metadata = {
  title: "Reze Studio",
  description: "Web-based MMD animation curve editor",
  keywords: ["MMD", "animation", "curve editor", "WebGPU", "Reze Engine"],
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    // The font variables belong on <html>, not on <body>: Tailwind's preflight
    // sets `font-family: var(--default-font-family)` on <html> itself, and a
    // custom property declared on <body> is invisible to its own parent. With
    // them a level down, that declaration referenced an undefined variable, was
    // dropped as invalid, and every element that did not ask for `font-mono` by
    // name inherited the browser's default face instead of the chosen one.
    <html lang="en" className={`dark select-none ${instrumentSans.variable} ${jetbrainsMono.variable}`}>
      <body className="antialiased outline-none">
        {/* Capture starts here so a crash report carries what led up to it. */}
        <CrashLogCapture />
        <NoStickyFocus />
        <NoNativeContextMenu />
        {children}
        <SiteNotice />
        <Analytics />
      </body>
    </html>
  )
}
