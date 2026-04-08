"use client"

import { Studio } from "@/context/studio-context"
import { Playback } from "@/context/playback-context"
import { StudioPage } from "@/components/studio"
import { StudioStatusProvider } from "@/components/studio-status"

export default function Home() {
  return (
    <Studio>
      <Playback>
        <StudioStatusProvider>
          <StudioPage />
        </StudioStatusProvider>
      </Playback>
    </Studio>
  )
}
