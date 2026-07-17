"use client"

import { Studio } from "@/context/studio-context"
import { Project } from "@/context/project-context"
import { Playback } from "@/context/playback-context"
import { StudioPage } from "@/components/studio"
import { StudioStatusProvider } from "@/components/studio-status"

export default function Home() {
  return (
    <Studio>
      <Project>
        <Playback>
          <StudioStatusProvider>
            <StudioPage />
          </StudioStatusProvider>
        </Playback>
      </Project>
    </Studio>
  )
}
