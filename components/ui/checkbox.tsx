"use client"

import * as React from "react"
import { CheckIcon } from "lucide-react"
import { Checkbox as CheckboxPrimitive } from "radix-ui"

import { cn } from "@/lib/utils"

function Checkbox({
  className,
  ...props
}: React.ComponentProps<typeof CheckboxPrimitive.Root>) {
  return (
    <CheckboxPrimitive.Root
      data-slot="checkbox"
      className={cn(
        "peer flex size-3.5 shrink-0 items-center justify-center rounded-[3px] border border-border bg-background outline-none transition-colors hover:border-muted-foreground/60 focus-visible:border-muted-foreground focus-visible:ring-1 focus-visible:ring-muted-foreground/40 disabled:cursor-not-allowed disabled:opacity-50 data-[state=checked]:border-muted-foreground/70 data-[state=checked]:bg-muted-foreground/20 data-[state=checked]:text-foreground",
        className
      )}
      {...props}
    >
      <CheckboxPrimitive.Indicator
        data-slot="checkbox-indicator"
        className="flex items-center justify-center text-current"
      >
        <CheckIcon className="size-2.5" strokeWidth={3} />
      </CheckboxPrimitive.Indicator>
    </CheckboxPrimitive.Root>
  )
}

export { Checkbox }
