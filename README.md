# Reze Studio

Web-native MMD animation editor in the browser (WebGPU).

![Screenshot](./screenshot.png)

## Features

- [x] PMX model and VMD animation loading and rendering with IK and physics
- [x] Timeline with dope sheet and per-channel curve editor
- [x] Bézier interpolation curve editing
- [x] Keyframe insert / delete at playhead
- [x] VMD import/export
- [x] Load user's PMX model from local folder
- [x] Bone list with grouped hierarchy
- [x] Morph list
- [x] Keyboard shortcuts (Space, Arrow keys, Home/End)
- [x] Unsaved-change warning on tab close / refresh (export VMD to keep your work)
- [x] Rotation / translation sliders with direct numeric input
- [x] Morph weight keyframing
- [ ] Animation layers with blend weights and bone masks
- [ ] Custom bone groups with mute / solo toggle
- [ ] Clip operations: cut, copy, paste, mirrored paste (左↔右), import, time stretch
- [ ] Undo / redo
- [ ] 3D transform gizmos in viewport
- [ ] Mocap import (video → VMD)
- [ ] Overleaf style real-time collaboration
- [ ] AI-assisted animation (generative infill, motion retargeting)

## Tech Stack

- **Engine**: [reze-engine](https://github.com/AmyangXYZ/reze-engine) — WebGPU, Ammo.js physics
- **Editor**: Next.js, shadcn/ui

## Architecture

```
<Studio>                          external store — clip + selection (undo/redo target)
  └─ <Playback>                   external store — currentFrame, playing (never touched by rAF ticks)
       └─ <StudioStatusProvider>  external store — pmx name, fps, message (isolated from page re-renders)
            └─ <StudioPage>       layout shell + file handlers
                 ├─ <EngineBridge>          headless — all engine-coupled effects, returns null
                 ├─ <StudioLeftPanel>       memo'd — bone list, morph list, file menu
                 ├─ <StudioViewport>        memo'd — WebGPU <canvas>
                 ├─ <Timeline>              slice-subscribed — dopesheet + curve editor
                 │    └─ <TimelineCanvas>   imperative playhead + drag redraw handles
                 ├─ <PropertiesInspector>   slice-subscribed — pose sliders, morph weight (self-samples via rAF during playback)
                 └─ <StudioStatusFooter>    slice-subscribed — pmx name, fps, clip name
```

### State layers

| Layer        | Lives in                            | Notes                                                                |
| ------------ | ----------------------------------- | -------------------------------------------------------------------- |
| Document     | `context/studio-context.ts`         | External store, slice subscriptions, undo/redo target                |
| Selection    | `context/studio-context.ts`         | Bone, morph, keyframes                                               |
| Transport    | `context/playback-context.ts`       | External store; `currentFrame`, `playing`; single store-owned `currentFrameRef` shared via `usePlaybackFrameRef()` — EngineBridge's rAF writes it, non-subscribing consumers read it |
| Status chrome| `components/studio-status.tsx`      | External store; pmx filename, fps, transient message                 |
| Engine refs  | `StudioPage`                        | `engineRef`, `modelRef`, `canvasRef`                                 |
| View         | local `useState` in `Timeline`      | Zoom, scroll, tab                                                    |
| Chrome       | local `useState` in `StudioPage`    | Menubar, file pick dialog                                            |

### Subscription model

Studio (document/selection), Playback (transport), and StudioStatus (chrome
footer) are all external stores backed by `useSyncExternalStore`. Components
read via `useStudioSelector(s => s.field)` / `usePlaybackSelector(...)` /
`useStudioStatusSelector(...)` so each re-renders only on its own slice, and
write via `use*Actions()` which return stable bags that never cause re-renders.
Wrapping the store's internal `set()` is the single hook needed to add
undo/redo — the reducer pattern is deliberate preparation for that.

### Hot paths — zero React updates while interacting

The three high-frequency interactions (playback, keyframe drag, pose slider
drag) all share the same shape: **mutate refs/objects imperatively, repaint
the canvas via an imperative handle, and touch React exactly once on release.**

- **Playback (60Hz)** — `<EngineBridge>`'s rAF loop reads the engine clock,
  writes the live frame into the playback store's `currentFrameRef` (the
  single ref shared via `usePlaybackFrameRef()`), and calls
  `playheadDrawRef.current(frame)` — a handle `<TimelineCanvas>` exposes that
  repaints the playhead overlay directly. No `setCurrentFrame` per-tick, so
  nothing re-renders, but any non-subscribing consumer (inspector pose sample,
  PMX swap snapshot) still sees the live frame via the ref. Auto-scroll
  (page-turn when the playhead leaves the viewport) lives in the same
  imperative path and only touches React at the rare page-turn boundary. On
  pause, the final frame is flushed to `setCurrentFrame` so the paused view
  matches what was last painted.
- **Live pose / morph readout** — `<PropertiesInspector>` samples the selected
  bone's pose and morph weight in isolated leaf subcomponents. While paused it
  subscribes to `currentFrame` and re-samples on change; while playing it runs
  its own small rAF loop reading `modelRef.current`'s `runtimeSkeleton` /
  `getMorphWeights()` directly, gated by equality so unchanged frames don't
  reconcile. This keeps the 60Hz work out of the parent inspector and out of
  `<StudioPage>` entirely.
- **Keyframe drag** — `<Timeline>`'s move callbacks mutate `kf.frame` /
  channel values / track ordering **in place** and fire `dragRedrawRef.current()`,
  which bumps an internal drag version used by the static-layer cache
  invalidation check and redraws the canvas. `selectedKeyframes` entries are
  mutated in place so highlights follow the drag. On mouseup, a single
  `commit()` clones the track `Map`s → undo/redo snapshot + one
  `model.loadClip` via `<EngineBridge>`.
- **Pose slider drag** — `<PropertiesInspector>`'s `apply*Axis` /
  `applyMorphWeight` functions run in `"preview"` mode per drag tick: mutate
  the matching keyframe (or insert one) in place, then
  `model.loadClip + seek` for the 3D viewport. No `commit()`, so Timeline
  stays static and the Inspector doesn't reconcile. `<AxisSliderRow>` keeps a
  local thumb value during the drag so Radix doesn't snap back to the stale
  controlled prop. On `onValueCommit`, a single clone + `commit()` fires.

### Where each piece lives

| File                                  | Responsibility                                               |
| ------------------------------------- | ------------------------------------------------------------ |
| `app/page.tsx`                        | Next.js entry — mounts all providers + `<StudioPage />`      |
| `context/studio-context.ts`           | Document + selection store, `useStudioSelector`, actions     |
| `context/playback-context.ts`         | Transport store, selectors, actions, `usePlaybackFrameRef`   |
| `components/studio.tsx`               | `StudioPage` — layout, file handlers, menubar, export        |
| `components/studio-status.tsx`        | Status-bar store + `<StudioStatusFooter>`                    |
| `components/engine-bridge.tsx`        | Engine-coupled effects (init, seek, play, rAF playback loop) |
| `components/timeline.tsx`             | Dopesheet + curve editor, imperative playhead / drag redraw  |
| `components/properties-inspector.tsx` | Pose sliders, morph weight, interpolation editor             |
| `components/axis-slider-row.tsx`      | Slider row with preview/commit split + local-drag value      |

## Development

```bash
npm install
npm run dev     # http://localhost:4000
```

## License

GPLv3
