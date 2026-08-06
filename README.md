# flying-01 — Unicorn Viz effect pack

**Version 0.2.0**


First-person flight sensations: Warp Drive, Cloud Surfer, Canyon Run,
Wingsuit Dive, Nebula Drift, Asteroid Run, Portal Flight.

Each effect subclasses `unicornviz.effects.base.BaseEffect` and is auto-discovered.

## Effects

- **Warp Drive** — hyperspace jump: streaking stars rushing past into infinity,
  with a glowing engine core and full FTL "jump" bursts on hard beats.
- **Cloud Surfer** — cruising above a glowing cloud layer at altitude, banking
  through gaps toward a blazing sunset core.
- **Canyon Run** — first-person trench run down a glowing neon canyon, walls
  and a scrolling grid floor racing past, banking hard into turns.
- **Wingsuit Dive** — plummeting past neon mountain ridgelines toward a rising
  city glow, wind motes streaking past on every side.
- **Nebula Drift** — a majestic slow glide deep into a vast, colourful gas
  cloud with embedded star clusters and drifting dust.
- **Asteroid Run** — first-person dash through a dense, tumbling asteroid
  field with near-miss spark bursts on the beat.
- **Portal Flight** — flying through a rapid sequence of glowing ring-gates,
  each one flashing and colour-cycling as you pass through.

All seven are pure GLSL fragment-shader effects (no CPU simulation), fully
audio-reactive on bass/mid/treble/beat, and randomised per activation (palette
hue, layout/variant seeds, starting phase) per the project's effect
randomisation requirements.

## Changelog

- **0.2.0** — float32 time/audio pass (2026-08-03 audit §8): audio-modulated speeds are now CPU-integrated phases (nebula_drift pattern) in Asteroid Run, Canyon Run, Portal Flight, Warp Drive, and Wingsuit Dive — motion accelerates with the music instead of teleporting/strobing every frame at large time phases; Cloud Surfer wraps its fbm time input before the hash (cloud detail no longer collapses into flat bands).
- **0.1.0** — initial release: seven flying effects.
