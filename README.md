# flying-01 — Unicorn Viz effect pack

**Version 0.4.0**


First-person flight sensations: Warp Drive, Cloud Surfer, Canyon Run,
Wingsuit Dive, Nebula Drift, Asteroid Run, Portal Flight — plus five 1.x
originals kept alongside their 2.0 successors under their own names: Vector
Hallway, Space Dance, Bass Hammer, Cloud Bounce, Space Tunnel.

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

- **0.4.0** — five of the 1.x effects are back as effects in their own right,
  alongside the 2.0 versions that replaced them. The 2.0 pass changed what each
  one *is*, and in five cases the original was worth keeping on its own terms
  rather than being superseded:

  | kept as | was | why it stands alone |
  |---|---|---|
  | **Vector Hallway** | 1.x Canyon Run | flat neon planes and a grid floor — a vector hallway, never rock |
  | **Space Dance** | 1.x Nebula Drift | parallax star layers dancing over a soft cloud, not gas |
  | **Bass Hammer** | 1.x Portal Flight | concentric rings hammered by the low end |
  | **Cloud Bounce** | 1.x Cloud Surfer | a cloud field bouncing on the beat, no horizon |
  | **Space Tunnel** | 1.x Warp Drive | a tunnel of star rings, not a jump |

  The three bugs found during the 2.0 pass were fixed in them on the way in,
  where they applied:

  - **`hash21` input guard** — all five lacked the `mod(p, 512.0)` the rest of
    the packs carry, so a hash input that grows lands past float32's usable
    mantissa and the field collapses into a fixed lattice. Added to all five.
  - **Angular seam** — three hashed a cell index taken from `atan`, which wraps
    at ±π. Fixed in Vector Hallway, Bass Hammer and Space Tunnel by wrapping
    the index to a whole cell count. Space Tunnel was the worst case: its count
    was `26 + layer*14 + iDensity*10`, and iDensity is a per-activation float,
    so the lattice could never close. Vector Hallway's wrap deliberately keeps
    251 cells — the original's own angular density — because rounding to 40
    closed the wrap but visibly thinned the streaks, and thinning them is not
    what this effect is being kept for.
  - **Hard-cell "ASCII dot" fill** — **not present in any of the five.** Space
    Dance already placed a soft round point inside each cell, and Space
    Tunnel's `step()` on a star seed is a per-star brightness, not a spatial
    fill. Nothing to fix; noted so the absence is on the record.

  Wingsuit Dive's pitch is roughly halved and its altitude floor raised: it was
  steep enough that the horizon left the frame for most of each swoop and the
  shot became ground-only.

  `canyon_run.py`'s `_FRAG` no longer swaps a placeholder at import time. A
  string replaced after the fact hides the real shader from every tool that
  reads the source, and fails silently if the pattern stops matching.

- **0.3.0** — full 2.0 pass over all seven. Each one is rebuilt around an
  actual model instead of a screen-space texture, and each gains a beat event
  that is part of the model rather than a flash on top of it.

  *What was wrong.* Asteroid Run measured a mean pixel value of 12/255 at
  saturation 7 on a loud frame — a black screen with two smudges. Canyon Run
  drew tinted planes rather than a canyon. Wingsuit Dive was a flat wash with
  hard yellow squares (the "ASCII dots" shape: a bare threshold on a cell hash,
  filling the whole cell). Nebula Drift sat at saturation 19, which is grey
  mush. Portal Flight never went anywhere — nothing approached and nothing was
  passed. Cloud Surfer sampled one field in screen space, so nothing compressed
  toward a horizon and there was no sense of flight at all.

  *What they are now.* Asteroid Run: a depth-sorted field of lumpy cratered
  rocks, projected on the CPU, that shatter into shards on the beat. Canyon
  Run: a real corridor from a CPU wall table, marched and interpolated so the
  silhouette is smooth, banking from its own curvature. Wingsuit Dive: a
  terrain heightfield with contour relief, altitude bleeding off into each
  swoop. Nebula Drift: two gas populations at two hues, ridged filaments, dust
  lanes, and stars that light the gas *from inside* it — saturation 19 to 52.
  Portal Flight: six gates closing on you, and going through one flashes and
  rotates the palette. Warp Drive: a three-layer log-polar streak field with a
  chromatic tear on the jump. Cloud Surfer: the deck put on a plane, so it
  compresses toward the horizon the way ground does.

  *Shared fixes.* `hash21` gained the `mod(p, 512.0)` guard the other packs
  have. Every effect now takes `zoom` and `glitter` (neither existed before)
  and declares a `_MIN_ZOOM` floor, since these are first-person and the same
  field-of-view trap applies. Log-polar fields use a noise that is genuinely
  periodic in the angular axis — lining the lattice up at the `atan` wrap is
  not enough, because `hash21(i)` and `hash21(i + N)` are unrelated, and the
  field tears along the negative x axis without it.

- **0.2.1** — fixed a crash-on-load bug: Wingsuit Dive (`iBass`) and Asteroid
  Run (`iBass`, `iMid`) declared and set audio uniforms from Python that were
  never referenced in the compiled fragment shader. GLSL compilers strip
  truly-unused uniforms, so moderngl's uniform map didn't contain them and
  the very first `render()` call raised `KeyError`. Rather than just
  deleting the dead uniforms, wired them into the GPU-side effects their own
  docstrings already promised but never implemented: Wingsuit Dive's bass now
  drives the "ground-glow swell as it nears" the docstring describes; Asteroid
  Run's bass now amplifies near-miss camera shake and its mid now widens the
  visible depth range (rock density flicker). Verified with a real moderngl
  render across quiet/moderate/loud-beat audio and a high-iTime stress test
  for both effects.
- **0.2.0** — float32 time/audio pass (2026-08-03 audit §8): audio-modulated speeds are now CPU-integrated phases (nebula_drift pattern) in Asteroid Run, Canyon Run, Portal Flight, Warp Drive, and Wingsuit Dive — motion accelerates with the music instead of teleporting/strobing every frame at large time phases; Cloud Surfer wraps its fbm time input before the hash (cloud detail no longer collapses into flat bands).
- **0.1.0** — initial release: seven flying effects.
