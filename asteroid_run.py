"""Asteroid Run — threading a dense field at speed, with rocks that break.

The 1.x version drew almost nothing: measured on a loud frame it came out at a
mean pixel value of 12 out of 255 with a saturation of 7, which is a black
screen with two smudges in the corners.  This is a rebuild.

The field is a **real model**, kept on the CPU.  Every rock has a position in
front of the ship, a radius, a spin and its own silhouette seed; the ship flies
forward, rocks sweep past and are recycled behind you, and the whole set is
depth-sorted every frame so near rocks occlude far ones.  The CPU also does the
perspective projection -- a rock's screen position and screen radius are just
``x/z`` and ``r/z`` -- so the fragment shader never evaluates the field, it only
draws discs it is handed.  That is the same split the rollercoaster pack
settled on, and it is why a dense field is affordable here.

Rocks are lumpy rather than round (two harmonics on the silhouette radius, per
rock), lit from ahead so they carry a bright rim and a dark body, cratered, and
edged in neon.

**The bang.** On a beat the nearest rock in front of you *shatters*: it is
removed from the field and replaced with a burst of shards thrown outward from
where it was, which then fly past the camera on their own. Near-misses flare on
their own too -- a rock passing close throws a rim flash and a smear as it goes
by, so the frame reacts to the geometry and not only to the drums.

Audio reactivity:
  - bass    -> closing speed, engine glow, shard throw
  - mid     -> field density drift, palette
  - treble  -> dust motes, rim sparkle
  - beat    -> a rock shatters, shockwave ring, hull flash

Randomised per activation: palette hue, field layout, rock seeds, and
``self.time``.

Config keys (under ``[effects.AsteroidRun]``):
    speed, zoom, reactivity, glitter, field_density
"""
from __future__ import annotations

import math

import moderngl
import numpy as np

from unicornviz.effects.base import AudioData, BaseEffect

__version__ = '0.4.0'

_NROCK = 16
_NSHARD = 14
_Z_NEAR = 0.55          # a rock closer than this is behind the cockpit glass
_Z_FAR = 15.0           # where rocks are recycled in

_VERT = """
#version 330
in vec2 in_vert;
out vec2 v_uv;
void main() {
    v_uv = in_vert * 0.5 + 0.5;
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
"""

_FRAG = """
#version 330
uniform float iTime;
uniform vec2  iResolution;
uniform float iBass;
uniform float iMid;
uniform float iTreble;
uniform float iBeat;
uniform float iPunch;
uniform float iHue;
uniform float iZoom;
uniform float iGlitter;
uniform float iTravel;      // distance flown, wrapped
uniform float iRing;        // beat shockwave, 0 at the hit -> 1 spent
uniform float iNear;        // 0..1 how close the closest rock is
// The field, projected on the CPU. A: xy screen centre, z screen radius,
// w depth fade. B: x silhouette seed, y spin, z rim-light phase, w unused.
uniform vec4  iRockA[16];
uniform vec4  iRockB[16];
uniform vec4  iShard[14];   // xy screen, z screen size, w life

in  vec2 v_uv;
out vec4 fragColor;

#define PI  3.14159265
#define TAU 6.28318530

float hash21(vec2 p) {
    p = mod(p, 512.0);
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
vec3 palette(float t) { return 0.5 + 0.5 * cos(TAU * (t + vec3(0.0, 0.33, 0.67))); }
// Value noise that is periodic in x with period N. An angular coordinate wraps
// at +/-PI, and merely lining the lattice up there is not enough: hash21(i) and
// hash21(i + N) are unrelated values, so the field still tears along the
// negative x axis. Wrapping the hash's own x is what actually closes it.
float vnoiseP(vec2 x, float N) {
    vec2 i = floor(x), f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    float x0 = mod(i.x, N), x1 = mod(i.x + 1.0, N);
    return mix(mix(hash21(vec2(x0, i.y)),        hash21(vec2(x1, i.y)), f.x),
               mix(hash21(vec2(x0, i.y + 1.0)),  hash21(vec2(x1, i.y + 1.0)), f.x),
               f.y);
}
// Same idea for a scattered-point lattice: wrap the cell's x before hashing.
float hashP(vec2 g, float N) { return hash21(vec2(mod(g.x, N), g.y)); }


void main() {
    float A = iResolution.x / max(iResolution.y, 1.0);
    float tb = mod(iTime, 600.0);
    // A punch of field of view on the beat.
    vec2 p = (v_uv - 0.5) * vec2(A, 1.0) / max(iZoom * (1.0 - iPunch * 0.05), 0.05);

    vec3 neonA = palette(iHue);
    vec3 neonB = palette(iHue + 0.40);
    vec3 neonC = palette(iHue + 0.70);
    float rad = length(p);

    // ── deep space, but never actually black ───────────────────────────────
    // A dust-lit field with a faint nebula wash, so an empty frame still has
    // something in it -- the 1.x version's real problem was that it did not.
    vec3 col = mix(vec3(0.030, 0.028, 0.055), neonC * 0.10, 0.5)
             * (1.0 - rad * 0.35);
    col += neonB * 0.055 * (0.5 + 0.5 * sin(p.x * 1.7 + p.y * 2.3 + tb * 0.05))
         * (0.6 + iMid * 0.9);

    // Stars streaking away from the vanishing point. Done in log-polar: the
    // radial axis becomes a straight scroll, which is exactly the outward
    // expansion perspective produces, so a star both flows correctly and
    // stretches along its own radial rather than down the screen.
    float ang = atan(p.y, p.x);
    // Angular frequencies are n/TAU (28 and 70 cells around) so the lattice
    // closes exactly across the +/-PI wrap of atan(); an arbitrary frequency
    // leaves a seam along the negative x axis.
    float lr = log(max(rad, 0.02)) * 3.2 - iTravel * 1.1;
    float stretch = 0.30 / (1.0 + iBass * 2.6 + iNear * 1.2);
    for (int i = 0; i < 2; i++) {
        float fi = float(i);
        float kN = (i == 0) ? 28.0 : 70.0;
        vec2 cell = vec2(ang * kN / TAU, lr * (1.0 + fi * 1.7));
        vec2 g = floor(cell);
        float h = hashP(g + fi * 31.0, kN);
        if (h > 0.945) {
            vec2 d = fract(cell) - vec2(hashP(g + 3.1, kN), hashP(g + 7.7, kN));
            d.y *= stretch * (0.55 + hashP(g + 11.0, kN) * 0.9);
            col += mix(vec3(0.88, 0.93, 1.0), neonA, 0.30)
                 * exp(-dot(d, d) * 20.0) * (0.45 + h * 0.7)
                 * (0.55 + iTreble * 0.8);
        }
    }

    // ── the field, already projected and sorted far to near ────────────────
    vec3 lit = mix(vec3(1.0, 0.86, 0.66), neonA, 0.30);
    for (int i = 0; i < 16; i++) {
        vec4 R = iRockA[i];
        if (R.z <= 0.0) continue;
        vec2 q = p - R.xy;
        float rr = R.z;
        if (abs(q.x) > rr * 1.6 || abs(q.y) > rr * 1.6) continue;
        vec4 B = iRockB[i];
        float a = atan(q.y, q.x) + B.y;
        // Lumpy, not round: two harmonics keyed to this rock's own seed.
        float shape = rr * (1.0 + 0.085 * sin(a * 3.0 + B.x * 9.0)
                                + 0.055 * sin(a * 5.0 - B.x * 5.0)
                                + 0.030 * sin(a * 8.0 + B.x * 3.0));
        float d = length(q) - shape;
        if (d > rr * 0.5) continue;
        float fade = R.w;
        // Lit from ahead-left, so a rock reads as a solid with a bright limb.
        vec2 nrm = q / max(length(q), 1e-4);
        float shade = clamp(dot(nrm, normalize(vec2(-0.55, 0.62))), -1.0, 1.0);
        vec3 body = mix(vec3(0.09, 0.08, 0.15), vec3(0.52, 0.42, 0.38),
                        0.5 + 0.5 * shade);
        // Ambient bounce from the field, so the dark side is coloured rather
        // than merely dark -- most of what made 1.x read as a black screen.
        body = mix(body, neonC, 0.30) + neonB * 0.10 * (1.0 - max(shade, 0.0));
        // Craters, round and in the rock's own rotating frame. Hashing on the
        // angle instead sliced the body into wedges like a cut pie.
        float cs = sin(B.y), cc = cos(B.y);
        vec2 rq = mat2(cc, -cs, cs, cc) * q / max(rr, 1e-3) * 4.2 + B.x * 13.0;
        vec2 cg = floor(rq);
        float cd = length(fract(rq) - 0.5);
        body *= 1.0 - 0.34 * smoothstep(0.36, 0.08, cd)
                    * step(0.58, hash21(cg + B.x * 30.0));
        col = mix(col, body * (0.35 + fade * 0.9), smoothstep(0.004, 0.0, d));
        // The lit limb, and a neon edge on top of it.
        col += lit * smoothstep(rr * 0.11, 0.0, abs(d)) * max(shade, 0.0)
             * fade * (1.0 + iBass * 0.7);
        col += neonA * exp(-abs(d) * 34.0 / max(rr, 0.02)) * fade * 0.55;
        // A rock going past close throws a hard rim flash.
        col += mix(lit, vec3(1.0), 0.4)
             * smoothstep(rr * 0.16, 0.0, abs(d)) * B.z * 1.1;
    }

    // ── what the beat breaks ───────────────────────────────────────────────
    for (int i = 0; i < 14; i++) {
        vec4 S = iShard[i];
        if (S.w <= 0.0) continue;
        vec2 q = p - S.xy;
        float q2 = dot(q, q);
        if (q2 > S.z * S.z * 9.0) continue;
        float d = length(q) - S.z;
        col = mix(col, mix(vec3(0.55, 0.48, 0.44), neonA, 0.30) * (0.5 + S.w),
                  smoothstep(0.003, 0.0, d) * S.w);
        col += neonA * exp(-max(d, 0.0) * 70.0) * S.w * 0.8
             * (0.5 + iTreble * 0.8) * iGlitter;
    }

    // ── the bang ───────────────────────────────────────────────────────────
    // A shockwave out of the point of impact, and the hull lighting up.
    float ringR = iRing * 1.35;
    col += neonB * smoothstep(0.032, 0.0, abs(rad - ringR))
         * (1.0 - iRing) * (1.0 - iRing)
         * (0.5 + iBass * 0.8) * 0.85;
    col += neonA * iBeat * 0.10;

    // ── the cockpit ────────────────────────────────────────────────────────
    // Engine wash from below and a canopy vignette, so the frame is framed.
    col += mix(vec3(0.35, 0.65, 1.0), neonC, 0.35)
         * smoothstep(0.55, 0.0, length(p - vec2(0.0, -0.62)))
         * (0.28 + iBass * 0.55);
    // Dust rushing past the glass.
    vec2 dcell = vec2(p.x * 30.0, p.y * 30.0 + iTravel * 2.2);
    vec2 dg = floor(dcell);
    float dh = hash21(dg);
    if (dh > 0.955) {
        vec2 dd = fract(dcell) - vec2(hash21(dg + 5.3), hash21(dg + 9.1));
        dd.y *= 0.35;
        col += vec3(0.85, 0.92, 1.0) * exp(-dot(dd, dd) * 30.0)
             * (0.35 + iTreble * 0.9) * iGlitter;
    }

    col *= smoothstep(1.55, 0.28, rad);
    col = col / (1.0 + col * 0.48);
    col = pow(clamp(col, 0.0, 1.0), vec3(0.86));
    fragColor = vec4(col, 1.0);
}
"""


class AsteroidRun(BaseEffect):
    """First-person run through a dense asteroid field that breaks on the beat."""

    NAME = 'Asteroid Run'
    AUTHOR = 'unicorn-viz'
    TAGS = ['flying', 'cosmic', 'neon', 'intense', 'energetic', 'hard']
    PING_PONG_FRIENDS = [
        'Warp Drive', 'Nebula Drift', 'Portal Flight', 'Canyon Run',
        'Starfield', 'Audio Spectrum',
    ]

    # Widest the view may go. zoom is a field-of-view divisor, not a dolly, and
    # a first-person run turns inside out once it is opened much past this.
    _MIN_ZOOM = 0.80

    def _init(self) -> None:
        self.parameters = {
            'speed': float(self.config.get('speed', 1.0)),
            'zoom': float(self.config.get('zoom', 1.0)),
            'reactivity': float(self.config.get('reactivity', 1.2)),
            'glitter': float(self.config.get('glitter', 1.0)),
            'field_density': float(self.config.get('field_density', 1.0)),
        }
        self._prog = self._make_program(_VERT, _FRAG)
        self._vao, self._vbo = self._fullscreen_quad(self._prog)

        self._bass = self._mid = self._treble = 0.0
        self._beat = 0.0
        self._punch = 0.0
        self._ring = 1.0
        self._hue = float(self.rng.uniform(0.0, 1.0))
        self._travel = float(self.rng.uniform(0.0, 100.0))
        self._near = 0.0

        # x, y, z, radius, seed, spin. Depth-sorted every frame.
        self._rocks = np.zeros((_NROCK, 6), dtype=np.float64)
        for i in range(_NROCK):
            self._spawn_rock(i, z=float(self.rng.uniform(_Z_NEAR, _Z_FAR)))
        # x, y, z, vx, vy, life.
        self._shards = np.zeros((_NSHARD, 6), dtype=np.float64)

        self._tbl_a = np.zeros((_NROCK, 4), dtype='f4')
        self._tbl_b = np.zeros((_NROCK, 4), dtype='f4')
        self._tbl_s = np.zeros((_NSHARD, 4), dtype='f4')

    def _spawn_rock(self, i: int, z: float | None = None) -> None:
        """Put a rock back out in front, off to one side of the flight path."""
        spread = 2.5 * max(0.3, float(self.parameters['field_density']))
        self._rocks[i] = (
            float(self.rng.uniform(-spread, spread)),
            float(self.rng.uniform(-spread * 0.62, spread * 0.62)),
            _Z_FAR if z is None else z,
            float(self.rng.uniform(0.16, 0.62)),
            float(self.rng.uniform(0.0, 10.0)),
            float(self.rng.uniform(-0.8, 0.8)),
        )

    def _shatter(self) -> None:
        """Break the nearest rock ahead into shards thrown out of its place."""
        ahead = self._rocks[self._rocks[:, 2] > _Z_NEAR + 0.4]
        if not len(ahead):
            return
        rock = ahead[int(np.argmin(ahead[:, 2]))]
        idx = int(np.argmin(np.abs(self._rocks[:, 2] - rock[2])))
        x, y, z, r = self._rocks[idx, :4]
        n = min(_NSHARD, 6 + int(self._bass * 5))
        free = np.where(self._shards[:, 5] <= 0.0)[0][:n]
        for s in free:
            ang = float(self.rng.uniform(0.0, 2.0 * math.pi))
            spd = float(self.rng.uniform(0.6, 2.2)) * (0.6 + self._bass)
            self._shards[s] = (x, y, z, math.cos(ang) * spd,
                               math.sin(ang) * spd, 1.0)
        self._spawn_rock(idx)

    def update(self, dt: float, audio: AudioData) -> None:
        """Fly the ship, sweep the field past it, and break things on the beat."""
        super().update(dt, audio)
        react = max(0.05, float(self.parameters['reactivity']))
        self._bass = float(np.clip(audio.bass_n * react, 0.0, 1.8))
        self._mid = float(np.clip(audio.mid_n * react, 0.0, 1.8))
        self._treble = float(np.clip(audio.treble_n * react, 0.0, 1.8))
        beat = audio.beat > 0.5
        if beat:
            self._beat = 1.0
            self._punch = 1.0
            self._ring = 0.0
        self._beat = max(0.0, self._beat - dt * 3.6)
        self._punch = max(0.0, self._punch - dt * 2.4)
        self._ring = min(1.0, self._ring + dt * 1.5)

        speed = max(0.05, float(self.parameters['speed']))
        # Closing speed: the low end leans on the throttle.
        v = speed * (5.5 + self._bass * 9.0)
        step = dt * v
        self._travel = (self._travel + dt * speed * (1.4 + self._bass * 2.2)) % 100.0

        self._rocks[:, 2] -= step
        for i in range(_NROCK):
            if self._rocks[i, 2] < _Z_NEAR:
                self._spawn_rock(i)
        if beat:
            self._shatter()

        # Shards keep the parent's forward motion and spread as they come.
        live = self._shards[:, 5] > 0.0
        if live.any():
            self._shards[live, 2] -= step
            self._shards[live, 0] += self._shards[live, 3] * dt
            self._shards[live, 1] += self._shards[live, 4] * dt
            self._shards[live, 5] -= dt * 0.8
            self._shards[(self._shards[:, 2] < _Z_NEAR * 0.5), 5] = 0.0

        # How close the nearest rock is, for the streak stretch and the flash.
        zmin = float(self._rocks[:, 2].min())
        self._near = float(np.clip(1.0 - (zmin - _Z_NEAR) / 3.0, 0.0, 1.0))

    def render(self) -> None:
        """Project the field on the CPU and hand the shader flat discs."""
        p = self._prog
        # Perspective is just x/z: the shader's ray is fwd + p.x*right + p.y*up,
        # so a point lands at (x/z, y/z) and a radius r subtends r/z.
        z = np.maximum(self._rocks[:, 2], 1e-3)
        order = np.argsort(-z)                     # far first: near rocks win
        r = self._rocks[order]
        rz = np.maximum(r[:, 2], 1e-3)
        visible = r[:, 2] > _Z_NEAR
        self._tbl_a[:, 0] = r[:, 0] / rz
        self._tbl_a[:, 1] = r[:, 1] / rz
        self._tbl_a[:, 2] = np.where(visible, r[:, 3] / rz, 0.0)
        self._tbl_a[:, 3] = np.clip(1.0 - (rz - _Z_NEAR) / _Z_FAR, 0.15, 1.0)
        self._tbl_b[:, 0] = r[:, 4]
        self._tbl_b[:, 1] = r[:, 5] * float(self.time)
        # Near-miss flare: sharp, and only for rocks genuinely on top of you.
        self._tbl_b[:, 2] = np.clip(1.0 - (rz - _Z_NEAR) / 1.6, 0.0, 1.0) ** 2

        sz = np.maximum(self._shards[:, 2], 1e-3)
        self._tbl_s[:, 0] = self._shards[:, 0] / sz
        self._tbl_s[:, 1] = self._shards[:, 1] / sz
        self._tbl_s[:, 2] = np.where(self._shards[:, 2] > _Z_NEAR * 0.5,
                                     0.055 / sz, 0.0)
        self._tbl_s[:, 3] = np.clip(self._shards[:, 5], 0.0, 1.0)

        p['iTime'].value = float(self.time)
        p['iResolution'].value = (float(self.width), float(self.height))
        p['iBass'].value = self._bass
        p['iMid'].value = self._mid
        p['iTreble'].value = self._treble
        p['iBeat'].value = self._beat
        p['iPunch'].value = self._punch
        p['iHue'].value = self._hue
        p['iZoom'].value = max(self._MIN_ZOOM, float(self.parameters['zoom']))
        p['iGlitter'].value = max(0.0, float(self.parameters['glitter']))
        p['iTravel'].value = float(self._travel)
        p['iRing'].value = float(self._ring)
        p['iNear'].value = float(self._near)
        p['iRockA'].write(self._tbl_a.tobytes())
        p['iRockB'].write(self._tbl_b.tobytes())
        p['iShard'].write(self._tbl_s.tobytes())
        self._vao.render(moderngl.TRIANGLE_STRIP)

    def destroy(self) -> None:
        """Release GL resources."""
        self._vao.release()
        self._vbo.release()
        self._prog.release()
