"""Canyon Run — threading a slot canyon at low altitude, banking through it.

The 1.x version drew two tinted planes and a grid, which read as abstract
geometry rather than as a place.  This is a rebuild around an actual corridor.

The canyon is a **table**, built on the CPU: thirty-two slices out in front of
the ship, each holding the left wall, the right wall, the floor and the rim
height at that distance.  The walls snake, narrow and widen along the run, and
the ship banks into the turns because the bank is taken from the corridor's own
curvature rather than animated on top of it.  The fragment shader marches that
table -- for each pixel, walk outward until the ray leaves the corridor, and
shade the wall it hit -- so the geometry is evaluated thirty-two times per
pixel as array reads, not as trig.  It is the same split the rollercoaster pack
uses, for the same reason.

Because the walls are real, everything else follows from them: the strata band
by world height, the rim catches the sky, the ship's landing lights wash the
rock nearest you, and a wall that comes close throws a proximity flash.

**The bang.** On a beat the walls crack: a burst of sparks tears off the
nearest rock face and blows past the canopy, a shockwave runs down the corridor,
and the strata flare.

Audio reactivity:
  - bass    -> speed, how tight the canyon squeezes, light wash
  - mid     -> strata palette drift, snake amplitude
  - treble  -> grit in the air, rim sparkle
  - beat    -> sparks off the wall, shockwave down the corridor

Randomised per activation: palette hue, the canyon's snake seed, rock variant,
start distance, and ``self.time``.

Config keys (under ``[effects.CanyonRun]``):
    speed, zoom, reactivity, glitter, squeeze
"""
from __future__ import annotations

import math

import moderngl
import numpy as np

from unicornviz.effects.base import AudioData, BaseEffect

__version__ = '0.4.0'

_SLICES = 48            # depth slices of canyon handed to the shader
_Z_NEAR = 0.35
_Z_FAR = 18.0
_NSPARK = 14

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
uniform float iRoll;        // bank angle, radians
uniform float iRing;        // beat shockwave running down the corridor
uniform float iProx;        // 0..1 how close the nearest wall is
uniform float iRock;        // per-activation rock variant
// The corridor: one slice per entry. x = left wall, y = right wall,
// z = floor height, w = rim height. Distances are the fixed _Z table below.
uniform vec4  iWall[48];
uniform vec4  iSpark[14];   // xy screen, z size, w life

in  vec2 v_uv;
out vec4 fragColor;

#define PI  3.14159265
#define TAU 6.28318530
#define NS  48

float hash21(vec2 p) {
    p = mod(p, 512.0);
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
float vnoise(vec2 x) {
    vec2 i = floor(x), f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash21(i),                  hash21(i + vec2(1.0, 0.0)), f.x),
               mix(hash21(i + vec2(0.0, 1.0)), hash21(i + vec2(1.0, 1.0)), f.x), f.y);
}
vec3 palette(float t) { return 0.5 + 0.5 * cos(TAU * (t + vec3(0.0, 0.33, 0.67))); }

// Distance of slice i. Quadratic, so the near corridor -- which fills most of
// the screen -- gets the resolution and the far end costs nothing.
float sliceZ(int i) {
    float f = float(i) / float(NS - 1);
    return 0.35 + pow(f, 1.55) * 17.65;
}

void main() {
    float A = iResolution.x / max(iResolution.y, 1.0);
    float tb = mod(iTime, 600.0);
    vec2 p = (v_uv - 0.5) * vec2(A, 1.0) / max(iZoom * (1.0 - iPunch * 0.05), 0.05);
    // Bank into the turn: the whole frame rolls, which is what selling a
    // canyon run depends on.
    float cs = cos(iRoll), sn = sin(iRoll);
    p = mat2(cs, -sn, sn, cs) * p;

    vec3 neonA = palette(iHue);
    vec3 neonB = palette(iHue + 0.38);
    vec3 neonC = palette(iHue + 0.68);

    // ── sky above the rim ──────────────────────────────────────────────────
    vec3 col = mix(vec3(0.05, 0.07, 0.16), neonC * 0.35, clamp(p.y * 0.8 + 0.4, 0.0, 1.0));
    // A slow aurora over the rim, so the sky above the canyon is not dead.
    col += neonB * 0.09 * smoothstep(0.1, 0.9, p.y) * (0.5 + iMid * 0.8);
    col += neonC * 0.055 * smoothstep(0.55, 1.0, sin(p.x * 2.3 + tb * 0.09))
         * smoothstep(0.0, 0.8, p.y) * (0.5 + iMid * 0.9);
    {   // stars over the canyon
        vec2 cell = p * 44.0;
        vec2 g = floor(cell);
        float h = hash21(g);
        if (h > 0.972 && p.y > 0.0) {
            vec2 d = fract(cell) - vec2(hash21(g + 3.1), hash21(g + 7.7));
            col += vec3(0.85, 0.9, 1.0) * exp(-dot(d, d) * 26.0)
                 * (0.4 + iTreble * 0.7);
        }
    }

    // ── march the corridor ─────────────────────────────────────────────────
    // Walk outward tracking how far inside the corridor the ray still is, and
    // when that margin goes negative, interpolate the crossing between the two
    // slices. Taking the slice itself as the hit -- which is what a plain
    // first-exit test does -- quantises the silhouette to the slice spacing
    // and the walls come out as visible stair steps.
    float hitZ = -1.0;
    float side = 0.0;         // -1 left wall, +1 right wall, 0 floor
    vec3 hitP = vec3(0.0);
    float mPrev = 1e9;
    float zPrev = sliceZ(0);
    float rimPrev = iWall[0].w;
    for (int i = 0; i < NS; i++) {
        float z = sliceZ(i);
        vec4 W = iWall[i];
        float x = p.x * z, y = p.y * z;
        // Walls flare outward as they rise, the way a weathered canyon does;
        // parallel walls read as a corridor cut in concrete.
        float flare = max(y - W.z, 0.0) * 0.20;
        float mL = x - (W.x - flare);
        float mR = (W.y + flare) - x;
        float mF = y - W.z;
        float m = min(mF, min(mL, mR));
        if (m < 0.0) {
            float t = clamp(mPrev / max(mPrev - m, 1e-5), 0.0, 1.0);
            float zh = mix(zPrev, z, t);
            vec3 ph = vec3(p.x * zh, p.y * zh, zh);
            // The rim is interpolated too. Breaking out of the march on a
            // per-slice rim test instead is what left the skyline as a
            // staircase: the silhouette quantised to the slice spacing.
            if (ph.y <= mix(rimPrev, W.w, t)) {
                hitZ = zh;
                hitP = ph;
                side = (mF <= mL && mF <= mR) ? 0.0 : (mL < mR ? -1.0 : 1.0);
            }
            break;
        }
        mPrev = m;
        zPrev = z;
        rimPrev = W.w;
    }

    if (hitZ > 0.0) {
        float fade = clamp(1.0 - (hitZ - 0.35) / 20.0, 0.06, 1.0);
        // Strata: horizontal bands of rock, banded by world height and broken
        // up so the bands do not read as wallpaper.
        float band = hitP.y * 2.6 + vnoise(vec2(hitP.z * 0.7, hitP.y * 1.5)) * 0.8;
        float bi = floor(band);
        float bh = hash21(vec2(bi, iRock * 20.0));
        vec3 rock = mix(vec3(0.30, 0.15, 0.10), vec3(0.52, 0.30, 0.18), bh);
        rock = mix(rock, neonA, 0.26 + bh * 0.30);
        // Grain, and a darker seam between the strata.
        rock *= 0.72 + 0.52 * vnoise(vec2(hitP.z * 4.0, hitP.y * 7.0 + iRock));
        // Erosion channels running down the face, which is what stops a canyon
        // wall reading as a smooth painted cone.
        float gully = vnoise(vec2(hitP.z * 2.4 + iRock * 5.0, hitP.y * 0.5));
        rock *= 0.78 + 0.44 * smoothstep(0.35, 0.75, gully);
        rock *= 1.0 - 0.45 * smoothstep(0.10, 0.0, abs(fract(band) - 0.5) - 0.42);
        // Side walls take a different key light than the floor.
        // One wall faces the light, the other is in shadow, and the floor is
        // darkest -- without that split the corridor reads flat.
        rock *= (side == 0.0) ? 0.42 : (side < 0.0 ? 1.0 : 0.55);
        col = rock * (0.10 + fade * 0.78);
        // The ship's own lights wash the rock closest to you.
        // Landing lights, kept off the near rock: at full strength they blew
        // the closest wall to a flat wash and took the strata with it.
        col += mix(vec3(1.0, 0.92, 0.75), neonB, 0.30)
             * smoothstep(5.0, 1.2, hitZ) * (0.10 + iBass * 0.34);
        // Neon along the rim, where the wall meets the sky.
        float rimD = abs(hitP.y - iWall[0].w);
        col += neonC * smoothstep(0.30, 0.0, rimD) * fade * (0.45 + iMid * 0.7);
        // Depth haze, so far canyon recedes instead of stacking flat.
        col = mix(col, mix(vec3(0.05, 0.07, 0.16), neonC * 0.20, 0.5),
                  (1.0 - fade) * 0.75);
    }

    // ── the bang ───────────────────────────────────────────────────────────
    for (int i = 0; i < 14; i++) {
        vec4 S = iSpark[i];
        if (S.w <= 0.0) continue;
        vec2 q = p - S.xy;
        float q2 = dot(q, q);
        if (q2 > 0.010) continue;
        col += mix(vec3(1.0, 0.80, 0.42), neonA, 0.25)
             * exp(-q2 / max(S.z * S.z, 1e-5)) * S.w
             * (0.7 + iTreble * 1.0) * iGlitter;
    }
    // A shockwave running away down the corridor.
    float rr = length(p);
    col += neonB * smoothstep(0.030, 0.0, abs(rr - iRing * 1.25))
         * (1.0 - iRing) * (1.0 - iRing) * (0.6 + iBass * 0.9);
    // A wall going past close lights the whole canopy.
    col += neonA * iProx * 0.22 * (0.5 + iBass * 0.8);
    col += neonA * iBeat * 0.08;

    // Grit blowing past the canopy.
    vec2 dcell = vec2(p.x * 20.0, p.y * 20.0 + iTravel * 2.6);
    vec2 dg = floor(dcell);
    if (hash21(dg) > 0.986) {
        vec2 dd = fract(dcell) - vec2(hash21(dg + 5.3), hash21(dg + 9.1));
        dd.y *= 0.55;
        col += vec3(1.0, 0.92, 0.78) * exp(-dot(dd, dd) * 34.0)
             * (0.22 + iTreble * 0.7) * iGlitter;
    }

    col *= smoothstep(1.55, 0.30, length(v_uv - 0.5));
    col = col / (1.0 + col * 0.46);
    col = pow(clamp(col, 0.0, 1.0), vec3(0.86));
    fragColor = vec4(col, 1.0);
}
"""


def _slice_z() -> np.ndarray:
    """Distances of the corridor slices; mirrors ``sliceZ`` in the shader."""
    f = np.arange(_SLICES, dtype=np.float64) / (_SLICES - 1)
    return _Z_NEAR + f ** 1.55 * (_Z_FAR - _Z_NEAR)


class CanyonRun(BaseEffect):
    """Low-altitude run down a snaking slot canyon."""

    NAME = 'Canyon Run'
    AUTHOR = 'unicorn-viz'
    TAGS = ['flying', 'neon', 'intense', 'energetic', 'hard']
    PING_PONG_FRIENDS = [
        'Wingsuit Dive', 'Cloud Surfer', 'Asteroid Run', 'Warp Drive',
        'Audio Spectrum',
    ]

    # Widest the view may go: a corridor turns inside out past this.
    _MIN_ZOOM = 0.80

    def _init(self) -> None:
        self.parameters = {
            'speed': float(self.config.get('speed', 1.0)),
            'zoom': float(self.config.get('zoom', 1.0)),
            'reactivity': float(self.config.get('reactivity', 1.2)),
            'glitter': float(self.config.get('glitter', 1.0)),
            'squeeze': float(self.config.get('squeeze', 1.0)),
        }
        self._prog = self._make_program(_VERT, _FRAG)
        self._vao, self._vbo = self._fullscreen_quad(self._prog)

        self._bass = self._mid = self._treble = 0.0
        self._beat = 0.0
        self._punch = 0.0
        self._ring = 1.0
        self._hue = float(self.rng.uniform(0.0, 1.0))
        self._rock = float(self.rng.uniform(0.0, 1.0))
        self._seed = float(self.rng.uniform(0.0, 30.0))
        self._dist = float(self.rng.uniform(0.0, 400.0))
        self._travel = float(self.rng.uniform(0.0, 100.0))
        self._roll = 0.0
        self._prox = 0.0

        self._z = _slice_z()
        self._tbl_wall = np.zeros((_SLICES, 4), dtype='f4')
        self._sparks = np.zeros((_NSPARK, 5), dtype=np.float64)  # x y vx vy life
        self._tbl_spark = np.zeros((_NSPARK, 4), dtype='f4')

    def _centre(self, d: np.ndarray) -> np.ndarray:
        """Where the corridor's centreline sits at distance ``d`` down the run."""
        return (1.05 * np.sin(d * 0.055 + self._seed)
                + 0.55 * np.sin(d * 0.023 - self._seed * 0.7)
                + 0.28 * np.sin(d * 0.101 + 1.7)) * (0.9 + self._mid * 0.5)

    def _half_width(self, d: np.ndarray) -> np.ndarray:
        """Half the corridor's width; it breathes, and the bass squeezes it."""
        squeeze = max(0.2, float(self.parameters['squeeze']))
        base = 1.95 + 0.70 * np.sin(d * 0.037 + self._seed * 1.9)
        return np.maximum(0.85, base - self._bass * 0.55 * squeeze)

    def update(self, dt: float, audio: AudioData) -> None:
        """Fly down the corridor, bank into it, and strike sparks on the beat."""
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
        v = speed * (7.0 + self._bass * 9.0)
        self._dist += dt * v
        self._travel = (self._travel + dt * speed * (1.5 + self._bass * 2.0)) % 100.0

        # Bank from the corridor's own curvature, a little way ahead.
        look = np.array([self._dist + 2.0, self._dist + 7.0])
        c = self._centre(look)
        target = float(np.clip((c[1] - c[0]) * 0.55, -0.65, 0.65))
        self._roll += (target - self._roll) * min(1.0, dt * 2.4)

        if beat:
            self._strike()
        live = self._sparks[:, 4] > 0.0
        if live.any():
            self._sparks[live, 0] += self._sparks[live, 2] * dt
            self._sparks[live, 1] += self._sparks[live, 3] * dt
            self._sparks[live, 4] -= dt * 1.4

    def _strike(self) -> None:
        """Tear a burst of sparks off whichever wall is nearest."""
        near = self._centre(np.array([self._dist + 1.2]))[0]
        half = self._half_width(np.array([self._dist + 1.2]))[0]
        side = -1.0 if near > 0.0 else 1.0
        x0 = (near + side * half) / 1.2
        free = np.where(self._sparks[:, 4] <= 0.0)[0][:8]
        for s in free:
            ang = float(self.rng.uniform(-0.9, 0.9)) + (math.pi if side < 0 else 0.0)
            spd = float(self.rng.uniform(0.8, 2.4)) * (0.6 + self._bass)
            self._sparks[s] = (x0, float(self.rng.uniform(-0.35, 0.25)),
                               math.cos(ang) * spd, math.sin(ang) * spd * 0.5, 1.0)

    def render(self) -> None:
        """Sample the corridor into the table and hand it to the shader."""
        d = self._dist + self._z
        centre = self._centre(d) * 1.6
        half = self._half_width(d)
        # Left and right walls in the ship's frame, so the shader compares
        # against the ray's x directly.
        self._tbl_wall[:, 0] = centre - half - self._centre(np.array([self._dist]))[0]
        self._tbl_wall[:, 1] = centre + half - self._centre(np.array([self._dist]))[0]
        self._tbl_wall[:, 2] = -0.95 - 0.22 * np.sin(d * 0.08 + self._seed)
        self._tbl_wall[:, 3] = 2.60 + 0.95 * np.sin(d * 0.045 - self._seed)

        self._prox = float(np.clip(1.0 - (half[0] - 0.85) / 1.1, 0.0, 1.0))

        self._tbl_spark[:, 0] = self._sparks[:, 0]
        self._tbl_spark[:, 1] = self._sparks[:, 1]
        self._tbl_spark[:, 2] = 0.030
        self._tbl_spark[:, 3] = np.clip(self._sparks[:, 4], 0.0, 1.0)

        p = self._prog
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
        p['iRoll'].value = float(self._roll)
        p['iRing'].value = float(self._ring)
        p['iProx'].value = float(self._prox)
        p['iRock'].value = float(self._rock)
        p['iWall'].write(self._tbl_wall.tobytes())
        p['iSpark'].write(self._tbl_spark.tobytes())
        self._vao.render(moderngl.TRIANGLE_STRIP)

    def destroy(self) -> None:
        """Release GL resources."""
        self._vao.release()
        self._vbo.release()
        self._prog.release()
