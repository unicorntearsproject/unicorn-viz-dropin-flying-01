"""Wingsuit Dive — steep terrain flight, the ground coming up to meet you.

The 1.x version was a flat wash with two ribbon lines and a scatter of hard
yellow squares -- the squares being the "ASCII dots" shape, a bare threshold on
a cell hash filling the whole cell.  This is a rebuild around actual ground.

The terrain is a **table**, built on the CPU: forty samples of ridge height out
in front along the flight path, plus the lateral profile of each ridge.  The
shader marches it -- walk outward until the ray drops below the ground, then
interpolate the crossing so the skyline is a smooth ridge and not a staircase.
Because the ground is real, the dive is real: altitude is integrated, the
terrain rises under you, and the flare at the bottom of each swoop is the
wingsuit actually pulling out rather than a curve being animated.

**The bang.** A beat is a surge: the dive steepens, the ground jumps closer,
wind tears past the visor in long streaks, and vapour rips off the wingtips.
Ridge lines catch a rim light so the terrain reads as edges rather than a
gradient.

Audio reactivity:
  - bass    -> dive rate, terrain relief, wind speed
  - mid     -> haze and palette drift, ridge roughness
  - treble  -> spindrift, rim sparkle on the ridges
  - beat    -> surge, wingtip vapour, horizon flash

Randomised per activation: palette hue, terrain seed, start altitude, heading,
and ``self.time``.

Config keys (under ``[effects.WingsuitDive]``):
    speed, zoom, reactivity, glitter, relief
"""
from __future__ import annotations

import moderngl
import numpy as np

from unicornviz.effects.base import AudioData, BaseEffect

__version__ = '0.4.0'

_SLICES = 40
_Z_NEAR = 0.5
_Z_FAR = 34.0
_NVAPOUR = 12

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
uniform float iTravel;      // ground rushing past, wrapped
uniform float iRoll;        // bank angle
uniform float iPitch;       // how steeply the dive is pointed down
uniform float iAlt;         // 0..1 how close the ground is
uniform float iSurge;       // beat surge envelope
uniform float iSeedT;       // per-activation terrain variant
// Terrain: x = ridge height at that distance, y = lateral tilt,
// z = roughness, w = snow line.
uniform vec4  iTerr[40];
uniform vec4  iVapour[12]; // xy screen, z size, w life

in  vec2 v_uv;
out vec4 fragColor;

#define PI  3.14159265
#define TAU 6.28318530
#define NS  40

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


float sliceZ(int i) {
    float f = float(i) / float(NS - 1);
    return 0.5 + pow(f, 1.6) * 33.5;
}

// Ground height at a lateral offset, given the slice's ridge sample. The ridge
// runs across the flight path, so x carves the valleys.
float ground(vec4 T, float x, float z) {
    // Two noise octaves, not three. This runs once per slice per pixel -- up
    // to forty times -- and the third octave's period falls below a pixel at
    // the grazing angle of a dive anyway, so it cost a third of the march to
    // produce aliasing. Its amplitude is folded into the second.
    return T.x
         + T.y * x
         + T.z * (sin(x * 0.9 + T.w * 3.0) * 0.75
                + sin(x * 2.1 - T.w * 5.0) * 0.42
                + sin(x * 4.3 + T.w * 2.0) * 0.20)
         + T.z * (vnoise(vec2(x * 0.7, z * 0.26 + T.w)) * 1.30
                + vnoise(vec2(x * 2.3, z * 0.9 + T.w)) * 0.72);
}

void main() {
    float A = iResolution.x / max(iResolution.y, 1.0);
    float tb = mod(iTime, 600.0);
    vec2 p = (v_uv - 0.5) * vec2(A, 1.0) / max(iZoom * (1.0 - iPunch * 0.05), 0.05);
    float cs = cos(iRoll), sn = sin(iRoll);
    p = mat2(cs, -sn, sn, cs) * p;
    // Pitch points the frame at the ground. Adding to p.y samples a *higher*
    // world point for the same pixel, which tilts the view up and pushes the
    // horizon down -- the opposite of a dive; it has to subtract.
    p.y -= iPitch;

    vec3 neonA = palette(iHue);
    vec3 neonB = palette(iHue + 0.36);
    vec3 neonC = palette(iHue + 0.66);

    // ── sky above the horizon ──────────────────────────────────────────────
    float up = clamp(p.y * 1.1 + 0.35, 0.0, 1.0);
    vec3 col = mix(mix(vec3(0.42, 0.62, 0.88), neonC, 0.30),
                   mix(vec3(0.04, 0.08, 0.26), neonB, 0.22), pow(up, 0.7));
    col += mix(vec3(1.0, 0.86, 0.60), neonA, 0.30)
         * exp(-length(p - vec2(0.42, 0.30)) * 3.4) * (0.55 + iBass * 0.5);

    // ── march the ground ───────────────────────────────────────────────────
    float hitZ = -1.0;
    vec3 hitP = vec3(0.0);
    float mPrev = 1e9, zPrev = sliceZ(0);
    for (int i = 0; i < NS; i++) {
        float z = sliceZ(i);
        vec4 T = iTerr[i];
        float x = p.x * z, y = p.y * z;
        // Margin above the ground; when it goes negative the ray has landed.
        float m = y - ground(T, x, z);
        if (m < 0.0) {
            float t = clamp(mPrev / max(mPrev - m, 1e-5), 0.0, 1.0);
            hitZ = mix(zPrev, z, t);
            hitP = vec3(p.x * hitZ, p.y * hitZ, hitZ);
            break;
        }
        // A ray that is already well clear of the ground and still climbing
        // can never come back down to it, so there is nothing left to march.
        // Without this every sky pixel walked all forty slices for nothing --
        // which only became the common case once the pitch was raised and the
        // horizon came back into frame, and it tripled the frame cost.
        if (p.y > 0.0 && m > 2.5) break;
        mPrev = m;
        zPrev = z;
    }

    if (hitZ > 0.0) {
        float fade = clamp(1.0 - hitZ / 30.0, 0.0, 1.0);
        float h = hitP.y;
        // Rock, grass and snow by height, then a slope shade so ridges read.
        vec3 low = mix(vec3(0.10, 0.20, 0.12), neonC, 0.22);
        vec3 mid = mix(vec3(0.30, 0.24, 0.16), neonA, 0.20);
        vec3 hi = vec3(0.92, 0.95, 1.0);
        float band = clamp((h + 1.2) * 0.55, 0.0, 1.0);
        vec3 gnd = mix(low, mid, smoothstep(0.15, 0.55, band));
        gnd = mix(gnd, hi, smoothstep(0.72, 0.95, band));
        gnd *= 0.62 + 0.78 * vnoise(vec2(hitP.x * 2.6, hitP.z * 1.1 + iSeedT));
        // Sun on the slopes facing it.
        float slope = ground(iTerr[0], hitP.x + 0.25, hitZ)
                    - ground(iTerr[0], hitP.x - 0.25, hitZ);
        gnd *= 0.58 + 0.90 * clamp(-slope * 1.6, 0.0, 1.0);
        col = gnd * (0.22 + fade * 1.15);
        // Ridge rim: a hit that happens right after a long clear run is a
        // skyline, and lighting that edge is what makes terrain read as
        // ridges rather than as a painted gradient.
        col += neonA * smoothstep(2.2, 0.0, hitZ - zPrev) * fade * 0.30
             * (0.5 + iTreble * 0.8);
        // Cloud shadows drifting across the slopes, which is most of what
        // makes high terrain look like it is under a sky.
        float shadow = smoothstep(0.40, 0.64,
                                  vnoise(vec2(hitP.x * 0.22 + tb * 0.05,
                                              hitP.z * 0.10 - tb * 0.03)));
        col *= 1.0 - 0.38 * shadow;
        // Aerial haze: distance turns the terrain into the sky's colour.
        // Contour banding by height: at the grazing angle of a dive, relief
        // compresses to nothing, and the bands are what let you read the shape
        // of the ground at all.
        float cont = fract(h * 3.4 + 0.5);
        col *= 0.86 + 0.28 * smoothstep(0.44, 0.5, abs(cont - 0.5));
        col = mix(col, mix(vec3(0.55, 0.68, 0.86), neonB, 0.25),
                  (1.0 - fade) * (0.48 + iMid * 0.15));
    }

    // ── the air ────────────────────────────────────────────────────────────
    // Wind tearing past, radially from the direction of travel, stretched by
    // how hard the dive is going.
    float rad = length(p);
    float ang = atan(p.y, p.x);
    // Angular frequencies are n/TAU so the noise lattice closes exactly
    // across the +/-PI wrap of atan(). An arbitrary frequency leaves a
    // visible seam along the negative x axis, where the cell index jumps
    // by a non-integer amount.
    vec2 lp = vec2(ang * 11.1408, log(max(rad, 0.02)) * 5.4 - iTravel * 2.6);
    // Thin. A small stretch factor with a wide gaussian makes each streak a
    // fat white bar rather than a line of air going past.
    float stretch = 0.55 / (1.0 + iBass * 1.4 + iSurge * 1.8);
    vec2 g = floor(lp);
    if (hashP(g, 70.0) > 0.905) {
        vec2 d = fract(lp) - vec2(hashP(g + 3.1, 70.0), hashP(g + 7.7, 70.0));
        d.y *= stretch * (0.6 + hashP(g + 13.0, 70.0) * 0.9);
        col += mix(vec3(0.92, 0.96, 1.0), neonB, 0.25)
             * exp(-dot(d, d) * 105.0) * (0.32 + iTreble * 0.9 + iSurge * 0.7)
             * iGlitter;
    }

    // Vapour ripping off the wingtips.
    for (int i = 0; i < 12; i++) {
        vec4 V = iVapour[i];
        if (V.w <= 0.0) continue;
        vec2 q = p - V.xy;
        float q2 = dot(q, q);
        if (q2 > 0.030) continue;
        col += mix(vec3(1.0), neonB, 0.30)
             * exp(-q2 / max(V.z * V.z, 1e-5)) * V.w * 0.75
             * (0.6 + iTreble * 0.7);
    }

    // The ground getting close washes the whole frame with reflected light.
    col += neonA * iAlt * iAlt * 0.30 * (0.5 + iBass * 0.8);
    col += neonB * iBeat * 0.10;

    col *= smoothstep(1.55, 0.30, length(v_uv - 0.5));
    col = col / (1.0 + col * 0.46);
    col = pow(clamp(col, 0.0, 1.0), vec3(0.86));
    fragColor = vec4(col, 1.0);
}
"""

def _slice_z() -> np.ndarray:
    """Distances of the terrain slices; mirrors ``sliceZ`` in the shader."""
    f = np.arange(_SLICES, dtype=np.float64) / (_SLICES - 1)
    return _Z_NEAR + f ** 1.6 * (_Z_FAR - _Z_NEAR)


class WingsuitDive(BaseEffect):
    """Steep terrain flight with the ground coming up to meet you."""

    NAME = 'Wingsuit Dive'
    AUTHOR = 'unicorn-viz'
    TAGS = ['flying', 'neon', 'intense', 'energetic', 'hard']
    PING_PONG_FRIENDS = [
        'Canyon Run', 'Cloud Surfer', 'Asteroid Run', 'Warp Drive',
        'Audio Spectrum',
    ]

    _MIN_ZOOM = 0.80

    def _init(self) -> None:
        self.parameters = {
            'speed': float(self.config.get('speed', 1.0)),
            'zoom': float(self.config.get('zoom', 1.0)),
            'reactivity': float(self.config.get('reactivity', 1.2)),
            'glitter': float(self.config.get('glitter', 1.0)),
            'relief': float(self.config.get('relief', 1.0)),
        }
        self._prog = self._make_program(_VERT, _FRAG)
        self._vao, self._vbo = self._fullscreen_quad(self._prog)

        self._bass = self._mid = self._treble = 0.0
        self._beat = 0.0
        self._punch = 0.0
        self._surge = 0.0
        self._hue = float(self.rng.uniform(0.0, 1.0))
        self._seed = float(self.rng.uniform(0.0, 24.0))
        self._dist = float(self.rng.uniform(0.0, 500.0))
        self._travel = float(self.rng.uniform(0.0, 100.0))
        self._roll = 0.0
        self._pitch = 0.10
        self._alt = 0.35
        self._altitude = 0.95

        self._z = _slice_z()
        self._tbl_terr = np.zeros((_SLICES, 4), dtype='f4')
        self._vapour = np.zeros((_NVAPOUR, 5), dtype=np.float64)
        self._tbl_vap = np.zeros((_NVAPOUR, 4), dtype='f4')

    def _ridge(self, d: np.ndarray) -> np.ndarray:
        """Ridge height along the flight path at distance ``d``."""
        relief = max(0.2, float(self.parameters['relief']))
        # Height is measured down from the flier, so the whole ridge line
        # rises as altitude bleeds off through the dive.
        return (-self._altitude
                + (0.62 * np.sin(d * 0.055 + self._seed)
                   + 0.34 * np.sin(d * 0.115 - self._seed * 0.6)
                   + 0.19 * np.sin(d * 0.031 + 2.1)) * relief
                * (0.8 + self._bass * 0.45))

    def update(self, dt: float, audio: AudioData) -> None:
        """Fly the dive; the terrain comes to meet it."""
        super().update(dt, audio)
        react = max(0.05, float(self.parameters['reactivity']))
        self._bass = float(np.clip(audio.bass_n * react, 0.0, 1.8))
        self._mid = float(np.clip(audio.mid_n * react, 0.0, 1.8))
        self._treble = float(np.clip(audio.treble_n * react, 0.0, 1.8))
        beat = audio.beat > 0.5
        if beat:
            self._beat = 1.0
            self._punch = 1.0
            self._surge = 1.0
            self._spit_vapour()
        self._beat = max(0.0, self._beat - dt * 3.6)
        self._punch = max(0.0, self._punch - dt * 2.4)
        self._surge = max(0.0, self._surge - dt * 1.5)

        speed = max(0.05, float(self.parameters['speed']))
        v = speed * (9.0 + self._bass * 12.0 + self._surge * 8.0)
        self._dist += dt * v
        self._travel = (self._travel + dt * speed * (2.0 + self._bass * 2.4)) % 100.0

        # The dive breathes: pitch down toward the ground, then flare out.
        cycle = 0.5 - 0.5 * np.cos(self._dist * 0.010)
        # Altitude bleeds off into each swoop and is regained on the flare, so
        # the ground genuinely comes up rather than merely being drawn nearer.
        # Pitch was steep enough that the horizon left the frame for most of
        # the swoop and the shot became ground-only. Roughly halved, and the
        # altitude floor raised, so the skyline stays in view through the dive
        # and the ground reads as something being flown over.
        self._altitude = 1.28 - cycle * 0.66
        self._pitch += (0.05 + cycle * 0.15 + self._surge * 0.06 - self._pitch) \
            * min(1.0, dt * 1.4)
        self._alt = float(np.clip(cycle, 0.0, 1.0))
        # Bank with the terrain's lateral tilt, a little way ahead.
        tilt = float(np.sin(self._dist * 0.021 + self._seed) * 0.42)
        self._roll += (tilt - self._roll) * min(1.0, dt * 1.8)

        live = self._vapour[:, 4] > 0.0
        if live.any():
            self._vapour[live, 0] += self._vapour[live, 2] * dt
            self._vapour[live, 1] += self._vapour[live, 3] * dt
            self._vapour[live, 4] -= dt * 1.6

    def _spit_vapour(self) -> None:
        """Tear vapour off both wingtips."""
        free = np.where(self._vapour[:, 4] <= 0.0)[0][:6]
        for k, s in enumerate(free):
            side = -1.0 if k % 2 == 0 else 1.0
            self._vapour[s] = (
                side * float(self.rng.uniform(0.32, 0.55)),
                float(self.rng.uniform(-0.22, 0.02)),
                side * float(self.rng.uniform(0.25, 0.8)),
                float(self.rng.uniform(-0.5, -0.1)),
                1.0,
            )

    def render(self) -> None:
        """Sample the terrain ahead into the table and draw."""
        d = self._dist + self._z
        relief = max(0.2, float(self.parameters['relief']))
        self._tbl_terr[:, 0] = self._ridge(d)
        self._tbl_terr[:, 1] = 0.16 * np.sin(d * 0.021 + self._seed)
        self._tbl_terr[:, 2] = (0.26 + 0.15 * np.sin(d * 0.045)) * relief
        self._tbl_terr[:, 3] = self._seed

        self._tbl_vap[:, 0] = self._vapour[:, 0]
        self._tbl_vap[:, 1] = self._vapour[:, 1]
        self._tbl_vap[:, 2] = 0.075
        self._tbl_vap[:, 3] = np.clip(self._vapour[:, 4], 0.0, 1.0)

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
        p['iPitch'].value = float(self._pitch)
        p['iAlt'].value = float(self._alt)
        p['iSurge'].value = float(self._surge)
        p['iSeedT'].value = float(self._seed)
        p['iTerr'].write(self._tbl_terr.tobytes())
        p['iVapour'].write(self._tbl_vap.tobytes())
        self._vao.render(moderngl.TRIANGLE_STRIP)

    def destroy(self) -> None:
        """Release GL resources."""
        self._vao.release()
        self._vbo.release()
        self._prog.release()
