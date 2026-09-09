"""Portal Flight — a chain of gates, and you go through them.

The 1.x version drew one ring and a sphere and never went anywhere: nothing
approached, nothing was passed, so the sensation of flight was missing from a
flying effect.  The rebuild is about **transit**.

Six gates hang at their own distances down the run, kept on the CPU.  They
close on the camera, and the frame is depth-sorted so a near gate genuinely
occludes the ones behind it.  When one reaches the camera you **pass through
it**: the frame whites out for an instant, the palette rotates -- you have come
out somewhere else -- and that gate is recycled to the far end of the chain.
So the effect has a cycle with an event in it rather than a loop.

Each gate is a real object: a ring with depth, an energy membrane stretched
across it that thins toward the middle, arcs crackling around the rim, and a
throat glow that grows as it fills the frame.

**The bang.** The beat charges the next gate: its rim brightens, its arcs
double, and it drags the whole chain toward you a little faster.

Audio reactivity:
  - bass    -> approach speed, throat glow, ring thickness
  - mid     -> membrane turbulence, palette drift
  - treble  -> rim arcs, sparks in the throat
  - beat    -> the next gate charges; passing through flashes and shifts hue

Randomised per activation: palette hue, gate spacing and radii, spin
directions, and ``self.time``.

Config keys (under ``[effects.PortalFlight]``):
    speed, zoom, reactivity, glitter, gate_gap
"""
from __future__ import annotations

import moderngl
import numpy as np

from unicornviz.effects.base import AudioData, BaseEffect

__version__ = '0.4.0'

_NGATE = 6
_Z_NEAR = 0.28
_Z_FAR = 16.0

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
uniform float iTravel;      // wrapped distance, drives the throat
uniform float iFlash;       // white-out when a gate is passed
uniform float iCharge;      // the next gate charging on the beat
// Gates, projected on the CPU and sorted far to near. A: xy screen centre,
// z screen radius, w depth fade. B: x hue offset, y spin, z charge, w unused.
uniform vec4  iGateA[6];
uniform vec4  iGateB[6];

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


void main() {
    float A = iResolution.x / max(iResolution.y, 1.0);
    float tb = mod(iTime, 600.0);
    vec2 p = (v_uv - 0.5) * vec2(A, 1.0) / max(iZoom * (1.0 - iPunch * 0.05), 0.05);
    float rad = length(p);
    float ang = atan(p.y, p.x);

    vec3 neonA = palette(iHue);
    vec3 neonB = palette(iHue + 0.38);

    // ── the throat ─────────────────────────────────────────────────────────
    // What lies between the gates: filaments converging on the vanishing
    // point, drawn in log-polar so they flow outward with the run.
    vec3 col = mix(vec3(0.020, 0.016, 0.045), neonB * 0.10, 0.5);
    // Angular frequencies are n/TAU so the noise lattice closes exactly
    // across the +/-PI wrap of atan(). An arbitrary frequency leaves a
    // visible seam along the negative x axis, where the cell index jumps
    // by a non-integer amount.
    vec2 lp = vec2(ang * 3.1831, log(max(rad, 0.02)) * 2.4 - iTravel * 0.9);
    float fil = vnoiseP(lp * vec2(2.0, 1.4), 40.0) * 0.6
              + vnoiseP(lp * vec2(5.0, 3.1), 100.0) * 0.4;
    col += mix(neonA, neonB, 0.5) * pow(clamp(fil, 0.0, 1.0), 3.0) * 0.75
         * (0.35 + iMid * 0.85) * smoothstep(0.02, 0.55, rad);
    // Sparks running out along the throat.
    vec2 sp = vec2(ang * 15.9155, lp.y * 5.0);
    vec2 g = floor(sp);
    if (hashP(g, 100.0) > 0.955) {
        vec2 d = fract(sp) - vec2(hashP(g + 3.1, 100.0), hashP(g + 7.7, 100.0));
        d.y *= 0.28;
        col += mix(vec3(0.9, 0.95, 1.0), neonA, 0.3) * exp(-dot(d, d) * 40.0)
             * (0.4 + iTreble * 0.9) * iGlitter;
    }

    // ── the gates, far to near ─────────────────────────────────────────────
    for (int i = 0; i < 6; i++) {
        vec4 G = iGateA[i];
        if (G.z <= 0.0) continue;
        vec4 B = iGateB[i];
        vec2 q = p - G.xy;
        float r = length(q);
        float R = G.z;
        if (r > R * 2.4) continue;
        vec3 gc = palette(iHue + B.x);
        float fade = G.w;
        float a = atan(q.y, q.x) + B.y;

        // The membrane: a stretched sheet across the opening, thin in the
        // middle so you can see the next gate through it.
        if (r < R) {
            float m = 1.0 - r / R;
            float turb = vnoiseP(vec2(a * 16.0 / TAU, m * 5.0 - iTravel * 1.6), 16.0);
            col = mix(col, gc * (0.35 + turb * 0.9) * fade,
                      smoothstep(0.0, 0.35, m) * 0.55 * (0.5 + iMid * 0.4));
            col += gc * pow(m, 3.0) * 0.55 * fade * (0.5 + iBass * 0.7);
        }

        // The ring itself: a band with depth, brighter where it is charged.
        float ringW = R * (0.085 + 0.035 * B.z + iBass * 0.020);
        float d = abs(r - R) - ringW;
        float lit = 0.75 + 0.55 * sin(a * 3.0 + tb * 0.6) + B.z * 1.2;
        col = mix(col, mix(gc, vec3(1.0), 0.30) * lit * fade,
                  smoothstep(0.004, 0.0, d));
        col += gc * exp(-max(d, 0.0) * 26.0 / max(R, 0.05)) * fade
             * (0.45 + iBass * 0.55 + B.z * 0.9);

        // Arcs crackling around the rim.
        float arc = vnoiseP(vec2(a * 44.0 / TAU, tb * 2.2 + float(i)), 44.0);
        col += mix(gc, vec3(1.0), 0.5)
             * smoothstep(0.010, 0.0, abs(r - R * (1.0 + 0.05 * arc)))
             * step(0.62, arc) * fade * (0.5 + iTreble * 1.1 + B.z * 1.4);
    }

    // ── the bang ───────────────────────────────────────────────────────────
    // Going through: the frame blows out for an instant.
    col += vec3(1.0) * iFlash * iFlash * 1.5;
    col += neonA * iCharge * 0.12 * (0.5 + iBass * 0.8);
    col += neonB * iBeat * 0.07;

    col *= smoothstep(1.60, 0.24, length(v_uv - 0.5));
    col = col / (1.0 + col * 0.45);
    col = pow(clamp(col, 0.0, 1.0), vec3(0.87));
    fragColor = vec4(col, 1.0);
}
"""


class PortalFlight(BaseEffect):
    """Flight down a chain of gates, passing through each one."""

    NAME = 'Portal Flight'
    AUTHOR = 'unicorn-viz'
    TAGS = ['flying', 'cosmic', 'neon', 'intense', 'energetic']
    PING_PONG_FRIENDS = [
        'Warp Drive', 'Nebula Drift', 'Asteroid Run', 'Wormhole',
        'Audio Spectrum',
    ]

    _MIN_ZOOM = 0.60

    def _init(self) -> None:
        self.parameters = {
            'speed': float(self.config.get('speed', 1.0)),
            'zoom': float(self.config.get('zoom', 1.0)),
            'reactivity': float(self.config.get('reactivity', 1.2)),
            'glitter': float(self.config.get('glitter', 1.0)),
            'gate_gap': float(self.config.get('gate_gap', 1.0)),
        }
        self._prog = self._make_program(_VERT, _FRAG)
        self._vao, self._vbo = self._fullscreen_quad(self._prog)

        self._bass = self._mid = self._treble = 0.0
        self._beat = 0.0
        self._punch = 0.0
        self._flash = 0.0
        self._charge = 0.0
        self._hue = float(self.rng.uniform(0.0, 1.0))
        self._travel = float(self.rng.uniform(0.0, 100.0))

        # x, y, z, radius, hue offset, spin rate.
        self._gates = np.zeros((_NGATE, 6), dtype=np.float64)
        gap = (_Z_FAR - _Z_NEAR) / _NGATE
        for i in range(_NGATE):
            self._place_gate(i, z=_Z_NEAR + gap * (i + 0.5))
        self._tbl_a = np.zeros((_NGATE, 4), dtype='f4')
        self._tbl_b = np.zeros((_NGATE, 4), dtype='f4')

    def _place_gate(self, i: int, z: float | None = None) -> None:
        """Hang a gate at the far end of the chain, offset off the flight path."""
        gap = max(0.3, float(self.parameters['gate_gap']))
        self._gates[i] = (
            float(self.rng.uniform(-0.55, 0.55)),
            float(self.rng.uniform(-0.35, 0.35)),
            _Z_FAR * gap if z is None else z,
            float(self.rng.uniform(0.85, 1.55)),
            float(self.rng.uniform(0.0, 1.0)),
            float(self.rng.uniform(-0.9, 0.9)),
        )

    def update(self, dt: float, audio: AudioData) -> None:
        """Close on the chain, and go through whatever reaches the camera."""
        super().update(dt, audio)
        react = max(0.05, float(self.parameters['reactivity']))
        self._bass = float(np.clip(audio.bass_n * react, 0.0, 1.8))
        self._mid = float(np.clip(audio.mid_n * react, 0.0, 1.8))
        self._treble = float(np.clip(audio.treble_n * react, 0.0, 1.8))
        if audio.beat > 0.5:
            self._beat = 1.0
            self._punch = 1.0
            self._charge = 1.0
        self._beat = max(0.0, self._beat - dt * 3.4)
        self._punch = max(0.0, self._punch - dt * 2.4)
        self._charge = max(0.0, self._charge - dt * 1.8)
        self._flash = max(0.0, self._flash - dt * 3.2)

        speed = max(0.05, float(self.parameters['speed']))
        v = speed * (2.2 + self._bass * 3.4 + self._charge * 1.6)
        self._gates[:, 2] -= dt * v
        self._travel = (self._travel + dt * speed * (0.8 + self._bass * 1.6)) % 100.0

        for i in range(_NGATE):
            if self._gates[i, 2] < _Z_NEAR:
                # Through it: flash, come out somewhere else, recycle the gate.
                self._flash = 1.0
                self._hue = float((self._hue + self.rng.uniform(0.08, 0.22)) % 1.0)
                self._place_gate(i)

    def render(self) -> None:
        """Project the chain on the CPU, sorted so near gates occlude far ones."""
        order = np.argsort(-self._gates[:, 2])
        g = self._gates[order]
        z = np.maximum(g[:, 2], 1e-3)
        self._tbl_a[:, 0] = g[:, 0] / z
        self._tbl_a[:, 1] = g[:, 1] / z
        self._tbl_a[:, 2] = np.where(g[:, 2] > _Z_NEAR * 0.6, g[:, 3] / z, 0.0)
        self._tbl_a[:, 3] = np.clip(1.15 - z / _Z_FAR, 0.12, 1.0)
        self._tbl_b[:, 0] = g[:, 4]
        self._tbl_b[:, 1] = g[:, 5] * float(self.time)
        # Only the nearest gate reads as charging: that is the one you are
        # about to go through.
        self._tbl_b[:, 2] = 0.0
        self._tbl_b[-1, 2] = self._charge

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
        p['iFlash'].value = float(self._flash)
        p['iCharge'].value = float(self._charge)
        p['iGateA'].write(self._tbl_a.tobytes())
        p['iGateB'].write(self._tbl_b.tobytes())
        self._vao.render(moderngl.TRIANGLE_STRIP)

    def destroy(self) -> None:
        """Release GL resources."""
        self._vao.release()
        self._vbo.release()
        self._prog.release()
