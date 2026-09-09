"""Warp Drive — the jump to light speed, and the moment it happens.

The 1.x version drew a fan of thin lines from screen centre.  It had no depth,
so it read as a starburst rather than as motion through anything, and the
"jump" was a brightness change.  The rebuild is about **the transition**.

Stars live in a log-polar field: the angular axis wraps and the radial axis is
``log(r)``, which is exactly the outward expansion perspective produces, so a
constant scroll along it is a correct flow rather than an approximation.  Three
layers at different depths scroll at different rates, which is what gives the
tunnel parallax and therefore depth.  The noise is periodic in the angular
axis, or the field tears along the negative x axis where ``atan`` wraps.

Cruise draws points.  As the drive spools up, each point stretches along its
own radial until it is a line -- the streak is the *same star*, elongated,
which is the read the effect wants and the reason length is a property of the
star rather than a separate layer of lines.

**The bang.** A beat fires the jump: the streaks stretch to full length, the
field splits into red, green and blue at slightly different stretches -- a
chromatic tear, because the three channels arrive at different times -- the
core blows out, and a shock ring runs out through the field.

Audio reactivity:
  - bass    -> warp rate, streak length, core swell
  - mid     -> palette drift, field density
  - treble  -> glints on near stars
  - beat    -> the jump: stretch, chromatic tear, white-out, shock ring

Randomised per activation: palette hue, field density, start distance, and
``self.time``.

Config keys (under ``[effects.WarpDrive]``):
    speed, zoom, reactivity, glitter, jump_force
"""
from __future__ import annotations

import moderngl
import numpy as np

from unicornviz.effects.base import AudioData, BaseEffect

__version__ = '0.4.0'

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
uniform float iWarp;        // wrapped warp distance
uniform float iJump;        // jump envelope, 1 at the beat -> 0
uniform float iRing;        // shock ring radius phase
uniform float iDensity;
uniform float iJumpForce;

in  vec2 v_uv;
out vec4 fragColor;

#define TAU 6.28318530

float hash21(vec2 p) {
    p = mod(p, 512.0);
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
// Hash wrapped in x, so an angular cell index can cross the +/-PI branch cut
// of atan() without the field tearing along the negative x axis.
float hashP(vec2 g, float N) { return hash21(vec2(mod(g.x, N), g.y)); }
vec3 palette(float t) { return 0.5 + 0.5 * cos(TAU * (t + vec3(0.0, 0.33, 0.67))); }

// One depth layer of stars, returned as brightness. `stretch` scales the
// radial extent of each point: small values draw a long streak, because it is
// the same star smeared along its own radial rather than a separate line.
float layer(float ang, float lr, float N, float seed, float stretch,
            float dens, out float glint) {
    vec2 cell = vec2(ang * N / TAU, lr);
    vec2 base = floor(cell);
    float sum = 0.0;
    glint = 0.0;
    // Neighbouring cells along the radial axis have to be sampled too. A
    // streak stretched over several cells is otherwise clipped at its own
    // cell's edge, which is what turned the stars into hard-ended dashes
    // instead of streaks that taper away behind you.
    for (int j = -2; j <= 2; j++) {
        vec2 g = base + vec2(0.0, float(j));
        float h = hashP(g + seed, N);
        if (h < 1.0 - dens) continue;
        vec2 d = cell - g - vec2(hashP(g + 3.1, N), hashP(g + 7.7, N));
        d.y *= stretch * (0.55 + hashP(g + 11.0, N) * 0.95);
        float v = exp(-dot(d, d) * 135.0);
        sum += v * (0.45 + h * 0.75);
        glint += v * step(0.985, h);
    }
    return sum;
}

void main() {
    float A = iResolution.x / max(iResolution.y, 1.0);
    float tb = mod(iTime, 600.0);
    vec2 p = (v_uv - 0.5) * vec2(A, 1.0) / max(iZoom * (1.0 - iPunch * 0.06), 0.05);
    float rad = length(p);
    float ang = atan(p.y, p.x);

    vec3 neonA = palette(iHue);
    vec3 neonB = palette(iHue + 0.42);
    float jump = clamp(iJump, 0.0, 1.0) * clamp(iJumpForce, 0.0, 3.0);

    // Cruise draws points; the drive stretches them into lines.
    float stretch = 0.85 / (1.0 + iBass * 2.4 + jump * 7.0);
    float dens = clamp(0.055 + iDensity * 0.045 + iMid * 0.020, 0.01, 0.30);

    vec3 col = mix(vec3(0.016, 0.014, 0.040), neonB * 0.06, 0.5);
    // A faint drifting wash behind the field, so the space between the stars
    // is space rather than a flat black plate.
    col += neonB * 0.045
         * (0.5 + 0.5 * sin(p.x * 2.1 + p.y * 1.6 + tb * 0.07))
         * (0.5 + iMid * 0.8) * smoothstep(0.0, 0.8, rad);

    // Three depth layers, scrolling at different rates: the parallax between
    // them is what makes this a tunnel instead of a starburst.
    float lr = log(max(rad, 0.02)) * 3.4;
    float g0, g1, g2;
    float s0 = layer(ang, lr * 1.00 - iWarp * 1.30, 28.0, 0.0, stretch, dens, g0);
    float s1 = layer(ang, lr * 1.55 - iWarp * 2.05, 44.0, 17.0, stretch * 0.8, dens, g1);
    float s2 = layer(ang, lr * 2.30 - iWarp * 3.10, 70.0, 41.0, stretch * 0.62, dens, g2);

    // The chromatic tear: during a jump the channels arrive at different
    // stretches, so the streaks split into colour at their leading edges.
    if (jump > 0.02) {
        float sr, sg, sb, gx;
        sr = layer(ang, lr * 1.55 - iWarp * 2.05, 44.0, 17.0, stretch * 0.66, dens, gx);
        sg = s1;
        sb = layer(ang, lr * 1.55 - iWarp * 2.05, 44.0, 17.0, stretch * 0.98, dens, gx);
        col += vec3(sr, sg, sb) * jump * 1.5;
    }

    vec3 star = mix(vec3(0.86, 0.92, 1.0), neonA, 0.28);
    col += star * (s0 * 1.05 + s1 * 1.30 + s2 * 1.55) * (0.75 + iBass * 0.55);
    col += mix(vec3(1.0), neonB, 0.35) * (g0 + g1 + g2) * 1.6
         * (0.4 + iTreble * 1.2) * iGlitter;

    // ── the drive itself ───────────────────────────────────────────────────
    // A core at the vanishing point, swelling with the low end and blowing out
    // on a jump.
    float core = exp(-rad * rad * (150.0 - jump * 90.0));
    col += mix(vec3(0.75, 0.90, 1.0), neonA, 0.35) * core
         * (0.45 + iBass * 0.75 + jump * 2.2);
    col += neonB * exp(-rad * 6.5) * 0.10 * (0.35 + iBass * 0.7);

    // Shock ring running out through the field.
    // Thin and short-lived, or it reads as a standing ring rather than as a
    // shock leaving the drive.
    col += mix(neonA, vec3(1.0), 0.35)
         * smoothstep(0.022, 0.0, abs(rad - iRing * 1.45))
         * pow(1.0 - iRing, 3.0) * (0.7 + iBass * 0.9);

    // The white-out at the instant of the jump.
    col += vec3(1.0) * pow(jump, 3.0) * 0.9;
    col += neonA * iBeat * 0.06;

    col *= smoothstep(1.60, 0.22, length(v_uv - 0.5));
    col = col / (1.0 + col * 0.44);
    col = pow(clamp(col, 0.0, 1.0), vec3(0.87));
    fragColor = vec4(col, 1.0);
}
"""


class WarpDrive(BaseEffect):
    """Hyperspace: points stretch to streaks, then the jump tears them open."""

    NAME = 'Warp Drive'
    AUTHOR = 'unicorn-viz'
    TAGS = ['flying', 'cosmic', 'neon', 'intense', 'energetic', 'hard']
    PING_PONG_FRIENDS = [
        'Portal Flight', 'Nebula Drift', 'Asteroid Run', 'Starfield',
        'Wormhole', 'Audio Spectrum',
    ]

    _MIN_ZOOM = 0.55

    def _init(self) -> None:
        self.parameters = {
            'speed': float(self.config.get('speed', 1.0)),
            'zoom': float(self.config.get('zoom', 1.0)),
            'reactivity': float(self.config.get('reactivity', 1.2)),
            'glitter': float(self.config.get('glitter', 1.0)),
            'jump_force': float(self.config.get('jump_force', 1.0)),
        }
        self._prog = self._make_program(_VERT, _FRAG)
        self._vao, self._vbo = self._fullscreen_quad(self._prog)

        self._bass = self._mid = self._treble = 0.0
        self._beat = 0.0
        self._punch = 0.0
        self._jump = 0.0
        self._ring = 1.0
        self._hue = float(self.rng.uniform(0.0, 1.0))
        self._density = float(self.rng.uniform(0.0, 1.0))
        self._warp = float(self.rng.uniform(0.0, 100.0))

    def update(self, dt: float, audio: AudioData) -> None:
        """Run the drive, and fire a jump on the beat."""
        super().update(dt, audio)
        react = max(0.05, float(self.parameters['reactivity']))
        self._bass = float(np.clip(audio.bass_n * react, 0.0, 1.8))
        self._mid = float(np.clip(audio.mid_n * react, 0.0, 1.8))
        self._treble = float(np.clip(audio.treble_n * react, 0.0, 1.8))
        if audio.beat > 0.5:
            self._beat = 1.0
            self._punch = 1.0
            self._jump = 1.0
            self._ring = 0.0
            self._hue = float((self._hue + self.rng.uniform(0.04, 0.14)) % 1.0)
        self._beat = max(0.0, self._beat - dt * 3.0)
        self._punch = max(0.0, self._punch - dt * 2.4)
        self._jump = max(0.0, self._jump - dt * 2.0)
        self._ring = min(1.0, self._ring + dt * 1.7)

        speed = max(0.05, float(self.parameters['speed']))
        # Wrapped, so the phase never grows into float32's dead zone.
        self._warp = (self._warp
                      + dt * speed * (0.55 + self._bass * 1.7 + self._jump * 3.2)
                      ) % 100.0

    def render(self) -> None:
        """Upload the drive state."""
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
        p['iWarp'].value = float(self._warp)
        p['iJump'].value = float(self._jump)
        p['iRing'].value = float(self._ring)
        p['iDensity'].value = float(self._density)
        p['iJumpForce'].value = max(0.0, float(self.parameters['jump_force']))
        self._vao.render(moderngl.TRIANGLE_STRIP)

    def destroy(self) -> None:
        """Release GL resources."""
        self._vao.release()
        self._vbo.release()
        self._prog.release()
