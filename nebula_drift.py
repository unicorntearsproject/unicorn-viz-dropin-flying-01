"""Nebula Drift — drifting through gas that is actually lit from inside.

The 1.x version was a single low-contrast cloud: measured on a loud frame it
came out at a saturation of 19, which is why it read as grey mush rather than
as a nebula.  The rebuild is about **structure and light**.

Gas is built from domain-warped fbm sampled twice, at two different scales and
two different hues, so the cloud has regions rather than one tint; ridged noise
on top of that gives the bright filaments a nebula actually has.  Dark lanes
are carved out by a third sample, because the dust between the gas is most of
what makes the shape readable.  All of it drifts on a wrapped phase, so the
field flows forever without the numbers growing.

Stars are the light source, not a decoration.  A handful of them sit **inside**
the gas at their own depths, and each one lights the cloud around it -- so the
gas is bright near a star and dim between, which is the difference between a
nebula and a fog.  They carry diffraction spikes and parallax past the field.

**The bang.** A beat ignites one: the star flares, and a shell of light expands
out of it through the gas, brightening whatever it passes.

Audio reactivity:
  - bass    -> drift rate, how hard the stars light the gas
  - mid     -> hue separation between the two gas layers, warp depth
  - treble  -> star spikes, dust glitter
  - beat    -> a star ignites and blows a shell through the cloud

Randomised per activation: the two gas hues, warp seed, star layout, drift
direction, and ``self.time``.

Config keys (under ``[effects.NebulaDrift]``):
    speed, zoom, reactivity, glitter, gas_density
"""
from __future__ import annotations

import moderngl
import numpy as np

from unicornviz.effects.base import AudioData, BaseEffect

__version__ = '0.4.0'

_NSTAR = 7

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
uniform float iHueA;        // the two gas hues
uniform float iHueB;
uniform float iZoom;
uniform float iGlitter;
uniform float iDrift;       // wrapped drift phase
uniform float iSeed;
uniform float iDensity;
uniform float iShellR;      // radius of the ignition shell, <0 when spent
uniform vec2  iShellP;      // where it is expanding from
uniform vec4  iStar[7];     // xy position, z brightness, w flare

in  vec2 v_uv;
out vec4 fragColor;

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
float fbm(vec2 x) {
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 5; i++) { v += a * vnoise(x); x *= 2.03; a *= 0.5; }
    return v;
}
// Ridged: the filament form. 1 - |2n-1| turns blobs into bright veins.
float ridged(vec2 x) {
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 4; i++) {
        v += a * (1.0 - abs(2.0 * vnoise(x) - 1.0));
        x *= 2.11; a *= 0.5;
    }
    return v;
}
vec3 palette(float t) { return 0.5 + 0.5 * cos(TAU * (t + vec3(0.0, 0.33, 0.67))); }

void main() {
    float A = iResolution.x / max(iResolution.y, 1.0);
    float tb = mod(iTime, 600.0);
    vec2 p = (v_uv - 0.5) * vec2(A, 1.0) / max(iZoom * (1.0 - iPunch * 0.04), 0.05);

    vec3 gasA = palette(iHueA);
    vec3 gasB = palette(iHueB);
    vec3 hot = mix(vec3(1.0, 0.92, 0.80), gasA, 0.25);

    // ── the gas ────────────────────────────────────────────────────────────
    // Warp the sample point with its own noise: straight fbm gives billows,
    // and a nebula is made of drawn-out sheets, not billows.
    vec2 dr = vec2(iDrift * 0.35, -iDrift * 0.22) + iSeed;
    vec2 w = vec2(fbm(p * 1.30 + dr), fbm(p * 1.30 + dr + 4.7));
    float warp = 0.55 + iMid * 0.55;
    vec2 q = p * 1.7 + (w - 0.5) * warp + dr * 0.5;

    float dens = max(0.25, iDensity);
    float a1 = fbm(q) * dens;
    float a2 = fbm(q * 2.15 + 9.3) * dens;
    float fil = ridged(q * 1.55 + 2.1);
    // The dust lane: a separate field that eats the gas where it is thick.
    float dust = smoothstep(0.42, 0.70, fbm(q * 0.72 - 3.4));

    vec3 col = vec3(0.020, 0.018, 0.040);
    // Two gas populations at two hues; without the split the whole field
    // collapses to one tint and reads as fog.
    // Exposure. Gas this bright before the tonemap clips to white and the
    // hue separation goes with it -- the whole point of two populations.
    col += gasA * pow(clamp(a1, 0.0, 1.0), 2.1) * 0.78;
    col += gasB * pow(clamp(a2, 0.0, 1.0), 2.4) * 0.55;
    col += mix(gasA, gasB, 0.5) * pow(clamp(fil - 0.52, 0.0, 1.0), 2.2) * 1.30
         * (0.45 + iMid * 0.6);
    col *= 1.0 - dust * 0.88;

    // ── the stars that light it ────────────────────────────────────────────
    for (int i = 0; i < 7; i++) {
        vec4 S = iStar[i];
        vec2 d = p - S.xy;
        float r = length(d);
        // The gas glows where a star is inside it. This coupling is the whole
        // difference between a nebula and a fog.
        float lightIt = exp(-r * 2.6) * S.z;
        col += mix(gasA, hot, 0.55) * lightIt * 0.55
             * (0.20 + a1 * 1.1) * (0.6 + iBass * 0.7);
        if (r < 0.30) {
            // Core, halo and the four-point spike a bright star draws.
            col += hot * exp(-r * r * 3400.0) * (1.0 + S.w * 2.4);
            col += hot * exp(-r * 34.0) * 0.30 * S.z;
            float spike = max(exp(-abs(d.x) * 190.0), exp(-abs(d.y) * 190.0))
                        * exp(-r * 11.0);
            col += hot * spike * (0.32 + S.w * 1.2) * (0.4 + iTreble * 0.8);
        }
    }

    // ── the bang ───────────────────────────────────────────────────────────
    // A shell of light expanding out of the star that just went, brightening
    // the gas it crosses rather than being drawn on top of it.
    if (iShellR >= 0.0) {
        float sr = length(p - iShellP);
        float shell = smoothstep(0.10, 0.0, abs(sr - iShellR));
        col += mix(hot, gasB, 0.35) * shell * (1.0 - iShellR * 0.55) * 0.55
             * (0.3 + a1 * 1.5) * (0.7 + iBass * 0.7);
    }

    // Foreground dust catching the light, and far stars behind the gas.
    vec2 cell = p * 52.0 + vec2(iDrift * 0.15, 0.0);
    vec2 g = floor(cell);
    float h = hash21(g);
    if (h > 0.976) {
        vec2 dd = fract(cell) - vec2(hash21(g + 3.1), hash21(g + 7.7));
        col += mix(vec3(0.85, 0.9, 1.0), gasB, 0.30)
             * exp(-dot(dd, dd) * 30.0)
             * (0.35 + iTreble * 0.8) * iGlitter
             * (0.35 + 0.65 * (1.0 - dust));
    }

    col += mix(gasA, gasB, 0.5) * iBeat * 0.07;
    col *= 0.92 + 0.16 * sin(tb * 0.05 + p.x);
    col *= smoothstep(1.60, 0.26, length(v_uv - 0.5));
    col = col / (1.0 + col * 0.42);
    col = pow(clamp(col, 0.0, 1.0), vec3(0.88));
    fragColor = vec4(col, 1.0);
}
"""


class NebulaDrift(BaseEffect):
    """Drift through lit gas: two hues, filaments, dust lanes, igniting stars."""

    NAME = 'Nebula Drift'
    AUTHOR = 'unicorn-viz'
    TAGS = ['flying', 'cosmic', 'neon', 'groovy', 'chill']
    PING_PONG_FRIENDS = [
        'Warp Drive', 'Asteroid Run', 'Portal Flight', 'Starfield',
        'Audio Spectrum',
    ]

    _MIN_ZOOM = 0.55        # no hard geometry to turn inside out

    def _init(self) -> None:
        self.parameters = {
            'speed': float(self.config.get('speed', 1.0)),
            'zoom': float(self.config.get('zoom', 1.0)),
            'reactivity': float(self.config.get('reactivity', 1.2)),
            'glitter': float(self.config.get('glitter', 1.0)),
            'gas_density': float(self.config.get('gas_density', 1.0)),
        }
        self._prog = self._make_program(_VERT, _FRAG)
        self._vao, self._vbo = self._fullscreen_quad(self._prog)

        self._bass = self._mid = self._treble = 0.0
        self._beat = 0.0
        self._punch = 0.0
        # Two hues a useful distance apart, so the gas separates into regions.
        self._hue_a = float(self.rng.uniform(0.0, 1.0))
        self._hue_b = float((self._hue_a + self.rng.uniform(0.18, 0.42)) % 1.0)
        self._seed = float(self.rng.uniform(0.0, 40.0))
        self._drift = float(self.rng.uniform(0.0, 100.0))
        self._dir = 1.0 if self.rng.random() < 0.5 else -1.0

        # x, y, brightness, flare, drift vx, vy.
        self._stars = np.zeros((_NSTAR, 6), dtype=np.float64)
        for i in range(_NSTAR):
            self._stars[i] = (
                float(self.rng.uniform(-0.85, 0.85)),
                float(self.rng.uniform(-0.5, 0.5)),
                float(self.rng.uniform(0.30, 1.0)),
                0.0,
                self._dir * float(self.rng.uniform(0.004, 0.030)),
                float(self.rng.uniform(-0.012, 0.012)),
            )
        self._tbl_star = np.zeros((_NSTAR, 4), dtype='f4')
        self._shell_r = -1.0
        self._shell_p = (0.0, 0.0)

    def update(self, dt: float, audio: AudioData) -> None:
        """Drift the field, and ignite a star when the beat lands."""
        super().update(dt, audio)
        react = max(0.05, float(self.parameters['reactivity']))
        self._bass = float(np.clip(audio.bass_n * react, 0.0, 1.8))
        self._mid = float(np.clip(audio.mid_n * react, 0.0, 1.8))
        self._treble = float(np.clip(audio.treble_n * react, 0.0, 1.8))
        if audio.beat > 0.5:
            self._beat = 1.0
            self._punch = 1.0
            self._ignite()
        self._beat = max(0.0, self._beat - dt * 3.0)
        self._punch = max(0.0, self._punch - dt * 2.2)

        speed = max(0.05, float(self.parameters['speed']))
        self._drift = (self._drift
                       + dt * speed * (0.30 + self._bass * 0.95)) % 100.0

        # Stars parallax across the gas and wrap around.
        self._stars[:, 0] += self._stars[:, 4] * dt * speed * (1.0 + self._bass)
        self._stars[:, 1] += self._stars[:, 5] * dt * speed
        self._stars[self._stars[:, 0] > 1.05, 0] = -1.05
        self._stars[self._stars[:, 0] < -1.05, 0] = 1.05
        self._stars[:, 3] = np.maximum(0.0, self._stars[:, 3] - dt * 2.2)

        if self._shell_r >= 0.0:
            self._shell_r += dt * (1.1 + self._bass * 0.8)
            if self._shell_r > 2.2:
                self._shell_r = -1.0

    def _ignite(self) -> None:
        """Flare one star and blow a shell of light out of it."""
        i = int(self.rng.integers(_NSTAR))
        self._stars[i, 3] = 1.0
        self._shell_p = (float(self._stars[i, 0]), float(self._stars[i, 1]))
        self._shell_r = 0.0

    def render(self) -> None:
        """Upload the gas phase and the stars lighting it."""
        self._tbl_star[:, 0] = self._stars[:, 0]
        self._tbl_star[:, 1] = self._stars[:, 1]
        self._tbl_star[:, 2] = self._stars[:, 2]
        self._tbl_star[:, 3] = np.clip(self._stars[:, 3], 0.0, 1.0)

        p = self._prog
        p['iTime'].value = float(self.time)
        p['iResolution'].value = (float(self.width), float(self.height))
        p['iBass'].value = self._bass
        p['iMid'].value = self._mid
        p['iTreble'].value = self._treble
        p['iBeat'].value = self._beat
        p['iPunch'].value = self._punch
        p['iHueA'].value = self._hue_a
        p['iHueB'].value = self._hue_b
        p['iZoom'].value = max(self._MIN_ZOOM, float(self.parameters['zoom']))
        p['iGlitter'].value = max(0.0, float(self.parameters['glitter']))
        p['iDrift'].value = float(self._drift)
        p['iSeed'].value = float(self._seed)
        p['iDensity'].value = max(0.25, float(self.parameters['gas_density']))
        p['iShellR'].value = float(self._shell_r)
        p['iShellP'].value = self._shell_p
        p['iStar'].write(self._tbl_star.tobytes())
        self._vao.render(moderngl.TRIANGLE_STRIP)

    def destroy(self) -> None:
        """Release GL resources."""
        self._vao.release()
        self._vbo.release()
        self._prog.release()
