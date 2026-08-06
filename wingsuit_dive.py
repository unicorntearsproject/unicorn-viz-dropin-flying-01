"""Wingsuit Dive — plummeting past neon mountain ridgelines toward a rising
city glow, wind streaking past on every side.

A vertical-flight sensation rather than forward-flight: silhouetted mountain
ridges with glowing edges slide upward past the camera as it dives, a hazy
city grid glows brighter as the ground approaches, and rushing wind-streak
lines fly past on beat-driven gusts.

Audio reactivity:
  - bass    -> dive speed, ground-glow swell as it nears
  - mid     -> ridge colour drift, sway
  - treble  -> wind-streak density/sparkle
  - beat    -> gust flash + camera kick
"""
from __future__ import annotations

import moderngl

from unicornviz.effects.base import AudioData, BaseEffect

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
uniform float iDive;      // altitude progress [0,1], loops as you "land" and reset high
uniform float iSpeed;
uniform float iHue;
uniform float iSway;
uniform float iRush;      // CPU-integrated bass-dive travel phase

in  vec2 v_uv;
out vec4 fragColor;

#define TAU 6.28318530

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
float ridge(float x, float seed) {
    float v = sin(x * 2.3 + seed * 6.0) * 0.5 + sin(x * 5.1 + seed * 3.0) * 0.25
            + sin(x * 11.0 + seed) * 0.12;
    return v;
}
vec3 palette(float t) { return 0.5 + 0.5 * cos(TAU * (t + vec3(0.0, 0.33, 0.67))); }

void main() {
    float A = iResolution.x / max(iResolution.y, 1.0);
    float t = iTime * (0.6 + iSpeed * 0.8);

    float sway = sin(t * 0.3 + iSway) * (0.06 + iMid * 0.08);
    vec2 p = vec2((v_uv.x - 0.5) * A + sway, v_uv.y - 0.5);

    // Sky darkens overhead, brightens toward the ground glow below.
    vec3 skyHigh = mix(vec3(0.01, 0.01, 0.04), palette(iHue + 0.6), 0.15);
    vec3 groundGlow = mix(vec3(0.15, 0.08, 0.20), palette(iHue), 0.4);
    float alt = clamp(iDive, 0.0, 1.0);
    vec3 col = mix(skyHigh, groundGlow, pow(alt, 1.6));

    // Mountain ridgelines: each layer is a horizon-height function of screen
    // x, and motion comes from phase-shifting the noise input over time —
    // NOT from accumulating raw elapsed time into the height comparison
    // itself (that flattened the whole silhouette to nothing after the
    // first moment, since it compared bounded screen-space against an
    // ever-growing raw accumulator).
    for (int layer = 0; layer < 4; layer++) {
        float fl = float(layer);
        // iRush is CPU-integrated (see update()): the old t * diveSpeed
        // form jumped the silhouette to an unrelated shape on every bass
        // delta once iTime grew large.
        float phase = iRush * (0.6 + fl * 0.45) * 0.6 + alt * (2.0 + fl * 1.2);
        float h = ridge(p.x * (0.8 + fl * 0.3) + fl * 5.0 + phase * 0.25, fl * 3.3)
                * (0.16 - fl * 0.015) - 0.06 - fl * 0.09;
        float edge = p.y - h;
        float below = smoothstep(0.015, -0.015, edge);
        vec3 ridgeCol = mix(vec3(0.03, 0.04, 0.08), palette(iHue + fl * 0.09 + 0.05), 0.35 - fl * 0.05);
        col = mix(col, ridgeCol, below * (0.85 - fl * 0.12));
        float glow = smoothstep(0.03, 0.0, abs(edge)) * (1.0 - fl * 0.2);
        col += palette(iHue + fl * 0.09 + 0.2) * glow * (0.5 + iTreble * 0.5);
    }

    // City-light speckle near the ground, brighter as altitude drops.
    vec2 cg = vec2(p.x * 30.0, (p.y + mod(iRush * 0.35, 400.0) + alt * 3.0) * 30.0);
    float cityN = hash21(floor(cg));
    float cityLight = step(0.965, cityN) * smoothstep(0.35, 0.85, alt);
    col += mix(vec3(1.0, 0.85, 0.5), palette(iHue + 0.15), 0.3) * cityLight * 0.8;

    // Rushing wind motes: short comet-trail streaks scattered in a scrolling
    // grid, denser near the screen edges (peripheral rush), each with its
    // own brief bounded trail rather than a full-height line — reads as
    // scattered wind rush, not highway lane markings.
    float edgeDist = abs(p.x) * 1.4;
    // mod() bounds the scroll offset before it reaches floor()/hash() — an
    // unbounded iTime (this can start anywhere up to 10000 by design) times
    // this term's ~8x multiplier loses enough float32 fractional precision
    // at floor() to collapse many grid cells to the same hash, turning
    // scattered motes into solid bars.
    float windScroll = mod(t * 3.0 + iRush * 2.0, 500.0);
    vec2 windGrid = vec2(p.x * 14.0, p.y * 7.0 + windScroll);
    vec2 windCell = floor(windGrid);
    float windSeed = hash21(windCell);
    if (windSeed > 0.90) {
        vec2 local = fract(windGrid) - vec2(hash21(windCell + 3.1), 0.5);
        float trail = exp(-abs(local.y) * 9.0) * exp(-local.x * local.x * 60.0)
                    * smoothstep(0.0, -0.3, local.y);
        col += vec3(0.9, 0.95, 1.0) * trail * smoothstep(0.0, 0.6, edgeDist)
             * (0.7 + iTreble * 1.0);
    }

    // Beat gust flash.
    col += palette(iHue + 0.4) * iBeat * 0.14 * smoothstep(1.2, 0.0, length(p));

    float vig = smoothstep(1.3, 0.15, length(v_uv - 0.5));
    col *= vig;
    col = col / (1.0 + col * 0.5);
    col = pow(clamp(col, 0.0, 1.0), vec3(0.85));
    fragColor = vec4(col, 1.0);
}
"""


class WingsuitDive(BaseEffect):
    """Vertical wingsuit free-fall past neon ridgelines toward city light."""

    NAME = 'Wingsuit Dive'
    AUTHOR = 'unicorn-viz'
    TAGS = ['flying', 'neon', 'intense', 'energetic', 'hard']
    PING_PONG_FRIENDS = [
        'Canyon Run', 'Warp Drive', 'Cloud Surfer', 'Tron Grid',
        'Reactor Breach', 'Audio Spectrum',
    ]

    def _init(self) -> None:
        self.parameters = {
            'speed': float(self.config.get('speed', 1.0)),
            'reactivity': float(self.config.get('reactivity', 1.3)),
        }
        self._prog = self._make_program(_VERT, _FRAG)
        self._vao, self._vbo = self._fullscreen_quad(self._prog)
        self._hue = float(self.rng.uniform(0.0, 1.0))
        self._sway = float(self.rng.uniform(0.0, 6.28))
        self._dive = float(self.rng.uniform(0.0, 0.4))
        # CPU-integrated travel phase (nebula_drift pattern).
        self._rush = float(self.rng.uniform(0.0, 400.0))
        self._bass = self._mid = self._treble = 0.0
        self._beat = 0.0

    def update(self, dt: float, audio: AudioData) -> None:
        super().update(dt, audio)
        react = max(0.05, float(self.parameters['reactivity']))
        self._bass = float(min(audio.bass_n * react, 1.6))
        self._mid = float(min(audio.mid_n * react, 1.6))
        self._treble = float(min(audio.treble_n * react, 1.6))
        if audio.beat > 0.5:
            self._beat = 1.0
            self._hue = float((self._hue + self.rng.uniform(0.02, 0.08)) % 1.0)
        self._beat = max(0.0, self._beat - dt * 2.8)
        speed = max(0.05, float(self.parameters['speed']))
        self._dive += dt * (0.02 + self._bass * 0.03) * speed
        if self._dive > 1.0:
            self._dive -= 1.0
        self._rush = (self._rush + dt * (0.6 + speed * 0.8) * (0.6 + self._bass * 1.3)) % 1000.0

    def render(self) -> None:
        p = self._prog
        p['iTime'].value = float(self.time)
        p['iResolution'].value = (float(self.width), float(self.height))
        p['iBass'].value = self._bass
        p['iMid'].value = self._mid
        p['iTreble'].value = self._treble
        p['iBeat'].value = self._beat
        p['iDive'].value = self._dive
        p['iSpeed'].value = max(0.05, float(self.parameters['speed']))
        p['iHue'].value = self._hue
        p['iSway'].value = self._sway
        p['iRush'].value = self._rush
        self._vao.render(moderngl.TRIANGLE_STRIP)

    def destroy(self) -> None:
        self._vao.release()
        self._vbo.release()
        self._prog.release()
