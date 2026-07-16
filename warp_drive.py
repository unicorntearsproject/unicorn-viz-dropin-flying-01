"""Warp Drive — hyperspace jump: streaking stars rushing past into infinity.

Pure first-person flight-through-space sensation: thousands of star streaks
radiate from a vanishing point at screen centre, lengthening into hard
light-speed lines as bass rises, with a glowing engine core and full FTL
"jump" bursts (extreme radial blur + chromatic flash) on hard beats.

Audio reactivity:
  - bass    -> streak length / warp speed, core swell
  - mid     -> streak density flicker, palette drift
  - treble  -> sparkle glints on near streaks
  - beat    -> FTL jump burst (radial blur + flash + palette shift)
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
uniform float iJump;      // FTL jump envelope [0,1], beat-triggered
uniform float iSpeed;
uniform float iHue;
uniform float iDensity;   // per-instance star-field density variant

in  vec2 v_uv;
out vec4 fragColor;

#define TAU 6.28318530

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
vec3 palette(float t) { return 0.5 + 0.5 * cos(TAU * (t + vec3(0.0, 0.33, 0.67))); }

void main() {
    float A = iResolution.x / max(iResolution.y, 1.0);
    vec2 p = (v_uv - 0.5) * vec2(A, 1.0);
    float t = iTime * (0.5 + iSpeed * 0.8);

    float speed = 0.35 + iBass * 1.4 + iJump * 3.5;
    vec3 col = vec3(0.0);

    // Layered star streaks: polar-warped rings of stars, each ring's radial
    // motion driven by speed so nearer rings blur into longer streaks.
    for (int layer = 0; layer < 3; layer++) {
        float fl = float(layer);
        float cellCount = 26.0 + fl * 14.0 + iDensity * 10.0;
        float ang = atan(p.y, p.x);
        float rad = length(p) + 1e-4;
        float cell = floor(ang / TAU * cellCount);
        float starSeed = hash21(vec2(cell, fl * 7.0));
        float depth = fract(starSeed * 9.7 - t * (speed * (0.5 + fl * 0.4)));
        float r0 = 0.02 + depth * 1.4;
        vec2 starDir = vec2(cos((cell + 0.5) / cellCount * TAU),
                            sin((cell + 0.5) / cellCount * TAU));
        vec2 toStar = p - starDir * r0;
        float along = dot(toStar, starDir);
        float perp = length(toStar - starDir * along);
        float streakLen = 0.02 + speed * (0.10 + depth * 0.10);
        float core = smoothstep(0.0028, 0.0, perp) * smoothstep(streakLen, 0.0, abs(along));
        float fade = smoothstep(1.5, 0.05, r0) * (0.4 + 0.6 * (1.0 - depth));
        vec3 tone = mix(vec3(0.85, 0.92, 1.0),
                       palette(iHue + starSeed * 0.4 + fl * 0.1 + iMid * 0.12), 0.45 + iMid * 0.15);
        col += tone * core * fade * (0.7 + iTreble * 0.7 * step(0.8, starSeed));
    }

    // Radial motion-blur haze that grows hard during an FTL jump.
    float rad2 = length(p);
    float blur = smoothstep(1.1, 0.0, rad2) * (0.10 + iJump * 0.55);
    col += palette(iHue + 0.5 + t * 0.02) * blur;

    // Engine core glow at the vanishing point, plus a sharp instant flash on
    // the beat itself (separate from iJump's slower-decaying warp blur).
    float core = exp(-rad2 * rad2 * 26.0);
    col += mix(vec3(0.7, 0.85, 1.0), palette(iHue + 0.15), 0.5)
         * core * (0.8 + iBass * 1.2 + iJump * 1.5 + iBeat * 1.8);

    // Jump chromatic streak flash.
    col.r += iJump * 0.20 * smoothstep(1.2, 0.0, rad2);
    col.b += iJump * 0.12 * smoothstep(1.2, 0.0, rad2) * -1.0;
    col = max(col, vec3(0.0));

    float vig = smoothstep(1.5, 0.35, length(v_uv - 0.5));
    col *= vig;
    col = col / (1.0 + col * 0.5);
    col = pow(clamp(col, 0.0, 1.0), vec3(0.82));
    fragColor = vec4(col, 1.0);
}
"""


class WarpDrive(BaseEffect):
    """First-person hyperspace jump: streaking stars into a vanishing point."""

    NAME = 'Warp Drive'
    AUTHOR = 'unicorn-viz'
    TAGS = ['flying', 'cosmic', 'neon', 'intense', 'energetic']
    PING_PONG_FRIENDS = [
        'Laser Tunnel', 'Wormhole', 'Nebula Drift', 'Portal Flight',
        'Reactor Breach', 'Starfield', 'Audio Spectrum',
    ]

    def _init(self) -> None:
        self.parameters = {
            'speed': float(self.config.get('speed', 1.0)),
            'reactivity': float(self.config.get('reactivity', 1.2)),
        }
        self._prog = self._make_program(_VERT, _FRAG)
        self._vao, self._vbo = self._fullscreen_quad(self._prog)
        self._hue = float(self.rng.uniform(0.0, 1.0))
        self._density = float(self.rng.uniform(0.0, 1.0))
        self._bass = self._mid = self._treble = 0.0
        self._beat = 0.0
        self._jump = 0.0

    def update(self, dt: float, audio: AudioData) -> None:
        super().update(dt, audio)
        react = max(0.05, float(self.parameters['reactivity']))
        self._bass = float(min(audio.bass_n * react, 1.6))
        self._mid = float(min(audio.mid_n * react, 1.6))
        self._treble = float(min(audio.treble_n * react, 1.6))
        if audio.beat > 0.5:
            self._beat = 1.0
            self._jump = 1.0
            self._hue = float((self._hue + self.rng.uniform(0.05, 0.18)) % 1.0)
        self._beat = max(0.0, self._beat - dt * 2.5)
        self._jump = max(0.0, self._jump - dt * 1.3)

    def render(self) -> None:
        p = self._prog
        p['iTime'].value = float(self.time)
        p['iResolution'].value = (float(self.width), float(self.height))
        p['iBass'].value = self._bass
        p['iMid'].value = self._mid
        p['iTreble'].value = self._treble
        p['iBeat'].value = self._beat
        p['iJump'].value = self._jump
        p['iSpeed'].value = max(0.05, float(self.parameters['speed']))
        p['iHue'].value = self._hue
        p['iDensity'].value = self._density
        self._vao.render(moderngl.TRIANGLE_STRIP)

    def destroy(self) -> None:
        self._vao.release()
        self._vbo.release()
        self._prog.release()
