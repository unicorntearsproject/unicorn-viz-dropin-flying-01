"""Asteroid Run — first-person dash through a dense, tumbling asteroid field.

Chunky glowing-edged rocks fly past at varying depths (parallax layers),
each one a rough polygon silhouette with a hot rim light, tumbling as it
passes. Near-misses punch a spark flash and a camera-shake kick on the beat
for real danger/tension.

Audio reactivity:
  - bass    -> flight speed, near-miss shake intensity
  - mid     -> tumble rate, rock density flicker
  - treble  -> rim-light sparkle, debris glints
  - beat    -> near-miss spark flash + camera kick
"""
from __future__ import annotations

__version__ = '0.1.0'

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
uniform float iSpeed;
uniform float iHue;
uniform float iShake;    // beat-driven camera-kick envelope

in  vec2 v_uv;
out vec4 fragColor;

#define TAU 6.28318530

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
vec3 palette(float t) { return 0.5 + 0.5 * cos(TAU * (t + vec3(0.0, 0.33, 0.67))); }

// Rough rock silhouette: a wobbled polygon distance field.
float rockSdf(vec2 p, float seed, float rot) {
    float c = cos(rot), s = sin(rot);
    p = mat2(c, -s, s, c) * p;
    float a = atan(p.y, p.x);
    float r = length(p);
    float wob = 0.75 + 0.14 * sin(a * 5.0 + seed * 9.0) + 0.09 * sin(a * 3.0 - seed * 5.0)
              + 0.06 * sin(a * 8.0 + seed * 13.0);
    return r - wob;
}

void main() {
    float A = iResolution.x / max(iResolution.y, 1.0);
    float t = iTime * (0.5 + iSpeed * 0.9);
    float speed = 0.6 + iBass * 1.3;

    // mod() bounds every time-derived value below before it reaches
    // floor()/hash() — an unbounded iTime (starts anywhere up to 10000 by
    // design) loses enough float32 fractional precision at floor() to
    // collapse many cells to the same hash (see wingsuit_dive.py).
    float tShake = mod(t * 30.0, 2000.0);
    vec2 shakeOff = vec2(hash21(vec2(floor(tShake), 1.0)),
                         hash21(vec2(floor(tShake), 2.0))) - 0.5;
    vec2 p = (v_uv - 0.5) * vec2(A, 1.0) + shakeOff * iShake * 0.06;

    vec3 col = vec3(0.008, 0.006, 0.014);

    // Distant sparkle dust: soft round twinkle points, not flat cell squares.
    vec2 dg = p * 40.0;
    vec2 dCell = floor(dg);
    float dustN = hash21(dCell);
    if (dustN > 0.988) {
        vec2 dLocal = fract(dg) - 0.5 - (hash21(dCell + 8.3) - 0.5) * 0.5;
        float dust = exp(-dot(dLocal, dLocal) * 26.0);
        col += vec3(0.6, 0.7, 0.9) * dust * (0.5 + iTreble * 0.6);
    }

    // Layered tumbling asteroids: each layer scrolls forward (scaling up)
    // with a bounded, wrapped depth so rocks recur endlessly without ever
    // comparing against an unbounded raw time accumulator.
    for (int layer = 0; layer < 5; layer++) {
        float fl = float(layer);
        float cellSize = 1.1;
        float tWrap = mod(t * speed * 0.5, cellSize * 600.0);
        float rawZ = fl * (cellSize / 5.0) - mod(tWrap, cellSize);
        float z = mod(rawZ, cellSize) + 0.04;
        float scale = 1.0 / z;
        float seedBase = fl + floor((tWrap + fl * (cellSize / 5.0)) / cellSize) * 5.0;

        for (int slot = 0; slot < 2; slot++) {
            float fs = float(slot);
            float rockSeed = hash21(vec2(seedBase, fs * 7.7));
            // Perspective divide: a rock at fixed world-space lateral offset
            // and depth z projects to screen position offset/z: comparing
            // p directly against that (rearranged to avoid dividing by a
            // near-zero z) as (p*z - offset)/R puts every rock test in the
            // same normalised unit-sphere space rockSdf expects, regardless
            // of depth. (Multiplying the offset by scale a second time, as
            // an earlier version of this did, compounded into a scale^2
            // blowup that pushed every rock off-screen.)
            vec2 offset = (vec2(hash21(vec2(seedBase, fs + 1.0)),
                               hash21(vec2(seedBase, fs + 2.0))) - 0.5) * 1.6;
            float R = 0.35;
            vec2 rp = (p * z - offset) / R;
            float rot = t * (0.3 + rockSeed * 0.8 + iMid * 0.5) + rockSeed * 20.0;
            float d = rockSdf(rp, rockSeed, rot);
            float body = smoothstep(0.02, -0.02, d);
            float rim = smoothstep(0.10, 0.0, abs(d)) * (1.0 - body * 0.5);

            vec3 rockCol = mix(vec3(0.16, 0.13, 0.15), vec3(0.38, 0.28, 0.24), rockSeed);
            vec3 hotRim = mix(vec3(1.0, 0.5, 0.2), palette(iHue + rockSeed * 0.3), 0.5);
            col = mix(col, rockCol * (0.5 + 0.1 / z), body * smoothstep(4.0, 0.3, scale));
            col += hotRim * rim * (0.5 + iTreble * 0.9) * smoothstep(4.0, 0.3, scale);
        }
    }

    // Near-miss spark burst on the beat.
    float tBurst = mod(t * 2.0, 2000.0);
    float burstSeed = hash21(vec2(floor(tBurst), 5.0));
    vec2 burstPos = (vec2(hash21(vec2(floor(tBurst), 6.0)),
                         hash21(vec2(floor(tBurst), 7.0))) - 0.5) * vec2(A, 1.0) * 0.8;
    float burstD = length(p - burstPos);
    vec3 sparkCol = mix(vec3(1.0, 0.7, 0.3), palette(iHue + 0.5), 0.4);
    col += sparkCol * exp(-burstD * burstD * 30.0) * iBeat * 1.6;
    for (int k = 0; k < 8; k++) {
        float fk = float(k);
        float ang = fk / 8.0 * TAU + burstSeed * 6.0;
        vec2 dir = vec2(cos(ang), sin(ang));
        float along = dot(p - burstPos, dir);
        float perp = length((p - burstPos) - dir * along);
        float ray = exp(-perp * perp * 400.0) * smoothstep(0.0, 0.25, along) * smoothstep(0.45, 0.05, along);
        col += sparkCol * ray * iBeat * 0.8;
    }

    // Beat chromatic kick + vignette.
    col.r += iShake * 0.06;
    col.b -= iShake * 0.03;
    col = max(col, vec3(0.0));

    float vig = smoothstep(1.3, 0.15, length(v_uv - 0.5));
    col *= vig;
    col = col / (1.0 + col * 0.5);
    col = pow(clamp(col, 0.0, 1.0), vec3(0.85));
    fragColor = vec4(col, 1.0);
}
"""


class AsteroidRun(BaseEffect):
    """First-person dash dodging through a dense tumbling asteroid field."""

    NAME = 'Asteroid Run'
    AUTHOR = 'unicorn-viz'
    TAGS = ['flying', 'cosmic', 'intense', 'hard', 'energetic']
    PING_PONG_FRIENDS = [
        'Warp Drive', 'Portal Flight', 'Nebula Drift', 'Reactor Breach',
        'Galaga', 'Audio Spectrum',
    ]

    def _init(self) -> None:
        self.parameters = {
            'speed': float(self.config.get('speed', 1.0)),
            'reactivity': float(self.config.get('reactivity', 1.3)),
        }
        self._prog = self._make_program(_VERT, _FRAG)
        self._vao, self._vbo = self._fullscreen_quad(self._prog)
        self._hue = float(self.rng.uniform(0.0, 1.0))
        self._bass = self._mid = self._treble = 0.0
        self._beat = 0.0
        self._shake = 0.0

    def update(self, dt: float, audio: AudioData) -> None:
        super().update(dt, audio)
        react = max(0.05, float(self.parameters['reactivity']))
        self._bass = float(min(audio.bass_n * react, 1.6))
        self._mid = float(min(audio.mid_n * react, 1.6))
        self._treble = float(min(audio.treble_n * react, 1.6))
        if audio.beat > 0.5:
            self._beat = 1.0
            self._shake = 1.0
            self._hue = float((self._hue + self.rng.uniform(0.03, 0.1)) % 1.0)
        self._beat = max(0.0, self._beat - dt * 3.2)
        self._shake = max(0.0, self._shake - dt * 4.5)

    def render(self) -> None:
        p = self._prog
        p['iTime'].value = float(self.time)
        p['iResolution'].value = (float(self.width), float(self.height))
        p['iBass'].value = self._bass
        p['iMid'].value = self._mid
        p['iTreble'].value = self._treble
        p['iBeat'].value = self._beat
        p['iSpeed'].value = max(0.05, float(self.parameters['speed']))
        p['iHue'].value = self._hue
        p['iShake'].value = self._shake
        self._vao.render(moderngl.TRIANGLE_STRIP)

    def destroy(self) -> None:
        self._vao.release()
        self._vbo.release()
        self._prog.release()
