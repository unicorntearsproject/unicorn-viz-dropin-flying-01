"""Nebula Drift — a majestic slow glide deep into a vast, colourful gas cloud.

The gentlest flight sensation in the pack: no fast streaks, just a slow
forward push into layered painterly plasma clouds with embedded star
clusters and drifting dust motes, parallax-parting as you sink deeper in.
Beat hits ripple a soft shockwave through the gas.

Audio reactivity:
  - bass    -> forward drift speed, gas swell/brightness
  - mid     -> colour churn rate, layer parallax
  - treble  -> star-cluster twinkle, dust sparkle
  - beat    -> soft shockwave ripple through the cloud
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
uniform float iSpeed;
uniform float iHue;
uniform float iDepth;    // per-instance forward-drift accumulator

in  vec2 v_uv;
out vec4 fragColor;

#define TAU 6.28318530

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    float a = hash21(i), b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0)), d = hash21(i + vec2(1.0, 1.0));
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
float fbm(vec2 p) {
    float v = 0.0, amp = 0.5;
    for (int i = 0; i < 5; i++) { v += noise(p) * amp; p *= 2.02; amp *= 0.55; }
    return v;
}
vec3 palette(float t) { return 0.5 + 0.5 * cos(TAU * (t + vec3(0.0, 0.33, 0.67))); }

void main() {
    float A = iResolution.x / max(iResolution.y, 1.0);
    vec2 p = (v_uv - 0.5) * vec2(A, 1.0);
    float t = iTime * (0.3 + iSpeed * 0.4);
    float drift = iDepth;

    vec3 col = vec3(0.01, 0.005, 0.02);

    // Layered churning gas: several fbm sheets at different parallax depths,
    // each drifting toward the camera (scaling up) as `drift` accumulates so
    // the whole scene reads as sinking deeper into the cloud over time.
    for (int layer = 0; layer < 4; layer++) {
        float fl = float(layer);
        float baseDepth = 1.0 + fl * 0.8;
        float localDepth = mod(baseDepth - drift * (0.5 + fl * 0.3), 3.2) + 0.25;
        vec2 lp = p * localDepth
                + vec2(mod(t * (0.10 + fl * 0.05), 400.0), fl * 4.1 - mod(t * 0.06, 400.0));
        float n = fbm(lp * 1.3 + fl * 2.0);
        float mask = smoothstep(0.35, 0.85, n);
        vec3 tone = mix(vec3(0.05, 0.02, 0.10), palette(iHue + n * 0.5 + fl * 0.08 + iMid * 0.15), 0.7);
        float fade = smoothstep(3.4, 0.3, localDepth) * smoothstep(0.05, 0.4, localDepth);
        col += tone * mask * fade * (0.55 + iBass * 0.5) / localDepth;
    }

    // Embedded star clusters: bright points scattered through the gas,
    // twinkling on treble, parallax-scrolling with depth like the gas.
    for (int layer = 0; layer < 3; layer++) {
        float fl = float(layer);
        float localDepth = mod(1.4 + fl * 1.1 - drift * (0.6 + fl * 0.25), 3.6) + 0.2;
        // mod() bounds the scroll offset before floor()/hash() — see
        // wingsuit_dive.py for why an unbounded iTime here collapses grid
        // cells into solid bars instead of scattered points.
        vec2 sp = p * localDepth * 6.0 + vec2(fl * 11.0, fl * 7.0 - mod(t * 0.4, 400.0));
        vec2 cell = floor(sp);
        float starSeed = hash21(cell + fl * 13.0);
        if (starSeed > 0.90) {
            vec2 local = fract(sp) - 0.5 - (hash21(cell + 5.0) - 0.5) * 0.6;
            float d2 = dot(local, local);
            float star = exp(-d2 * 220.0) / max(localDepth, 0.3);
            float tw = 0.5 + 0.5 * sin(t * (3.0 + starSeed * 6.0) + starSeed * 40.0);
            vec3 starCol = mix(vec3(1.0, 0.95, 0.85), palette(iHue + starSeed * 0.6), 0.4);
            col += starCol * star * (0.5 + tw * 0.5) * (0.6 + iTreble * 1.2);
        }
    }

    // Beat shockwave: a soft ring of brightened gas expanding outward.
    float ring = smoothstep(0.10, 0.0, abs(length(p) - (1.0 - iBeat) * 1.3)) * iBeat;
    col += palette(iHue + 0.35) * ring * 0.6;

    // Overall pulse + gentle vignette.
    col *= 1.0 + iBass * 0.25 + iBeat * 0.2;
    float vig = smoothstep(1.4, 0.2, length(v_uv - 0.5));
    col *= vig;
    col = col / (1.0 + col * 0.6);
    col = pow(clamp(col, 0.0, 1.0), vec3(0.82));
    fragColor = vec4(col, 1.0);
}
"""


class NebulaDrift(BaseEffect):
    """A slow, majestic glide deep into a churning, star-flecked nebula."""

    NAME = 'Nebula Drift'
    AUTHOR = 'unicorn-viz'
    TAGS = ['flying', 'cosmic', 'chill', 'groovy', 'psychedelic']
    PING_PONG_FRIENDS = [
        'Warp Drive', 'Cloud Surfer', 'Portal Flight', 'Cosmos',
        'Wavey Gravy', 'Rainbow Trance',
    ]

    def _init(self) -> None:
        self.parameters = {
            'speed': float(self.config.get('speed', 1.0)),
            'reactivity': float(self.config.get('reactivity', 1.2)),
        }
        self._prog = self._make_program(_VERT, _FRAG)
        self._vao, self._vbo = self._fullscreen_quad(self._prog)
        self._hue = float(self.rng.uniform(0.0, 1.0))
        self._depth = float(self.rng.uniform(0.0, 3.0))
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
            self._hue = float((self._hue + self.rng.uniform(0.02, 0.07)) % 1.0)
        self._beat = max(0.0, self._beat - dt * 1.8)
        speed = max(0.05, float(self.parameters['speed']))
        self._depth += dt * (0.10 + self._bass * 0.18 + self._mid * 0.05) * speed
        self._depth = self._depth % 1000.0

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
        p['iDepth'].value = self._depth
        self._vao.render(moderngl.TRIANGLE_STRIP)

    def destroy(self) -> None:
        self._vao.release()
        self._vbo.release()
        self._prog.release()
