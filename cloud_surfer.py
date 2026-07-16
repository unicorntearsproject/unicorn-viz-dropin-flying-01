"""Cloud Surfer — cruising above a glowing cloud layer at altitude, banking
through gaps toward a blazing sunset core.

A gentler flight sensation than Warp Drive's hyperspace jump: layered
volumetric-looking clouds drift and part beneath the camera, sun glare
blooms through the gaps, and the whole view banks gently side to side like a
glider riding thermals. Beat hits punch a burst of light through the clouds.

Audio reactivity:
  - bass    -> cloud parting / altitude dive punch, sun core swell
  - mid     -> banking sway amplitude, palette drift
  - treble  -> rim-light sparkle on cloud edges
  - beat    -> sunburst flash through the gaps
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
uniform float iBank;      // per-instance bank-sway phase offset

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
    vec2 uv = v_uv;
    float t = iTime * (0.25 + iSpeed * 0.35);

    // Gentle banking: rotate the whole frame slightly, sway with mid energy.
    float bank = sin(t * 0.35 + iBank) * (0.10 + iMid * 0.10);
    float cb = cos(bank), sb = sin(bank);
    vec2 p = vec2(uv.x * A - A * 0.5, uv.y - 0.5);
    p = mat2(cb, -sb, sb, cb) * p;

    // Sky gradient: deep dusk blue up top fading to a warm horizon glow.
    // Keep the sunset warm/pink-to-purple — a small hue tint, not a full
    // random-spectrum override, so it always reads as a sunset.
    vec3 skyTop = mix(vec3(0.03, 0.05, 0.14), palette(iHue + 0.55), 0.12);
    vec3 skyHorizon = mix(vec3(0.85, 0.45, 0.25), palette(iHue), 0.16);
    vec3 col = mix(skyHorizon, skyTop, smoothstep(-0.1, 0.55, p.y));

    // Distant sun with glare, offset slightly toward horizon.
    vec2 sunPos = vec2(0.15, 0.08);
    float sunD = length(p - sunPos);
    float sun = exp(-sunD * sunD * 22.0);
    vec3 sunCol = mix(vec3(1.0, 0.85, 0.55), palette(iHue + 0.1), 0.14);
    col += sunCol * sun * (0.7 + iBass * 0.5 + iBeat * 0.7);
    col += sunCol * exp(-sunD * 5.0) * (0.10 + iBeat * 0.18);

    // Layered cloud banks: fbm noise sheets scrolling past at different
    // depths/speeds for parallax, composited back-to-front so distant layers
    // show through the gaps of nearer ones instead of just flattening into a
    // single wash. Each layer keeps clear open-sky gaps (high threshold).
    float coverage = 0.0;
    float rim = 0.0;
    for (int layer = 0; layer < 3; layer++) {
        float fl = float(layer);
        float parallax = 1.0 - fl * 0.28;
        vec2 cp = p * (0.75 + fl * 0.45) + vec2(t * (0.55 + fl * 0.4) * parallax, fl * 3.7);
        cp.y += iBass * 0.22 * (fl + 1.0);
        float n = fbm(cp * 1.5);
        float thresh = 0.58 + fl * 0.05;
        float mask = smoothstep(thresh, thresh + 0.22, n);
        float shade = smoothstep(thresh, thresh + 0.45, n);         // brighter cloud tops
        vec3 layerCol = mix(vec3(0.42, 0.46, 0.58), vec3(1.0, 0.97, 0.92), shade);
        layerCol = mix(layerCol, skyHorizon, 0.12 + fl * 0.10);
        float alpha = mask * (0.85 - fl * 0.18);
        col = mix(col, layerCol, alpha);
        rim += smoothstep(thresh - 0.02, thresh + 0.02, n)
             * (1.0 - smoothstep(thresh + 0.02, thresh + 0.07, n)) * (1.0 - fl * 0.3);
        coverage = max(coverage, mask);
    }
    col += mix(vec3(1.0, 0.9, 0.65), palette(iHue + 0.2), 0.2) * rim * (0.25 + iTreble * 0.6);

    // Sunburst punch through the gaps on strong beats.
    float gapLight = (1.0 - coverage) * exp(-sunD * 2.0);
    col += sunCol * gapLight * iBeat * 1.4;

    // Rushing wind streak lines, subtle, speed-linked.
    float streak = sin(p.x * 40.0 + t * 6.0) * sin(p.y * 3.0);
    col += vec3(1.0) * smoothstep(0.996, 1.0, streak) * (0.15 + iBass * 0.3);

    col = col / (1.0 + col * 0.5);
    col = pow(clamp(col, 0.0, 1.0), vec3(0.88));
    fragColor = vec4(col, 1.0);
}
"""


class CloudSurfer(BaseEffect):
    """Gliding above a luminous cloud layer toward a blazing horizon."""

    NAME = 'Cloud Surfer'
    AUTHOR = 'unicorn-viz'
    TAGS = ['flying', 'sky', 'chill', 'groovy', 'neon']
    PING_PONG_FRIENDS = [
        'Warp Drive', 'Wingsuit Dive', 'Nebula Drift', 'Rainbow Trance',
        'Wavey Gravy', 'Audio Waveforms',
    ]

    def _init(self) -> None:
        self.parameters = {
            'speed': float(self.config.get('speed', 1.0)),
            'reactivity': float(self.config.get('reactivity', 1.2)),
        }
        self._prog = self._make_program(_VERT, _FRAG)
        self._vao, self._vbo = self._fullscreen_quad(self._prog)
        self._hue = float(self.rng.uniform(0.0, 1.0))
        self._bank = float(self.rng.uniform(0.0, 6.28))
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
            self._hue = float((self._hue + self.rng.uniform(0.03, 0.1)) % 1.0)
        self._beat = max(0.0, self._beat - dt * 2.2)

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
        p['iBank'].value = self._bank
        self._vao.render(moderngl.TRIANGLE_STRIP)

    def destroy(self) -> None:
        self._vao.release()
        self._vbo.release()
        self._prog.release()
