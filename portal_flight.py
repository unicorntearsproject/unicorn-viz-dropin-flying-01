"""Portal Flight — flying through a rapid sequence of glowing ring-gates,
each one flashing and colour-cycling as you pass through.

Rings recede toward a vanishing point and rush past the camera in sequence;
each ring gets its own hue and a "pass-through" flash timed to when it
crosses the camera plane, escalating with beat energy for a genuine
teleport-between-gates sensation.

Audio reactivity:
  - bass    -> flight speed, ring pulse thickness
  - mid     -> ring wobble / palette drift
  - treble  -> chromatic trail sparkle
  - beat    -> hard speed surge + pass-through flash
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
uniform float iSurge;    // beat-driven speed-surge envelope
uniform float iFlight;   // CPU-integrated flight distance (audio-speed safe)

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
    float t = iTime * (0.6 + iSpeed * 0.9);

    vec3 col = vec3(0.01, 0.008, 0.02);

    // Background streaks (mild, sets the "flying" backdrop between gates).
    float ang = atan(p.y, p.x);
    float rad = length(p) + 1e-4;
    float cell = floor(ang / TAU * 30.0);
    float streakSeed = hash21(vec2(cell, 1.0));
    float depth = fract(streakSeed * 7.3 - t * 0.15);
    vec2 dir = vec2(cos((cell + 0.5) / 30.0 * TAU), sin((cell + 0.5) / 30.0 * TAU));
    vec2 toStar = p - dir * (0.05 + depth * 1.3);
    float along = dot(toStar, dir);
    float perp = length(toStar - dir * along);
    float streak = smoothstep(0.0022, 0.0, perp) * smoothstep(0.05, 0.0, abs(along));
    col += palette(iHue + streakSeed * 0.3) * streak * 0.35;

    // Ring gates: an endless sequence marching toward the camera, spaced
    // evenly in a bounded, wrapped depth so gate index/hue never depends on
    // an ever-growing raw time value.
    float spacing = 0.62;
    // iFlight is CPU-integrated (see update()): the old t * (k + iBass +
    // iSurge) form re-scrubbed the whole gate sequence on every audio/surge
    // delta, so gates teleported instead of accelerating.
    float wrapped = mod(iFlight, spacing * 400.0);
    for (int i = 0; i < 9; i++) {
        float fi = float(i);
        float z = mod(fi * spacing - wrapped, spacing * 9.0) + 0.03;
        float gateIdx = floor((wrapped + fi * spacing) / (spacing * 9.0)) * 9.0 + fi;
        float gateHue = iHue + gateIdx * 0.09;
        float scale = 1.0 / z;
        if (scale > 40.0) continue;

        float wobble = sin(t * 1.3 + gateIdx * 2.7) * (0.05 + iMid * 0.08);
        vec2 rp = p * scale;
        float ringR = 0.62 + wobble;
        float ringD = abs(length(rp) - ringR);
        float thickness = 0.05 + iBass * 0.05;
        float ring = smoothstep(thickness, 0.0, ringD) / max(scale * 0.06, 1.0);
        vec3 ringCol = palette(gateHue);

        // Pass-through flash: brightest right as the gate crosses z≈0.08.
        float passFlash = smoothstep(0.22, 0.03, z) * smoothstep(0.0, 0.08, z);
        // Small floor so the distant, small-scale rings stay visible as a
        // receding tunnel instead of thinning below visibility.
        float distantBoost = smoothstep(1.0, 0.15, scale) * 0.6;
        col += ringCol * ring * (0.8 + distantBoost + iTreble * 0.6 + passFlash * 3.0 + iBeat * 1.0);
        col += ringCol * passFlash * smoothstep(1.0, 0.0, ringD * 3.0) * 0.5;
    }

    // Central engine glow + beat pulse — kept tight so it doesn't swallow
    // the distant receding rings (a wide glow here washed out the whole
    // ring-tunnel perspective down to just the nearest one or two gates).
    float core = exp(-rad * rad * 90.0);
    col += mix(vec3(0.8, 0.9, 1.0), palette(iHue + 0.5), 0.4) * core * (0.5 + iBeat * 1.3 + iSurge * 1.0);

    // Surge chromatic streak.
    col.r += iSurge * 0.15 * smoothstep(1.2, 0.0, rad);
    col.b -= iSurge * 0.08 * smoothstep(1.2, 0.0, rad);
    col = max(col, vec3(0.0));

    float vig = smoothstep(1.45, 0.3, length(v_uv - 0.5));
    col *= vig;
    col = col / (1.0 + col * 0.5);
    col = pow(clamp(col, 0.0, 1.0), vec3(0.82));
    fragColor = vec4(col, 1.0);
}
"""


class PortalFlight(BaseEffect):
    """First-person flight through a rapid sequence of glowing ring-gates."""

    NAME = 'Portal Flight'
    AUTHOR = 'unicorn-viz'
    TAGS = ['flying', 'neon', 'intense', 'energetic', 'hard']
    PING_PONG_FRIENDS = [
        'Warp Drive', 'Canyon Run', 'Laser Tunnel', 'Wormhole',
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
        self._bass = self._mid = self._treble = 0.0
        self._beat = 0.0
        self._surge = 0.0
        # CPU-integrated flight phase (nebula_drift pattern).
        self._flight = float(self.rng.uniform(0.0, 248.0))

    def update(self, dt: float, audio: AudioData) -> None:
        super().update(dt, audio)
        react = max(0.05, float(self.parameters['reactivity']))
        self._bass = float(min(audio.bass_n * react, 1.6))
        self._mid = float(min(audio.mid_n * react, 1.6))
        self._treble = float(min(audio.treble_n * react, 1.6))
        if audio.beat > 0.5:
            self._beat = 1.0
            self._surge = 1.0
            self._hue = float((self._hue + self.rng.uniform(0.04, 0.14)) % 1.0)
        self._beat = max(0.0, self._beat - dt * 3.0)
        self._surge = max(0.0, self._surge - dt * 1.4)
        rate = 0.6 + max(0.05, float(self.parameters['speed'])) * 0.9
        self._flight = (self._flight
                        + dt * rate * (0.7 + self._bass * 1.0 + self._surge * 2.2) * 0.5
                        ) % 1000.0

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
        p['iSurge'].value = self._surge
        p['iFlight'].value = self._flight
        self._vao.render(moderngl.TRIANGLE_STRIP)

    def destroy(self) -> None:
        self._vao.release()
        self._vbo.release()
        self._prog.release()
