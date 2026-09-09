"""Vector Hallway — the original neon trench, kept as its own effect.

This is the 1.x Canyon Run: flat neon walls, a scrolling grid floor and a hard
bank into the turns.  It never read as rock and was never trying to -- it reads
as a *vector hallway*, which is why it earns a name of its own rather than
being replaced.  Canyon Run went on to become an actual canyon; this stayed
what it always was.

Three bugs were fixed on the way in; see the pack README for the full write-up.
The one that mattered here is the angular seam: the speed streaks hashed
``floor(ang * 40.0)``, and ``atan`` wraps at +/-PI, so the cell index jumped by
251.3 -- not a whole number of cells -- and the streak field tore along the
negative x axis. The cells are now counted around the full turn and wrapped
before hashing, which closes it.

Audio reactivity:
  - bass    -> wall glow, speed
  - mid     -> palette drift
  - treble  -> streak density
  - beat    -> chromatic edge flash

Config keys (under ``[effects.VectorHallway]``):
    speed, reactivity
"""
from __future__ import annotations

import moderngl

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
uniform float iSpeed;
uniform float iHue;
uniform float iTurn;      // per-instance turn-phase offset
uniform float iFlight;    // CPU-integrated bass-speed flight phase

in  vec2 v_uv;
out vec4 fragColor;

#define TAU 6.28318530

float hash21(vec2 p) {
    // Bound the input before hashing. Callers add terms that grow with time or
    // depth; once one reaches ~200000 the multiply below (x123.34) lands past
    // float32's usable mantissa and the "random" field collapses into a fixed
    // lattice, which renders as a regular grid of marks instead of scatter.
    p = mod(p, 512.0);
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
vec3 palette(float t) { return 0.5 + 0.5 * cos(TAU * (t + vec3(0.0, 0.33, 0.67))); }

void main() {
    float A = iResolution.x / max(iResolution.y, 1.0);
    float t = iTime * (0.6 + iSpeed * 0.7);
    // iFlight is the CPU-integrated audio share of forward motion (see
    // update()): the old t * (k + iBass...) form multiplied unbounded time
    // by a per-frame audio delta, strobing walls/floor under music.
    float speed = 0.5 + iBass * 1.2;

    // Bank hard into turns, snap sharply on beat.
    float bank = sin(t * 0.22 + iTurn) * (0.35 + iMid * 0.25) + iBeat * 0.18;
    float cb = cos(bank), sb = sin(bank);
    vec2 p = vec2((v_uv.x - 0.5) * A, v_uv.y - 0.5);
    p = mat2(cb, -sb, sb, cb) * p;

    // Canyon walls: distance from centreline narrows/widens with bass squeeze
    // pulses, each side textured with scrolling glowing strata bands.
    float squeeze = 0.34 - 0.08 * (0.5 + 0.5 * sin(t * 1.3)) * iBass;
    float wallX = squeeze + 0.10 * sin(t * 0.5 + iTurn * 1.7);
    float side = sign(p.x);
    float distToWall = wallX - abs(p.x);
    float inCanyon = smoothstep(0.0, 0.02, distToWall);

    vec3 col = vec3(0.01, 0.01, 0.03);

    if (distToWall < 0.05) {
        // Perspective-mapped strata: bands recede toward the vanishing point.
        // za is clamped so the band frequency can't grow unbounded near the
        // vanishing point (that was aliasing into moire noise); the glow
        // itself fades out well before the clamp so it reads as bands
        // receding into a soft haze, not a flat wall of noise.
        float zaRaw = 1.0 / max(0.06, (0.5 - p.y));
        float za = min(zaRaw, 9.0);
        float band = floor((za * 2.2 - (t * 2.0 + iFlight)) * 3.0);
        float bandF = fract((za * 2.2 - (t * 2.0 + iFlight)) * 3.0);
        float glowLine = smoothstep(0.08, 0.0, min(bandF, 1.0 - bandF)) * smoothstep(9.0, 5.5, zaRaw);
        float hue = iHue + band * 0.06 + side * 0.08;
        vec3 wallCol = mix(vec3(0.05, 0.08, 0.16), palette(hue), 0.55);
        float wallShade = smoothstep(0.05, -0.02, distToWall);
        col = mix(col, wallCol * (0.4 + za * 0.06), wallShade);
        col += palette(hue + 0.15) * glowLine * wallShade * (0.6 + iTreble * 0.8);
        // Bright rim right at the canyon edge.
        col += palette(hue + 0.3) * smoothstep(0.018, 0.0, abs(distToWall)) * 1.2;
    }

    // Scrolling floor grid rushing underneath (below eye-level, p.y < -0.04).
    float floorDepth = -p.y - 0.04;
    if (floorDepth > 0.0 && distToWall > -0.2) {
        float fz = min(1.0 / max(0.03, floorDepth), 24.0);
        vec2 fp = vec2(p.x * fz, fz * 1.4 - (t * 2.75 + iFlight * 1.5));
        float gx = abs(fract(fp.x * 2.0) - 0.5);
        float gy = abs(fract(fp.y) - 0.5);
        float lineW = clamp(0.05 / max(fz * 0.15, 1.0), 0.01, 0.05);
        float grid = smoothstep(lineW, 0.0, min(gx, gy) / max(fz * 0.4, 1.0));
        vec3 floorCol = palette(iHue + 0.5);
        col = mix(col, vec3(0.02, 0.02, 0.06), smoothstep(0.0, 0.35, floorDepth));
        col += floorCol * grid * smoothstep(24.0, 3.0, fz) * (0.5 + iBass * 0.5);
    }

    // Forward speed streaks radiating from the vanishing point.
    vec2 vp = p;
    float ang = atan(vp.y, vp.x);
    float rad = length(vp);
    // The seed lattice is wrapped so it closes at atan()'s branch cut: the
    // original floor(ang * 40.0) moved the index by 251.3 there, which is not
    // a whole number of cells, so hash21() returned an unrelated value across
    // the negative x axis. 251 cells around the turn is the same angular
    // density the original had (40 per radian), so the streaks keep exactly
    // the spacing they always had -- picking a round 40 cells instead closed
    // the wrap but visibly thinned the field, which is not what this effect is
    // being kept for.
    float cell = mod(floor(ang / TAU * 251.0), 251.0);
    float streak = hash21(vec2(cell, 0.0));
    float streakLine = smoothstep(0.985, 1.0, sin(ang * 40.0 + streak * 6.0)) * (1.0 - smoothstep(0.05, 0.5, rad));
    col += vec3(0.9, 0.95, 1.0) * streakLine * (speed * 0.3 + iTreble * 0.4);

    // Beat chromatic edge flash.
    col.r += iBeat * 0.10 * (1.0 - inCanyon);
    col.b += iBeat * 0.06 * (1.0 - inCanyon) * -1.0;
    col = max(col, vec3(0.0));

    float vig = smoothstep(1.35, 0.2, length(v_uv - 0.5));
    col *= vig;
    col = col / (1.0 + col * 0.5);
    col = pow(clamp(col, 0.0, 1.0), vec3(0.85));
    fragColor = vec4(col, 1.0);
}
"""


class VectorHallway(BaseEffect):
    """First-person high-speed trench run through a glowing neon canyon."""

    NAME = 'Vector Hallway'
    AUTHOR = 'unicorn-viz'
    TAGS = ['flying', 'neon', 'intense', 'hard', 'energetic']
    PING_PONG_FRIENDS = [
        'Canyon Run',
        'Space Tunnel',
        'Bass Hammer',
        'Tron Grid',
        'Audio Spectrum',
    ]

    def _init(self) -> None:
        self.parameters = {
            'speed': float(self.config.get('speed', 1.0)),
            'reactivity': float(self.config.get('reactivity', 1.3)),
        }
        self._prog = self._make_program(_VERT, _FRAG)
        self._vao, self._vbo = self._fullscreen_quad(self._prog)
        self._hue = float(self.rng.uniform(0.0, 1.0))
        self._turn = float(self.rng.uniform(0.0, 6.28))
        self._bass = self._mid = self._treble = 0.0
        self._beat = 0.0
        # CPU-integrated audio flight phase (nebula_drift pattern).
        self._flight = float(self.rng.uniform(0.0, 1000.0))

    def update(self, dt: float, audio: AudioData) -> None:
        super().update(dt, audio)
        react = max(0.05, float(self.parameters['reactivity']))
        self._bass = float(min(audio.bass_n * react, 1.6))
        self._mid = float(min(audio.mid_n * react, 1.6))
        self._treble = float(min(audio.treble_n * react, 1.6))
        if audio.beat > 0.5:
            self._beat = 1.0
            self._hue = float((self._hue + self.rng.uniform(0.03, 0.12)) % 1.0)
        self._beat = max(0.0, self._beat - dt * 3.0)
        rate = 0.6 + max(0.05, float(self.parameters['speed'])) * 0.7
        self._flight = (self._flight + dt * rate * self._bass * 1.2) % 1000.0

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
        p['iTurn'].value = self._turn
        p['iFlight'].value = self._flight
        self._vao.render(moderngl.TRIANGLE_STRIP)

    def destroy(self) -> None:
        self._vao.release()
        self._vbo.release()
        self._prog.release()
