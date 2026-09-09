"""Cloud Surfer — running the top of a cloud deck at dawn.

The 1.x version sampled one fbm field across the whole screen.  With no horizon
and no ground plane there was nothing to move past, so it read as drifting
texture rather than as flight, and the deck had no shape.

The rebuild puts the clouds **on a plane**.  Each pixel's ray is intersected
with the deck below, and the cloud field is sampled at the point it lands on,
so the texture compresses toward the horizon exactly as ground does -- which is
the whole sensation of speed.  Above the deck is sky and a sun; below the tops,
the deck falls away into shadow.

Cloud tops are lit from the sun's side and shadowed on the other, sampled from
the same field a short step toward the light, so relief comes out of the cloud
shape rather than being painted on.  Towers rise out of the deck and pass
close; the horizon banks as the surfer carves.

**The bang.** A beat strikes lightning *inside* the deck: a flash localised to
one cloud cell, lighting that cloud from within and throwing its neighbours
into silhouette, then guttering out.

Audio reactivity:
  - bass    -> speed over the deck, tower height, sun bloom
  - mid     -> deck turbulence, palette drift
  - treble  -> spray off the tops, rim sparkle
  - beat    -> lightning inside the deck, horizon flash

Randomised per activation: palette hue, sky variant, cloud seed, heading, and
``self.time``.

Config keys (under ``[effects.CloudSurfer]``):
    speed, zoom, reactivity, glitter, deck_height
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
uniform float iRun;         // distance over the deck, wrapped
uniform float iRoll;        // bank
uniform float iAlt;         // height above the tops
uniform float iSky;
uniform float iSeed;
uniform float iBolt;        // lightning envelope
uniform vec2  iBoltP;       // where in the deck it struck

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
    for (int i = 0; i < 5; i++) { v += a * vnoise(x); x *= 2.07; a *= 0.5; }
    return v;
}
vec3 palette(float t) { return 0.5 + 0.5 * cos(TAU * (t + vec3(0.0, 0.33, 0.67))); }

// The deck's density at a point on the plane.
float deck(vec2 w, float detail) {
    return fbm(w * 0.55 + iSeed) * 1.15
         + fbm(w * 1.90 - iSeed * 0.5) * 0.42 * detail;
}

void main() {
    float A = iResolution.x / max(iResolution.y, 1.0);
    float tb = mod(iTime, 600.0);
    vec2 p = (v_uv - 0.5) * vec2(A, 1.0) / max(iZoom * (1.0 - iPunch * 0.05), 0.05);
    float cs = cos(iRoll), sn = sin(iRoll);
    p = mat2(cs, -sn, sn, cs) * p;

    vec3 neonA = palette(iHue);
    vec3 neonB = palette(iHue + 0.40);
    vec3 sunC = mix(vec3(1.0, 0.78, 0.42), neonA, 0.30);
    vec2 sunP = vec2(0.34, 0.16);

    // ── sky ────────────────────────────────────────────────────────────────
    float up = clamp(p.y * 1.2 + 0.42, 0.0, 1.0);
    vec3 lo, hi;
    if (iSky < 0.34)      { lo = vec3(1.00, 0.62, 0.34); hi = vec3(0.06, 0.14, 0.42); }
    else if (iSky < 0.67) { lo = vec3(0.98, 0.80, 0.52); hi = vec3(0.05, 0.28, 0.55); }
    else                  { lo = vec3(0.86, 0.34, 0.52); hi = vec3(0.05, 0.08, 0.30); }
    vec3 col = mix(lo, hi, pow(up, 0.80));
    col = mix(col, neonB, 0.14 + iMid * 0.10);
    col += sunC * exp(-length(p - sunP) * 4.2) * (0.85 + iBass * 0.8);
    col += sunC * exp(-length(p - sunP) * 16.0) * 1.2;

    // ── the deck ───────────────────────────────────────────────────────────
    // Intersect the ray with the plane the clouds sit on. Sampling the field
    // in screen space instead -- which is what 1.x did -- gives texture that
    // never compresses toward the horizon, so nothing reads as distance.
    if (p.y < -0.005) {
        float t = iAlt / (-p.y);
        if (t < 120.0) {
            vec2 w = vec2(p.x * t, t) + vec2(0.0, iRun);
            float fade = clamp(1.0 - t / 60.0, 0.0, 1.0);
            float d = deck(w, fade);
            // Light the tops from the sun's side by sampling a step toward it:
            // the relief then comes out of the cloud's own shape.
            float dl = deck(w + vec2(0.45, -0.25), fade);
            float slope = clamp((d - dl) * 2.6, -1.0, 1.0);

            float dens = smoothstep(0.42, 0.92, d + iBass * 0.10);
            vec3 lit = mix(vec3(0.98, 0.92, 0.88), sunC, 0.35);
            vec3 shade = mix(vec3(0.26, 0.30, 0.46), neonB, 0.30);
            vec3 cloud = mix(shade, lit, clamp(0.5 + slope * 0.9, 0.0, 1.0));
            // Towers catch the light on their tops.
            cloud += lit * smoothstep(0.86, 1.05, d) * 0.55 * (0.5 + iBass * 0.6);

            // Lightning inside the deck: it lights one cloud from within and
            // throws the rest into silhouette.
            float bd = length(w - iBoltP);
            cloud += mix(vec3(0.85, 0.92, 1.0), neonA, 0.2)
                   * exp(-bd * 0.35) * iBolt * (0.6 + dens * 2.4) * 2.2;

            // Below the tops the deck falls away into its own shadow.
            vec3 gap = mix(vec3(0.10, 0.12, 0.24), neonB * 0.4, 0.4);
            vec3 surf = mix(gap, cloud, dens);
            col = mix(col, surf, fade * 0.97 + 0.03);
            // Haze at the horizon so the deck meets the sky instead of ending.
            // Haze all the way to opaque at the horizon: leaving even a
            // little deck visible there puts a moire band on the skyline,
            // because the field's period falls below a pixel at that angle.
            col = mix(col, mix(lo, hi, 0.35), pow(1.0 - fade, 0.75) * 0.99);

            // Spray tearing off the tops.
            vec2 scell = w * 6.0;
            vec2 sg = floor(scell);
            if (hash21(sg) > 0.988) {
                vec2 sd2 = fract(scell) - vec2(hash21(sg + 3.1), hash21(sg + 7.7));
                col += vec3(1.0) * exp(-dot(sd2, sd2) * 24.0) * dens
                     * (0.3 + iTreble * 0.9) * iGlitter * fade;
            }
        }
    }

    // Sun shafts raking across the whole frame.
    vec2 sd = p - sunP;
    float shaft = pow(max(0.0, 1.0 - length(sd) * 0.55), 3.0)
                * (0.5 + 0.5 * sin(atan(sd.y, sd.x) * 14.0 + tb * 0.15));
    col += sunC * shaft * 0.13 * (0.4 + iMid * 0.9);

    col += vec3(0.9, 0.95, 1.0) * iBolt * 0.16;
    col += neonA * iBeat * 0.05;

    col *= smoothstep(1.60, 0.28, length(v_uv - 0.5));
    col = col / (1.0 + col * 0.45);
    col = pow(clamp(col, 0.0, 1.0), vec3(0.88));
    fragColor = vec4(col, 1.0);
}
"""


class CloudSurfer(BaseEffect):
    """Running the tops of a cloud deck, with lightning inside it."""

    NAME = 'Cloud Surfer'
    AUTHOR = 'unicorn-viz'
    TAGS = ['flying', 'neon', 'groovy', 'energetic', 'chill']
    PING_PONG_FRIENDS = [
        'Wingsuit Dive', 'Canyon Run', 'Nebula Drift', 'Audio Spectrum',
    ]

    _MIN_ZOOM = 0.75

    def _init(self) -> None:
        self.parameters = {
            'speed': float(self.config.get('speed', 1.0)),
            'zoom': float(self.config.get('zoom', 1.0)),
            'reactivity': float(self.config.get('reactivity', 1.2)),
            'glitter': float(self.config.get('glitter', 1.0)),
            'deck_height': float(self.config.get('deck_height', 1.0)),
        }
        self._prog = self._make_program(_VERT, _FRAG)
        self._vao, self._vbo = self._fullscreen_quad(self._prog)

        self._bass = self._mid = self._treble = 0.0
        self._beat = 0.0
        self._punch = 0.0
        self._bolt = 0.0
        self._bolt_p = (0.0, 0.0)
        self._hue = float(self.rng.uniform(0.0, 1.0))
        self._sky = float(self.rng.uniform(0.0, 1.0))
        self._seed = float(self.rng.uniform(0.0, 30.0))
        self._run = float(self.rng.uniform(0.0, 200.0))
        self._roll = 0.0
        self._alt = 0.55

    def update(self, dt: float, audio: AudioData) -> None:
        """Run the deck, carve across it, and strike inside it on the beat."""
        super().update(dt, audio)
        react = max(0.05, float(self.parameters['reactivity']))
        self._bass = float(np.clip(audio.bass_n * react, 0.0, 1.8))
        self._mid = float(np.clip(audio.mid_n * react, 0.0, 1.8))
        self._treble = float(np.clip(audio.treble_n * react, 0.0, 1.8))
        if audio.beat > 0.5:
            self._beat = 1.0
            self._punch = 1.0
            self._bolt = 1.0
            # Somewhere out in the deck ahead of the surfer.
            self._bolt_p = (float(self.rng.uniform(-9.0, 9.0)),
                            self._run + float(self.rng.uniform(6.0, 26.0)))
        self._beat = max(0.0, self._beat - dt * 3.2)
        self._punch = max(0.0, self._punch - dt * 2.4)
        self._bolt = max(0.0, self._bolt - dt * 3.4)

        speed = max(0.05, float(self.parameters['speed']))
        # Wrapped so the sampling coordinate never grows without bound.
        self._run = (self._run + dt * speed * (3.4 + self._bass * 6.0)) % 400.0
        # Carve: the surfer banks back and forth across the deck.
        self._roll += (np.sin(self._run * 0.020 + self._seed) * 0.34 - self._roll) \
            * min(1.0, dt * 1.6)
        height = max(0.2, float(self.parameters['deck_height']))
        self._alt = (0.42 + 0.20 * float(np.sin(self._run * 0.013))) * height

    def render(self) -> None:
        """Upload the run state."""
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
        p['iRun'].value = float(self._run)
        p['iRoll'].value = float(self._roll)
        p['iAlt'].value = float(self._alt)
        p['iSky'].value = float(self._sky)
        p['iSeed'].value = float(self._seed)
        p['iBolt'].value = float(self._bolt)
        p['iBoltP'].value = self._bolt_p
        self._vao.render(moderngl.TRIANGLE_STRIP)

    def destroy(self) -> None:
        """Release GL resources."""
        self._vao.release()
        self._vbo.release()
        self._prog.release()
