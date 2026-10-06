"""Lissajous visuals driven by the music analysis.

x(θ) = A·sin(a·θ + δ),  y(θ) = A·sin(b·θ)

Every bar shows a closed Lissajous figure with an integer ratio a:b. Moving to the
next bar never interpolates the ratio itself (fractional ratios give open curves);
instead the two closed curves are blended point by point, so the shape morphs
without ever breaking. Colour runs as a gradient along the curve and drifts through
the palette; the phase δ turns slowly and the amplitude breathes with the music.

Rendered with: manim -r 1920,1080 --fps 30 scene.py LissajousAmbient
The analysis JSON path comes from the YALIX_ANALYSIS environment variable.
"""

from __future__ import annotations

import json
import os

import numpy as np
from manim import ManimColor, Scene, Text, ValueTracker, VGroup, VMobject, always_redraw, interpolate_color, linear

BACKGROUND = "#000000"
# Catppuccin Mocha accents
PALETTE = ["#cba6f7", "#89b4fa", "#94e2d5", "#a6e3a1", "#f9e2af", "#fab387", "#f38ba8", "#b4befe"]
SUBTEXT = "#585b70"

# Eight closed figures, cycled bar by bar so the shapes keep changing.
SHAPES = [(3, 2), (4, 3), (5, 4), (5, 3), (2, 1), (3, 4), (5, 6), (4, 5)]

THETA = np.linspace(0, 2 * np.pi, 721)  # first point == last point: always closed


def smootherstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * x * (x * (6 * x - 15) + 10)


def lissajous(a: int, b: int, delta: float) -> np.ndarray:
    return np.stack([np.sin(a * THETA + delta), np.sin(b * THETA)], axis=1)


class LissajousAmbient(Scene):
    def construct(self):
        data = json.loads(open(os.environ["YALIX_ANALYSIS"]).read())
        fps = data["fps"]
        duration = data["duration"]
        bar = data["chord_seconds"]
        rms = np.array(data["rms"])

        self.camera.background_color = BACKGROUND
        t_tracker = ValueTracker(0.0)

        def colour(pos: float) -> ManimColor:
            """Continuous position on the palette ring (pos in palette units)."""
            i = int(np.floor(pos)) % len(PALETTE)
            f = pos - np.floor(pos)
            return interpolate_color(ManimColor(PALETTE[i]), ManimColor(PALETTE[(i + 1) % len(PALETTE)]), f)

        def figure(t: float) -> tuple[np.ndarray, float, float]:
            """Blended closed curve at time t, plus loudness and palette position."""
            idx = int(t // bar)
            local = (t - idx * bar) / bar
            a1, b1 = SHAPES[idx % len(SHAPES)]
            a2, b2 = SHAPES[(idx + 1) % len(SHAPES)]
            delta = 2 * np.pi * t / 60
            # Hold the figure for the first half of the bar, morph during the second half.
            m = smootherstep((local - 0.5) / 0.5)
            pts = (1 - m) * lissajous(a1, b1, delta) + m * lissajous(a2, b2, delta)
            level = float(rms[min(int(t * fps), len(rms) - 1)])
            return pts, level, t / (bar * 1.5)

        def curve(lag: float, scale: float, opacity: float, width: float, hue_shift: float = 0.0):
            def build():
                t = max(t_tracker.get_value() - lag, 0.0)
                pts, level, hue = figure(t)
                amp = 3.0 * scale * (0.85 + 0.15 * level)
                xy = np.column_stack([pts[:, 0] * amp * 1.35, pts[:, 1] * amp, np.zeros(len(pts))])
                mob = VMobject()
                mob.set_points_smoothly(xy)
                mob.set_stroke(
                    color=[colour(hue + hue_shift), colour(hue + hue_shift + 1.0)],
                    width=width,
                    opacity=opacity * (0.6 + 0.4 * level),
                )
                return mob

            return always_redraw(build)

        # Motion trail: echoes of the figure a moment ago, shifted half a colour ahead.
        trail = VGroup(*[curve(0.3 * k, 1.0, 0.2 * (1 - k / 7), 1.1, hue_shift=0.5) for k in range(1, 7)])
        layers = VGroup(
            trail,
            curve(0.0, 0.94, 0.18, 1.0, hue_shift=1.0),
            curve(0.0, 1.0, 0.30, 8.0),  # soft glow
            curve(0.0, 1.0, 0.95, 2.2),  # main stroke
        )
        mark = Text("yalix", font_size=22, color=SUBTEXT).to_corner(np.array([1, -1, 0]), buff=0.4)
        mark.set_opacity(0.8)

        self.add(layers, mark)
        self.play(t_tracker.animate.set_value(duration), run_time=duration, rate_func=linear)
