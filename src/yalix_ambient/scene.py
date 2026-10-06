"""Lissajous visuals driven by the music analysis.

x(θ) = A·sin(a·θ + δ),  y(θ) = A·sin(b·θ)

Each chord sets the frequency ratio a:b. Between chords the ratio is interpolated,
so the curve opens and closes as the harmony moves. The phase δ drifts slowly and
the amplitude A breathes with the loudness of the track.

Rendered with: manim -r 1920,1080 --fps 30 scene.py LissajousAmbient
The analysis JSON path comes from the YALIX_ANALYSIS environment variable.
"""

from __future__ import annotations

import json
import os

import numpy as np
from manim import ManimColor, Scene, Text, ValueTracker, VGroup, VMobject, always_redraw, interpolate_color, linear

# Catppuccin Mocha
BASE = "#1e1e2e"
PALETTE = ["#cba6f7", "#89b4fa", "#94e2d5", "#fab387"]  # mauve, blue, teal, peach
SUBTEXT = "#6c7086"


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


class LissajousAmbient(Scene):
    def construct(self):
        data = json.loads(open(os.environ["YALIX_ANALYSIS"]).read())
        fps = data["fps"]
        duration = data["duration"]
        chord_s = data["chord_seconds"]
        ratios = data["ratios"]
        rms = np.array(data["rms"])
        n_ratios = len(ratios)

        self.camera.background_color = BASE
        theta = np.linspace(0, 2 * np.pi, 361)  # closed: first point == last point
        t_tracker = ValueTracker(0.0)

        def state(t: float):
            """Interpolated ratio, colour and loudness at time t."""
            idx = int(t // chord_s)
            local = (t - idx * chord_s) / chord_s
            cur, nxt = ratios[idx % n_ratios], ratios[(idx + 1) % n_ratios]
            # Hold the chord's ratio, morph during the last 25% of the bar.
            m = smoothstep((local - 0.75) / 0.25)
            a = cur[0] + (nxt[0] - cur[0]) * m
            b = cur[1] + (nxt[1] - cur[1]) * m
            col = interpolate_color(
                ManimColor(PALETTE[idx % len(PALETTE)]), ManimColor(PALETTE[(idx + 1) % len(PALETTE)]), m
            )
            level = float(rms[min(int(t * fps), len(rms) - 1)])
            return a, b, col, level

        def curve(phase_offset: float, scale: float, opacity: float, width: float, lag: float = 0.0):
            def build():
                t = max(t_tracker.get_value() - lag, 0.0)
                a, b, col, level = state(t)
                delta = 2 * np.pi * t / 45 + phase_offset
                amp = 3.0 * scale * (0.82 + 0.18 * level)
                x = amp * 1.35 * np.sin(a * theta + delta)
                y = amp * np.sin(b * theta)
                pts = np.stack([x, y, np.zeros_like(x)], axis=1)
                mob = VMobject()
                mob.set_points_smoothly(pts)
                mob.set_stroke(col, width=width, opacity=opacity * (0.55 + 0.45 * level))
                return mob

            return always_redraw(build)

        # Layered strokes give a soft glow without post-processing.
        # Echoes of the curve a moment ago form a soft motion trail.
        trail = VGroup(*[curve(0.0, 1.0, 0.22 * (1 - k / 6), 1.2, lag=0.35 * k) for k in range(1, 6)])
        layers = VGroup(
            trail,
            curve(0.12, 0.92, 0.25, 1.4),
            curve(0.24, 0.84, 0.15, 1.0),
            curve(0.04, 1.00, 0.35, 7.0),
            curve(0.00, 1.00, 0.95, 2.2),
        )
        mark = Text("yalix", font_size=22, color=SUBTEXT).to_corner(np.array([1, -1, 0]), buff=0.4)
        mark.set_opacity(0.7)

        self.add(layers, mark)
        self.play(t_tracker.animate.set_value(duration), run_time=duration, rate_func=linear)
