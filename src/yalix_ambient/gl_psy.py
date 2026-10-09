"""Figure families for the psytechno mix: the geometry of a Goa night.

- yantra: a yantra of interlocking triangles (four pointing up, five down) inside two rings of
  lotus petals and a square with four gates. Each triangle breathes on its own beat; each hit
  makes them all pulse outwards.
- tunnel: a hyperspace tunnel. Rings of a polygon recede in depth, each one turned a little more
  than the last, so the walls twist; the camera flies forward and every kick pushes it faster.
- chaos: the chaos game. Every particle jumps by one of a few contracting maps chosen at random,
  so together they draw the maps' fractal (the set that the maps turn into itself). The maps
  drift slowly, so the fractal morphs like a flame; each hit nudges one of them.
- moire: two sets of concentric rings, each ring rippled by a sine around its circumference,
  laid over each other from two centres; their interference draws the moiré.

Same contract as gl_forms: screen-normalized positions (y up), a hue and a brightness.
"""

from __future__ import annotations

import numpy as np

from yalix_ambient.gl_cymatics import _Strikes
from yalix_ambient.gl_render import H, W

ASPECT = W / H


def _segments(pts: list[tuple[tuple[float, float], tuple[float, float]]], u: np.ndarray, which: np.ndarray) -> np.ndarray:
    a = np.array([p[0] for p in pts])[which]
    b = np.array([p[1] for p in pts])[which]
    return a + (b - a) * u[:, None]


class Yantra:
    # (base y, apex y): triangles pointing up have the apex above the base, down below.
    TRIANGLES = ((-0.62, 0.78), (-0.42, 0.55), (-0.25, 0.42), (-0.05, 0.28),  # up
                 (0.66, -0.78), (0.48, -0.6), (0.3, -0.45), (0.18, -0.28), (0.05, -0.15))  # down  # fmt: skip

    def __init__(self, n: int, rng: np.random.Generator, vis) -> None:
        self.rng = rng
        segs, tri_id = [], []
        for i, (base, apex) in enumerate(self.TRIANGLES):
            half = np.sqrt(max(0.0, 0.9**2 - base**2)) * 0.98  # base corners on the inner circle
            corners = [(-half, base), (half, base), (0.0, apex)]
            for j in range(3):
                segs.append((corners[j], corners[(j + 1) % 3]))
                tri_id.append(i)
        self.segs, self.tri = segs, np.array(tri_id)
        self.n_tri = int(n * 0.5)
        lens = np.array([np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in segs])
        self.which = rng.choice(len(segs), self.n_tri, p=lens / lens.sum())
        self.u = rng.uniform(0, 1, self.n_tri)
        m = n - self.n_tri
        self.ring = rng.choice(4, m, p=[0.15, 0.3, 0.35, 0.2])  # circle, 8 petals, 16 petals, square
        self.th = rng.uniform(0, 2 * np.pi, m)
        self.breath = rng.uniform(0, 2 * np.pi, len(self.TRIANGLES))
        self.strike = _Strikes()
        self.pulse = 0.0
        self.square = vis.system != "lotus"

    def step(self, t: float, level: float, kick: float, vis):
        if self.strike(kick):
            self.pulse = 1.0
        p = _segments(self.segs, self.u, self.which)
        k = self.tri[self.which]
        scale = 1 + 0.02 * np.sin(t * 0.8 + self.breath[k]) + 0.04 * self.pulse
        p = p * scale[:, None]
        th = self.th
        r = np.select(
            [self.ring == 0, self.ring == 1, self.ring == 2],
            [np.full_like(th, 0.92), 0.95 + 0.12 * np.abs(np.cos(4 * th)), 1.08 + 0.08 * np.abs(np.cos(8 * th))],
            default=0.0,
        )
        outer = np.column_stack([r * np.cos(th + 0.02 * t), r * np.sin(th + 0.02 * t)])
        sq = self.ring == 3
        if self.square:  # the square with a gate in the middle of each side
            side = (th[sq] // (np.pi / 2)).astype(int)
            s = (th[sq] % (np.pi / 2)) / (np.pi / 2) * 2.6 - 1.3
            gate = np.abs(s) < 0.22
            depth = np.where(gate, 1.18, 1.3)
            pts = np.column_stack([s, depth])
            rot = side * np.pi / 2
            outer[sq] = np.column_stack([pts[:, 0] * np.cos(rot) - pts[:, 1] * np.sin(rot),
                                         pts[:, 0] * np.sin(rot) + pts[:, 1] * np.cos(rot)])  # fmt: skip
        else:
            outer[sq] = np.column_stack([1.25 * np.cos(th[sq] - 0.01 * t), 1.25 * np.sin(th[sq] - 0.01 * t)])
        xy = np.vstack([p, outer * (1 + 0.02 * self.pulse)]) * 0.75
        self.pulse *= 0.9
        hue = np.concatenate([k / len(self.TRIANGLES), 0.6 + 0.1 * self.ring])
        fade = np.concatenate([np.full(len(p), 0.8), np.full(len(outer), 0.5)]) * (0.75 + 0.25 * level)
        return xy, np.clip(hue, 0, 1), fade


class Tunnel:
    RINGS = 28

    def __init__(self, n: int, rng: np.random.Generator, vis) -> None:
        self.rng = rng
        self.k = {"tri": 3, "square": 4, "star": 10}.get(vis.system, 6)
        self.ring = rng.integers(0, self.RINGS, n)
        self.u = rng.uniform(0, 1, n)
        self.side = rng.integers(0, self.k, n)
        self.z = 0.0
        self.speed = 0.0
        self.strike = _Strikes()
        self.twist = rng.choice([-1, 1]) * rng.uniform(0.08, 0.14)

    def step(self, t: float, level: float, kick: float, vis):
        s = self.strike(kick)
        self.speed = max(self.speed * 0.93, 0.6 * s)
        self.z += (0.025 + 0.06 * self.speed) * vis.speed
        depth = (self.ring - self.z) % self.RINGS + 0.3
        a0 = 2 * np.pi * self.side / self.k + self.twist * (self.ring - self.z) + 0.05 * t
        a1 = a0 + 2 * np.pi / self.k
        if self.k == 10:  # a star: every other corner pulled in
            r0 = np.where(self.side % 2 == 0, 1.0, 0.55)
            r1 = np.where(self.side % 2 == 0, 0.55, 1.0)
        else:
            r0 = r1 = np.ones_like(a0)
        p0 = np.column_stack([r0 * np.cos(a0), r0 * np.sin(a0)])
        p1 = np.column_stack([r1 * np.cos(a1), r1 * np.sin(a1)])
        p = p0 + (p1 - p0) * self.u[:, None]
        persp = 1.4 / depth
        xy = p * persp[:, None]
        hue = ((self.ring - self.z) / self.RINGS * 2 + 0.1 * t) % 1.0
        fade = np.clip(depth / 2, 0, 1) * np.clip(1.2 - depth / self.RINGS, 0, 1) * (0.7 + 0.3 * level)
        return xy, hue, fade


class Chaos:
    """The chaos game on a few affine maps that drift: a morphing fractal."""

    def __init__(self, n: int, rng: np.random.Generator, vis) -> None:
        self.rng, self.n = rng, n
        kind = vis.system
        if kind == "star":  # five rotated copies, shrunk towards the rim: a fivefold flower
            self.base = [(0.42, 2 * np.pi * i / 5 + 0.3, 0.55 * np.array([np.cos(2 * np.pi * i / 5), np.sin(2 * np.pi * i / 5)])) for i in range(5)]
        elif kind == "sierpinski":
            self.base = [(0.5, 0.0, 0.6 * np.array([np.cos(a), np.sin(a)])) for a in (np.pi / 2, 7 * np.pi / 6, 11 * np.pi / 6)]
        else:  # flame: four random maps
            angles = rng.uniform(0, 2 * np.pi) + np.arange(4) * np.pi / 2 + rng.normal(0, 0.3, 4)
            self.base = [(rng.uniform(0.5, 0.68), rng.uniform(0, 2 * np.pi), 0.6 * np.array([np.cos(a), np.sin(a)])) for a in angles]
        self.drift = rng.uniform(0, 2 * np.pi, (len(self.base), 3))
        self.p = rng.uniform(-1, 1, (n, 2))
        self.col = rng.uniform(0, 1, n)
        self.strike = _Strikes()
        self.nudge = np.zeros(len(self.base))
        self.swirl = kind == "flame"
        self.c, self.ext = None, None

    def step(self, t: float, level: float, kick: float, vis):
        if self.strike(kick):
            self.nudge[int(self.rng.integers(0, len(self.base)))] = 1.0
        for _ in range(2):
            pick = self.rng.integers(0, len(self.base), self.n)
            out = np.empty_like(self.p)
            for i, (s, a, c) in enumerate(self.base):
                m = pick == i
                d = self.drift[i]
                ss = s * (1 + 0.08 * np.sin(0.05 * t + d[0]))
                aa = a + 0.4 * np.sin(0.03 * t + d[1]) + 0.3 * self.nudge[i]
                cc = c * (1 + 0.15 * np.sin(0.04 * t + d[2]))
                q = self.p[m]
                rot = np.column_stack([q[:, 0] * np.cos(aa) - q[:, 1] * np.sin(aa), q[:, 0] * np.sin(aa) + q[:, 1] * np.cos(aa)])
                q = rot * ss + cc
                if self.swirl:  # the flame's swirl variation: a twist that grows with radius
                    r2 = (q**2).sum(1, keepdims=True)
                    q = np.column_stack([q[:, 0] * np.sin(r2[:, 0]) - q[:, 1] * np.cos(r2[:, 0]),
                                         q[:, 0] * np.cos(r2[:, 0]) + q[:, 1] * np.sin(r2[:, 0])]) * 0.25 + q * 0.75  # fmt: skip
                out[m] = q
            self.p = np.clip(out, -1.6, 1.6)
            self.col = 0.6 * self.col + 0.4 * pick / max(len(self.base) - 1, 1)
        self.nudge *= 0.95
        # Keep the fractal centred and filling the frame, eased so the camera never jumps.
        c = np.median(self.p, axis=0)
        ext = np.percentile(np.abs(self.p - c), 98) + 1e-6
        self.c = c if self.c is None else 0.97 * self.c + 0.03 * c
        self.ext = ext if self.ext is None else 0.97 * self.ext + 0.03 * ext
        xy = (self.p - self.c) / self.ext * 0.95
        fade = np.full(self.n, 0.7 + 0.3 * level)
        return xy, self.col, fade


class Moire:
    def __init__(self, n: int, rng: np.random.Generator, vis) -> None:
        self.rng = rng
        self.rings = 18  # fewer, bolder rings: the interference reads at a glance
        self.set = rng.integers(0, 2, n)
        self.i = rng.integers(1, self.rings + 1, n)
        self.th = rng.uniform(0, 2 * np.pi, n)
        self.k = {"petals": 6, "waves": 11}.get(vis.system, 8)
        self.strike = _Strikes()
        self.amp = 0.0

    def step(self, t: float, level: float, kick: float, vis):
        if self.strike(kick):
            self.amp = 1.0
        sep = 0.18 + 0.12 * np.sin(t * 0.07)
        cx = np.where(self.set == 0, -sep, sep)
        r = self.i / self.rings * 1.05
        sgn = np.where(self.set == 0, 1, -1)
        r = r * (1 + (0.025 + 0.03 * self.amp) * np.sin(self.k * self.th + sgn * 0.6 * t + self.i * 0.25))
        xy = np.column_stack([cx + r * np.cos(self.th + sgn * 0.01 * t), r * np.sin(self.th + sgn * 0.01 * t)])
        self.amp *= 0.92
        hue = (self.i / self.rings + 0.5 * self.set) % 1.0
        fade = np.clip(1.4 - 0.6 * r, 0.5, 1) * (0.8 + 0.2 * level)
        return xy, hue, fade


PSY = {"yantra": Yantra, "tunnel": Tunnel, "chaos": Chaos, "moire": Moire}
