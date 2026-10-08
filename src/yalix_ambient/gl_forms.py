"""More 2D/3D 'form' families for the particle renderer: Chladni plates, phyllotaxis, double
pendulums, torus knots, a rotating tesseract, point vortices and Maurer roses.

Same contract as gl_machines: positions in screen-normalized coordinates (x in [-W/H, W/H],
y in [-1, 1]), a hue in [0, 1] and a brightness in [0, 1] per particle, every frame. The
variant comes from `vis.system` where it matters and everything else from `vis.seed`.
"""

from __future__ import annotations

import itertools

import numpy as np

from yalix_ambient.gl_render import H, W

ASPECT = W / H


def _rot(xy: np.ndarray, a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.column_stack([xy[:, 0] * c - xy[:, 1] * s, xy[:, 0] * s + xy[:, 1] * c])


class Chladni:
    """Sand on a vibrating square plate: grains slide towards the nodal lines of the current
    mode and get shaken loose when the plate jumps to the next one (every ~24 s)."""

    def __init__(self, n: int, rng: np.random.Generator, vis) -> None:
        self.rng = rng
        pairs = [(1, 3), (2, 5), (3, 4), (1, 6), (3, 7), (2, 7), (4, 5), (3, 8), (5, 6), (1, 8)]
        order = rng.permutation(len(pairs))
        self.modes = [pairs[i] for i in order]
        self.p = rng.uniform(-1, 1, (n, 2))
        self.sign = 1.0 if rng.random() < 0.5 else -1.0
        self.period = 24.0

    def _field(self, p: np.ndarray, nm: tuple[int, int]):
        n, m = nm
        x, y = (p[:, 0] + 1) * np.pi / 2, (p[:, 1] + 1) * np.pi / 2
        f = np.sin(n * x) * np.sin(m * y) + self.sign * np.sin(m * x) * np.sin(n * y)
        fx = n * np.cos(n * x) * np.sin(m * y) + self.sign * m * np.cos(m * x) * np.sin(n * y)
        fy = m * np.sin(n * x) * np.cos(m * y) + self.sign * n * np.sin(m * x) * np.cos(n * y)
        return f, np.column_stack([fx, fy])

    def step(self, t: float, level: float, kick: float, vis):
        k = int(t // self.period)
        u = t / self.period - k
        f, g = self._field(self.p, self.modes[k % len(self.modes)])
        shake = 0.05 * np.exp(-u * 18) + 0.004 + 0.01 * kick
        self.p -= (f[:, None] * g) * 0.0025 * vis.speed
        self.p += self.rng.standard_normal(self.p.shape) * shake
        self.p = np.clip(self.p, -1, 1)
        xy = self.p * 0.9
        hue = np.clip(np.abs(f) * 2.0, 0, 1)
        fade = np.clip(1.0 - np.abs(f) * 1.5, 0.15, 1.0) * (0.8 + 0.2 * level)
        return xy, hue, fade


class Phyllotaxis:
    """The sunflower: seed i at radius sqrt(i) and angle i * phi. The angle sweeps slowly
    around the golden angle, so the spiral arms open, close and swap direction."""

    def __init__(self, n: int, rng: np.random.Generator, vis) -> None:
        self.i = np.arange(n, dtype=float)
        self.r = np.sqrt((self.i + 0.5) / n) * 0.92
        self.golden = np.pi * (3 - np.sqrt(5))
        self.amp = rng.uniform(0.004, 0.012)
        self.rate = rng.uniform(0.006, 0.012)
        self.ph = rng.uniform(0, 2 * np.pi)

    def step(self, t: float, level: float, kick: float, vis):
        a = self.golden + self.amp * np.sin(2 * np.pi * self.rate * t * vis.speed + self.ph)
        th = self.i * a + 0.03 * t
        r = self.r * (1 + 0.02 * kick + 0.015 * np.sin(self.i * 0.002 - t))
        xy = np.column_stack([r * np.cos(th), r * np.sin(th)])
        hue = (self.i / len(self.i) + 0.05 * t) % 1.0
        fade = np.full(len(xy), 0.75 + 0.25 * level)
        return xy, hue, fade


class Pendulums:
    """Thousands of double pendulums released from almost the same angle: one bright line at
    first, then the fan opens and the chaos fills the circle. Released again every ~65 s."""

    def __init__(self, n: int, rng: np.random.Generator, vis) -> None:
        self.n, self.rng = n, rng
        self.l1, self.l2 = 1.0, rng.uniform(0.7, 1.1)
        self.m1, self.m2 = 1.0, rng.uniform(0.6, 1.2)
        self.a0 = rng.uniform(1.6, 2.6, 2)
        self.period = 65.0
        self._release()
        self.k = 0

    def _release(self) -> None:
        eps = np.linspace(-1, 1, self.n) * 2e-4
        self.s = np.column_stack([self.a0[0] + eps, self.a0[1] + eps * 0.5, np.zeros(self.n), np.zeros(self.n)])

    def _f(self, s: np.ndarray) -> np.ndarray:
        g, l1, l2, m1, m2 = 9.81, self.l1, self.l2, self.m1, self.m2
        a1, a2, w1, w2 = s.T
        d = a1 - a2
        den = 2 * m1 + m2 - m2 * np.cos(2 * d)
        dw1 = (-g * (2 * m1 + m2) * np.sin(a1) - m2 * g * np.sin(a1 - 2 * a2)
               - 2 * np.sin(d) * m2 * (w2 * w2 * l2 + w1 * w1 * l1 * np.cos(d))) / (l1 * den)
        dw2 = (2 * np.sin(d) * (w1 * w1 * l1 * (m1 + m2) + g * (m1 + m2) * np.cos(a1)
               + w2 * w2 * l2 * m2 * np.cos(d))) / (l2 * den)  # fmt: skip
        return np.column_stack([w1, w2, dw1, dw2])

    def step(self, t: float, level: float, kick: float, vis):
        k = int(t // self.period)
        if k != self.k:
            self.k = k
            self._release()
        h = 0.012 * vis.speed
        for _ in range(2):
            s = self.s
            k1 = self._f(s)
            k2 = self._f(s + h / 2 * k1)
            k3 = self._f(s + h / 2 * k2)
            k4 = self._f(s + h * k3)
            self.s = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        a1, a2 = self.s[:, 0], self.s[:, 1]
        x1, y1 = self.l1 * np.sin(a1), -self.l1 * np.cos(a1)
        x2, y2 = x1 + self.l2 * np.sin(a2), y1 - self.l2 * np.cos(a2)
        scale = 0.9 / (self.l1 + self.l2)
        xy = np.column_stack([x2, -y2]) * scale  # screen y grows downwards
        u = (t % self.period) / self.period
        hue = np.linspace(0, 1, self.n)
        fade = np.full(self.n, 0.9) * np.clip(u * 30, 0, 1) * np.clip((1 - u) * 20, 0, 1) * (0.85 + 0.15 * level)
        return xy, hue, fade


class TorusKnot:
    """Particles flowing along a (p, q) torus knot that turns in 3D, seen in perspective."""

    def __init__(self, n: int, rng: np.random.Generator, vis) -> None:
        pairs = {"trefoil": (2, 3), "cinquefoil": (2, 5), "35": (3, 5), "37": (3, 7), "47": (4, 7), "58": (5, 8)}
        self.p, self.q = pairs.get(vis.system, list(pairs.values())[int(rng.integers(len(pairs)))])
        self.phi = rng.uniform(0, 2 * np.pi, n)
        self.v = rng.uniform(0.7, 1.2, n)
        self.off = rng.standard_normal((n, 3)) * 0.02
        self.tilt = rng.uniform(0.4, 0.9)

    def step(self, t: float, level: float, kick: float, vis):
        self.phi = (self.phi + self.v * 0.004 * vis.speed) % (2 * np.pi)
        p, q, phi = self.p, self.q, self.phi
        r = 0.38 * (1 + 0.15 * np.sin(t / 20))
        x = (1 + r * np.cos(q * phi)) * np.cos(p * phi)
        y = (1 + r * np.cos(q * phi)) * np.sin(p * phi)
        z = r * np.sin(q * phi)
        P = np.column_stack([x, y, z]) + self.off
        a, b = 0.05 * t, self.tilt + 0.2 * np.sin(t / 30)
        P = P @ np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]).T
        P = P @ np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]]).T
        persp = 1 / (1 + 0.25 * P[:, 2])
        xy = P[:, :2] * persp[:, None] * 0.58 * (1 + 0.012 * kick)
        hue = (phi / (2 * np.pi) * 2) % 1.0
        fade = np.clip(0.55 + 0.45 * persp, 0, 1) * (0.85 + 0.15 * level)
        return xy, hue, fade


class Tesseract:
    """A 4D polytope (hypercube or 16-cell) turning in two planes at once, projected to 3D
    and then to the screen; particles stream along its edges."""

    def __init__(self, n: int, rng: np.random.Generator, vis) -> None:
        if vis.system == "16cell":
            V = np.array([s * e for e in np.eye(4) for s in (1, -1)])
            E = [(i, j) for i, j in itertools.combinations(range(8), 2) if np.abs(V[i] + V[j]).sum() > 0]
        else:
            V = np.array(list(itertools.product((-1, 1), repeat=4)), dtype=float)
            E = [(i, j) for i, j in itertools.combinations(range(16), 2) if np.abs(V[i] - V[j]).sum() == 2]
        self.V, self.E = V / np.linalg.norm(V[0]), np.array(E)  # unit radius, so both polytopes fit
        self.e = rng.integers(0, len(E), n)
        self.s = rng.uniform(0, 1, n)
        self.v = rng.uniform(0.5, 1.5, n) * rng.choice([-1, 1], n)
        self.w = rng.uniform(0.08, 0.14, 2)

    def step(self, t: float, level: float, kick: float, vis):
        self.s = (self.s + self.v * 0.01 * vis.speed) % 1.0
        a, b = self.w * t * vis.speed
        R = np.eye(4)
        R[[0, 0, 3, 3], [0, 3, 0, 3]] = np.cos(a), -np.sin(a), np.sin(a), np.cos(a)  # XW plane
        R2 = np.eye(4)
        R2[[1, 1, 2, 2], [1, 2, 1, 2]] = np.cos(b), -np.sin(b), np.sin(b), np.cos(b)  # YZ plane
        V = self.V @ (R @ R2).T
        A, B = V[self.E[self.e, 0]], V[self.E[self.e, 1]]
        P = A * (1 - self.s[:, None]) + B * self.s[:, None]
        w = 1 / (2.6 - P[:, 3])  # 4D -> 3D
        P3 = P[:, :3] * w[:, None]
        c = 0.3 * t
        P3 = P3 @ np.array([[np.cos(c), 0, np.sin(c)], [0, 1, 0], [-np.sin(c), 0, np.cos(c)]]).T
        persp = 1 / (1 + 0.4 * P3[:, 2])
        xy = P3[:, :2] * persp[:, None] * 1.35 * (1 + 0.012 * kick)
        hue = np.clip((P[:, 3] + 1.5) / 3, 0, 1)
        fade = np.clip(0.4 + 0.6 * persp * w * 2, 0, 1) * (0.85 + 0.15 * level)
        return xy, hue, fade


class Vortices:
    """A few point vortices push each other around (Kirchhoff's equations); tracer particles
    are carried by their combined flow and spiral into the cores."""

    def __init__(self, n: int, rng: np.random.Generator, vis) -> None:
        k = max(3, min(int(vis.dipoles) + 1, 6))
        self.rng = rng
        self.z = rng.uniform(-0.5, 0.5, (k, 2)) * np.array([ASPECT, 1.0])
        self.g = rng.choice([-1.0, 1.0], k) * rng.uniform(0.6, 1.2, k)
        self.p = rng.uniform(-1, 1, (n, 2)) * np.array([ASPECT, 1.0])
        self.age = rng.uniform(0, 1, n)

    def _vel(self, pts: np.ndarray, eps: float = 0.04) -> np.ndarray:
        v = np.zeros_like(pts)
        for zj, gj in zip(self.z, self.g):
            d = pts - zj
            r2 = (d * d).sum(1) + eps * eps
            v += gj * np.column_stack([-d[:, 1], d[:, 0]]) / r2[:, None]
        return v / (2 * np.pi)

    def step(self, t: float, level: float, kick: float, vis):
        h = 0.004 * vis.speed
        vz = np.zeros_like(self.z)
        for i in range(len(self.z)):
            others = [j for j in range(len(self.z)) if j != i]
            d = self.z[i] - self.z[others]
            r2 = (d * d).sum(1) + 0.01
            vz[i] = (self.g[others, None] * np.column_stack([-d[:, 1], d[:, 0]]) / r2[:, None]).sum(0) / (2 * np.pi)
        self.z += vz * h * 8
        self.z = np.clip(self.z, [-ASPECT * 0.7, -0.7], [ASPECT * 0.7, 0.7])
        v = self._vel(self.p)
        sp = np.linalg.norm(v, axis=1)
        self.p += v / (1 + sp[:, None] * 0.15) * h * 8
        self.age += 1 / (30 * 10)
        out = (np.abs(self.p[:, 0]) > ASPECT) | (np.abs(self.p[:, 1]) > 1) | (self.age > 1)
        k = int(out.sum())
        if k:
            self.p[out] = self.rng.uniform(-1, 1, (k, 2)) * np.array([ASPECT, 1.0])
            self.age[out] = 0
        hue = np.clip(np.log1p(sp) / 3, 0, 1)
        fade = np.minimum(self.age * 8, 1) * np.minimum((1 - self.age) * 8, 1) * (0.8 + 0.2 * level)
        return self.p * 0.95, hue, fade


class MaurerRose:
    """A Maurer rose: 361 points of the rose r = sin(n th) joined by chords every d degrees.
    Particles run along the chords while d drifts, so the web keeps re-weaving itself."""

    def __init__(self, n: int, rng: np.random.Generator, vis) -> None:
        self.nn = int(rng.choice([2, 3, 4, 5, 6, 7]))
        self.d0 = float(rng.choice([29, 31, 37, 39, 41, 47, 59, 71, 97]))
        self.k = rng.integers(0, 360, n)
        self.s = rng.uniform(0, 1, n)
        self.v = rng.uniform(0.5, 1.2, n)

    def step(self, t: float, level: float, kick: float, vis):
        self.s += self.v * 0.02 * vis.speed
        wrap = self.s >= 1
        self.k[wrap] = (self.k[wrap] + 1) % 360
        self.s[wrap] -= 1
        d = np.radians(self.d0 + 0.6 * np.sin(t / 40))

        def pt(k):
            th = k * d
            r = np.sin(self.nn * th)
            return np.column_stack([r * np.cos(th), r * np.sin(th)])

        a, b = pt(self.k), pt(self.k + 1)
        xy = a * (1 - self.s[:, None]) + b * self.s[:, None]
        xy = _rot(xy, 0.01 * t) * 0.9 * (1 + 0.012 * kick)
        hue = self.k / 360
        fade = np.full(len(xy), 0.7 + 0.3 * level)
        return xy, hue, fade


FORMS = {
    "chladni": Chladni,
    "phyllotaxis": Phyllotaxis,
    "pendulums": Pendulums,
    "knot": TorusKnot,
    "tesseract": Tesseract,
    "vortices": Vortices,
    "rose": MaurerRose,
}
