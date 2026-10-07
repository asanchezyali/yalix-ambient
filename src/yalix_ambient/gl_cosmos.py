"""2D 'cosmos' families for the particle renderer: n-body orbits, spiral galaxies, harmonographs,
iterated maps (Clifford, De Jong) and spirographs.

Same contract as gl_machines: positions in screen-normalized coordinates (x in [-W/H, W/H],
y in [-1, 1]), a hue in [0, 1] and a brightness in [0, 1] per particle, every frame. The
variant is picked with `vis.system` and the rest comes from `vis.seed`, so two episodes of
the same family never draw the same figure.
"""

from __future__ import annotations

import numpy as np

from yalix_ambient.gl_render import H, W

ASPECT = W / H


class Orbits:
    """Massive bodies on a known n-body solution; thousands of test particles fall around them.

    system: figure8 (Chenciner-Montgomery choreography), binary, hierarchical (a binary with a
    distant third), quad (four equal masses on a rotating square, slowly breaking up).
    """

    def __init__(self, n: int, rng: np.random.Generator, vis) -> None:
        self.rng, self.n = rng, n
        kind = vis.system if vis.system in ("figure8", "binary", "hierarchical", "quad") else "figure8"
        if kind == "figure8":
            p1, v3 = np.array([0.97000436, -0.24308753]), np.array([-0.93240737, -0.86473146])
            self.x = np.array([p1, -p1, [0.0, 0.0]])
            self.v = np.array([-v3 / 2, -v3 / 2, v3])
            self.m = np.ones(3)
            period = 6.3259
        elif kind == "binary":
            self.x = np.array([[-0.5, 0.0], [0.5, 0.0]])
            self.v = np.array([[0.0, -0.5], [0.0, 0.5]]) * np.sqrt(2.0)
            self.m = np.array([1.0, 1.0])
            period = 2 * np.pi * 0.5 / (0.5 * np.sqrt(2.0))
        elif kind == "hierarchical":
            self.x = np.array([[-0.18, 0.0], [0.18, 0.0], [1.1, 0.0]])
            vb = np.sqrt(1.0 / (4 * 0.18))
            vo = np.sqrt(2.0 / 1.1) * 0.9
            self.v = np.array([[0.0, -vb], [0.0, vb], [0.0, vo]])
            self.m = np.array([1.0, 1.0, 0.3])
            period = 2 * np.pi * 0.18 / vb
        else:
            r = 0.6
            ang = np.arange(4) * np.pi / 2 + np.pi / 4
            self.x = r * np.column_stack([np.cos(ang), np.sin(ang)])
            vq = np.sqrt((1 / (2 * r)) * (2 / np.sqrt(2) + 0.5))  # equal masses on a square, rotating
            self.v = vq * np.column_stack([-np.sin(ang), np.cos(ang)])
            self.m = np.ones(4)
            period = 2 * np.pi * r / vq
        self.x -= (self.m[:, None] * self.x).sum(0) / self.m.sum()
        self.v -= (self.m[:, None] * self.v).sum(0) / self.m.sum()
        self.dt = period / (24.0 * 30)  # one period every ~24 s at speed 1
        self.view = 0.62 / max(np.abs(self.x).max(), 0.6)
        self.p = np.zeros((n, 2))
        self.q = np.zeros((n, 2))
        self.age = np.zeros(n)
        self._spawn(np.arange(n))
        self.age = rng.uniform(0, 1, n)
        self.life = rng.uniform(0.5, 1.0, n)

    def _acc(self, pts: np.ndarray, eps: float) -> np.ndarray:
        a = np.zeros_like(pts)
        for xj, mj in zip(self.x, self.m):
            d = xj - pts
            r2 = (d * d).sum(1) + eps * eps
            a += mj * d / (r2 * np.sqrt(r2))[:, None]
        return a

    def _spawn(self, idx: np.ndarray) -> None:
        k = len(idx)
        if not k:
            return
        j = self.rng.choice(len(self.m), size=k, p=self.m / self.m.sum())
        r = self.rng.uniform(0.05, 0.32, k) * np.sqrt(self.m[j])
        th = self.rng.uniform(0, 2 * np.pi, k)
        u = np.column_stack([np.cos(th), np.sin(th)])
        vc = np.sqrt(self.m[j] / r) * self.rng.uniform(0.85, 1.1, k)
        sign = np.where(self.rng.random(k) < 0.85, 1.0, -1.0)
        self.p[idx] = self.x[j] + u * r[:, None]
        self.q[idx] = self.v[j] + sign[:, None] * vc[:, None] * np.column_stack([-u[:, 1], u[:, 0]])
        self.age[idx] = 0.0

    def step(self, t: float, level: float, kick: float, vis) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h = self.dt * vis.speed / 4
        for _ in range(4):
            # bodies: leapfrog on their own mutual gravity
            ab = np.zeros_like(self.x)
            for i in range(len(self.m)):
                d = self.x - self.x[i]
                r2 = (d * d).sum(1) + 1e-6
                r2[i] = np.inf
                ab[i] = (self.m[:, None] * d / (r2 * np.sqrt(r2))[:, None]).sum(0)
            self.v += ab * h
            self.x += self.v * h
            self.q += self._acc(self.p, 0.035) * h
            self.p += self.q * h
        self.age += 1.0 / (30 * 14) / self.life
        r_min = np.min([np.linalg.norm(self.p - xj, axis=1) for xj in self.x], axis=0)
        out = (np.abs(self.p).max(1) > 2.4) | (r_min < 0.012) | (self.age > 1)
        self._spawn(np.flatnonzero(out))
        xy = self.p * self.view * (1 + 0.01 * kick)
        spd = np.linalg.norm(self.q, axis=1)
        hue = np.clip(spd / (np.percentile(spd, 95) + 1e-9), 0, 1)
        fade = np.minimum(self.age * 10, 1) * np.minimum((1 - self.age) * 6, 1) * (0.8 + 0.2 * level)
        return xy, hue, fade


class Galaxy:
    """Stars on nested, slowly twisted ellipses: the density wave draws the arms by itself.

    system: spiral2, spiral3, barred. Inner stars turn faster (flat rotation curve), the
    pattern turns slowly, and the disc is seen tilted while the camera drifts.
    """

    def __init__(self, n: int, rng: np.random.Generator, vis) -> None:
        self.rng = rng
        kind = vis.system if vis.system in ("spiral2", "spiral3", "barred") else "spiral2"
        self.m = 3 if kind == "spiral3" else 2
        self.a = np.clip(rng.exponential(0.32, n), 0.02, 1.0)
        bulge = rng.random(n) < 0.18
        self.a[bulge] = np.abs(rng.normal(0, 0.07, bulge.sum())) + 0.01
        self.e = np.where(bulge, 0.05, rng.uniform(0.18, 0.28, n))
        if kind == "barred":
            bar = self.a < 0.22
            self.e[bar] = 0.55
        self.twist = rng.uniform(2.6, 3.6) * (1 if rng.random() < 0.5 else -1)
        self.th = rng.uniform(0, 2 * np.pi, n)
        self.w = 0.9 / np.maximum(self.a, 0.06)  # flat rotation: angular speed ~ 1 / r
        self.tilt = rng.uniform(0.35, 0.6)
        self.spin0 = rng.uniform(0, 2 * np.pi)
        self.jit = rng.standard_normal((n, 2)) * 0.012 * (0.4 + self.a[:, None])
        self.bulge = bulge

    def step(self, t: float, level: float, kick: float, vis) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.th += self.w * 0.004 * vis.speed
        phase = self.twist * self.a + 0.02 * t * vis.speed
        r = self.a * (1 + self.e * np.cos(self.m * (self.th - phase)))
        x, y = r * np.cos(self.th), r * np.sin(self.th)
        spin = self.spin0 + 0.01 * t
        xr, yr = x * np.cos(spin) - y * np.sin(spin), x * np.sin(spin) + y * np.cos(spin)
        tilt = self.tilt + 0.08 * np.sin(t / 40)
        xy = np.column_stack([xr * 1.05, yr * np.cos(tilt) * 1.05]) + self.jit
        xy *= 1 + 0.01 * kick
        hue = np.clip(self.a / 0.9, 0, 1)
        fade = np.where(self.bulge, 0.45, 0.9) * (0.85 + 0.15 * level)
        return xy, hue, fade


class Harmonograph:
    """Two damped pendulums per axis draw a closing spiral; particles run along the pen line
    while the pendulums' phases drift, so the drawing slowly changes shape."""

    def __init__(self, n: int, rng: np.random.Generator, vis) -> None:
        ratios = [(2, 3), (3, 4), (3, 5), (4, 5), (1, 3), (5, 6)]
        a, b = ratios[int(rng.integers(len(ratios)))]
        self.f = np.array([a, b, b, a], dtype=float) + rng.normal(0, 0.006, 4)
        self.ph = rng.uniform(0, 2 * np.pi, 4)
        self.drift = rng.normal(0, 0.02, 4)
        self.amp = rng.uniform(0.35, 0.5, 4)
        self.damp = rng.uniform(0.0025, 0.004)
        self.S = 260.0
        self.s = rng.uniform(0, self.S, n)
        self.v = rng.uniform(0.6, 1.2, n)

    def step(self, t: float, level: float, kick: float, vis) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.s = (self.s + self.v * 0.05 * vis.speed) % self.S
        s, ph = self.s, self.ph + self.drift * t
        dec = np.exp(-self.damp * s)
        x = (self.amp[0] * np.sin(self.f[0] * s + ph[0]) + self.amp[1] * np.sin(self.f[1] * s + ph[1])) * dec
        y = (self.amp[2] * np.sin(self.f[2] * s + ph[2]) + self.amp[3] * np.sin(self.f[3] * s + ph[3])) * dec
        xy = np.column_stack([x * 1.15, y]) * (1 + 0.012 * kick)
        hue = s / self.S
        fade = (0.35 + 0.65 * dec) * (0.85 + 0.15 * level)
        return xy, hue, fade


class IteratedMap:
    """A 2D chaotic map applied once per frame: the cloud settles on the attractor and shimmers.
    Its four parameters breathe slowly, so the figure keeps changing shape.

    system: clifford or dejong.
    """

    PRESETS = {
        "clifford": [(-1.4, 1.6, 1.0, 0.7), (1.7, 1.7, 0.6, 1.2), (-1.7, 1.3, -0.1, -1.2), (1.5, -1.8, 1.6, 0.9)],
        "dejong": [(1.4, -2.3, 2.4, -2.1), (2.01, -2.53, 1.61, -0.33), (-2.7, -0.09, -0.86, -2.2), (-2.0, -2.0, -1.2, 2.0)],
    }

    def __init__(self, n: int, rng: np.random.Generator, vis) -> None:
        self.kind = vis.system if vis.system in self.PRESETS else "clifford"
        presets = self.PRESETS[self.kind]
        self.base = np.array(presets[int(rng.integers(len(presets)))])
        self.wob = rng.uniform(0.04, 0.09, 4)
        self.rate = rng.uniform(0.015, 0.035, 4)
        self.p = rng.uniform(-0.5, 0.5, (n, 2))
        self.prev = self.p.copy()
        for _ in range(30):
            self.p = self._map(self.p, self.base)

    def _map(self, p: np.ndarray, k: np.ndarray) -> np.ndarray:
        a, b, c, d = k
        x, y = p[:, 0], p[:, 1]
        if self.kind == "clifford":
            return np.column_stack([np.sin(a * y) + c * np.cos(a * x), np.sin(b * x) + d * np.cos(b * y)])
        return np.column_stack([np.sin(a * y) - np.cos(b * x), np.sin(c * x) - np.cos(d * y)])

    def step(self, t: float, level: float, kick: float, vis) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        k = self.base + self.wob * np.sin(2 * np.pi * self.rate * t * vis.speed)
        self.prev, self.p = self.p, self._map(self.p, k)
        ext = (1 + abs(k[2]), 1 + abs(k[3])) if self.kind == "clifford" else (2.0, 2.0)
        spin = 0.006 * t
        x, y = self.p[:, 0] / ext[0], self.p[:, 1] / ext[1]
        xy = np.column_stack([x * np.cos(spin) - y * np.sin(spin), x * np.sin(spin) + y * np.cos(spin)])
        xy *= np.array([0.95 * ASPECT * 0.62, 0.92]) * (1 + 0.012 * kick)
        jump = np.linalg.norm(self.p - self.prev, axis=1)
        hue = np.clip(jump / 2.5, 0, 1)
        fade = np.full(len(xy), 0.55 + 0.25 * level)
        return xy, hue, fade


class Spirograph:
    """A hypotrochoid: a wheel rolling inside a ring with a pen at distance d; the pen arm
    breathes and the drawing rotates."""

    def __init__(self, n: int, rng: np.random.Generator, vis) -> None:
        pairs = [(5, 3), (7, 4), (8, 5), (11, 4), (9, 7), (13, 5), (10, 3)]
        R, r = pairs[int(rng.integers(len(pairs)))]
        self.R, self.r = float(R), float(r)
        self.turns = r / np.gcd(R, r)
        self.d0 = rng.uniform(0.6, 1.1) * r
        self.s = rng.uniform(0, 2 * np.pi * self.turns, n)
        self.v = rng.uniform(0.7, 1.2, n)
        self.scale = 0.85 / (self.R - self.r + self.d0 * 1.25)

    def step(self, t: float, level: float, kick: float, vis) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        L = 2 * np.pi * self.turns
        self.s = (self.s + self.v * 0.006 * vis.speed) % L
        R, r, s = self.R, self.r, self.s
        d = self.d0 * (1 + 0.25 * np.sin(t / 25))
        x = (R - r) * np.cos(s) + d * np.cos((R - r) / r * s)
        y = (R - r) * np.sin(s) - d * np.sin((R - r) / r * s)
        spin = 0.012 * t
        xy = np.column_stack([x * np.cos(spin) - y * np.sin(spin), x * np.sin(spin) + y * np.cos(spin)])
        xy *= self.scale * (1 + 0.012 * kick)
        hue = np.clip(np.hypot(x, y) / (R - r + d), 0, 1)
        fade = np.full(len(xy), 0.8 + 0.2 * level)
        return xy, hue, fade


COSMOS = {
    "orbits": Orbits,
    "galaxy": Galaxy,
    "harmonograph": Harmonograph,
    "map": IteratedMap,
    "spirograph": Spirograph,
}
