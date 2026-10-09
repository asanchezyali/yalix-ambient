"""Figure families for the gothic trap mix: the geometry of a cathedral at night.

- apollonian: a rose window of mutually tangent circles. Start from four circles that touch,
  and keep filling every curved triangle with the one circle that touches all three sides
  (Descartes' theorem, in complex numbers so it gives the centre too). Particles run around the
  circles; each hit lights one generation of them.
- nave: an endless corridor of pointed arches (each side an arc whose centre sits on the other
  springing point, the equilateral gothic arch) on columns, seen in perspective while the
  camera walks down the nave; each hit sends a glow through the bays.
- flock: bats. Each one steers towards a leader flying a slow Lissajous path, keeps its speed,
  and is pushed by a turbulent wind; each hit scatters the flock and it closes again.
- glass: a stained-glass lancet. Seeds drift slowly; the plane splits into their Voronoi cells
  (the region closest to each seed). Particles near a border are the lead lines, the rest glow
  as coloured glass; each hit lights one pane.

Same contract as gl_forms: screen-normalized positions (y up), a hue and a brightness.
"""

from __future__ import annotations

import numpy as np

from yalix_ambient.gl_cymatics import _Strikes
from yalix_ambient.gl_render import H, W

ASPECT = W / H


def _gasket(kind: str, r_min: float = 0.012) -> list[tuple[float, complex, int]]:
    """Circles (curvature, centre, generation) of an Apollonian gasket inside the unit circle."""
    if kind == "triple":
        k = 1 + 2 / np.sqrt(3)
        d = 1 - 1 / k
        start = [(-1.0, 0j)] + [(k, d * np.exp(1j * (np.pi / 2 + 2 * np.pi * i / 3))) for i in range(3)]
    else:  # classic: two halves and the two thirds between them
        start = [(-1.0, 0j), (2.0, 0.5 + 0j), (2.0, -0.5 + 0j), (3.0, 2j / 3)]
    circles = [(k, z, 0) for k, z in start]
    queue = []
    idx = [0, 1, 2, 3]
    for d in idx:  # each circle of the first four, reflected across the other three
        a, b, c = (start[i] for i in idx if i != d)
        queue.append((a, b, c, start[d], 1))
    seen = {(round(k, 6), round(z.real, 6), round(z.imag, 6)) for k, z in start}
    while queue:
        (k1, z1), (k2, z2), (k3, z3), (k0, z0), g = queue.pop()
        k4 = 2 * (k1 + k2 + k3) - k0
        if k4 <= 0 or 1 / k4 < r_min:
            continue
        z4 = (2 * (k1 * z1 + k2 * z2 + k3 * z3) - k0 * z0) / k4
        key = (round(k4, 6), round(z4.real, 6), round(z4.imag, 6))
        if key in seen:
            continue
        seen.add(key)
        circles.append((k4, z4, g))
        n = (k4, z4)
        queue += [((k1, z1), (k2, z2), n, (k3, z3), g + 1), ((k1, z1), (k3, z3), n, (k2, z2), g + 1),
                  ((k2, z2), (k3, z3), n, (k1, z1), g + 1)]  # fmt: skip
    return circles


class Apollonian:
    def __init__(self, n: int, rng: np.random.Generator, vis) -> None:
        self.rng = rng
        circles = _gasket(vis.system)
        r = np.array([1 / abs(k) for k, _, _ in circles])
        w = r**0.6
        self.which = rng.choice(len(circles), n, p=w / w.sum())
        self.r = r[self.which] * 0.92
        z = np.array([c[1] for c in circles])[self.which] * 0.92
        self.c = np.column_stack([z.real, z.imag])
        self.gen = np.array([c[2] for c in circles])[self.which]
        self.th = rng.uniform(0, 2 * np.pi, n)
        self.dir = np.where(self.gen % 2, 1, -1)
        self.strike = _Strikes()
        self.lit, self.glow = 0, 0.0

    def step(self, t: float, level: float, kick: float, vis):
        if self.strike(kick):
            self.lit = int(self.rng.integers(0, 5))
            self.glow = 1.0
        self.th += self.dir * 0.006 / np.sqrt(self.r) * vis.speed
        rot = 0.01 * t
        p = self.c + self.r[:, None] * np.column_stack([np.cos(self.th), np.sin(self.th)])
        c, s = np.cos(rot), np.sin(rot)
        xy = np.column_stack([p[:, 0] * c - p[:, 1] * s, p[:, 0] * s + p[:, 1] * c])
        hue = np.clip(0.12 * self.gen + 0.1 * np.sin(self.th * 3 + t * 0.2), 0, 1)
        fade = 0.45 + 0.25 * level + 0.6 * self.glow * (self.gen == self.lit)
        self.glow *= 0.94
        return xy * (1 + 0.008 * kick), hue, np.clip(fade, 0, 1)


class Nave:
    """Bays of pointed arches every 1 unit along z; the camera walks forward forever."""

    BAYS = 12

    def __init__(self, n: int, rng: np.random.Generator, vis) -> None:
        self.rng = rng
        self.bay = rng.integers(0, self.BAYS, n)
        self.part = rng.choice(4, n, p=[0.45, 0.25, 0.2, 0.1])  # arch, columns, side arcade, floor
        self.u = rng.uniform(0, 1, n)
        self.side = rng.choice([-1, 1], n)
        self.strike = _Strikes()
        self.pulse = -10.0
        self.z_cam = 0.0
        self.width = {"high": 0.9, "wide": 1.3}.get(vis.system, 1.05)

    def _arch(self, u: np.ndarray, w: float, spring: float) -> np.ndarray:
        """An equilateral pointed arch of span w springing at height `spring`: u in [0, 1]."""
        left = u < 0.5
        a = np.where(left, u * 2, (1 - u) * 2) * np.pi / 3  # 0 at the springing, 60 deg at the apex
        x = np.where(left, w / 2 - w * np.cos(a), -(w / 2 - w * np.cos(a)))
        y = spring + w * np.sin(a)
        return np.column_stack([x, y])

    def step(self, t: float, level: float, kick: float, vis):
        if self.strike(kick):
            self.pulse = t
        self.z_cam += 0.006 * vis.speed
        w = self.width
        z = (self.bay - self.z_cam) % self.BAYS + 0.6
        pts = np.zeros((len(z), 2))
        a = self.part == 0
        pts[a] = self._arch(self.u[a], w * 2, 0.4)
        c = self.part == 1
        pts[c] = np.column_stack([self.side[c] * w, -1.0 + 1.4 * self.u[c]])
        s = self.part == 2  # small arches between the columns along the walls
        sub = self._arch(self.u[s], 0.5, -0.2)
        pts[s] = np.column_stack([self.side[s] * w, sub[:, 1]])
        zs = z.copy()
        zs[s] += sub[:, 0]  # the side arcade runs along z
        f = self.part == 3
        pts[f] = np.column_stack([self.side[f] * w * self.u[f], np.full(f.sum(), -1.0)])
        zs[f] += self.u[f] - 0.5
        persp = 1.2 / zs
        xy = np.column_stack([pts[:, 0] * persp, (pts[:, 1] + 0.15) * persp - 0.05])
        fog = np.clip(1.2 - zs / self.BAYS, 0, 1) ** 1.5
        wave = np.exp(-(((t - self.pulse) * 6 - zs) ** 2) / 0.8)  # the glow runs down the nave
        hue = np.clip(0.2 + 0.05 * zs + 0.6 * wave, 0, 1)
        fade = np.clip(fog * (0.6 + 0.3 * level) + 0.8 * wave * fog, 0, 1)
        fade *= np.clip(zs - 0.65, 0, 1) * 4  # nothing too close to the eye
        return xy, hue, np.clip(fade, 0, 1)


class Flock:
    def __init__(self, n: int, rng: np.random.Generator, vis) -> None:
        self.rng, self.n = rng, n
        self.k = {"swarm": 1, "colony": 5}.get(vis.system, 3)
        self.lead = rng.integers(0, self.k, n)
        self.p = rng.normal(0, 0.3, (n, 2))
        self.v = rng.normal(0, 0.01, (n, 2))
        self.off = rng.normal(0, 0.09, (n, 2)) * rng.uniform(0.3, 1.8, (n, 1))  # dense core, loose edge
        self.ph = rng.uniform(0, 2 * np.pi, (self.k, 3))
        self.strike = _Strikes()
        self.flap = rng.uniform(0, 2 * np.pi, n)

    def _leaders(self, t: float) -> np.ndarray:
        a = self.ph
        return np.column_stack([1.2 * np.sin(0.11 * t + a[:, 0]) * np.cos(0.05 * t + a[:, 2]),
                                0.6 * np.sin(0.17 * t + a[:, 1])])  # fmt: skip

    def step(self, t: float, level: float, kick: float, vis):
        s = self.strike(kick)
        L = self._leaders(t)
        target = L[self.lead] + self.off * (1 + 0.5 * np.sin(t * 0.3))
        acc = (target - self.p) * 0.008
        wind = np.column_stack([np.sin(self.p[:, 1] * 4 + t * 0.7), np.cos(self.p[:, 0] * 3 - t * 0.5)]) * 0.0006
        self.v = (self.v + acc + wind) * 0.96
        if s:  # scatter away from each leader
            d = self.p - L[self.lead]
            self.v += d / (np.linalg.norm(d, axis=1, keepdims=True) + 0.05) * 0.03 * s
        sp = np.linalg.norm(self.v, axis=1, keepdims=True) + 1e-9
        self.v = np.where(sp > 0.04, self.v / sp * 0.04, self.v)
        self.p += self.v * vis.speed
        self.flap += 0.5
        wing = np.abs(np.sin(self.flap))  # wings catch the light on every beat of the flap
        hue = np.clip(sp[:, 0] / 0.04, 0, 1) * 0.8
        fade = (0.35 + 0.65 * wing) * (0.7 + 0.3 * level)
        return self.p.copy(), hue, fade


class Glass:
    """A lancet window of Voronoi panes with lead between them."""

    def __init__(self, n: int, rng: np.random.Generator, vis) -> None:
        self.rng = rng
        m = {"rose": 60, "small": 36}.get(vis.system, 18)  # panes per window
        wins = (0.0,) if vis.system == "rose" else (-1.15, 0.0, 1.15)  # one tall window or a triple lancet
        seeds, pts, self.box = [], [], []
        for ox in wins:
            sd = self._inside(rng.uniform(-1, 1, (m * 3, 2)))[:m]
            seeds.append(sd + [ox, 0])
            pts.append(self._inside(rng.uniform(-1, 1, (n * 3 // len(wins), 2)))[: n // len(wins)] + [ox, 0])
            self.box += [ox] * len(sd)
        self.seeds, self.pts, self.box = np.vstack(seeds), np.vstack(pts), np.array(self.box)
        self.vel = rng.normal(0, 0.0006, self.seeds.shape)
        self.hue = rng.uniform(0, 1, len(self.seeds))
        self.strike = _Strikes()
        self.lit, self.glow = 0, 0.0

    @staticmethod
    def _inside(p: np.ndarray) -> np.ndarray:
        """Keep points inside a lancet: a rectangle topped by an equilateral pointed arch."""
        w, spring = 0.5, 0.1  # apex at spring + w * sqrt(3) = 0.97
        x, y = p[:, 0] * w, p[:, 1]
        body = (np.abs(x) <= w) & (y <= spring) & (y >= -0.95)
        arch = ((x + w) ** 2 + (y - spring) ** 2 <= (2 * w) ** 2) & ((x - w) ** 2 + (y - spring) ** 2 <= (2 * w) ** 2)
        top = (y > spring) & arch
        return np.column_stack([x, y])[body | top]

    def step(self, t: float, level: float, kick: float, vis):
        if self.strike(kick):
            self.lit = int(self.rng.integers(0, len(self.seeds)))
            self.glow = 1.0
        self.seeds += self.vel * vis.speed
        back = np.abs(self.seeds[:, 0] - self.box) > 0.45
        self.vel[back, 0] *= -1
        back = (self.seeds[:, 1] > 0.95) | (self.seeds[:, 1] < -0.9)
        self.vel[back, 1] *= -1
        d = np.linalg.norm(self.pts[:, None, :] - self.seeds[None, :, :], axis=2)
        two = np.partition(d, 1, axis=1)[:, :2]
        near = np.argmin(d, axis=1)
        lead = (two[:, 1] - two[:, 0]) < 0.01  # the tracery glows, the panes behind it are dimmer
        hue = np.where(lead, 0.95, self.hue[near] * 0.85)
        fade = np.where(lead, 1.0, 0.6 + 0.25 * level)
        fade = fade + 0.6 * self.glow * ((near == self.lit) & ~lead)
        self.glow *= 0.95
        jitter = self.rng.normal(0, 0.002, self.pts.shape)
        return (self.pts + jitter) * 0.95, hue, np.clip(fade, 0, 1)


GOTHIC = {"apollonian": Apollonian, "nave": Nave, "flock": Flock, "glass": Glass}
