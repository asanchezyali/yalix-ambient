"""Figure families for the handpan mix: the shapes a struck shell, a drum skin and a pond make.

- membrane: sand on a circular drum skin. The skin vibrates in a Bessel mode J_m(k r) cos(m th),
  the grains slide to the nodal lines, and every handpan strike shakes them loose a little;
  after a number of strikes the skin jumps to the next mode.
- ripples: the surface of water seen from above. Each strike drops a stone, and the rings
  spread, cross and interfere; particles light up on the crests (like caustics).
- mandala: a slow flow folded into k mirrored wedges, a kaleidoscope that breathes with the notes.
- turing: a Gray-Scott reaction-diffusion (two chemicals, one feeding on the other) growing
  coral, spots, worms or mazes; particles gather on the pattern and each strike seeds new growth.

Same contract as gl_forms: screen-normalized positions, a hue and a brightness per particle.
The strikes reach these scenes through the `kick` curve, which the handpan engine fills with
its notes; a jump in it is one strike.
"""

from __future__ import annotations

import numpy as np
from scipy.special import jn_zeros, jv

from yalix_ambient.gl_render import H, W

ASPECT = W / H


class _Strikes:
    """Turns the smooth `kick` curve back into discrete strikes (its rising edges)."""

    def __init__(self) -> None:
        self.prev = 0.0

    def __call__(self, kick: float) -> float:
        jump = kick - self.prev
        self.prev = kick
        return jump if jump > 0.25 else 0.0


class Membrane:
    MODES = {
        "drum": [(1, 2), (2, 2), (3, 1), (0, 3), (2, 3), (4, 2), (1, 3), (5, 1), (3, 3), (6, 2)],
        "rings": [(0, 2), (0, 3), (0, 4), (1, 3), (0, 5), (2, 4), (0, 6), (1, 5)],
        "petals": [(3, 2), (4, 2), (5, 2), (6, 2), (7, 2), (8, 2), (5, 3), (6, 3)],
        "deep": [(2, 4), (3, 4), (4, 3), (5, 4), (6, 3), (7, 3), (8, 3), (9, 2)],
    }

    def __init__(self, n: int, rng: np.random.Generator, vis) -> None:
        self.rng = rng
        modes = self.MODES.get(vis.system, self.MODES["drum"])
        self.modes = [modes[i] for i in rng.permutation(len(modes))]
        r = np.sqrt(rng.uniform(0, 1, n)) * 0.95
        th = rng.uniform(0, 2 * np.pi, n)
        self.p = np.column_stack([r * np.cos(th), r * np.sin(th)])
        self.strike = _Strikes()
        self.count, self.k = 0, 0
        self.per_mode = 28
        self.shake = 0.0

    def _f(self, p: np.ndarray, t: float) -> np.ndarray:
        m, nn = self.modes[self.k % len(self.modes)]
        r = np.linalg.norm(p, axis=1) / 0.95
        th = np.arctan2(p[:, 1], p[:, 0]) - 0.02 * t
        return jv(m, jn_zeros(m, nn)[-1] * r) * np.cos(m * th)

    def step(self, t: float, level: float, kick: float, vis):
        s = self.strike(kick)
        if s:
            self.count += 1
            self.shake = max(self.shake, 0.012 * s)
            if self.count % self.per_mode == 0:
                self.k += 1
                self.shake = 0.06
        f = self._f(self.p, t)
        e = 1e-3
        gx = (self._f(self.p + [e, 0], t) - f) / e
        gy = (self._f(self.p + [0, e], t) - f) / e
        self.p -= (f[:, None] * np.column_stack([gx, gy])) * 0.0035 * vis.speed
        self.p += self.rng.standard_normal(self.p.shape) * (0.0015 + self.shake)
        self.shake *= 0.9
        r = np.linalg.norm(self.p, axis=1, keepdims=True)
        self.p = np.where(r > 0.95, self.p / r * 0.95, self.p)
        hue = np.clip(0.6 * r[:, 0] + 1.5 * np.abs(f), 0, 1)
        fade = np.clip(1.0 - np.abs(f) * 2.2, 0.12, 1.0) * (0.75 + 0.25 * level)
        return self.p * (1 + 0.01 * kick), hue, fade


class Ripples:
    """Rings on water: every strike drops a stone and its crests spread out and fade. Particles
    are dealt to the eight newest stones (four crests each); the rest glint on the calm surface."""

    SLOTS = 9  # eight stones and the calm water

    def __init__(self, n: int, rng: np.random.Generator, vis) -> None:
        self.rng, self.system, self.n = rng, vis.system, n
        self.src: list[tuple[float, float, float, float] | None] = [None] * (self.SLOTS - 1)  # x, y, t0, strength
        self.drops = 0
        self.strike = _Strikes()
        self.next_drop = 0.0
        self.c, self.lam, self.tau = 0.16, 0.06, 7.0  # speed, crest spacing, fade (s)
        idx = np.arange(n)
        self.slot = idx % self.SLOTS
        self.ring = (idx // self.SLOTS) % 5
        self.th = rng.uniform(0, 2 * np.pi, n)
        self.calm = rng.uniform(-1, 1, (n, 2)) * np.array([ASPECT, 1.0])
        self.fixed = [(-0.5, 0.0), (0.5, 0.0)] if self.system == "two" else [(0.0, 0.0)] if self.system == "spring" else []
        self.last_fixed = -10.0

    def _drop(self, t: float, x: float, y: float, a: float) -> None:
        self.src[self.drops % len(self.src)] = (x, y, t, a)  # each stone keeps its slot: no jumps
        self.drops += 1

    def step(self, t: float, level: float, kick: float, vis):
        s = self.strike(kick)
        if s:
            self._drop(t, self.rng.uniform(-ASPECT * 0.75, ASPECT * 0.75), self.rng.uniform(-0.75, 0.75), 0.6 + 0.4 * s)
        if self.system == "rain" and t >= self.next_drop:
            self.next_drop = t + self.rng.exponential(0.9)
            self._drop(t, self.rng.uniform(-ASPECT, ASPECT), self.rng.uniform(-1, 1), 0.4)
        if not self.fixed and self.system != "rain" and t >= self.next_drop:  # a drop now and then, even in silence
            self.next_drop = t + self.rng.uniform(2.5, 4.5)
            self._drop(t, self.rng.uniform(-ASPECT * 0.7, ASPECT * 0.7), self.rng.uniform(-0.7, 0.7), 0.5)
        if self.fixed and t - self.last_fixed > 1.6:  # steady sources pulse on their own
            self.last_fixed = t
            for x, y in self.fixed:
                self._drop(t, x, y, 0.55)
        xy = self.calm + 0.004 * np.column_stack([np.sin(t * 0.3 + self.th * 5), np.cos(t * 0.25 + self.th * 3)])
        fade = 0.16 + 0.25 * np.sin(t * 1.2 + self.th * 11) ** 12  # glints
        hue = np.full(self.n, 0.15)
        fade = np.asarray(fade, dtype=float).copy()
        for k, src in enumerate(self.src):
            if src is None:
                continue
            x, y, t0, a = src
            m = self.slot == k
            age = t - t0
            r = self.c * age - self.ring[m] * self.lam + 0.004 * np.sin(self.th[m] * 7 + age * 3)
            ok = r > 0
            xy[m] = np.column_stack([x + r * np.cos(self.th[m]), y + r * np.sin(self.th[m])])
            fade[m] = np.where(ok, a * np.exp(-age / self.tau) * (1 - 0.16 * self.ring[m]), 0.0)
            hue[m] = np.clip(0.2 + 0.18 * self.ring[m] + age / 10, 0, 1)
        return xy, hue, np.clip(1.5 * fade * (0.8 + 0.2 * level), 0, 1)


class Mandala:
    def __init__(self, n: int, rng: np.random.Generator, vis) -> None:
        self.k = int(vis.system) if vis.system.isdigit() else 8
        self.rng = rng
        self.m = n // (2 * self.k)
        self.p = self._spawn(self.m)
        self.age = rng.uniform(0, 1, self.m)
        self.waves = rng.uniform(-1, 1, (5, 4)) * [5, 5, 0.25, 1]  # kx, ky, omega, amplitude
        self.strike = _Strikes()
        self.pulse = 0.0

    def _spawn(self, k: int) -> np.ndarray:
        r = np.sqrt(self.rng.uniform(0.0, 1.0, k)) * 0.9
        th = self.rng.uniform(0, np.pi / self.k, k)
        return np.column_stack([r * np.cos(th), r * np.sin(th)])

    def step(self, t: float, level: float, kick: float, vis):
        if self.strike(kick):
            self.pulse = 1.0
        x, y = self.p[:, 0], self.p[:, 1]
        vx = np.zeros_like(x)
        vy = np.zeros_like(y)
        for kx, ky, om, a in self.waves:  # velocity from a stream function: no sources or sinks
            c = a * np.cos(kx * x + ky * y + om * t)
            vx += c * ky
            vy -= c * kx
        self.p += np.column_stack([vx, vy]) * 0.0022 * vis.speed
        r = np.linalg.norm(self.p, axis=1)
        th = np.arctan2(self.p[:, 1], self.p[:, 0]) % (2 * np.pi / self.k)
        th = np.where(th > np.pi / self.k, 2 * np.pi / self.k - th, th)  # fold back into the wedge
        self.p = np.column_stack([r * np.cos(th), r * np.sin(th)])
        self.age += 1 / (30 * 12)
        out = (r > 0.95) | (self.age > 1)
        if out.any():
            self.p[out] = self._spawn(int(out.sum()))
            self.age[out] = 0
        r = np.linalg.norm(self.p, axis=1)
        th = np.arctan2(self.p[:, 1], self.p[:, 0])
        rr = r * (1 + 0.035 * self.pulse * np.sin(r * 18 - t * 3))
        self.pulse *= 0.93
        rot = 0.015 * t
        xs, ys = [], []
        for j in range(self.k):
            for sgn in (1, -1):
                a = sgn * th + 2 * np.pi * j / self.k + rot
                xs.append(rr * np.cos(a))
                ys.append(rr * np.sin(a))
        xy = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
        hue = np.tile((r * 1.3 + 0.1 * np.sin(t / 7)) % 1.0, 2 * self.k)
        f = np.minimum(self.age * 6, 1) * np.minimum((1 - self.age) * 6, 1)
        fade = np.tile(f, 2 * self.k) * (0.8 + 0.2 * level)
        return xy, hue, fade


class Turing:
    PARAMS = {  # Gray-Scott feed and kill rates
        "coral": (0.0545, 0.062),
        "spots": (0.030, 0.062),
        "mitosis": (0.0367, 0.0649),
        "maze": (0.029, 0.057),
        "worms": (0.078, 0.061),
    }

    def __init__(self, n: int, rng: np.random.Generator, vis) -> None:
        self.rng = rng
        self.F, self.K = self.PARAMS.get(vis.system, self.PARAMS["coral"])
        self.gw, self.gh = 192, 108
        self.u = np.ones((self.gh, self.gw))
        self.v = np.zeros((self.gh, self.gw))
        for _ in range(int(rng.integers(6, 12))):
            self._seed()
        for _ in range(1800):  # grow a pattern before the scene starts
            self._react()
        self.p = rng.uniform(-1, 1, (n, 2)) * np.array([ASPECT, 1.0])
        self.strike = _Strikes()

    def _seed(self, x: int | None = None, y: int | None = None) -> None:
        x = int(self.rng.integers(10, self.gw - 10)) if x is None else x
        y = int(self.rng.integers(10, self.gh - 10)) if y is None else y
        self.u[y - 3 : y + 3, x - 3 : x + 3] = 0.5
        self.v[y - 3 : y + 3, x - 3 : x + 3] = 0.25 + self.rng.uniform(0, 0.05, (6, 6))

    def _react(self) -> None:
        u, v = self.u, self.v

        def lap(z):
            return np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1) - 4 * z

        uvv = u * v * v
        self.u = u + 0.16 * lap(u) - uvv + self.F * (1 - u)
        self.v = v + 0.08 * lap(v) + uvv - (self.F + self.K) * v

    def _cell(self, p: np.ndarray):
        gx = np.clip(((p[:, 0] / ASPECT + 1) / 2 * (self.gw - 1)).astype(int), 1, self.gw - 2)
        gy = np.clip(((p[:, 1] + 1) / 2 * (self.gh - 1)).astype(int), 1, self.gh - 2)
        return gx, gy

    def step(self, t: float, level: float, kick: float, vis):
        if self.strike(kick):
            self._seed()
        for _ in range(int(4 * vis.speed) or 1):
            self._react()
        gx, gy = self._cell(self.p)
        v = self.v
        val = v[gy, gx]
        grad = np.column_stack([v[gy, gx + 1] - v[gy, gx - 1], v[gy + 1, gx] - v[gy - 1, gx]])
        self.p += grad * 0.05 + self.rng.standard_normal(self.p.shape) * 0.0025
        lost = (val < 0.08) & (self.rng.random(len(val)) < 0.03)  # stragglers jump onto the pattern
        if lost.any():
            w = v.ravel() ** 2
            idx = self.rng.choice(w.size, int(lost.sum()), p=w / w.sum())
            cy, cx = np.divmod(idx, self.gw)
            self.p[lost] = np.column_stack([(cx / (self.gw - 1) * 2 - 1) * ASPECT, cy / (self.gh - 1) * 2 - 1])
        self.p = np.clip(self.p, [-ASPECT, -1], [ASPECT, 1])
        hue = np.clip(val * 2.2, 0, 1)
        fade = np.clip(val * 3.0, 0.08, 1) * (0.8 + 0.2 * level)
        return self.p * 0.98, hue, fade


CYMATICS = {"membrane": Membrane, "ripples": Ripples, "mandala": Mandala, "turing": Turing}
