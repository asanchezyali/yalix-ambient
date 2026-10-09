"""2D 'machine' families for the particle renderer: code rain, a packet network and clockwork.

Each family places N particles in screen-normalized coordinates (x in [-W/H, W/H], y in
[-1, 1]) and, every frame, returns positions, a hue in [0, 1] and a brightness in [0, 1].
"""

from __future__ import annotations

import numpy as np

from yalix_ambient.gl_render import H, W

ASPECT = W / H


class CodeRain:
    """Glyphs fixed on a grid; a bright head runs down each column and leaves a fading tail."""

    def __init__(self, n: int, rng: np.random.Generator, vis) -> None:
        self.rng = rng
        cols, rows = 120, 54
        self.cols, self.rows = cols, rows
        cells = rng.choice(cols * rows, size=min(n, cols * rows), replace=False)
        self.col, row = cells % cols, cells // cols
        self.x = ((self.col + 0.5) / cols * 2 - 1) * ASPECT * 0.98
        self.y = (row + 0.5) / rows * 2 - 1
        self.head = rng.uniform(-1.2, 1.4, cols)
        self.speed = rng.uniform(0.35, 1.25, cols)
        self.tail = rng.uniform(0.25, 0.9, cols)
        self.flick = rng.uniform(0, 2 * np.pi, len(cells))

    def step(self, t: float, level: float, kick: float, vis) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.head -= self.speed * 0.0075 * vis.speed * (1 + 0.5 * level)
        gone = self.head < -1.0 - self.tail
        self.head[gone] = self.rng.uniform(1.05, 1.6, gone.sum())
        d = self.y - self.head[self.col]  # > 0: above the head, inside the tail
        fade = np.where(d >= 0, np.exp(-d / self.tail[self.col]), 0.0)
        fade *= 0.75 + 0.25 * np.sin(self.flick + t * 9)  # glyphs flicker as they change
        hue = np.clip(1 - fade, 0, 1)  # the head sits at one end of the palette, the tail at the other
        return np.column_stack([self.x, self.y]), hue, np.clip(fade * (1 + 0.3 * kick), 0, 1)


class PacketGraph:
    """Nodes on a sparse graph; packets travel edge to edge and pick a random next hop."""

    def __init__(self, n: int, rng: np.random.Generator, vis) -> None:
        self.rng = rng
        nodes: list[np.ndarray] = []
        while len(nodes) < 42:
            p = rng.uniform(-0.9, 0.9, 2) * np.array([ASPECT * 0.95, 1.0])
            if all(np.linalg.norm(p - q) > 0.22 for q in nodes):
                nodes.append(p)
        self.nodes = np.array(nodes)
        dist = np.linalg.norm(self.nodes[:, None] - self.nodes[None], axis=2)
        self.nbrs = np.argsort(dist, axis=1)[:, 1:4]
        self.n_node = min(len(nodes) * 30, n // 3)
        self.node_of = rng.integers(0, len(nodes), self.n_node)
        self.jit = rng.standard_normal((self.n_node, 2)) * 0.012
        m = n - self.n_node
        self.frm = rng.integers(0, len(nodes), m)
        self.to = self.nbrs[self.frm, rng.integers(0, 3, m)]
        self.s = rng.uniform(0, 1, m)
        self.v = rng.uniform(0.4, 1.3, m)

    def step(self, t: float, level: float, kick: float, vis) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.s += self.v * 0.007 * vis.speed * (1 + 0.8 * kick)
        done = self.s >= 1
        if done.any():
            self.frm[done] = self.to[done]
            self.to[done] = self.nbrs[self.frm[done], self.rng.integers(0, 3, done.sum())]
            self.s[done] -= 1
        pk = self.nodes[self.frm] * (1 - self.s[:, None]) + self.nodes[self.to] * self.s[:, None]
        nd = self.nodes[self.node_of] + self.jit * (1 + 1.5 * kick)
        xy = np.vstack([nd, pk])
        hue = np.concatenate([np.full(self.n_node, 0.95), (self.v - 0.4) / 0.9 * 0.7])
        fade = np.concatenate([np.full(self.n_node, 0.55 + 0.45 * kick), np.full(len(pk), 0.9)])
        return xy, hue, fade


class Clockwork:
    """A train of meshed gears that advance like an escapement: a tick, then stillness."""

    def __init__(self, n: int, rng: np.random.Generator, vis) -> None:
        gears = []  # (centre, radius, teeth, angular speed)
        c, r, w = np.array([-ASPECT * 0.45, -0.1]), 0.42, 0.35
        gears.append((c, r, 24, w))
        tries = 0
        while len(gears) < 7 and tries < 400:
            tries += 1
            pc, pr, _, pw = gears[int(rng.integers(len(gears)))]
            nr = rng.uniform(0.12, 0.38)
            ang = rng.uniform(0, 2 * np.pi)
            nc = pc + (pr + nr + 0.01) * np.array([np.cos(ang), np.sin(ang)])
            inside = abs(nc[0]) + nr < ASPECT * 0.97 and abs(nc[1]) + nr < 0.97
            clear = all(np.linalg.norm(nc - g[0]) > g[1] + nr + 0.005 for g in gears)
            if inside and clear:
                gears.append((nc, nr, max(int(24 * nr / 0.42), 8), -pw * pr / nr))  # meshing ratio
        lo = np.min([g[0] - g[1] for g in gears], axis=0)
        hi = np.max([g[0] + g[1] for g in gears], axis=0)
        shift = -(lo + hi) / 2  # centre the whole train on screen
        self.gears = [(g[0] + shift, g[1], g[2], g[3]) for g in gears]
        gears = self.gears
        circ = np.array([g[1] for g in gears])
        share = circ / circ.sum()
        self.g = rng.choice(len(gears), size=n, p=share)
        self.kind = rng.choice(3, size=n, p=[0.6, 0.15, 0.25])  # 0 rim with teeth, 1 hub, 2 spokes
        self.th0 = rng.uniform(0, 2 * np.pi, n)
        self.spoke = rng.integers(0, 6, n)
        self.u = rng.uniform(0.2, 0.85, n)

    def step(self, t: float, level: float, kick: float, vis) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tick = np.floor(t * 2) + np.clip((t * 2 % 1) * 5, 0, 1)  # two ticks a second
        centres = np.array([g[0] for g in self.gears])[self.g]
        R = np.array([g[1] for g in self.gears])[self.g]
        teeth = np.array([g[2] for g in self.gears])[self.g]
        w = np.array([g[3] for g in self.gears])[self.g]
        rot = w * tick * 0.5 * vis.speed
        tooth = np.clip(np.sin(teeth * self.th0) * 3, -1, 1)
        r = np.select([self.kind == 0, self.kind == 1], [R + 0.06 * R * tooth, R * 0.18], R * self.u)
        th = np.where(self.kind == 2, self.spoke * np.pi / 3, self.th0) + rot
        xy = centres + np.column_stack([np.cos(th), np.sin(th)]) * r[:, None] * (1 + 0.01 * kick)
        hue = (self.g / max(len(self.gears) - 1, 1)) * 0.8 + 0.1 * (self.kind == 1)
        fade = np.where(self.kind == 0, 0.9, 0.6) * (0.85 + 0.15 * level)
        return xy, hue, fade


MACHINES = {"rain": CodeRain, "graph": PacketGraph, "clockwork": Clockwork}

from yalix_ambient.gl_cosmos import COSMOS  # noqa: E402

MACHINES.update(COSMOS)

from yalix_ambient.gl_forms import FORMS  # noqa: E402

MACHINES.update(FORMS)

from yalix_ambient.gl_cymatics import CYMATICS  # noqa: E402

MACHINES.update(CYMATICS)

from yalix_ambient.gl_industrial import INDUSTRIAL  # noqa: E402

MACHINES.update(INDUSTRIAL)
