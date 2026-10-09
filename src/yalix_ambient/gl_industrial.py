"""Figure families for the industrial mix: the mathematics of a factory.

- sparks: ballistic sparks from a grinder, a weld or a ladle. Each spark follows a parabola
  under gravity with air drag, bounces off the floor and cools from white to dark red; every
  metal hit in the music throws a burst.
- linkage: four-bar linkages, the oldest machine elements. A crank turns, a rocker swings, and a
  point on the coupler draws its curve (Hoekens and Chebyshev's straight-line motions among them).
  Particles light the links and stream along the curve; each hit jolts the cranks.
- lattice: a steel structure in 3D (a cubic lattice, a Warren truss, a pylon, a geodesic dome)
  turning slowly in perspective. A hit sends a wave of vibration through the members from the
  point it lands on.
- convection: molten metal in a crucible. Rayleigh-Bénard rolls (a stream function
  sin(kx) sin(y)) that wobble in time, so the flow mixes chaotically; colour is temperature,
  hot at the bottom and cold at the top. Each hit kicks the rolls.

Same contract as gl_forms: screen-normalized positions, a hue and a brightness per particle.
The hits arrive through the `kick` curve, which the industrial engine fills with its kicks and
metal strikes.
"""

from __future__ import annotations

import itertools

import numpy as np

from yalix_ambient.gl_cymatics import _Strikes
from yalix_ambient.gl_render import H, W

ASPECT = W / H


class Sparks:
    """Sparks under gravity: a fan from a grinder, a sphere from a weld, drops from a ladle."""

    def __init__(self, n: int, rng: np.random.Generator, vis) -> None:
        self.rng, self.n, self.kind = rng, n, vis.system
        self.p = np.zeros((n, 2))
        self.v = np.zeros((n, 2))
        self.age = np.full(n, 10.0)
        self.life = rng.uniform(1.2, 2.6, n)
        self.strike = _Strikes()
        self.next = 0
        self.src = np.array([0.0, -0.1])
        self.floor = 0.82

    def _emit(self, k: int, t: float, power: float) -> None:
        idx = (self.next + np.arange(k)) % self.n
        self.next = (self.next + k) % self.n
        rng = self.rng
        if self.kind == "weld":  # a point wandering along a seam, sparks in every direction
            src = np.array([0.9 * np.sin(t * 0.11), -0.15 + 0.1 * np.sin(t * 0.07)])
            a = rng.uniform(0, 2 * np.pi, k)
            s = rng.exponential(0.5, k) * power
        elif self.kind == "ladle":  # a pour from above that splashes where it lands
            src = np.array([0.35 * np.sin(t * 0.05), -0.95])
            a = rng.normal(np.pi / 2, 0.08, k)
            s = rng.uniform(0.1, 0.3, k)
        else:  # grinder: a fan thrown forward and down off the wheel
            src = np.array([-0.55 + 0.25 * np.sin(t * 0.09), -0.3 + 0.1 * np.sin(t * 0.13)])
            a = rng.normal(-0.15, 0.22, k)
            s = rng.uniform(0.6, 1.6, k) * power
        self.p[idx] = src + rng.normal(0, 0.01, (k, 2))
        self.v[idx] = np.column_stack([np.cos(a), np.sin(a)]) * s[:, None]
        self.age[idx] = 0.0
        self.life[idx] = rng.uniform(2.2, 3.8, k) if self.kind == "ladle" else rng.uniform(1.0, 2.8, k)

    def step(self, t: float, level: float, kick: float, vis):
        dt = 1 / 30 * vis.speed
        s = self.strike(kick)
        self._emit(int(40 + 110 * level), t, 0.7 + 0.5 * level)  # the steady stream
        if s:
            self._emit(int(900 * s), t, 1.0 + s)  # a hit: a burst
        self.v[:, 1] += 1.6 * dt  # gravity (screen y grows downwards)
        self.v *= 1 - 0.9 * dt  # air drag
        self.p += self.v * dt
        hit = (self.p[:, 1] > self.floor) & (self.v[:, 1] > 0)
        self.v[hit, 1] *= -self.rng.uniform(0.15, 0.45, hit.sum())
        self.v[hit, 0] = self.v[hit, 0] * 0.6 + self.rng.normal(0, 1.2, hit.sum()) * np.abs(self.v[hit, 1])  # splash
        self.p[hit, 1] = self.floor
        self.age += dt
        u = np.clip(self.age / self.life, 0, 1)
        fade = np.clip(1.4 * (1 - u), 0, 1) * (self.age < self.life)
        return self.p * [1, -1], u * 0.9, fade  # the renderer's y points up


def _four_bar(theta: np.ndarray, a: float, b: float, c: float, d: float, px: float, py: float) -> tuple:
    """Crank a from (0,0), ground d along x, coupler b, rocker c from (d,0). Returns B, C, P."""
    B = np.column_stack([a * np.cos(theta), a * np.sin(theta)])
    D = np.array([d, 0.0])
    BD = D - B
    r = np.linalg.norm(BD, axis=1)
    x = (b * b - c * c + r * r) / (2 * r)
    h = np.sqrt(np.clip(b * b - x * x, 0, None))
    u = BD / r[:, None]
    C = B + u * x[:, None] + np.column_stack([-u[:, 1], u[:, 0]]) * h[:, None]
    e = (C - B) / b
    P = B + e * px + np.column_stack([-e[:, 1], e[:, 0]]) * py
    return B, C, P


class Linkage:
    PRESETS = {  # crank, coupler, rocker, ground, coupler point (along, across), in crank units
        "hoekens": (1.0, 2.5, 2.5, 2.0, 5.0, 0.0),
        "chebyshev": (1.0, 2.5, 2.5, 2.0, 1.25, 1.2),
        "watt": (1.0, 2.2, 2.0, 2.6, 1.1, -0.9),
        "egg": (1.0, 3.0, 2.2, 2.8, 1.5, 1.6),
    }

    def __init__(self, n: int, rng: np.random.Generator, vis) -> None:
        self.rng, self.n = rng, n
        names = list(self.PRESETS)
        if vis.system in self.PRESETS:
            chosen = [vis.system] * 3
        else:
            chosen = [names[i] for i in rng.permutation(len(names))[:3]]
        self.mechs = []
        for j, name in enumerate(chosen):
            a, b, c, d, px, py = self.PRESETS[name]
            k = rng.uniform(0.85, 1.15)
            b, px, py = b * k, px * k, py * k
            B, C, P = _four_bar(np.linspace(0, 2 * np.pi, 400), a, b, c, d, px, py)
            allp = np.vstack([B, C, P, [[0, 0], [d, 0]]])  # the whole machine, pivots included
            ctr = (allp.max(0) + allp.min(0)) / 2
            scale = min(1.05 / np.ptp(allp[:, 0]), 1.6 / np.ptp(allp[:, 1]))
            off = np.array([(j - 1) * 1.15, 0.08 * (j % 2) - 0.04])
            self.mechs.append((a, b, c, d, px, py, scale, off, ctr, rng.uniform(0.5, 0.8) * (1 if j % 2 else -1)))
        per = n // 3
        self.per = per
        self.role = np.tile(np.arange(per) % 7 - 2, 3).clip(1)  # 1: the curve (four in seven); 2 crank; 3 coupler; 4 rocker
        self.u = rng.uniform(0, 1, 3 * per)  # place along a rod, or lag along the curve
        self.theta = np.zeros(3)
        self.strike = _Strikes()
        self.jolt = 0.0

    def step(self, t: float, level: float, kick: float, vis):
        s = self.strike(kick)
        if s:
            self.jolt = max(self.jolt, 0.8 * s)
        out, hues, fades = [], [], []
        for j, (a, b, c, d, px, py, scale, off, ctr, w) in enumerate(self.mechs):
            self.theta[j] += w * (1 + 2.5 * self.jolt) * vis.speed / 30
            sl = slice(j * self.per, (j + 1) * self.per)
            role, u = self.role[sl], self.u[sl]
            th = np.where(role < 2, self.theta[j] - u * 2 * np.pi, self.theta[j])
            B, C, P = _four_bar(th, a, b, c, d, px, py)
            A, D = np.zeros_like(B), np.tile([d, 0.0], (len(B), 1))
            pts = np.where(
                (role < 2)[:, None], P,
                np.where((role == 2)[:, None], A + (B - A) * u[:, None],
                         np.where((role == 3)[:, None], B + (C - B) * u[:, None] + (P - B) * (u * (1 - u))[:, None],
                                  D + (C - D) * u[:, None])),
            )  # fmt: skip
            xy = (pts - ctr) * scale
            xy = xy + off
            out.append(xy)
            hue = np.where(role < 2, 0.15 + 0.6 * u, 0.85)
            fade = np.where(role < 2, 0.2 + 0.8 * (1 - u) ** 3, 0.55) * (0.8 + 0.2 * level)
            hues.append(hue)
            fades.append(fade)
        self.jolt *= 0.92
        return np.concatenate(out), np.concatenate(hues), np.concatenate(fades)


def _edges(kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and member index pairs of a steel structure, centred, about 1 across."""
    if kind == "truss":  # a Warren girder: two chords and the zigzag between them
        k = 10
        bottom = [(i, 0.0, 0.0) for i in range(k + 1)]
        top = [(i + 0.5, 1.0, 0.0) for i in range(k)]
        nodes = bottom + top
        e = [(i, i + 1) for i in range(k)] + [(k + 1 + i, k + 2 + i) for i in range(k - 1)]
        e += [(i, k + 1 + i) for i in range(k)] + [(i + 1, k + 1 + i) for i in range(k)]
        back = [(x, y, 1.0) for x, y, _ in nodes]
        m = len(nodes)
        e += [(a + m, b + m) for a, b in e] + [(i, i + m) for i in range(m)]
        nodes = np.array(nodes + back) - [k / 2, 0.5, 0.5]
        return nodes / k * 2.6 * np.array([1, 2.2, 2.2]), np.array(e)
    if kind == "pylon":  # a lattice tower narrowing to the top, with crossed bracing
        lv = 7
        nodes, e = [], []
        for i in range(lv + 1):
            w = 1.0 - 0.75 * i / lv
            nodes += [(sx * w, i * 0.5, sz * w) for sx, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        for i in range(lv + 1):
            e += [(4 * i + q, 4 * i + (q + 1) % 4) for q in range(4)]
        for i in range(lv):
            for q in range(4):
                e += [(4 * i + q, 4 * (i + 1) + q), (4 * i + q, 4 * (i + 1) + (q + 1) % 4),
                      (4 * i + (q + 1) % 4, 4 * (i + 1) + q)]  # fmt: skip
        nodes = np.array(nodes) - [0, lv * 0.25, 0]
        return nodes / (lv * 0.5) * 1.7 * np.array([0.7, -1, 0.7]), np.array(e)
    if kind == "dome":  # a geodesic sphere: an icosahedron split once and pushed out
        g = (1 + 5**0.5) / 2
        v = [(-1, g, 0), (1, g, 0), (-1, -g, 0), (1, -g, 0), (0, -1, g), (0, 1, g), (0, -1, -g), (0, 1, -g),
             (g, 0, -1), (g, 0, 1), (-g, 0, -1), (-g, 0, 1)]  # fmt: skip
        v = [np.array(p) / np.linalg.norm(p) for p in v]
        faces = [f for f in itertools.combinations(range(12), 3)
                 if all(abs(np.linalg.norm(v[a] - v[b]) - 1.0515) < 0.01 for a, b in itertools.combinations(f, 2))]  # fmt: skip
        nodes = list(v)
        mid: dict[tuple[int, int], int] = {}

        def m(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key not in mid:
                p = nodes[a] + nodes[b]
                nodes.append(p / np.linalg.norm(p))
                mid[key] = len(nodes) - 1
            return mid[key]

        e = set()
        for a, b, c in faces:
            ab, bc, ca = m(a, b), m(b, c), m(c, a)
            for x, y in ((a, ab), (ab, b), (b, bc), (bc, c), (c, ca), (ca, a), (ab, bc), (bc, ca), (ca, ab)):
                e.add((min(x, y), max(x, y)))
        return np.array(nodes) * 0.8, np.array(sorted(e))
    # cube: a 3 x 3 x 3 lattice of beams
    r = (-1, 0, 1)
    nodes = list(itertools.product(r, r, r))
    idx = {p: i for i, p in enumerate(nodes)}
    e = []
    for p in nodes:
        for ax in range(3):
            q = list(p)
            q[ax] += 1
            if tuple(q) in idx:
                e.append((idx[p], idx[tuple(q)]))
    return np.array(nodes, dtype=float) * 0.55, np.array(e)


class Lattice:
    def __init__(self, n: int, rng: np.random.Generator, vis) -> None:
        self.rng = rng
        self.nodes, self.e = _edges(vis.system)
        L = np.linalg.norm(self.nodes[self.e[:, 1]] - self.nodes[self.e[:, 0]], axis=1)
        self.which = rng.choice(len(self.e), n, p=L / L.sum())  # particles spread evenly over steel
        self.u = rng.uniform(0, 1, n)
        self.strike = _Strikes()
        self.waves: list[tuple[np.ndarray, float, float]] = []  # origin, start, strength
        self.tilt = rng.uniform(0.25, 0.45)
        self.spin = rng.choice([-1, 1]) * rng.uniform(0.04, 0.07)

    def step(self, t: float, level: float, kick: float, vis):
        s = self.strike(kick)
        a, b = self.nodes[self.e[self.which, 0]], self.nodes[self.e[self.which, 1]]
        p = a + (b - a) * self.u[:, None]
        if s:
            self.waves.append((self.nodes[self.rng.integers(len(self.nodes))], t, s))
            self.waves = self.waves[-6:]
        disp = np.zeros(len(p))
        for o, t0, amp in self.waves:  # a ring of vibration running out from where the hit landed
            d = np.linalg.norm(p - o, axis=1)
            age = t - t0
            disp += amp * np.exp(-age / 1.2) * np.exp(-((d - 0.9 * age) ** 2) / 0.02) * np.sin(d * 40 - age * 30)
        n_dir = np.column_stack([np.sin(self.which * 1.7), np.cos(self.which * 2.3), np.sin(self.which * 0.9)])
        p = p + n_dir * disp[:, None] * 0.03
        ang = self.spin * t * vis.speed
        ca, sa, ct, st = np.cos(ang), np.sin(ang), np.cos(self.tilt), np.sin(self.tilt)
        x = p[:, 0] * ca + p[:, 2] * sa
        z = -p[:, 0] * sa + p[:, 2] * ca
        y = p[:, 1] * ct - z * st
        z = p[:, 1] * st + z * ct
        persp = 1 / (1 + 0.35 * z)
        xy = np.column_stack([x * persp, -y * persp]) * (1 + 0.01 * kick)
        hue = np.clip(0.1 + 0.25 * (z + 1) + 2.5 * np.abs(disp), 0, 1)
        fade = np.clip(0.45 + 0.35 * persp - 0.2 * z + 3 * np.abs(disp), 0, 1) * (0.8 + 0.2 * level)
        return xy, hue, fade


class Convection:
    """Rayleigh-Bénard rolls in a crucible: psi = A sin(k (x - B sin(w t) sin(y))) sin(y)."""

    ROLLS = {"rolls": 4, "cells": 6, "plume": 2}

    def __init__(self, n: int, rng: np.random.Generator, vis) -> None:
        self.rng, self.n = rng, n
        self.k = self.ROLLS.get(vis.system, 4)
        self.p = np.column_stack([rng.uniform(0, self.k * np.pi, n), rng.uniform(0, np.pi, n)])
        self.age = rng.uniform(0, 1, n)
        self.strike = _Strikes()
        self.push = 0.0
        self.w = rng.uniform(0.35, 0.55)

    def step(self, t: float, level: float, kick: float, vis):
        if self.strike(kick):
            self.push = 1.0
        x, y = self.p[:, 0], self.p[:, 1]
        B = 0.35 + 0.25 * self.push
        ph = x - B * np.sin(self.w * t) * np.sin(y)
        # u = dpsi/dy, v = -dpsi/dx for psi = sin(ph) sin(y)
        dph_dy = -B * np.sin(self.w * t) * np.cos(y)
        u = np.cos(ph) * dph_dy * np.sin(y) + np.sin(ph) * np.cos(y)
        v = -np.cos(ph) * np.sin(y)
        self.p += np.column_stack([u, v]) * 0.02 * vis.speed * (1 + 0.4 * level + self.push)
        self.p[:, 0] %= self.k * np.pi
        self.p[:, 1] = np.clip(self.p[:, 1], 0.01, np.pi - 0.01)
        self.push *= 0.95
        self.age += 1 / (30 * 14)
        old = self.age > 1
        if old.any():
            self.p[old] = np.column_stack([self.rng.uniform(0, self.k * np.pi, old.sum()), self.rng.uniform(0, np.pi, old.sum())])
            self.age[old] = 0
        xs = (self.p[:, 0] / (self.k * np.pi) * 2 - 1) * ASPECT * 0.92
        ys = (self.p[:, 1] / np.pi * 2 - 1) * 0.8  # y = 0 at the hot floor, drawn at the bottom
        # temperature: hot floor, cold lid, and the rising (cos > 0) and sinking halves of each roll
        temp = 0.55 * (1 - y / np.pi) + 0.45 * np.cos(ph) * np.sin(y)
        hue = np.clip(1 - temp, 0, 1) * 0.9
        speed = np.hypot(u, v)
        fade = np.minimum(self.age * 6, 1) * np.minimum((1 - self.age) * 6, 1) * (0.75 + 0.25 * level)
        fade *= np.clip(speed / 0.8, 0.12, 1)  # the still eye of each roll stays dark
        return np.column_stack([xs, ys]), hue, fade


INDUSTRIAL = {"sparks": Sparks, "linkage": Linkage, "lattice": Lattice, "convection": Convection}
