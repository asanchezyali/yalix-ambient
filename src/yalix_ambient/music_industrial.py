"""Industrial meditation: the weight of German industrial metal at the pace of a slow breath
(the genre, never anyone's songs).

Every sound here is new to the channel: no choir, strings, organ, bells, music box, plucks or
pads from the other mixes. The palette is a factory at night:
- the wall: power chords (root, fifth, octave) from detuned saws, driven through an asymmetric
  clipper and a dark cabinet (two low-passes at 3 kHz and a bump at 120 Hz), quad-tracked (two
  takes hard left and right, two at 70 %) and cut by a hard gate, so every chug stops dead. The
  silence between hits is the stomp;
- a pulse-wave bass doubling the riff an octave down, ducked by the kick;
- a dry, heavy kick (110 to 42 Hz) and a short snare of noise and bitcrushed body;
- struck metal from the scrapyard, all low: sheet metal (dense inharmonic modes), oil drums,
  tuned pipes and a hydraulic press;
- the furnace: a drone of detuned saws under a filter that opens and closes with the breath, and
  a ring-modulated hum (two sines multiplied, the sound of a transformer);
- the hook on a low mono lead (square and saw with glide, G3 to G4): a falling minor line,
  four times, the fourth one varied;
- a concrete hall instead of a reverb: discrete early reflections and a dark, short tail.

The master is dark on purpose: everything above 3 kHz is rolled off, so an hour of it never tires.

Every theme follows the same arc: struck metal and the drone in the first seconds, the hook by
~10 s, the drums halfway through the first riff, a quiet break, a 4-bar build of tearing noise,
and the peak around 2:00, then the riff again and a short outro.

Styles:
- stomp: kick and snare on the backbeat, gated eighth chugs, sheet metal;
- march: four on the floor, sixteenth chugs at the peak;
- foundry: half time, chords left to ring, oil drums;
- press: a syncopated hydraulic press, pipes, the wall only at the peak;
- machine: no wall: the pulse bass and the scrapyard carry the groove;
- furnace: no drums and no wall: drone, transformer hum and slow metal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter, oaconvolve, sosfiltfilt

from yalix_ambient.music import SR, _env, euclidean, lowpass, midi_to_hz
from yalix_ambient.music_dark import bandpass, highpass

AEOLIAN = (0, 2, 3, 5, 7, 8, 10)
PHRYGIAN = (0, 1, 3, 5, 7, 8, 10)
DORIAN = (0, 2, 3, 5, 7, 9, 10)
HARMONIC = (0, 2, 3, 5, 7, 8, 11)

FORM = (("intro", 4), ("a", 8), ("a2", 8), ("break", 4), ("build", 4), ("b", 12), ("a", 6), ("outro", 4))
HEAVY = ("a", "a2", "b")

# Riffs over one bar of sixteenths. m muted root, p ringing root, o ringing octave,
# f muted flat second, t muted flat third, - hold, . rest.
RIFFS = {
    "stomp": ("m.m...m.m.m.f.o-", "p-------m.m.m.o-"),
    "march": ("m.m.m.m.m.mmm.o-", "mmmmmmmmp---p---"),
    "foundry": ("p---------------", "p-------o-------"),
    "press": ("", "p-----m.m.p-----"),
    "machine": ("", ""),
    "furnace": ("", ""),
}
OFFSET = {"m": 0, "p": 0, "o": 12, "f": 1, "t": 3}

STYLES = {  # style -> (kick steps, snare steps, layers)
    "stomp": ((0, 8), (4, 12), ("sheet", "drone", "hall")),
    "march": ((0, 4, 8, 12), (4, 12), ("sheet", "pipe", "drone")),
    "foundry": ((0,), (8,), ("drum", "drone", "hum")),
    "press": ((0, 6, 10), (8,), ("press", "pipe", "sheet", "hum", "drone")),
    "machine": ((0, 4, 8, 12), (), ("pulse", "sheet", "drum", "hall", "hum")),
    "furnace": ((), (), ("drone", "hum", "sheet", "pipe")),
}


@dataclass
class Spec:
    duration: float = 194.159
    bpm: float = 72.0  # one bar = 16 sixteenths
    tonic: int = 36  # C2, the low string in drop C
    mode: tuple[int, ...] = AEOLIAN
    progression: tuple[int, ...] = (0, 5, 2, 6)  # scale degrees, one chord every `chord_bars`
    chord_bars: int = 2
    style: str = "stomp"  # stomp | march | foundry | press | machine | furnace
    hook: tuple[tuple[int, int], ...] | None = None  # (step, scale degree) per bar; None: composed
    extra: tuple[str, ...] = ()  # add layers: sheet pipe drum press pulse drone hum hall
    drive: float = 9.0
    seed: int = 1
    fps: int = 30

    @property
    def step(self) -> float:
        return 60 / self.bpm / 4


# ------------------------------------------------------------------ instruments


def saw(freq: float, n: int, phase: float = 0.0) -> np.ndarray:
    return 2 * ((phase + freq * np.arange(n) / SR) % 1.0) - 1


def wall(freqs: list[float], n: int, muted: bool, rng: np.random.Generator) -> np.ndarray:
    """One take of a power chord before the amp: detuned saws, a pick, a palm that darkens it."""
    t = np.arange(n) / SR
    x = np.zeros(n)
    for f in freqs:
        for c in (-7, 6):
            x += saw(f * 2 ** ((c + rng.normal(0, 2)) / 1200), n, rng.random())
    if muted:  # the palm: the string dies fast and only its low part is left
        x = lowpass(x, 420) * np.exp(-t / 0.11) + lowpass(x, 1100) * np.exp(-t / 0.018) * 0.5
    else:
        x = lowpass(x, 1300) * (0.75 + 0.25 * np.exp(-t / 0.4))
    pick = lowpass(rng.standard_normal(n), 1800) * np.exp(-t / 0.004) * 0.6
    return (x + pick) * _env(n, 0.002, 0.015)


def drive(x: np.ndarray, gain: float) -> np.ndarray:
    """The amp: asymmetric clipping twice, then a dark cabinet."""
    x = bandpass(x, 70, 2200)
    y = np.tanh(gain * x + 0.25) - np.tanh(0.25)
    y = np.tanh(2.2 * bandpass(y, 80, 2600))
    y = lowpass(lowpass(y, 4200), 3600)
    y += 0.5 * bandpass(y, 90, 160)  # the chest of the cabinet
    y -= 0.35 * bandpass(y, 600, 1000)  # scooped mids
    return highpass(y, 60)


def kick_x(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / SR
    f = 42 + 68 * np.exp(-t / 0.03)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.32)
    thud = lowpass(rng.standard_normal(n), 900) * np.exp(-t / 0.006) * 0.5
    return np.tanh(2.5 * (body + thud)) * 0.8


def snare_x(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / SR
    body = (np.sin(2 * np.pi * 175 * t) + 0.5 * np.sin(2 * np.pi * 318 * t)) * np.exp(-t / 0.05)
    noise = bandpass(rng.standard_normal(n), 300, 3200) * np.exp(-t / 0.09)
    x = np.round((body + noise) * 6) / 6  # crushed: a cheap sampler from 1989
    return lowpass(x, 3500) * (t < 0.2) * 0.7


def sheet(freq: float, n: int, vel: float, rng: np.random.Generator) -> np.ndarray:
    """A struck sheet of steel: many inharmonic modes, the high ones die first."""
    t = np.arange(n) / SR
    x = np.zeros(n)
    for _ in range(18):
        r = rng.uniform(1, 9)
        x += np.sin(2 * np.pi * freq * r * t + rng.random() * 6) * np.exp(-t * (1.2 + 1.6 * r)) / r**0.6
    x += lowpass(rng.standard_normal(n), 1500) * np.exp(-t / 0.01) * 2
    return lowpass(np.tanh(x * 0.8), 2400) * vel * 0.4


def oil_drum(freq: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """An empty barrel hit with a hammer: a low thump and a rattling shell."""
    t = np.arange(n) / SR
    f = freq * (1 + 0.4 * np.exp(-t / 0.02))
    thump = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.25)
    shell = sum(np.sin(2 * np.pi * freq * r * t) * np.exp(-t / 0.15) for r in (2.32, 3.41, 4.87, 6.1)) * 0.25
    rattle = bandpass(rng.standard_normal(n), 200, 900) * np.exp(-t / 0.06) * 0.6
    return np.tanh(1.8 * (thump + shell + rattle)) * 0.6


def pipe(freq: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """A steel pipe hit with a wrench, tuned to the chord: free-bar modes, darkened."""
    t = np.arange(n) / SR
    x = sum(np.sin(2 * np.pi * freq * r * t) * np.exp(-t * d) * a
            for r, d, a in ((1, 1.2, 1), (2.76, 3.5, 0.45), (5.40, 8, 0.2)))  # fmt: skip
    x += bandpass(rng.standard_normal(n), 400, 1800) * np.exp(-t / 0.004)
    return lowpass(x, 2200) * 0.35


def press(n: int, rng: np.random.Generator) -> np.ndarray:
    """A hydraulic press: the hiss of the valve, then the chunk of the die 0.3 s later."""
    t = np.arange(n) / SR
    hiss = lowpass(rng.standard_normal(n), 700) * np.clip(t / 0.15, 0, 1) * (t < 0.3) * 0.25
    tc = np.clip(t - 0.3, 0, None)
    f = 55 + 90 * np.exp(-tc / 0.015)
    chunk = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tc / 0.18) * (t >= 0.3)
    clatter = bandpass(rng.standard_normal(n), 150, 1200) * np.exp(-tc / 0.03) * (t >= 0.3) * 0.6
    return np.tanh(2 * (hiss + chunk + clatter)) * 0.6


def pulse_bass(freq: float, n: int) -> np.ndarray:
    t = np.arange(n) / SR
    x = np.where((freq * t) % 1.0 < 0.3, 1.0, -0.3)
    return lowpass(np.tanh(2 * lowpass(x, 500)), 320) * _env(n, 0.003, 0.03)


def drone(freqs: list[float], n: int, breath: float, rng: np.random.Generator) -> np.ndarray:
    """The furnace: detuned saws under a filter that opens and closes with the breath."""
    x = sum(saw(f * 2 ** (c / 1200), n, rng.random()) for f in freqs for c in (-9, 0, 8))
    lo, hi = lowpass(x, 220), lowpass(x, 1100)
    t = np.arange(n) / SR
    o = 0.5 - 0.5 * np.cos(2 * np.pi * t / breath + rng.random() * 6)
    return (lo * (1 - o) + hi * o) * _env(n, 2.0, 2.5) * 0.08


def hum(freq: float, n: int) -> np.ndarray:
    """A transformer: two sines multiplied, so only their sum and difference sound."""
    t = np.arange(n) / SR
    x = np.sin(2 * np.pi * freq * t) * np.sin(2 * np.pi * freq * 1.5 * t + 0.3 * np.sin(2 * np.pi * 0.07 * t))
    return lowpass(np.tanh(1.5 * x), 900) * _env(n, 1.5, 1.5)


def lead(notes: list[tuple[float, float, float]], total: int) -> np.ndarray:
    """A mono lead with glide: (start s, duration s, midi). Square and saw, filtered low."""
    f = np.zeros(total)
    amp = np.zeros(total)
    for ts, dur, m in notes:
        s, e = int(ts * SR), min(int((ts + dur) * SR), total)
        f[s:e] = midi_to_hz(m)
        amp[s:e] = 1.0
    if not notes:
        return np.zeros(total)
    idx = np.maximum.accumulate(np.where(f > 0, np.arange(total), 0))  # hold the pitch between notes
    f = f[idx]
    f[f == 0] = midi_to_hz(notes[0][2])
    f = lfilter([0.0015], [1, -0.9985], f, zi=[f[0] * 0.9985])[0]  # 15 ms glide
    t = np.arange(total) / SR
    f *= 1 + 0.004 * np.sin(2 * np.pi * 4.8 * t)
    ph = np.cumsum(f) / SR
    x = np.sign(np.sin(2 * np.pi * ph)) * 0.6 + (2 * (ph * 1.003 % 1) - 1) * 0.5
    a = lfilter([0.001], [1, -0.999], amp)  # soft attack and release
    return lowpass(lowpass(np.tanh(1.5 * x), 1500), 1500) * a


def hall(x: np.ndarray, rng: np.random.Generator, length: float = 1.6) -> np.ndarray:
    """A concrete hall: discrete early reflections, then a dark tail. x is (n, 2)."""
    n = int(length * SR)
    t = np.arange(n) / SR
    out = np.zeros_like(x)
    for ch in range(2):
        ir = lowpass(rng.standard_normal(n), 1800) * np.exp(-t / (length / 6.9)) * np.clip(t / 0.03, 0, 1) * 0.08
        for d, g in zip(rng.uniform(0.011, 0.07, 6), (0.5, 0.4, 0.35, 0.3, 0.25, 0.2)):
            ir[int(d * SR)] += g * (1 if rng.random() < 0.5 else -1)
        out[:, ch] = oaconvolve(x[:, ch], lowpass(ir, 3000))[: len(x)]
    return out


# ------------------------------------------------------------------ composition


def triad(mode: tuple[int, ...], degree: int) -> list[int]:
    return [mode[(degree + k) % 7] + 12 * ((degree + k) // 7) for k in (0, 2, 4)]


def usable(mode: tuple[int, ...], degree: int) -> bool:
    a, _, c = triad(mode, degree)
    return c - a == 7


def fit(m: int, lo: int, hi: int) -> int:
    while m < lo:
        m += 12
    while m > hi:
        m -= 12
    return m


def plan(n_bars: int) -> list[str]:
    weights = np.array([b for _, b in FORM], dtype=float)
    lengths = np.maximum(np.round(weights / weights.sum() * n_bars), 2).astype(int)
    lengths[-3] += n_bars - lengths.sum()  # the length mismatch goes to the second riff section
    return [name for (name, _), k in zip(FORM, lengths) for _ in range(k)]


def compose_hook(rng: np.random.Generator) -> tuple[tuple[int, int], ...]:
    """A falling minor line over one bar: 4 to 6 notes, mostly steps down, one leap."""
    rhythms = ((0, 2, 4, 6, 8), (0, 3, 6, 8, 12), (0, 2, 4, 8, 10, 12), (0, 4, 6, 8))
    steps = rhythms[int(rng.integers(0, len(rhythms)))]
    deg = int(rng.choice([4, 7, 9]))
    out = []
    for i, s in enumerate(steps):
        out.append((s, deg))
        deg += -1 if rng.random() < 0.8 else int(rng.choice([2, 3]))
        if i == len(steps) - 2 and rng.random() < 0.5:
            deg -= 2  # the leap near the end
    return tuple(out)


def put(buf: np.ndarray, start: float, x: np.ndarray, g: float = 1.0) -> None:
    s = int(start)
    if s >= len(buf) or s < 0:
        return
    m = min(len(x), len(buf) - s)
    buf[s : s + m] += x[:m] * g


# ------------------------------------------------------------------ synthesis


def synthesize(spec: Spec) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(spec.seed)
    total = int(spec.duration * SR)
    st = spec.step
    bar = 16 * st
    n_bars = max(int(spec.duration / bar), 12)
    sections = plan(n_bars)
    kick_steps, snare_steps, layers = STYLES[spec.style]
    L = set(layers) | set(spec.extra)
    prog = [d for d in spec.progression if usable(spec.mode, d)] or [0]
    riff_a, riff_b = RIFFS[spec.style]
    hook = spec.hook or compose_hook(rng)
    sc = spec.mode
    breath = rng.uniform(9, 12)  # seconds per breath of the furnace filter

    def chord_at(b: int) -> int:
        return prog[(b // spec.chord_bars) % len(prog)]

    takes = np.zeros((4, total))  # quad-tracked: L, R, L70, R70
    gate = np.zeros(total)
    bass = np.zeros(total)
    kicks = np.zeros(total)
    snare_bus = np.zeros(total)
    metal = np.zeros((total, 2))
    furnace = np.zeros((total, 2))
    swell = np.zeros(total)
    lead_notes: list[tuple[float, float, float]] = []
    hits: list[tuple[float, float]] = []
    sec_start = [0] * n_bars  # first bar of the section each bar belongs to ("a" comes twice)
    for i in range(1, n_bars):
        sec_start[i] = sec_start[i - 1] if sections[i] == sections[i - 1] else i
    hook_count = 0

    def strike(buf: np.ndarray, ts: float, x: np.ndarray, g: float, pan: float | None = None) -> None:
        p = rng.uniform(0.25, 0.75) if pan is None else pan
        put(buf[:, 0], ts * SR, x, g * (1 - p) * 2)
        put(buf[:, 1], ts * SR, x, g * p * 2)

    for b in range(n_bars):
        sec = sections[b]
        t0 = b * bar
        new_sec = b == 0 or sections[b - 1] != sec
        k_in = b - sec_start[b]
        sec_len = next((j for j, x in enumerate(sections[sec_start[b]:]) if x != sec), n_bars - sec_start[b])
        deg = chord_at(b)
        root = fit(spec.tonic + sc[deg % 7] + 12 * (deg // 7), spec.tonic - 2, spec.tonic + 9)
        tri = [spec.tonic + x for x in triad(sc, deg)]
        pcs = {x % 12 for x in tri}
        heavy = sec in HEAVY
        first_of_chord = b % spec.chord_bars == 0
        drums_on = (heavy or sec == "build") and not (sec == "a" and sec_start[b] < n_bars / 2 and k_in < sec_len // 2)

        # ---- the wall, gated: every note opens the gate, nothing sounds between notes
        pat = riff_b if sec == "b" else (riff_a if sec in ("a", "a2") else "")
        for s, ch in enumerate(pat):
            if ch not in OFFSET:
                continue
            ln = 1
            while s + ln < 16 and pat[s + ln] == "-":
                ln += 1
            muted = ch in "mft"
            dur = ln * st * (0.7 if muted else 1.0)
            m0 = root + OFFSET[ch]
            freqs = [midi_to_hz(m0), midi_to_hz(m0 + 7)] + ([] if muted else [midi_to_hz(m0 + 12)])
            ts = t0 + s * st
            for k in range(4):
                jit = rng.uniform(0.004, 0.012) * (k % 2) + rng.uniform(0, 0.004)
                put(takes[k], (ts + jit) * SR, wall(freqs, int((dur + 0.02) * SR), muted, rng))
            gate[int(ts * SR) : min(int((ts + dur + 0.03) * SR), total)] = 1.0
            put(bass, ts * SR, pulse_bass(midi_to_hz(root - 12 + OFFSET[ch] % 12), int(dur * SR)), 0.9)
            if s == 0:
                hits.append((ts, 0.8))
        if "pulse" in L and (heavy or sec == "build"):  # the machine bass: eighths on the root
            for s in range(0, 16, 2):
                if sec != "b" and s in (6, 14) and rng.random() < 0.5:
                    continue
                m = root - 12 + (12 if s == 14 and sec == "b" else 0)
                g = 0.5 if sec != "build" else 0.2 + 0.1 * k_in
                put(bass, (t0 + s * st) * SR, pulse_bass(midi_to_hz(m), int(st * 1.6 * SR)), g)

        # ---- drums
        if kick_steps and drums_on:
            ks = kick_steps if sec != "build" else (0, 4, 8, 12)
            if spec.style == "stomp" and sec == "b":
                ks = (0, 8, 10)
            for s in ks:
                g = 1.0 if sec != "build" else 0.35 + 0.15 * k_in
                put(kicks, (t0 + s * st) * SR, kick_x(int(0.6 * SR), rng), g)
                hits.append((t0 + s * st, 1.0 if s == 0 else 0.75))
            if sec != "build":
                for s in snare_steps or ((4, 12) if sec == "b" else ()):
                    put(snare_bus, (t0 + s * st) * SR, snare_x(int(0.25 * SR), rng), 0.6)
                    hits.append((t0 + s * st, 0.7))
        if sec == "build" and k_in == sec_len - 1 and spec.style != "furnace":  # oil drums roll into the peak
            for s in range(16):
                strike(metal, t0 + s * st, oil_drum(midi_to_hz(fit(root, 28, 35)), int(0.5 * SR), rng), 0.08 + 0.02 * s)

        # ---- the scrapyard
        if new_sec and sec in ("intro", "break", "b", "outro"):  # every section opens on a big hit
            strike(metal, t0, sheet(midi_to_hz(fit(root, 40, 47)), int(4 * SR), 1.0, rng), 0.35, 0.5)
            put(kicks, t0 * SR, kick_x(int(0.6 * SR), rng), 0.7)
            hits.append((t0, 1.0))
        if "sheet" in L and (heavy or (sec == "build" and spec.style != "furnace")):
            steps = (6, 14) if spec.style != "machine" else tuple(s for s, h in enumerate(euclidean(5, 16, 3)) if h)
            for s in steps:
                if rng.random() < 0.25:
                    continue
                strike(metal, t0 + s * st, sheet(midi_to_hz(fit(root + 12, 48, 59)), int(1.2 * SR), 0.6, rng), 0.3 if spec.style == "machine" else 0.18)
                hits.append((t0 + s * st, 0.5))
        if "sheet" in L and spec.style == "furnace" and (first_of_chord or rng.random() < 0.3):
            s = 0 if first_of_chord else int(rng.choice([6, 10]))
            strike(metal, t0 + s * st, sheet(midi_to_hz(fit(root, 40, 47)), int(4 * SR), 0.7, rng), 0.22)
            hits.append((t0 + s * st, 0.6))
        if "drum" in L and (heavy or sec == "build"):
            for s in (0, 10) if spec.style == "foundry" else (3, 7, 11, 15):
                strike(metal, t0 + s * st, oil_drum(midi_to_hz(fit(root, 28, 35)), int(0.6 * SR), rng), 0.2)
                hits.append((t0 + s * st, 0.6))
        if "pipe" in L and (first_of_chord or heavy):
            for s in (0,) if not heavy else ((0, 12) if spec.style != "march" else (2, 6, 10, 14)):
                m = fit(tri[int(rng.integers(0, 3))] + 12, 52, 63)
                strike(metal, t0 + s * st, pipe(midi_to_hz(m), int(2.5 * SR), rng), 0.14 if heavy else 0.2)
                hits.append((t0 + s * st, 0.45))
        if "press" in L and (heavy or sec == "build"):
            for s in (0, 6, 10) if sec == "b" else (0, 10):
                put(kicks, (t0 + s * st - 0.3) * SR, press(int(1.0 * SR), rng), 0.4)
                hits.append((t0 + s * st, 0.9))

        # ---- the furnace: drone and transformer, held under everything
        if first_of_chord:
            n = int((spec.chord_bars * bar + 1.5) * SR)
            if "drone" in L and sec != "b":
                fr = [midi_to_hz(fit(root, 36, 47) - 12), midi_to_hz(fit(root, 36, 47) - 5)]
                x = drone(fr, n, breath, rng)
                g = 0.6 if heavy else 1.0
                put(furnace[:, 0], t0 * SR, x, g)
                put(furnace[:, 1], t0 * SR, np.roll(x, 400), g)
            if "hum" in L and sec in ("intro", "break", "build", "outro", "a2"):
                x = hum(midi_to_hz(fit(root, 48, 59)), n) * 0.07
                put(furnace[:, 0], t0 * SR, x)
                put(furnace[:, 1], t0 * SR, x)
            if not pat and "pulse" not in L and sec != "build":  # a sub floor when nothing else holds the root
                t = np.arange(n) / SR
                put(bass, t0 * SR, np.sin(2 * np.pi * midi_to_hz(fit(root, 28, 39)) * t) * _env(n, 1.5, 1.5), 0.15)
        if sec == "build":  # noise tearing up, then gone when the peak lands
            n = int(bar * SR)
            ramp = (k_in + np.arange(n) / n) / sec_len
            x = bandpass(rng.standard_normal(n), 80, 1500) + 0.8 * saw(midi_to_hz(root), n)
            put(swell, t0 * SR, np.tanh((1 + 6 * ramp) * x) * ramp**2, 1.0)

        # ---- the hook on the low lead: four times, the fourth one varied
        if sec != "build" and not (sec == "intro" and k_in < 1) and not (sec == "a" and k_in < 2 and b < n_bars / 2):
            hook_count += 1
            notes = list(hook)
            if hook_count % 4 == 0:
                s_last, d_last = notes[-1]
                notes[-1] = (s_last, d_last + int(rng.choice([-2, 2, 3])))
            sparse = sec in ("intro", "outro") or spec.style == "furnace"
            for j, (s, d) in enumerate(notes):
                if sparse and j % 2:
                    continue
                m = spec.tonic + 24 + sc[d % 7] + 12 * (d // 7)
                if s in (0, 8):
                    m = min((x for x in range(m - 4, m + 5) if x % 12 in pcs), key=lambda x: abs(x - m))
                m = fit(m, 55, 67)
                nxt = notes[j + 1][0] if j + 1 < len(notes) else 16
                lead_notes.append((t0 + s * st, (nxt - s) * st * 0.92, float(m)))
                hits.append((t0 + s * st, 0.35))

    # ------------------------------------------------------------------ buses
    mix = np.zeros((total, 2))
    if takes.any():
        a_g, r_g = np.exp(-1 / (0.001 * SR)), np.exp(-1 / (0.03 * SR))
        rise = lfilter([1 - a_g], [1, -a_g], gate)
        fall = lfilter([1 - r_g], [1, -r_g], gate)
        g_env = np.where(gate > 0, rise, fall)
        out = [drive(tk, spec.drive) * g_env for tk in takes]
        mix[:, 0] += (out[0] + 0.7 * out[2] + 0.3 * out[3]) * 0.4
        mix[:, 1] += (out[1] + 0.7 * out[3] + 0.3 * out[2]) * 0.4

    # The kick ducks the bass, the furnace and the lead: the whole room pumps.
    k_env = lfilter([0.0005], [1, -0.9995], np.abs(lowpass(kicks, 150)))
    duck = 1 - 0.5 * np.clip(k_env / (k_env.max() + 1e-9) * 3, 0, 1)
    b2 = lowpass(np.tanh(1.4 * bass), 260) * 0.3 * duck
    mix += b2[:, None]
    mix += kicks[:, None] * 0.55
    mix += snare_bus[:, None] * 0.45
    mix += metal
    mix += furnace * duck[:, None]
    ld = lead(lead_notes, total) * 0.09 * duck
    mix += np.stack([ld * 0.9, ld * 1.1], axis=1)
    mix += np.stack([swell, swell], axis=1) * 0.06
    t_all = np.arange(total) / SR
    if "hall" in L:  # the machine hall breathing at the half-bar
        trem = lowpass(0.5 + 0.5 * np.sign(np.sin(2 * np.pi * t_all / (bar / 2))), 30)
        mask = np.zeros(total)
        for b in range(n_bars):
            if sections[b] in HEAVY or sections[b] == "build":
                mask[int(b * bar * SR) : int((b + 1) * bar * SR)] = 1
        room = bandpass(rng.standard_normal(total), 150, 700)
        mix += (room * trem * lowpass(mask, 2) * 0.03)[:, None]

    wet = metal * 0.7 + np.stack([snare_bus, snare_bus], axis=1) * 0.25 + np.stack([ld, ld], axis=1) * 0.5
    mix += 0.45 * hall(wet, rng)

    mix *= _env(total, 1.0, 6.0)[:, None]
    mix = np.tanh(1.2 * mix)
    mix = lowpass(mix.T, 7000).T * 0.75 + lowpass(mix.T, 3000).T * 0.25  # the dark master
    mix = highpass(mix.T, 28).T
    # A low shelf that keeps the sub at ~40 % of the energy: heavy, never a rumble.
    low = sosfiltfilt(butter(2, 100, btype="low", fs=SR, output="sos"), mix, axis=0)
    e_low, e_high = np.sum(low**2), np.sum((mix - low) ** 2)
    mix += (min(1.0, np.sqrt(2 / 3 * e_high / (e_low + 1e-9))) - 1) * low
    mix /= np.max(np.abs(mix)) + 1e-9
    mix *= 10 ** (-1.0 / 20)
    return mix.astype(np.float32), analyze(mix, b2, hits, spec)


def analyze(mix, bass, hits, spec: Spec) -> dict:
    hop = SR // spec.fps
    n_frames = len(mix) // hop
    mono = mix.mean(axis=1)

    def env_of(x, smooth):
        r = np.sqrt(np.array([np.mean(x[i * hop : (i + 1) * hop] ** 2) for i in range(n_frames)]))
        k = np.exp(-np.arange(-30, 31) ** 2 / (2 * smooth**2))
        r = np.convolve(r, k / k.sum(), mode="same")
        return [round(float(v), 4) for v in (r - r.min()) / (r.max() - r.min() + 1e-9)]

    hit = np.zeros(n_frames)  # every kick, metal hit and lead note: a jump that decays
    for ts, v in hits:
        f0 = int(ts * spec.fps)
        for j in range(f0, min(f0 + 12, n_frames)):
            hit[j] = max(hit[j], v * np.exp(-(j - f0) / 3))
    return {
        "fps": spec.fps,
        "duration": spec.duration,
        "chord_seconds": 16 * spec.step * spec.chord_bars,
        "rms": env_of(mono, 8),
        "bass": env_of(bass, 3),
        "kick": [round(float(v), 4) for v in hit],
    }


def render_track(spec: Spec, wav_path: Path, analysis_path: Path) -> None:
    mix, analysis = synthesize(spec)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(wav_path, SR, mix)
    analysis_path.write_text(json.dumps(analysis))
