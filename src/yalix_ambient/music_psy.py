"""Psytechno: the structure of techno with the drive of Goa trance (the genre, never anyone's songs).

Every sound here is new to the channel:
- a hard kick (150 to 48 Hz) with its rumble: the kick sent through a short dark room, distorted,
  low-passed and ducked by the kick itself, so the low end rolls between the beats;
- the rolling psy bass: three short notes after every kick (K-B-B-B on the sixteenths), each
  with its own filter snap;
- a sitar: additive strings with the jawari buzz (the curved bridge that makes the high
  harmonics flare) and meend, the slide into a note;
- a tanpura: four strings (fifth, octave, octave, root) plucked in a cycle, the harmonics
  rising and falling in turn, the drone of Indian music;
- tabla: the right drum tuned and harmonic, the left one bending up after each stroke;
- flamenco: palmas in a twelve-beat compás laid over the four-four (they meet again every
  three bars) and rasgueado on a nylon guitar;
- psy effects: falling zaps, a low chord cut by a trance gate, a distorted growl whose filter
  sweeps across the bar, tribal toms, risers and snare rolls into the drop;
- a temple instead of a reverb.

Scales come from the Middle East and India: phrygian dominant and double harmonic. Every theme
follows the arc of a club track: tanpura and kick, the rolling bass, the hook, a break where the
sitar plays alone, a build of rolls and risers, the drop around 2:00, then the groove again.

Styles:
- goa: rolling bass, tanpura, sitar hook, zaps;
- warrior: hard kick and rumble, gated chords, palmas at the drop;
- temple: tabla over the groove, tanpura throughout, low gated chords;
- hypno: minimal: kick, rumble, rolling bass, zaps, a few sitar notes;
- flamenco: palmas in compás, rasgueado, phrygian dominant;
- ritual: tribal toms and a distorted growl over the roll, the sitar only at the drop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter, oaconvolve, sosfiltfilt

from yalix_ambient.music import SR, _env, lowpass, midi_to_hz
from yalix_ambient.music_dark import bandpass, highpass

PHRYGIAN = (0, 1, 3, 5, 7, 8, 10)
PHRYGIAN_DOM = (0, 1, 4, 5, 7, 8, 10)
DOUBLE_HARMONIC = (0, 1, 4, 5, 7, 8, 11)
AEOLIAN = (0, 2, 3, 5, 7, 8, 10)

FORM = (("intro", 4), ("a", 8), ("a2", 8), ("break", 4), ("build", 4), ("b", 12), ("a", 6), ("outro", 4))
HEAVY = ("a", "a2", "b")

STYLES = {  # bass pattern, layers
    "goa": ("roll", ("tanpura", "sitar", "zaps")),
    "warrior": ("roll", ("gate", "palmas_b", "zaps", "toms_b", "growl_b")),
    "temple": ("roll", ("tabla", "tanpura", "gate")),
    "hypno": ("roll2", ("zaps", "sitar_sparse")),
    "flamenco": ("gallop", ("palmas", "rasgueado", "sitar_b")),
    "ritual": ("roll", ("toms", "growl", "tanpura", "sitar_b")),
}


@dataclass
class Spec:
    duration: float = 194.159
    bpm: float = 142.0
    tonic: int = 40  # E2
    mode: tuple[int, ...] = PHRYGIAN_DOM
    roots: tuple[int, ...] = (0, 0, 1, 0)  # semitones above the tonic, one per `chord_bars`
    chord_bars: int = 4
    style: str = "goa"
    extra: tuple[str, ...] = ()
    seed: int = 1
    fps: int = 30

    @property
    def step(self) -> float:
        return 60 / self.bpm / 4


# ------------------------------------------------------------------ instruments


def kick(n: int) -> np.ndarray:
    t = np.arange(n) / SR
    f = 48 + 102 * np.exp(-t / 0.022)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.26)
    click = np.exp(-t / 0.002) * 0.4
    return np.tanh(3.2 * (body + click)) * 0.8


def psy_bass(freq: float, n: int) -> np.ndarray:
    """One short rolling-bass note: saw and square, a filter that snaps shut."""
    t = np.arange(n) / SR
    ph = freq * t
    x = 0.6 * (2 * (ph % 1) - 1) + 0.4 * np.sign(np.sin(2 * np.pi * ph))
    y = lowpass(x, 180) + lowpass(x, 900) * np.exp(-t / 0.025)
    return np.tanh(2.6 * y) * _env(n, 0.001, 0.01)


def hat(n: int, rng: np.random.Generator, open_: bool = False) -> np.ndarray:
    t = np.arange(n) / SR
    x = lowpass(bandpass(rng.standard_normal(n), 3500, 7000), 5000)
    return x * np.exp(-t / (0.09 if open_ else 0.018)) * (0.16 if open_ else 0.1)


def clap(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / SR
    x = np.zeros(n)
    for j in range(3):
        s = int(j * 0.011 * SR)
        x[s:] += bandpass(rng.standard_normal(n - s), 800, 3200) * np.exp(-t[: n - s] / (0.01 if j < 2 else 0.09))
    return lowpass(x, 3500) * 0.35


def palma(n: int, rng: np.random.Generator, accent: bool) -> np.ndarray:
    """One flamenco hand clap: a single sharp, dry hit (accents are 'sordas' no more)."""
    t = np.arange(n) / SR
    x = bandpass(rng.standard_normal(n), 1200, 4000) * np.exp(-t / 0.012)
    body = np.sin(2 * np.pi * rng.uniform(900, 1100) * t) * np.exp(-t / 0.006) * 0.3
    return lowpass(x + body, 3200) * (0.4 if accent else 0.22)


def sitar(freq: float, n: int, rng: np.random.Generator, bend: float = 0.0) -> np.ndarray:
    """A sitar string: harmonics 1..28, the jawari makes the upper ones flare a moment after the
    pluck and buzz; `bend` semitones of meend slide into the note over 150 ms."""
    t = np.arange(n) / SR
    f = freq * 2 ** (bend * np.exp(-t / 0.05) / 12)
    ph = np.cumsum(f) / SR
    x = np.zeros(n)
    for k in range(1, 29):
        if freq * k > 4000:
            break
        flare = 1 + 0.4 * (k / 28) * np.exp(-((t - 0.04 - 0.004 * k) ** 2) / 0.002)  # the jawari bloom
        x += np.sin(2 * np.pi * k * ph + rng.random()) * flare * np.exp(-t * (0.9 + 0.25 * k)) / k**0.7
    buzz = bandpass(np.tanh(3 * x), 900, 2500) * 0.12
    return lowpass(lowpass(x + buzz, 3200), 4000) * _env(n, 0.001, 0.05) * 0.25


def tanpura(freq: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Four strings (pa, sa, sa, low Sa) plucked in turn every ~1.3 s; in each, the harmonics
    swell one after another, the shimmer of the jawari."""
    t = np.arange(n) / SR
    out = np.zeros(n)
    strings = (freq * 1.5, freq * 2, freq * 2, freq)
    period = rng.uniform(1.2, 1.5)
    for j, f in enumerate(strings):
        for start in np.arange(j * period / 4, n / SR, period):
            s = int(start * SR)
            m = min(int(period * 4 * SR), n - s)
            if m <= 0:
                continue
            tt = t[:m]
            y = np.zeros(m)
            for k in range(1, 16):
                if f * k > 5000:
                    break
                swell = np.exp(-((tt - 0.15 * k) ** 2) / (0.25 + 0.05 * k))
                y += np.sin(2 * np.pi * f * k * tt + rng.random()) * (0.5 + 0.3 * swell) / k**1.3 * np.exp(-tt / 2.5)
            out[s : s + m] += y
    return lowpass(out, 2500) * _env(n, 1.0, 1.5) * 0.035


def tabla(kind: str, freq: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """dayan: the tuned right drum, harmonic thanks to the black paste; bayan: the left, bass
    drum whose pitch rises under the palm (ghe)."""
    t = np.arange(n) / SR
    if kind == "dayan":
        x = sum(np.sin(2 * np.pi * freq * k * t) * np.exp(-t * (6 + 4 * k)) / k for k in (1, 2, 3, 4, 5))
        x += bandpass(rng.standard_normal(n), 1200, 3000) * np.exp(-t / 0.004) * 0.2
        return lowpass(x, 3000) * 0.35
    f = freq * (1 + 0.35 * np.clip(t / 0.18, 0, 1))
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.35)
    return np.tanh(1.5 * x) * 0.4


def nylon(freq: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """A nylon string: soft, few harmonics, a quick decay of the top."""
    t = np.arange(n) / SR
    x = np.zeros(n)
    for k in range(1, 14):
        x += abs(np.sin(np.pi * k * 0.2)) / k * np.sin(2 * np.pi * freq * k * t + rng.random()) * np.exp(-t * (2 + 1.5 * k))
    return lowpass(x, 2200) * _env(n, 0.002, 0.04) * 0.25


def zap(n: int, rng: np.random.Generator) -> np.ndarray:
    """A psy zap: an FM blip whose pitch falls from about 2 kHz to 150 Hz."""
    t = np.arange(n) / SR
    f = 80 + rng.uniform(500, 1000) * np.exp(-t / rng.uniform(0.03, 0.08))
    ph = np.cumsum(f) / SR
    x = np.sin(2 * np.pi * ph + 2.5 * np.sin(2 * np.pi * ph * 1.5))
    return lowpass(x * np.exp(-t / 0.12), 2200) * 0.2


def gated_chord(freqs: list[float], n: int, gate: str, step: float, rng: np.random.Generator) -> np.ndarray:
    """A detuned pulse chord cut by a trance gate (one character per sixteenth, 1 open, 0 shut)."""
    t = np.arange(n) / SR
    x = sum(np.sign(np.sin(2 * np.pi * f * 2 ** (c / 1200) * t + rng.random() * 6)) for f in freqs for c in (-8, 7))
    x = lowpass(lowpass(x, 900), 1400)
    g = np.array([gate[int(i) % len(gate)] == "1" for i in t / step], dtype=float)
    a = np.exp(-1 / (0.003 * SR))
    g = lfilter([1 - a], [1, -a], g)
    return np.tanh(1.5 * x) * g * _env(n, 0.01, 0.05) * 0.08


def tom(freq: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """A floor tom hit hard: the pitch sags as the skin relaxes."""
    t = np.arange(n) / SR
    f = freq * (1 + 0.5 * np.exp(-t / 0.03))
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.25)
    x += lowpass(rng.standard_normal(n), 1200) * np.exp(-t / 0.01) * 0.3
    return np.tanh(2 * x) * 0.5


def growl(freq: float, n: int, cutoff: float, rng: np.random.Generator) -> np.ndarray:
    """One sixteenth of the psy growl: two detuned saws driven hard, the filter where the bar's
    sweep has reached."""
    t = np.arange(n) / SR
    x = sum(2 * ((freq * 2 ** (c / 1200) * t + rng.random()) % 1) - 1 for c in (-12, 10))
    y = np.tanh(4 * x)
    return lowpass(lowpass(y, cutoff), cutoff * 1.3) * _env(n, 0.002, 0.01) * 0.3


def temple(x: np.ndarray, rng: np.random.Generator, length: float = 2.2) -> np.ndarray:
    """A stone temple: a dense, warm tail. x is (n, 2)."""
    n = int(length * SR)
    t = np.arange(n) / SR
    out = np.zeros_like(x)
    for ch in range(2):
        ir = lowpass(rng.standard_normal(n), 3200) * np.exp(-t / (length / 6.9)) * np.clip(t / 0.02, 0, 1) * 0.06
        out[:, ch] = oaconvolve(x[:, ch], ir)[: len(x)]
    return out


# ------------------------------------------------------------------ composition


def fit(m: int, lo: int, hi: int) -> int:
    while m < lo:
        m += 12
    while m > hi:
        m -= 12
    return m


def plan(n_bars: int) -> list[str]:
    weights = np.array([b for _, b in FORM], dtype=float)
    lengths = np.maximum(np.round(weights / weights.sum() * n_bars), 2).astype(int)
    lengths[-3] += n_bars - lengths.sum()
    return [name for (name, _), k in zip(FORM, lengths) for _ in range(k)]


def compose_hook(rng: np.random.Generator) -> tuple[tuple[int, int, float], ...]:
    """A two-bar sitar phrase: (step 0..31, scale degree, meend in semitones)."""
    rhythms = ((0, 3, 6, 8, 12, 14, 16, 19, 22, 24), (0, 4, 6, 10, 12, 16, 20, 22, 26), (0, 2, 3, 6, 8, 16, 18, 19, 22, 24))
    steps = rhythms[int(rng.integers(0, len(rhythms)))]
    deg = [4, 5, 4, 2, 1, 0, 1, 2, 1, 0, -1, 0]
    s0 = int(rng.integers(0, 4))
    return tuple((s, deg[(i + s0) % len(deg)], float(rng.choice([0, 0, 0, -1]))) for i, s in enumerate(steps))


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
    bass_pat, layers = STYLES[spec.style]
    L = set(layers) | set(spec.extra)
    sc = spec.mode
    hook = compose_hook(rng)
    gate = rng.choice(["1011011010110110", "1101101101101101", "1110111011101011"])

    kicks = np.zeros(total)
    bass = np.zeros(total)
    perc = np.zeros((total, 2))
    tops = np.zeros((total, 2))
    melo = np.zeros((total, 2))
    drone = np.zeros((total, 2))
    fx = np.zeros((total, 2))
    hits: list[tuple[float, float]] = []
    sec_start = [0] * n_bars
    for i in range(1, n_bars):
        sec_start[i] = sec_start[i - 1] if sections[i] == sections[i - 1] else i
    hook_count = 0
    compas = 0  # palmas run in twelve, across the bar lines

    def pan(buf: np.ndarray, ts: float, x: np.ndarray, g: float, p: float) -> None:
        put(buf[:, 0], ts * SR, x, g * (1 - p) * 2)
        put(buf[:, 1], ts * SR, x, g * p * 2)

    sitar_cache: dict[tuple[int, float], np.ndarray] = {}

    def sitar_note(ts: float, m: int, g: float, bend: float, p: float = 0.55) -> None:
        key = (m, bend)
        if key not in sitar_cache:
            sitar_cache[key] = sitar(midi_to_hz(m), int(1.6 * SR), rng, bend)
        pan(melo, ts, sitar_cache[key], g, p)

    for b in range(n_bars):
        sec = sections[b]
        t0 = b * bar
        new_sec = b == 0 or sections[b - 1] != sec
        k_in = b - sec_start[b]
        sec_len = next((j for j, x in enumerate(sections[sec_start[b]:]) if x != sec), n_bars - sec_start[b])
        root = spec.tonic + spec.roots[(b // spec.chord_bars) % len(spec.roots)]
        heavy = sec in HEAVY
        kick_on = sec != "break" and not (sec == "outro" and k_in >= sec_len - 1)
        if sec == "intro" and spec.style not in ("warrior", "hypno") and k_in < 2:
            kick_on = False  # each style opens on its own colour before the kick arrives
        bass_on = (heavy and not (sec == "a" and sec_start[b] < n_bars / 2 and k_in < 2)) or (sec == "build" and k_in < 2)
        tri = [root + sc[0], root + sc[2], root + sc[4]]

        # ---- kick: four on the floor
        if kick_on:
            for s in (0, 4, 8, 12):
                if sec == "build" and k_in == sec_len - 1 and s == 12:
                    continue  # a gap before the drop
                put(kicks, (t0 + s * st) * SR, kick(int(0.4 * SR)), 1.0)
                hits.append((t0 + s * st, 1.0 if s == 0 else 0.8))

        # ---- the rolling bass
        if bass_on:
            for beat in range(4):
                for j in (1, 2, 3):
                    s = beat * 4 + j
                    if bass_pat == "gallop" and j == 2:
                        continue
                    if bass_pat == "roll2" and beat % 2 and j == 3:
                        m = root - 12 + 12  # an octave flick every other beat
                    elif sec == "b" and beat == 3 and j == 3:
                        m = root - 12 + sc[1]  # the half step of the scale, the Goa sting
                    else:
                        m = root - 12
                    put(bass, (t0 + s * st) * SR, psy_bass(midi_to_hz(fit(m, 28, 40)), int(st * 0.9 * SR)), 1.0)

        # ---- tops: offbeat open hats, closed sixteenths, the clap
        if heavy or sec == "build":
            for beat in range(4):
                pan(tops, t0 + (beat * 4 + 2) * st, hat(int(0.2 * SR), rng, True), 0.8, 0.55)
                if heavy and not (sec == "a" and k_in < sec_len // 2):
                    for j in (1, 3):
                        pan(tops, t0 + (beat * 4 + j) * st, hat(int(0.05 * SR), rng), 0.6, 0.35 + 0.3 * (j == 3))
            if sec != "build":
                for s in (4, 12):
                    pan(perc, t0 + s * st, clap(int(0.3 * SR), rng), 0.7, 0.5)
        if sec == "build":  # a snare roll that doubles every bar, and a riser
            div = (1, 2, 4, 8)[min(k_in, 3)]
            for i in range(4 * div):
                s = i * 4 / div
                pan(perc, t0 + s * st, clap(int(0.12 * SR), rng), 0.25 + 0.12 * k_in, 0.5)
            n = int(bar * SR)
            ramp = (k_in + np.arange(n) / n) / sec_len
            noise = rng.standard_normal(n)
            x = bandpass(noise, 400, 2500) * (1 - ramp) + bandpass(noise, 1500, 6000) * ramp
            pan(fx, t0, x * ramp**2, 0.06, 0.5)

        # ---- percussion from the styles
        if "tabla" in L and (heavy or sec in ("intro", "break")):
            bols = "dt.dtd.ttd.dt.td"  # d dayan, t bayan, . rest: a theka feel over sixteenths
            for s, ch in enumerate(bols):
                if ch == ".":
                    continue
                kind = "dayan" if ch == "d" else "bayan"
                f = midi_to_hz(fit(spec.tonic + 24, 60, 71)) if kind == "dayan" else 90.0
                pan(perc, t0 + s * st, tabla(kind, f, int(0.5 * SR), rng), 0.5 if s % 4 == 0 else 0.35, 0.65 if kind == "dayan" else 0.4)
        palmas_on = ("palmas" in L and (heavy or sec in ("intro", "break"))) or ("palmas_b" in L and sec == "b")
        if palmas_on:
            for s in range(16):
                beat12 = compas % 12
                compas += 1
                if beat12 in (2, 5, 7, 9, 11) or rng.random() < 0.25:  # the bulería accents: 3 6 8 10 12
                    pan(perc, t0 + s * st, palma(int(0.1 * SR), rng, beat12 in (2, 5, 7, 9, 11)), 0.9, rng.uniform(0.3, 0.7))
        else:
            compas += 16

        # ---- drones and chords
        if b % spec.chord_bars == 0:
            n = int((spec.chord_bars * bar + 1.0) * SR)
            if "tanpura" in L:
                x = tanpura(midi_to_hz(fit(root, 48, 59)), n, rng)
                put(drone[:, 0], t0 * SR, x, 1.0)
                put(drone[:, 1], t0 * SR, np.roll(x, 300), 1.0)
            if "gate" in L and sec in ("a2", "break", "b"):
                x = gated_chord([midi_to_hz(fit(m, 45, 57)) for m in tri], n, gate, st, rng)
                pan(drone, t0, x, 1.0 if sec != "break" else 0.6, 0.5)
        if "rasgueado" in L and (heavy or (sec in ("intro", "break") and b % 2 == 0)):  # strums, the fingers 12 ms apart
            for s in (0, 6, 10) if sec != "break" else (0,):
                for j, m in enumerate(sorted(fit(x, 45, 57) for x in tri + [tri[0] + 12, tri[2] - 12])):
                    pan(melo, t0 + s * st + 0.012 * j, nylon(midi_to_hz(m), int(1.2 * SR), rng), 0.5, 0.35 + 0.06 * j)

        # ---- tribal toms and the growl
        if ("toms" in L and (heavy or sec in ("intro", "build"))) or ("toms_b" in L and sec == "b"):
            for s, m in ((0, 0), (3, 0), (6, 5), (10, 7), (11, 5), (14, 0)) if b % 2 else ((0, 0), (6, 5), (10, 0), (13, 7)):
                pan(perc, t0 + s * st, tom(midi_to_hz(fit(root + m, 40, 52)), int(0.4 * SR), rng), 0.45, 0.3 + 0.04 * m)
        if ("growl" in L and sec in ("a2", "b")) or ("growl_b" in L and sec == "b"):
            ph = rng.uniform(0, 2 * np.pi)
            for s in range(16):
                if s % 4 == 0:
                    continue  # leave the kick alone
                c = 350 + 900 * (0.5 + 0.5 * np.sin(2 * np.pi * (b % 4 * 16 + s) / 64 + ph))  # one sweep per 4 bars
                pan(melo, t0 + s * st, growl(midi_to_hz(fit(root, 40, 52)), int(st * 0.85 * SR), c, rng), 0.5, 0.5)

        # ---- zaps
        if "zaps" in L and (heavy or sec == "build" or (sec == "intro" and spec.style == "goa")) and rng.random() < 0.6:
            s = int(rng.choice([3, 7, 11, 14]))
            pan(fx, t0 + s * st, zap(int(0.4 * SR), rng), 0.5, rng.uniform(0.15, 0.85))

        # ---- the sitar hook: two bars, four times, the fourth varied
        sitar_on = ("sitar" in L and sec in ("a2", "break", "b", "outro")) or ("sitar_b" in L and sec in ("break", "b"))
        sitar_on = sitar_on or ("sitar_sparse" in L and sec in ("break", "b") and b % 4 == 0)
        if sitar_on and b % 2 == 0 and sec != "build":
            hook_count += 1
            for i, (s, d, bend) in enumerate(hook):
                if sec == "break" and i % 3 == 2:
                    continue
                dd = d + (1 if hook_count % 4 == 0 and i >= len(hook) - 2 else 0)
                if s % 4 == 0:  # strong steps take a chord tone: no rub against the bass and the drone
                    dd = min((0, 2, 4, 7), key=lambda x: abs(x - dd))
                m = fit(spec.tonic + 24 + sc[dd % 7] + 12 * (dd // 7), 50, 64)
                sitar_note(t0 + s * st, m, 0.2 if sec != "break" else 0.3, bend)
                hits.append((t0 + s * st, 0.4))
        if new_sec and sec in ("break", "b"):
            hits.append((t0, 1.0))

    # ------------------------------------------------------------------ buses
    k_env = lfilter([0.002], [1, -0.998], np.abs(kicks))
    duck = 1 - 0.75 * np.clip(k_env / (k_env.max() + 1e-9) * 2.5, 0, 1)
    # The rumble: the kick in a small dark room, driven, kept under 140 Hz, ducked by the kick.
    n_ir = int(0.5 * SR)
    ir = lowpass(rng.standard_normal(n_ir), 400) * np.exp(-np.arange(n_ir) / (0.12 * SR))
    rumble = lowpass(np.tanh(6 * oaconvolve(kicks, ir)[:total] * 0.05), 150) * duck * 0.5
    mix = np.zeros((total, 2))
    mix += (kicks * 0.8 + rumble)[:, None]
    mix += (lowpass(bass, 1200) * 0.4 * duck)[:, None]
    mix += perc + tops * 0.4 + melo + drone + fx
    wet = melo * 0.6 + perc * 0.3 + drone * 0.5 + fx * 0.5
    mix += 0.4 * temple(wet, rng)

    mix *= _env(total, 0.5, 4.0)[:, None]
    mix = np.tanh(1.15 * mix)
    mix = lowpass(mix.T, 6000).T * 0.6 + lowpass(mix.T, 2800).T * 0.4  # dark master: nothing sharp
    mix = highpass(mix.T, 28).T
    low = sosfiltfilt(butter(2, 100, btype="low", fs=SR, output="sos"), mix, axis=0)
    e_low, e_high = np.sum(low**2), np.sum((mix - low) ** 2)
    mix += (min(1.0, np.sqrt(2 / 3 * e_high / (e_low + 1e-9))) - 1) * low
    mix /= np.max(np.abs(mix)) + 1e-9
    mix *= 10 ** (-1.0 / 20)
    return mix.astype(np.float32), analyze(mix, bass, hits, spec)


def analyze(mix, bass, hits, spec: Spec) -> dict:
    hop = SR // spec.fps
    n_frames = len(mix) // hop
    mono = mix.mean(axis=1)

    def env_of(x, smooth):
        r = np.sqrt(np.array([np.mean(x[i * hop : (i + 1) * hop] ** 2) for i in range(n_frames)]))
        k = np.exp(-np.arange(-30, 31) ** 2 / (2 * smooth**2))
        r = np.convolve(r, k / k.sum(), mode="same")
        return [round(float(v), 4) for v in (r - r.min()) / (r.max() - r.min() + 1e-9)]

    hit = np.zeros(n_frames)
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
