"""Cyberpunk music, v2: every episode is its own song, not a preset with a new key.

Styles of genres, never anyone's songs. Each episode picks independently:
- groove: four-on-the-floor, half-time, broken beat, drum & bass, trap, electro, downtempo,
  or none; swing (late off-beat sixteenths) and how much the mix pumps under the kick
  (0 = no pumping at all, which is most of them);
- bass: rolling octaves, pulse, detuned reese, FM growl, acid (resonant 303-style with
  accents), 808 with glide, or a plain sub;
- pad: supersaw, FM glass, formant choir, strings, or none;
- lead: PWM square, gliding saw, FM bell, acid, Karplus-Strong pluck, formant voice, or none;
- arpeggio: none, square, pluck or bell, in eighths, triplets or sixteenths;
- texture: rain, city rumble with distant horns, radio static with Morse, vinyl crackle;
- form: club, anthem, build, ambient or loop (layers rotate every 8 bars).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter, sawtooth, square

from yalix_ambient.music import SR, _env, euclidean, lowpass, midi_to_hz, reverb
from yalix_ambient.music_dark import bandpass, hat, highpass, kick, snare
from yalix_ambient.music_series import FORMANTS_AH, FORMANTS_OH, bell, choir, strings
from yalix_ambient.music_v2 import ks_pluck

AEOLIAN = (0, 2, 3, 5, 7, 8, 10)
DORIAN = (0, 2, 3, 5, 7, 9, 10)
PHRYGIAN = (0, 1, 3, 5, 7, 8, 10)
HARMONIC_MINOR = (0, 2, 3, 5, 7, 8, 11)
LYDIAN = (0, 2, 4, 6, 7, 9, 11)


@dataclass
class Spec:
    duration: float = 194.159
    bpm: float = 110.0
    tonic: int = 45  # A2
    scale: tuple[int, ...] = AEOLIAN
    progression: tuple[int, ...] = (0, 5, 2, 6)  # scale degrees
    chord_bars: int = 2
    sevenths: bool = False
    groove: str = "four"  # four halftime break dnb trap electro downtempo minimal idm heartbeat chip none
    swing: float = 0.0  # fraction of a sixteenth that off-beat sixteenths arrive late
    pump: float = 0.0  # sidechain depth under the kick, 0..1
    bass: str = "rolling"  # rolling | pulse | reese | fm | acid | 808 | seq | sub | none
    pad: str = "supersaw"  # supersaw | glass | choir | strings | drone | none
    lead: str = "square"  # square | saw | bell | acid | pluck | vox | sine | chip | none
    arp: str = "square"  # square | pluck | bell | chip | none
    arp_rate: int = 16  # 8 eighths, 12 triplets, 16 sixteenths
    arp_shape: str = "up"  # up | updown | pendulum | random
    texture: tuple[str, ...] = ()  # rain city radio vinyl typing typewriter clockwork steam modem servers
    form: str = "club"  # club | anthem | build | ambient | loop | slowburn | pulse
    lead_density: int = 6  # notes per two-bar phrase
    stutter: float = 0.0  # audio glitches: buffer repeats and bit-crushed moments, 0..1
    reverb_s: float = 3.2
    seed: int = 1
    fps: int = 30


# ------------------------------------------------------------------ helpers


def chord(spec: Spec, degree: int) -> list[int]:
    out = []
    for k in (0, 2, 4, 6) if spec.sevenths else (0, 2, 4):
        octv, idx = divmod(degree + k, len(spec.scale))
        out.append(spec.tonic + 12 * octv + spec.scale[idx])
    return out


def scale_note(spec: Spec, degree: int, base: int) -> int:
    octv, idx = divmod(int(degree), len(spec.scale))
    return base + 12 * octv + spec.scale[idx]


def resonant_lowpass(x: np.ndarray, fc: float, q: float) -> np.ndarray:
    """RBJ biquad low-pass; high Q gives the squelch of an acid line."""
    w0 = 2 * np.pi * min(fc, SR * 0.45) / SR
    alpha = np.sin(w0) / (2 * q)
    c = np.cos(w0)
    b = np.array([(1 - c) / 2, 1 - c, (1 - c) / 2])
    a = np.array([1 + alpha, -2 * c, 1 - alpha])
    return lfilter(b / a[0], a / a[0], x)


def supersaw(freq: float, t: np.ndarray, voices: int = 5, spread: float = 14.0) -> np.ndarray:
    cents = np.linspace(-spread, spread, voices)
    return sum(sawtooth(2 * np.pi * freq * 2 ** (c / 1200) * t + c) for c in cents) / voices


def fm_glass(freq: float, t: np.ndarray, ratio: float = 3.5, index: float = 1.6) -> np.ndarray:
    mod = index * np.exp(-t * 0.6) * np.sin(2 * np.pi * freq * ratio * t)
    return np.sin(2 * np.pi * freq * t + mod)


def kick_808(freq: float, n: int) -> np.ndarray:
    t = np.arange(n) / SR
    f = freq * (1 + 2.5 * np.exp(-t * 30))
    return np.tanh(1.6 * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 1.8))


def clap(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / SR
    env = sum(np.exp(-np.maximum(t - d, 0) * 60) * (t >= d) for d in (0.0, 0.011, 0.022)) + np.exp(-t * 14) * 0.6
    return bandpass(rng.standard_normal(n), 900, 5000) * env * 0.6


def tight_kick(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / SR
    f = 50 + 130 * np.exp(-t * 45)
    return np.tanh(2.2 * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 8)) * 0.9 + kick(n, rng) * 0.2


def chip_kick(n: int) -> np.ndarray:
    t = np.arange(n) / SR
    f = 60 + 300 * np.exp(-t * 60)
    return square(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 18)


def rim(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / SR
    return (np.sin(2 * np.pi * 1700 * t) * 0.6 + bandpass(rng.standard_normal(n), 2000, 8000)) * np.exp(-t * 90)


def keystroke(rng: np.random.Generator, heavy: bool) -> np.ndarray:
    """One key: a plastic click (modern) or a typebar strike on the platen (typewriter)."""
    n = int((0.09 if heavy else 0.05) * SR)
    t = np.arange(n) / SR
    click = highpass(rng.standard_normal(n), 2500 if not heavy else 1200) * np.exp(-t * (180 if not heavy else 90))
    thock = np.sin(2 * np.pi * rng.uniform(180, 260) * t) * np.exp(-t * (70 if not heavy else 35))
    clack = np.zeros(n)
    if heavy:  # the typebar hitting the platen a few ms later
        d = int(0.012 * SR)
        clack[d:] = bandpass(rng.standard_normal(n - d), 800, 4000) * np.exp(-t[: n - d] * 120)
    return click * (0.7 if not heavy else 0.9) + thock * 0.35 + clack * 0.8


def typing_track(total: int, duration: float, rng: np.random.Generator, heavy: bool) -> np.ndarray:
    """Bursts of typing with human timing, pauses and, on a typewriter, the carriage-return bell."""
    out = np.zeros((total, 2))
    ts = rng.uniform(6, 14)
    while ts < duration - 8:
        n_keys = int(rng.integers(12, 60))
        rate = rng.uniform(6, 11)
        col = 0
        for _ in range(n_keys):
            k = keystroke(rng, heavy) * rng.uniform(0.6, 1.0)
            pan = 0.42 + 0.16 * np.tanh((col - 30) / 20) if heavy else rng.uniform(0.4, 0.6)
            s = int(ts * SR)
            m = min(len(k), total - s)
            if m <= 0:
                break
            out[s : s + m, 0] += k[:m] * (1 - pan)
            out[s : s + m, 1] += k[:m] * pan
            ts += rng.gamma(4, 1 / (4 * rate)) + (rng.uniform(0.25, 0.6) if rng.random() < 0.08 else 0)
            col += 1
            if heavy and col >= 55:  # end of line: ding, then the carriage slides back
                n = int(1.2 * SR)
                t = np.arange(n) / SR
                ding = bell(2093, n) * 0.18
                slide = bandpass(rng.standard_normal(n), 300, 2500) * np.clip(t / 0.05, 0, 1) * (t < 0.45) * 0.25
                s = int(ts * SR)
                m = min(n, total - s)
                if m > 0:
                    out[s : s + m] += np.stack([ding + slide, ding * 0.8 + slide], axis=1)[:m]
                ts += 0.6
                col = 0
        ts += rng.uniform(6, 22)
    return out


# ------------------------------------------------------------------ form

ALL = {"drums", "bass", "pad", "arp", "lead"}
FORMS = {
    "club": [("intro", 8, {"pad", "arp"}), ("verse", 16, {"drums", "bass", "pad", "arp"}), ("drop", 16, ALL - {"lead"}),
             ("break", 8, {"pad", "lead"}), ("drop", 16, ALL)],
    "anthem": [("intro", 4, {"pad"}), ("verse", 8, {"light", "bass", "pad", "lead"}), ("chorus", 8, ALL),
               ("verse", 8, {"light", "bass", "arp", "lead"}), ("chorus", 8, ALL), ("bridge", 8, {"pad", "arp"}),
               ("chorus", 12, ALL)],
    "build": [("intro", 8, {"pad"}), ("rise", 16, {"pad", "arp", "light"}), ("rise", 12, {"pad", "arp", "light", "bass"}),
              ("drop", 24, ALL), ("fall", 8, {"pad", "lead", "arp"})],
    "ambient": [("intro", 8, {"pad"}), ("a", 16, {"pad", "arp", "bass"}), ("b", 16, {"pad", "lead", "arp"}),
                ("a", 16, {"pad", "arp", "bass", "lead", "light"})],
    "loop": [("l1", 8, {"pad", "bass"}), ("l2", 8, {"drums", "bass", "arp"}), ("l3", 8, {"drums", "pad", "lead"}),
             ("l4", 8, ALL), ("l5", 8, {"light", "bass", "lead"}), ("l6", 8, ALL - {"arp"}), ("l7", 8, ALL)],
    "slowburn": [("s1", 12, {"pad"}), ("s2", 12, {"pad", "bass"}), ("s3", 12, {"pad", "bass", "light"}),
                 ("s4", 12, {"pad", "bass", "light", "arp"}), ("s5", 16, ALL)],
    "pulse": [("p1", 8, {"drums", "bass"}), ("p2", 8, {"drums", "bass", "lead"}), ("p3", 8, {"drums", "bass", "arp"}),
              ("p4", 8, {"drums", "bass", "pad", "lead"}), ("p5", 8, {"light", "pad", "arp"}), ("p6", 8, ALL)],
}  # fmt: skip


def plan(form: str, n_bars: int) -> list[tuple[str, set[str]]]:
    out: list[tuple[str, set[str]]] = []
    blocks = FORMS[form]
    last_full = next(b for b in reversed(blocks) if len(b[2]) >= 4)
    for name, bars, roles in blocks:
        out += [(name, roles)] * bars
    body = n_bars - 6
    out = out[:body] + [(last_full[0], last_full[2])] * max(body - len(out), 0)
    return out + [("outro", {"pad"})] * (n_bars - len(out))


# ------------------------------------------------------------------ composition


def synthesize(spec: Spec) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(spec.seed)
    total = int(spec.duration * SR)
    s16 = 60 / spec.bpm / 4
    bar = 16 * s16
    n_bars = math.ceil(spec.duration / bar)
    sections = plan(spec.form, n_bars)
    progress = np.clip(np.arange(n_bars) / max(n_bars - 8, 1), 0, 1)

    def at(b: int, i: float) -> float:  # time of sixteenth i in bar b, with swing on the off-beats
        return b * bar + (i + (spec.swing if int(i) % 2 else 0.0)) * s16

    def put(buf, s, x, gain=1.0):
        s = int(s)
        n = min(len(x), total - s)
        if n > 0 and s >= 0:
            buf[s : s + n] += x[:n] * gain

    def put2(buf, s, x, gain, pan):
        put(buf[:, 0], s, x, gain * (1 - pan))
        put(buf[:, 1], s, x, gain * pan)

    def degree_of(b: int) -> int:
        return spec.progression[(b // spec.chord_bars) % len(spec.progression)]

    def on(b: int, role: str) -> bool:
        return role in sections[b][1]

    # ---------------------------------------------------------------- drums
    drums, gated = np.zeros((total, 2)), np.zeros(total)
    kicks: list[float] = []
    gated_times: list[float] = []
    tk = tight_kick(int(0.5 * SR), rng)
    for b in range(n_bars):
        full, light = on(b, "drums"), on(b, "light")
        if spec.groove == "none" or not (full or light):
            continue
        g = spec.groove
        idm_k = tuple(i for i, h in enumerate(euclidean(int(rng.integers(3, 6)), 16, int(rng.integers(0, 16)))) if h)
        idm_g = tuple(int(x) for x in sorted(rng.choice(16, 3, replace=False)))
        k_pos, s_pos, ghosts = {
            "minimal": ((0, 8) if b % 4 != 3 else (0, 8, 14), (), ()),
            "idm": (idm_k, (4,) if b % 2 else (12,), idm_g),
            "heartbeat": ((0, 3), (), ()),
            "chip": ((0, 6, 8) if b % 2 == 0 else (0, 6, 10, 14), (4, 12), ()),
            "four": ((0, 4, 8, 12), (4, 12), ()),
            "halftime": ((0, 10), (8,), (14,)),
            "break": ((0, 6, 10) if b % 2 == 0 else (0, 3, 10, 11), (4, 12), (7, 15)),
            "dnb": ((0, 10) if b % 2 == 0 else (0, 7, 10), (4, 12), (14,) if b % 2 else (9,)),
            "trap": ((0, 7, 11) if b % 2 == 0 else (0, 3, 10), (8,), ()),
            "electro": ((0, 3, 6, 10, 14) if b % 2 == 0 else (0, 3, 10), (4, 12), ()),
            "downtempo": ((0, 9) if b % 2 == 0 else (0, 6, 9), (8,), (15,)),
        }[g]
        if light:
            k_pos, s_pos, ghosts = (0,), (), ()
        for i in k_pos:
            t0 = at(b, i)
            if g == "trap":
                put2(drums, t0 * SR, kick_808(midi_to_hz(chord(spec, degree_of(b))[0] - 12), int(0.9 * SR)), 0.5, 0.5)
            elif g == "heartbeat":
                put2(drums, t0 * SR, lowpass(tk, 180), 0.45 if i == 0 else 0.3, 0.5)
            elif g == "chip":
                put2(drums, t0 * SR, chip_kick(int(0.18 * SR)), 0.35, 0.5)
            else:
                put2(drums, t0 * SR, tk, 0.6, 0.5)
            kicks.append(t0)
        for i in s_pos:
            if g == "chip":
                hit = rng.standard_normal(int(0.09 * SR)) * np.exp(-np.arange(int(0.09 * SR)) / SR * 40) * 0.6
            else:
                hit = clap(int(0.35 * SR), rng) if g in ("trap", "electro") else snare(int(0.4 * SR), rng)
            put2(drums, at(b, i) * SR, hit, 0.3, 0.5)
            if g in ("four", "halftime", "electro"):
                put(gated, at(b, i) * SR, hit, 1.0)
                gated_times.append(at(b, i))
        for i in ghosts:
            put2(drums, at(b, i) * SR, snare(int(0.2 * SR), rng), 0.08, 0.55)
        # hats: a different figure per groove
        if g == "trap":
            steps = list(range(0, 16, 2))
            if b % 4 == 3:  # thirty-second triplet roll into the next bar
                steps += [12 + k * 4 / 6 for k in range(6)]
        elif g in ("halftime", "downtempo"):
            steps = list(range(0, 16, 2))
        elif g == "dnb":
            steps = [i for i in range(16) if i % 4 != 0]
        elif g == "heartbeat":
            steps = []
        elif g == "chip":
            steps = list(range(0, 16, 2))
        elif g == "idm":
            steps = [i for i in range(16) if rng.random() < 0.55] + [i + 0.5 for i in range(16) if rng.random() < 0.12]
        else:
            steps = list(range(16))
        for i in steps:
            accent = 1.8 if (int(i) % 4 == 2) else 1.0
            put2(drums, at(b, i) * SR, hat(int(0.07 * SR), rng), 0.02 * accent, 0.35 if int(i) % 2 else 0.65)
        if g == "four" and full and b % 2:
            for i in (2, 6, 10, 14):
                n = int(0.3 * SR)
                oh = highpass(rng.standard_normal(n), 6000) * np.exp(-np.arange(n) / SR * 12)
                put2(drums, at(b, i) * SR, oh, 0.03, 0.5)
        if g == "minimal":  # rimshot clicks carry the groove instead of a snare
            for i in ((3, 11) if b % 2 == 0 else (3, 6, 11, 14)):
                put2(drums, at(b, i) * SR, rim(int(0.06 * SR), rng), 0.12, 0.6 if i % 2 else 0.4)
        if g == "electro":  # cowbell-like square pair on the syncopation
            for i in (5, 13):
                n = int(0.12 * SR)
                tt = np.arange(n) / SR
                cb = (square(2 * np.pi * 560 * tt) + square(2 * np.pi * 845 * tt)) * np.exp(-tt * 30)
                put2(drums, at(b, i) * SR, bandpass(cb, 500, 3000), 0.03, 0.7)

    if gated.any():  # the 80s snare: big hall chopped after a quarter second
        g_wet = reverb(np.stack([gated, gated], axis=1), 2.5, rng)
        gate = np.zeros(total)
        for gt in gated_times:
            gate[int(gt * SR) : int((gt + 0.25) * SR)] = 1.0
        drums += g_wet * lowpass(gate, 60)[:, None] * 0.5

    duck = np.ones(total)
    if spec.pump > 0:
        for kt in kicks:
            s = int(kt * SR)
            n = min(int(0.45 * SR), total - s)
            duck[s : s + n] = np.minimum(duck[s : s + n], 1 - 0.7 * spec.pump * np.exp(-np.arange(n) / SR / 0.12))

    # ---------------------------------------------------------------- bass
    low = np.zeros(total)
    if spec.bass != "none":
        acid_cut = 400.0
        for b in range(n_bars):
            if not on(b, "bass"):
                continue
            root = chord(spec, degree_of(b))[0] - 12
            kind = spec.bass
            if kind in ("rolling", "pulse"):
                step = 1 if kind == "rolling" else 2
                for i in range(0, 16, step):
                    note = root + (12 if (i // step) % 2 else 0)
                    n = int(step * s16 * 0.9 * SR)
                    tt = np.arange(n) / SR
                    f = midi_to_hz(note)
                    x = sawtooth(2 * np.pi * f * tt) + 0.7 * np.sin(2 * np.pi * f * tt)
                    env = np.minimum(tt / 0.003, 1) * np.exp(-tt * (9 if step == 1 else 5))
                    put(low, at(b, i) * SR, lowpass(x * env, 900), 0.5)
            elif kind == "reese":
                n = int(bar * SR)
                tt = np.arange(n) / SR
                f = midi_to_hz(root)
                x = sawtooth(2 * np.pi * f * tt) + sawtooth(2 * np.pi * f * 1.012 * tt) + 0.8 * np.sin(2 * np.pi * f * tt)
                put(low, b * bar * SR, np.tanh(1.8 * lowpass(x, 380 + 220 * (b % 4) / 3) * _env(n, 0.02, 0.15)), 0.42)
            elif kind == "fm":
                for i in [j for j, h in enumerate(euclidean(5, 8, 1)) if h]:
                    n = int(2 * s16 * 0.95 * SR)
                    tt = np.arange(n) / SR
                    f = midi_to_hz(root + (12 if i in (3, 6) else 0))
                    x = np.sin(2 * np.pi * f * tt + (1 + 3.5 * np.exp(-tt * 6)) * np.sin(2 * np.pi * f * tt))
                    put(low, at(b, 2 * i) * SR, np.tanh(2 * x) * np.minimum(tt / 0.004, 1) * np.exp(-tt * 4), 0.5)
            elif kind == "acid":
                pattern = euclidean(11, 16, b % 3)
                line = (0, 0, 12, 0, 3, 0, 7, 10, 0, 12, 0, 5, 0, 15, 7, 0)
                for i, h in enumerate(pattern):
                    if not h:
                        continue
                    accent = i in (0, 6, 11)
                    acid_cut = 0.85 * acid_cut + 0.15 * (300 + 1800 * progress[b] * (1.6 if accent else 1.0))
                    n = int(s16 * 0.95 * SR)
                    tt = np.arange(n) / SR
                    x = sawtooth(2 * np.pi * midi_to_hz(root + line[i]) * tt)
                    x = resonant_lowpass(x, acid_cut * 2.0, 7.0 if accent else 4.5)
                    put(low, at(b, i) * SR, np.tanh(1.6 * x * np.exp(-tt * 10)), 0.32 if accent else 0.24)
            elif kind == "seq":  # Berlin-school ostinato through a slowly sweeping resonant filter
                line = (0, 12, 7, 0, 10, 12, 0, 7, 0, 12, 5, 0, 10, 7, 12, 3)
                for i in range(16):
                    n = int(s16 * 0.9 * SR)
                    tt = np.arange(n) / SR
                    x = sawtooth(2 * np.pi * midi_to_hz(root + line[i]) * tt) * np.exp(-tt * 8)
                    cut = 350 + 1300 * (0.5 + 0.5 * np.sin(2 * np.pi * (b * 16 + i) / 128))
                    put(low, at(b, i) * SR, np.tanh(1.3 * resonant_lowpass(x, cut, 3.0)), 0.36)
            elif kind == "808":
                hits = ((0, 0), (7, 0), (10, 12)) if b % 2 else ((0, 0), (7, 0), (11, 0))
                for h, (i, oct_) in enumerate(hits):
                    n = int(((hits[h + 1][0] if h + 1 < len(hits) else 16) - i) * s16 * SR)
                    tt = np.arange(n) / SR
                    f0, f1 = midi_to_hz(root - 12 + oct_), midi_to_hz(root - 12)
                    f = f1 + (f0 - f1) * np.exp(-tt * 18)  # glide
                    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 1.4)
                    put(low, at(b, i) * SR, np.tanh(2.2 * x), 0.5)
            # every bass sits on a sub
            n = int(bar * SR)
            tt = np.arange(n) / SR
            put(low, b * bar * SR, np.sin(2 * np.pi * midi_to_hz(root - 12) * tt) * _env(n, 0.01, 0.05),
                0.3 if kind == "sub" else 0.18)  # fmt: skip

    wide, dry = np.zeros((total, 2)), np.zeros((total, 2))

    # ---------------------------------------------------------------- pad
    if spec.pad != "none":
        for b0 in range(0, n_bars, spec.chord_bars):
            if not on(b0, "pad"):
                continue
            n = int((spec.chord_bars * bar + 1.5) * SR)
            tt = np.arange(n) / SR
            notes = [m + 12 for m in chord(spec, degree_of(b0))]
            if spec.pad == "supersaw":
                x = lowpass(sum(supersaw(midi_to_hz(m), tt) for m in notes) / len(notes), 1400 + 1200 * progress[b0])
                gain = 0.11
            elif spec.pad == "glass":
                x = sum(fm_glass(midi_to_hz(m + 12), tt) for m in notes) / len(notes)
                gain = 0.10
            elif spec.pad == "choir":
                x = choir([midi_to_hz(m) for m in notes], n, rng, FORMANTS_AH if (b0 // spec.chord_bars) % 2 else FORMANTS_OH)
                gain = 0.5
            elif spec.pad == "drone":  # root and fifth, slow beating, a breath of filtered noise
                r0 = midi_to_hz(notes[0] - 12)
                x = (np.sin(2 * np.pi * r0 * tt) + np.sin(2 * np.pi * r0 * 1.003 * tt)
                     + 0.6 * np.sin(2 * np.pi * r0 * 1.5 * tt) + 0.6 * np.sin(2 * np.pi * r0 * 1.497 * tt))
                x = x / 3.2 + bandpass(rng.standard_normal(n), 200, 900) * 0.25
                gain = 0.2
            else:  # strings
                x = strings([midi_to_hz(m + 12) for m in notes], n)
                gain = 0.16
            x = x * _env(n, 0.8, 1.2) * gain
            s = int(b0 * bar * SR)
            m = min(n, total - s)
            if m > 0:
                wide[s : s + m] += np.stack([x, np.roll(x, int(0.011 * SR))], axis=1)[:m]

    # ---------------------------------------------------------------- arpeggio
    if spec.arp != "none":
        arp = np.zeros((total, 2))
        k = 0
        per_bar = {8: 8, 12: 12, 16: 16}[spec.arp_rate]
        for b in range(n_bars):
            if not on(b, "arp"):
                continue
            notes = chord(spec, degree_of(b))
            line = [m + 12 for m in notes] + [m + 24 for m in notes]
            if spec.arp_shape == "updown":
                line = line + line[-2:0:-1]
            elif spec.arp_shape == "pendulum":
                line = [line[i] for i in (0, 3, 1, 4, 2, 5) if i < len(line)]
            for j in range(per_bar):
                i = j * 16 / per_bar
                note = line[rng.integers(len(line))] if spec.arp_shape == "random" else line[k % len(line)]
                k += 1
                n = int(16 / per_bar * s16 * 1.8 * SR)
                tt = np.arange(n) / SR
                f = midi_to_hz(note)
                if spec.arp == "pluck":
                    x = ks_pluck(f, n, rng, 0.994) * 0.6
                elif spec.arp == "bell":
                    x = bell(f, n) * 0.35
                elif spec.arp == "chip":
                    x = square(2 * np.pi * f * tt, 0.25) * np.exp(-tt * 20) * 0.5
                else:
                    x = (square(2 * np.pi * f * tt, 0.3) * 0.6 + sawtooth(2 * np.pi * f * tt) * 0.4) * np.exp(-tt * 14)
                    x = lowpass(x, 900 + 3600 * progress[b])
                put2(arp, at(b, i) * SR, x * np.minimum(tt / 0.002, 1), 0.07, 0.5 + 0.25 * np.sin(k * 0.9))
        d = int(3 * s16 * SR)  # dotted-eighth ping-pong
        echo = np.zeros_like(arp)
        for r in range(1, 4):
            echo[r * d :, r % 2] += lowpass(arp[: total - r * d, 1 - r % 2], 3000) * 0.4**r
        dry += arp + echo

    # ---------------------------------------------------------------- lead
    if spec.lead != "none":
        lead = np.zeros((total, 2))
        motifs = []
        for _ in range(3):  # phrases A, B, C of two bars each
            hits = [i for i, h in enumerate(euclidean(spec.lead_density, 32, int(rng.integers(0, 5)))) if h]
            walk = np.cumsum(rng.choice([-2, -1, 1, 1, 2, 3, -3], len(hits)))
            motifs.append(list(zip(hits, walk - walk[0] + int(rng.integers(2, 6)))))
        order = (0, 1, 0, 2)  # A B A C
        for b in range(0, n_bars, 2):
            if not on(b, "lead"):
                continue
            motif = motifs[order[(b // 2) % 4]]
            chord_now = chord(spec, degree_of(b))
            prev_f = None
            for j, (i, dg) in enumerate(motif):
                nxt = motif[j + 1][0] if j + 1 < len(motif) else 32
                note = scale_note(spec, dg, spec.tonic + 24)
                if i % 8 == 0:
                    note = min((c + 24 + 12 * o for c in chord_now for o in (0, 1)), key=lambda c: abs(c - note))
                n = int((nxt - i) * s16 * 0.95 * SR)
                tt = np.arange(n) / SR
                f_t = midi_to_hz(note)
                kind = spec.lead
                if kind == "saw":  # legato glide from the previous note
                    f0 = prev_f or f_t
                    f = f_t + (f0 - f_t) * np.exp(-tt * 25)
                    x = lowpass(supersaw(1.0, np.cumsum(f) / SR, 3, 10), 3000) * _env(n, 0.02, 0.1)
                    g = 0.075
                elif kind == "bell":
                    x, g = bell(f_t, n), 0.05
                elif kind == "pluck":
                    x, g = ks_pluck(f_t, n, rng, 0.997), 0.11
                elif kind == "acid":
                    x = resonant_lowpass(sawtooth(2 * np.pi * f_t * tt), 1200 + 1500 * progress[b], 6.0)
                    x, g = np.tanh(1.4 * x) * np.exp(-tt * 3), 0.05
                elif kind == "sine":  # pure, slightly detuned, slow vibrato: eerie
                    vib = 1 + 0.006 * np.sin(2 * np.pi * 4.2 * tt) * np.minimum(tt / 0.5, 1)
                    x = np.sin(2 * np.pi * np.cumsum(f_t * vib) / SR) * _env(n, 0.08, 0.2)
                    g = 0.09
                elif kind == "chip":  # 12.5% pulse with a fast chord arpeggio, the 8-bit trick
                    tri = np.array(chord_now)[(np.floor(tt * 30).astype(int)) % len(chord_now)] - chord_now[0]
                    f = f_t * 2 ** (tri / 12)
                    x = square(2 * np.pi * np.cumsum(f) / SR, 0.125) * _env(n, 0.005, 0.05)
                    g = 0.05
                elif kind == "vox":
                    x = choir([f_t], n, rng, FORMANTS_AH if j % 2 else FORMANTS_OH) * _env(n, 0.05, 0.15)
                    g = 0.35
                else:  # square with vibrato and PWM
                    f = f_t * (1 + 0.004 * np.sin(2 * np.pi * 5.5 * tt) * np.minimum(tt / 0.3, 1))
                    ph = 2 * np.pi * np.cumsum(f) / SR
                    x = lowpass((square(ph, 0.5 + 0.2 * np.sin(2 * np.pi * 0.7 * tt)) * 0.55 + np.sin(ph) * 0.45) * _env(n, 0.01, 0.12), 3200)
                    g = 0.075
                prev_f = f_t
                put2(lead, at(b, i) * SR, x, g, 0.48)
        d = int(6 * s16 * SR)
        tail = np.zeros_like(lead)
        for r in range(1, 4):
            tail[r * d :] += lowpass(lead[: total - r * d][:, ::-1].T, 2500).T * 0.36**r
        dry += lead + tail

    # ---------------------------------------------------------------- texture
    tex = np.zeros((total, 2))
    T = set(spec.texture)
    if "rain" in T:
        drops = (rng.random((total, 2)) < 0.0004) * rng.uniform(0.2, 1.0, (total, 2))
        tex += bandpass(rng.standard_normal((2, total)), 900, 7000).T * 0.009 + highpass(drops.T, 2500).T * 0.05
    if "city" in T:
        rumble = np.cumsum(rng.standard_normal((2, total)), axis=1)
        rumble = lowpass(rumble - lowpass(rumble, 20), 180).T
        tex += rumble / (np.abs(rumble).max() + 1e-9) * 0.05
        for _ in range(int(spec.duration / 25)):
            ts = rng.uniform(5, spec.duration - 5)
            n = int(rng.uniform(0.4, 0.9) * SR)
            tt = np.arange(n) / SR
            horn = (sawtooth(2 * np.pi * 311 * tt) + sawtooth(2 * np.pi * 392 * tt)) * _env(n, 0.05, 0.2)
            put2(tex, ts * SR, lowpass(horn, 900), 0.012, rng.uniform(0.2, 0.8))
    if "radio" in T:
        static = bandpass(rng.standard_normal((2, total)), 1500, 4000).T
        gate = lowpass((rng.random(total) < 0.00006).astype(float), 3) * 4000
        tex += static * np.clip(gate, 0, 1)[:, None] * 0.03
        ts = 6.0
        while ts < spec.duration - 6:  # Morse groups, far away
            for dur in rng.choice([0.06, 0.18], int(rng.integers(3, 7))):
                n = int(dur * SR)
                beep = np.sin(2 * np.pi * 640 * np.arange(n) / SR) * _env(n, 0.003, 0.005)
                put2(tex, ts * SR, beep, 0.012, 0.75)
                ts += dur + 0.07
            ts += rng.uniform(9, 20)
    if "vinyl" in T:
        crackle = (rng.random((total, 2)) < 0.00025) * rng.standard_normal((total, 2))
        tex += highpass(crackle.T, 1500).T * 0.12 + lowpass(rng.standard_normal((2, total)), 4000).T * 0.004

    if "typing" in T:
        tex += typing_track(total, spec.duration, rng, heavy=False) * 0.22
    if "typewriter" in T:
        tex += typing_track(total, spec.duration, rng, heavy=True) * 0.2
    if "clockwork" in T:  # escapement tick-tock and the odd ratchet of a winding gear
        for j in range(int(spec.duration * 2)):
            n = int(0.03 * SR)
            tt = np.arange(n) / SR
            tick = bandpass(rng.standard_normal(n), 2500 if j % 2 else 3400, 7000) * np.exp(-tt * 150)
            put2(tex, j * 0.5 * SR, tick, 0.05, 0.35 if j % 2 else 0.65)
            if rng.random() < 0.02:
                for r in range(int(rng.integers(6, 14))):
                    put2(tex, (j * 0.5 + 0.12 + r * 0.035) * SR, tick * 0.7, 0.04, 0.5)
    if "steam" in T:  # hiss bursts from a valve, sometimes a whistle
        for _ in range(int(spec.duration / 18)):
            ts = rng.uniform(4, spec.duration - 6)
            n = int(rng.uniform(1.2, 3.0) * SR)
            hiss = bandpass(rng.standard_normal(n), 2500, 9000) * _env(n, 0.08, 0.9)
            if rng.random() < 0.35:
                tt = np.arange(n) / SR
                hiss = hiss + np.sin(2 * np.pi * rng.uniform(900, 1500) * tt) * _env(n, 0.2, 0.6) * 0.15
            put2(tex, ts * SR, hiss, 0.035, rng.uniform(0.2, 0.8))
    if "modem" in T:  # a dial-up handshake, far in the background, two or three times
        for ts in np.sort(rng.uniform(15, spec.duration - 15, int(rng.integers(2, 4)))):
            seq = [(1300, 0.4), (2100, 0.5), (1650, 0.3), (980, 0.25)]
            t_ = ts
            for f, d in seq:
                n = int(d * SR)
                tt = np.arange(n) / SR
                put2(tex, t_ * SR, np.sin(2 * np.pi * f * tt) * _env(n, 0.01, 0.02), 0.008, 0.7)
                t_ += d
            n = int(1.6 * SR)
            screech = bandpass(rng.standard_normal(n), 1200, 3500) * (0.6 + 0.4 * square(2 * np.pi * 11 * np.arange(n) / SR))
            put2(tex, t_ * SR, screech * _env(n, 0.02, 0.2), 0.012, 0.7)
    if "servers" in T:  # mains hum and fans in a server room
        tt = np.arange(total) / SR
        hum = sum(np.sin(2 * np.pi * 60 * h * tt) / h for h in (1, 2, 3, 5))
        fans = lowpass(rng.standard_normal((2, total)), 700).T
        tex += hum[:, None] * 0.006 + fans * 0.02

    stereo = drums + (low * duck)[:, None] + wide * duck[:, None] + dry
    wet = reverb(0.4 * stereo + 0.6 * dry + 0.5 * wide, spec.reverb_s, rng)
    hiss = lowpass(rng.standard_normal((2, total)), 6000).T * 0.003
    mix = 0.8 * stereo + 0.45 * wet + tex + hiss
    if spec.stutter > 0:  # buffer repeats and bit-crushed moments, on the grid
        n_events = int(spec.stutter * spec.duration / 6) if n_bars > 10 else 0
        for _ in range(n_events):
            b = int(rng.integers(4, n_bars - 4))
            i = int(rng.integers(0, 4)) * 4
            s = int(at(b, i) * SR)
            if rng.random() < 0.6:
                sl = int(s16 / (2 if rng.random() < 0.5 else 1) * SR)
                reps = int(rng.integers(3, 7))
                piece = mix[s : s + sl].copy()
                for r in range(1, reps):
                    a = s + r * sl
                    m = min(sl, total - a)
                    if m > 0:
                        mix[a : a + m] = piece[:m] * (1 - 0.08 * r)
            else:
                n = int(4 * s16 * SR)
                seg = mix[s : s + n]
                held = np.repeat(seg[::8], 8, axis=0)[: len(seg)]
                mix[s : s + len(seg)] = np.round(held * 24) / 24
    mix *= _env(total, 3.0, 6.0)[:, None]
    mix = np.tanh(1.15 * mix)
    mix /= np.max(np.abs(mix)) + 1e-9
    mix *= 10 ** (-1.0 / 20)
    return mix.astype(np.float32), analyze(mix, low * duck, kicks, spec, bar)


def analyze(mix, bass, kicks, spec: Spec, bar: float) -> dict:
    hop = SR // spec.fps
    n_frames = len(mix) // hop
    mono = mix.mean(axis=1)

    def env_of(x, smooth):
        r = np.sqrt(np.array([np.mean(x[i * hop : (i + 1) * hop] ** 2) for i in range(n_frames)]))
        k = np.exp(-np.arange(-30, 31) ** 2 / (2 * smooth**2))
        r = np.convolve(r, k / k.sum(), mode="same")
        return [round(float(v), 4) for v in (r - r.min()) / (r.max() - r.min() + 1e-9)]

    kick_env = np.zeros(n_frames)
    for kt in kicks:
        f0 = int(kt * spec.fps)
        for j in range(f0, min(f0 + 10, n_frames)):
            kick_env[j] = max(kick_env[j], np.exp(-(j - f0) / 3))
    return {
        "fps": spec.fps,
        "duration": spec.duration,
        "chord_seconds": bar * spec.chord_bars,
        "rms": env_of(mono, 8),
        "bass": env_of(bass, 3),
        "kick": [round(float(v), 4) for v in kick_env],
    }


def render_track(spec: Spec, wav_path: Path, analysis_path: Path) -> None:
    mix, analysis = synthesize(spec)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(wav_path, SR, mix)
    analysis_path.write_text(json.dumps(analysis))
