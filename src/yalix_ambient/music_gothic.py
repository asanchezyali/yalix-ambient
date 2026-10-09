"""Gothic trap: a harpsichord in a crypt over a distorted 808 (the genres, never anyone's songs).

Every sound here is new to the channel. The recipe comes from baroque trap, phonk and witch
house:
- a harpsichord (additive strings plucked near the end, a brighter 4' stop an octave up, the
  thump of the jack) playing broken-chord figures that repeat bar after bar, the baroque ostinato;
- a phonk cowbell: two square waves a fifth-and-a-bit apart (the 808 cowbell), tuned to the
  scale and saturated, playing a syncopated two-bar hook, four times with the fourth varied;
- a large distorted 808 that is also the kick: long tail, a pitch drop on the attack, glides
  between notes;
- a theremin: a sine with a slow portamento and a vibrato that widens as the note holds;
- half-time trap drums: a clap layered on a snare, preceded by a reversed swell; dark hi-hats
  with triplet and rising rolls;
- witch house textures: tape stops into the breaks (the whole mix slows to a halt), reversed
  harpsichord chords, vinyl crackle, cassette wow and whispers (noise through moving formants);
- a crypt instead of a reverb: long pre-delay, dark tail.

Harmony is gothic on purpose: harmonic minor, the Andalusian descent i - VII - VI - V and the
lament bass. Every theme follows the same arc: crackle and harpsichord alone, drums halfway into
the first part, the hook, a tape stop into the break, a build of hat rolls and a reversed riser,
the drop around 2:00, the first part again and an outro that slows down like a dying tape.

Styles:
- baroque: harpsichord counterpoint in sixteenths, 808 and claps, cowbell at the drop;
- phonk: the cowbell hook from the start, a busy 808, open harpsichord chords;
- witch: dragged and hazy: theremin, whispers, reversed chords, sparse drums;
- requiem: slow lament bass, harpsichord in eighths, theremin at the drop;
- crypt: minimal: 808, hats, crackle and a few harpsichord notes;
- masquerade: triplet feel, a waltzing harpsichord, cowbell answers.
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

# Chords as (root in semitones above the tonic, quality); m minor, M major.
PROGRESSIONS = {
    "andalusian": ((0, "m"), (10, "M"), (8, "M"), (7, "M")),  # i VII VI V
    "lament": ((0, "m"), (8, "M"), (5, "m"), (7, "M")),  # i VI iv V
    "descent": ((0, "m"), (8, "M"), (3, "M"), (10, "M")),  # i VI III VII
    "cadence": ((0, "m"), (5, "m"), (7, "M"), (0, "m")),  # i iv V i
    "drone": ((0, "m"), (1, "M")),  # i bII, the Phrygian shadow
}
HARMONIC = (0, 2, 3, 5, 7, 8, 11)

FORM = (("intro", 4), ("a", 8), ("a2", 8), ("break", 4), ("build", 4), ("b", 12), ("a", 6), ("outro", 4))
HEAVY = ("a", "a2", "b")

STYLES = {  # harpsichord figure, 808 pattern, hats, layers
    "baroque": ("counter", "steady", "16", ("cowbell_b", "reverse")),
    "phonk": ("chords", "busy", "16", ("cowbell", "reverse")),
    "witch": ("broken", "sparse", "8", ("theremin", "whisper", "reverse")),
    "requiem": ("lament", "steady", "8", ("theremin_b", "reverse")),
    "crypt": ("sparse", "busy", "16", ("whisper",)),
    "masquerade": ("waltz", "steady", "trip", ("cowbell_b", "theremin_b")),
}


@dataclass
class Spec:
    duration: float = 194.159
    bpm: float = 140.0  # trap tempo; the half-time snare makes it feel like 70
    tonic: int = 38  # D2
    progression: str = "andalusian"
    chord_bars: int = 1
    style: str = "baroque"
    extra: tuple[str, ...] = ()
    swing: float = 0.0
    seed: int = 1
    fps: int = 30

    @property
    def step(self) -> float:
        return 60 / self.bpm / 4


# ------------------------------------------------------------------ instruments


def harpsichord(freq: float, n: int, rng: np.random.Generator, bright: float = 1.0) -> np.ndarray:
    """A plucked string, additive: the pluck point near the end leaves every harmonic, the high
    ones die first; a 4' stop an octave up and the thump of the jack."""
    t = np.arange(n) / SR
    x = np.zeros(n)
    pluck = rng.uniform(0.1, 0.14)
    for stop, gain in ((1.0, 1.0), (2.0, 0.35 * bright)):
        f0 = freq * stop * 2 ** (rng.normal(0, 3) / 1200)
        for k in range(1, 26):
            fk = f0 * k * (1 + 0.0004 * k * k)  # a little stiffness
            if fk > 7000:
                break
            a = abs(np.sin(np.pi * k * pluck)) / k**0.9
            x += gain * a * np.sin(2 * np.pi * fk * t + rng.random() * 6) * np.exp(-t * (1.5 + 0.9 * k))
    jack = bandpass(rng.standard_normal(n), 300, 2500) * np.exp(-t / 0.006) * 0.15
    return lowpass(x + jack, 5000) * _env(n, 0.001, 0.03) * 0.3


def cowbell(freq: float, n: int) -> np.ndarray:
    """The 808 cowbell: two square waves at a ratio of 1.48, band-passed, driven."""
    t = np.arange(n) / SR
    x = np.sign(np.sin(2 * np.pi * freq * t)) + np.sign(np.sin(2 * np.pi * freq * 1.48 * t))
    x = bandpass(x, freq * 0.8, freq * 4)
    env = np.exp(-t / 0.05) * 0.6 + np.exp(-t / 0.35) * 0.4
    return lowpass(np.tanh(2.5 * x * env), 3500) * _env(n, 0.001, 0.02) * 0.4


def bass808(notes: list[tuple[float, float, float, bool]], total: int) -> np.ndarray:
    """One continuous 808: (start, duration, midi, glide). Each note restarts the envelope with a
    pitch drop on the attack; a glide note slides from the previous pitch."""
    f = np.zeros(total)
    env = np.zeros(total)
    prev = None
    for ts, dur, m, glide in notes:
        s, e = int(ts * SR), min(int((ts + dur) * SR), total)
        if s >= total:
            continue
        k = np.arange(e - s) / SR
        target = midi_to_hz(m)
        release = np.clip((e - s - np.arange(e - s)) / (0.02 * SR), 0, 1)
        if glide and prev is not None:  # the same note keeps ringing while its pitch slides
            g = np.clip(k / 0.09, 0, 1)
            f[s:e] = prev * (target / prev) ** (g * g * (3 - 2 * g))
            level = env[s - 1] if s > 0 and env[s - 1] > 0 else 0.6
            env[s:e] = level * np.exp(-k / max(dur * 0.9, 0.3)) * release
        else:
            f[s:e] = target * (1 + 0.6 * np.exp(-k / 0.018))  # a fresh hit: a pitch drop on the attack
            env[s:e] = np.exp(-k / max(dur * 0.9, 0.3)) * release
        prev = target
    ph = np.cumsum(f) / SR
    x = (np.sin(2 * np.pi * ph) + 0.25 * np.sin(4 * np.pi * ph)) * env
    growl = lowpass(np.tanh(3.5 * x), 1400)
    click = np.zeros(total)
    for ts, _, _, glide in notes:
        if not glide and int(ts * SR) < total:
            s = int(ts * SR)
            m = min(int(0.004 * SR), total - s)
            click[s : s + m] += np.hanning(2 * m)[:m] * 0.3
    return 0.55 * growl + 0.45 * np.tanh(1.5 * x) + lowpass(click, 3000)


def clap(n: int, rng: np.random.Generator) -> np.ndarray:
    """A clap of four hands 9 ms apart on a short snare body."""
    t = np.arange(n) / SR
    x = np.zeros(n)
    for j in range(4):
        s = int(j * 0.009 * SR)
        tail = 0.012 if j < 3 else 0.12
        x[s:] += bandpass(rng.standard_normal(n - s), 900, 4500) * np.exp(-t[: n - s] / tail)
    body = np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.04) * 0.6
    return lowpass(x * 0.5 + body, 6000) * 0.5


def hat(n: int, rng: np.random.Generator, tone: float = 1.0) -> np.ndarray:
    t = np.arange(n) / SR
    x = bandpass(rng.standard_normal(n), 4500 * tone, 9000)
    return x * np.exp(-t / 0.025) * 0.25


def theremin(notes: list[tuple[float, float, float]], total: int) -> np.ndarray:
    """A sine with a slow portamento and a vibrato that widens as the note holds."""
    if not notes:
        return np.zeros(total)
    f = np.zeros(total)
    amp = np.zeros(total)
    held = np.zeros(total)
    for ts, dur, m in notes:
        s, e = int(ts * SR), min(int((ts + dur) * SR), total)
        f[s:e] = midi_to_hz(m)
        amp[s:e] = 1
        held[s:e] = np.arange(e - s) / SR
    idx = np.maximum.accumulate(np.where(f > 0, np.arange(total), 0))
    f = f[idx]
    f[f == 0] = midi_to_hz(notes[0][2])
    a = np.exp(-1 / (0.12 * SR))
    f = lfilter([1 - a], [1, -a], f, zi=[f[0] * a])[0]  # 120 ms portamento
    t = np.arange(total) / SR
    depth = 0.004 + 0.012 * np.clip(held / 1.5, 0, 1)
    f *= 1 + depth * np.sin(2 * np.pi * 5.8 * t)
    ph = np.cumsum(f) / SR
    x = np.sin(2 * np.pi * ph) + 0.12 * np.sin(4 * np.pi * ph) + 0.05 * np.sin(6 * np.pi * ph)
    g = np.exp(-1 / (0.06 * SR))
    return x * lfilter([1 - g], [1, -g], amp)


def whisper(n: int, rng: np.random.Generator) -> np.ndarray:
    """Breath through a mouth that keeps changing shape: noise through two moving formants."""
    t = np.arange(n) / SR
    noise = rng.standard_normal(n)
    out = np.zeros(n)
    blk = 2048
    f1s = 400 + 300 * np.sin(2 * np.pi * t / rng.uniform(1.5, 3) + rng.random() * 6)
    f2s = 1300 + 700 * np.sin(2 * np.pi * t / rng.uniform(1, 2.2) + rng.random() * 6)
    for s in range(0, n, blk):
        seg = noise[s : s + blk + 512]
        f1, f2 = f1s[s], f2s[s]
        y = bandpass(seg, f1 * 0.8, f1 * 1.25) + 0.7 * bandpass(seg, f2 * 0.85, f2 * 1.2)
        out[s : s + blk] = y[: min(blk, n - s)]
    return out * _env(n, 0.4, 0.6) * (0.6 + 0.4 * np.sin(2 * np.pi * t / 0.9) ** 2)


def crackle(total: int, rng: np.random.Generator) -> np.ndarray:
    """Vinyl: sparse clicks of random size and a soft hiss."""
    x = lowpass(rng.standard_normal(total), 3000) * 0.004
    k = rng.poisson(6 * total / SR)
    pos = rng.integers(0, total - 200, k)
    amp = rng.pareto(3, k) * 0.03
    for p, a in zip(pos, amp):
        x[p : p + 40] += a * np.exp(-np.arange(40) / 6) * rng.choice([-1, 1])
    return lowpass(x, 5000)


def crypt(x: np.ndarray, rng: np.random.Generator, length: float = 3.2) -> np.ndarray:
    """A stone crypt: 40 ms of pre-delay, then a long, dark tail. x is (n, 2)."""
    n = int(length * SR)
    t = np.arange(n) / SR
    out = np.zeros_like(x)
    pre = int(0.04 * SR)
    for ch in range(2):
        tail = rng.standard_normal(n) * np.exp(-t / (length / 6.9)) * np.clip(t / 0.08, 0, 1)
        dark = lowpass(tail, 2600) * (1 - np.clip(t / length, 0, 1)) + lowpass(tail, 900) * np.clip(t / length, 0, 1)
        ir = np.concatenate([np.zeros(pre), dark * 0.06])
        out[:, ch] = oaconvolve(x[:, ch], ir)[: len(x)]
    return out


def tape_stop(x: np.ndarray, s: int, e: int) -> None:
    """The tape slows to a halt between samples s and e (in place, x is (n, 2))."""
    n = e - s
    if n <= 0:
        return
    rate = (1 - np.arange(n) / n) ** 1.6
    pos = s + np.cumsum(rate)
    for ch in range(x.shape[1]):
        x[s:e, ch] = np.interp(pos, np.arange(len(x)), x[:, ch]) * (1 - np.arange(n) / n) ** 0.5


# ------------------------------------------------------------------ composition


def chord_tones(tonic: int, chord: tuple[int, str]) -> list[int]:
    r, q = chord
    return [tonic + r, tonic + r + (3 if q == "m" else 4), tonic + r + 7]


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


def compose_hook(rng: np.random.Generator) -> tuple[tuple[int, int], ...]:
    """A two-bar cowbell figure: (step 0..31, scale degree), syncopated, falling at the end."""
    rhythms = (
        (0, 3, 6, 8, 11, 14, 16, 19, 22, 24, 27, 30),
        (0, 3, 6, 10, 12, 16, 19, 22, 26, 28),
        (0, 2, 3, 6, 8, 10, 11, 14, 16, 18, 19, 22, 24, 28),
    )
    steps = rhythms[int(rng.integers(0, len(rhythms)))]
    shape = [0, 2, 4, 2, 3, 2, 0, -1, 0, 2, 4, 5, 4, 2]
    start = int(rng.integers(0, 3))
    return tuple((s, shape[(i + start) % len(shape)] + (0 if i < len(steps) - 2 else -1)) for i, s in enumerate(steps))


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
    figure, bass_pat, hat_pat, layers = STYLES[spec.style]
    L = set(layers) | set(spec.extra)
    prog = PROGRESSIONS[spec.progression]
    hook = compose_hook(rng)
    trip = hat_pat == "trip"

    def at(b: int, s: float) -> float:
        sw = spec.swing * st if int(s) % 2 else 0.0
        return b * bar + s * st + sw

    harp = np.zeros((total, 2))
    bell = np.zeros(total)
    snares = np.zeros(total)
    hats = np.zeros((total, 2))
    swells = np.zeros((total, 2))
    air = np.zeros((total, 2))
    bass_notes: list[tuple[float, float, float, bool]] = []
    ther_notes: list[tuple[float, float, float]] = []
    hits: list[tuple[float, float]] = []
    sec_start = [0] * n_bars
    for i in range(1, n_bars):
        sec_start[i] = sec_start[i - 1] if sections[i] == sections[i - 1] else i
    hook_count = 0
    stops: list[tuple[int, int]] = []

    cache: dict[tuple[int, float, int], np.ndarray] = {}

    def harp_note(ts: float, m: int, g: float, dur: float = 1.6) -> None:
        key = (m, dur, int(rng.integers(0, 3)))  # three plucks per string, so repeats differ
        if key not in cache:
            cache[key] = harpsichord(midi_to_hz(m), int(dur * SR), rng)
        x = cache[key]
        p = np.clip(0.5 + (m - 60) / 40, 0.2, 0.8)  # low strings left, high right, like the instrument
        put(harp[:, 0], ts * SR, x, g * (1 - p) * 2)
        put(harp[:, 1], ts * SR, x, g * p * 2)

    for b in range(n_bars):
        sec = sections[b]
        t0 = b * bar
        new_sec = b == 0 or sections[b - 1] != sec
        k_in = b - sec_start[b]
        sec_len = next((j for j, x in enumerate(sections[sec_start[b]:]) if x != sec), n_bars - sec_start[b])
        ci = (b // spec.chord_bars) % len(prog)
        tones = chord_tones(spec.tonic, prog[ci])
        root = fit(tones[0], spec.tonic - 2, spec.tonic + 9)
        nxt_root = fit(chord_tones(spec.tonic, prog[(ci + 1) % len(prog)])[0], spec.tonic - 2, spec.tonic + 9)
        heavy = sec in HEAVY
        drums_on = (heavy or sec == "build") and not (sec == "a" and sec_start[b] < n_bars / 2 and k_in < sec_len // 2)
        up = [fit(x, 60, 71) for x in tones]
        up.sort()
        mid = [fit(x, 48, 59) for x in tones]
        mid.sort()

        # ---- the harpsichord: the ostinato the whole theme hangs on
        g_h = 0.5 if sec in ("intro", "break", "outro") else 0.42
        if sec != "build" or k_in < 2:
            if figure == "counter":  # broken chord in sixteenths over a walking bass in eighths
                fig = [mid[0], up[0], up[2], up[1], up[2], up[0], up[1], up[2]] * 2
                for s, m in enumerate(fig):
                    harp_note(at(b, s), m, g_h * (1.0 if s % 4 == 0 else 0.7), 0.6)
                for s in range(0, 16, 4):
                    harp_note(at(b, s), fit(root + (0, 7, 12, 7)[s // 4], 36, 50), g_h * 0.8, 0.9)
            elif figure == "chords":  # open chords, struck and left to ring
                for s in (0, 6, 10):
                    for j, m in enumerate(up + [mid[0]]):
                        harp_note(at(b, s) + 0.008 * j, m, g_h * 0.55, 1.8)
            elif figure == "broken":  # slow arpeggio in eighths, up and back
                for s, m in zip(range(0, 16, 2), [mid[0], mid[2], up[0], up[1], up[2], up[1], up[0], mid[2]]):
                    harp_note(at(b, s), m, g_h * 0.8, 1.2)
            elif figure == "lament":  # eighths, the top voice falling one step per beat
                for s in range(0, 16, 2):
                    harp_note(at(b, s), up[(s // 2) % 3], g_h * 0.7, 1.0)
                for s in (0, 8):
                    harp_note(at(b, s), fit(root, 36, 47), g_h * 0.9, 1.6)
            elif figure == "sparse":  # a few notes, far apart
                for s in (0, 7) if b % 2 == 0 else (3,):
                    harp_note(at(b, s), up[int(rng.integers(0, 3))], g_h * 0.8, 2.0)
            elif figure == "waltz":  # oom-pah-pah in triplet beats: bass, chord, chord
                for beat in range(4):
                    s = beat * 4
                    harp_note(at(b, s), fit(root, 36, 47), g_h * 0.85, 0.9)
                    for j, m in enumerate(up):
                        harp_note(at(b, s + 4 / 3) + 0.006 * j, m, g_h * 0.4, 0.5)
                        harp_note(at(b, s + 8 / 3) + 0.006 * j, m, g_h * 0.35, 0.5)

        # ---- the 808: kick and bass at once
        if drums_on or (sec == "b"):
            pat = {"steady": (0, 10), "busy": (0, 3, 7, 10, 14), "sparse": (0,)}[bass_pat]
            if sec == "b" and bass_pat == "steady":
                pat = (0, 7, 10)
            for j, s in enumerate(pat):
                m = root - 12
                glide = False
                if s == 14:
                    m = nxt_root - 12  # slide into the next chord
                    glide = True
                if s == 10 and sec == "b" and b % 2:
                    m += 12  # the octave jump
                    glide = True
                nxt = pat[j + 1] if j + 1 < len(pat) else 16
                dur = (nxt - s) * st
                g_build = 0.35 + 0.2 * k_in if sec == "build" else 1.0
                bass_notes.append((at(b, s), dur * g_build if sec == "build" else dur, float(m), glide))
                if not glide:
                    hits.append((at(b, s), 1.0 if s == 0 else 0.8))

        # ---- the snare: half time, a reversed swell into it
        if drums_on and sec != "build":
            for s in (8,):
                x = clap(int(0.4 * SR), rng)
                put(snares, at(b, s) * SR, x, 0.8)
                hits.append((at(b, s), 0.9))
                if "reverse" in L and b % 4 == 3:  # a reversed tail sucks into the snare
                    n = int(bar / 2 * SR)
                    tail = lowpass(rng.standard_normal(n), 3000) * np.exp(-np.arange(n) / (0.25 * SR))
                    put(snares, at(b, s) * SR - n, tail[::-1], 0.12)

        # ---- hats: eighths, sixteenths or triplets, with rolls
        if drums_on or sec == "build":
            if sec == "build":  # the roll speeds up through the build
                div = (2, 3, 4, 6)[min(k_in, 3)]
                pos = np.arange(0, 16, 4 / div)
            elif trip:
                pos = np.arange(0, 16, 4 / 3)
            else:
                pos = np.arange(0, 16, 2 if hat_pat == "8" else 1)
            for s in pos:
                acc = 1.0 if s % 4 == 0 else 0.6
                p = 0.5 + 0.25 * np.sin(s)
                x = hat(int(0.06 * SR), rng)
                put(hats[:, 0], at(b, s) * SR, x, acc * (1 - p) * 0.9)
                put(hats[:, 1], at(b, s) * SR, x, acc * p * 0.9)
            if sec != "build" and b % 2 == 1:  # a roll at the end of every other bar, rising
                kind = int(rng.integers(0, 3))
                start, cnt = (12, 6) if kind == 0 else ((14, 4) if kind == 1 else (12, 8))
                for j in range(cnt):
                    s = start + j * (16 - start) / cnt
                    x = hat(int(0.04 * SR), rng, tone=0.8 + 0.06 * j)
                    put(hats[:, j % 2], at(b, s) * SR, x, 0.5 + 0.06 * j)

        # ---- the cowbell hook: two bars, four times, the fourth one varied
        bell_on = ("cowbell" in L and (heavy or sec == "intro")) or ("cowbell_b" in L and sec == "b")
        if bell_on and sec != "build" and b % 2 == 0:
            hook_count += 1
            sc = HARMONIC
            for i, (s, d) in enumerate(hook):
                if sec == "intro" and i % 2:
                    continue
                dd = d + (2 if hook_count % 4 == 0 and i >= len(hook) - 3 else 0)
                m = spec.tonic + 24 + sc[dd % 7] + 12 * (dd // 7)
                if s % 8 == 0:  # strong steps take the nearest chord tone
                    pcs = {x % 12 for x in tones}
                    m = min((x for x in range(m - 4, m + 5) if x % 12 in pcs), key=lambda x: abs(x - m))
                m = fit(m, 62, 74)
                put(bell, at(b, s) * SR, cowbell(midi_to_hz(m), int(0.5 * SR)), 1.0)
                hits.append((at(b, s), 0.4))

        # ---- the theremin: a long, slow line over the chords
        ther_on = ("theremin" in L and sec in ("a2", "break", "b", "outro")) or ("theremin_b" in L and sec == "b")
        if ther_on and b % 2 == 0:
            line = [up[2], up[1]] if (b // 2) % 2 == 0 else [up[1], up[0]]
            for j, m in enumerate(line):
                ther_notes.append((at(b, 16 * j), 2 * bar / len(line) * 0.95, float(fit(m + 12, 64, 79))))

        # ---- witch house air: reversed chords, whispers
        if "reverse" in L and new_sec and sec in ("break", "b", "outro"):
            n = int(bar * SR)
            x = sum(harpsichord(midi_to_hz(m), n, rng) for m in up + [mid[0]])
            x = lowpass(x[::-1], 3000)
            put(swells[:, 0], t0 * SR - n, x, 0.5)
            put(swells[:, 1], t0 * SR - n, x, 0.5)
        if "whisper" in L and sec in ("intro", "a2", "break", "outro") and rng.random() < 0.5:
            x = whisper(int(rng.uniform(1.5, 3) * SR), rng)
            p = rng.uniform(0.15, 0.85)
            s = rng.uniform(0, 8)
            put(air[:, 0], at(b, s) * SR, x, 0.05 * (1 - p))
            put(air[:, 1], at(b, s) * SR, x, 0.05 * p)
        if sec == "build":  # a reversed noise riser into the drop
            n = int(bar * SR)
            ramp = (k_in + np.arange(n) / n) / sec_len
            x = bandpass(rng.standard_normal(n), 300, 4000) * ramp**3
            put(swells[:, 0], t0 * SR, x, 0.08)
            put(swells[:, 1], t0 * SR, x, 0.08)

        # ---- tape stops: the last beat before the break, and the very end
        if b + 1 < n_bars and sections[b + 1] == "break" and sec != "break":
            stops.append((int(at(b, 12) * SR), int((t0 + bar) * SR)))
        if new_sec and sec in ("break", "b"):
            hits.append((t0, 1.0))

    # ------------------------------------------------------------------ buses
    b8 = bass808(bass_notes, total) * 0.5
    th = theremin(ther_notes, total) * 0.07
    mix = np.zeros((total, 2))
    k_env = lfilter([0.0006], [1, -0.9994], np.abs(lowpass(b8, 120)))
    duck = 1 - 0.35 * np.clip(k_env / (k_env.max() + 1e-9) * 3, 0, 1)  # the 808 pumps the room
    mix += b8[:, None]
    mix += harp * duck[:, None]
    mix += np.stack([bell * 0.12 * 0.55, bell * 0.12 * 0.45], axis=1)
    mix += snares[:, None] * 0.5
    mix += hats * 0.5
    mix += swells + air
    mix += np.stack([th * 0.45, th * 0.55], axis=1) * duck[:, None]
    wet = harp * 0.6 + snares[:, None] * 0.35 + np.stack([th, th], axis=1) * 0.6 + swells * 0.5 + air
    wet += np.stack([bell, bell], axis=1) * 0.03
    mix += 0.5 * crypt(wet, rng)
    mix += crackle(total, rng)[:, None] * np.array([1.0, 0.8])

    # Cassette wow: the whole mix drifts a few cents, slowly.
    t_all = np.arange(total) / SR
    wobble = 0.0009 * np.sin(2 * np.pi * 0.5 * t_all) + 0.0004 * np.sin(2 * np.pi * 3.1 * t_all)
    pos = np.clip(np.arange(total) + np.cumsum(wobble) * 6, 0, total - 1)
    for ch in range(2):
        mix[:, ch] = np.interp(pos, np.arange(total), mix[:, ch])
    for s, e in stops:
        tape_stop(mix, s, min(e, total))
    out_n = int(min(3.0, spec.duration * 0.02) * SR)
    tape_stop(mix, total - out_n, total)  # the outro: the tape dies

    mix *= _env(total, 0.8, 0.2)[:, None]
    mix = np.tanh(1.15 * mix)
    mix = lowpass(mix.T, 9000).T * 0.7 + lowpass(mix.T, 3500).T * 0.3  # dark, with a little air for the hats
    mix = highpass(mix.T, 28).T
    low = sosfiltfilt(butter(2, 100, btype="low", fs=SR, output="sos"), mix, axis=0)
    e_low, e_high = np.sum(low**2), np.sum((mix - low) ** 2)
    mix += (min(1.0, np.sqrt(2 / 3 * e_high / (e_low + 1e-9))) - 1) * low
    mix /= np.max(np.abs(mix)) + 1e-9
    mix *= 10 ** (-1.0 / 20)
    return mix.astype(np.float32), analyze(mix, b8, hits, spec)


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
