"""Handpan reggae for focus: a steel handpan over slow reggae grooves (the genre, no one's songs).

The handpan is modal synthesis of a tuned steel shell: each tone field rings with its fundamental,
the octave and the compound fifth (the three partials a handpan maker tunes), a little beating
between near-identical modes, and a soft finger strike. Layouts are the classic handpan scales
(Kurd, Celtic minor, Pygmy, Amara, Equinox, Aegean, Oxalis), with the ding at the centre. Taps on
the shell and the gu (the hole underneath) give it its percussive side.

Consonance rules, so the steel and the band never fight:
- the band only plays chords whose tones the handpan can agree with (no diminished chords);
- on the strong steps the handpan plays tones of the sounding chord; in between it may pass
  through scale tones, but never one a semitone or a whole tone from a chord tone;
- a note that would still ring into a chord it clashes with is damped at the change;
- the echo and the melodica follow the same chord tones.

Each theme has a style, so the mix keeps changing colour:
- roots: one drop, organ bubble, guitar skank, melodica in the B sections;
- dub: the band drops out for long stretches, bass and rimshot through a tape echo;
- lovers: lovers rock, soft rockers drums, electric piano chords and a fingerpicked guitar;
- steppers: kick on every beat, sixteenth hats, organ stabs, a driving bass;
- nyabinghi: hand drums (the heartbeat, the funde, the repeater) with bass and pad, no band;
- ambient: handpan, gu, pad and the sea, no drums at all.

The analysis exports the handpan strikes as the 'kick' curve, so the figures move with each note.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from yalix_ambient.music import SR, _env, euclidean, lowpass, midi_to_hz, pad_voice, reverb
from yalix_ambient.music_dark import bandpass, highpass
from yalix_ambient.music_v2 import ks_pluck

AEOLIAN = (0, 2, 3, 5, 7, 8, 10)
DORIAN = (0, 2, 3, 5, 7, 9, 10)
MIXOLYDIAN = (0, 2, 4, 5, 7, 9, 10)
LYDIAN = (0, 2, 4, 6, 7, 9, 11)
IONIAN = (0, 2, 4, 5, 7, 9, 11)

# Handpan layouts: semitones from the ding, lowest to highest tone field.
LAYOUTS = {
    "kurd": (0, 7, 8, 10, 12, 14, 15, 17, 19),
    "celtic": (0, 7, 10, 12, 14, 15, 17, 19, 22),
    "pygmy": (0, 3, 5, 7, 10, 12, 15, 17, 19),
    "amara": (0, 7, 10, 12, 14, 15, 19, 22),
    "equinox": (0, 3, 7, 8, 10, 12, 14, 15, 19),
    "aegean": (0, 4, 7, 11, 12, 14, 16, 18, 19),
    "oxalis": (0, 7, 10, 12, 14, 16, 17, 19, 22),
    "sabye": (0, 5, 7, 9, 11, 12, 14, 16, 17),  # major, the ding on the fifth degree's fourth
}

# Bass lines over one bar of sixteenths: (step, length, scale degrees above the chord root).
BASS = {
    "onedrop": ((2, 2, 0), (4, 2, 0), (7, 1, 4), (8, 4, 7), (12, 2, 4), (14, 2, 2)),
    "roots": ((0, 3, 0), (4, 2, 2), (6, 2, 4), (8, 3, 4), (12, 4, 0)),
    "steppers": ((0, 2, 0), (2, 2, 0), (4, 2, 4), (6, 2, 0), (8, 2, 0), (10, 2, 2), (12, 2, 4), (14, 2, 7)),
    "dub": ((0, 6, 0), (8, 2, 4), (11, 5, 0)),
    "rockers": ((0, 2, 0), (3, 1, 0), (4, 3, 4), (8, 2, 0), (11, 1, 2), (12, 4, 4)),
    "lovers": ((0, 4, 0), (6, 2, 4), (8, 4, 2), (12, 2, 4), (14, 2, 7)),
}

FORMS = {
    "song": (("intro", 4), ("a", 8), ("b", 8), ("dub", 6), ("a", 8), ("b", 8), ("outro", 4)),
    "dub": (("intro", 4), ("a", 6), ("dub", 8), ("b", 6), ("dub", 8), ("outro", 4)),
    "flow": (("intro", 6), ("a", 12), ("b", 12), ("outro", 6)),
}

STYLES = {  # style -> (drums, bass line, band layers, form)
    "roots": ("onedrop", "onedrop", ("organ", "skank", "melodica"), "song"),
    "dub": ("onedrop", "dub", ("skank",), "dub"),
    "lovers": ("rockers", "lovers", ("rhodes", "guitar"), "song"),
    "steppers": ("steppers", "steppers", ("organ",), "song"),
    "nyabinghi": ("nyabinghi", "dub", ("pad",), "flow"),
    "ambient": ("none", "none", ("pad", "sea"), "flow"),
}


@dataclass
class Spec:
    duration: float = 194.159
    bpm: float = 72.0  # one bar = 16 sixteenths
    tonic: int = 50  # the ding (D3)
    layout: str = "kurd"
    mode: tuple[int, ...] = AEOLIAN  # harmony for the band, matching the layout
    progression: tuple[int, ...] = (0, 3)  # scale degrees, one chord every `chord_bars`
    chord_bars: int = 2
    style: str = "roots"  # roots | dub | lovers | steppers | nyabinghi | ambient
    hand: str = "groove"  # groove | melodic | flow: how the handpan is played
    hits: int = 5  # handpan notes per bar (Euclidean)
    ring: float = 2.8  # seconds for a mid handpan note to fade by 60 dB
    swing: float = 0.12  # delay of the off sixteenths, as a fraction of a sixteenth
    extra: tuple[str, ...] = ("gu", "taps")  # + shaker sea pad melodica
    echo: float = 3.0  # dub delay in sixteenths (3 = dotted eighth)
    feedback: float = 0.5
    reverb_s: float = 3.0
    seed: int = 1
    fps: int = 30

    @property
    def step(self) -> float:
        return 60 / self.bpm / 4


# ------------------------------------------------------------------ instruments


def handpan(freq: float, n: int, vel: float, ring: float, rng: np.random.Generator) -> np.ndarray:
    """A tone field: fundamental, octave and compound fifth (all harmonic), beating twins and a
    finger strike. Higher fields fade sooner, like the real shell."""
    t = np.arange(n) / SR
    rate = 6.9 / ring * (freq / 300) ** 0.5
    out = np.zeros(n)
    for ratio, amp, k in ((1.0, 1.0, 1.0), (2.0, 0.30 + 0.20 * vel, 1.6), (3.0, 0.08 + 0.10 * vel, 2.4)):
        f = freq * ratio
        ph = rng.uniform(0, 2 * np.pi)
        beat = 1 + rng.uniform(0.0004, 0.0012)
        out += amp * np.exp(-t * rate * k) * (np.sin(2 * np.pi * f * t + ph) + 0.25 * np.sin(2 * np.pi * f * beat * t + ph))
    strike = lowpass(rng.standard_normal(n), 1800) * np.exp(-t * 300) * 0.15 * vel
    return (out * np.minimum(t / 0.004, 1) + strike) * (0.3 + 0.7 * vel)


def shell_tap(n: int, rng: np.random.Generator) -> np.ndarray:
    """A muted tap on the rim of the shell: a short, dull tick."""
    t = np.arange(n) / SR
    return bandpass(rng.standard_normal(n), 700, 3000) * np.exp(-t * 80)


def gu(n: int) -> np.ndarray:
    """The hole underneath: a Helmholtz thump that bends down."""
    t = np.arange(n) / SR
    f = 82 + 40 * np.exp(-t * 18)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 5) * np.minimum(t / 0.004, 1)


def kick_r(n: int) -> np.ndarray:
    t = np.arange(n) / SR
    f = 48 + 55 * np.exp(-t * 30)
    return np.tanh(1.3 * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 8))


def rimshot(n: int, rng: np.random.Generator) -> np.ndarray:
    """A cross-stick: woody, short, with a ring of the snare shell."""
    t = np.arange(n) / SR
    wood = np.sin(2 * np.pi * 820 * t) * np.exp(-t * 60) + 0.6 * np.sin(2 * np.pi * 1630 * t) * np.exp(-t * 90)
    return wood + 0.7 * bandpass(rng.standard_normal(n), 1500, 6000) * np.exp(-t * 55)


def hat_r(n: int, rng: np.random.Generator, open_: bool = False) -> np.ndarray:
    t = np.arange(n) / SR
    return highpass(rng.standard_normal(n), 7000) * np.exp(-t * (14 if open_ else 80))


def shaker(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / SR
    return bandpass(rng.standard_normal(n), 4000, 11000) * np.minimum(t / 0.012, 1) * np.exp(-t * 35)


def hand_drum(freq: float, n: int, rng: np.random.Generator, slap: bool = False) -> np.ndarray:
    """Nyabinghi drums: a skin with a pitch drop; slaps add the high crack."""
    t = np.arange(n) / SR
    f = freq * (1 + 0.25 * np.exp(-t * 30))
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * (6 if freq < 120 else 12))
    skin = bandpass(rng.standard_normal(n), 600 if slap else 250, 5000 if slap else 1800) * np.exp(-t * 40)
    return np.tanh(1.4 * (body + (0.8 if slap else 0.25) * skin))


def dub_bass(freq: float, n: int) -> np.ndarray:
    t = np.arange(n) / SR
    x = np.sin(2 * np.pi * freq * t) + 0.25 * np.sin(4 * np.pi * freq * t) + 0.07 * np.sin(6 * np.pi * freq * t)
    env = np.minimum(t / 0.008, 1) * np.exp(-t * 0.8) * np.clip((n / SR - t) / 0.03, 0, 1)
    return x * env


def organ(freqs: list[float], n: int, t0: float) -> np.ndarray:
    """Drawbar organ (16', 8', 5 1/3', 4') with a slow rotary tremolo."""
    t = np.arange(n) / SR
    out = np.zeros(n)
    for f in freqs:
        for h, a in ((0.5, 0.5), (1, 1.0), (1.5, 0.4), (2, 0.35)):
            out += a * np.sin(2 * np.pi * f * h * t)
    trem = 1 + 0.12 * np.sin(2 * np.pi * 5.6 * (t + t0))
    env = np.minimum(t / 0.004, 1) * np.clip((n / SR - t) / 0.02, 0, 1)
    return out * trem * env / max(len(freqs), 1)


def rhodes(freqs: list[float], n: int) -> np.ndarray:
    """Electric piano: a struck tine (FM, bright on the attack, mellow after) with a bell."""
    t = np.arange(n) / SR
    out = np.zeros(n)
    for f in freqs:
        index = 0.25 + 1.3 * np.exp(-t * 6)
        out += np.sin(2 * np.pi * f * t + index * np.sin(2 * np.pi * f * t)) * np.exp(-t * 1.3)
        out += 0.12 * np.sin(2 * np.pi * f * 7.0 * t) * np.exp(-t * 20)
    env = np.minimum(t / 0.003, 1) * np.clip((n / SR - t) / 0.08, 0, 1)
    return out * env / max(len(freqs), 1)


def skank(freqs: list[float], n: int, rng: np.random.Generator) -> np.ndarray:
    """The chop: a short, muted strum of the chord on the offbeat."""
    t = np.arange(n) / SR
    x = sum(ks_pluck(f, n, rng, damping=0.985) * (1 - 0.1 * i) for i, f in enumerate(freqs))
    return x * np.exp(-t * 22)


def melodica(freq: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """A free reed: harmonics, a little breath, vibrato that comes in late."""
    t = np.arange(n) / SR
    vib = 1 + 0.004 * np.sin(2 * np.pi * 5.2 * t) * np.clip((t - 0.3) / 0.4, 0, 1)
    ph = 2 * np.pi * np.cumsum(freq * vib) / SR
    x = sum(np.sin(k * ph) / k**1.2 for k in range(1, 9))
    breath = bandpass(rng.standard_normal(n), 1500, 6000) * 0.04
    env = np.minimum(t / 0.06, 1) * np.clip((n / SR - t) / 0.15, 0, 1)
    return lowpass(x + breath, 3000) * env


def tape_echo(x: np.ndarray, delay: float, fb: float, repeats: int = 6, tone: float = 2600) -> np.ndarray:
    """Dub delay: each repeat is quieter and darker, like a worn tape loop."""
    d = int(delay * SR)
    out, y = np.zeros_like(x), x
    for _ in range(repeats):
        y = np.concatenate([np.zeros((d,) + x.shape[1:]), y[:-d]]) * fb
        y = lowpass(y.T, tone).T if y.ndim > 1 else lowpass(y, tone)
        out += y
    return out


# ------------------------------------------------------------------ harmony


def triad(mode: tuple[int, ...], degree: int) -> list[int]:
    return [mode[(degree + k) % 7] + 12 * ((degree + k) // 7) for k in (0, 2, 4)]


def is_diminished(mode: tuple[int, ...], degree: int) -> bool:
    a, _, c = triad(mode, degree)
    return c - a == 6


def chord_tones(spec: Spec, degree: int, base: int) -> list[int]:
    return [base + x for x in triad(spec.mode, degree)]


def degree_note(spec: Spec, root_degree: int, steps: int, base: int) -> int:
    d = root_degree + steps
    return base + spec.mode[d % 7] + 12 * (d // 7)


def consonant(pc: int, chord_pcs: set[int]) -> bool:
    """A passing tone may not rub against the chord: no semitone or whole tone from any chord tone."""
    return all(min((pc - c) % 12, (c - pc) % 12) > 2 for c in chord_pcs)


def plan(spec: Spec, n_bars: int) -> list[str]:
    """The form scaled to the number of bars: one section name per bar."""
    form = FORMS[STYLES[spec.style][3]]
    weights = np.array([b for _, b in form], dtype=float)
    lengths = np.maximum(np.round(weights / weights.sum() * n_bars), 2).astype(int)
    lengths[-2] += n_bars - lengths.sum()
    return [name for (name, _), k in zip(form, lengths) for _ in range(k)]


def handpan_bar(spec: Spec, rng: np.random.Generator, notes: np.ndarray, chord_pcs: set[int]):
    """One bar of handpan: (step, field index, velocity). The ding on the one, chord tones on the
    beats, consonant neighbours on the off steps."""
    if spec.hand == "flow":  # a rolling pattern of eighths, chord tones only
        steps = list(range(0, 16, 2))
    else:
        k = spec.hits if spec.hand == "groove" else max(3, spec.hits - 2)
        steps = [i for i, h in enumerate(euclidean(k, 16, int(rng.integers(0, 3)))) if h]
    chordy = [i for i, m in enumerate(notes) if m % 12 in chord_pcs and i > 0] or [1]
    passing = [i for i, m in enumerate(notes) if i > 0 and consonant(m % 12, chord_pcs)]
    out, cur = [], int(rng.choice(chordy))
    ding_ok = notes[0] % 12 in chord_pcs
    for s in steps:
        if s == 0 and ding_ok:
            out.append((0, 0, 0.85))
            continue
        strong = s % 4 == 0 or spec.hand == "flow"
        pool = chordy if strong or not passing else chordy + passing
        near = sorted(pool, key=lambda i: (abs(i - cur), rng.random()))[:3]
        cur = int(rng.choice(near))
        out.append((s, cur, float(rng.uniform(0.7, 0.95) if strong else rng.uniform(0.45, 0.65))))
    return out


def melody_phrase(rng: np.random.Generator, bars: int) -> list[tuple[int, int]]:
    """A slow melodica rhythm: (step from phrase start, length in steps)."""
    rhythms = (((0, 6), (8, 4), (12, 4)), ((2, 6), (8, 8)), ((0, 4), (4, 4), (8, 8)), ((4, 4), (10, 6)))
    return [(b * 16 + s, ln) for b in range(bars) for s, ln in rhythms[int(rng.integers(0, len(rhythms)))]]


# ------------------------------------------------------------------ synthesis


def put(buf: np.ndarray, start: float, x: np.ndarray, g: float = 1.0) -> None:
    s = int(start)
    if s >= len(buf):
        return
    m = min(len(x), len(buf) - s)
    buf[s : s + m] += x[:m] * g


def synthesize(spec: Spec) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(spec.seed)
    total = int(spec.duration * SR)
    st = spec.step
    bar = 16 * st
    n_bars = int(spec.duration / bar)
    sections = plan(spec, n_bars)
    groove, bass_line, band_layers, _ = STYLES[spec.style]
    L = set(band_layers) | set(spec.extra)
    notes = np.array([spec.tonic + o for o in LAYOUTS[spec.layout]])
    prog = [d for d in spec.progression if not is_diminished(spec.mode, d)] or [0]
    base = spec.tonic - 12
    while base > 45:
        base -= 12

    def when(b: int, s: float) -> float:  # seconds, with swing on the off sixteenths
        return b * bar + (s + (spec.swing if int(s) % 2 else 0.0)) * st

    def chord_at(b: int) -> int:
        return prog[(b // spec.chord_bars) % len(prog)]

    hp = np.zeros((total, 2))
    taps = np.zeros(total)
    drums = np.zeros(total)
    rim_bus = np.zeros(total)
    bass = np.zeros(total)
    keys = np.zeros((total, 2))
    skank_bus = np.zeros(total)
    mel = np.zeros(total)
    pad = np.zeros(total)
    strikes: list[tuple[float, float]] = []

    # Handpan fields sit in a circle around the ding: pan each one by its place on the shell.
    angle = np.linspace(0, 2 * np.pi, len(notes), endpoint=False) + rng.uniform(0, np.pi)
    pan = 0.5 + 0.3 * np.sin(angle)
    pan[0] = 0.5

    motif: list | None = None
    phrase = melody_phrase(rng, 4)
    mel_deg = 4
    for b in range(n_bars):
        sec = sections[b]
        deg = chord_at(b)
        chord = chord_tones(spec, deg, base)
        pcs = {c % 12 for c in chord}
        change = (b + 1) % spec.chord_bars == 0  # the chord changes after this bar
        next_pcs = {c % 12 for c in chord_tones(spec, chord_at(b + 1), base)}
        band = sec in ("a", "b")

        # ---- handpan: a two-bar motif per chord, varied a little each time
        density = {"intro": 0.55, "a": 1.0, "b": 0.85, "dub": 0.5, "outro": 0.5}[sec]
        if b % spec.chord_bars == 0 or motif is None:
            motif = [handpan_bar(spec, rng, notes, pcs) for _ in range(2)]
        for s, i, v in motif[b % 2]:
            if s and rng.random() > density:
                continue
            ts = when(b, s)
            ring = spec.ring * (1.4 if i == 0 else 1.0) * (1.0 if v > 0.68 else 0.5)
            n = int(min(ring * 1.2, 6.0) * SR)
            if change and notes[i] % 12 not in next_pcs:  # damp before it clashes with the next chord
                n = min(n, int(((b + 1) * bar - ts + 0.25) * SR))
            x = handpan(midi_to_hz(notes[i]), max(n, 1), v, ring, rng)
            x *= np.clip((n - np.arange(n)) / (0.2 * SR), 0, 1)
            put(hp[:, 0], ts * SR, x, (1 - pan[i]) * 0.9)
            put(hp[:, 1], ts * SR, x, pan[i] * 0.9)
            strikes.append((ts, v))
        if "taps" in L and sec != "intro":
            for s in range(2, 16, 4):
                if rng.random() < (0.5 if spec.hand == "groove" else 0.2):
                    put(taps, when(b, s) * SR, shell_tap(int(0.1 * SR), rng), rng.uniform(0.06, 0.12))
        if "gu" in L and b % 2 == 0 and sec != "intro":
            put(drums, when(b, 0) * SR, gu(int(0.8 * SR)), 0.4)

        # ---- drums
        if band or (sec == "dub" and groove != "none"):
            kick_steps = {"onedrop": {8}, "rockers": {0, 8}, "steppers": {0, 4, 8, 12}}.get(groove, set())
            if sec == "dub" and groove != "nyabinghi":
                kick_steps = {8}
            for s in kick_steps:
                put(drums, when(b, s) * SR, kick_r(int(0.5 * SR)), 0.75)
            if groove in ("onedrop", "rockers", "steppers"):
                put(rim_bus, when(b, 8) * SR, rimshot(int(0.25 * SR), rng), 0.5)
                if b % 4 == 3 and sec != "dub" and spec.style != "lovers":
                    for s in (14, 15):
                        put(rim_bus, when(b, s) * SR, rimshot(int(0.2 * SR), rng), 0.22)
            if groove in ("onedrop", "rockers", "steppers") and sec != "dub":
                for s in range(0, 16, 1 if groove == "steppers" else 2):
                    open_ = groove != "steppers" and s % 8 == 6 and rng.random() < 0.5
                    g = (0.07 if s % 4 else 0.10) * (0.7 if spec.style == "lovers" else 1.0)
                    put(drums, when(b, s) * SR, hat_r(int(0.3 * SR), rng, open_), g)
            if groove == "nyabinghi":
                put(drums, when(b, 0) * SR, hand_drum(70, int(1.0 * SR), rng), 0.9)  # the heartbeat
                put(drums, when(b, 8) * SR, hand_drum(70, int(0.5 * SR), rng), 0.45)
                for s in (4, 6, 12, 14):
                    put(drums, when(b, s) * SR, hand_drum(190, int(0.4 * SR), rng), 0.32)
                for s, h in enumerate(euclidean(int(rng.integers(3, 6)), 16, int(rng.integers(0, 16)))):
                    if h:
                        put(drums, when(b, s) * SR, hand_drum(380, int(0.3 * SR), rng, slap=True), 0.2)
        if "shaker" in L and sec not in ("intro", "outro"):
            for s in range(16):
                put(drums, when(b, s) * SR, shaker(int(0.15 * SR), rng), 0.04 if s % 2 else 0.065)

        # ---- bass
        if bass_line != "none" and (band or sec == "dub"):
            for s, ln, d in BASS["dub" if sec == "dub" else bass_line]:
                m = degree_note(spec, deg, d, base)
                while m > base + 7:
                    m -= 12
                put(bass, when(b, s) * SR, dub_bass(midi_to_hz(m), int(ln * st * SR * 0.95)))

        # ---- keys and guitars
        voicing = [midi_to_hz(c + 12) for c in chord]
        if band:
            if "skank" in L:
                for s in (4, 12):
                    put(skank_bus, when(b, s) * SR, skank([f * 2 for f in voicing], int(0.25 * SR), rng), 0.5)
            if "organ" in L:
                stabs = (4, 12) if spec.style != "steppers" else (2, 6, 10, 14)
                for s in stabs:
                    x = organ(voicing, int(1.2 * st * SR), when(b, s))
                    put(keys[:, 0], when(b, s) * SR, x, 0.16)
                    put(keys[:, 1], when(b, s) * SR, x, 0.24)
                if spec.style == "roots":  # the bubble: low chord tones on the off sixteenths
                    for s in (2, 3, 6, 7, 10, 11, 14, 15):
                        c = chord[0] if s % 4 == 2 else chord[2]
                        x = organ([midi_to_hz(c)], int(0.8 * st * SR), when(b, s))
                        put(keys[:, 0], when(b, s) * SR, x, 0.12)
                        put(keys[:, 1], when(b, s) * SR, x, 0.08)
            if "rhodes" in L:
                for s, ln in ((4, 3), (12, 3)) if b % 2 else ((0, 6), (8, 3), (12, 3)):
                    x = rhodes(voicing, int(ln * st * SR * 1.6))
                    put(keys[:, 0], when(b, s) * SR, x, 0.22)
                    put(keys[:, 1], when(b, s) * SR, x, 0.18)
            if "guitar" in L:  # fingerpicked: root, fifth, third, fifth on the eighths
                pick = [chord[0] + 12, chord[2] + 12, chord[1] + 24, chord[2] + 12]
                for j, s in enumerate(range(0, 16, 2)):
                    put(skank_bus, when(b, s) * SR, ks_pluck(midi_to_hz(pick[j % 4]), int(1.2 * SR), rng, 0.996), 0.16)

        # ---- melodica in the B sections: every note starts on a tone of the sounding chord
        if "melodica" in L and sec == "b":
            for s, ln in phrase:
                if s // 16 == b % 4:
                    mel_deg = int(np.clip(mel_deg + rng.choice([-2, -1, 1, 2]), 2, 11))
                    cands = [d for d in range(2, 12) if degree_note(spec, 0, d, 0) % 12 in pcs]
                    mel_deg = min(cands, key=lambda d: abs(d - mel_deg)) if cands else mel_deg
                    m = degree_note(spec, 0, mel_deg, spec.tonic + 12)
                    put(mel, when(b, s % 16) * SR, melodica(midi_to_hz(m), int(ln * st * SR), rng), 0.18)

        # ---- pad under the quiet sections (and everywhere without drums)
        if "pad" in L and (sec in ("intro", "dub", "outro") or groove in ("none", "nyabinghi")) and b % spec.chord_bars == 0:
            n = int(spec.chord_bars * bar * SR) + int(0.5 * SR)
            for c in chord:
                put(pad, b * bar * SR, pad_voice(midi_to_hz(c + 12), n, rng) * _env(n, 1.0, 1.5), 0.045)

    # ------------------------------------------------------------------ buses
    mix = np.zeros((total, 2))
    mix += hp * 0.6
    mix += np.stack([taps * 0.6, taps * 0.4], axis=1)
    mix += drums[:, None] * 0.7
    mix += np.stack([rim_bus * 0.55, rim_bus * 0.45], axis=1)
    b2 = lowpass(np.tanh(1.8 * bass), 600) * 0.42
    mix += b2[:, None]
    mix += lowpass(keys.T, 4000).T * 0.6
    sk = highpass(np.tanh(2.0 * skank_bus), 300) * 0.32
    mix += np.stack([sk * 0.65, sk * 0.35], axis=1)
    mix += np.stack([mel * 0.5, mel * 0.5], axis=1)
    mix += np.stack([pad, pad], axis=1)

    if "sea" in L:  # waves on the shore: filtered noise that swells every ~9 s
        t_all = np.arange(total) / SR
        swell = (0.5 - 0.5 * np.cos(2 * np.pi * t_all / rng.uniform(8, 11))) ** 2
        mix += lowpass(rng.standard_normal((2, total)), 1200).T * (0.008 + 0.035 * swell)[:, None]

    # Dub: tape echo on the rimshot and the skank (not on the handpan: it would blur the harmony).
    fb = spec.feedback + (0.12 if spec.style == "dub" else 0.0)
    send = np.stack([rim_bus * 0.5, rim_bus * 0.5], axis=1) + np.stack([sk * 0.4, sk * 0.4], axis=1)
    mix += (0.75 if spec.style == "dub" else 0.45) * tape_echo(send, spec.echo * st, fb)
    rv = hp * 0.35 + np.stack([mel, mel], axis=1) * 0.5 + keys * 0.3 + np.stack([rim_bus, rim_bus], axis=1) * 0.2
    mix += 0.35 * reverb(rv, spec.reverb_s, rng)

    mix *= _env(total, 2.0, 6.0)[:, None]
    mix = np.tanh(1.05 * mix)
    mix /= np.max(np.abs(mix)) + 1e-9
    mix *= 10 ** (-1.0 / 20)
    return mix.astype(np.float32), analyze(mix, b2, strikes, spec)


def analyze(mix, bass, strikes, spec: Spec) -> dict:
    hop = SR // spec.fps
    n_frames = len(mix) // hop
    mono = mix.mean(axis=1)

    def env_of(x, smooth):
        r = np.sqrt(np.array([np.mean(x[i * hop : (i + 1) * hop] ** 2) for i in range(n_frames)]))
        k = np.exp(-np.arange(-30, 31) ** 2 / (2 * smooth**2))
        r = np.convolve(r, k / k.sum(), mode="same")
        return [round(float(v), 4) for v in (r - r.min()) / (r.max() - r.min() + 1e-9)]

    hit = np.zeros(n_frames)  # every handpan strike: a jump that decays over ~0.3 s
    for ts, v in strikes:
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
