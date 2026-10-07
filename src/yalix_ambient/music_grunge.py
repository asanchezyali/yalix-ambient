"""Grunge series music: sludgy, melancholic, Seattle-style (the genre, never anyone's songs).

Hallmarks synthesised here:
- down-tuned heavy guitars (drop-D / E♭ register), double-tracked hard left and right, with
  bends and slides into riff notes; one amp (tanh) and cabinet per side, so notes interact;
- a melody played by a singing lead guitar with a second guitar in harmony (twin guitars).
  Notes on the beat sit on the chord that is sounding (the riff's chord in the verses) and the
  harmony takes the next chord tone above, so the line never fights the band. The verse and
  chorus melodies are composed once per song (A B A D, hook / answer), so each piece has a hook;
  `sing` (formant vocals) stays available but is not used;
- a wah lead (a band-pass that sweeps) over the solo section, or a cello lead when unplugged;
- the acoustic side: fingerpicked and strummed steel strings (Karplus-Strong), cello, drone;
- roomy rock drums with a big snare, crashes on section changes and tom fills;
- a real song form (intro, verses, choruses, bridge, solo, outro) scaled to the duration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter

from yalix_ambient.music import SR, _env, euclidean, lowpass, midi_to_hz, reverb
from yalix_ambient.music_dark import bandpass, hat, highpass
from yalix_ambient.music_v2 import bass_tone, ks_pluck, tom

AEOLIAN = (0, 2, 3, 5, 7, 8, 10)
DORIAN = (0, 2, 3, 5, 7, 9, 10)
PHRYGIAN = (0, 1, 3, 5, 7, 8, 10)
MIXOLYDIAN = (0, 2, 4, 5, 7, 9, 10)
IONIAN = (0, 2, 4, 5, 7, 9, 11)
PENTATONIC = (0, 3, 5, 7, 10)
MAJOR_PENTATONIC = (0, 2, 4, 7, 9)

VOWELS = {
    "a": ((730, 1.0), (1090, 0.5), (2440, 0.22)),
    "o": ((570, 1.0), (840, 0.55), (2410, 0.15)),
    "e": ((530, 1.0), (1840, 0.4), (2480, 0.2)),
    "u": ((320, 1.0), (870, 0.35), (2240, 0.1)),
}

FORM = (("intro", 4), ("verse", 8), ("chorus", 8), ("verse", 8), ("chorus", 8),
        ("bridge", 8), ("solo", 8), ("chorus", 8), ("outro", 6))  # fmt: skip
ENERGY = {"intro": 0.7, "verse": 0.8, "chorus": 1.0, "bridge": 0.6, "solo": 0.95, "outro": 0.7}


@dataclass
class Spec:
    duration: float = 194.159
    bpm: float = 84.0
    meter: int = 8  # steps per bar
    subdiv: int = 2  # steps per beat: 2 = eighths (4/4, 7/8...), 3 = triplets (6/8, 12/8 feel)
    tonic: int = 38  # D2: drop-D register
    scale: tuple[int, ...] = AEOLIAN
    form: tuple[tuple[str, int], ...] = FORM
    # Riff: (step, length in steps, interval from the root, articulation) over `riff_bars` bars.
    # Articulations: n note, p power chord, m palm-muted power chord, b bend up into the note,
    # s slide up into the note, v sustained note with vibrato, t tritone dyad.
    riff: tuple[tuple[float, float, int, str], ...] = ((0, 2, 0, "p"), (3, 1, 3, "b"), (4, 2, 0, "m"), (6, 2, 1, "s"))
    riff_bars: int = 1
    verse_roots: tuple[int, ...] = (0,)  # transposition per riff cycle
    chorus_roots: tuple[int, ...] = (0, 3, -2, 5)
    bridge_roots: tuple[int, ...] = (0, -2, -4, -2)
    chord_bars: int = 2  # bars per chord in choruses and bridges
    verse: str = "electric"  # electric | acoustic | both
    chorus: str = "electric"
    intro: str = "riff"  # riff | acoustic | drone
    drums: str = "rock"  # rock | halftime | shuffle | sludge | brush | none
    kick_pat: str | None = None  # explicit 'x.' strings per bar, override the style
    snare_pat: str | None = None
    drive: float = 7.0
    wah: str = "lead"  # off | lead | riff
    harmony: int = 2  # diatonic steps between voices (2 third, 3 fourth, 4 fifth)
    harmony_semi: int | None = None  # fixed parallel interval in semitones instead
    voice_center: int = 55  # lead vocal register (MIDI)
    vowels: str = "aoea"
    layers: tuple[str, ...] = ("bass", "vocals", "lead")  # + acoustic cello drone rain
    lead_scale: tuple[int, ...] = PENTATONIC
    riff_b: tuple[tuple[float, float, int, str], ...] | None = None  # turnaround; None -> walk-up
    quiet_verse: bool = False  # Nirvana dynamics: clean verse, loud chorus
    chorus_strum: str = "x.xxx.xo"  # x down, o up, m muted, . ring
    chorus_strum_b: str | None = None  # every fourth bar
    reverb_s: float = 3.0
    seed: int = 1
    fps: int = 30

    @property
    def step(self) -> float:
        return 60 / self.bpm / self.subdiv


# ------------------------------------------------------------------ instruments


def kick_g(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / SR
    f = 52 + 75 * np.exp(-t * 28)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 7)
    click = highpass(rng.standard_normal(n), 2000) * np.exp(-t * 300) * 0.4
    return np.tanh(1.5 * (body + click))


def snare_g(n: int, rng: np.random.Generator) -> np.ndarray:
    """A big, open rock snare: two shell modes plus a long noise tail."""
    t = np.arange(n) / SR
    noise = bandpass(rng.standard_normal(n), 900, 7000) * np.exp(-t * 9)
    body = np.sin(2 * np.pi * 190 * t) * np.exp(-t * 16) + 0.4 * np.sin(2 * np.pi * 330 * t) * np.exp(-t * 22)
    return np.tanh(1.6 * (0.9 * noise + 0.7 * body))


def crash(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / SR
    x = highpass(rng.standard_normal(n), 3000) * np.exp(-t * 1.8)
    return x + 0.5 * bandpass(rng.standard_normal(n), 5000, 12000) * np.exp(-t * 4)


def open_hat(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / SR
    return highpass(rng.standard_normal(n), 6000) * np.exp(-t * 14)


def ride(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / SR
    bell_ = sum(np.sin(2 * np.pi * f * t) * a for f, a in ((510, 1.0), (763, 0.6), (1231, 0.4), (2597, 0.25)))
    return 0.25 * bell_ * np.exp(-t * 4) + 0.3 * highpass(rng.standard_normal(n), 6000) * np.exp(-t * 10)


def brush(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / SR
    return bandpass(rng.standard_normal(n), 2000, 8000) * np.minimum(t / 0.04, 1) * np.exp(-t * 7)


def wah(x: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """A swept band-pass, done as a cross-fade between fixed bands (smooth and vectorised)."""
    centers = np.geomspace(420, 2200, 6)
    idx = np.clip(pos, 0, 1) * (len(centers) - 1)
    out = 0.15 * x
    for k, c in enumerate(centers):
        out = out + np.clip(1 - np.abs(idx - k), 0, 1) * bandpass(x, c * 0.75, c * 1.33) * 2.5
    return out


def amp(x: np.ndarray, drive: float) -> np.ndarray:
    """Two gain stages and a 4x12-ish cabinet: mud cut, presence bump, fizz roll-off."""
    y = np.tanh(drive * highpass(x, 90))
    y = np.tanh(1.5 * y)
    y = y + 0.35 * bandpass(y, 1400, 3200)
    return lowpass(lowpass(y, 5200), 6500)


def guitar_dry(freqs: list[float], dur: float, art: str, detune: float, rng: np.random.Generator) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    semis = np.zeros(n)
    if art == "b":
        semis -= 2 * np.exp(-t / 0.07)
    elif art == "s":
        semis -= 3 * np.clip(1 - t / 0.12, 0, 1)
    if art in ("v", "b") and dur > 0.4:
        semis += 0.3 * np.sin(2 * np.pi * 5.5 * t) * np.clip((t - 0.25) / 0.3, 0, 1)
    out = np.zeros(n)
    for f in freqs:
        for d in (-detune, detune):
            ph = np.cumsum(f * 2 ** ((semis + d / 100) / 12)) / SR
            out += 2 * (ph % 1) - 1
    out /= 2 * len(freqs)
    decay = 14.0 if art == "m" else 0.7
    env = np.minimum(t / 0.002, 1) * np.exp(-t * decay) * np.clip((dur - t) / 0.015, 0, 1)
    if art == "m":
        out = lowpass(out, 700)
    pick = highpass(rng.standard_normal(n), 2500) * np.exp(-t * 300) * 0.25
    return out * env + pick


def acoustic_pluck(freq: float, n: int, rng: np.random.Generator) -> np.ndarray:
    x = ks_pluck(freq, n, rng, damping=0.998)
    return x + 0.4 * bandpass(x, 90, 300)  # body resonance


def cello(freq: float, dur: float, rng: np.random.Generator) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    vib = 0.15 * np.sin(2 * np.pi * 5.0 * t + rng.uniform(0, 6.28)) * np.clip(t / 0.5, 0, 1)
    ph = np.cumsum(freq * 2 ** (vib / 12)) / SR
    x = 2 * (ph % 1) - 1
    x = lowpass(x, 1500) + 0.5 * bandpass(x, 250, 700)
    return x * _env(n, min(0.25, dur / 3), min(0.4, dur / 3))


def sing(events: list[tuple[float, float, float, str, float]], total: int, rng: np.random.Generator,
         detune_cents: float = 0.0) -> np.ndarray:  # fmt: skip
    """One voice: (start, duration, midi, vowel, gain) events -> formant-synthesised audio."""
    if not events:
        return np.zeros(total)
    pitch = np.full(total, np.nan)
    gate = np.zeros(total)
    since = np.zeros(total)
    vow = np.zeros(total, dtype=int)
    keys = list(VOWELS)
    for t0, d, m, v, g in events:
        s, e = int(t0 * SR), min(int((t0 + d) * SR), total)
        if e <= s:
            continue
        pitch[s:e] = m
        gate[s:e] = g
        since[s:e] = np.arange(e - s) / SR
        vow[s:e] = keys.index(v)
    # Fill the rests with the surrounding notes so the glide never swoops from zero.
    idx = np.where(np.isnan(pitch), 0, np.arange(total))
    np.maximum.accumulate(idx, out=idx)
    pitch = pitch[idx]
    first = np.argmax(~np.isnan(pitch))
    pitch[:first] = pitch[first]
    a = np.exp(-1 / (0.045 * SR))  # 45 ms portamento
    pitch = lfilter([1 - a], [1, -a], pitch - pitch[0], zi=[0.0])[0] + pitch[0]
    vib = 0.28 * np.sin(2 * np.pi * 5.3 * np.arange(total) / SR + rng.uniform(0, 6.28)) * np.clip((since - 0.35) / 0.4, 0, 1)
    drift = lowpass(rng.standard_normal(total), 3)
    drift *= 0.08 / (drift.std() + 1e-9)  # slow, human pitch wander (~8 cents)
    f = midi_to_hz(pitch + vib + drift + detune_cents / 100)
    ph = np.cumsum(f) / SR
    src = lowpass(2 * (ph % 1) - 1, 2800) + 0.08 * bandpass(rng.standard_normal(total), 1500, 6000)
    a_env = np.exp(-1 / (0.03 * SR))
    amp_ = lfilter([1 - a_env], [1, -a_env], gate)
    out = np.zeros(total)
    a_v = np.exp(-1 / (0.06 * SR))
    for k, v in enumerate(keys):
        mask = (vow == k).astype(float) * (gate > 0)
        if not mask.any():
            continue
        w = lfilter([1 - a_v], [1, -a_v], mask)
        out += w * sum(g * bandpass(src, fc * 0.9, fc * 1.1) for fc, g in VOWELS[v])
    out *= amp_
    out = np.tanh(2.0 * out / (np.max(np.abs(out)) + 1e-9))  # a little grit
    return out


# ------------------------------------------------------------------ composition


def layout(spec: Spec) -> list[tuple[float, int, str, int, int]]:
    """Bars as (start, steps, section, index within section, section number)."""
    bar_len = spec.meter * spec.step
    n_bars = int(spec.duration / bar_len)
    weights = [b for _, b in spec.form]
    scale = n_bars / sum(weights)
    counts = [max(2, int(b * scale / 2) * 2) for b in weights[:-1]]
    counts.append(n_bars - sum(counts))
    k = 0
    while counts[-1] > weights[-1] * scale + 3:  # hand spare bars to the choruses, not the outro
        names = [i for i, (nm, _) in enumerate(spec.form) if nm == "chorus"] or [1]
        counts[names[k % len(names)]] += 2
        counts[-1] -= 2
        k += 1
    bars, t = [], 0.0
    for k, ((name, _), c) in enumerate(zip(spec.form, counts)):
        for i in range(c):
            if t + 0.5 * bar_len > spec.duration:
                break
            bars.append((t, spec.meter, name, i, k))
            t += bar_len
    return bars


def compose_melody(rng: np.random.Generator, bars: int, S: int, sub: int, density: tuple[int, int],
                   start_deg: int) -> list[tuple[int, float, float, int]]:  # fmt: skip
    """Two-bar phrases: (bar, step, length, scale degree). Breath at the end of each phrase."""
    out = []
    deg = start_deg
    for ph in range(0, bars, 2):
        steps = 2 * S
        k = int(rng.integers(*density))
        cand = np.arange(0, int(steps * 0.72))
        starts = np.sort(rng.choice(cand, size=min(k, len(cand)), replace=False))
        starts[0] = int(rng.choice([0, 1]))
        starts = np.unique(starts)
        for j, s in enumerate(starts):
            end = starts[j + 1] if j + 1 < len(starts) else steps - sub
            deg = int(np.clip(deg + rng.choice([-2, -1, -1, 0, 1, 1, 2]), start_deg - 3, start_deg + 5))
            if j == len(starts) - 1:
                deg = start_deg + int(rng.choice([0, 2, 4]))  # land on a chord tone
            bar, step = divmod(int(s), S)
            out.append((ph + bar, step, float(max(end - s - 0.3, 0.7)), deg))
    return out


def synthesize(spec: Spec) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(spec.seed)
    total = int(spec.duration * SR)
    L = set(spec.layers)
    st, sub = spec.step, spec.subdiv
    bars = layout(spec)
    sc = spec.scale

    def deg_midi(base: int, deg: int) -> int:
        o, i = divmod(deg, len(sc))
        return base + 12 * o + sc[i]

    def third_of(off: int) -> int:
        return 3 if (off + 3) % 12 in sc else 4

    def root_off(bar) -> int:
        _, _, name, i, _ = bar
        if name in ("chorus", "solo"):
            return spec.chorus_roots[(i // spec.chord_bars) % len(spec.chorus_roots)]
        if name == "bridge":
            return spec.bridge_roots[(i // spec.chord_bars) % len(spec.bridge_roots)]
        return spec.verse_roots[(i // spec.riff_bars) % len(spec.verse_roots)]

    def put(buf, s, x, gain=1.0):
        s = int(s)
        n = min(len(x), len(buf) - s)
        if n > 0 and s >= 0:
            buf[s : s + n] += x[:n] * gain

    gtr = [np.zeros(total), np.zeros(total)]  # rhythm guitars, hard L / R
    clean = np.zeros(total)
    lead = np.zeros(total)
    bass_ev: list[tuple[float, float, int]] = []
    ac = np.zeros((total, 2))
    vc = np.zeros((total, 2))  # cello
    drums, snare_bus, cym = np.zeros(total), np.zeros(total), np.zeros((total, 2))
    kicks: list[float] = []
    voice_lead, voice_harm = [], []
    drone = np.zeros(total)  # kept for specs that ask for it

    base_v = spec.tonic + 12 * round((spec.voice_center - spec.tonic) / 12)
    S0 = spec.meter
    sec_len: dict[int, int] = {}
    for bar in bars:
        sec_len[bar[4]] = sec_len.get(bar[4], 0) + 1

    # Melodies: verse A B A D and chorus hook / answer / hook / new answer, 8 bars each,
    # so the hook comes back but no two halves are identical.
    def phrase(dens, start):
        return compose_melody(rng, 2, S0, sub, dens, start)

    def shift(ph, by):
        return [(b + by, s, ln, d) for b, s, ln, d in ph]

    va, vb, vd = phrase((4, 7), 0), phrase((4, 7), 1), phrase((3, 5), 2)
    verse_mel = va + shift(vb, 2) + shift(va, 4) + shift(vd, 6)
    hook, ans1, ans2 = phrase((3, 5), 4), phrase((3, 6), 3), phrase((2, 4), 2)
    chorus_mel = hook + shift(ans1, 2) + shift(hook, 4) + shift(ans2, 6)
    vowels = spec.vowels

    # Riff A, and a turnaround B: same riff, last half bar replaced by a walk-up to the root.
    cyc_steps = spec.riff_bars * S0
    if spec.riff_b:
        riff_b = spec.riff_b
    else:
        cut = cyc_steps - S0 // 2
        walk = (-5, -3, -2, -1) if S0 // 2 >= 4 else (-2, -1)
        ln_w = (S0 // 2) / len(walk)
        riff_b = tuple((s_, min(ln, cut - s_), iv, a) for s_, ln, iv, a in spec.riff if s_ < cut) + tuple(
            (cut + j * ln_w, ln_w, iv, "n" if j < len(walk) - 1 else "s") for j, iv in enumerate(walk))

    def riff_chord(root, i, ls):
        """The chord the riff is sitting on at this step (chords and landing notes, not passing notes)."""
        riff = riff_b if (i // spec.riff_bars) % 4 == 3 else spec.riff
        pos = (i % spec.riff_bars) * S0 + ls
        cand = [e for e in riff if e[0] <= pos and e[3] in "pmvb"]
        return root + (cand[-1][2] if cand else 0)

    def guitar(t0, dur, freqs, art, gain):
        for side in (0, 1):
            jit = abs(rng.normal(0, 0.006)) if side else 0.0
            x = guitar_dry(freqs, dur, art, 6 + 4 * side, rng)
            put(gtr[side], (t0 + jit) * SR, x, gain)

    def clean_note(t0, dur, freqs, gain):
        put(clean, t0 * SR, guitar_dry(freqs, dur + 0.3, "n", 4, rng), gain)

    def riff_bar(t0, S, root, i, gain, quiet):
        cyc_i = i // spec.riff_bars
        riff = riff_b if cyc_i % 4 == 3 else spec.riff
        cyc = i % spec.riff_bars
        for step, ln, iv, art in riff:
            bi, ls = divmod(step, S0)
            if bi != cyc or ls >= S:
                continue
            m = root + iv
            if quiet:  # clean verse: chord tones ring instead of distorted power chords
                third = third_of(m - spec.tonic)
                freqs = [m + 12, m + 12 + third, m + 19] if art in ("p", "m") else [m + 12]
                clean_note(t0 + ls * st, ln * st, [midi_to_hz(x) for x in freqs], 0.8)
            else:
                freqs = {"p": [m, m + 7, m + 12], "m": [m, m + 7, m + 12], "t": [m, m + 6]}.get(art, [m])
                guitar(t0 + ls * st, ln * st * 0.95, [midi_to_hz(x) for x in freqs], art, gain)
            bass_ev.append((t0 + ls * st, ln * st * 0.92, m - 12))

    def strum_bar(t0, S, root, pat, gain, acoustic=False):
        hits = [k for k, c in enumerate(pat[:S]) if c in "xom"]
        third = third_of(root - spec.tonic)
        for j, k in enumerate(hits):
            end = hits[j + 1] if j + 1 < len(hits) else S
            c = pat[k]
            if acoustic:
                voicing = [root + 12, root + 19, root + 24, root + 24 + third, root + 31]
                strings = voicing if c == "x" else voicing[::-1][:3]
                for q, m in enumerate(strings):
                    p = acoustic_pluck(midi_to_hz(m), int((0.25 if c == "m" else 1.6) * SR), rng)
                    pan = 0.3 + 0.4 * q / len(strings)
                    g = gain * {"x": 0.16, "o": 0.10, "m": 0.07}[c]
                    put(ac[:, 0], (t0 + k * st + q * 0.011) * SR, p, g * (1 - pan))
                    put(ac[:, 1], (t0 + k * st + q * 0.011) * SR, p, g * pan)
            else:
                art = "m" if c == "m" else "p"
                guitar(t0 + k * st, (end - k) * st * 0.97, [midi_to_hz(root + x) for x in (0, 7, 12)], art,
                       gain * (1.0 if c == "x" else 0.85))  # fmt: skip

    def pick_bar(t0, S, root, i, gain):
        """Fingerpicking: alternating bass on the beats, treble notes between, a new figure each bar."""
        third = third_of(root - spec.tonic)
        treble = [root + 24, root + 24 + third, root + 31, root + 36]
        for k in range(S):
            if k % sub == 0:
                m = root + 12 if (k // sub) % 2 == 0 else root + 19
            else:
                m = treble[(k + i) % len(treble)]
                if rng.random() < 0.25:
                    continue
            p = acoustic_pluck(midi_to_hz(m), int(2.0 * SR), rng)
            pan = 0.35 if k % sub == 0 else 0.65
            put(ac[:, 0], (t0 + k * st) * SR, p, gain * 0.2 * (1 - pan))
            put(ac[:, 1], (t0 + k * st) * SR, p, gain * 0.2 * pan)

    LEAD_RHYTHMS = {  # (start, length) in eighths, rock-solo phrasing
        8: [((0, 2), (2, 1), (3, 1), (4, 4)), ((0, 3), (3, 1), (4, 2), (6, 2)), ((1, 1), (2, 2), (4, 1), (5, 3)),
            ((0, 6), (6, 1), (7, 1)), ((0, 1), (1, 1), (2, 1), (3, 1), (4, 4))],
        12: [((0, 3), (3, 3), (6, 6)), ((0, 2), (2, 1), (3, 3), (6, 3), (9, 3)), ((0, 9), (9, 1), (10, 2))],
        6: [((0, 3), (3, 3)), ((0, 2), (2, 1), (3, 3)), ((0, 6),)],
    }  # fmt: skip
    KICKS = {
        8: {"verse": ("x...x...", "x..xx...", "x...x...", "x...x.x."),
            "chorus": ("x..xx.x.", "x.xx..x.", "x..xx.x.", "x.x.x.xx"),
            "bridge": ("x.....x.", "x.......", "x.....x.", "x...x...")},
        12: {"verse": ("x......x....", "x.....x.x...", "x......x....", "x..x...x...."),
             "chorus": ("x.....x.x...", "x..x..x.x...", "x.....x.x...", "x..x..x..xx."),
             "bridge": ("x...........", "x.....x.....", "x...........", "x........x..")},
        6: {"verse": ("x.....", "x..x..", "x.....", "x....x"),
            "chorus": ("x..x..", "x.xx..", "x..x..", "x..x.x"),
            "bridge": ("x.....", "x.....", "x.....", "x..x..")},
    }  # fmt: skip
    SNARES = {8: ("..x...x.", "....x..."), 12: ("...x.....x..", "......x....."), 6: ("...x..", "...x..")}

    # ---------------------------------------------------------------- bars
    lead_deg = 2
    n_bars = len(bars)
    for b, bar in enumerate(bars):
        t0, S, name, i, sec = bar
        nxt = bars[b + 1] if b + 1 < n_bars else None
        last_in_sec = nxt is None or nxt[4] != sec
        n_sec = sec_len[sec]
        root = spec.tonic + root_off(bar)
        en = ENERGY[name]
        bar_dur = S * st
        electric_verse = spec.verse in ("electric", "both")
        quiet = spec.quiet_verse and name == "verse" and i < n_sec - 2

        if (name == "verse" and electric_verse) or (name in ("intro", "outro") and spec.intro == "riff"):
            if name == "outro" and i >= n_sec - 2:
                if i == n_sec - 2:  # last chord rings out
                    guitar(t0, 2 * bar_dur, [midi_to_hz(spec.tonic + x) for x in (0, 7, 12)], "v", 1.0)
                    bass_ev.append((t0, 2 * bar_dur, spec.tonic - 12))
            else:
                intro_clean = name == "intro" and spec.quiet_verse and i < n_sec // 2
                riff_bar(t0, S, root, i, en, quiet or intro_clean)
        if name in ("chorus", "solo") and spec.chorus in ("electric", "both"):
            pat = spec.chorus_strum_b if i % 4 == 3 and spec.chorus_strum_b else spec.chorus_strum
            strum_bar(t0, S, root, pat, 1.0 if name == "chorus" else 0.8)
        if name in ("chorus", "solo") or (name == "verse" and not electric_verse):
            nxt_root = spec.tonic + root_off(nxt) if nxt is not None else root
            for k in range(0, S, 1 if name != "verse" else sub):  # driving eighths, passing note into the change
                m = root - 12
                if k == S - 1 and nxt_root != root:
                    m = nxt_root - 12 + (1 if nxt_root < root else -1)
                bass_ev.append((t0 + k * st, st * 0.9 if name != "verse" else sub * st * 0.9, m))
        if name == "bridge":
            bass_ev.append((t0, bar_dur * 0.95, root - 12))
            third = third_of(root - spec.tonic)
            arp = [root + 12, root + 19, root + 24, root + 12 + third + 12, root + 19, root + 24]
            for k in range(0, S, 1):  # clean arpeggio, a new order each bar
                if "acoustic" in L:
                    break
                m = arp[(k * (1 + i % 2)) % len(arp)]
                clean_note(t0 + k * st, st * 2, [midi_to_hz(m)], 0.55)

        acoustic_now = (
            (name == "verse" and spec.verse in ("acoustic", "both"))
            or (name in ("intro", "outro") and spec.intro == "acoustic")
            or (name == "bridge" and "acoustic" in L)
        )
        if acoustic_now:
            pick_bar(t0, S, root, i, 1.0 if spec.verse == "acoustic" else 0.6)
        if name in ("chorus", "solo") and spec.chorus in ("acoustic", "both"):
            pat = spec.chorus_strum_b if i % 4 == 3 and spec.chorus_strum_b else spec.chorus_strum
            strum_bar(t0, S, root, pat, 1.0 if spec.chorus == "acoustic" else 0.6, acoustic=True)

        if "cello" in L and name in ("chorus", "bridge") and i % spec.chord_bars == 0:
            third = third_of(root - spec.tonic)
            line = (root + 12, root + 12 + third, root + 19, root + 12 + third)
            m = line[(i // spec.chord_bars) % 4]
            x = cello(midi_to_hz(m), spec.chord_bars * bar_dur * 0.98, rng) * 0.12
            put(vc[:, 0], t0 * SR, x, 0.55)
            put(vc[:, 1], t0 * SR, x, 0.45)

        # Lead over the chorus chords: rock phrasing, bends and vibrato; cello when unplugged.
        if name == "solo" and "lead" in L:
            rhythms = LEAD_RHYTHMS.get(S, LEAD_RHYTHMS[8])
            for k, ln in rhythms[int(rng.integers(0, len(rhythms)))]:
                if k >= S:
                    continue
                lead_deg = int(np.clip(lead_deg + rng.choice([-2, -1, 1, 1, 2, 3]), 0, 9))
                if i == n_sec - 1 and k == 0:
                    lead_deg = len(spec.lead_scale)  # end the solo on the octave
                o, q = divmod(lead_deg, len(spec.lead_scale))
                m = spec.tonic + 24 + 12 * o + spec.lead_scale[q]
                dur = min(ln, S - k) * st * 0.95
                if spec.verse == "acoustic" and spec.chorus == "acoustic":
                    x = cello(midi_to_hz(m), dur + 0.2, rng) * 0.16
                    put(vc[:, 0], (t0 + k * st) * SR, x, 0.4)
                    put(vc[:, 1], (t0 + k * st) * SR, x, 0.6)
                else:
                    art = "b" if rng.random() < 0.35 else "v"
                    put(lead, (t0 + k * st) * SR, guitar_dry([midi_to_hz(m)], dur, art, 3, rng))

        # Vocals.
        if "vocals" in L:
            if name in ("verse", "chorus"):
                mel = verse_mel if name == "verse" else chorus_mel
                for mb, step, ln, deg in mel:
                    if mb != i % 8 or step >= S:
                        continue
                    ts, d = t0 + step * st, ln * st
                    m = deg_midi(base_v, deg)
                    chord_r = riff_chord(root, i, step) if name == "verse" and electric_verse else root
                    third = third_of(chord_r - spec.tonic)
                    tones = {(chord_r + x) % 12 for x in (0, third, 7)}
                    if (step % sub == 0 or ln >= 2 * sub) and m % 12 not in tones:  # strong notes sit on the chord
                        for dm in (-1, 1, -2, 2):
                            if (m + dm) % 12 in tones:
                                m += dm
                                break
                    v = vowels[(step + mb) % len(vowels)]
                    voice_lead.append((ts, d, m, v, 1.0))
                    if name == "chorus" or (name == "verse" and i % 8 >= 4 and sec >= 3):
                        if m % 12 in tones:  # the next chord tone above: always consonant
                            h = next(m + k for k in range(1, 9) if (m + k) % 12 in tones)
                        elif spec.harmony_semi is not None and (m + spec.harmony_semi - spec.tonic) % 12 in sc:
                            h = m + spec.harmony_semi
                        else:
                            h = deg_midi(base_v, deg + 2)
                        voice_harm.append((ts, d, h, v, 0.8))
            elif name == "bridge" and i % 2 == 0:
                third = third_of(root - spec.tonic)
                m = base_v + ((root - spec.tonic) % 12)
                if m > spec.voice_center + 7:
                    m -= 12
                d = 2 * bar_dur * 0.9
                voice_lead.append((t0 + 0.1, d, m, "u", 0.7))
                voice_harm.append((t0 + 0.1, d, m + (third if spec.harmony_semi is None else spec.harmony_semi), "o", 0.6))

        # Drums: per-section grooves with bar-to-bar variation, ghost notes, crashes and fills.
        drums_on = spec.drums != "none" and (
            name in ("verse", "chorus", "solo", "bridge") or (name == "outro" and spec.intro == "riff" and i < n_sec - 2))
        if drums_on:
            Sk = S if S in KICKS else 8
            part = "chorus" if name in ("chorus", "solo") else "bridge" if name == "bridge" else "verse"
            soft = 0.6 if name == "bridge" else 0.85 if quiet else 1.0
            kp = KICKS[Sk][part][i % 4]
            sp = SNARES[Sk][1 if name == "bridge" else 0]
            if spec.drums in ("halftime", "sludge") and S == 8:  # snare on 3 only, the kick leaves room
                sp = "....x..."
                kp = (("x.......", "x..x....", "x.......", "x.x...x.") if spec.drums == "sludge"
                      else ("x.......", "x.....x.", "x..x....", "x.....x."))[i % 4]  # fmt: skip
            kp, sp = spec.kick_pat or kp, spec.snare_pat or sp
            fill = "big" if last_in_sec and name != "outro" else "small" if i % 4 == 3 else None
            fill_from = S - (S // 2 if fill == "big" else S // 4) if fill else S
            for k, c in enumerate(kp[:S]):
                if c == "x" and k < fill_from:
                    put(drums, (t0 + k * st) * SR, kick_g(int(0.5 * SR), rng), 0.55 * soft)
                    kicks.append(t0 + k * st)
            for k, c in enumerate(sp[:S]):
                if k >= fill_from:
                    continue
                if c == "x":
                    g = 0.33 * soft
                    put(snare_bus, (t0 + k * st) * SR, (brush if spec.drums == "brush" else snare_g)(int(0.6 * SR), rng), g)
                elif name == "verse" and k % sub and rng.random() < 0.18:  # ghost note
                    put(snare_bus, (t0 + k * st) * SR, snare_g(int(0.2 * SR), rng), 0.05)
            if name in ("chorus", "solo"):  # open hats, crash every four bars
                for k in range(0, min(S, fill_from)):
                    put(cym[:, 1], (t0 + k * st) * SR, open_hat(int(0.35 * SR), rng), 0.045 if k % sub == 0 else 0.03)
            elif name == "bridge":
                for k in range(0, S, sub):
                    r = ride(int(0.8 * SR), rng)
                    put(cym[:, 0], (t0 + k * st) * SR, r, 0.04)
                    put(cym[:, 1], (t0 + k * st) * SR, r, 0.05)
            else:
                pattern = [k for k in range(S) if not (S == 12 and k % 3 == 1)]
                for k in pattern:
                    if k < fill_from:
                        acc = 1.0 if k % sub == 0 else 0.6
                        put(cym[:, 1], (t0 + k * st) * SR, hat(int(0.1 * SR), rng), 0.05 * acc * soft)
            if (i == 0 and name in ("chorus", "solo", "verse")) or (name in ("chorus", "solo") and i % 4 == 0):
                c = crash(int(2.5 * SR), rng)
                put(cym[:, 0], t0 * SR, c, 0.16 if i == 0 else 0.11)
                put(cym[:, 1], t0 * SR, c, 0.10 if i == 0 else 0.07)
                put(drums, t0 * SR, kick_g(int(0.5 * SR), rng), 0.3)
            if fill:
                kind = int(rng.integers(0, 3)) if fill == "big" else 3
                hits = np.arange(fill_from, S, 0.5)  # sixteenths
                for j, k in enumerate(hits):
                    ts = (t0 + k * st) * SR
                    if kind == 0:  # toms, high to low
                        f = (190, 150, 118, 92)[min(int(j * 4 / len(hits)), 3)]
                        tm = tom(f, int(0.5 * SR), rng)
                        pan = 0.75 - 0.5 * j / max(len(hits) - 1, 1)
                        put(cym[:, 0], ts, tm, 0.32 * (1 - pan))
                        put(cym[:, 1], ts, tm, 0.32 * pan)
                    elif kind == 1:  # snare roll, crescendo
                        put(snare_bus, ts, snare_g(int(0.3 * SR), rng), 0.12 + 0.2 * j / len(hits))
                    elif kind == 2:  # snare and floor tom, alternating
                        if j % 2:
                            tm = tom(92, int(0.5 * SR), rng)
                            put(cym[:, 0], ts, tm, 0.12)
                            put(cym[:, 1], ts, tm, 0.2)
                        else:
                            put(snare_bus, ts, snare_g(int(0.3 * SR), rng), 0.25)
                    elif j % 2 == 0 or j == len(hits) - 1:  # small pickup on the snare
                        put(snare_bus, ts, snare_g(int(0.3 * SR), rng), 0.18)
        elif name == "intro" and i == n_sec - 1 and spec.drums != "none":
            for j, k in enumerate(np.arange(S // 2, S, 0.5)):  # roll in
                put(snare_bus, (t0 + k * st) * SR, snare_g(int(0.3 * SR), rng), 0.08 + 0.2 * j / S)

    # ---------------------------------------------------------------- buses
    mix = np.zeros((total, 2))

    if clean.any():  # clean guitar: light breakup, chorus (modulated delay) on the right side
        t_all = np.arange(total) / SR
        x = lowpass(np.tanh(1.6 * clean) / 1.6, 5000)
        delay = (0.012 + 0.003 * np.sin(2 * np.pi * 0.8 * t_all)) * SR
        wet = np.interp(np.arange(total) - delay, np.arange(total), x, left=0.0)
        mix[:, 0] += 0.16 * x + 0.05 * wet
        mix[:, 1] += 0.08 * x + 0.13 * wet

    if any(g.any() for g in gtr):
        t_all = np.arange(total) / SR
        for side in (0, 1):
            x = gtr[side]
            if spec.wah == "riff":
                pos = 0.5 - 0.5 * np.cos(2 * np.pi * t_all / (sub * st * 2) + side)
                x = wah(x, pos)
            y = amp(x, spec.drive)
            mix[:, side] += y * 0.30
            mix[:, 1 - side] += y * 0.05

    if lead.any():
        t_all = np.arange(total) / SR
        x = lead
        if spec.wah in ("lead", "riff"):
            x = wah(x, 0.5 - 0.5 * np.cos(2 * np.pi * t_all / (sub * st * 4)))
        y = amp(x, spec.drive * 1.4) * 0.16
        d = int(3 * st * SR)
        echo = np.zeros(total)
        echo[d:] = lowpass(y[:-d], 2500) * 0.35
        mix[:, 0] += 0.45 * y + 0.6 * echo
        mix[:, 1] += 0.55 * y + 0.3 * echo

    bass = np.zeros(total)
    if "bass" in L and bass_ev:
        bass_ev.sort()
        for j, (ts, d, m) in enumerate(bass_ev):
            if j + 1 < len(bass_ev):
                d = min(d, bass_ev[j + 1][0] - ts)
            n = int(max(d, 0.05) * SR)
            tt = np.arange(n) / SR
            env = np.minimum(tt / 0.004, 1) * np.exp(-tt * 1.2) * np.clip((n / SR - tt) / 0.015, 0, 1)
            put(bass, ts * SR, bass_tone(midi_to_hz(m), n, "saw", ts) * env)
        bass = lowpass(np.tanh(2.2 * bass), 1100) * 0.36
        mix += bass[:, None]

    # The melody is played, not sung: a singing lead guitar, and a second guitar in harmony.
    vocal = np.zeros((total, 2))
    for events, gain, pan, seed in ((voice_lead, 1.0, 0.42, 1), (voice_harm, 0.75, 0.62, 3)):
        if not events:
            continue
        mrng = np.random.default_rng(spec.seed + seed)
        x = np.zeros(total)
        for ts, d, m, v, g in events:
            art = "b" if mrng.random() < 0.15 and d > 0.3 else "v"
            put(x, ts * SR, guitar_dry([midi_to_hz(m + 12)], d + 0.15, art, 2, mrng), g)
        y = lowpass(np.tanh(2.5 * lowpass(x, 3500)) / 2.5, 4200) * 0.55 * gain
        dly = int(3 * st * SR)
        echo = np.zeros(total)
        echo[dly:] = lowpass(y[:-dly], 2200) * 0.3
        vocal[:, 0] += (1 - pan) * y + pan * echo
        vocal[:, 1] += pan * y + (1 - pan) * echo

    kick_bus = drums
    mix += kick_bus[:, None] + 0.45 * np.stack([snare_bus, snare_bus], axis=1) + cym
    mix += ac + vc + vocal + drone[:, None]

    if "rain" in L:
        r = highpass(rng.standard_normal((2, total)), 2500).T * 0.004
        for _ in range(int(spec.duration * 6)):  # sparse droplets
            ts = rng.uniform(0, spec.duration)
            n = int(0.05 * SR)
            tt = np.arange(n) / SR
            dr = np.sin(2 * np.pi * rng.uniform(2000, 5000) * tt) * np.exp(-tt * 90) * 0.01
            put(r[:, int(rng.integers(0, 2))], ts * SR, dr)
        mix += r

    send = 0.6 * vocal + 0.5 * np.stack([snare_bus, snare_bus], axis=1) + 0.6 * ac + 0.7 * vc + 0.3 * cym
    mix += 0.45 * reverb(send, spec.reverb_s, rng)

    mix *= _env(total, 2.0, 6.0)[:, None]
    mix = np.tanh(1.1 * mix)
    mix /= np.max(np.abs(mix)) + 1e-9
    mix *= 10 ** (-1.0 / 20)
    return mix.astype(np.float32), analyze(mix, bass + drone, kicks, spec)


def analyze(mix, bass, kicks, spec: Spec) -> dict:
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
        for j in range(f0, min(f0 + 12, n_frames)):
            kick_env[j] = max(kick_env[j], np.exp(-(j - f0) / 4))
    return {
        "fps": spec.fps,
        "duration": spec.duration,
        "chord_seconds": spec.meter * spec.step * spec.chord_bars,
        "rms": env_of(mono, 8),
        "bass": env_of(bass, 3),
        "kick": [round(float(v), 4) for v in kick_env],
    }


def render_track(spec: Spec, wav_path: Path, analysis_path: Path) -> None:
    mix, analysis = synthesize(spec)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(wav_path, SR, mix)
    analysis_path.write_text(json.dumps(analysis))
