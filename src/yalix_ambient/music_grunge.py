"""Grunge series music: sludgy, melancholic, Seattle-style (the genre, never anyone's songs).

Hallmarks synthesised here:
- down-tuned heavy guitars (drop-D / E♭ register), double-tracked hard left and right, with
  bends and slides into riff notes; one amp (tanh) and cabinet per side, so notes interact;
- a two-voice vocal harmony without lyrics: a lead line plus a second voice a diatonic third,
  fourth or fifth away (or a fixed, dissonant interval), formant-synthesised with portamento,
  delayed vibrato and vowel morphing. The verse and chorus melodies are composed once per
  song and repeat, so every piece has its own hook;
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
PENTATONIC = (0, 3, 5, 7, 10)

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
        if name in ("chorus",):
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
    lead = np.zeros(total)
    bass_ev: list[tuple[float, float, int]] = []
    ac = np.zeros((total, 2))
    vc = np.zeros((total, 2))  # cello
    drums, snare_bus, cym = np.zeros(total), np.zeros(total), np.zeros((total, 2))
    kicks: list[float] = []
    voice_lead, voice_harm = [], []
    drone = np.zeros(total)

    base_v = spec.tonic + 12 * round((spec.voice_center - spec.tonic) / 12)
    S0 = spec.meter
    verse_mel = compose_melody(rng, 4, S0, sub, (4, 7), 0)
    chorus_mel = compose_melody(rng, 4, S0, sub, (3, 5), 4)
    vowels = spec.vowels

    def guitar(t0, dur, freqs, art, gain):
        for side in (0, 1):
            jit = rng.normal(0, 0.006) if side else 0.0
            x = guitar_dry(freqs, dur, art, 6 + 4 * side, rng)
            put(gtr[side], (t0 + max(jit, 0)) * SR, x, gain)

    def pick_bar(t0, S, root, gain, strum=False):
        third = third_of(root - spec.tonic)
        voicing = [root + 12, root + 19, root + 24, root + 24 + third, root + 31]
        if strum:
            for i in range(0, S, sub):
                down = (i // sub) % 2 == 0
                strings = voicing if down else voicing[::-1][:4]
                for j, m in enumerate(strings):
                    p = acoustic_pluck(midi_to_hz(m), int(1.6 * SR), rng)
                    pan = 0.3 + 0.4 * j / len(strings)
                    s = (t0 + i * st + j * 0.012) * SR
                    g = gain * (0.16 if down else 0.10)
                    put(ac[:, 0], s, p, g * (1 - pan))
                    put(ac[:, 1], s, p, g * pan)
            return
        for k, h in enumerate(euclidean(min(6, S), S)):
            if h:
                m = voicing[(k * 2 + (k // 3)) % len(voicing)]
                p = acoustic_pluck(midi_to_hz(m), int(2.2 * SR), rng)
                pan = 0.25 + 0.5 * ((k * 3) % 5) / 4
                put(ac[:, 0], (t0 + k * st) * SR, p, gain * 0.2 * (1 - pan))
                put(ac[:, 1], (t0 + k * st) * SR, p, gain * 0.2 * pan)

    # ---------------------------------------------------------------- bars
    lead_deg = 0
    n_bars = len(bars)
    for b, bar in enumerate(bars):
        t0, S, name, i, sec = bar
        nxt = bars[b + 1] if b + 1 < n_bars else None
        last_in_sec = nxt is None or nxt[4] != sec
        root = spec.tonic + root_off(bar)
        en = ENERGY[name]
        bar_dur = S * st

        riff_now = (name in ("verse", "solo") and spec.verse in ("electric", "both")) or (
            name in ("intro", "outro") and spec.intro == "riff")
        if riff_now:
            cyc = i % spec.riff_bars
            for step, ln, iv, art in spec.riff:
                bi, ls = divmod(step, S0)
                if bi != cyc or ls >= S:
                    continue
                m = root + iv
                freqs = {"p": [m, m + 7, m + 12], "m": [m, m + 7, m + 12], "t": [m, m + 6]}.get(art, [m])
                guitar(t0 + ls * st, ln * st * 0.95, [midi_to_hz(x) for x in freqs], art, en)
                bass_ev.append((t0 + ls * st, ln * st * 0.92, m - 12))
        if name == "chorus" and spec.chorus in ("electric", "both"):
            chord = [midi_to_hz(root + x) for x in (0, 7, 12)]
            half = (S // 2 // sub) * sub or S
            guitar(t0, half * st * 0.98, chord, "p", 1.0)
            guitar(t0 + half * st, (S - half) * st * 0.98, chord, "p", 0.9)
        if name in ("chorus", "solo", "verse") and not riff_now or name == "chorus":
            for k in range(0, S, sub):  # bass pumps the beat when there is no riff to follow
                if name == "verse" and k % (2 * sub):
                    continue
                bass_ev.append((t0 + k * st, sub * st * 0.9, root - 12))
        if name == "bridge":
            bass_ev.append((t0, bar_dur * 0.95, root - 12))
            if spec.chorus != "acoustic" and "acoustic" not in L:
                guitar(t0, bar_dur * 0.98, [midi_to_hz(root + x) for x in (0, 7, 12)], "v", 0.45)

        acoustic_now = (
            (name == "verse" and spec.verse in ("acoustic", "both"))
            or (name in ("intro", "outro") and spec.intro == "acoustic")
            or (name == "bridge" and "acoustic" in L)
            or (name == "solo" and spec.verse == "acoustic")
        )
        if acoustic_now:
            pick_bar(t0, S, root, 1.0 if spec.verse == "acoustic" else 0.7)
        if name == "chorus" and spec.chorus in ("acoustic", "both"):
            pick_bar(t0, S, root, 1.0, strum=True)

        if "cello" in L and name in ("chorus", "bridge", "outro") and i % spec.chord_bars == 0:
            third = third_of(root - spec.tonic)
            line = (root + 12, root + 12 + third, root + 19)
            m = line[(i // spec.chord_bars) % 3]
            x = cello(midi_to_hz(m), spec.chord_bars * bar_dur * 0.98, rng) * 0.12
            put(vc[:, 0], t0 * SR, x, 0.55)
            put(vc[:, 1], t0 * SR, x, 0.45)

        if ("drone" in L or spec.intro == "drone") and name in ("intro", "bridge", "outro") and i % 2 == 0:
            n = int(2 * bar_dur * SR) + SR
            tt = np.arange(n) / SR
            f0 = midi_to_hz(spec.tonic - 12)
            x = lowpass(sum(2 * ((f0 * r * tt * (1 + d)) % 1) - 1 for r in (1, 1.5, 2) for d in (-0.002, 0.002)), 400)
            put(drone, t0 * SR, x * _env(n, 1.5, 1.5), 0.05)

        # Lead: wah guitar when plugged in, cello when unplugged.
        if name == "solo" and "lead" in L:
            hits = euclidean(int(rng.integers(3, 6)), S, int(rng.integers(0, 3)))
            on = [k for k, h in enumerate(hits) if h]
            for j, k in enumerate(on):
                end = on[j + 1] if j + 1 < len(on) else S
                lead_deg = int(np.clip(lead_deg + rng.choice([-2, -1, 1, 1, 2, 3]), 0, 9))
                o, q = divmod(lead_deg, len(PENTATONIC))
                m = spec.tonic + 24 + 12 * o + PENTATONIC[q]
                dur = (end - k) * st * 0.95
                if spec.verse == "acoustic":
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
                    if mb != i % 4 or step >= S:
                        continue
                    ts, d = t0 + step * st, ln * st
                    m = deg_midi(base_v, deg)
                    tones = {(root + x) % 12 for x in (0, third_of(root - spec.tonic), 7)}
                    if ln >= 2 * sub and m % 12 not in tones:  # long notes settle on the chord
                        for dm in (-1, 1, -2, 2):
                            if (m + dm) % 12 in tones:
                                m += dm
                                break
                    v = vowels[(step + mb) % len(vowels)]
                    voice_lead.append((ts, d, m, v, 1.0))
                    if name == "chorus" or (name == "verse" and i % 4 >= 2 and sec >= 3):
                        h = m + spec.harmony_semi if spec.harmony_semi is not None else deg_midi(base_v, deg + spec.harmony)
                        voice_harm.append((ts, d, h, v, 0.8))
            elif name in ("bridge", "outro") and i % 2 == 0 and (name == "bridge" or i < 4):
                third = third_of(root - spec.tonic)
                m = base_v + ((root - spec.tonic) % 12) + (0 if name == "bridge" else 7)
                if m > spec.voice_center + 7:
                    m -= 12
                d = 2 * bar_dur * 0.9
                voice_lead.append((t0 + 0.1, d, m, "u", 0.7))
                voice_harm.append((t0 + 0.1, d, m + (third if spec.harmony_semi is None else spec.harmony_semi), "o", 0.6))
            elif name == "intro" and spec.intro == "drone" and i >= 2 and i % 2 == 0:
                voice_lead.append((t0, 2 * bar_dur * 0.9, base_v, "u", 0.6))

        # Drums.
        drums_on = spec.drums != "none" and (
            name in ("verse", "chorus", "solo", "bridge") or (name == "outro" and spec.intro == "riff" and i < 4))
        if drums_on:
            style = spec.drums
            if name == "bridge":
                style = "brush" if style == "brush" else "halftime"
            soft = 0.6 if name == "bridge" else 1.0
            kp, sp = spec.kick_pat, spec.snare_pat
            if style == "halftime" or style == "sludge" or kp is None or name == "bridge":
                kp, sp = {
                    "rock": ("x..xx...", "..x...x."),
                    "halftime": ("x.....x.", "....x..."),
                    "sludge": ("x..x....", "....x..."),
                    "shuffle": ("x......x....", "...x.....x.."),
                    "brush": ("x...x...", "..x...x."),
                }.get(style, ("x..xx...", "..x...x."))
                if len(kp) != S:  # odd meters without an explicit pattern
                    kp = "".join("x" if k in (0, S // 2) else "." for k in range(S))
                    sp = "".join("x" if k in (int(S * 0.3), int(S * 0.78)) else "." for k in range(S))
            for k, c in enumerate(kp[:S]):
                if c == "x":
                    put(drums, (t0 + k * st) * SR, kick_g(int(0.5 * SR), rng), 0.55 * soft)
                    kicks.append(t0 + k * st)
            fill = last_in_sec and name != "outro" and S >= 6
            for k, c in enumerate(sp[:S]):
                if c == "x" and not (fill and k >= S - 3):
                    if style == "brush":
                        put(snare_bus, (t0 + k * st) * SR, brush(int(0.4 * SR), rng), 0.35 * soft)
                    else:
                        put(snare_bus, (t0 + k * st) * SR, snare_g(int(0.6 * SR), rng), 0.33 * soft)
            # cymbals
            if style == "brush":
                for k in range(0, S, sub):
                    put(snare_bus, (t0 + k * st) * SR, brush(int(0.3 * SR), rng), 0.08)
            elif style == "sludge" or (name == "chorus" and style != "shuffle"):
                for k in range(0, S, sub):
                    r = ride(int(0.8 * SR), rng)
                    put(cym[:, 0], (t0 + k * st) * SR, r, 0.05)
                    put(cym[:, 1], (t0 + k * st) * SR, r, 0.07)
            else:
                pattern = range(S) if style != "shuffle" else [k for k in range(S) if k % 3 != 1]
                for k in pattern:
                    acc = 1.0 if k % sub == 0 else 0.6
                    put(cym[:, 1], (t0 + k * st) * SR, hat(int(0.1 * SR), rng), 0.05 * acc * soft)
            if i == 0 and name in ("chorus", "solo", "verse"):
                c = crash(int(2.5 * SR), rng)
                put(cym[:, 0], t0 * SR, c, 0.16)
                put(cym[:, 1], t0 * SR, c, 0.10)
            if fill:  # tom fill into the next section
                for j, k in enumerate(range(S - 3, S)):
                    tm = tom((175, 128, 92)[j], int(0.6 * SR), rng)
                    pan = (0.7, 0.5, 0.3)[j]
                    put(cym[:, 0], (t0 + k * st) * SR, tm, 0.35 * (1 - pan))
                    put(cym[:, 1], (t0 + k * st) * SR, tm, 0.35 * pan)
        elif name == "intro" and i == len([x for x in bars if x[4] == sec]) - 1 and spec.drums != "none":
            for j, k in enumerate(range(max(S - 4, 0), S)):  # roll in
                put(snare_bus, (t0 + k * st) * SR, snare_g(int(0.4 * SR), rng), 0.12 + 0.06 * j)

    # ---------------------------------------------------------------- buses
    mix = np.zeros((total, 2))

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

    vocal = np.zeros((total, 2))
    if voice_lead:
        a = sing(voice_lead, total, np.random.default_rng(spec.seed + 1))
        b = sing([(t + 0.012, d, m, v, g) for t, d, m, v, g in voice_lead], total, np.random.default_rng(spec.seed + 2), 7)
        vocal[:, 0] += 0.20 * a + 0.12 * b
        vocal[:, 1] += 0.12 * a + 0.20 * b
    if voice_harm:
        h = sing(voice_harm, total, np.random.default_rng(spec.seed + 3), -5)
        vocal[:, 0] += 0.09 * h
        vocal[:, 1] += 0.17 * h

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
