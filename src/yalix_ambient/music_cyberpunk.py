"""Cyberpunk music: synthwave and darksynth built from oscillators, no samples.

Styles of genres, never anyone's songs. Every episode chooses:
- a diatonic progression (scale degrees) in a minor mode, two bars per chord;
- drums: four-on-the-floor, half-time, broken beat, or none; snare with a gated reverb;
- bass: rolling sixteenth-note octaves, eighth-note pulse, detuned 'reese', or FM growl;
- a sixteenth-note arpeggio whose filter opens over the song, with dotted-eighth ping-pong delay;
- a pulse-wave lead that plays a two-phrase motif snapped to the chord tones;
- a supersaw pad that pumps under the kick, and optional rain.

Form, in bars: intro (pad, arp) → verse (drums, bass) → drop (everything) → break (pad, lead)
→ drop → outro.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import sawtooth, square

from yalix_ambient.music import SR, _env, euclidean, lowpass, midi_to_hz, reverb
from yalix_ambient.music_dark import bandpass, hat, highpass, kick, snare

AEOLIAN = (0, 2, 3, 5, 7, 8, 10)
DORIAN = (0, 2, 3, 5, 7, 9, 10)
PHRYGIAN = (0, 1, 3, 5, 7, 8, 10)
HARMONIC_MINOR = (0, 2, 3, 5, 7, 8, 11)


@dataclass
class Spec:
    duration: float = 194.159
    bpm: float = 110.0
    tonic: int = 45  # A2
    scale: tuple[int, ...] = AEOLIAN
    progression: tuple[int, ...] = (0, 5, 2, 6)  # scale degrees: i–VI–III–VII
    bars_per_chord: int = 2
    drums: str = "four"  # four | halftime | break | none
    bass: str = "rolling"  # rolling | pulse | reese | fm
    arp_rate: int = 16  # 16 = sixteenths, 8 = eighths
    arp_shape: str = "up"  # up | updown | pendulum
    layers: tuple[str, ...] = ("pad", "arp", "lead", "bass")  # + rain
    cutoff: tuple[float, float] = (900.0, 4200.0)  # arp filter, start -> peak
    reverb_s: float = 3.2
    seed: int = 1
    fps: int = 30

    @property
    def sixteenth(self) -> float:
        return 60 / self.bpm / 4


# ------------------------------------------------------------------ instruments


def supersaw(freq: float, t: np.ndarray, voices: int = 5, spread: float = 14.0) -> np.ndarray:
    cents = np.linspace(-spread, spread, voices)
    return sum(sawtooth(2 * np.pi * freq * 2 ** (c / 1200) * t + c) for c in cents) / voices


def pulse(freq: float, t: np.ndarray, duty: float | np.ndarray = 0.5) -> np.ndarray:
    return square(2 * np.pi * freq * t, duty=duty)


def chord(spec: Spec, degree: int) -> list[int]:
    """Diatonic triad on a scale degree, as MIDI notes from the tonic."""
    out = []
    for k in (0, 2, 4):
        d = degree + k
        octv, idx = divmod(d, len(spec.scale))
        out.append(spec.tonic + 12 * octv + spec.scale[idx])
    return out


def synth_kick(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / SR
    f = 48 + 120 * np.exp(-t * 40)  # punchier pitch drop than the industrial kick
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 7)
    return np.tanh(2.2 * body) * 0.9 + kick(n, rng) * 0.25


# ------------------------------------------------------------------ composition


def form(n_bars: int) -> list[str]:
    plan = [("intro", 8), ("verse", 16), ("drop", 16), ("break", 8), ("drop", 16)]
    out: list[str] = []
    for name, bars in plan:
        out += [name] * bars
    out = out[: max(n_bars - 6, 0)]
    out += ["drop"] * max(n_bars - 6 - len(out), 0)
    return out + ["outro"] * (n_bars - len(out))


def synthesize(spec: Spec) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(spec.seed)
    total = int(spec.duration * SR)
    t_all = np.arange(total) / SR
    L = set(spec.layers)
    s16 = spec.sixteenth
    bar = 16 * s16
    n_bars = math.ceil(spec.duration / bar)
    sections = form(n_bars)

    def put(buf, s, x, gain=1.0):
        s = int(s)
        n = min(len(x), total - s)
        if n > 0 and s >= 0:
            buf[s : s + n] += x[:n] * gain

    def degree_of(b: int) -> int:
        return spec.progression[(b // spec.bars_per_chord) % len(spec.progression)]

    has_drums = lambda sec: spec.drums != "none" and sec in ("verse", "drop")  # noqa: E731
    has_bass = lambda sec: "bass" in L and sec in ("verse", "drop", "outro")  # noqa: E731

    # ---------------------------------------------------------------- drums
    drums, gated = np.zeros((total, 2)), np.zeros(total)
    kicks: list[float] = []
    k_hit = synth_kick(int(0.5 * SR), rng)
    for b, sec in enumerate(sections):
        if not has_drums(sec):
            continue
        t0 = b * bar
        if spec.drums == "four":
            k_pos, s_pos = (0, 4, 8, 12), (4, 12)
        elif spec.drums == "halftime":
            k_pos, s_pos = (0, 10), (8,)
        else:  # broken beat
            k_pos, s_pos = (0, 6, 10) if b % 2 == 0 else (0, 3, 10), (4, 12)
        for i in k_pos:
            put(drums[:, 0], (t0 + i * s16) * SR, k_hit, 0.62)
            put(drums[:, 1], (t0 + i * s16) * SR, k_hit, 0.62)
            kicks.append(t0 + i * s16)
        for i in s_pos:
            sn = snare(int(0.4 * SR), rng)
            put(drums[:, 0], (t0 + i * s16) * SR, sn, 0.28)
            put(drums[:, 1], (t0 + i * s16) * SR, sn, 0.28)
            put(gated, (t0 + i * s16) * SR, sn, 1.0)
        hat_steps = range(16) if spec.drums != "halftime" else range(0, 16, 2)
        for i in hat_steps:
            g = 0.045 if i % 4 == 2 else 0.022
            pan = 0.35 if i % 2 else 0.65
            hh = hat(int(0.08 * SR), rng)
            put(drums[:, 0], (t0 + i * s16) * SR, hh, g * (1 - pan))
            put(drums[:, 1], (t0 + i * s16) * SR, hh, g * pan)
        if spec.drums == "four" and sec == "drop":  # open hats on the off-beats
            for i in (2, 6, 10, 14):
                oh = highpass(rng.standard_normal(int(0.3 * SR)), 6000) * np.exp(-np.arange(int(0.3 * SR)) / SR * 12)
                put(drums[:, 0], (t0 + i * s16) * SR, oh, 0.03)
                put(drums[:, 1], (t0 + i * s16) * SR, oh, 0.03)

    # Gated reverb: a big hall cut off after a quarter of a second, the 80s snare.
    if gated.any():
        g_wet = reverb(np.stack([gated, gated], axis=1), 2.5, rng)
        gate = np.zeros(total)
        for b, sec in enumerate(sections):
            if has_drums(sec):
                for i in range(16):
                    s = int((b * bar + i * s16) * SR)
                    gate[s : s + int(0.25 * SR)] = 1.0
        gate = lowpass(gate, 60)
        drums += g_wet * gate[:, None] * 0.55

    duck = np.ones(total)
    for kt in kicks:
        s = int(kt * SR)
        n = min(int(0.45 * SR), total - s)
        duck[s : s + n] = np.minimum(duck[s : s + n], 1 - 0.6 * np.exp(-np.arange(n) / SR / 0.12))

    # ---------------------------------------------------------------- bass
    low = np.zeros(total)
    if "bass" in L:
        for b, sec in enumerate(sections):
            if not has_bass(sec):
                continue
            t0 = b * bar
            root = chord(spec, degree_of(b))[0] - 12
            if spec.bass in ("rolling", "pulse"):
                step = 1 if spec.bass == "rolling" else 2
                for i in range(0, 16, step):
                    note = root + (12 if (i // step) % 2 else 0)
                    n = int(step * s16 * 0.9 * SR)
                    tt = np.arange(n) / SR
                    x = sawtooth(2 * np.pi * midi_to_hz(note) * tt) + 0.7 * np.sin(2 * np.pi * midi_to_hz(note) * tt)
                    env = np.minimum(tt / 0.003, 1) * np.exp(-tt * (9 if step == 1 else 5))
                    put(low, (t0 + i * s16) * SR, lowpass(x * env, 900), 0.55)
            elif spec.bass == "reese":
                n = int(bar * SR)
                tt = np.arange(n) / SR
                f = midi_to_hz(root)
                x = sawtooth(2 * np.pi * f * tt) + sawtooth(2 * np.pi * f * 1.012 * tt) + 0.8 * np.sin(2 * np.pi * f * tt)
                cut = 380 + 260 * np.sin(2 * np.pi * (b % 4 + tt / bar) / 4)
                x = lowpass(x, float(cut.mean())) * _env(n, 0.02, 0.15)
                put(low, t0 * SR, np.tanh(1.8 * x), 0.45)
            else:  # fm growl on eighths with a syncopated skip
                for i in [j for j, h in enumerate(euclidean(5, 8, 1)) if h]:
                    n = int(2 * s16 * 0.95 * SR)
                    tt = np.arange(n) / SR
                    f = midi_to_hz(root + (12 if i in (3, 6) else 0))
                    index = 1.0 + 3.5 * np.exp(-tt * 6)
                    x = np.sin(2 * np.pi * f * tt + index * np.sin(2 * np.pi * f * tt)) + 0.6 * np.sin(np.pi * f * tt)
                    env = np.minimum(tt / 0.004, 1) * np.exp(-tt * 4)
                    put(low, (t0 + 2 * i * s16) * SR, np.tanh(2.0 * x) * env, 0.5)
        # sub under every bar so the low end holds when the bass is busy
        for b, sec in enumerate(sections):
            if has_bass(sec):
                n = int(bar * SR)
                tt = np.arange(n) / SR
                f = midi_to_hz(chord(spec, degree_of(b))[0] - 24)
                put(low, b * bar * SR, np.sin(2 * np.pi * f * tt) * _env(n, 0.01, 0.05), 0.22)

    wide, dry = np.zeros((total, 2)), np.zeros((total, 2))
    progress = np.clip(np.arange(n_bars) / max(n_bars - 8, 1), 0, 1)

    # ---------------------------------------------------------------- pad
    if "pad" in L:
        for b0 in range(0, n_bars, spec.bars_per_chord):
            sec = sections[b0]
            n = int((spec.bars_per_chord * bar + 1.5) * SR)
            tt = np.arange(n) / SR
            notes = chord(spec, degree_of(b0))
            x = sum(supersaw(midi_to_hz(m + 12), tt) for m in notes) / 3
            cut = 1200 if sec in ("intro", "break", "outro") else 2200
            x = lowpass(x, cut) * _env(n, 0.6, 1.2) * (0.11 if sec != "break" else 0.14)
            s = int(b0 * bar * SR)
            m = min(n, total - s)
            if m > 0:
                wide[s : s + m] += np.stack([x, np.roll(x, int(0.011 * SR))], axis=1)[:m]

    # ---------------------------------------------------------------- arpeggio
    if "arp" in L:
        arp = np.zeros((total, 2))
        k = 0
        step = 16 // spec.arp_rate
        for b, sec in enumerate(sections):
            if sec in ("break",) or b >= n_bars - 2:
                continue
            notes = chord(spec, degree_of(b))
            line = [m + 12 for m in notes] + [m + 24 for m in notes]
            if spec.arp_shape == "updown":
                line = line + line[-2:0:-1]
            elif spec.arp_shape == "pendulum":
                line = [line[0], line[3], line[1], line[4], line[2], line[5]]
            lo, hi = spec.cutoff
            cut = lo + (hi - lo) * progress[b] * (1.0 if sec == "drop" else 0.6)
            for i in range(0, 16, step):
                note = line[k % len(line)]
                k += 1
                n = int(step * s16 * 1.6 * SR)
                tt = np.arange(n) / SR
                x = pulse(midi_to_hz(note), tt, 0.3) * 0.6 + sawtooth(2 * np.pi * midi_to_hz(note) * tt) * 0.4
                x = lowpass(x * np.exp(-tt * 14) * np.minimum(tt / 0.002, 1), cut) * 0.07
                pan = 0.5 + 0.25 * np.sin(k * 0.9)
                put(arp[:, 0], (b * bar + i * s16) * SR, x, 1 - pan)
                put(arp[:, 1], (b * bar + i * s16) * SR, x, pan)
        d = int(3 * s16 * SR)  # dotted eighth
        echo = np.zeros_like(arp)
        for r in range(1, 4):
            ch = r % 2
            echo[r * d :, ch] += lowpass(arp[: total - r * d, 1 - ch], 3000) * 0.42**r
        dry += arp + echo

    # ---------------------------------------------------------------- lead
    if "lead" in L:
        lead = np.zeros((total, 2))
        motifs = []
        for _ in range(2):  # phrase A and phrase B, two bars each
            hits = [i for i, h in enumerate(euclidean(int(rng.integers(5, 8)), 32, int(rng.integers(0, 4)))) if h]
            degs = np.cumsum(rng.choice([-2, -1, 1, 2, 3], len(hits)))
            motifs.append(list(zip(hits, degs - degs[0] + int(rng.integers(2, 5)))))
        break_start = sections.index("break") if "break" in sections else n_bars
        for b in range(0, n_bars, 2):
            sec = sections[b]
            if not (sec == "break" or (sec == "drop" and b > break_start)):  # the lead enters at the break
                continue
            motif = motifs[(b // 2) % 2]
            chord_now = chord(spec, degree_of(b))
            for j, (i, dg) in enumerate(motif):
                nxt = motif[j + 1][0] if j + 1 < len(motif) else 32
                octv, idx = divmod(int(dg), len(spec.scale))
                note = spec.tonic + 24 + 12 * octv + spec.scale[idx]
                if i % 8 == 0:  # strong beats land on the nearest chord tone
                    note = min((c + 24 + 12 * o for c in chord_now for o in (0, 1)), key=lambda c: abs(c - note))
                n = int((nxt - i) * s16 * 0.95 * SR)
                tt = np.arange(n) / SR
                f = midi_to_hz(note) * (1 + 0.004 * np.sin(2 * np.pi * 5.5 * tt) * np.minimum(tt / 0.3, 1))
                ph = 2 * np.pi * np.cumsum(f) / SR
                duty = 0.5 + 0.2 * np.sin(2 * np.pi * 0.7 * tt)
                x = square(ph, duty=duty) * 0.55 + np.sin(ph) * 0.45
                x = lowpass(x * _env(n, 0.01, 0.12), 3200) * 0.075
                put(lead[:, 0], (b * bar + i * s16) * SR, x, 0.55)
                put(lead[:, 1], (b * bar + i * s16) * SR, x, 0.45)
        d = int(6 * s16 * SR)  # dotted quarter
        tail = np.zeros_like(lead)
        for r in range(1, 4):
            tail[r * d :] += lowpass(lead[: total - r * d][:, ::-1].T, 2500).T * 0.38**r
        dry += lead + tail

    rain = np.zeros((total, 2))
    if "rain" in L:
        r = bandpass(rng.standard_normal((2, total)), 900, 7000).T * 0.010
        drops = (rng.random((total, 2)) < 0.0004) * rng.uniform(0.2, 1.0, (total, 2))
        rain = r + highpass(drops.T, 2500).T * 0.05

    stereo = drums + (low * duck)[:, None] + wide * duck[:, None] + dry
    wet = reverb(0.4 * stereo + 0.6 * dry + 0.5 * wide, spec.reverb_s, rng)
    hiss = lowpass(rng.standard_normal((2, total)), 6000).T * 0.004  # tape hiss
    mix = 0.8 * stereo + 0.45 * wet + rain + hiss
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
        "chord_seconds": bar * spec.bars_per_chord,
        "rms": env_of(mono, 8),
        "bass": env_of(bass, 3),
        "kick": [round(float(v), 4) for v in kick_env],
    }


def render_track(spec: Spec, wav_path: Path, analysis_path: Path) -> None:
    mix, analysis = synthesize(spec)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(wav_path, SR, mix)
    analysis_path.write_text(json.dumps(analysis))
