"""Ambient music synthesized from scratch with numpy.

Everything here is math: additive synthesis (sums of harmonics), Euclidean rhythms
for the arpeggio, exponential envelopes and a convolution reverb built from
decaying noise. No samples, no third-party audio, so the output is fully owned.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, fftconvolve, sosfilt

SR = 44_100


def midi_to_hz(note: float) -> float:
    return 440.0 * 2 ** ((note - 69) / 12)


def euclidean(pulses: int, steps: int, rotation: int = 0) -> list[int]:
    """Bjorklund-style Euclidean rhythm: spread `pulses` hits as evenly as possible over `steps`."""
    pattern = [1 if (i * pulses) % steps < pulses else 0 for i in range(steps)]
    return pattern[rotation:] + pattern[:rotation]


@dataclass
class Chord:
    name: str
    notes: list[int]  # MIDI notes
    ratio: tuple[int, int]  # Lissajous frequency ratio used by the visuals


# D Dorian progression: Dm9 – Cmaj7 – Bbmaj7 – C6. Each chord maps to a small-integer
# frequency ratio, the same idea as consonance: simple ratios sound (and look) stable.
DEFAULT_PROGRESSION = [
    Chord("Dm9", [50, 57, 60, 64, 65], (3, 2)),
    Chord("Cmaj7", [48, 55, 59, 62, 64], (4, 3)),
    Chord("Bbmaj7", [46, 53, 57, 60, 62], (5, 4)),
    Chord("C6", [48, 55, 57, 60, 64], (5, 3)),
]


@dataclass
class TrackSpec:
    duration: float = 64.0  # seconds
    bpm: float = 72.0
    chord_seconds: float = 8.0
    progression: list[Chord] = field(default_factory=lambda: list(DEFAULT_PROGRESSION))
    seed: int = 7
    fps: int = 30  # frame rate of the analysis exported for the visuals


def _env(n: int, attack: float, release: float) -> np.ndarray:
    t = np.arange(n) / SR
    a = np.clip(t / max(attack, 1e-4), 0, 1)
    r = np.clip((n / SR - t) / max(release, 1e-4), 0, 1)
    return np.minimum(a, r) ** 2


def pad_voice(freq: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Warm pad: a few detuned partials with 1/k^1.5 amplitudes and slow vibrato."""
    t = np.arange(n) / SR
    out = np.zeros(n)
    for k in range(1, 7):
        detune = 1 + rng.uniform(-0.002, 0.002)
        lfo = 1 + 0.0015 * np.sin(2 * np.pi * rng.uniform(0.08, 0.2) * t + rng.uniform(0, 2 * np.pi))
        phase = 2 * np.pi * np.cumsum(freq * k * detune * lfo) / SR
        out += np.sin(phase) / k**1.5
    return out


def pluck(freq: float, n: int) -> np.ndarray:
    """Soft bell-like pluck: two inharmonic partials with exponential decay."""
    t = np.arange(n) / SR
    decay = np.exp(-t * 3.5)
    return (np.sin(2 * np.pi * freq * t) + 0.35 * np.sin(2 * np.pi * freq * 2.76 * t) * np.exp(-t * 6)) * decay


def lowpass(x: np.ndarray, cutoff: float) -> np.ndarray:
    sos = butter(2, cutoff, btype="low", fs=SR, output="sos")
    return sosfilt(sos, x)


def reverb(x: np.ndarray, seconds: float, rng: np.random.Generator) -> np.ndarray:
    """Convolution with exponentially decaying stereo noise: a cheap, smooth hall."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    ir = rng.standard_normal((n, 2)) * np.exp(-t * (6.9 / seconds))[:, None]
    ir = lowpass(ir.T, 5000).T
    ir /= np.sqrt((ir**2).sum(axis=0))
    wet = np.stack([fftconvolve(x[:, c], ir[:, c])[: len(x)] for c in range(2)], axis=1)
    return wet


def synthesize(spec: TrackSpec) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(spec.seed)
    total = int(spec.duration * SR)
    mix = np.zeros((total, 2))
    chord_n = int(spec.chord_seconds * SR)
    n_chords = int(np.ceil(spec.duration / spec.chord_seconds))

    # Pads and bass: one chord per block with long overlapping crossfades.
    for i in range(n_chords):
        chord = spec.progression[i % len(spec.progression)]
        start = i * chord_n
        n = min(chord_n + int(2.5 * SR), total - start)
        if n <= 0:
            break
        env = _env(n, attack=2.5, release=2.5)
        for j, note in enumerate(chord.notes[1:]):
            voice = pad_voice(midi_to_hz(note), n, rng) * env * 0.09
            pan = 0.5 + 0.35 * np.sin(j * 1.7)
            mix[start : start + n, 0] += voice * (1 - pan)
            mix[start : start + n, 1] += voice * pan
        bass = np.sin(2 * np.pi * midi_to_hz(chord.notes[0] - 12) * np.arange(n) / SR) * env * 0.16
        mix[start : start + n] += bass[:, None]

    # Arpeggio on a Euclidean rhythm E(5,16), notes walking up the current chord.
    step = 60 / spec.bpm / 4  # sixteenth note
    pattern = euclidean(5, 16)
    n_steps = int(spec.duration / step)
    pluck_n = int(1.6 * SR)
    hits = 0
    for s in range(n_steps):
        if not pattern[s % 16]:
            continue
        t0 = s * step
        chord = spec.progression[int(t0 // spec.chord_seconds) % len(spec.progression)]
        note = chord.notes[1:][hits % (len(chord.notes) - 1)] + 12
        hits += 1
        start = int(t0 * SR)
        n = min(pluck_n, total - start)
        if n <= 0:
            continue
        p = pluck(midi_to_hz(note), n) * 0.06
        pan = 0.5 + 0.4 * np.sin(hits * 0.9)
        mix[start : start + n, 0] += p * (1 - pan)
        mix[start : start + n, 1] += p * pan

    # Soft filtered noise, like distant rain.
    noise = lowpass(rng.standard_normal((total, 2)).T, 900).T * 0.012
    mix += noise

    # Reverb and master.
    mix = 0.65 * mix + 0.55 * reverb(mix, 3.5, rng)
    fade = _env(total, attack=4.0, release=5.0)[:, None]
    mix *= fade
    mix /= np.max(np.abs(mix)) + 1e-9
    mix *= 10 ** (-1.0 / 20)  # peak at -1 dBFS

    analysis = analyze(mix, spec)
    return mix.astype(np.float32), analysis


def analyze(mix: np.ndarray, spec: TrackSpec) -> dict:
    """Per-frame loudness (RMS) and chord index, consumed by the Manim scene."""
    hop = SR // spec.fps
    mono = mix.mean(axis=1)
    n_frames = len(mono) // hop
    rms = np.sqrt(np.array([np.mean(mono[i * hop : (i + 1) * hop] ** 2) for i in range(n_frames)]))
    # Smooth so the visuals breathe instead of flicker.
    k = np.exp(-np.arange(-30, 31) ** 2 / (2 * 8**2))
    rms = np.convolve(rms, k / k.sum(), mode="same")
    rms = (rms - rms.min()) / (rms.max() - rms.min() + 1e-9)
    chords = [int((i / spec.fps) // spec.chord_seconds) % len(spec.progression) for i in range(n_frames)]
    return {
        "fps": spec.fps,
        "duration": spec.duration,
        "chord_seconds": spec.chord_seconds,
        "ratios": [list(c.ratio) for c in spec.progression],
        "chord_names": [c.name for c in spec.progression],
        "rms": [round(float(x), 4) for x in rms],
        "chord": chords,
    }


def render_track(spec: TrackSpec, wav_path: Path, analysis_path: Path) -> None:
    mix, analysis = synthesize(spec)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(wav_path, SR, mix)
    analysis_path.write_text(json.dumps(analysis))
