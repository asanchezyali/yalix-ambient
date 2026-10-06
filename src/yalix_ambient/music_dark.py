"""Dark, slow industrial track: heavy but relaxing. Synthesized from scratch with numpy.

E Phrygian (the flat second gives the dark colour), 74 BPM half-time:
- kick: pitch-swept sine with a soft clip; snare on beat 3 drowned in reverb;
- bass: distorted sawtooth riff in eighths with a slowly sweeping low-pass;
- wall: detuned, distorted power chords far in the background;
- sidechain: bass and pads duck under every kick, the industrial "breathing".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sawtooth, sosfilt

from yalix_ambient.music import SR, _env, euclidean, lowpass, midi_to_hz, reverb


def bandpass(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return sosfilt(butter(2, [lo, hi], btype="band", fs=SR, output="sos"), x)


def highpass(x: np.ndarray, cutoff: float) -> np.ndarray:
    return sosfilt(butter(2, cutoff, btype="high", fs=SR, output="sos"), x)


@dataclass
class DarkSpec:
    duration: float = 180.0
    bpm: float = 74.0
    roots: tuple[int, ...] = (40, 41, 40, 38)  # E2 F2 E2 D2, two bars each
    bars_per_chord: int = 2
    seed: int = 13
    fps: int = 30

    @property
    def beat(self) -> float:
        return 60 / self.bpm

    @property
    def chord_seconds(self) -> float:
        return 4 * self.beat * self.bars_per_chord


def kick(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / SR
    f = 44 + 90 * np.exp(-t * 32)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 5.5)
    click = rng.standard_normal(n) * np.exp(-t * 400) * 0.3
    return np.tanh(1.8 * (body + click))


def snare(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / SR
    noise = bandpass(rng.standard_normal(n), 1400, 6000) * np.exp(-t * 13)
    body = np.sin(2 * np.pi * 185 * t) * np.exp(-t * 22)
    return np.tanh(1.5 * (0.8 * noise + 0.6 * body))


def hat(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / SR
    return highpass(rng.standard_normal(n), 7000) * np.exp(-t * 70)


def saw_voice(freq: float, t: np.ndarray, detune_cents: float = 0.0) -> np.ndarray:
    f = freq * 2 ** (detune_cents / 1200)
    return sawtooth(2 * np.pi * f * t)


def synthesize(spec: DarkSpec) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(spec.seed)
    total = int(spec.duration * SR)
    t_all = np.arange(total) / SR
    beat, bar = spec.beat, 4 * spec.beat
    n_bars = int(np.ceil(spec.duration / bar))
    drums = np.zeros(total)
    snare_bus = np.zeros(total)
    kicks: list[float] = []

    intro_bars, outro_bars = 2, 2
    for b in range(n_bars):
        if b < intro_bars or b >= n_bars - outro_bars:
            continue
        t0 = b * bar
        for step in (0, 10):  # half-time groove: kick on 1 and a syncopated push
            ts = t0 + step * beat / 4
            s = int(ts * SR)
            n = min(int(0.6 * SR), total - s)
            if n > 0:
                drums[s : s + n] += kick(n, rng) * 0.55
                kicks.append(ts)
        s = int((t0 + 2 * beat) * SR)  # snare on beat 3
        n = min(int(0.5 * SR), total - s)
        if n > 0:
            snare_bus[s : s + n] += snare(n, rng) * 0.32
        pattern = euclidean(7, 16, rotation=b % 3)
        for step, hit in enumerate(pattern):
            if hit:
                s = int((t0 + step * beat / 4) * SR)
                n = min(int(0.12 * SR), total - s)
                if n > 0:
                    drums[s : s + n] += hat(n, rng) * 0.035

    # Sidechain envelope: duck to 45% at each kick, recover in ~180 ms.
    duck = np.ones(total)
    for kt in kicks:
        s = int(kt * SR)
        n = min(int(0.6 * SR), total - s)
        tt = np.arange(n) / SR
        duck[s : s + n] = np.minimum(duck[s : s + n], 1 - 0.55 * np.exp(-tt / 0.18))

    # Bass riff in eighths following the root, distorted and filtered.
    bass = np.zeros(total)
    riff = [1, 0, 1, 1, 0, 1, 1, 0]  # eighth-note accents
    octave = [0, 0, 0, 12, 0, 0, 7, 0]  # occasional octave and fifth
    eighth = beat / 2
    n_eighths = int(spec.duration / eighth)
    for e in range(n_eighths):
        if not riff[e % 8]:
            continue
        ts = e * eighth
        chord_idx = int(ts // spec.chord_seconds) % len(spec.roots)
        note = spec.roots[chord_idx] - 12 + octave[e % 8]
        s = int(ts * SR)
        n = min(int(eighth * 1.6 * SR), total - s)
        if n <= 0:
            continue
        tt = np.arange(n) / SR
        env = np.minimum(tt / 0.006, 1) * np.exp(-tt * 2.2)
        v = saw_voice(midi_to_hz(note), tt) + 0.6 * np.sin(2 * np.pi * midi_to_hz(note) * tt)
        bass[s : s + n] += v * env
    cutoff_lfo = 380 + 320 * (0.5 + 0.5 * np.sin(2 * np.pi * t_all / (spec.chord_seconds * 4)))
    bass = np.tanh(2.8 * bass)
    # Time-varying low-pass approximated by crossfading two fixed filters.
    lo, hi = lowpass(bass, 300), lowpass(bass, 900)
    mix_w = (cutoff_lfo - 300) / 600
    bass = (lo * (1 - mix_w) + hi * mix_w) * 0.42
    sub = np.zeros(total)
    for i in range(int(np.ceil(spec.duration / spec.chord_seconds))):
        s = int(i * spec.chord_seconds * SR)
        n = min(int((spec.chord_seconds + 0.5) * SR), total - s)
        if n <= 0:
            break
        tt = np.arange(n) / SR
        root = spec.roots[i % len(spec.roots)] - 24
        sub[s : s + n] += np.sin(2 * np.pi * midi_to_hz(root) * tt) * _env(n, 0.4, 0.6) * 0.30

    # Distorted power-chord wall and a dark pad, both slow and far back.
    wall = np.zeros(total)
    pad = np.zeros(total)
    for i in range(int(np.ceil(spec.duration / spec.chord_seconds))):
        s = int(i * spec.chord_seconds * SR)
        n = min(int((spec.chord_seconds + 1.5) * SR), total - s)
        if n <= 0:
            break
        tt = np.arange(n) / SR
        root = spec.roots[i % len(spec.roots)]
        env = _env(n, 1.6, 1.8)
        chord = sum(saw_voice(midi_to_hz(root + iv), tt, d) for iv in (0, 7, 12) for d in (-8, 8))
        wall[s : s + n] += np.tanh(3.5 * chord / 6) * env
        triad = sum(np.sin(2 * np.pi * midi_to_hz(root + 12 + iv) * tt) for iv in (0, 3, 7, 10))
        pad[s : s + n] += triad * env / 4
    wall = lowpass(wall, 1600) * 0.10
    pad = pad * 0.10

    # Build the stereo mix.
    stereo = np.zeros((total, 2))
    stereo += (drums[:, None]) * np.array([1.0, 1.0])
    stereo += ((bass + sub) * duck)[:, None]
    stereo[:, 0] += wall * duck * 0.9
    stereo[:, 1] += np.roll(wall, int(0.012 * SR)) * duck * 0.9  # tiny delay for width
    stereo += (pad * duck)[:, None] * np.array([0.8, 1.0])
    snare_stereo = np.stack([snare_bus, snare_bus], axis=1)
    noise = lowpass(rng.standard_normal((total, 2)).T, 500).T * 0.008

    wet = reverb(stereo * np.array([1, 1]) * 0.5 + snare_stereo, 4.2, rng)
    mix = 0.75 * stereo + 0.45 * snare_stereo + 0.5 * wet + noise
    mix *= _env(total, 3.0, 6.0)[:, None]
    mix = np.tanh(1.2 * mix)
    mix /= np.max(np.abs(mix)) + 1e-9
    mix *= 10 ** (-1.0 / 20)

    analysis = analyze(mix, bass * duck, kicks, spec)
    return mix.astype(np.float32), analysis


def analyze(mix: np.ndarray, bass: np.ndarray, kicks: list[float], spec: DarkSpec) -> dict:
    hop = SR // spec.fps
    n_frames = len(mix) // hop
    mono = mix.mean(axis=1)

    def env_of(x: np.ndarray, smooth: float) -> list[float]:
        r = np.sqrt(np.array([np.mean(x[i * hop : (i + 1) * hop] ** 2) for i in range(n_frames)]))
        k = np.exp(-np.arange(-30, 31) ** 2 / (2 * smooth**2))
        r = np.convolve(r, k / k.sum(), mode="same")
        r = (r - r.min()) / (r.max() - r.min() + 1e-9)
        return [round(float(v), 4) for v in r]

    kick_env = np.zeros(n_frames)
    for kt in kicks:
        f0 = int(kt * spec.fps)
        for j in range(f0, min(f0 + 12, n_frames)):
            kick_env[j] = max(kick_env[j], np.exp(-(j - f0) / 4))
    return {
        "fps": spec.fps,
        "duration": spec.duration,
        "chord_seconds": spec.chord_seconds,
        "rms": env_of(mono, 8),
        "bass": env_of(bass, 3),
        "kick": [round(float(v), 4) for v in kick_env],
    }


def render_track(spec: DarkSpec, wav_path: Path, analysis_path: Path) -> None:
    mix, analysis = synthesize(spec)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(wav_path, SR, mix)
    analysis_path.write_text(json.dumps(analysis))
