"""Parametric dark-metal-ambient tracks for the series. Everything synthesized with numpy.

Genre vocabulary (styles, not anyone's songs):
- industrial (Manson-like): distorted saw bass, noise, eerie music-box bells, half-time drums;
- NDH march (Rammstein-like): palm-muted chug guitar in eighths, four-on-the-floor kick, snare on 2 and 4;
- symphonic (Nightwish-like): formant choir, detuned string ensemble, timpani, harmonic minor.
All kept slow and spacious so the result relaxes instead of agitating.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from yalix_ambient.music import SR, _env, euclidean, lowpass, midi_to_hz, reverb
from yalix_ambient.music_dark import bandpass, hat, highpass, kick, saw_voice, snare


@dataclass
class SeriesSpec:
    duration: float = 180.0
    bpm: float = 74.0
    roots: tuple[int, ...] = (40, 41, 40, 38)  # MIDI roots, one chord per `bars_per_chord`
    minor_third: bool = True
    bars_per_chord: int = 2
    drums: str = "halftime"  # halftime | march | sparse | none
    layers: tuple[str, ...] = ("bass", "wall")  # bass, wall, chug, choir, strings, timpani, bells, noise
    bass_drive: float = 2.8
    reverb_s: float = 4.2
    seed: int = 13
    fps: int = 30
    melody_scale: tuple[int, ...] = (0, 2, 3, 5, 7, 8, 10)  # for bells, relative to root

    @property
    def beat(self) -> float:
        return 60 / self.bpm

    @property
    def chord_seconds(self) -> float:
        return 4 * self.beat * self.bars_per_chord


# ---------------------------------------------------------------- instruments


def chug(freq: float, n: int) -> np.ndarray:
    """Palm-muted power chord: detuned saws (root + fifth), hard clip, cabinet band."""
    t = np.arange(n) / SR
    v = sum(saw_voice(freq * r, t, d) for r in (1.0, 1.4983) for d in (-9, 9))
    env = np.minimum(t / 0.004, 1) * np.exp(-t * 11)
    x = np.tanh(6.0 * v * env / 4)
    return bandpass(x, 90, 3200)


FORMANTS_AH = ((800, 1.0), (1150, 0.5), (2900, 0.18))
FORMANTS_OH = ((450, 1.0), (800, 0.45), (2830, 0.12))


def choir(freqs: list[float], n: int, rng: np.random.Generator, vowel=FORMANTS_AH) -> np.ndarray:
    """Formant choir: several detuned saw 'voices' per note, shaped by vowel band-passes."""
    t = np.arange(n) / SR
    src = np.zeros(n)
    for f in freqs:
        for _ in range(3):
            vib = 1 + 0.006 * np.sin(2 * np.pi * rng.uniform(4.5, 5.8) * t + rng.uniform(0, 6.28))
            phase = 2 * np.pi * np.cumsum(f * rng.uniform(0.997, 1.003) * vib) / SR
            src += 2 * (phase / (2 * np.pi) % 1) - 1
    out = sum(g * bandpass(src, fc * 0.85, fc * 1.15) for fc, g in vowel)
    return out / (len(freqs) * 3)


def strings(freqs: list[float], n: int) -> np.ndarray:
    t = np.arange(n) / SR
    v = sum(saw_voice(f, t, d) for f in freqs for d in (-12, -4, 4, 12))
    return lowpass(v / (len(freqs) * 4), 2800)


def timpani(freq: float, n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / SR
    f = freq * (1 + 0.15 * np.exp(-t * 20))
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 2.2)
    skin = lowpass(rng.standard_normal(n), 900) * np.exp(-t * 18) * 0.5
    return np.tanh(1.3 * (body + skin))


def bell(freq: float, n: int) -> np.ndarray:
    """Music-box / glass bell: inharmonic partials with fast decay."""
    t = np.arange(n) / SR
    parts = ((1.0, 1.0, 2.5), (2.76, 0.45, 5.0), (5.40, 0.25, 9.0), (8.93, 0.12, 14.0))
    return sum(a * np.sin(2 * np.pi * freq * r * t) * np.exp(-t * d) for r, a, d in parts)


# ---------------------------------------------------------------- composition


def chord_tones(root: int, minor: bool) -> list[int]:
    return [root, root + (3 if minor else 4), root + 7]


def synthesize(spec: SeriesSpec) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(spec.seed)
    total = int(spec.duration * SR)
    t_all = np.arange(total) / SR
    beat, bar = spec.beat, 4 * spec.beat
    n_bars = int(np.ceil(spec.duration / bar))
    n_chords = int(np.ceil(spec.duration / spec.chord_seconds))
    L = set(spec.layers)

    def put(buf, s, x, gain=1.0):
        n = min(len(x), total - s)
        if n > 0 and s >= 0:
            buf[s : s + n] += x[:n] * gain

    def root_at(ts: float) -> int:
        return spec.roots[int(ts // spec.chord_seconds) % len(spec.roots)]

    drums, snare_bus = np.zeros(total), np.zeros(total)
    kicks: list[float] = []
    intro, outro = 2, 2
    for b in range(intro, n_bars - outro):
        t0 = b * bar
        if spec.drums == "halftime":
            k_steps, s_steps = (0, 10), (8,)
        elif spec.drums == "march":
            k_steps, s_steps = (0, 4, 8, 12), (4, 12)
        elif spec.drums == "sparse":
            k_steps, s_steps = (0,), (8,) if b % 2 else ()
        else:
            k_steps, s_steps = (), ()
        for st in k_steps:
            ts = t0 + st * beat / 4
            put(drums, int(ts * SR), kick(int(0.6 * SR), rng), 0.55)
            kicks.append(ts)
        for st in s_steps:
            put(snare_bus, int((t0 + st * beat / 4) * SR), snare(int(0.5 * SR), rng), 0.30)
        if spec.drums in ("halftime", "march"):
            for st, hit in enumerate(euclidean(7 if spec.drums == "halftime" else 8, 16, rotation=b % 3)):
                if hit:
                    put(drums, int((t0 + st * beat / 4) * SR), hat(int(0.12 * SR), rng), 0.03)

    duck = np.ones(total)
    for kt in kicks:
        s = int(kt * SR)
        n = min(int(0.6 * SR), total - s)
        duck[s : s + n] = np.minimum(duck[s : s + n], 1 - 0.5 * np.exp(-np.arange(n) / SR / 0.18))

    low = np.zeros(total)  # bass + sub + chug (mono, ducked)
    wide = np.zeros((total, 2))  # pads, wall, choir, strings (stereo, ducked)
    dry = np.zeros((total, 2))  # bells, timpani (stereo, not ducked)

    if "bass" in L:
        riff, octave = [1, 0, 1, 1, 0, 1, 1, 0], [0, 0, 0, 12, 0, 0, 7, 0]
        eighth = beat / 2
        bass = np.zeros(total)
        for e in range(int(spec.duration / eighth)):
            if not riff[e % 8]:
                continue
            ts = e * eighth
            note = root_at(ts) - 12 + octave[e % 8]
            n = int(eighth * 1.6 * SR)
            tt = np.arange(n) / SR
            env = np.minimum(tt / 0.006, 1) * np.exp(-tt * 2.2)
            put(bass, int(ts * SR), (saw_voice(midi_to_hz(note), tt) + 0.6 * np.sin(2 * np.pi * midi_to_hz(note) * tt)) * env)
        bass = np.tanh(spec.bass_drive * bass)
        w = 0.5 + 0.5 * np.sin(2 * np.pi * t_all / (spec.chord_seconds * 4))
        low += (lowpass(bass, 300) * (1 - w) + lowpass(bass, 900) * w) * 0.40
    for i in range(n_chords):  # sub bass always
        s = int(i * spec.chord_seconds * SR)
        n = int((spec.chord_seconds + 0.5) * SR)
        tt = np.arange(n) / SR
        put(low, s, np.sin(2 * np.pi * midi_to_hz(spec.roots[i % len(spec.roots)] - 24) * tt) * _env(n, 0.4, 0.6), 0.28)

    if "chug" in L:
        pattern = [1, 1, 0, 1, 1, 0, 1, 0]  # eighths, palm-muted gallop
        eighth = beat / 2
        ch = np.zeros(total)
        for e in range(int(spec.duration / eighth)):
            b = int(e * eighth // bar)
            if b < intro or b >= n_bars - outro or not pattern[e % 8]:
                continue
            ts = e * eighth
            put(ch, int(ts * SR), chug(midi_to_hz(root_at(ts) - 12), int(0.3 * SR)))
        low += ch * 0.16

    for i in range(n_chords):
        s = int(i * spec.chord_seconds * SR)
        n = int((spec.chord_seconds + 2.0) * SR)
        root = spec.roots[i % len(spec.roots)]
        tones = chord_tones(root, spec.minor_third)
        env = _env(n, 1.8, 2.0)
        tt = np.arange(n) / SR
        if "wall" in L:
            v = sum(saw_voice(midi_to_hz(root + iv), tt, d) for iv in (0, 7, 12) for d in (-8, 8))
            x = lowpass(np.tanh(3.5 * v / 6) * env, 1600) * 0.10
            wide[s : s + min(n, total - s), 0] += x[: total - s]
            wide[s : s + min(n, total - s), 1] += np.roll(x, int(0.012 * SR))[: total - s]
        if "choir" in L:
            vowel = FORMANTS_AH if i % 2 == 0 else FORMANTS_OH
            c = choir([midi_to_hz(n_ + 12) for n_ in tones], n, rng, vowel) * env * 0.5
            wide[s : s + min(n, total - s)] += np.stack([c, np.roll(c, int(0.008 * SR))], axis=1)[: total - s]
        if "strings" in L:
            st = strings([midi_to_hz(n_ + 24) for n_ in tones] + [midi_to_hz(root + 12)], n) * env * 0.16
            wide[s : s + min(n, total - s)] += np.stack([np.roll(st, int(0.01 * SR)), st], axis=1)[: total - s]
        if "timpani" in L and intro * bar <= i * spec.chord_seconds < (n_bars - outro) * bar:
            for hit, gain in ((0.0, 0.5), (beat * 0.5, 0.25)):
                tp = timpani(midi_to_hz(root - 12), int(1.8 * SR), rng)
                put(dry[:, 0], s + int(hit * SR), tp, gain * 0.5)
                put(dry[:, 1], s + int(hit * SR), tp, gain * 0.5)

    if "bells" in L:  # slow random walk on the scale, two octaves up
        deg = 0
        step = beat * 2
        for j in range(int(spec.duration / step)):
            ts = j * step
            if rng.random() < 0.35 or ts < bar or ts > spec.duration - 2 * bar:
                continue
            deg = int(np.clip(deg + rng.choice([-2, -1, 1, 2]), -3, 9))
            octv, idx = divmod(deg, len(spec.melody_scale))
            note = root_at(ts) + 24 + 12 * octv + spec.melody_scale[idx]
            b = bell(midi_to_hz(note), int(2.5 * SR)) * 0.05
            pan = rng.uniform(0.2, 0.8)
            put(dry[:, 0], int(ts * SR), b, 1 - pan)
            put(dry[:, 1], int(ts * SR), b, pan)

    noise = lowpass(rng.standard_normal((total, 2)).T, 500).T * (0.016 if "noise" in L else 0.007)

    stereo = drums[:, None] + (low * duck)[:, None] + wide * duck[:, None] + dry
    snare_st = np.stack([snare_bus, snare_bus], axis=1)
    wet = reverb(0.5 * stereo + snare_st + 0.6 * dry, spec.reverb_s, rng)
    mix = 0.75 * stereo + 0.45 * snare_st + 0.5 * wet + noise
    mix *= _env(total, 3.0, 6.0)[:, None]
    mix = np.tanh(1.2 * mix)
    mix /= np.max(np.abs(mix)) + 1e-9
    mix *= 10 ** (-1.0 / 20)
    return mix.astype(np.float32), analyze(mix, low * duck, kicks, spec)


def analyze(mix: np.ndarray, bass: np.ndarray, kicks: list[float], spec: SeriesSpec) -> dict:
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


def render_track(spec: SeriesSpec, wav_path: Path, analysis_path: Path) -> None:
    mix, analysis = synthesize(spec)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(wav_path, SR, mix)
    analysis_path.write_text(json.dumps(analysis))


