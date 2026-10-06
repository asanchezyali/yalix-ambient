"""Series music, v2: every episode gets its own musical identity.

What changes per episode (styles of genres, never anyone's songs):
- meter: eighth-note steps per bar, cycling (4/4 = 8, 7/8 = 7, 5/4 = 10, 13/8 = 13, or a
  sequence like 9-8-7 in the spirit of progressive metal);
- drums: half-time industrial, NDH march, tribal toms in polyrhythm (Tool-like), sparse, none;
- bass timbre: saw (industrial), square/PWM (march), FM growl, octave fuzz, clean sub;
- bass riff: its own rhythm (explicit pattern or Euclidean) and interval line, including
  tritones and minor seconds, optionally accented on Fibonacci positions;
- lead: Karplus-Strong clean guitar arpeggio with ping-pong delay, music-box bells, choir, strings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter, square

from yalix_ambient.music import SR, _env, euclidean, lowpass, midi_to_hz, reverb
from yalix_ambient.music_dark import bandpass, hat, kick, saw_voice, snare
from yalix_ambient.music_series import FORMANTS_AH, FORMANTS_OH, bell, choir, chord_tones, chug, strings, timpani


@dataclass
class Spec:
    duration: float = 194.159
    bpm: float = 74.0
    meters: tuple[int, ...] = (8,)  # eighth-note steps per bar, cycled
    bars_per_chord: int = 2
    roots: tuple[int, ...] = (40, 41, 40, 38)
    minor: bool = True
    scale: tuple[int, ...] = (0, 1, 3, 5, 7, 8, 10)  # Phrygian by default
    drums: str = "halftime"  # halftime | march | tribal | sparse | none
    kick_pitch: float = 44.0
    bass_timbre: str = "saw"  # saw | square | fm | fuzz | sub
    bass_pattern: tuple[int, ...] | None = None  # explicit hits per step, cycled; None -> Euclidean
    bass_hits: int = 5
    bass_notes: tuple[int, ...] = (0, 0, 12, 0, 7)  # interval line relative to the root, cycled per hit
    bass_drive: float = 2.8
    bass_cutoff: tuple[float, float] = (300.0, 900.0)
    fibonacci: bool = False  # accent bass and toms on Fibonacci positions
    arp_hits: int = 5  # Euclidean hits per bar for the clean-guitar arpeggio
    layers: tuple[str, ...] = ("bass", "wall")  # bass wall chug choir strings timpani bells arp noise drone
    reverb_s: float = 4.2
    seed: int = 1
    fps: int = 30

    @property
    def step(self) -> float:
        return 60 / self.bpm / 2  # eighth note


# ------------------------------------------------------------------ instruments


def bass_tone(freq: float, n: int, kind: str, t0: float) -> np.ndarray:
    t = np.arange(n) / SR
    if kind == "saw":
        x = saw_voice(freq, t) + 0.6 * np.sin(2 * np.pi * freq * t)
    elif kind == "square":
        duty = 0.5 + 0.22 * np.sin(2 * np.pi * 0.35 * (t + t0))  # slow PWM
        x = square(2 * np.pi * freq * t, duty=duty) * 0.8 + 0.5 * np.sin(2 * np.pi * freq * t)
    elif kind == "fm":
        index = 1.2 + 4.0 * np.exp(-t * 3.0)  # bright attack, darker tail: a growl
        x = np.sin(2 * np.pi * freq * t + index * np.sin(2 * np.pi * freq * t)) + 0.5 * np.sin(np.pi * freq * t)
    elif kind == "fuzz":
        y = np.tanh(8 * (np.sin(2 * np.pi * freq * t) + 0.5 * saw_voice(freq, t)))
        x = y + 0.6 * (np.abs(y) - 0.5)  # full-wave rectification adds the octave
    else:  # sub
        x = np.sin(2 * np.pi * freq * t) + 0.15 * np.sin(4 * np.pi * freq * t)
    return x


def ks_pluck(freq: float, n: int, rng: np.random.Generator, damping: float = 0.996) -> np.ndarray:
    """Karplus-Strong: a noise burst circulating in a delay line with an averaging filter."""
    L = max(int(SR / freq), 2)
    burst = np.zeros(n)
    burst[:L] = lowpass(rng.uniform(-1, 1, L), 3500)
    a = np.zeros(L + 2)
    a[0], a[L], a[L + 1] = 1.0, -0.5 * damping, -0.5 * damping
    return lfilter([1.0], a, burst)


def tom(freq: float, n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / SR
    f = freq * (1 + 0.5 * np.exp(-t * 25))
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 7)
    skin = bandpass(rng.standard_normal(n), 200, 2500) * np.exp(-t * 30) * 0.3
    return np.tanh(1.6 * (body + skin))


def fibonacci_positions(steps: int) -> set[int]:
    a, b, pos, out = 1, 2, 0, {0}
    while pos < steps:
        pos += a
        out.add(pos % steps)
        a, b = b, a + b
    return out


# ------------------------------------------------------------------ composition


def synthesize(spec: Spec) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(spec.seed)
    total = int(spec.duration * SR)
    t_all = np.arange(total) / SR
    L = set(spec.layers)
    st = spec.step

    # Timeline of bars with (start time, steps) following the meter cycle.
    bars: list[tuple[float, int]] = []
    t = 0.0
    while t < spec.duration:
        steps = spec.meters[len(bars) % len(spec.meters)]
        bars.append((t, steps))
        t += steps * st
    n_bars = len(bars)
    chords = [(bars[i][0], spec.roots[(i // spec.bars_per_chord) % len(spec.roots)]) for i in range(0, n_bars, spec.bars_per_chord)]

    def root_at(ts: float) -> int:
        r = chords[0][1]
        for c_t, c_r in chords:
            if c_t <= ts:
                r = c_r
            else:
                break
        return r

    def put(buf, s, x, gain=1.0):
        s = int(s)
        n = min(len(x), total - s)
        if n > 0 and s >= 0:
            buf[s : s + n] += x[:n] * gain

    intro, outro = 2, 2
    drums, snare_bus = np.zeros(total), np.zeros(total)
    toms = np.zeros((total, 2))
    kicks: list[float] = []

    for b, (t0, S) in enumerate(bars):
        if b < intro or b >= n_bars - outro or spec.drums == "none":
            continue
        fib = fibonacci_positions(S) if spec.fibonacci else set()
        if spec.drums == "halftime":
            k_pos, s_pos = {0, int(S * 0.6) | 1}, {S // 2}
        elif spec.drums == "march":
            k_pos, s_pos = set(range(0, S, 2)), {2, 6} if S == 8 else {S // 2}
        elif spec.drums == "tribal":
            k_pos = {i for i, h in enumerate(euclidean(3, S)) if h}
            s_pos = {S - 2} if b % 2 else set()
            for freq, hits, rot, pan in ((92, 5, 1, 0.35), (128, 3, 2, 0.65), (175, 2, 3, 0.5)):
                for i, h in enumerate(euclidean(min(hits, S), S, rot)):
                    if h:
                        g = 0.28 * (1.4 if i in fib else 1.0)
                        tm = tom(freq, int(0.7 * SR), rng)
                        put(toms[:, 0], (t0 + i * st) * SR, tm, g * (1 - pan))
                        put(toms[:, 1], (t0 + i * st) * SR, tm, g * pan)
        elif spec.drums == "sparse":
            k_pos, s_pos = {0}, ({S // 2} if b % 2 else set())
        else:
            k_pos, s_pos = set(), set()
        for i in sorted(k_pos):
            put(drums, (t0 + i * st) * SR, kick(int(0.6 * SR), rng) * (1.0 if spec.kick_pitch >= 44 else 1.1), 0.55)
            kicks.append(t0 + i * st)
        for i in s_pos:
            put(snare_bus, (t0 + i * st) * SR, snare(int(0.5 * SR), rng), 0.30)
        if spec.drums in ("halftime", "march", "tribal"):
            for i, h in enumerate(euclidean(max(S - 3, 3), 2 * S, b % 3)):  # sixteenth hats
                if h:
                    put(drums, (t0 + i * st / 2) * SR, hat(int(0.12 * SR), rng), 0.028)

    duck = np.ones(total)
    for kt in kicks:
        s = int(kt * SR)
        n = min(int(0.6 * SR), total - s)
        duck[s : s + n] = np.minimum(duck[s : s + n], 1 - 0.5 * np.exp(-np.arange(n) / SR / 0.18))

    low, wide, dry = np.zeros(total), np.zeros((total, 2)), np.zeros((total, 2))

    # Bass riff: its own rhythm, interval line and timbre.
    if "bass" in L:
        bass = np.zeros(total)
        hit_count = 0
        for b, (t0, S) in enumerate(bars):
            fib = fibonacci_positions(S) if spec.fibonacci else set()
            pattern = (
                [spec.bass_pattern[i % len(spec.bass_pattern)] for i in range(S)]
                if spec.bass_pattern
                else euclidean(min(spec.bass_hits, S), S)
            )
            hits = [i for i, h in enumerate(pattern) if h]
            for j, i in enumerate(hits):
                ts = t0 + i * st
                nxt = hits[j + 1] if j + 1 < len(hits) else S
                n = int((nxt - i) * st * 0.92 * SR)
                note = root_at(ts) - 12 + spec.bass_notes[hit_count % len(spec.bass_notes)]
                hit_count += 1
                tt = np.arange(n) / SR
                env = np.minimum(tt / 0.006, 1) * np.exp(-tt * 1.6) * np.minimum((n / SR - tt) / 0.02, 1)
                acc = 1.35 if i in fib else 1.0
                put(bass, ts * SR, bass_tone(midi_to_hz(note), n, spec.bass_timbre, ts) * env * acc)
        bass = np.tanh(spec.bass_drive * bass)
        lo_c, hi_c = spec.bass_cutoff
        w = 0.5 + 0.5 * np.sin(2 * np.pi * t_all / (bars[0][1] * st * 8))
        low += (lowpass(bass, lo_c) * (1 - w) + lowpass(bass, hi_c) * w) * 0.40

    # Sub drone under every chord.
    for i, (c_t, c_r) in enumerate(chords):
        end = chords[i + 1][0] if i + 1 < len(chords) else spec.duration
        n = int((end - c_t + 0.5) * SR)
        tt = np.arange(n) / SR
        put(low, c_t * SR, np.sin(2 * np.pi * midi_to_hz(c_r - 24) * tt) * _env(n, 0.4, 0.6), 0.26 if "drone" not in L else 0.36)

    if "chug" in L:
        ch = np.zeros(total)
        for b, (t0, S) in enumerate(bars):
            if b < intro or b >= n_bars - outro:
                continue
            for i, h in enumerate(euclidean(max(S - 2, 3), S, 1)):
                if h:
                    ts = t0 + i * st
                    put(ch, ts * SR, chug(midi_to_hz(root_at(ts) - 12), int(0.3 * SR)))
        low += ch * 0.15

    # Pads per chord.
    for i, (c_t, root) in enumerate(chords):
        end = chords[i + 1][0] if i + 1 < len(chords) else spec.duration
        n = int((end - c_t + 2.0) * SR)
        s = int(c_t * SR)
        tones = chord_tones(root, spec.minor)
        env = _env(n, 1.8, 2.0)
        tt = np.arange(n) / SR
        m = min(n, total - s)
        if m <= 0:
            continue
        if "wall" in L:
            v = sum(saw_voice(midi_to_hz(root + iv), tt, d) for iv in (0, 7, 12) for d in (-8, 8))
            x = lowpass(np.tanh(3.5 * v / 6) * env, 1600) * 0.09
            wide[s : s + m] += np.stack([x, np.roll(x, int(0.012 * SR))], axis=1)[:m]
        if "choir" in L:
            c = choir([midi_to_hz(n_ + 12) for n_ in tones], n, rng, FORMANTS_AH if i % 2 == 0 else FORMANTS_OH)
            c = c * env * 0.5
            wide[s : s + m] += np.stack([c, np.roll(c, int(0.008 * SR))], axis=1)[:m]
        if "strings" in L:
            x = strings([midi_to_hz(n_ + 24) for n_ in tones] + [midi_to_hz(root + 12)], n) * env * 0.15
            wide[s : s + m] += np.stack([np.roll(x, int(0.01 * SR)), x], axis=1)[:m]
        if "timpani" in L and intro * 1.0 <= i * spec.bars_per_chord < n_bars - outro:
            tp = timpani(midi_to_hz(root - 12), int(1.8 * SR), rng)
            put(dry[:, 0], s, tp, 0.25)
            put(dry[:, 1], s, tp, 0.25)

    # Clean guitar arpeggio with ping-pong delay.
    if "arp" in L:
        arp = np.zeros((total, 2))
        k = 0
        for b, (t0, S) in enumerate(bars):
            if b < 1 or b >= n_bars - 1:
                continue
            for i, h in enumerate(euclidean(min(spec.arp_hits, S), S, b % 2)):
                if not h:
                    continue
                ts = t0 + i * st
                root = root_at(ts)
                line = [root + 12, root + 19, root + 12 + spec.scale[2], root + 24, root + 12 + spec.scale[4]]
                note = line[k % len(line)]
                k += 1
                p = ks_pluck(midi_to_hz(note), int(1.8 * SR), rng) * 0.10
                pan = 0.5 + 0.3 * np.sin(k)
                put(arp[:, 0], ts * SR, p, 1 - pan)
                put(arp[:, 1], ts * SR, p, pan)
        d = int(3 * st * SR)
        echo = np.zeros_like(arp)
        for r in range(1, 5):  # ping-pong: alternate channels, 0.45 feedback
            g = 0.45**r
            ch = r % 2
            echo[r * d :, ch] += lowpass(arp[: total - r * d, 1 - ch], 2500) * g
        dry += arp + echo

    if "bells" in L:
        deg = 0
        for j in range(int(spec.duration / (4 * st))):
            ts = j * 4 * st
            if rng.random() < 0.4 or ts < 4 * st * 4 or ts > spec.duration - 8 * st * 4:
                continue
            deg = int(np.clip(deg + rng.choice([-2, -1, 1, 2]), -3, 9))
            octv, idx = divmod(deg, len(spec.scale))
            note = root_at(ts) + 24 + 12 * octv + spec.scale[idx]
            b_ = bell(midi_to_hz(note), int(2.5 * SR)) * 0.05
            pan = rng.uniform(0.2, 0.8)
            put(dry[:, 0], ts * SR, b_, 1 - pan)
            put(dry[:, 1], ts * SR, b_, pan)

    noise = lowpass(rng.standard_normal((total, 2)).T, 500).T * (0.016 if "noise" in L else 0.007)
    stereo = drums[:, None] + toms + (low * duck)[:, None] + wide * duck[:, None] + dry
    snare_st = np.stack([snare_bus, snare_bus], axis=1)
    wet = reverb(0.5 * stereo + snare_st + 0.5 * dry + 0.4 * toms, spec.reverb_s, rng)
    mix = 0.75 * stereo + 0.45 * snare_st + 0.5 * wet + noise
    mix *= _env(total, 3.0, 6.0)[:, None]
    mix = np.tanh(1.2 * mix)
    mix /= np.max(np.abs(mix)) + 1e-9
    mix *= 10 ** (-1.0 / 20)
    return mix.astype(np.float32), analyze(mix, low * duck, kicks, spec, bars)


def analyze(mix, bass, kicks, spec: Spec, bars) -> dict:
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
        "chord_seconds": bars[0][1] * spec.step * spec.bars_per_chord,
        "rms": env_of(mono, 8),
        "bass": env_of(bass, 3),
        "kick": [round(float(v), 4) for v in kick_env],
    }


def render_track(spec: Spec, wav_path: Path, analysis_path: Path) -> None:
    mix, analysis = synthesize(spec)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(wav_path, SR, mix)
    analysis_path.write_text(json.dumps(analysis))
