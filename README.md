# yalix-ambient

Math-generated ambient videos for focus and relaxation. Everything is synthesized from scratch:

- **Music** (`music.py`): additive synthesis (sums of harmonics), a D Dorian progression, an arpeggio on a
  Euclidean rhythm E(5,16), a sub bass, filtered noise and a convolution reverb built from decaying noise.
  No samples or third-party audio.
- **Visuals, GPU engine (default)** (`gl_render.py`): GLSL shaders through moderngl. An fbm-noise nebula with a
  breathing radial glow, closed Lissajous figures drawn as soft lines, a feedback trail and a Gaussian bloom,
  piped frame by frame into ffmpeg. About 2× faster than real time on an M1 Pro.
- **Visuals, Manim engine** (`scene.py`, `--engine manim`): Manim Lissajous curves `x = A·sin(aθ + δ)`, `y = A·sin(bθ)`. Each chord sets the
  ratio a:b (3:2, 4:3, 5:4, 5:3); between chords the ratio morphs, the phase drifts and the amplitude breathes
  with the loudness of the track. Catppuccin Mocha palette.
- **Pipeline** (`pipeline.py`): music → Manim render → ffmpeg mux, loudness-normalized to -16 LUFS.

```bash
uv sync
uv run yalix-ambient --minutes 3 --name video01   # output/video01.mp4
```

Series of ten episodes at 3:14 each, one figure per episode and no repeats inside a series:

```bash
uv run yalix-series       # dark: industrial, symphonic and progressive metal styles
uv run yalix-grunge       # grunge: Seattle-style songs, twin lead guitars
uv run yalix-cyberpunk    # cyberpunk: synthwave and darksynth, neon palettes, scanlines and glitch
```

The cyberpunk music (`music_cyberpunk.py`) uses supersaw pads that pump under the kick, a sixteenth-note
arpeggio whose filter opens over the song, a pulse-wave lead, gated-reverb snares and four bass types
(rolling, pulse, reese, FM). Its visuals add four attractors: Sprott, Burke-Shaw, Lü and Rucklidge.

Render speed at 1080p30: GPU engine ~0.5× the video length (3 min in ~1.5 min); Manim ~4×.

Next: more visual families (harmonograph, Fourier epicycles, vector fields), more progressions and
timbres per seed, thumbnails, and upload through the YouTube Data API.
