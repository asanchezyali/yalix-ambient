"""End-to-end pipeline: synthesize music -> render visuals with Manim -> mux with ffmpeg.

Usage:
    uv run yalix-ambient --minutes 1 --seed 7 --name prototype

Output: output/<name>.mp4 (1920x1080, 30 fps, AAC 192k, loudness normalized to -16 LUFS).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path

from yalix_ambient.gl_render import render_video
from yalix_ambient.music import TrackSpec, render_track

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output"
MEDIA = ROOT / "media"
SCENE = Path(__file__).with_name("scene.py")


def run(cmd: list[str], env: dict | None = None) -> None:
    subprocess.run(cmd, check=True, env=env)


def build(minutes: float, seed: int, name: str, bpm: float = 72.0, engine: str = "gl") -> Path:
    OUTPUT.mkdir(exist_ok=True)
    wav = OUTPUT / f"{name}.wav"
    analysis = OUTPUT / f"{name}.json"
    final = OUTPUT / f"{name}.mp4"

    t0 = time.time()
    spec = TrackSpec(duration=minutes * 60, seed=seed, bpm=bpm)
    render_track(spec, wav, analysis)
    print(f"[1/3] music: {wav.name} ({time.time() - t0:.0f}s)")

    t1 = time.time()
    if engine == "gl":
        silent = OUTPUT / f"{name}.silent.mp4"
        render_video(analysis, silent, fps=spec.fps)
    else:
        env = {**os.environ, "YALIX_ANALYSIS": str(analysis)}
        run(
            [
                "manim",
                "-r",
                "1920,1080",
                "--fps",
                str(spec.fps),
                "--media_dir",
                str(MEDIA),
                "-o",
                name,
                "--progress_bar",
                "none",
                str(SCENE),
                "LissajousAmbient",
            ],
            env=env,
        )
        silent = MEDIA / "videos" / "scene" / f"1080p{spec.fps}" / f"{name}.mp4"
    print(f"[2/3] visuals: {silent.name} ({time.time() - t1:.0f}s)")

    t2 = time.time()
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(silent),
            "-i",
            str(wav),
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "copy",
            "-af",
            "loudnorm=I=-16:TP=-1.5:LRA=7",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-shortest",
            "-movflags",
            "+faststart",
            str(final),
        ]
    )
    print(f"[3/3] mux: {final} ({time.time() - t2:.0f}s, total {time.time() - t0:.0f}s)")
    return final


def main() -> None:
    p = argparse.ArgumentParser(description="Math-generated ambient video")
    p.add_argument("--minutes", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--bpm", type=float, default=72.0)
    p.add_argument("--name", default="prototype")
    p.add_argument("--engine", choices=["gl", "manim"], default="gl")
    a = p.parse_args()
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found")
    build(a.minutes, a.seed, a.name, a.bpm, a.engine)


if __name__ == "__main__":
    main()
