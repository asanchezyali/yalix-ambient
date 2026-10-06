"""The dark series: ten episodes, each a distinct (music preset, visual family, palette).

Every episode lasts 194.159 s, which shows as 3:14 on YouTube (π, give or take).

    uv run yalix-series            # render all ten
    uv run yalix-series 3 7        # render episodes 3 and 7
"""

from __future__ import annotations

import sys
import time
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path

from yalix_ambient.gl_particles import Visual, render_video
from yalix_ambient.music_series import SeriesSpec, render_track
from yalix_ambient.pipeline import OUTPUT, run

DURATION = 194.159  # 3:14

INDUSTRIAL = ("bass", "wall", "bells", "noise")
SYMPHONIC = ("choir", "strings", "timpani", "bass")
MARCH = ("chug", "bass", "wall")


@dataclass
class Episode:
    number: int
    title: str
    music: SeriesSpec
    visual: Visual

    @property
    def slug(self) -> str:
        name = unicodedata.normalize("NFKD", self.title.split("·")[0].strip().lower())
        return f"{self.number:02d}-" + name.encode("ascii", "ignore").decode().replace(" ", "-")


EPISODES = [
    Episode(1, "Lorenz · Ashes",
            SeriesSpec(bpm=74, roots=(40, 41, 40, 38), drums="halftime", layers=INDUSTRIAL, seed=101),
            Visual(system="lorenz", palette="ember", nebula="crimson", seed=11)),
    Episode(2, "Thomas · Cathedral of Ice",
            SeriesSpec(bpm=66, roots=(45, 41, 43, 40), drums="sparse", layers=SYMPHONIC, seed=102),
            Visual(system="thomas", palette="ice", nebula="abyss", seed=12)),
    Episode(3, "Aizawa · Furnace",
            SeriesSpec(bpm=82, roots=(40, 40, 43, 38), bars_per_chord=1, drums="march", layers=MARCH, seed=103),
            Visual(system="aizawa", palette="fire", nebula="inferno", seed=13)),
    Episode(4, "Dipoles · Static",
            SeriesSpec(bpm=70, roots=(38, 39, 38, 36), drums="halftime", layers=INDUSTRIAL, bass_drive=3.4, seed=104),
            Visual(family="magnetic", dipoles=2, palette="toxic", nebula="venom", particles=9000, size=1.8,
                   trail=0.965, seed=14)),
    Episode(5, "Halvorsen · Requiem",
            SeriesSpec(bpm=64, roots=(43, 39, 41, 38), drums="sparse", layers=SYMPHONIC + ("wall",), seed=105),
            Visual(system="halvorsen", palette="aurora", nebula="void", particles=9000, seed=15)),
    Episode(6, "Rössler · Music Box",
            SeriesSpec(bpm=72, roots=(41, 42, 41, 39), drums="sparse", layers=("bass", "bells", "noise", "wall"),
                       seed=106),
            Visual(system="rossler", palette="blood", nebula="crimson", seed=16)),
    Episode(7, "Quadrupole · Iron",
            SeriesSpec(bpm=84, roots=(40, 43, 38, 40), bars_per_chord=1, drums="march", layers=MARCH + ("choir",),
                       seed=107),
            Visual(family="magnetic", dipoles=4, palette="fire", nebula="inferno", particles=9000, size=1.8,
                   trail=0.965, seed=17)),
    Episode(8, "Nebula · Choir of the Void",
            SeriesSpec(bpm=60, roots=(45, 43, 41, 40), drums="none", layers=("choir", "strings", "bass"),
                       reverb_s=5.5, seed=108),
            Visual(family="flow", palette="aurora", nebula="void", particles=10000, size=1.8, trail=0.965, seed=18)),
    Episode(9, "Dadras · Machine",
            SeriesSpec(bpm=78, roots=(38, 38, 39, 36), drums="halftime", layers=("bass", "chug", "noise", "bells"),
                       seed=109),
            Visual(system="dadras", palette="toxic", nebula="venom", seed=19)),
    Episode(10, "Lorenz · Finale",
            SeriesSpec(bpm=80, roots=(40, 36, 43, 38), drums="march",
                       layers=("chug", "choir", "strings", "timpani", "bass", "wall"), seed=110),
            Visual(system="lorenz", palette="blood", nebula="crimson", particles=9000, seed=20)),
]  # fmt: skip


def build_episode(ep: Episode) -> Path:
    out_dir = OUTPUT / "series"
    out_dir.mkdir(parents=True, exist_ok=True)
    wav, analysis = out_dir / f"{ep.slug}.wav", out_dir / f"{ep.slug}.json"
    silent, final = out_dir / f"{ep.slug}.silent.mp4", out_dir / f"{ep.slug}.mp4"
    t0 = time.time()
    render_track(replace(ep.music, duration=DURATION), wav, analysis)
    render_video(analysis, silent, ep.visual)
    run(
        [
            "ffmpeg", "-v", "error", "-y", "-i", str(silent), "-i", str(wav),
            "-map", "0:v", "-map", "1:a", "-c:v", "copy",
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=9", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-shortest", "-movflags", "+faststart", str(final),
        ]
    )  # fmt: skip
    silent.unlink(missing_ok=True)
    print(f"[{ep.number:02d}] {ep.title} -> {final.name} ({time.time() - t0:.0f}s)", flush=True)
    return final


def main() -> None:
    wanted = {int(a) for a in sys.argv[1:]}
    for ep in EPISODES:
        if not wanted or ep.number in wanted:
            build_episode(ep)


if __name__ == "__main__":
    main()
