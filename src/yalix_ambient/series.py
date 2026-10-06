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
from yalix_ambient.music_v2 import Spec, render_track
from yalix_ambient.pipeline import OUTPUT, run

DURATION = 194.159  # 3:14

PHRYGIAN = (0, 1, 3, 5, 7, 8, 10)
HARMONIC_MINOR = (0, 2, 3, 5, 7, 8, 11)
AEOLIAN = (0, 2, 3, 5, 7, 8, 10)


@dataclass
class Episode:
    number: int
    title: str
    music: Spec
    visual: Visual

    @property
    def slug(self) -> str:
        name = unicodedata.normalize("NFKD", self.title.split("·")[0].strip().lower())
        return f"{self.number:02d}-" + name.encode("ascii", "ignore").decode().replace(" ", "-")


EPISODES = [
    # Industrial (Manson-like): 4/4 half-time, gritty saw riff with a minor-second sting, music-box bells.
    Episode(1, "Lorenz · Ashes",
            Spec(bpm=74, meters=(8,), roots=(40, 41, 40, 38), drums="halftime", bass_timbre="saw",
                 bass_pattern=(1, 0, 1, 1, 0, 1, 1, 0), bass_notes=(0, 0, 0, 12, 0, 0, 7, 1),
                 layers=("bass", "wall", "bells", "noise"), seed=101),
            Visual(system="lorenz", palette="ember", nebula="crimson", seed=11)),
    # Symphonic (Nightwish-like): 6/8 lilt, harmonic minor, choir and strings over a clean sub.
    Episode(2, "Thomas · Cathedral of Ice",
            Spec(bpm=72, meters=(6,), roots=(45, 41, 38, 40), scale=HARMONIC_MINOR, drums="sparse",
                 bass_timbre="sub", bass_pattern=(1, 0, 0, 1, 0, 0), bass_notes=(0, 7), bass_drive=1.2,
                 layers=("choir", "strings", "timpani", "bass"), reverb_s=5.0, seed=102),
            Visual(system="thomas", palette="ice", nebula="abyss", seed=12)),
    # NDH march (Rammstein-like): square bass locked to a palm-muted gallop, E-F minor second.
    Episode(3, "Aizawa · Furnace",
            Spec(bpm=84, meters=(8,), bars_per_chord=1, roots=(40, 40, 41, 40), drums="march",
                 bass_timbre="square", bass_pattern=(1, 1, 0, 1, 1, 0, 1, 0), bass_notes=(0, 0, 0, 0, 1, 0),
                 bass_cutoff=(400, 1400), layers=("chug", "bass", "wall"), seed=103),
            Visual(system="aizawa", palette="fire", nebula="inferno", seed=13)),
    # Progressive (Tool-like): 7/8, tribal toms on Fibonacci accents, FM growl bass with tritones, clean delay guitar.
    Episode(4, "Dipoles · Static",
            Spec(bpm=78, meters=(7,), roots=(38, 38, 44, 38), drums="tribal", bass_timbre="fm", bass_hits=4,
                 bass_notes=(0, 6, 0, 5, 3), fibonacci=True, arp_hits=4,
                 layers=("bass", "arp", "drone", "noise"), seed=104),
            Visual(family="magnetic", dipoles=2, palette="toxic", nebula="venom", particles=9000, size=1.8,
                   trail=0.965, seed=14)),
    # Symphonic doom: 5/4, octave-fuzz bass, choir, strings and timpani.
    Episode(5, "Halvorsen · Requiem",
            Spec(bpm=64, meters=(10,), roots=(43, 39, 41, 38), scale=HARMONIC_MINOR, drums="sparse",
                 bass_timbre="fuzz", bass_hits=3, bass_notes=(0, 7, 10), bass_drive=1.8,
                 layers=("choir", "strings", "timpani", "bass", "wall"), seed=105),
            Visual(system="halvorsen", palette="aurora", nebula="void", particles=9000, seed=15)),
    # Eerie waltz (Manson-like): 3/4, music box over a clean sub with a tritone, almost no drums.
    Episode(6, "Rössler · Music Box",
            Spec(bpm=60, meters=(6,), roots=(42, 43, 42, 37), drums="sparse", bass_timbre="sub",
                 bass_pattern=(1, 0, 0, 0, 1, 0), bass_notes=(0, 0, 6), bass_drive=1.4,
                 layers=("bells", "bass", "noise", "drone"), seed=106),
            Visual(system="rossler", palette="blood", nebula="crimson", seed=16)),
    # NDH anthem: driving octave saw bass, march, chug and choir in C# Phrygian.
    Episode(7, "Quadrupole · Iron",
            Spec(bpm=88, meters=(8,), roots=(37, 37, 40, 35), drums="march", bass_timbre="saw",
                 bass_pattern=(1, 0, 1, 0, 1, 0, 1, 1), bass_notes=(0, 12, 0, 12, 0, 12, 0, 10),
                 bass_cutoff=(500, 1600), layers=("chug", "bass", "choir", "wall"), seed=107),
            Visual(family="magnetic", dipoles=4, palette="fire", nebula="inferno", particles=9000, size=1.8,
                   trail=0.965, seed=17)),
    # Dark ambient: no drums, drone, choir and strings, a slow clean arpeggio far away.
    Episode(8, "Nebula · Choir of the Void",
            Spec(bpm=54, meters=(8,), roots=(45, 43, 41, 40), scale=AEOLIAN, drums="none", bass_timbre="sub",
                 bass_pattern=(1, 0, 0, 0, 0, 0, 0, 0), bass_notes=(0,), bass_drive=1.0, arp_hits=3,
                 layers=("choir", "strings", "drone", "arp", "bass"), reverb_s=6.0, seed=108),
            Visual(family="flow", palette="aurora", nebula="void", particles=10000, size=1.8, trail=0.965, seed=18)),
    # Progressive (Tool-like): 13/8, tribal toms, octave-fuzz riff with minor seconds and a tritone, chug.
    Episode(9, "Dadras · Machine",
            Spec(bpm=84, meters=(13,), roots=(38, 38, 41, 37), drums="tribal", bass_timbre="fuzz", bass_hits=7,
                 bass_notes=(0, 0, 3, 0, 1, 0, 6), fibonacci=True, arp_hits=5,
                 layers=("bass", "chug", "arp", "noise"), seed=109),
            Visual(system="dadras", palette="toxic", nebula="venom", seed=19)),
    # Finale: 9-8-7 cycling meter, tribal toms, square bass on Fibonacci accents, choir, strings, timpani.
    Episode(10, "Lorenz · Finale",
            Spec(bpm=76, meters=(9, 8, 7), bars_per_chord=3, roots=(40, 36, 43, 38), drums="tribal",
                 bass_timbre="square", bass_hits=5, bass_notes=(0, 7, 0, 1, 10), fibonacci=True,
                 layers=("bass", "chug", "choir", "strings", "timpani", "arp", "wall"), seed=110),
            Visual(system="lorenz", palette="blood", nebula="crimson", particles=9000, seed=20)),
]  # fmt: skip


def build_episode(ep: Episode) -> Path:
    out_dir = OUTPUT / "series-v2"
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
