"""The grunge series: ten pieces in the Seattle style, each with its own song.

Every episode keeps the mystical look (abstract figure, glowing gradients, nebula) and gets a
unique musical identity: meter, feel, tuning, riff, chorus progression, vocal harmony
interval, plugged or unplugged, and its own palette. 194.159 s each (3:14).

    uv run yalix-grunge            # render all ten
    uv run yalix-grunge 3 7        # render episodes 3 and 7
"""

from __future__ import annotations

import sys

from yalix_ambient.gl_particles import Visual
from yalix_ambient.music_grunge import AEOLIAN, DORIAN, PHRYGIAN, Spec, render_track
from yalix_ambient.series import Episode, build_episode

FOLDER = "grunge"

EPISODES = [
    # Mid-tempo drop-D sludge: bent riff with a minor-second slide, thirds in the chorus, wah solo.
    Episode(1, "Lorenz · Rust",
            Spec(bpm=84, tonic=38, scale=AEOLIAN, riff_bars=2,
                 riff=((0, 3, 0, "p"), (3, 1, 0, "m"), (4, 2, 3, "b"), (6, 2, 1, "s"),
                       (8, 2, 0, "p"), (10, 1, -2, "n"), (11, 1, -1, "n"), (12, 2, 5, "p"), (14, 2, 6, "t")),
                 chorus_roots=(0, 3, -2, 5), drums="rock", harmony=2, voice_center=55, vowels="aoea",
                 layers=("bass", "vocals", "lead"), seed=201),
            Visual(system="lorenz", palette="rust", nebula="sepia", seed=31)),
    # Slow 12/8 lament: unplugged verses, heavy choruses, parallel fourths, cello and drone.
    Episode(2, "Thomas · Undertow",
            Spec(bpm=56, meter=12, subdiv=3, tonic=38, scale=AEOLIAN, verse="acoustic", chorus="both",
                 intro="acoustic", chorus_roots=(0, -4, -2, -5), drums="shuffle", harmony=3, voice_center=55,
                 vowels="oaou", layers=("bass", "vocals", "lead", "acoustic", "cello", "drone"), reverb_s=4.0,
                 seed=202),
            Visual(system="thomas", palette="ice", nebula="abyss", seed=32)),
    # Jagged 7/8 in E♭ Phrygian: wah on the riff itself, tritone dyads, fifths in the harmony.
    Episode(3, "Chen · Splinter",
            Spec(bpm=100, meter=7, tonic=39, scale=PHRYGIAN,
                 riff=((0, 2, 0, "p"), (2, 1, 1, "m"), (3, 1, 0, "m"), (4, 1, 6, "t"), (5, 2, 3, "b")),
                 chorus_roots=(0, 1, -2, 3), drums="rock", kick_pat="x..x.x.", snare_pat="..x...x",
                 wah="riff", harmony=4, voice_center=57, vowels="aeao", layers=("bass", "vocals", "lead"),
                 seed=203),
            Visual(system="chen", palette="ash", nebula="smoke", seed=33)),
    # Fully unplugged: fingerpicking, strummed choruses, brushes, cello solo and rain.
    Episode(4, "Dipoles · Moth",
            Spec(bpm=76, tonic=40, scale=DORIAN, verse="acoustic", chorus="acoustic", intro="acoustic",
                 chorus_roots=(0, -2, -4, -5), drums="brush", harmony=2, voice_center=57, vowels="aoua",
                 layers=("bass", "vocals", "lead", "acoustic", "cello", "rain"), reverb_s=3.6, seed=204),
            Visual(family="magnetic", dipoles=2, palette="nicotine", nebula="sepia", particles=9000, size=1.8,
                   trail=0.965, seed=34)),
    # Doom crawl in C#: half-step and tritone riff, ride-heavy sludge drums, chromatic parallel fourths.
    Episode(5, "Halvorsen · Sludge",
            Spec(bpm=56, tonic=37, scale=PHRYGIAN, riff_bars=2, drive=9.0,
                 riff=((0, 4, 0, "p"), (4, 2, 1, "s"), (6, 2, 0, "m"),
                       (8, 3, -1, "b"), (11, 1, 0, "m"), (12, 4, 6, "t")),
                 chorus_roots=(0, 1, 6, 1), drums="sludge", harmony_semi=5, voice_center=54, vowels="ouao",
                 intro="drone", layers=("bass", "vocals", "lead", "drone"), reverb_s=4.2, seed=205),
            Visual(system="halvorsen", palette="moss", nebula="swamp", particles=9000, seed=35)),
    # 5/4 under a grey sky: acoustic intro, verses with both guitars, harmony a third below, rain.
    Episode(6, "Rössler · Overcast",
            Spec(bpm=72, meter=10, tonic=38, scale=DORIAN, verse="both", intro="acoustic",
                 riff=((0, 2, 0, "p"), (2, 2, 3, "n"), (4, 1, 5, "b"), (5, 3, 3, "v"), (8, 2, -2, "p")),
                 chorus_roots=(0, 5, 3, -2), drums="rock", kick_pat="x....x.x..", snare_pat="...x....x.",
                 harmony=-2, voice_center=59, vowels="eaoa",
                 layers=("bass", "vocals", "lead", "acoustic", "cello", "rain"), seed=206),
            Visual(system="rossler", palette="rain", nebula="slate", seed=36)),
    # The fast one: driving palm-muted riff in E♭, high lead voice, wah solo.
    Episode(7, "Four-Wing · Static Sky",
            Spec(bpm=112, tonic=39, scale=AEOLIAN, drive=8.0,
                 riff=((0, 1, 0, "m"), (1, 1, 0, "m"), (2, 1, 3, "n"), (3, 1, 0, "m"),
                       (4, 1, 5, "n"), (5, 1, 6, "n"), (6, 2, 3, "b")),
                 chorus_roots=(0, -4, -2, 3), drums="rock", harmony=2, voice_center=60, vowels="aeoa",
                 layers=("bass", "vocals", "lead"), seed=207),
            Visual(system="fourwing", palette="bruise", nebula="dusk", seed=37)),
    # 6/8 ballad: drone and hummed intro, acoustic verses, big choruses, cello, fourths.
    Episode(8, "Nebula · Hollow",
            Spec(bpm=50, meter=6, subdiv=3, tonic=38, scale=AEOLIAN, verse="acoustic", chorus="both",
                 intro="drone", chorus_roots=(0, 3, -4, -2), drums="rock", kick_pat="x.....", snare_pat="...x..",
                 harmony=3, voice_center=55, vowels="ouoa",
                 layers=("bass", "vocals", "lead", "acoustic", "cello", "drone"), reverb_s=4.5, seed=208),
            Visual(family="flow", palette="ember", nebula="dusk", particles=10000, size=1.8, trail=0.965, seed=38)),
    # 9/8 grinder: wah on the riff, parallel minor thirds that never resolve.
    Episode(9, "Aizawa · Needle",
            Spec(bpm=96, meter=9, tonic=38, scale=PHRYGIAN,
                 riff=((0, 2, 0, "p"), (2, 1, 1, "m"), (3, 2, 0, "p"), (5, 1, 6, "t"), (6, 3, 5, "b")),
                 chorus_roots=(0, 1, 3, 1), drums="rock", kick_pat="x..x..x..", snare_pat="..x..x..x",
                 wah="riff", harmony_semi=3, voice_center=57, vowels="aeau", layers=("bass", "vocals", "lead"),
                 seed=209),
            Visual(system="aizawa", palette="blood", nebula="smoke", seed=39)),
    # Finale: half-time weight, descending power chords, acoustic and electric together, everything.
    Episode(10, "Dadras · Last Light",
            Spec(bpm=76, tonic=38, scale=AEOLIAN, riff_bars=2, drive=8.0, verse="both", chorus="both",
                 intro="acoustic",
                 riff=((0, 2, 0, "p"), (2, 2, -2, "p"), (4, 2, -4, "p"), (6, 2, -5, "b"),
                       (8, 3, 0, "p"), (11, 1, 1, "m"), (12, 4, 3, "v")),
                 form=(("intro", 4), ("verse", 8), ("chorus", 8), ("bridge", 6), ("verse", 8), ("chorus", 8),
                       ("solo", 8), ("chorus", 12), ("outro", 6)),
                 chorus_roots=(0, -4, 3, -2), drums="halftime", harmony=2, voice_center=57, vowels="aoea",
                 layers=("bass", "vocals", "lead", "acoustic", "cello", "drone"), seed=210),
            Visual(system="dadras", palette="fire", nebula="sepia", particles=9000, seed=40)),
]  # fmt: skip


def main() -> None:
    wanted = {int(a) for a in sys.argv[1:]}
    for ep in EPISODES:
        if not wanted or ep.number in wanted:
            build_episode(ep, FOLDER, render_track)


if __name__ == "__main__":
    main()
