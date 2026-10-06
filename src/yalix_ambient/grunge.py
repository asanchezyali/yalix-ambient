"""The grunge series: ten upbeat pieces in the Seattle style, each with its own song and name.

Every episode keeps the mystical look (abstract figure, glowing gradients, nebula) and gets a
unique musical identity: meter, feel, tuning, riff, chorus progression, vocal harmony
interval, plugged or unplugged, and its own palette. 194.159 s each (3:14).

    uv run yalix-grunge            # render all ten
    uv run yalix-grunge 3 7        # render episodes 3 and 7
"""

from __future__ import annotations

import sys

from yalix_ambient.gl_particles import Visual
from yalix_ambient.music_grunge import DORIAN, IONIAN, MAJOR_PENTATONIC, MIXOLYDIAN, Spec, render_track
from yalix_ambient.series import Episode, build_episode

FOLDER = "grunge"

EPISODES = [
    # Upbeat drop-D crunch in E Mixolydian: bluesy bends, thirds in the chorus, wah solo.
    Episode(1, "Sunburn",
            Spec(bpm=122, tonic=40, scale=MIXOLYDIAN, riff_bars=2, drive=5.5, chord_bars=1,
                 riff=((0, 2, 0, "p"), (2, 1, 3, "b"), (3, 1, 0, "m"), (4, 2, -2, "p"), (6, 2, 5, "p"),
                       (8, 2, 0, "p"), (10, 1, 7, "n"), (11, 1, 9, "n"), (12, 2, 10, "b"), (14, 2, 5, "p")),
                 chorus_roots=(0, -2, 5, 0), drums="rock", harmony=2, voice_center=60, vowels="aoea",
                 lead_scale=MAJOR_PENTATONIC, layers=("bass", "vocals", "lead"), reverb_s=2.4, seed=301),
            Visual(system="chen", palette="nicotine", nebula="sepia", seed=41)),
    # Swinging 12/8 shuffle: acoustic verses, electric choruses, parallel fourths.
    Episode(2, "Tidewater",
            Spec(bpm=96, meter=12, subdiv=3, tonic=38, scale=MIXOLYDIAN, verse="acoustic", chorus="both",
                 intro="acoustic", chorus_roots=(0, 5, -2, 0), chord_bars=1, drums="shuffle", drive=5.0,
                 harmony=3, voice_center=60, vowels="oaea", lead_scale=MAJOR_PENTATONIC,
                 layers=("bass", "vocals", "lead", "acoustic"), reverb_s=2.6, seed=302),
            Visual(system="fourwing", palette="rain", nebula="slate", seed=42)),
    # Quick 7/8 in D Dorian: wah on the riff, fifths in the harmony.
    Episode(3, "Splinter",
            Spec(bpm=132, meter=7, tonic=38, scale=DORIAN, drive=6.0,
                 riff=((0, 2, 0, "p"), (2, 1, 3, "n"), (3, 1, 5, "n"), (4, 1, 0, "m"), (5, 2, 9, "b")),
                 chorus_roots=(0, 5, 3, 7), chord_bars=1, drums="rock", kick_pat="x..x.x.", snare_pat="..x...x",
                 wah="riff", harmony=4, voice_center=60, vowels="aeao", layers=("bass", "vocals", "lead"),
                 reverb_s=2.4, seed=303),
            Visual(system="arneodo", palette="ash", nebula="smoke", seed=43)),
    # Unplugged and bright: strummed G major, cello solo, a full rock beat on the drums.
    Episode(4, "Porch Light",
            Spec(bpm=116, tonic=43, scale=IONIAN, verse="acoustic", chorus="acoustic", intro="acoustic",
                 chorus_roots=(0, 7, 9, 5), chord_bars=1, drums="rock", harmony=2, voice_center=62,
                 vowels="aoea", lead_scale=MAJOR_PENTATONIC,
                 layers=("bass", "vocals", "lead", "acoustic", "cello"), reverb_s=2.6, seed=304),
            Visual(family="magnetic", dipoles=3, palette="nicotine", nebula="sepia", particles=9000, size=1.8,
                   trail=0.965, seed=44)),
    # Punk-tempo grunge in A: palm-muted gallop, a big major chorus.
    Episode(5, "Gravel Road",
            Spec(bpm=144, tonic=45, scale=MIXOLYDIAN, drive=6.5,
                 riff=((0, 1, 0, "m"), (1, 1, 0, "m"), (2, 1, 0, "p"), (3, 1, -2, "p"),
                       (4, 1, 0, "m"), (5, 1, 0, "m"), (6, 2, 5, "p")),
                 chorus_roots=(0, -2, 5, 7), chord_bars=1, drums="rock", harmony_semi=5, voice_center=62,
                 vowels="aeoa", lead_scale=MAJOR_PENTATONIC, layers=("bass", "vocals", "lead"), reverb_s=2.0,
                 seed=305),
            Visual(system="aizawa", palette="fire", nebula="inferno", seed=45)),
    # 5/4 that keeps moving: acoustic intro, both guitars in the verses, harmony a third below.
    Episode(6, "Sun Break",
            Spec(bpm=120, meter=10, tonic=38, scale=DORIAN, verse="both", intro="acoustic", drive=5.5,
                 riff=((0, 2, 0, "p"), (2, 2, 3, "n"), (4, 1, 5, "b"), (5, 3, 7, "v"), (8, 2, 10, "p")),
                 chorus_roots=(0, 5, 3, 10), chord_bars=1, drums="rock", kick_pat="x....x.x..",
                 snare_pat="...x....x.", harmony=-2, voice_center=62, vowels="eaoa",
                 layers=("bass", "vocals", "lead", "acoustic"), reverb_s=2.4, seed=306),
            Visual(system="thomas", palette="aurora", nebula="slate", seed=46)),
    # The fastest: driving E♭ riff, high lead voice, wah solo.
    Episode(7, "Static Sky",
            Spec(bpm=138, tonic=39, scale=MIXOLYDIAN, drive=6.5,
                 riff=((0, 1, 0, "m"), (1, 1, 0, "m"), (2, 1, 4, "n"), (3, 1, 0, "m"),
                       (4, 1, 5, "n"), (5, 1, 7, "n"), (6, 2, 10, "b")),
                 chorus_roots=(0, -2, 5, 0), chord_bars=1, drums="rock", harmony=2, voice_center=63,
                 vowels="aeoa", lead_scale=MAJOR_PENTATONIC, layers=("bass", "vocals", "lead"), reverb_s=2.0,
                 seed=307),
            Visual(system="lorenz", palette="bruise", nebula="dusk", seed=47)),
    # Rolling 6/8 in C major: acoustic verses, electric choruses, fourths.
    Episode(8, "Paper Kite",
            Spec(bpm=84, meter=6, subdiv=3, tonic=36, scale=IONIAN, verse="acoustic", chorus="both",
                 intro="acoustic", chorus_roots=(0, 5, 9, 7), chord_bars=1, drums="rock", kick_pat="x..x..",
                 snare_pat="...x..", drive=5.0, harmony=3, voice_center=60, vowels="oaea",
                 lead_scale=MAJOR_PENTATONIC, layers=("bass", "vocals", "lead", "acoustic", "cello"), reverb_s=2.8,
                 seed=308),
            Visual(family="flow", palette="ember", nebula="dusk", particles=10000, size=1.8, trail=0.965, seed=48)),
    # Driving 9/8 in E Dorian: wah on the riff, parallel minor thirds.
    Episode(9, "Copper Wire",
            Spec(bpm=126, meter=9, tonic=40, scale=DORIAN, drive=6.0,
                 riff=((0, 2, 0, "p"), (2, 1, 3, "n"), (3, 2, 5, "p"), (5, 1, 7, "n"), (6, 3, 10, "b")),
                 chorus_roots=(0, 5, -2, 3), chord_bars=1, drums="rock", kick_pat="x..x..x..",
                 snare_pat="..x..x..x", wah="riff", harmony_semi=3, voice_center=61, vowels="aeau",
                 layers=("bass", "vocals", "lead"), reverb_s=2.4, seed=309),
            Visual(system="halvorsen", palette="rust", nebula="sepia", particles=9000, seed=49)),
    # Finale: an anthem in D, acoustic and electric together, long last chorus.
    Episode(10, "Last Light",
            Spec(bpm=128, tonic=38, scale=IONIAN, riff_bars=2, drive=6.0, verse="both", chorus="both",
                 intro="acoustic", chord_bars=1,
                 riff=((0, 2, 0, "p"), (2, 2, 7, "p"), (4, 2, 9, "p"), (6, 2, 5, "b"),
                       (8, 3, 0, "p"), (11, 1, 2, "n"), (12, 4, 4, "v")),
                 form=(("intro", 4), ("verse", 8), ("chorus", 8), ("bridge", 6), ("verse", 8), ("chorus", 8),
                       ("solo", 8), ("chorus", 12), ("outro", 6)),
                 chorus_roots=(0, 7, 9, 5), drums="rock", harmony=2, voice_center=62, vowels="aoea",
                 lead_scale=MAJOR_PENTATONIC, layers=("bass", "vocals", "lead", "acoustic", "cello"),
                 reverb_s=2.6, seed=310),
            Visual(system="dadras", palette="fire", nebula="sepia", particles=9000, seed=50)),
]  # fmt: skip


def main() -> None:
    wanted = {int(a) for a in sys.argv[1:]}
    for ep in EPISODES:
        if not wanted or ep.number in wanted:
            build_episode(ep, FOLDER, render_track)


if __name__ == "__main__":
    main()
