"""The grunge series: ten upbeat pieces in the Seattle style, each with its own song and name.

Every episode keeps the mystical look (abstract figure, glowing gradients, nebula) and gets a
unique musical identity: feel, tempo, key, riff and turnaround, chorus progression and strum,
quiet or loud verses, plugged or unplugged, and its own palette. The melody is played by twin
lead guitars. 194.159 s each (3:14).

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
    # Upbeat crunch in E Mixolydian: clean verse, loud chorus, wah solo.
    Episode(1, "Sunburn",
            Spec(bpm=122, tonic=40, scale=MIXOLYDIAN, riff_bars=2, drive=5.5, chord_bars=1, quiet_verse=True,
                 riff=((0, 2, 0, "p"), (2, 1, 3, "b"), (3, 1, 0, "m"), (4, 2, -2, "p"), (6, 2, 5, "p"),
                       (8, 2, 0, "p"), (10, 1, 7, "n"), (11, 1, 9, "n"), (12, 2, 10, "b"), (14, 2, 5, "p")),
                 chorus_roots=(0, -2, 5, 0), chorus_strum="x.xxx.xx", chorus_strum_b="x.x.x.xx", drums="rock",
                 voice_center=60, lead_scale=MAJOR_PENTATONIC, layers=("bass", "vocals", "lead"), reverb_s=2.4,
                 seed=301),
            Visual(system="chen", palette="nicotine", nebula="sepia", seed=41)),
    # Swinging 12/8 shuffle: acoustic verses, electric choruses.
    Episode(2, "Tidewater",
            Spec(bpm=96, meter=12, subdiv=3, tonic=38, scale=MIXOLYDIAN, verse="acoustic", chorus="both",
                 intro="acoustic", chorus_roots=(0, 5, -2, 0), chord_bars=1, chorus_strum="x..x..x..x.o",
                 drums="shuffle", drive=5.0, voice_center=60, lead_scale=MAJOR_PENTATONIC,
                 layers=("bass", "vocals", "lead", "acoustic"), reverb_s=2.6, seed=302),
            Visual(system="fourwing", palette="rain", nebula="slate", seed=42)),
    # Chugging D Dorian riff with wah on it, syncopated chorus hits.
    Episode(3, "Splinter",
            Spec(bpm=128, tonic=38, scale=DORIAN, drive=6.0,
                 riff=((0, 1, 0, "m"), (1, 1, 0, "m"), (2, 2, 3, "p"), (4, 1, 0, "m"), (5, 1, 5, "p"), (6, 2, 3, "b")),
                 chorus_roots=(0, 5, 3, 7), chord_bars=1, chorus_strum="x..x..x.", chorus_strum_b="x..x.xxx",
                 drums="rock", wah="riff", voice_center=60, layers=("bass", "vocals", "lead"), reverb_s=2.4,
                 seed=303),
            Visual(system="arneodo", palette="ash", nebula="smoke", seed=43)),
    # Unplugged and bright: strummed G major, cello solo.
    Episode(4, "Porch Light",
            Spec(bpm=116, tonic=43, scale=IONIAN, verse="acoustic", chorus="acoustic", intro="acoustic",
                 chorus_roots=(0, 7, 9, 5), chord_bars=1, chorus_strum="x.xo.oxo", chorus_strum_b="x.xox.xx",
                 drums="rock", voice_center=62, lead_scale=MAJOR_PENTATONIC,
                 layers=("bass", "vocals", "lead", "acoustic", "cello"), reverb_s=2.6, seed=304),
            Visual(family="magnetic", dipoles=3, palette="nicotine", nebula="sepia", particles=9000, size=1.8,
                   trail=0.965, seed=44)),
    # Fast and loud in A: palm-muted gallop, quiet verse, eighth-note chorus wall.
    Episode(5, "Gravel Road",
            Spec(bpm=144, tonic=45, scale=MIXOLYDIAN, drive=6.5, quiet_verse=True,
                 riff=((0, 1, 0, "m"), (1, 1, 0, "m"), (2, 1, 0, "p"), (3, 1, -2, "p"),
                       (4, 1, 0, "m"), (5, 1, 0, "m"), (6, 2, 5, "p")),
                 chorus_roots=(0, -2, 5, 7), chord_bars=1, chorus_strum="xxxxxxxx", chorus_strum_b="x.x.xxxx",
                 drums="rock", voice_center=62, lead_scale=MAJOR_PENTATONIC, layers=("bass", "vocals", "lead"),
                 reverb_s=2.0, seed=305),
            Visual(system="aizawa", palette="fire", nebula="inferno", seed=45)),
    # Ringing D Dorian riff over acoustic picking; sustained chorus chords.
    Episode(6, "Sun Break",
            Spec(bpm=118, tonic=38, scale=DORIAN, verse="both", intro="acoustic", drive=5.5,
                 riff=((0, 2, 0, "p"), (2, 2, 3, "n"), (4, 1, 5, "b"), (5, 3, 7, "v")),
                 chorus_roots=(0, 5, 3, 10), chord_bars=1, chorus_strum="x...x.xo", chorus_strum_b="x.x.x.xx",
                 drums="rock", voice_center=62, layers=("bass", "vocals", "lead", "acoustic"), reverb_s=2.4,
                 seed=306),
            Visual(system="thomas", palette="aurora", nebula="slate", seed=46)),
    # The fastest: driving E♭ riff, quiet verse, wah solo.
    Episode(7, "Static Sky",
            Spec(bpm=138, tonic=39, scale=MIXOLYDIAN, drive=6.5, quiet_verse=True,
                 riff=((0, 1, 0, "m"), (1, 1, 0, "m"), (2, 1, 4, "n"), (3, 1, 0, "m"),
                       (4, 1, 5, "n"), (5, 1, 7, "n"), (6, 2, 10, "b")),
                 chorus_roots=(0, -2, 5, 0), chord_bars=1, chorus_strum="x.xxx.xx", drums="rock",
                 voice_center=63, lead_scale=MAJOR_PENTATONIC, layers=("bass", "vocals", "lead"), reverb_s=2.0,
                 seed=307),
            Visual(system="lorenz", palette="bruise", nebula="dusk", seed=47)),
    # Rolling 6/8 in C major: acoustic verses, electric choruses, cello.
    Episode(8, "Paper Kite",
            Spec(bpm=84, meter=6, subdiv=3, tonic=36, scale=IONIAN, verse="acoustic", chorus="both",
                 intro="acoustic", chorus_roots=(0, 5, 9, 7), chord_bars=1, chorus_strum="x..x.o",
                 drums="rock", drive=5.0, voice_center=60, lead_scale=MAJOR_PENTATONIC,
                 layers=("bass", "vocals", "lead", "acoustic", "cello"), reverb_s=2.8, seed=308),
            Visual(family="flow", palette="ember", nebula="dusk", particles=10000, size=1.8, trail=0.965, seed=48)),
    # E Dorian two-bar riff with wah on it; quiet verse, eighth-note chorus.
    Episode(9, "Copper Wire",
            Spec(bpm=126, tonic=40, scale=DORIAN, riff_bars=2, drive=6.0, quiet_verse=True,
                 riff=((0, 2, 0, "p"), (2, 1, 3, "n"), (3, 1, 5, "p"), (4, 2, 7, "p"), (6, 2, 10, "b"),
                       (8, 3, 0, "p"), (11, 1, 3, "n"), (12, 4, 5, "v")),
                 chorus_roots=(0, 5, -2, 3), chord_bars=1, chorus_strum="xxxxxxxx", chorus_strum_b="x..x..xx",
                 drums="rock", wah="riff", voice_center=61, layers=("bass", "vocals", "lead"), reverb_s=2.4,
                 seed=309),
            Visual(system="halvorsen", palette="rust", nebula="sepia", particles=9000, seed=49)),
    # Finale: an anthem in D, acoustic and electric together, long last chorus.
    Episode(10, "Last Light",
            Spec(bpm=128, tonic=38, scale=IONIAN, riff_bars=2, drive=6.0, verse="both", chorus="both",
                 intro="acoustic", chord_bars=1,
                 riff=((0, 2, 0, "p"), (2, 2, 7, "p"), (4, 2, 9, "p"), (6, 2, 5, "b"),
                       (8, 3, 0, "p"), (11, 1, 2, "n"), (12, 4, 4, "v")),
                 form=(("intro", 4), ("verse", 8), ("chorus", 8), ("bridge", 6), ("verse", 8), ("chorus", 8),
                       ("solo", 8), ("chorus", 12), ("outro", 6)),
                 chorus_roots=(0, 7, 9, 5), chorus_strum="x.xxx.xo", chorus_strum_b="x.x.xxxx", drums="rock",
                 voice_center=62, lead_scale=MAJOR_PENTATONIC,
                 layers=("bass", "vocals", "lead", "acoustic", "cello"), reverb_s=2.6, seed=310),
            Visual(system="dadras", palette="fire", nebula="sepia", particles=9000, seed=50)),
]  # fmt: skip


def main() -> None:
    wanted = {int(a) for a in sys.argv[1:]}
    for ep in EPISODES:
        if not wanted or ep.number in wanted:
            build_episode(ep, FOLDER, render_track)


if __name__ == "__main__":
    main()
