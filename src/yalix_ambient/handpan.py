"""Handpan Reggae: nineteen themes for focus, 194.159 s each (3:14).

A synthesized steel handpan over slow reggae, each theme in its own style (roots, lovers rock,
dub, steppers, nyabinghi or ambient), handpan layout and key, so neighbouring themes never share
a colour. The figures are the cymatics families, all new for this mix: sand on a drum membrane,
rings on water, mandalas and Turing patterns, in daylight palettes. Every handpan strike moves
the figure.

    uv run yalix-mix handpan
"""

from __future__ import annotations

from yalix_ambient.gl_particles import Visual
from yalix_ambient.music_handpan import AEOLIAN, DORIAN, IONIAN, LYDIAN, MIXOLYDIAN, Spec
from yalix_ambient.series import Episode


def _v(family: str, system: str, palette: str, nebula: str, seed: int, **kw) -> Visual:
    base = dict(particles=9000, size=1.8, trail=0.965, intensity=0.9, bloom=0.45)
    if family == "ripples":
        base.update(trail=0.55, intensity=1.3, size=2.0)
    return Visual(family=family, system=system, palette=palette, nebula=nebula, seed=seed, **{**base, **kw})


EPISODES = [
    Episode(1, "Drum Membrane · Ding",
            Spec(bpm=72, tonic=50, layout="kurd", mode=AEOLIAN, progression=(0, 5, 6, 5), style="roots",
                 hand="groove", hits=5, seed=901),
            _v("membrane", "drum", "lagoon", "shallows", 201)),
    Episode(2, "Ripples · Low Tide",
            Spec(bpm=60, tonic=50, layout="aegean", mode=LYDIAN, progression=(0, 1), style="ambient", hand="melodic",
                 ring=4.0, extra=("gu",), reverb_s=5.0, seed=902),
            _v("ripples", "pond", "tide", "shallows", 202)),
    Episode(3, "Mandala of Eight · Island Time",
            Spec(bpm=82, tonic=50, layout="sabye", mode=IONIAN, progression=(0, 3, 4, 3), style="lovers",
                 hand="melodic", hits=6, extra=("taps", "shaker"), seed=903),
            _v("mandala", "8", "jungle", "canopy", 203)),
    Episode(4, "Turing Coral · Coral Garden",
            Spec(bpm=68, tonic=48, layout="celtic", mode=DORIAN, progression=(0, 3), style="dub", hand="melodic",
                 extra=("gu",), seed=904),
            _v("turing", "coral", "reef", "shallows", 204)),
    Episode(5, "Bessel Rings · Salt Air",
            Spec(bpm=64, tonic=52, layout="pygmy", mode=AEOLIAN, progression=(0, 3), style="nyabinghi", hand="groove",
                 extra=("gu", "taps"), seed=905),
            _v("membrane", "rings", "sandstone", "clay", 205)),
    Episode(6, "Raindrops · Hammock",
            Spec(bpm=86, tonic=53, layout="oxalis", mode=MIXOLYDIAN, progression=(0, 6), style="steppers",
                 hand="flow", extra=("taps", "shaker"), seed=906),
            _v("ripples", "rain", "reef", "twilight", 206)),
    Episode(7, "Mandala of Twelve · Trade Winds",
            Spec(bpm=74, tonic=52, layout="amara", mode=DORIAN, progression=(0, 3, 6, 3), style="roots",
                 hand="groove", hits=6, swing=0.16, seed=907),
            _v("mandala", "12", "sunrise", "golden", 207)),
    Episode(8, "Turing Maze · Slow Current",
            Spec(bpm=62, tonic=53, layout="equinox", mode=AEOLIAN, progression=(0, 5), style="ambient",
                 hand="groove", hits=4, ring=3.6, extra=("gu", "sea"), reverb_s=5.0, seed=908),
            _v("turing", "maze", "roots", "canopy", 208)),
    Episode(9, "Bessel Petals · Banyan Shade",
            Spec(bpm=80, tonic=55, layout="sabye", mode=IONIAN, progression=(0, 5, 3, 4), style="lovers",
                 hand="groove", hits=5, extra=("taps",), seed=909),
            _v("membrane", "petals", "jungle", "canopy", 209)),
    Episode(10, "Two Springs · Morning Dub",
             Spec(bpm=70, tonic=50, layout="kurd", mode=AEOLIAN, progression=(0, 3), style="dub", hand="groove",
                  hits=4, extra=("gu", "taps"), echo=4, seed=910),
             _v("ripples", "two", "lagoon", "shallows", 210)),
    Episode(11, "Turing Spots · Sunwater",
             Spec(bpm=88, tonic=50, layout="celtic", mode=DORIAN, progression=(0, 3, 0, 6), style="steppers",
                  hand="groove", hits=6, extra=("taps",), seed=911),
             _v("turing", "spots", "sunrise", "golden", 211)),
    Episode(12, "Mandala of Six · Seashell",
             Spec(bpm=66, tonic=50, layout="amara", mode=DORIAN, progression=(0, 6), style="nyabinghi", hand="flow",
                  extra=("gu", "sea"), seed=912),
             _v("mandala", "6", "sandstone", "clay", 212)),
    Episode(13, "Drum Membrane · Lagoon",
             Spec(bpm=70, tonic=48, layout="pygmy", mode=DORIAN, progression=(0, 3), style="roots", hand="flow",
                  extra=("gu", "taps", "shaker"), seed=913),
             _v("membrane", "deep", "tide", "shallows", 213)),
    Episode(14, "Spring · Driftwood",
             Spec(bpm=58, tonic=52, layout="celtic", mode=AEOLIAN, progression=(0, 5, 6, 0), style="ambient",
                  hand="melodic", ring=4.2, extra=("gu", "sea"), reverb_s=5.5, seed=914),
             _v("ripples", "spring", "sandstone", "golden", 214)),
    Episode(15, "Turing Mitosis · Blue Hour",
             Spec(bpm=78, tonic=53, layout="oxalis", mode=MIXOLYDIAN, progression=(0, 3, 6, 3), style="lovers",
                  hand="flow", extra=("taps", "shaker"), seed=915),
             _v("turing", "mitosis", "tide", "twilight", 215)),
    Episode(16, "Mandala of Ten · Mango Rain",
             Spec(bpm=84, tonic=52, layout="equinox", mode=AEOLIAN, progression=(0, 5, 6, 5), style="steppers",
                  hand="groove", hits=6, extra=("gu", "shaker"), seed=916),
             _v("mandala", "10", "roots", "canopy", 216)),
    Episode(17, "Bessel Drum · Steel Garden",
             Spec(bpm=66, tonic=50, layout="aegean", mode=LYDIAN, progression=(0, 1), style="dub", hand="melodic",
                  extra=("gu",), feedback=0.55, seed=917),
             _v("membrane", "drum", "sunrise", "golden", 217)),
    Episode(18, "Turing Worms · Moonwater",
             Spec(bpm=62, tonic=48, layout="kurd", mode=AEOLIAN, progression=(0, 3), style="nyabinghi",
                  hand="melodic", hits=6, extra=("gu", "taps"), seed=918),
             _v("turing", "worms", "lagoon", "twilight", 218)),
    Episode(19, "Ripples · Last Ferry",
             Spec(bpm=56, tonic=50, layout="amara", mode=DORIAN, progression=(0, 6, 3, 0), style="ambient",
                  hand="melodic", hits=5, ring=4.5, extra=("gu", "sea"), reverb_s=6.0, seed=919),
             _v("ripples", "pond", "tide", "twilight", 219)),
]  # fmt: skip
