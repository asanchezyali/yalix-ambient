"""Industrial Meditation: nineteen themes for deep work, 194.159 s each (3:14).

The weight of German industrial metal at the pace of a slow breath, every sound synthesized:
a gated wall of guitars, a pulse bass, struck sheet metal, oil drums, pipes and a hydraulic
press, a furnace drone and a low lead. Each theme has its own style (stomp, march, foundry,
press, machine or furnace) and key, so neighbouring themes never share a colour. The figures
are the factory families, new for this mix: grinder and weld sparks, four-bar linkages, steel
structures that ring when struck and convection rolls in molten metal. Every hit moves them.

    uv run yalix-mix industrial
"""

from __future__ import annotations

from yalix_ambient.gl_particles import Visual
from yalix_ambient.music_industrial import DORIAN, PHRYGIAN, Spec
from yalix_ambient.series import Episode


def _v(family: str, system: str, palette: str, nebula: str, seed: int, **kw) -> Visual:
    base = dict(particles=9000, size=1.8, trail=0.93, intensity=0.9, bloom=0.45, hue_speed=0.3)
    base.update({
        "sparks": dict(trail=0.9, intensity=1.2, size=1.7, hue_speed=0.0),
        "linkage": dict(trail=0.9, intensity=1.1, size=2.0),
        "lattice": dict(trail=0.8, intensity=1.0, size=1.9),
        "convection": dict(trail=0.965, intensity=0.8, size=1.7),
    }[family])  # fmt: skip
    return Visual(family=family, system=system, palette=palette, nebula=nebula, seed=seed, **{**base, **kw})


EPISODES = [
    Episode(1, "Grinder Sparks · First Shift",
            Spec(bpm=74, tonic=36, style="stomp", progression=(0, 5, 2, 6), seed=1001),
            _v("sparks", "grinder", "molten", "smelter", 301)),
    Episode(2, "Hoekens Linkage · Piston Hall",
            Spec(bpm=80, tonic=38, style="machine", mode=DORIAN, progression=(0, 3), seed=1002),
            _v("linkage", "hoekens", "steel", "concrete", 302)),
    Episode(3, "Steel Lattice · Cold Iron",
            Spec(bpm=66, tonic=39, style="foundry", mode=PHRYGIAN, progression=(0, 1), seed=1003),
            _v("lattice", "cube", "steel", "arcnight", 303)),
    Episode(4, "Convection Rolls · Smelter",
            Spec(bpm=64, tonic=40, style="furnace", progression=(0, 5), seed=1004),
            _v("convection", "rolls", "molten", "smelter", 304)),
    Episode(5, "Warren Truss · Rail Yard",
            Spec(bpm=78, tonic=40, style="march", progression=(0, 5, 6, 0), seed=1005),
            _v("lattice", "truss", "hazard", "nightshift", 305)),
    Episode(6, "Weld Arc · Blue Seam",
            Spec(bpm=70, tonic=37, style="press", progression=(0, 3, 5, 4), seed=1006),
            _v("sparks", "weld", "arc", "arcnight", 306)),
    Episode(7, "Four Linkages · Assembly Line",
            Spec(bpm=76, tonic=38, style="stomp", progression=(0, 6, 5, 6), seed=1007),
            _v("linkage", "mixed", "sodium", "nightshift", 307)),
    Episode(8, "Pylon · High Voltage",
            Spec(bpm=68, tonic=36, style="foundry", mode=PHRYGIAN, progression=(0, 1, 0, 6), seed=1008),
            _v("lattice", "pylon", "sodium", "concrete", 308)),
    Episode(9, "Ladle Pour · Molten Core",
            Spec(bpm=62, tonic=38, style="furnace", mode=PHRYGIAN, progression=(0, 1), seed=1009),
            _v("sparks", "ladle", "molten", "smelter", 309)),
    Episode(10, "Convection Cells · Blast Furnace",
             Spec(bpm=80, tonic=37, style="march", progression=(0, 5, 3, 6), seed=1010),
             _v("convection", "cells", "sodium", "smelter", 310)),
    Episode(11, "Geodesic Dome · Night Shift",
             Spec(bpm=72, tonic=40, style="press", mode=DORIAN, progression=(0, 3), seed=1011),
             _v("lattice", "dome", "arc", "arcnight", 311)),
    Episode(12, "Chebyshev Linkage · Iron Lung",
             Spec(bpm=78, tonic=36, style="machine", progression=(0, 5), seed=1012),
             _v("linkage", "chebyshev", "hazard", "concrete", 312)),
    Episode(13, "Grinder Sparks · Steel Rain",
             Spec(bpm=72, tonic=40, style="stomp", mode=PHRYGIAN, progression=(0, 1, 5, 1), seed=1013),
             _v("sparks", "grinder", "sodium", "nightshift", 313)),
    Episode(14, "Steel Lattice · Hammer Hall",
             Spec(bpm=76, tonic=38, style="march", progression=(0, 2, 5, 6), seed=1014),
             _v("lattice", "cube", "molten", "smelter", 314)),
    Episode(15, "Convection Plume · Slag",
             Spec(bpm=66, tonic=40, style="foundry", progression=(0, 5, 6, 5), seed=1015),
             _v("convection", "plume", "molten", "nightshift", 315)),
    Episode(16, "Weld Arc · Spark Gap",
             Spec(bpm=82, tonic=37, style="machine", mode=PHRYGIAN, progression=(0, 1), seed=1016),
             _v("sparks", "weld", "steel", "arcnight", 316)),
    Episode(17, "Watt Linkage · Pressworks",
             Spec(bpm=70, tonic=38, style="press", progression=(0, 5, 2, 6), seed=1017),
             _v("linkage", "watt", "sodium", "smelter", 317)),
    Episode(18, "Pylon · Transformer",
             Spec(bpm=74, tonic=37, style="stomp", mode=DORIAN, progression=(0, 3, 0, 6), seed=1018),
             _v("lattice", "pylon", "steel", "arcnight", 318)),
    Episode(19, "Ladle Pour · Last Shift",
             Spec(bpm=60, tonic=36, style="furnace", progression=(0, 5, 3, 4), seed=1019),
             _v("sparks", "ladle", "sodium", "smelter", 319)),
]  # fmt: skip
