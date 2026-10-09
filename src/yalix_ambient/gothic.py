"""Gothic Trap: nineteen themes for night work, 194.159 s each (3:14).

A harpsichord in a crypt over a distorted 808: baroque ostinatos, a phonk cowbell hook, a
theremin, half-time trap drums and witch house tape stops, every sound synthesized. Each theme
has its own style (baroque, phonk, witch, requiem, crypt or masquerade), progression and key,
so neighbouring themes never share a colour. The figures are the cathedral families, new for
this mix: Apollonian rose windows, an endless nave of pointed arches, flocks of bats and
stained-glass lancets. Every hit moves them.

    uv run yalix-mix gothic
"""

from __future__ import annotations

from yalix_ambient.gl_particles import Visual
from yalix_ambient.music_gothic import Spec
from yalix_ambient.series import Episode


def _v(family: str, system: str, palette: str, nebula: str, seed: int, **kw) -> Visual:
    base = {"particles": 9000, "size": 1.8, "trail": 0.9, "intensity": 1.0, "bloom": 0.45, "hue_speed": 0.2}
    base.update({
        "apollonian": {"trail": 0.9, "intensity": 1.0},
        "nave": {"trail": 0.85, "intensity": 1.1, "size": 1.9},
        "flock": {"trail": 0.93, "intensity": 1.1, "size": 2.0},
        "glass": {"trail": 0.8, "intensity": 1.6, "size": 2.3},
    }[family])  # fmt: skip
    return Visual(family=family, system=system, palette=palette, nebula=nebula, seed=seed, **{**base, **kw})


EPISODES = [
    Episode(1, "Apollonian Window · Vespers",
            Spec(bpm=140, tonic=38, style="baroque", progression="andalusian", seed=2001),
            _v("apollonian", "classic", "candle", "crypt", 401)),
    Episode(2, "Nave · Midnight Mass",
            Spec(bpm=144, tonic=40, style="phonk", progression="descent", swing=0.08, seed=2002),
            _v("nave", "nave", "moonlight", "moonstone", 402)),
    Episode(3, "Bat Flock · Belfry",
            Spec(bpm=130, tonic=36, style="witch", progression="drone", chord_bars=2, seed=2003),
            _v("flock", "flock", "velvet", "wine", 403)),
    Episode(4, "Triple Lancet · Requiem",
            Spec(bpm=136, tonic=33, style="requiem", progression="lament", seed=2004),
            _v("glass", "lancet", "cathedral", "crypt", 404)),
    Episode(5, "Apollonian Window · Amethyst Hour",
            Spec(bpm=138, tonic=31, style="masquerade", progression="andalusian", seed=2005),
            _v("apollonian", "triple", "amethyst", "moonstone", 405)),
    Episode(6, "Nave · Crypt Walk",
            Spec(bpm=142, tonic=41, style="crypt", progression="cadence", chord_bars=2, seed=2006),
            _v("nave", "wide", "candle", "crypt", 406)),
    Episode(7, "Bat Colony · Black Velvet",
            Spec(bpm=146, tonic=38, style="phonk", progression="andalusian", swing=0.1, seed=2007),
            _v("flock", "colony", "velvet", "wine", 407)),
    Episode(8, "Rose Lancet · Absinthe",
            Spec(bpm=138, tonic=40, style="baroque", progression="lament", seed=2008),
            _v("glass", "rose", "absinthe", "crypt", 408)),
    Episode(9, "Bat Swarm · Gargoyles",
            Spec(bpm=128, tonic=35, style="witch", progression="drone", chord_bars=2, seed=2009),
            _v("flock", "swarm", "moonlight", "moonstone", 409)),
    Episode(10, "Apollonian Window · Candlemas",
             Spec(bpm=134, tonic=36, style="requiem", progression="descent", seed=2010),
             _v("apollonian", "classic", "candle", "wine", 410)),
    Episode(11, "Nave · Masked Ball",
             Spec(bpm=140, tonic=38, style="masquerade", progression="cadence", seed=2011),
             _v("nave", "high", "amethyst", "moonstone", 411)),
    Episode(12, "Small Lancets · Ossuary",
             Spec(bpm=144, tonic=40, style="crypt", progression="andalusian", chord_bars=2, seed=2012),
             _v("glass", "small", "moonlight", "crypt", 412)),
    Episode(13, "Bat Flock · Nocturne",
             Spec(bpm=136, tonic=33, style="baroque", progression="descent", seed=2013),
             _v("flock", "flock", "amethyst", "wine", 413)),
    Episode(14, "Apollonian Window · Black Candle",
             Spec(bpm=142, tonic=36, style="phonk", progression="lament", swing=0.08, seed=2014),
             _v("apollonian", "triple", "velvet", "crypt", 414)),
    Episode(15, "Nave · Cloister",
             Spec(bpm=132, tonic=31, style="requiem", progression="andalusian", seed=2015),
             _v("nave", "nave", "candle", "crypt", 415)),
    Episode(16, "Triple Lancet · Vigil",
             Spec(bpm=130, tonic=38, style="witch", progression="drone", chord_bars=2, seed=2016),
             _v("glass", "lancet", "cathedral", "wine", 416)),
    Episode(17, "Bat Colony · Danse Macabre",
             Spec(bpm=140, tonic=40, style="masquerade", progression="lament", seed=2017),
             _v("flock", "colony", "absinthe", "moonstone", 417)),
    Episode(18, "Rose Lancet · Last Rites",
             Spec(bpm=138, tonic=41, style="baroque", progression="cadence", seed=2018),
             _v("glass", "rose", "velvet", "crypt", 418)),
    Episode(19, "Nave · Dawn",
             Spec(bpm=136, tonic=36, style="crypt", progression="descent", chord_bars=2, seed=2019),
             _v("nave", "wide", "moonlight", "moonstone", 419)),
]  # fmt: skip
