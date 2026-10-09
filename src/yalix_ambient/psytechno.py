"""Psytechno: nineteen themes for high-energy focus, 194.159 s each (3:14).

The structure of techno with the drive of Goa trance: a hard kick and its rumble, the rolling
psy bass, tribal toms, a distorted growl, tabla, a tanpura drone, palmas in compás and a sitar
kept low, every sound synthesized. Each theme has its own style (warrior, ritual, hypno, goa,
flamenco or temple) and key, so neighbouring themes never share a colour. The figures are the
psychedelic families, new for this mix: yantras, hyperspace tunnels, chaos-game fractals and
moiré rings. Every hit moves them.

    uv run yalix-mix psytechno
"""

from __future__ import annotations

from yalix_ambient.gl_particles import Visual
from yalix_ambient.music_psy import DOUBLE_HARMONIC, PHRYGIAN, PHRYGIAN_DOM, Spec
from yalix_ambient.series import Episode


def _v(family: str, system: str, palette: str, nebula: str, seed: int, **kw) -> Visual:
    base = {"particles": 9000, "size": 1.8, "trail": 0.9, "intensity": 1.0, "bloom": 0.45, "hue_speed": 0.3}
    base.update({
        "yantra": {"trail": 0.85, "intensity": 1.1},
        "tunnel": {"trail": 0.8, "intensity": 1.2, "size": 2.0},
        "chaos": {"trail": 0.7, "intensity": 0.9, "size": 1.6},
        "moire": {"trail": 0.75, "intensity": 0.9, "size": 1.7},
    }[family])  # fmt: skip
    return Visual(family=family, system=system, palette=palette, nebula=nebula, seed=seed, **{**base, **kw})


EPISODES = [
    Episode(1, "Hyperspace Tunnel · Warriors",
            Spec(bpm=147, tonic=41, style="warrior", mode=PHRYGIAN, roots=(0, 0, 1, 0), seed=3001),
            _v("tunnel", "hex", "blacklight", "uvfloor", 501)),
    Episode(2, "Yantra · Goa Gate",
            Spec(bpm=145, tonic=40, style="goa", mode=DOUBLE_HARMONIC, roots=(0, 0, 1, 0), seed=3002),
            _v("yantra", "square", "saffron", "redsmoke", 502)),
    Episode(3, "Moiré Rings · Trance State",
            Spec(bpm=147, tonic=31, style="hypno", mode=PHRYGIAN, roots=(0, 0, 0, 1), seed=3003),
            _v("moire", "rings", "blacklight", "uvfloor", 503)),
    Episode(4, "Fivefold Fractal · Fire Ritual",
            Spec(bpm=146, tonic=38, style="ritual", mode=PHRYGIAN, roots=(0, 0, 1, 0), seed=3004),
            _v("chaos", "star", "lava", "redsmoke", 504)),
    Episode(5, "Lotus Yantra · Battle Cry",
            Spec(bpm=148, tonic=40, style="warrior", mode=PHRYGIAN, roots=(0, 0, -2, 0), seed=3005),
            _v("yantra", "lotus", "toxic", "nightjungle", 505)),
    Episode(6, "Star Tunnel · Duende",
            Spec(bpm=145, tonic=33, style="flamenco", mode=PHRYGIAN_DOM, roots=(0, -2, -4, -5), chord_bars=1, seed=3006),
            _v("tunnel", "star", "saffron", "redsmoke", 506)),
    Episode(7, "Sierpinski Flame · Temple Steps",
            Spec(bpm=144, tonic=38, style="temple", mode=DOUBLE_HARMONIC, roots=(0,), seed=3007),
            _v("chaos", "sierpinski", "shaman", "nightjungle", 507)),
    Episode(8, "Moiré Petals · Iron Horses",
            Spec(bpm=147, tonic=36, style="warrior", mode=PHRYGIAN, roots=(0, 0, 1, -2), seed=3008),
            _v("moire", "petals", "lava", "redsmoke", 508)),
    Episode(9, "Square Tunnel · Hypnosis",
            Spec(bpm=146, tonic=33, style="hypno", mode=PHRYGIAN_DOM, roots=(0, 0, 0, 1), seed=3009),
            _v("tunnel", "square", "toxic", "nightjungle", 509)),
    Episode(10, "Fivefold Fractal · Third Eye",
             Spec(bpm=145, tonic=38, style="goa", mode=DOUBLE_HARMONIC, roots=(0, 0, 1, 0), seed=3010),
             _v("chaos", "star", "blacklight", "uvfloor", 510)),
    Episode(11, "Yantra · Andalusian Night",
             Spec(bpm=145, tonic=35, style="flamenco", mode=PHRYGIAN_DOM, roots=(0, -2, -4, -5), chord_bars=1, seed=3011),
             _v("yantra", "square", "lava", "redsmoke", 511)),
    Episode(12, "Triangle Tunnel · Warpath",
             Spec(bpm=148, tonic=38, style="warrior", mode=PHRYGIAN, roots=(0, 0, 1, 0), seed=3012),
             _v("tunnel", "tri", "blacklight", "uvfloor", 512)),
    Episode(13, "Moiré Waves · Shaman",
             Spec(bpm=147, tonic=40, style="ritual", mode=PHRYGIAN, roots=(0, 0, 1, 0), seed=3013),
             _v("moire", "waves", "shaman", "nightjungle", 513)),
    Episode(14, "Lotus Yantra · Incense",
             Spec(bpm=144, tonic=36, style="temple", mode=DOUBLE_HARMONIC, roots=(0,), seed=3014),
             _v("yantra", "lotus", "saffron", "redsmoke", 514)),
    Episode(15, "Sierpinski Flame · Black Light",
             Spec(bpm=147, tonic=38, style="hypno", mode=PHRYGIAN, roots=(0, 0, 0, 1), seed=3015),
             _v("chaos", "sierpinski", "blacklight", "uvfloor", 515)),
    Episode(16, "Hex Tunnel · War Drums",
             Spec(bpm=148, tonic=41, style="warrior", mode=PHRYGIAN, roots=(0, 0, -2, 0), seed=3016),
             _v("tunnel", "hex", "lava", "redsmoke", 516)),
    Episode(17, "Moiré Rings · Goa Dawn",
             Spec(bpm=145, tonic=36, style="goa", mode=PHRYGIAN_DOM, roots=(0, 0, 1, 0), seed=3017),
             _v("moire", "rings", "toxic", "nightjungle", 517)),
    Episode(18, "Fivefold Fractal · Palmas",
             Spec(bpm=145, tonic=38, style="flamenco", mode=PHRYGIAN_DOM, roots=(0, -2, -4, -5), chord_bars=1, seed=3018),
             _v("chaos", "star", "saffron", "redsmoke", 518)),
    Episode(19, "Yantra · Last Ritual",
             Spec(bpm=146, tonic=40, style="ritual", mode=PHRYGIAN, roots=(0, 0, 1, 0), seed=3019),
             _v("yantra", "square", "shaman", "uvfloor", 519)),
]  # fmt: skip
