"""Deep Space: nineteen drumless themes for the one-hour mix, 194.159 s each (3:14).

No drums anywhere: pads, bells, choirs, drones and slow arpeggios, alternating the two synth
engines (music_cyberpunk with groove 'none', music_v2 with drums 'none') so neighbouring
themes never share a sound. The figures are mostly the cosmos families (n-body orbits, spiral
galaxies, harmonographs, Clifford and De Jong maps, spirographs) plus three attractors seen
from a slow orbiting camera.

    uv run yalix-mix deepspace
"""

from __future__ import annotations

from yalix_ambient import music_cyberpunk as cp
from yalix_ambient.gl_particles import Visual
from yalix_ambient.music_v2 import Spec
from yalix_ambient.series import AEOLIAN, HARMONIC_MINOR, PHRYGIAN, Episode

DORIAN = (0, 2, 3, 5, 7, 9, 10)


def _v(family: str, system: str = "lorenz", **kw) -> Visual:
    base = dict(particles=9000, size=1.8, trail=0.965, intensity=0.9, bloom=0.45)
    if family == "attractor":
        base = {}
    return Visual(family=family, system=system, **{**base, **kw})


EPISODES = [
    Episode(1, "Galaxy · Andromeda",
            cp.Spec(bpm=60, tonic=41, scale=cp.LYDIAN, progression=(0, 1, 4, 1), chord_bars=4, groove="none",
                    bass="sub", pad="strings", lead="bell", arp="bell", arp_rate=8, arp_shape="random",
                    form="ambient", lead_density=3, reverb_s=6.0, seed=801),
            _v("galaxy", "spiral2", palette="ice", nebula="abyss", seed=101)),
    Episode(2, "Three Bodies · Figure Eight",
            Spec(bpm=56, meters=(8,), roots=(45, 41, 43, 40), scale=AEOLIAN, drums="none", bass_timbre="sub",
                 bass_pattern=(1, 0, 0, 0, 0, 0, 0, 0), bass_notes=(0, 7), bass_drive=1.0,
                 layers=("strings", "bells", "drone", "bass"), reverb_s=6.0, seed=802),
            _v("orbits", "figure8", palette="glacier", nebula="void", seed=102)),
    Episode(3, "Harmonograph · Pulsar",
            cp.Spec(bpm=76, tonic=43, scale=cp.DORIAN, progression=(0, 3, 1, 4), groove="none", bass="seq",
                    pad="glass", lead="sine", arp="pluck", arp_rate=16, arp_shape="updown", texture=("radio",),
                    form="slowburn", lead_density=4, reverb_s=5.0, seed=803),
            _v("harmonograph", palette="aurora", nebula="night", seed=103)),
    Episode(4, "Binary Star",
            Spec(bpm=62, meters=(6,), roots=(38, 41, 36, 40), scale=HARMONIC_MINOR, drums="none", bass_timbre="fm",
                 bass_pattern=(1, 0, 0, 0, 0, 0), bass_notes=(0, 7), bass_drive=1.2,
                 layers=("choir", "bells", "drone", "bass"), reverb_s=6.0, seed=804),
            _v("orbits", "binary", palette="fire", nebula="dusk", seed=104)),
    Episode(5, "De Jong · Cosmic Web",
            cp.Spec(bpm=70, tonic=44, scale=cp.PHRYGIAN, progression=(0, 1, 0, 6), chord_bars=4, groove="none",
                    bass="sub", pad="drone", lead="vox", arp="chip", arp_rate=16, arp_shape="random",
                    texture=("modem",), form="loop", lead_density=4, reverb_s=5.5, seed=805),
            _v("map", "dejong", palette="vapor", nebula="ultraviolet", intensity=0.7, seed=105)),
    Episode(6, "Aizawa · Event Horizon",
            Spec(bpm=50, meters=(8,), roots=(36, 37, 36, 34), scale=PHRYGIAN, drums="none", bass_timbre="sub",
                 bass_pattern=(1, 0, 0, 0, 0, 0, 0, 0), bass_notes=(0,), bass_drive=1.5,
                 layers=("drone", "choir", "noise", "bass"), reverb_s=7.0, seed=806),
            _v("attractor", "aizawa", palette="ember", nebula="blackout", orbit=0.75, zoom=0.5, sway=0, speed=0.7,
               seed=106)),
    Episode(7, "Galaxy · Three Arms",
            cp.Spec(bpm=66, tonic=40, scale=cp.LYDIAN, progression=(0, 4, 1, 4), groove="none", bass="sub",
                    pad="supersaw", lead="bell", arp="pluck", arp_rate=12, arp_shape="pendulum", form="build",
                    lead_density=4, reverb_s=5.5, seed=807),
            _v("galaxy", "spiral3", palette="sunset", nebula="night", seed=107)),
    Episode(8, "Spirograph · Kepler",
            cp.Spec(bpm=80, tonic=45, scale=cp.LYDIAN, progression=(0, 1, 4, 3), groove="none", bass="seq",
                    pad="glass", lead="pluck", arp="bell", arp_rate=16, arp_shape="up", form="loop",
                    lead_density=6, reverb_s=4.5, seed=808),
            _v("spirograph", palette="glacier", nebula="slate", seed=108)),
    Episode(9, "Flow · Solar Wind",
            Spec(bpm=58, meters=(10,), roots=(40, 43, 38, 41), scale=AEOLIAN, drums="none", bass_timbre="sub",
                 bass_pattern=(1, 0, 0, 0, 0, 0, 0, 0, 0, 0), bass_notes=(0, 5), bass_drive=1.0, arp_hits=3,
                 layers=("strings", "drone", "noise", "arp"), reverb_s=6.0, seed=809),
            _v("flow", palette="aurora", nebula="void", particles=10000, size=1.7, trail=0.975, intensity=0.6,
               seed=109)),
    Episode(10, "Three Bodies · Hierarchy",
            cp.Spec(bpm=64, tonic=42, scale=cp.DORIAN, progression=(0, 3, 4, 3), chord_bars=4, groove="none",
                    bass="sub", pad="choir", lead="sine", arp="none", texture=("radio",), form="slowburn",
                    lead_density=3, reverb_s=6.0, seed=810),
            _v("orbits", "hierarchical", palette="ice", nebula="abyss", seed=110)),
    Episode(11, "Galaxy · Barred Spiral",
            Spec(bpm=60, meters=(8, 6), roots=(43, 40, 45, 41), scale=DORIAN, drums="none", bass_timbre="sub",
                 bass_pattern=(1, 0, 0, 0, 0, 0, 0, 0), bass_notes=(0, 7), bass_drive=1.0,
                 layers=("choir", "bells", "strings", "bass"), reverb_s=6.5, seed=811),
            _v("galaxy", "barred", palette="vapor", nebula="neonfog", seed=111)),
    Episode(12, "Clifford · Stardust",
            cp.Spec(bpm=72, tonic=40, scale=cp.AEOLIAN, progression=(0, 5, 3, 6), sevenths=True, groove="none",
                    bass="sub", pad="strings", lead="bell", arp="bell", arp_rate=8, arp_shape="random",
                    form="ambient", lead_density=3, reverb_s=6.0, seed=812),
            _v("map", "clifford", palette="glacier", nebula="slate", intensity=0.7, seed=112)),
    Episode(13, "Halvorsen · Drift",
            Spec(bpm=54, meters=(8,), roots=(41, 37, 39, 36), scale=HARMONIC_MINOR, drums="none", bass_timbre="sub",
                 bass_pattern=(1, 0, 0, 0, 1, 0, 0, 0), bass_notes=(0, 7), bass_drive=1.0,
                 layers=("drone", "choir", "bells", "bass"), reverb_s=6.5, seed=813),
            _v("attractor", "halvorsen", palette="monochrome", nebula="smoke", orbit=0.5, sway=10, speed=0.6,
               seed=113)),
    Episode(14, "Four Bodies · Quadrant",
            cp.Spec(bpm=84, tonic=43, scale=cp.DORIAN, progression=(0, 4, 3, 4), groove="none", bass="seq",
                    pad="glass", lead="pluck", arp="square", arp_rate=16, arp_shape="pendulum", form="pulse",
                    lead_density=6, reverb_s=4.5, seed=814),
            _v("orbits", "quad", palette="aurora", nebula="ultraviolet", seed=114)),
    Episode(15, "Harmonograph · Lissajous Sky",
            Spec(bpm=66, meters=(6,), roots=(45, 43, 40, 41), scale=AEOLIAN, drums="none", bass_timbre="sub",
                 bass_pattern=(1, 0, 0, 0, 0, 0), bass_notes=(0, 7), bass_drive=1.0, arp_hits=4,
                 layers=("strings", "arp", "bells", "bass"), reverb_s=5.5, seed=815),
            _v("harmonograph", palette="ember", nebula="dusk", seed=115)),
    Episode(16, "Thomas · Nebula",
            cp.Spec(bpm=58, tonic=38, scale=cp.PHRYGIAN, progression=(0, 1, 6, 1), chord_bars=4, groove="none",
                    bass="sub", pad="choir", lead="vox", arp="none", texture=("radio",), form="ambient",
                    lead_density=3, reverb_s=6.5, seed=816),
            _v("attractor", "thomas", palette="bruise", nebula="dusk", orbit=0.25, zoom=0.4, sway=15, speed=0.7,
               seed=116)),
    Episode(17, "Spirograph · Hypotrochoid",
            cp.Spec(bpm=90, tonic=45, scale=cp.LYDIAN, progression=(0, 1, 4, 1), groove="none", bass="pulse",
                    pad="glass", lead="bell", arp="chip", arp_rate=16, arp_shape="updown", form="loop",
                    lead_density=6, reverb_s=4.5, seed=817),
            _v("spirograph", palette="ice", nebula="abyss", seed=117)),
    Episode(18, "Quadrupole · Magnetar",
            Spec(bpm=60, meters=(8,), roots=(38, 39, 38, 36), scale=PHRYGIAN, drums="none", bass_timbre="sub",
                 bass_pattern=(1, 0, 0, 0, 0, 0, 0, 0), bass_notes=(0, 1), bass_drive=1.2,
                 layers=("drone", "noise", "choir", "bass"), reverb_s=6.5, seed=818),
            _v("magnetic", dipoles=4, palette="laser", nebula="night", intensity=0.6, seed=118)),
    Episode(19, "Galaxy · Heat Death",
            cp.Spec(bpm=52, tonic=40, scale=cp.AEOLIAN, progression=(0, 5, 3, 4), chord_bars=4, groove="none",
                    bass="sub", pad="strings", lead="sine", arp="pluck", arp_rate=8, arp_shape="pendulum",
                    form="slowburn", lead_density=3, reverb_s=7.0, seed=819),
            _v("galaxy", "spiral2", palette="ash", nebula="blackout", seed=119)),
]  # fmt: skip
