"""The cyberpunk series: ten neon pieces, synthwave and darksynth, 194.159 s each (3:14).

No figure repeats inside the series: ten different systems (four of them new to the channel:
Sprott, Burke-Shaw, Lü and Rucklidge), each with its own palette, nebula and track. The cyber
style adds scanlines, a chromatic split that opens on the kick and short glitch bursts.

    uv run yalix-cyberpunk            # render all ten
    uv run yalix-cyberpunk 3 7        # render episodes 3 and 7
"""

from __future__ import annotations

import sys

from yalix_ambient.gl_particles import Visual
from yalix_ambient.music_cyberpunk import AEOLIAN, DORIAN, HARMONIC_MINOR, PHRYGIAN, Spec, render_track
from yalix_ambient.series import Episode, build_episode

FOLDER = "cyberpunk"

EPISODES = [
    # Rain on neon: four-on-the-floor synthwave in A minor, rolling bass.
    Episode(1, "Neon Rain",
            Spec(bpm=104, tonic=45, scale=AEOLIAN, progression=(0, 5, 2, 6), drums="four", bass="rolling",
                 layers=("pad", "arp", "lead", "bass", "rain"), seed=401),
            Visual(system="sprott", palette="neon", nebula="night", style="cyber", seed=61)),
    # Darksynth crawl: half-time, reese bass, Phrygian minor second.
    Episode(2, "Ghost Protocol",
            Spec(bpm=88, tonic=40, scale=PHRYGIAN, progression=(0, 1, 0, 6), drums="halftime", bass="reese",
                 arp_rate=8, arp_shape="pendulum", cutoff=(700.0, 3000.0), seed=402),
            Visual(system="burke_shaw", palette="matrix", nebula="terminal", style="cyber", seed=62)),
    # Bright and driving: D Dorian, eighth-note pulse bass.
    Episode(3, "Chrome Heart",
            Spec(bpm=118, tonic=38, scale=DORIAN, progression=(0, 3, 0, 6), drums="four", bass="pulse",
                 arp_shape="updown", seed=403),
            Visual(system="lu", palette="chrome", nebula="ultraviolet", style="cyber", seed=63)),
    # Field lines over a server farm: F# minor, rolling bass.
    Episode(4, "Data Haven",
            Spec(bpm=110, tonic=42, scale=AEOLIAN, progression=(0, 2, 6, 5), drums="four", bass="rolling",
                 arp_shape="pendulum", seed=404),
            Visual(family="magnetic", dipoles=3, palette="synthwave", nebula="neonfog", particles=9000, size=1.8,
                   trail=0.965, style="cyber", seed=64)),
    # The fastest: harmonic minor with a major V, rolling bass, wide filter sweep.
    Episode(5, "Night City",
            Spec(bpm=124, tonic=40, scale=HARMONIC_MINOR, progression=(0, 5, 3, 4), drums="four", bass="rolling",
                 cutoff=(1100.0, 5200.0), seed=405),
            Visual(system="rucklidge", palette="laser", nebula="night", style="cyber", seed=65)),
    # Cold and slow: half-time, reese, i–VII–VI–v.
    Episode(6, "Black Ice",
            Spec(bpm=92, tonic=43, scale=AEOLIAN, progression=(0, 6, 5, 4), drums="halftime", bass="reese",
                 arp_rate=8, arp_shape="updown", cutoff=(600.0, 2600.0), reverb_s=4.0, seed=406),
            Visual(system="fourwing", palette="glacier", nebula="ultraviolet", style="cyber", seed=66)),
    # Broken beat, FM growl, smoke-like flow of particles.
    Episode(7, "Wetware",
            Spec(bpm=98, tonic=41, scale=DORIAN, progression=(0, 3, 2, 4), drums="break", bass="fm",
                 arp_shape="pendulum", layers=("pad", "arp", "lead", "bass", "rain"), seed=407),
            Visual(family="flow", palette="vapor", nebula="neonfog", particles=10000, size=1.8, trail=0.965,
                   style="cyber", seed=67)),
    # Overdriven: 128 BPM, FM bass, Phrygian.
    Episode(8, "Overclock",
            Spec(bpm=128, tonic=44, scale=PHRYGIAN, progression=(0, 1, 6, 1), drums="four", bass="fm",
                 cutoff=(1200.0, 5600.0), seed=408),
            Visual(system="chen", palette="acid", nebula="night", style="cyber", seed=68)),
    # After the rain: no drums, reese drone, pad and lead.
    Episode(9, "Afterglow",
            Spec(bpm=84, tonic=45, scale=AEOLIAN, progression=(0, 5, 3, 6), drums="none", bass="reese",
                 arp_rate=8, arp_shape="updown", layers=("pad", "arp", "lead", "bass", "rain"), reverb_s=4.5,
                 seed=409),
            Visual(system="halvorsen", palette="sunset", nebula="ultraviolet", particles=9000, style="cyber",
                   seed=69)),
    # Finale: everything on, broken beat into four-on-the-floor energy, B minor.
    Episode(10, "Last Signal",
            Spec(bpm=116, tonic=47, scale=AEOLIAN, progression=(0, 5, 2, 6), drums="four", bass="rolling",
                 arp_shape="updown", cutoff=(1000.0, 5000.0), layers=("pad", "arp", "lead", "bass"), seed=410),
            Visual(system="aizawa", palette="neon", nebula="neonfog", particles=9000, style="cyber", seed=70)),
]  # fmt: skip


def main() -> None:
    wanted = {int(a) for a in sys.argv[1:]}
    for ep in EPISODES:
        if not wanted or ep.number in wanted:
            build_episode(ep, FOLDER, render_track)


if __name__ == "__main__":
    main()
