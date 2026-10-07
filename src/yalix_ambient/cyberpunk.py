"""The cyberpunk series: ten neon pieces, 194.159 s each (3:14). No two sound or move alike.

Each episode has its own groove, bass, pad, lead, texture and song form (see music_cyberpunk),
and its own figure, palette and camera: some sway, some orbit, some push in; glitch is
frequent in a few, absent in others. Only two episodes pump under the kick.

    uv run yalix-cyberpunk            # render all ten
    uv run yalix-cyberpunk 3 7        # render episodes 3 and 7
"""

from __future__ import annotations

import sys

from yalix_ambient.gl_particles import Visual
from yalix_ambient.music_cyberpunk import AEOLIAN, DORIAN, HARMONIC_MINOR, LYDIAN, PHRYGIAN, Spec, render_track
from yalix_ambient.series import Episode, build_episode

FOLDER = "cyberpunk"

EPISODES = [
    # Downtempo in the rain: swung half-speed beat, plucked arpeggio, PWM lead, no pumping.
    Episode(1, "Neon Rain",
            Spec(bpm=86, tonic=45, scale=AEOLIAN, progression=(0, 5, 2, 6), sevenths=True, groove="downtempo",
                 swing=0.18, bass="sub", pad="supersaw", lead="square", arp="pluck", arp_rate=8,
                 arp_shape="updown", texture=("rain",), form="anthem", lead_density=5, seed=401),
            Visual(system="sprott", palette="neon", nebula="night", style="cyber", sway=15, orbit=0.5,
                   speed=0.8, glitch=0.25, scanlines=0.6, seed=61)),
    # Drum & bass at 172: reese bass, choir pad, FM bell lead, radio chatter. Long build to one drop.
    Episode(2, "Ghost Protocol",
            Spec(bpm=172, tonic=40, scale=PHRYGIAN, progression=(0, 1, 0, 6), chord_bars=4, groove="dnb",
                 bass="reese", pad="choir", lead="bell", arp="none", texture=("radio",), form="build",
                 lead_density=4, reverb_s=4.0, seed=402),
            Visual(system="burke_shaw", palette="matrix", nebula="terminal", style="cyber", sway=35,
                   speed=1.7, hue_speed=2.0, glitch=1.0, scanlines=1.0, chroma=1.4, seed=62)),
    # The one classic synthwave track: four on the floor, gated snare, pulse bass, gliding saw lead.
    Episode(3, "Chrome Heart",
            Spec(bpm=118, tonic=38, scale=DORIAN, progression=(0, 3, 0, 6), groove="four", pump=0.45,
                 bass="pulse", pad="supersaw", lead="saw", arp="square", arp_shape="updown", form="club",
                 seed=403),
            Visual(system="lu", palette="chrome", nebula="ultraviolet", style="cyber", sway=25, zoom=0.35,
                   glitch=0.5, seed=63)),
    # Electro: syncopated kicks, claps, cowbell squares, acid bass, glass pad. Layers rotate.
    Episode(4, "Data Haven",
            Spec(bpm=112, tonic=42, scale=AEOLIAN, progression=(0, 0, 5, 6), groove="electro", bass="acid",
                 pad="glass", lead="bell", arp="none", texture=("radio",), form="loop", lead_density=7,
                 seed=404),
            Visual(family="magnetic", dipoles=3, palette="synthwave", nebula="neonfog", particles=9000, size=1.8,
                   trail=0.965, style="cyber", speed=1.3, hue_speed=1.6, glitch=0.6, chroma=0.6, seed=64)),
    # Trap: 808 with glide, hat rolls, strings, plucked lead and bell triplets over city noise.
    Episode(5, "Night City",
            Spec(bpm=140, tonic=40, scale=HARMONIC_MINOR, progression=(0, 5, 3, 4), groove="trap",
                 bass="808", pad="strings", lead="pluck", arp="bell", arp_rate=12, arp_shape="random",
                 texture=("city",), form="anthem", lead_density=8, seed=405),
            Visual(system="rucklidge", palette="laser", nebula="night", style="cyber", orbit=1.0, sway=10,
                   glitch=0.4, scanlines=0.5, seed=65)),
    # No drums at all: glass pad, sub drone, sparse bells, vinyl. Slow and cold.
    Episode(6, "Black Ice",
            Spec(bpm=70, tonic=43, scale=LYDIAN, progression=(0, 1, 0, 4), chord_bars=4, sevenths=True,
                 groove="none", bass="sub", pad="glass", lead="bell", arp="pluck", arp_rate=8,
                 arp_shape="random", texture=("vinyl",), form="ambient", lead_density=3, reverb_s=5.5,
                 seed=406),
            Visual(system="fourwing", palette="glacier", nebula="ultraviolet", style="cyber", sway=8,
                   speed=0.55, hue_speed=0.5, glitch=0.0, scanlines=0.3, chroma=0.3, trail=0.975, seed=66)),
    # Broken beat with swing, FM growl, formant choir, acid lead, rain. Layers rotate.
    Episode(7, "Wetware",
            Spec(bpm=96, tonic=41, scale=DORIAN, progression=(0, 3, 2, 4), groove="break", swing=0.22,
                 bass="fm", pad="choir", lead="acid", arp="none", texture=("rain",), form="loop",
                 lead_density=9, seed=407),
            Visual(family="flow", palette="vapor", nebula="neonfog", particles=10000, size=1.8, trail=0.97,
                   style="cyber", speed=1.2, hue_speed=1.4, glitch=0.5, seed=67)),
    # The hardest: 132, four on the floor, acid bass, PWM lead, frequent glitches.
    Episode(8, "Overclock",
            Spec(bpm=132, tonic=44, scale=PHRYGIAN, progression=(0, 1, 6, 1), groove="four", pump=0.3,
                 bass="acid", pad="none", lead="square", arp="square", arp_shape="pendulum", form="build",
                 lead_density=10, seed=408),
            Visual(system="chen", palette="acid", nebula="night", style="cyber", sway=40, speed=1.6, zoom=0.25,
                   hue_speed=2.5, glitch=1.2, chroma=1.5, seed=68)),
    # Lo-fi afterglow: very swung downtempo, strings, gliding saw lead, vinyl and rain.
    Episode(9, "Afterglow",
            Spec(bpm=78, tonic=45, scale=AEOLIAN, progression=(0, 5, 3, 6), sevenths=True, groove="downtempo",
                 swing=0.28, bass="808", pad="strings", lead="saw", arp="none", texture=("vinyl", "rain"),
                 form="ambient", lead_density=5, reverb_s=4.5, seed=409),
            Visual(system="halvorsen", palette="sunset", nebula="ultraviolet", particles=9000, style="cyber",
                   orbit=0.75, sway=5, speed=0.7, glitch=0.0, scanlines=0.4, chroma=0.5, seed=69)),
    # Finale: half-time, reese, supersaw and choir together, formant voice lead, radio. Slow push-in.
    Episode(10, "Last Signal",
            Spec(bpm=90, tonic=47, scale=AEOLIAN, progression=(0, 5, 2, 6), groove="halftime", pump=0.2,
                 bass="reese", pad="supersaw", lead="vox", arp="square", arp_rate=16, arp_shape="pendulum",
                 texture=("radio",), form="build", lead_density=6, reverb_s=4.0, seed=410),
            Visual(system="aizawa", palette="neon", nebula="neonfog", particles=9000, style="cyber", sway=20,
                   orbit=0.25, zoom=0.5, glitch=0.35, seed=70)),
]  # fmt: skip


def main() -> None:
    wanted = {int(a) for a in sys.argv[1:]}
    for ep in EPISODES:
        if not wanted or ep.number in wanted:
            build_episode(ep, FOLDER, render_track)


if __name__ == "__main__":
    main()
