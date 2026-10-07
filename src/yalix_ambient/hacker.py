"""Dark steampunk hackers: ten pieces, 194.159 s each (3:14). Inspired by an aesthetic, not a show.

Phosphor screens, brass and copper, server rooms and clockwork. Every episode has its own
groove (minimal, IDM, heartbeat, broken beat, downtempo, 8-bit, none, electro, half-time,
drum & bass), its own instruments, texture and form, and its own figure: code rain, a packet
network and a clockwork train are new families; the rest are attractors and fields. Keyboard
typing appears in a few episodes only (modern in 1, 5, 7; typewriter in 3, 9).

    uv run yalix-hacker            # render all ten
    uv run yalix-hacker 3 7        # render episodes 3 and 7
"""

from __future__ import annotations

import sys

from yalix_ambient.gl_particles import Visual
from yalix_ambient.music_cyberpunk import AEOLIAN, DORIAN, HARMONIC_MINOR, LYDIAN, PHRYGIAN, Spec, render_track
from yalix_ambient.series import Episode, build_episode

FOLDER = "hacker"

EPISODES = [
    # Minimal techno at 3 a.m.: kick on 1 and 3, rimshots, a Berlin-school sequence, typing.
    Episode(1, "Zero Day",
            Spec(bpm=124, tonic=40, scale=PHRYGIAN, progression=(0, 0, 1, 0), chord_bars=4, groove="minimal",
                 bass="seq", pad="drone", lead="sine", arp="none", texture=("typing", "servers"), form="slowburn",
                 lead_density=4, stutter=0.2, seed=501),
            Visual(family="rain", palette="greenterm", nebula="blackout", particles=6400, size=3.4, trail=0.93,
                   intensity=2.2,
                   style="cyber", speed=1.0, glitch=0.4, scanlines=1.0, chroma=0.5, seed=81)),
    # IDM: shifting euclidean kicks, cut-up hats, FM bass, glass pad, chip arpeggio, a modem calling.
    Episode(2, "Root Access",
            Spec(bpm=100, tonic=43, scale=DORIAN, progression=(0, 3, 1, 4), groove="idm", bass="fm", pad="glass",
                 lead="pluck", arp="chip", arp_rate=16, arp_shape="random", texture=("modem",), form="loop",
                 lead_density=7, stutter=0.8, seed=502),
            Visual(family="graph", palette="bluescreen", nebula="blackout", particles=8000, size=1.8, trail=0.97,
                   style="cyber", speed=1.2, glitch=0.9, chroma=1.2, seed=82)),
    # A slow heartbeat in a clock tower: choir, bells, ticking escapement, a typewriter somewhere.
    Episode(3, "Clockwork Daemon",
            Spec(bpm=64, tonic=38, scale=HARMONIC_MINOR, progression=(0, 5, 3, 4), chord_bars=4, groove="heartbeat",
                 bass="sub", pad="choir", lead="bell", arp="pluck", arp_rate=8, arp_shape="pendulum",
                 texture=("clockwork", "typewriter"), form="ambient", lead_density=4, reverb_s=5.0, seed=503),
            Visual(family="clockwork", palette="brass", nebula="soot", particles=9000, size=1.9, trail=0.93,
                   style="cyber", glitch=0.0, scanlines=0.3, chroma=0.3, seed=83)),
    # Broken beat at 130 with buffer stutters, acid bass, no pad, in a humming server room.
    Episode(4, "Packet Storm",
            Spec(bpm=130, tonic=45, scale=AEOLIAN, progression=(0, 6, 5, 6), groove="break", swing=0.1,
                 bass="acid", pad="none", lead="square", arp="square", arp_shape="updown", texture=("servers",),
                 form="club", lead_density=8, stutter=0.6, seed=504),
            Visual(family="magnetic", dipoles=2, palette="redalert", nebula="redroom", particles=9000, size=1.8,
                   trail=0.965, style="cyber", speed=1.5, hue_speed=2.0, glitch=1.1, chroma=1.4, seed=84)),
    # Downtempo trap: swung 808, strings, gliding saw lead, bell triplets, vinyl and a keyboard.
    Episode(5, "Honeypot",
            Spec(bpm=90, tonic=41, scale=AEOLIAN, progression=(0, 5, 2, 6), sevenths=True, groove="downtempo",
                 swing=0.2, bass="808", pad="strings", lead="saw", arp="bell", arp_rate=12, arp_shape="up",
                 texture=("vinyl", "typing"), form="anthem", lead_density=5, seed=505),
            Visual(system="rossler", palette="phosphor", nebula="furnace", style="cyber", sway=15, orbit=0.5,
                   speed=0.8, glitch=0.2, scanlines=0.8, seed=85)),
    # 8-bit at 150: square kick, noise snare, chip lead and arpeggio, steam valves and clockwork.
    Episode(6, "Pressure Valve",
            Spec(bpm=150, tonic=44, scale=DORIAN, progression=(0, 4, 3, 4), groove="chip", bass="pulse", pad="none",
                 lead="chip", arp="chip", arp_shape="updown", texture=("steam", "clockwork"), form="pulse",
                 lead_density=9, stutter=0.4, seed=506),
            Visual(system="thomas", palette="copper", nebula="furnace", style="cyber", sway=30, speed=1.4,
                   hue_speed=1.5, glitch=0.6, scanlines=1.0, seed=86)),
    # No drums: drone, slow sequence, formant voice, rain on the window, someone typing.
    Episode(7, "Dead Drop",
            Spec(bpm=72, tonic=40, scale=AEOLIAN, progression=(0, 1, 0, 6), chord_bars=4, groove="none",
                 bass="seq", pad="drone", lead="vox", arp="none", texture=("rain", "typing"), form="slowburn",
                 lead_density=4, reverb_s=5.5, seed=507),
            Visual(family="flow", palette="monochrome", nebula="blackout", particles=10000, size=1.7, trail=0.975,
                   style="cyber", speed=0.6, hue_speed=0.6, glitch=0.15, scanlines=0.6, chroma=0.4, seed=87)),
    # Electro with a reese, supersaw and acid lead, modem and server hum, a few stutters.
    Episode(8, "Backdoor",
            Spec(bpm=116, tonic=42, scale=PHRYGIAN, progression=(0, 1, 6, 5), groove="electro", bass="reese",
                 pad="supersaw", lead="acid", arp="none", texture=("modem", "servers"), form="pulse",
                 lead_density=8, stutter=0.3, seed=508),
            Visual(system="arneodo", palette="verdigris", nebula="patina", style="cyber", sway=20, zoom=0.4,
                   speed=1.1, glitch=0.5, seed=88)),
    # Half-time in a steam room: rolling bass, drone, plucked lead, bells, a typewriter.
    Episode(9, "Brass Cipher",
            Spec(bpm=84, tonic=45, scale=LYDIAN, progression=(0, 1, 4, 3), groove="halftime", pump=0.2,
                 bass="rolling", pad="drone", lead="pluck", arp="bell", arp_rate=8, arp_shape="random",
                 texture=("typewriter", "steam"), form="build", lead_density=6, reverb_s=4.0, seed=509),
            Visual(system="dadras", palette="ember", nebula="soot", particles=9000, style="cyber", orbit=0.75,
                   sway=8, speed=0.9, glitch=0.25, scanlines=0.5, seed=89)),
    # Finale: drum & bass at 170, reese, choir, sine lead, radio voices, long build.
    Episode(10, "Exit Node",
            Spec(bpm=170, tonic=38, scale=AEOLIAN, progression=(0, 5, 6, 4), chord_bars=4, groove="dnb",
                 bass="reese", pad="choir", lead="sine", arp="square", arp_shape="pendulum", texture=("radio",),
                 form="build", lead_density=5, stutter=0.5, reverb_s=4.0, seed=510),
            Visual(system="fourwing", palette="ash", nebula="blackout", particles=9000, style="cyber", sway=35,
                   orbit=0.25, zoom=0.3, speed=1.5, glitch=0.8, chroma=1.2, seed=90)),
]  # fmt: skip


def main() -> None:
    wanted = {int(a) for a in sys.argv[1:]}
    for ep in EPISODES:
        if not wanted or ep.number in wanted:
            build_episode(ep, FOLDER, render_track)


if __name__ == "__main__":
    main()
