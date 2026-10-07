"""Long mixes: themes of 3:14 joined into one video of about an hour, figures morphing between them.

Audio: every theme is rendered on its own, levelled to the same loudness and crossfaded into
the next with an equal-power curve over OVERLAP seconds. Video: one continuous render; during
each crossfade both scenes run and every particle travels from its place in the old figure to
a place in the new one (particles are paired by angle around the centre so the flow is
orderly), while colours, nebula, trail and bloom blend. Chapters for YouTube are written next
to the video.

    uv run yalix-mix dark          # output/mixes/dark-mix.mp4 + dark-mix.chapters.txt
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from yalix_ambient.gl_particles import Renderer, Scene, Visual, glitch_schedule
from yalix_ambient import music_cyberpunk as cp
from yalix_ambient.cyberpunk import EPISODES as CYBER
from yalix_ambient.hacker import EPISODES as HACKER
from yalix_ambient.grunge import EPISODES as GRUNGE
from yalix_ambient import music_grunge as gr
from yalix_ambient.music import SR
from yalix_ambient.music_v2 import Spec, render_track
from yalix_ambient.pipeline import OUTPUT, run
from yalix_ambient.series import AEOLIAN, DURATION, HARMONIC_MINOR, Episode
from yalix_ambient.series import EPISODES as DARK

OVERLAP = 6.0  # seconds of crossfade and morph between themes
N_POINTS = 9000
FPS = 30
SCALE = 4 / 3  # render at 2560 x 1440: YouTube serves 1440p uploads with a better codec, even at 1080p

# Nine more dark themes, so an hour never repeats a figure.
DARK_EXTRA = [
    Episode(11, "Lü · Iron Lung",
            Spec(bpm=70, meters=(8,), roots=(38, 39, 38, 36), drums="halftime", bass_timbre="fuzz", bass_hits=4,
                 bass_notes=(0, 1, 0, 7), layers=("bass", "chug", "choir", "noise"), seed=111),
            Visual(system="lu", palette="rust", nebula="inferno", sway=20, seed=21)),
    Episode(12, "Sprott · Lantern Procession",
            Spec(bpm=62, meters=(6,), roots=(45, 41, 43, 40), scale=HARMONIC_MINOR, drums="sparse", bass_timbre="sub",
                 bass_pattern=(1, 0, 0, 1, 0, 0), bass_notes=(0, 7), bass_drive=1.2,
                 layers=("strings", "bells", "choir", "bass"), reverb_s=5.5, seed=112),
            Visual(system="sprott", palette="rain", nebula="abyss", orbit=0.5, sway=10, seed=22)),
    Episode(13, "Burke-Shaw · Foundry",
            Spec(bpm=86, meters=(7,), roots=(40, 40, 43, 38), drums="march", bass_timbre="square",
                 bass_pattern=(1, 1, 0, 1, 0, 1, 0), bass_notes=(0, 0, 0, 12, 0, 1), bass_cutoff=(450, 1500),
                 layers=("chug", "bass", "wall"), seed=113),
            Visual(system="burke_shaw", palette="copper", nebula="furnace", speed=1.2, seed=23)),
    Episode(14, "Rucklidge · Black Mass",
            Spec(bpm=66, meters=(10,), roots=(41, 37, 39, 36), scale=HARMONIC_MINOR, drums="tribal", bass_timbre="fm",
                 bass_hits=4, bass_notes=(0, 6, 0, 1), fibonacci=True,
                 layers=("choir", "timpani", "drone", "bass"), seed=114),
            Visual(system="rucklidge", palette="blood", nebula="redroom", sway=30, seed=24)),
    Episode(15, "Chen · Glass Cathedral",
            Spec(bpm=58, meters=(8,), roots=(45, 43, 40, 41), scale=AEOLIAN, drums="none", bass_timbre="sub",
                 bass_pattern=(1, 0, 0, 0, 0, 0, 0, 0), bass_notes=(0,), bass_drive=1.0,
                 layers=("strings", "choir", "bells", "drone"), reverb_s=6.5, seed=115),
            Visual(system="chen", palette="glacier", nebula="abyss", speed=0.6, sway=10, seed=25)),
    Episode(16, "Four-wing · Moth Swarm",
            Spec(bpm=80, meters=(9,), roots=(38, 38, 44, 41), drums="tribal", bass_timbre="saw", bass_hits=5,
                 bass_notes=(0, 0, 12, 6, 0), fibonacci=True, layers=("bass", "arp", "noise", "drone"), seed=116),
            Visual(system="fourwing", palette="moss", nebula="swamp", speed=1.1, seed=26)),
    Episode(17, "Clockwork · Bell Tower",
            Spec(bpm=66, meters=(6,), roots=(40, 45, 43, 38), drums="sparse", bass_timbre="sub",
                 bass_pattern=(1, 0, 0, 0, 0, 0), bass_notes=(0, 7), bass_drive=1.2,
                 layers=("bells", "strings", "timpani", "bass"), reverb_s=5.0, seed=117),
            Visual(family="clockwork", palette="brass", nebula="soot", size=1.9, trail=0.93, seed=27)),
    Episode(18, "Tripole · Eclipse",
            Spec(bpm=72, meters=(8,), roots=(42, 43, 42, 37), drums="halftime", bass_timbre="fuzz", bass_hits=3,
                 bass_notes=(0, 7, 10), layers=("bass", "wall", "choir"), seed=118),
            Visual(family="magnetic", dipoles=3, palette="bruise", nebula="dusk", size=1.8, trail=0.965, seed=28)),
    Episode(19, "Network · Last Rite",
            Spec(bpm=60, meters=(8, 6), roots=(40, 36, 38, 35), drums="sparse", bass_timbre="sub",
                 bass_pattern=(1, 0, 0, 0, 1, 0, 0, 0), bass_notes=(0, 7), bass_drive=1.0,
                 layers=("choir", "strings", "timpani", "drone"), reverb_s=6.0, seed=119),
            Visual(family="graph", palette="ash", nebula="smoke", size=1.7, trail=0.97, seed=29)),
]  # fmt: skip

_BY_NUMBER = {e.number: e for e in DARK + DARK_EXTRA}
# Calm opening, rising through the middle, glassy and slow towards the end.
DARK_ORDER = [12, 1, 6, 11, 2, 13, 3, 17, 4, 14, 5, 16, 7, 9, 15, 18, 8, 10, 19]

# Nine more cyberpunk themes: figures the series had not used yet (code rain and a network in
# neon, Lorenz, Thomas, Rössler, Dadras, Arneodo, two- and five-pole fields) and grooves it
# had not used (minimal, IDM, 8-bit, heartbeat).
CYBER_EXTRA = [
    Episode(11, "Neon Grid",
            cp.Spec(bpm=122, tonic=41, scale=cp.PHRYGIAN, progression=(0, 0, 1, 0), chord_bars=4, groove="minimal",
                    bass="seq", pad="glass", lead="sine", arp="none", texture=("city",), form="slowburn",
                    lead_density=4, seed=411),
            Visual(family="rain", palette="neon", nebula="night", particles=6400, size=4.2, trail=0.93,
                   intensity=3.6, style="cyber", glitch=0.4, chroma=0.6, seed=71)),
    Episode(12, "Signal Ghost",
            cp.Spec(bpm=104, tonic=44, scale=cp.DORIAN, progression=(0, 3, 1, 4), groove="idm", bass="fm",
                    pad="choir", lead="pluck", arp="chip", arp_shape="random", texture=("radio",), form="loop",
                    lead_density=7, stutter=0.5, seed=412),
            Visual(family="graph", palette="chrome", nebula="ultraviolet", size=1.8, trail=0.97, style="cyber",
                   speed=1.2, glitch=0.8, chroma=1.2, seed=72)),
    Episode(13, "Synth Lagoon",
            cp.Spec(bpm=80, tonic=43, scale=cp.AEOLIAN, progression=(0, 5, 3, 6), sevenths=True, groove="downtempo",
                    swing=0.25, bass="808", pad="glass", lead="bell", arp="pluck", arp_rate=8, arp_shape="updown",
                    texture=("rain",), form="ambient", lead_density=4, reverb_s=4.5, seed=413),
            Visual(system="lorenz", palette="vapor", nebula="neonfog", style="cyber", orbit=0.5, sway=10, speed=0.7,
                   glitch=0.0, scanlines=0.5, chroma=0.4, seed=73)),
    Episode(14, "Laser Cathedral",
            cp.Spec(bpm=90, tonic=38, scale=cp.HARMONIC_MINOR, progression=(0, 5, 3, 4), groove="halftime", pump=0.2,
                    bass="reese", pad="choir", lead="saw", arp="square", arp_shape="pendulum", form="anthem",
                    lead_density=5, reverb_s=4.0, seed=414),
            Visual(system="thomas", palette="laser", nebula="night", style="cyber", sway=30, zoom=0.3, glitch=0.4,
                   seed=74)),
    Episode(15, "Arcade Heart",
            cp.Spec(bpm=140, tonic=45, scale=cp.DORIAN, progression=(0, 3, 4, 3), groove="chip", bass="pulse",
                    pad="none", lead="chip", arp="chip", arp_shape="updown", form="pulse", lead_density=9,
                    stutter=0.3, seed=415),
            Visual(system="rossler", palette="acid", nebula="ultraviolet", style="cyber", sway=20, speed=1.5,
                   hue_speed=2.0, glitch=0.9, seed=75)),
    Episode(16, "Midnight Drive",
            cp.Spec(bpm=108, tonic=40, scale=cp.AEOLIAN, progression=(0, 5, 2, 6), groove="four", pump=0.4,
                    bass="rolling", pad="supersaw", lead="square", arp="square", arp_shape="up", form="club",
                    seed=416),
            Visual(system="dadras", palette="synthwave", nebula="neonfog", style="cyber", orbit=0.5, sway=15,
                   glitch=0.3, seed=76)),
    Episode(17, "Hologram",
            cp.Spec(bpm=72, tonic=42, scale=cp.LYDIAN, progression=(0, 1, 4, 1), chord_bars=4, groove="none",
                    bass="sub", pad="strings", lead="vox", arp="bell", arp_rate=8, arp_shape="random",
                    texture=("vinyl",), form="ambient", lead_density=3, reverb_s=5.5, seed=417),
            Visual(system="arneodo", palette="glacier", nebula="ultraviolet", style="cyber", sway=10, speed=0.6,
                   hue_speed=0.6, glitch=0.0, scanlines=0.4, chroma=0.3, trail=0.97, seed=77)),
    Episode(18, "Dipole Drift",
            cp.Spec(bpm=118, tonic=43, scale=cp.PHRYGIAN, progression=(0, 1, 6, 1), groove="electro", bass="acid",
                    pad="glass", lead="square", arp="none", texture=("radio",), form="pulse", lead_density=7,
                    stutter=0.2, seed=418),
            Visual(family="magnetic", dipoles=2, palette="matrix", nebula="terminal", size=1.8, trail=0.965,
                   style="cyber", speed=1.3, glitch=0.6, seed=78)),
    Episode(19, "Neon Requiem",
            cp.Spec(bpm=60, tonic=45, scale=cp.AEOLIAN, progression=(0, 5, 3, 4), chord_bars=4, groove="heartbeat",
                    bass="sub", pad="choir", lead="sine", arp="pluck", arp_rate=8, arp_shape="pendulum",
                    texture=("rain",), form="ambient", lead_density=3, reverb_s=6.0, seed=419),
            Visual(family="magnetic", dipoles=5, palette="sunset", nebula="ultraviolet", size=1.8, trail=0.97,
                   style="cyber", speed=0.7, glitch=0.0, scanlines=0.5, seed=79)),
]  # fmt: skip

_CYBER = {e.number: e for e in CYBER + CYBER_EXTRA}
# Calm start, energy through the middle, no two neighbouring themes with the same groove.
CYBER_ORDER = [1, 11, 3, 17, 4, 14, 7, 15, 5, 12, 2, 16, 18, 8, 13, 6, 10, 9, 19]

# Nine more dark steampunk hacker themes: the attractors the series had not used (Sprott, Lü,
# Burke-Shaw, Rucklidge, Chen, Aizawa, Halvorsen, Lorenz) plus a three-pole field, and two
# grooves it lacked (four-on-the-floor, trap). Typing in 13 and 17, a typewriter in 15.
HACKER_EXTRA = [
    Episode(11, "Cold Boot",
            cp.Spec(bpm=68, tonic=40, scale=cp.AEOLIAN, progression=(0, 5, 0, 3), chord_bars=4, groove="none",
                    bass="sub", pad="drone", lead="bell", arp="pluck", arp_rate=8, arp_shape="pendulum",
                    texture=("servers",), form="ambient", lead_density=3, reverb_s=5.5, seed=611),
            Visual(system="sprott", palette="greenterm", nebula="terminal", style="cyber", orbit=0.5, sway=10,
                   speed=0.6, hue_speed=0.6, glitch=0.1, scanlines=0.9, chroma=0.4, seed=91)),
    Episode(12, "Kernel Panic",
            cp.Spec(bpm=128, tonic=42, scale=cp.PHRYGIAN, progression=(0, 1, 0, 6), groove="break", swing=0.1,
                    bass="reese", pad="none", lead="acid", arp="square", arp_shape="random", texture=("modem",),
                    form="club", lead_density=8, stutter=0.7, seed=612),
            Visual(system="lu", palette="redalert", nebula="redroom", style="cyber", sway=25, speed=1.4,
                   hue_speed=1.8, glitch=1.0, chroma=1.3, seed=92)),
    Episode(13, "Steam Shell",
            cp.Spec(bpm=120, tonic=45, scale=cp.DORIAN, progression=(0, 3, 4, 3), groove="four", pump=0.35,
                    bass="rolling", pad="strings", lead="square", arp="square", arp_shape="updown",
                    texture=("steam", "typing"), form="build", lead_density=6, seed=613),
            Visual(system="burke_shaw", palette="copper", nebula="furnace", style="cyber", orbit=0.5, sway=15,
                   speed=1.1, glitch=0.3, scanlines=0.7, seed=93)),
    Episode(14, "Phreaker",
            cp.Spec(bpm=112, tonic=43, scale=cp.DORIAN, progression=(0, 4, 3, 1), groove="electro", bass="fm",
                    pad="glass", lead="chip", arp="chip", arp_rate=16, arp_shape="up", texture=("radio",),
                    form="pulse", lead_density=7, stutter=0.3, seed=614),
            Visual(system="rucklidge", palette="phosphor", nebula="blackout", style="cyber", sway=20, speed=1.2,
                   glitch=0.6, scanlines=1.0, chroma=0.8, seed=94)),
    Episode(15, "Difference Engine",
            cp.Spec(bpm=82, tonic=38, scale=cp.HARMONIC_MINOR, progression=(0, 5, 3, 4), groove="halftime",
                    pump=0.15, bass="808", pad="choir", lead="bell", arp="pluck", arp_rate=12, arp_shape="pendulum",
                    texture=("typewriter", "clockwork"), form="anthem", lead_density=5, reverb_s=4.5, seed=615),
            Visual(family="magnetic", dipoles=3, palette="brass", nebula="soot", particles=9000, size=1.8,
                   trail=0.965, style="cyber", speed=0.9, glitch=0.2, scanlines=0.4, chroma=0.4, seed=95)),
    Episode(16, "Brute Force",
            cp.Spec(bpm=140, tonic=41, scale=cp.AEOLIAN, progression=(0, 6, 5, 4), groove="trap", bass="808",
                    pad="supersaw", lead="saw", arp="square", arp_rate=16, arp_shape="updown",
                    texture=("servers",), form="club", lead_density=7, stutter=0.5, seed=616),
            Visual(system="chen", palette="bluescreen", nebula="night", style="cyber", sway=30, zoom=0.3,
                   speed=1.5, hue_speed=1.5, glitch=0.9, chroma=1.2, seed=96)),
    Episode(17, "Night Shift",
            cp.Spec(bpm=84, tonic=44, scale=cp.AEOLIAN, progression=(0, 5, 3, 6), sevenths=True, groove="downtempo",
                    swing=0.25, bass="sub", pad="glass", lead="vox", arp="bell", arp_rate=8, arp_shape="random",
                    texture=("rain", "typing"), form="slowburn", lead_density=4, reverb_s=5.0, seed=617),
            Visual(system="aizawa", palette="monochrome", nebula="slate", style="cyber", orbit=0.5, sway=10,
                   speed=0.7, hue_speed=0.5, glitch=0.1, scanlines=0.6, chroma=0.3, trail=0.97, seed=97)),
    Episode(18, "Lockpick",
            cp.Spec(bpm=96, tonic=40, scale=cp.LYDIAN, progression=(0, 1, 4, 1), groove="idm", bass="seq",
                    pad="drone", lead="pluck", arp="square", arp_rate=16, arp_shape="random",
                    texture=("clockwork",), form="loop", lead_density=7, stutter=0.6, seed=618),
            Visual(system="halvorsen", palette="verdigris", nebula="patina", style="cyber", sway=20, speed=1.1,
                   glitch=0.5, chroma=0.9, seed=98)),
    Episode(19, "Logout",
            cp.Spec(bpm=60, tonic=45, scale=cp.AEOLIAN, progression=(0, 5, 3, 4), chord_bars=4, groove="heartbeat",
                    bass="sub", pad="choir", lead="sine", arp="pluck", arp_rate=8, arp_shape="pendulum",
                    texture=("radio",), form="slowburn", lead_density=3, reverb_s=6.0, seed=619),
            Visual(system="lorenz", palette="ash", nebula="smoke", style="cyber", orbit=0.5, sway=10, speed=0.6,
                   hue_speed=0.5, glitch=0.0, scanlines=0.5, chroma=0.3, trail=0.97, seed=99)),
]  # fmt: skip

_HACKER = {e.number: e for e in HACKER + HACKER_EXTRA}
# A cold start, the middle alternating heavy and quiet, drum & bass near the end, a heartbeat to log out.
HACKER_ORDER = [11, 1, 17, 2, 13, 3, 14, 5, 12, 7, 16, 9, 18, 6, 15, 4, 8, 10, 19]

# Nine more grunge songs: the drum feels the series had not used (brushes, sludge, half-time,
# none), drone and rain layers, Aeolian and Phrygian keys, and figures it had not shown.
GRUNGE_EXTRA = [
    Episode(11, "Rain Dog",
            gr.Spec(bpm=92, meter=6, subdiv=3, tonic=41, scale=gr.IONIAN, verse="acoustic", chorus="acoustic",
                    intro="acoustic", chorus_roots=(0, 7, 5, 9), chord_bars=1, chorus_strum="x..x.o", drums="brush",
                    voice_center=60, lead_scale=gr.MAJOR_PENTATONIC,
                    layers=("bass", "vocals", "lead", "acoustic", "rain"), reverb_s=2.8, seed=711),
            Visual(system="lu", palette="ice", nebula="slate", sway=15, seed=51)),
    Episode(12, "Sludge Tide",
            gr.Spec(bpm=72, tonic=38, scale=gr.PHRYGIAN, drive=7.5, quiet_verse=True,
                    riff=((0, 3, 0, "p"), (3, 1, 1, "b"), (4, 2, 0, "m"), (6, 2, -2, "s")),
                    chorus_roots=(0, 1, -2, 0), chord_bars=1, chorus_strum="x...x.x.", drums="sludge",
                    voice_center=58, layers=("bass", "vocals", "lead", "drone"), reverb_s=3.0, seed=712),
            Visual(system="rucklidge", palette="blood", nebula="crimson", sway=20, seed=52)),
    Episode(13, "Halfway Home",
            gr.Spec(bpm=100, tonic=43, scale=gr.MIXOLYDIAN, drive=5.5, verse="both", intro="acoustic",
                    riff=((0, 2, 0, "p"), (2, 1, 5, "n"), (3, 1, 7, "n"), (4, 2, 10, "b"), (6, 2, 7, "p")),
                    chorus_roots=(0, -2, 5, 0), chord_bars=1, chorus_strum="x.x.x.xo", drums="halftime",
                    voice_center=62, lead_scale=gr.MAJOR_PENTATONIC, layers=("bass", "vocals", "lead", "acoustic"),
                    reverb_s=2.6, seed=713),
            Visual(system="sprott", palette="moss", nebula="swamp", orbit=0.5, sway=10, seed=53)),
    Episode(14, "Feedback Loop",
            gr.Spec(bpm=132, tonic=40, scale=gr.AEOLIAN, drive=6.5, quiet_verse=True, wah="riff",
                    riff=((0, 1, 0, "m"), (1, 1, 0, "m"), (2, 1, 3, "n"), (3, 1, 5, "p"), (4, 2, 7, "b"),
                          (6, 1, 5, "n"), (7, 1, 3, "n")),
                    chorus_roots=(0, 8, 3, 10), chord_bars=1, chorus_strum="xxxxxxxx", chorus_strum_b="x.x.xxxx",
                    drums="rock", voice_center=61, layers=("bass", "vocals", "lead"), reverb_s=2.2, seed=714),
            Visual(system="burke_shaw", palette="toxic", nebula="venom", speed=1.2, seed=54)),
    Episode(15, "Driftwood",
            gr.Spec(bpm=108, tonic=45, scale=gr.IONIAN, verse="acoustic", chorus="acoustic", intro="acoustic",
                    chorus_roots=(0, 5, 9, 7), chord_bars=1, chorus_strum="x.xo.oxo", chorus_strum_b="x.xox.xx",
                    drums="brush", voice_center=63, lead_scale=gr.MAJOR_PENTATONIC,
                    layers=("bass", "vocals", "lead", "acoustic", "cello"), reverb_s=2.6, seed=715),
            Visual(family="magnetic", dipoles=2, palette="sunset", nebula="dusk", particles=9000, size=1.8,
                   trail=0.965, seed=55)),
    Episode(16, "Undertow",
            gr.Spec(bpm=90, meter=12, subdiv=3, tonic=40, scale=gr.DORIAN, verse="acoustic", chorus="both",
                    intro="acoustic", chorus_roots=(0, 5, 3, 7), chord_bars=1, chorus_strum="x..x..x.ox..",
                    drums="shuffle", drive=5.5, voice_center=60, layers=("bass", "vocals", "lead", "acoustic"),
                    reverb_s=2.8, seed=716),
            Visual(system="rossler", palette="glacier", nebula="abyss", sway=15, orbit=0.5, seed=56)),
    Episode(17, "Fault Line",
            gr.Spec(bpm=140, tonic=42, scale=gr.MIXOLYDIAN, drive=6.5, quiet_verse=True, riff_bars=2,
                    riff=((0, 1, 0, "m"), (1, 1, 0, "m"), (2, 2, 3, "b"), (4, 1, 0, "m"), (5, 1, 5, "n"),
                          (6, 2, 7, "p"), (8, 2, 10, "b"), (10, 2, 7, "p"), (12, 4, 0, "v")),
                    chorus_roots=(0, -2, 5, 7), chord_bars=1, chorus_strum="x.xxx.xx", drums="rock",
                    voice_center=62, lead_scale=gr.MAJOR_PENTATONIC, layers=("bass", "vocals", "lead"),
                    reverb_s=2.0, seed=717),
            Visual(family="magnetic", dipoles=4, palette="brass", nebula="sepia", particles=9000, size=1.8,
                   trail=0.965, seed=57)),
    Episode(18, "Grey Harbor",
            gr.Spec(bpm=80, tonic=38, scale=gr.AEOLIAN, verse="acoustic", chorus="acoustic", intro="acoustic",
                    chorus_roots=(0, 8, 3, 10), chord_bars=2, chorus_strum="x...x.xo", drums="none",
                    voice_center=58, layers=("bass", "vocals", "lead", "acoustic", "cello", "drone", "rain"),
                    reverb_s=3.2, seed=718),
            Visual(family="graph", palette="verdigris", nebula="smoke", size=1.7, trail=0.97, seed=58)),
    Episode(19, "Long Way Down",
            gr.Spec(bpm=124, tonic=40, scale=gr.IONIAN, riff_bars=2, drive=6.0, verse="both", chorus="both",
                    intro="acoustic", chord_bars=1,
                    riff=((0, 2, 0, "p"), (2, 2, 4, "p"), (4, 2, 7, "p"), (6, 2, 9, "b"),
                          (8, 3, 5, "p"), (11, 1, 4, "n"), (12, 4, 2, "v")),
                    chorus_roots=(0, 5, 9, 7), chorus_strum="x.xxx.xo", chorus_strum_b="x.x.xxxx", drums="rock",
                    voice_center=62, lead_scale=gr.MAJOR_PENTATONIC, layers=("bass", "vocals", "lead", "acoustic"),
                    reverb_s=2.6, seed=719),
            Visual(family="magnetic", dipoles=5, palette="copper", nebula="void", particles=9000, size=1.8,
                   trail=0.965, seed=59)),
]  # fmt: skip

_GRUNGE = {e.number: e for e in GRUNGE + GRUNGE_EXTRA}
# Brushes to open, fast and slow songs alternating, the two anthems to close.
GRUNGE_ORDER = [11, 1, 15, 3, 12, 5, 2, 14, 4, 7, 16, 9, 18, 13, 6, 17, 8, 19, 10]

MIXES = {
    "dark": ([_BY_NUMBER[n] for n in DARK_ORDER], render_track),
    "cyberpunk": ([_CYBER[n] for n in CYBER_ORDER], cp.render_track),
    "hackers": ([_HACKER[n] for n in HACKER_ORDER], cp.render_track),
    "grunge": ([_GRUNGE[n] for n in GRUNGE_ORDER], gr.render_track),
}


def stamp(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60:02d}:{s % 60:02d}"


def smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def fit(x: np.ndarray, n: int) -> np.ndarray:
    """Pad (by repeating) or trim so every scene returns exactly n particles."""
    if len(x) == n:
        return x
    return x[np.arange(n) % len(x)]


def build_audio(name: str, episodes: list[Episode], track_fn, folder: Path) -> tuple[Path, dict, list[float]]:
    tracks = folder / "tracks"
    tracks.mkdir(parents=True, exist_ok=True)
    step = DURATION - OVERLAP
    total = int((step * (len(episodes) - 1) + DURATION) * SR)
    mix = np.zeros((total, 2), dtype=np.float32)
    n_frames = int(total / SR * FPS)
    curves = {k: np.zeros(n_frames) for k in ("rms", "bass", "kick")}
    weight = np.zeros(n_frames)
    starts = []
    fade_n = int(OVERLAP * SR)
    ramp = np.sin(np.linspace(0, np.pi / 2, fade_n)) ** 2  # equal power when combined with its mirror
    for i, ep in enumerate(episodes):
        wav, js = tracks / f"{ep.slug}.wav", tracks / f"{ep.slug}.json"
        if not wav.exists():
            track_fn(replace(ep.music, duration=DURATION), wav, js)
        _, x = wavfile.read(wav)
        x = x.astype(np.float32)
        x *= 0.12 / (np.sqrt(np.mean(x**2)) + 1e-9)  # level every theme to the same RMS
        g = np.ones(len(x), dtype=np.float32)
        if i > 0:
            g[:fade_n] = np.sqrt(ramp)
        if i < len(episodes) - 1:
            g[-fade_n:] = np.sqrt(ramp[::-1])
        s = int(i * step * SR)
        m = min(len(x), total - s)
        mix[s : s + m] += x[:m] * g[:m, None]
        starts.append(i * step)
        a = json.loads(js.read_text())
        f0 = int(i * step * FPS)
        for k in curves:
            v = np.array(a[k])[: n_frames - f0]
            curves[k][f0 : f0 + len(v)] += v
        weight[f0 : f0 + len(a["rms"])] += 1
    for k in curves:
        curves[k] /= np.maximum(weight, 1)
    peak = np.max(np.abs(mix)) + 1e-9
    mix = np.tanh(1.1 * mix / peak) / np.tanh(1.1) * 10 ** (-1.0 / 20)
    out = folder / f"{name}-mix.wav"
    wavfile.write(out, SR, mix.astype(np.float32))
    analysis = {"fps": FPS, "duration": total / SR, **{k: v.tolist() for k, v in curves.items()}}
    return out, analysis, starts


def render_mix_video(episodes: list[Episode], analysis: dict, starts: list[float], out: Path) -> None:
    duration = analysis["duration"]
    rms, bass, kick = (np.array(analysis[k]) for k in ("rms", "bass", "kick"))
    n_frames = int(duration * FPS)
    cyber = any(e.visual.style == "cyber" for e in episodes)
    rd = Renderer(out, N_POINTS, cyber=cyber, fps=FPS, scale=SCALE)
    glitches: dict[int, np.ndarray] = {}
    boost, bloom_add = (1.45, 0.25) if cyber else (1.0, 0.0)
    scenes: dict[int, Scene] = {}
    pairing: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    delay = np.random.default_rng(7).uniform(0, 0.35, N_POINTS)  # particles leave at slightly different times
    t0 = time.time()

    def scene(i: int) -> Scene:
        if i not in scenes:
            scenes[i] = Scene(episodes[i].visual, DURATION, N_POINTS)
            if cyber:
                glitches[i] = glitch_schedule(episodes[i].visual, int(DURATION * FPS) + 2, FPS)
        return scenes[i]

    def sample(i: int, t_local: float, level: float, k_: float):
        sx, sy, col, size = scene(i).step(t_local, level, k_, FPS)
        return fit(sx, N_POINTS), fit(sy, N_POINTS), fit(col, N_POINTS), fit(size, N_POINTS)

    for k in range(n_frames):
        t = k / FPS
        level, bass_l, kick_l = float(rms[min(k, len(rms) - 1)]), float(bass[min(k, len(bass) - 1)]), float(kick[min(k, len(kick) - 1)])
        # Theme j fades in over [starts[j], starts[j] + OVERLAP) while theme j - 1 fades out.
        j = min(int(t // (DURATION - OVERLAP)), len(episodes) - 1)
        morphing = j > 0 and t < starts[j] + OVERLAP
        cur = j - 1 if morphing else j
        a = sample(cur, t - starts[cur], level, kick_l)
        vis = episodes[cur].visual
        nebula = scene(cur).nebula(t - starts[cur])
        intensity, decay, bloom = vis.intensity, vis.trail, vis.bloom
        scan, chroma = vis.scanlines, vis.chroma
        glitch = float(glitches[cur][int((t - starts[cur]) * FPS)]) if cyber else 0.0
        if morphing:
            u = (t - starts[j]) / OVERLAP
            b = sample(j, t - starts[j], level, kick_l)
            if j not in pairing:  # pair particles by angle around the centre, once per transition
                ang_a = np.arctan2(a[1] - 540, a[0] - 960)
                ang_b = np.arctan2(b[1] - 540, b[0] - 960)
                pairing[j] = (np.argsort(ang_a), np.argsort(ang_b))
            ia, ib = pairing[j]
            w = smooth((u - delay) / 0.65)
            sx = a[0][ia] * (1 - w) + b[0][ib] * w
            sy = a[1][ia] * (1 - w) + b[1][ib] * w
            col = a[2][ia] * (1 - w[:, None]) + b[2][ib] * w[:, None]
            size = a[3][ia] * (1 - w) + b[3][ib] * w
            v2, uu = episodes[j].visual, float(smooth(u))
            nebula = nebula * (1 - uu) + scene(j).nebula(t - starts[j]) * uu
            intensity = intensity * (1 - uu) + v2.intensity * uu
            decay = decay * (1 - uu) + v2.trail * uu
            bloom = bloom * (1 - uu) + v2.bloom * uu
            scan = scan * (1 - uu) + v2.scanlines * uu
            chroma = chroma * (1 - uu) + v2.chroma * uu
            if cyber:
                glitch = max(glitch * (1 - uu), float(glitches[j][int((t - starts[j]) * FPS)]) * uu)
        else:
            scenes.pop(j - 1, None)  # the previous figure is gone for good
            glitches.pop(j - 1, None)
            sx, sy, col, size = a
        rd.frame(
            np.column_stack([sx, sy, col, size]),
            t=t, nebula=nebula, level=level, bg_level=0.4 * level + 0.6 * bass_l,
            intensity=intensity * boost * (1 + 0.35 * level + 0.25 * kick_l), decay=decay, bloom=bloom + bloom_add,
            fade=max(min(t / 4.0, (duration - t) / 6.0, 1.0), 0.0), kick=kick_l,
            cyber_params=(glitch * (0.4 + 0.6 * level), scan, chroma),
        )  # fmt: skip
        if k % (FPS * 300) == 0 and k:
            done = k / n_frames
            print(f"  video {done:.0%} ({(time.time() - t0) / done * (1 - done) / 60:.0f} min left)", flush=True)
    rd.close()


def build(name: str) -> Path:
    episodes, track_fn = MIXES[name]
    folder = OUTPUT / "mixes"
    folder.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    wav, analysis, starts = build_audio(name, episodes, track_fn, folder / name)
    print(f"[{name}] audio {analysis['duration'] / 60:.1f} min ({time.time() - t0:.0f}s)", flush=True)
    silent, final = folder / f"{name}-mix.silent.mp4", folder / f"{name}-mix.mp4"
    render_mix_video(episodes, analysis, starts, silent)
    run(
        [
            "ffmpeg", "-v", "error", "-y", "-i", str(silent), "-i", str(wav),
            "-map", "0:v", "-map", "1:a", "-c:v", "copy",
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=9", "-c:a", "aac", "-b:a", "384k", "-ar", "48000",
            "-shortest", "-movflags", "+faststart", str(final),
        ]
    )  # fmt: skip
    silent.unlink(missing_ok=True)
    chapters = "\n".join(f"{stamp(s)} {ep.title}" for s, ep in zip(starts, episodes))
    (folder / f"{name}-mix.chapters.txt").write_text(chapters + "\n")
    print(f"[{name}] -> {final} ({(time.time() - t0) / 60:.0f} min)\n{chapters}", flush=True)
    return final


def main() -> None:
    for name in sys.argv[1:] or ["dark"]:
        build(name)


if __name__ == "__main__":
    main()
