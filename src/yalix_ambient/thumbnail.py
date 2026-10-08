"""YouTube thumbnails for the one-hour mixes, built from a frame of the rendered video.

The layout is composed at 2560x1440 (twice YouTube's 1280x720) with the figure taken from the
1440p render at full resolution, then downsampled with Lanczos and a light unsharp mask, so the
particle threads and the type stay crisp after YouTube re-encodes the image.

    uv run yalix-thumb dark            # uses the frame time set in THUMBS
    uv run yalix-thumb dark --at 640   # try another frame
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MIXES = ROOT / "output" / "mixes"
FONTS = Path("/System/Library/Fonts/Supplemental")
IMPACT, DIN = FONTS / "Impact.ttf", FONTS / "DIN Alternate Bold.ttf"


@dataclass
class Thumb:
    at: float  # seconds into the mix
    title: str
    accent: str  # title colour
    tag: str  # colour of the bottom line
    lines: tuple[str, str]
    crop: tuple[int, int, int, int] = (1600, 1440, 0, 0)  # w, h, x offset, y offset from centre
    lift: int = 55  # white point of the figure, in percent (lower = brighter)
    saturation: int = 125
    title_size: int = 236


THUMBS = {
    "dark": Thumb(1700, "DARK CHAOS", "#ff2a3a", "#ff5a5a", ("MUSIC & VISUALS", "GENERATED WITH MATH")),
    "cyberpunk": Thumb(280, "NEON CHAOS", "#ff2bd6", "#22e6ff", ("CYBERPUNK & SYNTHWAVE", "GENERATED WITH MATH")),
    "hackers": Thumb(1229, "HACKER CHAOS", "#ffb52e", "#3dff8a", ("DARK STEAMPUNK CODING MUSIC", "GENERATED WITH MATH"),
                     title_size=224),
    "grunge": Thumb(1030, "GRUNGE CHAOS", "#ff8a2a", "#f2d38a", ("SEATTLE GRUNGE FOR FOCUS", "GENERATED WITH MATH"),
                    title_size=224),
    "deepspace": Thumb(1218, "DEEP SPACE", "#ffb070", "#9fc8ff", ("SPACE AMBIENT, NO DRUMS", "GENERATED WITH MATH")),
}  # fmt: skip


def run(*args: object) -> None:
    subprocess.run([str(a) for a in args], check=True)


def build(name: str, at: float | None = None) -> Path:
    t = THUMBS[name]
    video, out = MIXES / f"{name}-mix.mp4", MIXES / f"{name}-mix.thumbnail.jpg"
    w, h, dx, dy = t.crop
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        run("ffmpeg", "-v", "error", "-y", "-ss", at if at is not None else t.at, "-i", video,
            "-frames:v", "1", "-vf", "scale=2560:1440:flags=lanczos", d / "full.png")  # 1080p mixes too
        # The figure fills the right 1520x1440, faded into black towards the text on the left.
        run("magick", d / "full.png", "-filter", "Lanczos", "-gravity", "center",
            "-crop", f"{w}x{h}{dx:+d}{dy:+d}", "+repage", "-resize", "1520x1440^", "-extent", "1520x1440",
            "-level", f"0%,{t.lift}%", "-modulate", f"100,{t.saturation}", d / "fig.png")
        run("magick", "-size", "1440x1520", "gradient:black-white", "-rotate", "-90", "-level", "0%,45%",
            d / "mask.png")
        run("magick", d / "fig.png", d / "mask.png", "-alpha", "off", "-compose", "copyopacity", "-composite",
            d / "figm.png")
        run("magick", "-size", "2560x1440", "xc:black", d / "figm.png", "-gravity", "east", "-compose", "over",
            "-composite", "-gravity", "northwest",
            "-font", IMPACT, "-fill", "white", "-pointsize", 420, "-annotate", "+120+140", "1 HOUR",
            "-font", IMPACT, "-fill", t.accent, "-pointsize", t.title_size, "-annotate", "+128+600", t.title,
            "-font", DIN, "-fill", "#e0e0e0", "-pointsize", 80, "-annotate", "+136+900", t.lines[0],
            "-annotate", "+136+1000", t.lines[1],
            "-font", DIN, "-fill", t.tag, "-pointsize", 60, "-annotate", "+136+1240", "19 THEMES",
            d / "big.png")
        run("magick", d / "big.png", "-filter", "Lanczos", "-resize", "1280x720", "-unsharp", "0x0.6+0.6+0.01",
            "-sampling-factor", "4:4:4", "-quality", 95, out)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Build a mix thumbnail from a frame of its video.")
    p.add_argument("name", choices=sorted(THUMBS))
    p.add_argument("--at", type=float, help="frame time in seconds (overrides the default)")
    a = p.parse_args()
    print(build(a.name, a.at))


if __name__ == "__main__":
    main()
