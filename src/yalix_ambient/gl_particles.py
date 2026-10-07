"""Generalized GPU particle renderer for the dark series.

Three families, one pipeline (nebula background -> additive particles -> feedback trail
-> bloom -> composite, piped to ffmpeg):

- attractor: chaotic flows (Lorenz, Thomas, Aizawa, Halvorsen, Rössler, Dadras, Chen,
  four-wing, Arneodo).
  Particles are sampled along one long trajectory so the whole attractor shows from
  frame one. The camera frames it automatically with PCA: it looks along the axis of
  least variance (the most open view) and sways ±25° around it. The time step is
  calibrated so particles move ~1.6 px per frame on screen: calm whatever the system.
- magnetic: 2D field lines of slowly orbiting dipoles (B = (3(m·r̂)r̂ − m)/|r|³);
  particles drift along B and respawn, so the field lines draw themselves.
- flow: particles in a curl-noise velocity field (divergence-free), like smoke in a
  nebula.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import moderngl
import numpy as np

from yalix_ambient.gl_lorenz import POINT_FS, POINT_VS
from yalix_ambient.gl_render import BG_FS, BLUR_FS, COMP_FS, QUAD_VS, TRAIL_FS, H, W, palette_at

# ------------------------------------------------------------------ palettes

PALETTES = {
    "ember": [[0.55, 0.04, 0.10], [0.85, 0.10, 0.35], [0.55, 0.18, 0.85], [1.00, 0.55, 0.18]],
    "ice": [[0.25, 0.35, 0.95], [0.55, 0.40, 0.95], [0.75, 0.85, 1.00], [0.35, 0.80, 0.95]],
    "fire": [[0.95, 0.25, 0.05], [1.00, 0.55, 0.10], [0.80, 0.08, 0.05], [1.00, 0.85, 0.45]],
    "toxic": [[0.45, 0.85, 0.25], [0.85, 0.90, 0.70], [0.65, 0.20, 0.15], [0.25, 0.65, 0.45]],
    "blood": [[0.80, 0.05, 0.10], [0.45, 0.02, 0.06], [0.95, 0.35, 0.35], [0.60, 0.10, 0.40]],
    "aurora": [[0.20, 0.95, 0.65], [0.35, 0.55, 1.00], [0.80, 0.35, 0.95], [0.15, 0.75, 0.90]],
    # grunge: muted, earthy colours that still glow
    "rust": [[0.75, 0.30, 0.10], [0.95, 0.55, 0.25], [0.45, 0.15, 0.08], [0.85, 0.70, 0.45]],
    "moss": [[0.35, 0.55, 0.25], [0.70, 0.75, 0.40], [0.20, 0.35, 0.30], [0.85, 0.80, 0.55]],
    "bruise": [[0.45, 0.25, 0.65], [0.70, 0.35, 0.55], [0.25, 0.30, 0.65], [0.85, 0.60, 0.75]],
    "rain": [[0.40, 0.55, 0.75], [0.70, 0.80, 0.90], [0.25, 0.35, 0.55], [0.55, 0.75, 0.80]],
    "nicotine": [[0.85, 0.70, 0.30], [0.95, 0.85, 0.55], [0.60, 0.40, 0.15], [0.75, 0.55, 0.35]],
    "ash": [[0.70, 0.68, 0.65], [0.90, 0.85, 0.80], [0.45, 0.45, 0.50], [0.80, 0.55, 0.40]],
    # cyberpunk: saturated neon that reads well over near-black
    "neon": [[1.00, 0.10, 0.75], [0.10, 0.90, 1.00], [0.55, 0.20, 1.00], [1.00, 0.35, 0.55]],
    "matrix": [[0.10, 1.00, 0.35], [0.60, 1.00, 0.60], [0.00, 0.55, 0.25], [0.85, 1.00, 0.85]],
    "chrome": [[0.60, 0.90, 1.00], [0.90, 0.95, 1.00], [0.30, 0.55, 0.95], [0.75, 0.60, 1.00]],
    "synthwave": [[1.00, 0.30, 0.60], [1.00, 0.60, 0.15], [0.60, 0.20, 0.90], [0.20, 0.60, 1.00]],
    "laser": [[1.00, 0.10, 0.20], [0.20, 0.40, 1.00], [1.00, 0.45, 0.75], [0.40, 0.90, 1.00]],
    "glacier": [[0.20, 0.80, 1.00], [0.50, 0.50, 1.00], [0.85, 0.95, 1.00], [0.10, 0.50, 0.90]],
    "vapor": [[1.00, 0.55, 0.85], [0.35, 0.95, 0.90], [0.75, 0.60, 1.00], [1.00, 0.80, 0.55]],
    "acid": [[0.80, 1.00, 0.10], [1.00, 0.20, 0.80], [0.10, 0.90, 0.70], [1.00, 0.90, 0.30]],
    "sunset": [[1.00, 0.45, 0.20], [1.00, 0.20, 0.55], [0.55, 0.15, 0.75], [1.00, 0.75, 0.35]],
}
NEBULAE = {
    "crimson": [[0.16, 0.01, 0.04], [0.10, 0.02, 0.15], [0.04, 0.01, 0.08], [0.14, 0.02, 0.09]],
    "abyss": [[0.02, 0.04, 0.16], [0.08, 0.03, 0.18], [0.01, 0.02, 0.06], [0.03, 0.10, 0.16]],
    "inferno": [[0.18, 0.04, 0.01], [0.10, 0.01, 0.02], [0.05, 0.02, 0.01], [0.16, 0.06, 0.02]],
    "venom": [[0.03, 0.10, 0.04], [0.08, 0.02, 0.06], [0.02, 0.04, 0.03], [0.06, 0.09, 0.02]],
    "void": [[0.06, 0.02, 0.10], [0.02, 0.02, 0.06], [0.10, 0.02, 0.06], [0.03, 0.05, 0.10]],
    "sepia": [[0.14, 0.07, 0.03], [0.08, 0.05, 0.04], [0.04, 0.03, 0.02], [0.12, 0.08, 0.04]],
    "swamp": [[0.05, 0.09, 0.05], [0.08, 0.07, 0.03], [0.02, 0.04, 0.04], [0.06, 0.08, 0.06]],
    "dusk": [[0.09, 0.04, 0.13], [0.04, 0.03, 0.10], [0.12, 0.04, 0.08], [0.05, 0.05, 0.12]],
    "slate": [[0.04, 0.07, 0.12], [0.06, 0.06, 0.09], [0.02, 0.04, 0.07], [0.07, 0.09, 0.12]],
    "smoke": [[0.08, 0.08, 0.09], [0.11, 0.08, 0.06], [0.04, 0.04, 0.05], [0.09, 0.09, 0.11]],
    "night": [[0.05, 0.01, 0.12], [0.01, 0.04, 0.10], [0.10, 0.01, 0.08], [0.02, 0.02, 0.06]],
    "neonfog": [[0.12, 0.02, 0.10], [0.02, 0.06, 0.12], [0.06, 0.01, 0.12], [0.01, 0.08, 0.10]],
    "terminal": [[0.01, 0.07, 0.03], [0.00, 0.03, 0.02], [0.02, 0.05, 0.05], [0.00, 0.04, 0.01]],
    "ultraviolet": [[0.08, 0.00, 0.14], [0.03, 0.00, 0.08], [0.12, 0.02, 0.10], [0.04, 0.02, 0.12]],
}

# ------------------------------------------------------------------ attractors


def _lorenz(p):
    x, y, z = p.T
    return np.stack([10 * (y - x), x * (28 - z) - y, x * y - 8 / 3 * z], axis=1)


def _thomas(p):
    x, y, z = p.T
    b = 0.208186
    return np.stack([np.sin(y) - b * x, np.sin(z) - b * y, np.sin(x) - b * z], axis=1)


def _aizawa(p):
    x, y, z = p.T
    a, b, c, d, e, f = 0.95, 0.7, 0.6, 3.5, 0.25, 0.1
    return np.stack(
        [(z - b) * x - d * y, d * x + (z - b) * y, c + a * z - z**3 / 3 - (x * x + y * y) * (1 + e * z) + f * z * x**3],
        axis=1,
    )


def _halvorsen(p):
    x, y, z = p.T
    a = 1.89
    return np.stack([-a * x - 4 * y - 4 * z - y * y, -a * y - 4 * z - 4 * x - z * z, -a * z - 4 * x - 4 * y - x * x], axis=1)


def _rossler(p):
    x, y, z = p.T
    return np.stack([-y - z, x + 0.2 * y, 0.2 + z * (x - 5.7)], axis=1)


def _dadras(p):
    x, y, z = p.T
    a, b, c, d, e = 3.0, 2.7, 1.7, 2.0, 9.0
    return np.stack([y - a * x + b * y * z, c * y - x * z + z, d * x * y - e * z], axis=1)


def _chen(p):
    x, y, z = p.T
    a, b, c = 35.0, 3.0, 28.0
    return np.stack([a * (y - x), (c - a) * x - x * z + c * y, x * y - b * z], axis=1)


def _fourwing(p):
    x, y, z = p.T
    a, b, c = 0.2, 0.01, -0.4
    return np.stack([a * x + y * z, b * x + c * y - x * z, -z - x * y], axis=1)


def _arneodo(p):
    x, y, z = p.T
    return np.stack([y, z, 5.5 * x - 3.5 * y - z - x**3], axis=1)


def _lu(p):
    x, y, z = p.T
    a, b, c = 36.0, 3.0, 20.0
    return np.stack([a * (y - x), -x * z + c * y, x * y - b * z], axis=1)


def _burke_shaw(p):
    x, y, z = p.T
    s, v = 10.0, 4.272
    return np.stack([-s * (x + y), -y - s * x * z, s * x * y + v], axis=1)


def _sprott(p):
    # Sprott (2014): a strange attractor that coexists with invariant tori.
    x, y, z = p.T
    a, b = 2.07, 1.79
    return np.stack([y + a * x * y + x * z, 1 - b * x * x + y * z, x - x * x - y * y], axis=1)


def _rucklidge(p):
    x, y, z = p.T
    k, a = 2.0, 6.7
    return np.stack([-k * x + a * y - y * z, x, -z + y * y], axis=1)


ATTRACTORS = {
    # name: (derivative, start, integration dt for sampling)
    "lu": (_lu, (0.1, 0.3, -0.6), 0.002),
    "burke_shaw": (_burke_shaw, (0.6, 0.0, 0.0), 0.002),
    "sprott": (_sprott, (0.63, 0.47, -0.54), 0.005),
    "rucklidge": (_rucklidge, (1.0, 0.0, 4.5), 0.005),
    "lorenz": (_lorenz, (1.0, 1.0, 20.0), 0.005),
    "thomas": (_thomas, (0.1, 0.0, -0.1), 0.05),
    "aizawa": (_aizawa, (0.1, 0.0, 0.0), 0.01),
    "halvorsen": (_halvorsen, (-1.48, -1.51, 2.04), 0.005),
    "rossler": (_rossler, (1.0, 1.0, 0.0), 0.02),
    "dadras": (_dadras, (1.1, 2.1, -2.0), 0.005),
    "chen": (_chen, (-10.0, 0.0, 37.0), 0.002),
    "fourwing": (_fourwing, (1.3, -0.18, 0.01), 0.02),
    "arneodo": (_arneodo, (0.1, 0.06, 0.04), 0.01),
}


def rk4(f, p, dt):
    k1 = f(p)
    k2 = f(p + 0.5 * dt * k1)
    k3 = f(p + 0.5 * dt * k2)
    k4 = f(p + dt * k3)
    return p + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


# ------------------------------------------------------------------ fields (2D)


def curl_noise(xy: np.ndarray, t: float, seed: int) -> np.ndarray:
    """Divergence-free 2D velocity from a sum-of-sines potential ψ: v = (∂ψ/∂y, −∂ψ/∂x)."""
    rng = np.random.default_rng(seed)
    v = np.zeros_like(xy)
    for _ in range(6):
        k = rng.normal(0, 1, 2) * rng.uniform(1.2, 3.5)
        w = rng.uniform(0.02, 0.06)
        ph = rng.uniform(0, 2 * np.pi)
        a = 1.0 / np.linalg.norm(k)
        arg = xy @ k + w * t * 2 * np.pi + ph
        dpsi = a * np.cos(arg)[:, None] * k[None, :]
        v += np.stack([dpsi[:, 1], -dpsi[:, 0]], axis=1)
    return v


def dipole_field(xy: np.ndarray, t: float, n_dipoles: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    B = np.zeros_like(xy)
    for i in range(n_dipoles):
        orbit_r = rng.uniform(0.15, 0.55) if n_dipoles > 1 else 0.0
        speed = rng.uniform(0.01, 0.025) * (1 if i % 2 else -1)
        ang0 = 2 * np.pi * i / n_dipoles
        c = orbit_r * np.array([np.cos(ang0 + speed * t * 2 * np.pi), 0.6 * np.sin(ang0 + speed * t * 2 * np.pi)])
        m_ang = rng.uniform(0, 2 * np.pi) + 0.03 * t * (1 if i % 2 else -1)
        m = np.array([np.cos(m_ang), np.sin(m_ang)]) * (1 if i % 2 == 0 or n_dipoles < 4 else -1)
        r = xy - c
        d = np.linalg.norm(r, axis=1, keepdims=True) + 0.03
        rh = r / d
        B += (3 * (rh @ m)[:, None] * rh - m) / d**3
    return B


@dataclass
class Visual:
    family: str = "attractor"  # attractor | magnetic | flow
    system: str = "lorenz"
    dipoles: int = 2
    palette: str = "ember"
    nebula: str = "crimson"
    particles: int = 7000
    trail: float = 0.955
    intensity: float = 0.55
    size: float = 2.0
    bloom: float = 0.35
    seed: int = 3
    style: str = "default"  # default | cyber (scanlines, chromatic aberration, glitch bursts)


# Cyberpunk composite: same tone mapping as COMP_FS, plus RGB split that opens on the kick,
# CRT scanlines, a vignette and short horizontal-slice glitches.
CYBER_COMP_FS = """
#version 330
in vec2 uv; out vec4 frag;
uniform sampler2D bg; uniform sampler2D trail; uniform sampler2D bloom;
uniform float fade; uniform float bloom_k;
uniform float t; uniform float kick; uniform float glitch; uniform vec2 res;
float hash(float n) { return fract(sin(n) * 43758.5453); }
vec3 scene(vec2 p) {
    return texture(bg, p).rgb + texture(trail, p).rgb + texture(bloom, p).rgb * bloom_k;
}
void main() {
    vec2 p = uv;
    float band = floor(p.y * 36.0) + floor(t * 15.0) * 7.0;
    float h = hash(band);
    if (h > 1.0 - 0.35 * glitch) p.x += (hash(band + 1.7) - 0.5) * 0.06 * glitch;
    vec2 dir = p - 0.5;
    float ca = 0.0012 + 0.0045 * kick + 0.008 * glitch;
    vec3 c = vec3(scene(p + dir * ca).r, scene(p).g, scene(p - dir * ca).b);
    c = 1.0 - exp(-c * 1.15);
    c = pow(c, vec3(0.95));
    c *= 0.90 + 0.10 * sin(p.y * res.y * 3.14159);
    c *= 1.0 - 0.35 * dot(dir, dir) * 2.0;
    frag = vec4(c * fade, 1.0);
}
"""


# ------------------------------------------------------------------ renderer


def render_video(analysis_path: Path, out_path: Path, vis: Visual, fps: int = 30) -> None:
    data = json.loads(analysis_path.read_text())
    duration = data["duration"]
    rms, bass, kick = (np.array(data[k]) for k in ("rms", "bass", "kick"))
    n_frames = int(duration * fps)
    rng = np.random.default_rng(vis.seed)
    pal = np.array(PALETTES[vis.palette], dtype=np.float32)
    neb = np.array(NEBULAE[vis.nebula], dtype=np.float32)
    N = vis.particles
    cx, cy = W / 2, H / 2

    if vis.family == "attractor":
        f, start, dt_s = ATTRACTORS[vis.system]
        p = np.array([start], dtype=float)
        for _ in range(4000):
            p = rk4(f, p, dt_s)
        traj = np.empty((N * 6, 3))
        for i in range(len(traj)):
            p = rk4(f, p, dt_s)
            traj[i] = p[0]
        centre = traj.mean(axis=0)
        _, _, vt = np.linalg.svd(traj - centre, full_matrices=False)
        e1, e2, e3 = vt[0], vt[1], vt[2]  # view along e3: the flattest direction
        if abs(e2[2]) < abs(e1[2]) and abs(e1[2]) > 0.5:  # keep the 'up' axis vertical when obvious
            e1, e2 = e2, e1
        proj = np.column_stack([(traj - centre) @ e1, (traj - centre) @ e2])
        extent = np.percentile(np.abs(proj), 99.5, axis=0)
        scale0 = min(W * 0.40 / extent[0], H * 0.40 / extent[1])
        spread = np.ptp(traj, axis=0).max()
        pts = traj[rng.permutation(len(traj))[:N]] + rng.standard_normal((N, 3)) * spread * 0.0015
        speed_px = np.linalg.norm(f(traj[:2000]), axis=1).mean() * scale0
        dt_frame = 1.6 / speed_px  # ~1.6 px per frame on screen
        vel_ref = np.percentile(np.linalg.norm(f(traj), axis=1), [5, 95])
    else:
        pts = rng.uniform(-1, 1, (N, 2)) * np.array([W / H, 1.0])
        age = rng.uniform(0, 1, N)

    ctx = moderngl.create_standalone_context(require=330)
    ctx.enable(moderngl.PROGRAM_POINT_SIZE)
    quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4"))

    def prog(fs):
        pr = ctx.program(vertex_shader=QUAD_VS, fragment_shader=fs)
        return pr, ctx.vertex_array(pr, [(quad, "2f", "in_pos")])

    bg_p, bg_vao = prog(BG_FS)
    trail_p, trail_vao = prog(TRAIL_FS)
    blur_p, blur_vao = prog(BLUR_FS)
    cyber = vis.style == "cyber"
    comp_p, comp_vao = prog(CYBER_COMP_FS if cyber else COMP_FS)
    if cyber:
        comp_p["res"].value = (W, H)
        # Glitch bursts: a few frames long, rare, and only where the music is already loud.
        glitch = np.zeros(n_frames)
        g_rng = np.random.default_rng(vis.seed + 7)
        k_ = int(4 * fps)
        while k_ < n_frames - 5 * fps:
            k_ += int(g_rng.uniform(5, 12) * fps)
            span = int(g_rng.integers(3, 8))
            glitch[k_ : k_ + span] = np.linspace(1.0, 0.3, len(glitch[k_ : k_ + span]))
    pt_p = ctx.program(vertex_shader=POINT_VS, fragment_shader=POINT_FS)
    pt_buf = ctx.buffer(reserve=N * 6 * 4)
    pt_vao = ctx.vertex_array(pt_p, [(pt_buf, "2f 3f 1f", "in_pos", "in_color", "in_size")])

    def target(w, h):
        tex = ctx.texture((w, h), 4, dtype="f2")
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        return tex, ctx.framebuffer([tex])

    bg_tex, bg_fbo = target(W, H)
    cur_tex, cur_fbo = target(W, H)
    trail = [target(W, H), target(W, H)]
    hw, hh = W // 2, H // 2
    blur_a, blur_b = target(hw, hh), target(hw, hh)
    out_fbo = ctx.framebuffer([ctx.texture((W, H), 3)])
    bg_p["res"].value = (W, H)
    pt_p["res"].value = (W, H)

    ff = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
            "-vf", "vflip", "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
            str(out_path),
        ],
        stdin=subprocess.PIPE,
    )  # fmt: skip

    for k in range(n_frames):
        t = k / fps
        fi = min(int(t * data["fps"]), len(rms) - 1)
        level, bass_l, kick_l = float(rms[fi]), float(bass[fi]), float(kick[fi])

        if vis.family == "attractor":
            for _ in range(2):
                pts = rk4(f, pts, dt_frame / 2)
            sway = np.radians(25) * np.sin(2 * np.pi * t / 90)
            rel = pts - centre
            a, b, c = rel @ e1, rel @ e2, rel @ e3
            sx_ = a * np.cos(sway) + c * np.sin(sway)
            depth = -a * np.sin(sway) + c * np.cos(sway)
            depth_n = depth / (np.abs(c).max() + 1e-9)
            persp = 1.0 / (1.0 + 0.18 * depth_n)
            scale = scale0 * (1.0 + 0.012 * kick_l)
            sx, sy = cx + sx_ * scale * persp, cy + b * scale * persp
            spd = np.linalg.norm(f(pts), axis=1)
            hue = np.clip((spd - vel_ref[0]) / (vel_ref[1] - vel_ref[0] + 1e-9), 0, 1)
            sizes = vis.size * persp * (1 + 0.12 * kick_l)
        else:
            if vis.family == "magnetic":
                v = dipole_field(pts, t, vis.dipoles, vis.seed)
                mag = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
                v = v / mag
                step = 0.0022
                hue = np.clip(np.log10(mag[:, 0] + 1e-9) / 3.0 + 0.3, 0, 1)
            else:
                v = curl_noise(pts, t, vis.seed)
                step = 0.0016
                hue = np.clip(np.linalg.norm(v, axis=1) / 2.5, 0, 1)
            pts = pts + v * step * (1 + 0.3 * level)
            age += 1.0 / (fps * 9)
            out = (np.abs(pts[:, 0]) > W / H * 1.05) | (np.abs(pts[:, 1]) > 1.05) | (age > 1)
            if vis.family == "magnetic":
                out |= mag[:, 0] > 2e3  # swallowed by a pole
            n_out = int(out.sum())
            if n_out:
                pts[out] = rng.uniform(-1, 1, (n_out, 2)) * np.array([W / H, 1.0])
                age[out] = 0
            sx, sy = cx + pts[:, 0] * H / 2, cy + pts[:, 1] * H / 2
            fade_age = np.minimum(age * 8, 1) * np.minimum((1 - age) * 8, 1)
            sizes = vis.size * (0.6 + 0.4 * fade_age) * (1 + 0.12 * kick_l)

        colors = palette_at(hue * 2.6 + t / 30, pal)
        if vis.family != "attractor":
            colors = colors * fade_age[:, None]
        pt_buf.write(np.column_stack([sx, sy, colors, sizes]).astype(np.float32).tobytes())

        cyc = t / 35.0
        bg_p["t"].value = t
        bg_p["level"].value = 0.4 * level + 0.6 * bass_l
        for name, off in (("c0", 0.0), ("c1", 1.3), ("c2", 2.6)):
            bg_p[name].value = tuple(palette_at(np.array([cyc + off]), neb)[0])
        bg_fbo.use()
        bg_vao.render(moderngl.TRIANGLE_STRIP)

        cur_fbo.use()
        cur_fbo.clear(0, 0, 0, 1)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        boost = 1.45 if cyber else 1.0  # neon should glow, not smoulder
        pt_p["intensity"].value = vis.intensity * boost * (1 + 0.35 * level + 0.25 * kick_l)
        pt_vao.render(moderngl.POINTS)
        ctx.disable(moderngl.BLEND)

        (prev_tex, _), (next_tex, next_fbo) = trail[k % 2], trail[(k + 1) % 2]
        next_fbo.use()
        prev_tex.use(0)
        cur_tex.use(1)
        trail_p["prev"].value, trail_p["cur"].value = 0, 1
        trail_p["decay"].value = vis.trail
        trail_vao.render(moderngl.TRIANGLE_STRIP)

        blur_a[1].use()
        next_tex.use(0)
        blur_p["src"].value = 0
        blur_p["dir"].value = (2.0 / hw, 0.0)
        blur_vao.render(moderngl.TRIANGLE_STRIP)
        blur_b[1].use()
        blur_a[0].use(0)
        blur_p["dir"].value = (0.0, 2.0 / hh)
        blur_vao.render(moderngl.TRIANGLE_STRIP)

        out_fbo.use()
        bg_tex.use(0)
        next_tex.use(1)
        blur_b[0].use(2)
        comp_p["bg"].value, comp_p["trail"].value, comp_p["bloom"].value = 0, 1, 2
        comp_p["fade"].value = max(min(t / 4.0, (duration - t) / 5.0, 1.0), 0.0)
        comp_p["bloom_k"].value = vis.bloom + (0.25 if cyber else 0.0)
        if cyber:
            comp_p["t"].value = t
            comp_p["kick"].value = kick_l
            comp_p["glitch"].value = float(glitch[k]) * (0.4 + 0.6 * level)
        comp_vao.render(moderngl.TRIANGLE_STRIP)
        ff.stdin.write(out_fbo.read(components=3))

    ff.stdin.close()
    ff.wait()
    ctx.release()
