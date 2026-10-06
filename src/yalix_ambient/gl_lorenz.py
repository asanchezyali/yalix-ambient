"""GPU renderer: thousands of particles flowing along the Lorenz attractor.

    dx/dt = σ(y − x),   dy/dt = x(ρ − z) − y,   dz/dt = xy − βz     (σ=10, ρ=28, β=8/3)

Particles are seeded near the origin and integrated until they settle on the
attractor, so the "butterfly" is fully drawn from the first frame. Each frame the
system advances a little (RK4), the camera turns slowly around the vertical axis,
and the particles are drawn as soft additive points into a feedback buffer that
leaves glowing ribbons. Colour follows particle speed; the kick drum adds a subtle
pulse and the bass breathes in the background nebula.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import moderngl
import numpy as np

from yalix_ambient.gl_render import BG_FS, BLUR_FS, COMP_FS, QUAD_VS, TRAIL_FS, H, W, palette_at

SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0
N_PARTICLES = 7000

# Embers: crimson -> magenta -> violet -> amber for the fastest particles
EMBER = np.array(
    [
        [0.55, 0.04, 0.10],
        [0.85, 0.10, 0.35],
        [0.55, 0.18, 0.85],
        [1.00, 0.55, 0.18],
    ],
    dtype=np.float32,
)
NEBULA_DARK = np.array(
    [
        [0.16, 0.01, 0.04],  # blood red
        [0.10, 0.02, 0.15],  # dark violet
        [0.04, 0.01, 0.08],  # near-black indigo
        [0.14, 0.02, 0.09],  # wine
    ],
    dtype=np.float32,
)

POINT_VS = """
#version 330
in vec2 in_pos; in vec3 in_color; in float in_size;
out vec3 v_color;
uniform vec2 res;
void main() {
    v_color = in_color;
    gl_PointSize = in_size;
    gl_Position = vec4(in_pos / res * 2.0 - 1.0, 0.0, 1.0);
}
"""

POINT_FS = """
#version 330
in vec3 v_color; out vec4 frag;
uniform float intensity;
void main() {
    vec2 d = gl_PointCoord - 0.5;
    float a = exp(-dot(d, d) * 26.0);
    frag = vec4(v_color * a * intensity, 1.0);
}
"""


def lorenz(p: np.ndarray) -> np.ndarray:
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    return np.stack([SIGMA * (y - x), x * (RHO - z) - y, x * y - BETA * z], axis=1)


def rk4(p: np.ndarray, dt: float) -> np.ndarray:
    k1 = lorenz(p)
    k2 = lorenz(p + 0.5 * dt * k1)
    k3 = lorenz(p + 0.5 * dt * k2)
    k4 = lorenz(p + dt * k3)
    return p + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def render_video(analysis_path: Path, out_path: Path, fps: int = 30, seed: int = 3) -> None:
    data = json.loads(analysis_path.read_text())
    duration = data["duration"]
    rms, bass, kick = (np.array(data[k]) for k in ("rms", "bass", "kick"))
    n_frames = int(duration * fps)

    # Sample the starting positions along one long trajectory so the particles cover
    # the whole attractor from the first frame (a tight cloud would take minutes to spread).
    rng = np.random.default_rng(seed)
    p = np.array([[1.0, 1.0, 20.0]])
    for _ in range(3000):
        p = rk4(p, 0.005)
    traj = np.empty((N_PARTICLES * 6, 3))
    for i in range(len(traj)):
        p = rk4(p, 0.005)
        traj[i] = p[0]
    pts = traj[rng.permutation(len(traj))[:N_PARTICLES]] + rng.standard_normal((N_PARTICLES, 3)) * 0.05

    ctx = moderngl.create_standalone_context(require=330)
    ctx.enable(moderngl.PROGRAM_POINT_SIZE)
    quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4"))

    def prog(fs):
        p = ctx.program(vertex_shader=QUAD_VS, fragment_shader=fs)
        return p, ctx.vertex_array(p, [(quad, "2f", "in_pos")])

    bg_p, bg_vao = prog(BG_FS)
    trail_p, trail_vao = prog(TRAIL_FS)
    blur_p, blur_vao = prog(BLUR_FS)
    comp_p, comp_vao = prog(COMP_FS)
    pt_p = ctx.program(vertex_shader=POINT_VS, fragment_shader=POINT_FS)
    pt_buf = ctx.buffer(reserve=N_PARTICLES * 6 * 4)
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
    out_tex = ctx.texture((W, H), 3)
    out_fbo = ctx.framebuffer([out_tex])
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

    cx, cy = W / 2, H / 2
    for k in range(n_frames):
        t = k / fps
        fi = min(int(t * data["fps"]), len(rms) - 1)
        level, bass_l, kick_l = float(rms[fi]), float(bass[fi]), float(kick[fi])

        # Advance the flow slowly: relaxing, not frantic.
        for _ in range(2):
            pts = rk4(pts, 0.0022)
        vel = np.linalg.norm(lorenz(pts), axis=1)

        # Camera: slow turn around the vertical (z) axis plus a gentle sway.
        # The butterfly opens widest at a yaw of ~130° and collapses edge-on at 45°:
        # sway ±28° around the open view so the wings are always readable.
        yaw = np.radians(130) + 0.5 * np.sin(2 * np.pi * t / 90)
        tilt = 0.25 * np.sin(2 * np.pi * t / 47)
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2] - 25.0
        xr = x * np.cos(yaw) - y * np.sin(yaw)
        yr = x * np.sin(yaw) + y * np.cos(yaw)
        zr = z * np.cos(tilt) - yr * np.sin(tilt)
        depth = yr * np.cos(tilt) + z * np.sin(tilt)
        persp = 160.0 / (160.0 + depth)
        scale = H * 0.0175 * (1.0 + 0.012 * kick_l)
        sx = cx + xr * scale * persp
        sy = cy + zr * scale * persp

        speed = np.clip((vel - 20) / 160, 0, 1)
        colors = palette_at(speed * 2.6 + t / 30, EMBER)
        sizes = (2.0 * persp) * (1 + 0.12 * kick_l)
        verts = np.column_stack([sx, sy, colors, sizes]).astype(np.float32)
        pt_buf.write(verts.tobytes())

        # 1. background
        cyc = t / 35.0
        bg_p["t"].value = t
        bg_p["level"].value = 0.4 * level + 0.6 * bass_l
        bg_p["c0"].value = tuple(palette_at(np.array([cyc]), NEBULA_DARK)[0])
        bg_p["c1"].value = tuple(palette_at(np.array([cyc + 1.3]), NEBULA_DARK)[0])
        bg_p["c2"].value = tuple(palette_at(np.array([cyc + 2.6]), NEBULA_DARK)[0])
        bg_fbo.use()
        bg_vao.render(moderngl.TRIANGLE_STRIP)

        # 2. particles (additive)
        cur_fbo.use()
        cur_fbo.clear(0, 0, 0, 1)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        pt_p["intensity"].value = 0.55 + 0.20 * level + 0.15 * kick_l
        pt_vao.render(moderngl.POINTS)
        ctx.disable(moderngl.BLEND)

        # 3. trail
        (prev_tex, _), (next_tex, next_fbo) = trail[k % 2], trail[(k + 1) % 2]
        next_fbo.use()
        prev_tex.use(0)
        cur_tex.use(1)
        trail_p["prev"].value, trail_p["cur"].value = 0, 1
        trail_p["decay"].value = 0.955
        trail_vao.render(moderngl.TRIANGLE_STRIP)

        # 4. bloom
        blur_a[1].use()
        next_tex.use(0)
        blur_p["src"].value = 0
        blur_p["dir"].value = (2.0 / hw, 0.0)
        blur_vao.render(moderngl.TRIANGLE_STRIP)
        blur_b[1].use()
        blur_a[0].use(0)
        blur_p["dir"].value = (0.0, 2.0 / hh)
        blur_vao.render(moderngl.TRIANGLE_STRIP)

        # 5. composite
        out_fbo.use()
        bg_tex.use(0)
        next_tex.use(1)
        blur_b[0].use(2)
        comp_p["bg"].value, comp_p["trail"].value, comp_p["bloom"].value = 0, 1, 2
        fade = max(min(t / 4.0, (duration - t) / 5.0, 1.0), 0.0)
        comp_p["fade"].value = fade
        comp_p["bloom_k"].value = 0.35
        comp_vao.render(moderngl.TRIANGLE_STRIP)
        ff.stdin.write(out_fbo.read(components=3))

    ff.stdin.close()
    ff.wait()
    ctx.release()
