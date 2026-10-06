"""GPU renderer (GLSL via moderngl): nebula background + glowing Lissajous curves.

Per frame:
  1. background: fbm noise nebula with a radial glow that breathes with the music;
  2. curve: closed Lissajous figure drawn as a soft triangle-strip line (additive);
  3. trail: feedback buffer, previous frame faded and the new curve added on top;
  4. bloom: separable Gaussian blur of the trail at half resolution;
  5. composite + tone mapping, piped as raw RGB into ffmpeg.

Shapes morph between closed figures by blending points, never by interpolating the
frequency ratio, so the curve never opens.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import moderngl
import numpy as np

W, H = 1920, 1080

SHAPES = [(3, 2), (4, 3), (5, 4), (5, 3), (2, 1), (3, 4), (5, 6), (4, 5)]
# Catppuccin Mocha accents for the curves
PALETTE = np.array(
    [
        [0xCB, 0xA6, 0xF7],
        [0x89, 0xB4, 0xFA],
        [0x94, 0xE2, 0xD5],
        [0xA6, 0xE3, 0xA1],
        [0xF9, 0xE2, 0xAF],
        [0xFA, 0xB3, 0x87],
        [0xF3, 0x8B, 0xA8],
        [0xB4, 0xBE, 0xFE],
    ],
    dtype=np.float32,
) / 255.0
# Deep nebula tones for the background, cycled slowly
NEBULA = np.array(
    [
        [0.20, 0.06, 0.36],  # violet
        [0.05, 0.09, 0.30],  # indigo
        [0.02, 0.20, 0.24],  # deep teal
        [0.30, 0.05, 0.22],  # magenta
    ],
    dtype=np.float32,
)

N_POINTS = 1200
THETA = np.linspace(0, 2 * np.pi, N_POINTS)

QUAD_VS = """
#version 330
in vec2 in_pos;
out vec2 uv;
void main() { uv = in_pos * 0.5 + 0.5; gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

BG_FS = """
#version 330
in vec2 uv; out vec4 frag;
uniform float t; uniform float level; uniform vec2 res;
uniform vec3 c0; uniform vec3 c1; uniform vec3 c2;
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p); vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), u.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), u.x), u.y);
}
float fbm(vec2 p) {
    float v = 0.0, a = 0.5; mat2 r = mat2(0.8, -0.6, 0.6, 0.8);
    for (int i = 0; i < 6; i++) { v += a * noise(p); p = r * p * 2.02; a *= 0.5; }
    return v;
}
void main() {
    vec2 p = (uv - 0.5) * vec2(res.x / res.y, 1.0);
    float r = length(p);
    vec2 q = vec2(fbm(p * 1.4 + t * 0.025), fbm(p * 1.4 - t * 0.02 + 3.1));
    float n = fbm(p * 2.0 + q * 1.9 + t * 0.015);
    vec3 col = mix(c0, c1, smoothstep(0.25, 0.85, n));
    col = mix(col, c2, smoothstep(0.5, 0.95, q.x) * 0.8);
    float glow = exp(-r * r * 2.6) * (0.75 + 0.35 * level);
    col *= glow * 2.1 + 0.06 * n;
    col *= smoothstep(1.15, 0.2, r);
    frag = vec4(col, 1.0);
}
"""

LINE_VS = """
#version 330
in vec2 in_pos; in vec3 in_color; in float in_side;
out vec3 v_color; out float v_side;
uniform vec2 res;
void main() {
    v_color = in_color; v_side = in_side;
    gl_Position = vec4(in_pos / res * 2.0 - 1.0, 0.0, 1.0);
}
"""

LINE_FS = """
#version 330
in vec3 v_color; in float v_side; out vec4 frag;
uniform float intensity;
void main() {
    float a = exp(-v_side * v_side * 3.0);
    frag = vec4(v_color * a * intensity, 1.0);
}
"""

TRAIL_FS = """
#version 330
in vec2 uv; out vec4 frag;
uniform sampler2D prev; uniform sampler2D cur; uniform float decay;
void main() { frag = max(texture(prev, uv) * decay, texture(cur, uv)); }
"""

BLUR_FS = """
#version 330
in vec2 uv; out vec4 frag;
uniform sampler2D src; uniform vec2 dir;
void main() {
    float w[5] = float[](0.2270, 0.1945, 0.1216, 0.0541, 0.0162);
    vec3 c = texture(src, uv).rgb * w[0];
    for (int i = 1; i < 5; i++) {
        c += texture(src, uv + dir * float(i)).rgb * w[i];
        c += texture(src, uv - dir * float(i)).rgb * w[i];
    }
    frag = vec4(c, 1.0);
}
"""

COMP_FS = """
#version 330
in vec2 uv; out vec4 frag;
uniform sampler2D bg; uniform sampler2D trail; uniform sampler2D bloom;
uniform float fade; uniform float bloom_k;
void main() {
    vec3 c = texture(bg, uv).rgb + texture(trail, uv).rgb + texture(bloom, uv).rgb * bloom_k;
    c = 1.0 - exp(-c * 1.15);           // soft tone mapping
    c = pow(c, vec3(0.95));
    frag = vec4(c * fade, 1.0);
}
"""


def smootherstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * x * (x * (6 * x - 15) + 10)


def palette_at(pos: np.ndarray, table: np.ndarray) -> np.ndarray:
    i = np.floor(pos).astype(int) % len(table)
    f = (pos - np.floor(pos))[:, None]
    return table[i] * (1 - f) + table[(i + 1) % len(table)] * f


def figure(t: float, bar: float) -> np.ndarray:
    idx = int(t // bar)
    local = (t - idx * bar) / bar
    a1, b1 = SHAPES[idx % len(SHAPES)]
    a2, b2 = SHAPES[(idx + 1) % len(SHAPES)]
    delta = 2 * np.pi * t / 60
    m = smootherstep((local - 0.5) / 0.5)
    p1 = np.stack([np.sin(a1 * THETA + delta), np.sin(b1 * THETA)], axis=1)
    p2 = np.stack([np.sin(a2 * THETA + delta), np.sin(b2 * THETA)], axis=1)
    return (1 - m) * p1 + m * p2


def strip(points: np.ndarray, colors: np.ndarray, width: float) -> np.ndarray:
    """Turn a closed polyline (pixel coords) into a triangle strip with a soft cross-section."""
    nxt = np.roll(points, -1, axis=0)
    prv = np.roll(points, 1, axis=0)
    tangent = nxt - prv
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-9
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    left = points + normal * width
    right = points - normal * width
    n = len(points)
    verts = np.empty((2 * n, 6), dtype=np.float32)
    verts[0::2, :2], verts[1::2, :2] = left, right
    verts[0::2, 2:5], verts[1::2, 2:5] = colors, colors
    verts[0::2, 5], verts[1::2, 5] = 1.0, -1.0
    return verts


def render_video(analysis_path: Path, out_path: Path, fps: int = 30) -> None:
    data = json.loads(analysis_path.read_text())
    duration = data["duration"]
    bar = data["chord_seconds"]
    rms = np.array(data["rms"])
    n_frames = int(duration * fps)

    ctx = moderngl.create_standalone_context(require=330)
    quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4"))

    def prog(fs):
        p = ctx.program(vertex_shader=QUAD_VS, fragment_shader=fs)
        return p, ctx.vertex_array(p, [(quad, "2f", "in_pos")])

    bg_p, bg_vao = prog(BG_FS)
    trail_p, trail_vao = prog(TRAIL_FS)
    blur_p, blur_vao = prog(BLUR_FS)
    comp_p, comp_vao = prog(COMP_FS)
    line_p = ctx.program(vertex_shader=LINE_VS, fragment_shader=LINE_FS)
    line_buf = ctx.buffer(reserve=2 * N_POINTS * 6 * 4 * 2)
    line_vao = ctx.vertex_array(line_p, [(line_buf, "2f 3f 1f", "in_pos", "in_color", "in_side")])

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

    for p in (bg_p, line_p):
        p["res"].value = (W, H)

    ff = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
            "-vf", "vflip", "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
            str(out_path),
        ],
        stdin=subprocess.PIPE,
    )  # fmt: skip

    cx, cy, base_amp = W / 2, H / 2, H * 0.36
    for k in range(n_frames):
        t = k / fps
        level = float(rms[min(int(t * data["fps"]), len(rms) - 1)])

        # 1. background nebula
        cyc = t / 40.0
        bg_p["t"].value = t
        bg_p["level"].value = level
        bg_p["c0"].value = tuple(palette_at(np.array([cyc]), NEBULA)[0])
        bg_p["c1"].value = tuple(palette_at(np.array([cyc + 1.3]), NEBULA)[0])
        bg_p["c2"].value = tuple(palette_at(np.array([cyc + 2.6]), NEBULA)[0])
        bg_fbo.use()
        bg_vao.render(moderngl.TRIANGLE_STRIP)

        # 2. curves (main stroke + thinner inner echo), additive
        pts = figure(t, bar)
        amp = base_amp * (0.86 + 0.14 * level)
        hue = t / (bar * 1.5) + np.linspace(0, 1.0, N_POINTS)
        colors = palette_at(hue, PALETTE)
        main = np.column_stack([cx + pts[:, 0] * amp * 1.35, cy + pts[:, 1] * amp])
        inner = np.column_stack([cx + pts[:, 0] * amp * 1.27, cy + pts[:, 1] * amp * 0.94])
        verts = np.concatenate([strip(main, colors, 2.6), strip(inner, palette_at(hue + 1.0, PALETTE), 1.4)])
        line_buf.write(verts.tobytes())
        cur_fbo.use()
        cur_fbo.clear(0, 0, 0, 1)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        line_p["intensity"].value = 0.55 + 0.45 * level
        line_vao.render(moderngl.TRIANGLE_STRIP, vertices=2 * N_POINTS)
        line_vao.render(moderngl.TRIANGLE_STRIP, first=2 * N_POINTS, vertices=2 * N_POINTS)
        ctx.disable(moderngl.BLEND)

        # 3. trail feedback
        (prev_tex, _), (next_tex, next_fbo) = trail[k % 2], trail[(k + 1) % 2]
        next_fbo.use()
        prev_tex.use(0)
        cur_tex.use(1)
        trail_p["prev"].value, trail_p["cur"].value = 0, 1
        trail_p["decay"].value = 0.93
        trail_vao.render(moderngl.TRIANGLE_STRIP)

        # 4. bloom (half resolution, two separable passes)
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
        fade_in = min(t / 3.0, 1.0)
        fade_out = min((duration - t) / 4.0, 1.0)
        comp_p["fade"].value = max(min(fade_in, fade_out), 0.0)
        comp_p["bloom_k"].value = 0.9
        comp_vao.render(moderngl.TRIANGLE_STRIP)
        ff.stdin.write(out_fbo.read(components=3))

    ff.stdin.close()
    ff.wait()
    ctx.release()
