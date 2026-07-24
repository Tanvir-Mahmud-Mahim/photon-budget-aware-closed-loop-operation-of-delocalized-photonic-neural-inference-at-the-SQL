"""Shared schematic drawing toolchain (lane/slot layout system).

Layout discipline: every panel is drawn on a fixed lane/slot grid; text
occupies reserved bands only, so nothing can collide. All vector art.
Provides card/stage/chip/arrow/mixer/antenna/waves/comb/drone/camera/
sensor/panel_title helpers plus the AUTOFIT measured-shrink pass used by
fig_schematics_q.py.
"""
import json
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon
import figstyle as fs
from figstyle import C

fs.use_style()
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)

TINT = {"blue": "#eaf2fc", "aqua": "#e6f7f0", "yellow": "#fdf3dd",
        "violet": "#edebf8", "red": "#fdecec", "gray": "#f4f3f1",
        "orange": "#fdeee6", "green": "#e9f4e9", "panel": "#fafaf8"}
EDGE = {"blue": C["blue"], "aqua": "#128a60", "yellow": "#b57b00",
        "violet": C["violet"], "red": C["red"], "gray": C["ink2"],
        "orange": C["orange"], "green": C["green"]}
SH = "#d9d8d4"     # shadow tone
AUTOFIT = []       # (ax, text_artist, box_width, box_height) for post-fit pass


# ---------------------------------------------------------------- helpers --
def card(ax, x, y, w, h, fc, ec, lw=1.0, rad=0.028, shadow=True, zorder=3):
    if shadow:
        ax.add_patch(FancyBboxPatch((x + 0.006, y - 0.010), w, h,
                     boxstyle=f"round,pad=0.004,rounding_size={rad}",
                     fc=SH, ec="none", zorder=zorder - 1, alpha=0.55))
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0.004,rounding_size={rad}",
                 fc=fc, ec=ec, lw=lw, zorder=zorder))


def stage(ax, cx, cy, w, h, title, sub=None, tint="gray", fsz=6.4, bold=True,
          accent=True, zorder=3, shadow=True):
    """Stage box centered at (cx, cy); title/sub occupy fixed interior bands
    with guaranteed clearance from the accent bar and the box edges."""
    x, y = cx - w / 2, cy - h / 2
    card(ax, x, y, w, h, TINT[tint], EDGE[tint], zorder=zorder, shadow=shadow)
    if accent:
        ax.plot([x + 0.016, x + w - 0.016], [y + h - 0.030] * 2,
                color=EDGE[tint], lw=1.8, solid_capstyle="round",
                zorder=zorder + 1, alpha=0.9)
    if sub:
        ty, sy = cy + 0.10 * h, cy - 0.24 * h
    else:
        ty, sy = cy - (0.05 * h if accent else 0.0), None
    t = ax.text(cx, ty, title, ha="center", va="center", fontsize=fsz,
                fontweight="bold" if bold else "normal", color=C["ink"],
                zorder=zorder + 2)
    AUTOFIT.append((ax, t, w, h))
    if sub:
        s = ax.text(cx, sy, sub, ha="center", va="center",
                    fontsize=fsz - 1.2, color=C["ink2"], zorder=zorder + 2)
        AUTOFIT.append((ax, s, w, h))


def chip(ax, cx, cy, text, tint="red", fsz=5.6, zorder=6, ec=None):
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fsz,
            color=EDGE[tint] if ec is None else ec, zorder=zorder,
            bbox=dict(boxstyle="round,pad=0.42", fc="white",
                      ec=EDGE[tint] if ec is None else ec, lw=0.7))


def arrow(ax, p1, p2, color=None, lw=1.2, style="-|>", ls="-", ms=8,
          zorder=5, rad=0.0, shrinkA=2, shrinkB=2):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=ms,
                 color=color or C["ink2"], lw=lw, linestyle=ls, zorder=zorder,
                 shrinkA=shrinkA, shrinkB=shrinkB,
                 connectionstyle=f"arc3,rad={rad}"))


def mixer(ax, x, y, r=0.045, color=None, lw=1.3, zorder=6):
    col = color or C["violet"]
    ax.add_patch(Circle((x + 0.004, y - 0.007), r, fc=SH, ec="none",
                 zorder=zorder - 1, alpha=0.5))
    ax.add_patch(Circle((x, y), r, fc="white", ec=col, lw=lw, zorder=zorder))
    d = r * 0.5
    ax.plot([x - d, x + d], [y - d, y + d], color=col, lw=lw * 0.85, zorder=zorder + 1)
    ax.plot([x - d, x + d], [y + d, y - d], color=col, lw=lw * 0.85, zorder=zorder + 1)


def antenna(ax, x, y, s=0.10, color=None, lw=1.3, zorder=6):
    col = color or C["ink"]
    ax.plot([x, x], [y, y + s * 0.62], color=col, lw=lw, zorder=zorder)
    ax.add_patch(Polygon([(x - s * 0.42, y + s), (x + s * 0.42, y + s),
                          (x, y + s * 0.40)], closed=True, fill=False,
                 ec=col, lw=lw, zorder=zorder))


def waves(ax, x, y, n=3, r0=0.05, dr=0.036, color=None, th1=25, th2=155,
          lw=1.0, zorder=5):
    col = color or C["blue"]
    th = np.linspace(math.radians(th1), math.radians(th2), 48)
    for i in range(n):
        r = r0 + i * dr
        ax.plot(x + r * np.cos(th), y + r * np.sin(th), color=col, lw=lw,
                alpha=0.9 - 0.22 * i, zorder=zorder)


def comb(ax, x, y, w, h, n, color, seed=3, zorder=6):
    rng = np.random.RandomState(seed)
    xs = np.linspace(x, x + w, n)
    hs = h * (0.40 + 0.60 * rng.rand(n))
    for xi, hi in zip(xs, hs):
        ax.plot([xi, xi], [y, y + hi], color=color, lw=1.0, zorder=zorder,
                solid_capstyle="round")
    ax.plot([x - 0.010, x + w + 0.010], [y, y], color=C["ink2"], lw=0.7,
            zorder=zorder)


def drone(ax, x, y, s=0.10, color=None, zorder=7):
    col = color or C["ink2"]
    ax.add_patch(FancyBboxPatch((x - s * 0.34, y - s * 0.20), s * 0.68, s * 0.38,
                 boxstyle="round,pad=0.003,rounding_size=0.015",
                 fc="white", ec=col, lw=1.0, zorder=zorder))
    for dx in (-s * 0.52, s * 0.52):
        ax.plot([x + dx * 0.62, x + dx], [y + s * 0.14, y + s * 0.36],
                color=col, lw=1.0, zorder=zorder)
        ax.plot([x + dx - s * 0.26, x + dx + s * 0.26], [y + s * 0.36] * 2,
                color=col, lw=1.3, zorder=zorder, solid_capstyle="round")


def camera(ax, x, y, s=0.10, color=None, zorder=7):
    col = color or C["ink2"]
    ax.add_patch(FancyBboxPatch((x - s * 0.46, y - s * 0.32), s * 0.92, s * 0.62,
                 boxstyle="round,pad=0.003,rounding_size=0.015",
                 fc="white", ec=col, lw=1.0, zorder=zorder))
    ax.add_patch(Circle((x, y - s * 0.01), s * 0.17, fill=False, ec=col,
                 lw=1.0, zorder=zorder + 1))
    ax.add_patch(FancyBboxPatch((x - s * 0.16, y + s * 0.30), s * 0.32, s * 0.13,
                 boxstyle="round,pad=0.002,rounding_size=0.008",
                 fc="white", ec=col, lw=0.9, zorder=zorder))


def sensor(ax, x, y, s=0.10, color=None, zorder=7):
    col = color or C["ink2"]
    ax.add_patch(FancyBboxPatch((x - s * 0.38, y - s * 0.38), s * 0.76, s * 0.76,
                 boxstyle="round,pad=0.003,rounding_size=0.015",
                 fc="white", ec=col, lw=1.0, zorder=zorder))
    for i in range(3):
        yy = y - s * 0.18 + i * s * 0.18
        ax.plot([x - s * 0.20, x + s * 0.20], [yy, yy], color=col, lw=0.9,
                zorder=zorder + 1)


def panel_title(ax, text, color=None):
    ax.text(0.53, 0.955, text, ha="center", va="center", fontsize=7.0,
            fontweight="bold", color=color or C["ink"])


def autofit(fig, margin=0.90):
    """Shrink registered box texts until they fit inside their boxes."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for ax, artist, w, h in AUTOFIT:
        for _ in range(8):
            bb = artist.get_window_extent(renderer=r)
            inv = ax.transData.inverted()
            p0 = inv.transform((bb.x0, bb.y0))
            p1 = inv.transform((bb.x1, bb.y1))
            bw, bh = abs(p1[0] - p0[0]), abs(p1[1] - p0[1])
            if bw <= w * margin and bh <= h * 0.55:
                break
            artist.set_fontsize(artist.get_fontsize() * 0.93)
