#!/usr/bin/env python3
"""Generate a publication-style NAFNet Light architecture diagram.

The figure is intentionally self-contained and does not require PlotNeuralNet
or a LaTeX installation. It produces PNG and PDF files in this directory.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


OUT_DIR = Path(__file__).resolve().parent
PNG_PATH = OUT_DIR / "nafnet_light_architecture.png"
PDF_PATH = OUT_DIR / "nafnet_light_architecture.pdf"


COLORS = {
    "input": "#E7EEF8",
    "conv": "#D8E7C3",
    "enc": "#BFD7EA",
    "down": "#F6D7A7",
    "middle": "#D8C7F2",
    "up": "#F5C6CB",
    "dec": "#CFE8D5",
    "out": "#EDEDED",
    "inset": "#FFFFFF",
    "edge": "#2D3748",
    "skip": "#6B7280",
}


def add_box(ax, x, y, w, h, label, color, fontsize=8.5, lw=1.0):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=lw,
        edgecolor=COLORS["edge"],
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#111827",
        linespacing=1.15,
    )
    return patch


def add_volume(ax, x, y, w, h, depth, label, color, fontsize=8.5):
    """Draw a light PlotNeuralNet-like 3D block."""
    front = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.0,
        edgecolor=COLORS["edge"],
        facecolor=color,
    )
    top = Polygon(
        [(x, y + h), (x + depth, y + h + depth), (x + w + depth, y + h + depth), (x + w, y + h)],
        closed=True,
        linewidth=0.8,
        edgecolor=COLORS["edge"],
        facecolor=lighten(color, 0.16),
    )
    side = Polygon(
        [(x + w, y), (x + w + depth, y + depth), (x + w + depth, y + h + depth), (x + w, y + h)],
        closed=True,
        linewidth=0.8,
        edgecolor=COLORS["edge"],
        facecolor=lighten(color, -0.05),
    )
    ax.add_patch(top)
    ax.add_patch(side)
    ax.add_patch(front)
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#111827",
        linespacing=1.15,
    )
    return front


def lighten(hex_color, amount):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16) / 255
    g = int(hex_color[2:4], 16) / 255
    b = int(hex_color[4:6], 16) / 255
    if amount >= 0:
        r = r + (1 - r) * amount
        g = g + (1 - g) * amount
        b = b + (1 - b) * amount
    else:
        r = r * (1 + amount)
        g = g * (1 + amount)
        b = b * (1 + amount)
    return (r, g, b)


def arrow(ax, x1, y1, x2, y2, color="#374151", lw=1.1, style="-|>", rad=0.0, mutation=9):
    patch = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle=style,
        mutation_scale=mutation,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(patch)
    return patch


def add_text(ax, x, y, text, fontsize=8.0, ha="center", va="center", color="#111827", weight=None):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize, color=color, fontweight=weight, linespacing=1.15)


def draw_main_architecture(ax):
    y = 3.15
    h = 0.78
    w = 1.15
    gap = 0.52
    x = 0.3

    blocks = []
    specs = [
        ("Input\nRGB\n3 x H x W", COLORS["input"], "box"),
        ("Intro Conv\n3x3\nC=16", COLORS["conv"], "vol"),
        ("Enc 1\nNAFBlock x1\nC=16", COLORS["enc"], "vol"),
        ("Down\n2x2 / s=2", COLORS["down"], "box"),
        ("Enc 2\nNAFBlock x1\nC=32", COLORS["enc"], "vol"),
        ("Down\n2x2 / s=2", COLORS["down"], "box"),
        ("Enc 3\nNAFBlock x1\nC=64", COLORS["enc"], "vol"),
        ("Down\n2x2 / s=2", COLORS["down"], "box"),
        ("Enc 4\nNAFBlock x1\nC=128", COLORS["enc"], "vol"),
        ("Down\n2x2 / s=2", COLORS["down"], "box"),
        ("Middle\nNAFBlock x1\nC=256", COLORS["middle"], "vol"),
        ("Up\nPixelShuffle", COLORS["up"], "box"),
        ("Dec 4\nNAFBlock x1\nC=128", COLORS["dec"], "vol"),
        ("Up\nPixelShuffle", COLORS["up"], "box"),
        ("Dec 3\nNAFBlock x1\nC=64", COLORS["dec"], "vol"),
        ("Up\nPixelShuffle", COLORS["up"], "box"),
        ("Dec 2\nNAFBlock x1\nC=32", COLORS["dec"], "vol"),
        ("Up\nPixelShuffle", COLORS["up"], "box"),
        ("Dec 1\nNAFBlock x1\nC=16", COLORS["dec"], "vol"),
        ("Outro Conv\n3x3", COLORS["conv"], "box"),
        ("Output\nRGB\n3 x H x W", COLORS["out"], "box"),
    ]

    for idx, (label, color, kind) in enumerate(specs):
        xi = x + idx * (w + gap)
        if kind == "vol":
            add_volume(ax, xi, y, w, h, 0.12, label, color)
        else:
            add_box(ax, xi, y, w, h, label, color)
        blocks.append((xi, y, w, h))

    for (x1, y1, w1, h1), (x2, y2, _w2, _h2) in zip(blocks[:-1], blocks[1:]):
        arrow(ax, x1 + w1 + 0.06, y1 + h1 / 2, x2 - 0.06, y2 + h1 / 2)

    # Skip connections from encoder outputs to matching decoder blocks.
    skip_pairs = [(2, 18), (4, 16), (6, 14), (8, 12)]
    skip_heights = [5.62, 5.27, 4.92, 4.57]
    for (src, dst), sy in zip(skip_pairs, skip_heights):
        xs, ys, ws, hs = blocks[src]
        xd, yd, _wd, hd = blocks[dst]
        arrow(ax, xs + ws / 2, ys + hs + 0.04, xs + ws / 2, sy, color=COLORS["skip"], lw=0.9, style="-", mutation=1)
        arrow(ax, xs + ws / 2, sy, xd + w / 2, sy, color=COLORS["skip"], lw=0.9, style="-", mutation=1)
        arrow(ax, xd + w / 2, sy, xd + w / 2, yd + hd + 0.04, color=COLORS["skip"], lw=0.9, style="-|>", mutation=7)

    add_text(ax, 18.2, 5.88, "skip connections", fontsize=7.5, color=COLORS["skip"])
    add_text(
        ax,
        17.2,
        2.55,
        "Backbone used by bare NAFNet Light and by FGHC variants;\nfilter-guided fusion modules are attached outside this backbone.",
        fontsize=8.3,
        color="#374151",
    )


def draw_nafblock_inset(ax):
    x0, y0 = 7.3, 0.43
    panel_w, panel_h = 14.0, 1.75
    panel = FancyBboxPatch(
        (x0, y0),
        panel_w,
        panel_h,
        boxstyle="round,pad=0.04,rounding_size=0.06",
        linewidth=1.0,
        edgecolor="#4B5563",
        facecolor=COLORS["inset"],
    )
    ax.add_patch(panel)
    add_text(ax, x0 + 0.35, y0 + panel_h - 0.28, "NAFBlock internals", fontsize=9.0, ha="left", weight="bold")

    y1 = y0 + 1.05
    y2 = y0 + 0.46
    bw = 1.25
    bh = 0.33
    labels_top = ["LayerNorm2d", "DW Conv", "SimpleGate", "SCA", "PW Conv", "Residual scale"]
    labels_bot = ["LayerNorm2d", "FFN Conv", "SimpleGate", "PW Conv", "Residual scale"]
    xs_top = [x0 + 1.05 + i * 1.85 for i in range(len(labels_top))]
    xs_bot = [x0 + 1.95 + i * 1.95 for i in range(len(labels_bot))]

    for i, label in enumerate(labels_top):
        add_box(ax, xs_top[i], y1, bw, bh, label, "#EEF2FF", fontsize=7.0, lw=0.8)
        if i > 0:
            arrow(ax, xs_top[i - 1] + bw, y1 + bh / 2, xs_top[i], y1 + bh / 2, lw=0.8, mutation=7)

    for i, label in enumerate(labels_bot):
        add_box(ax, xs_bot[i], y2, bw, bh, label, "#F0FDF4", fontsize=7.0, lw=0.8)
        if i > 0:
            arrow(ax, xs_bot[i - 1] + bw, y2 + bh / 2, xs_bot[i], y2 + bh / 2, lw=0.8, mutation=7)

    add_text(ax, x0 + 0.45, y1 + bh / 2, "branch 1", fontsize=7.0, ha="left", color="#4B5563")
    add_text(ax, x0 + 0.45, y2 + bh / 2, "branch 2", fontsize=7.0, ha="left", color="#4B5563")


def draw_legend(ax):
    items = [
        ("convolution", COLORS["conv"]),
        ("encoder", COLORS["enc"]),
        ("downsample", COLORS["down"]),
        ("middle", COLORS["middle"]),
        ("upsample", COLORS["up"]),
        ("decoder", COLORS["dec"]),
    ]
    x0, y0 = 0.45, 0.55
    for i, (label, color) in enumerate(items):
        x = x0 + i * 1.85
        add_box(ax, x, y0, 0.26, 0.18, "", color, lw=0.6)
        add_text(ax, x + 0.36, y0 + 0.09, label, fontsize=6.8, ha="left", color="#374151")


def main():
    fig, ax = plt.subplots(figsize=(20, 6.4), dpi=220)
    ax.set_xlim(0, 35.5)
    ax.set_ylim(0, 6.35)
    ax.axis("off")

    add_text(ax, 0.35, 6.08, "NAFNet Light backbone architecture", fontsize=13.5, ha="left", weight="bold")
    add_text(
        ax,
        0.35,
        5.75,
        "Base width = 16; encoder blocks = (1, 1, 1, 1); middle blocks = 1; decoder blocks = (1, 1, 1, 1)",
        fontsize=9.2,
        ha="left",
        color="#374151",
    )

    draw_main_architecture(ax)
    draw_nafblock_inset(ax)
    draw_legend(ax)

    fig.savefig(PNG_PATH, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(PDF_PATH, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"saved {PNG_PATH}")
    print(f"saved {PDF_PATH}")


if __name__ == "__main__":
    main()
