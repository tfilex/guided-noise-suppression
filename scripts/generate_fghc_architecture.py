#!/usr/bin/env python3
"""Generate a publication-style FGHC architecture diagram."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT_DIR = Path(__file__).resolve().parent
PNG_PATH = OUT_DIR / "fghc_architecture.png"
PDF_PATH = OUT_DIR / "fghc_architecture.pdf"


COLORS = {
    "input": "#E7EEF8",
    "filter": "#FDECC8",
    "prior": "#FFF7DD",
    "fusion": "#FFE4E6",
    "backbone": "#DDEBFF",
    "output": "#E7F5E8",
    "new": "#F97316",
    "edge": "#2D3748",
    "shade": "#FFF7ED",
    "text": "#111827",
    "muted": "#4B5563",
}


def box(ax, x, y, w, h, text, color, fs=10.5, lw=1.1, edge=None, dashed=False):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.055",
        linewidth=lw,
        edgecolor=edge or COLORS["edge"],
        facecolor=color,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color=COLORS["text"],
        linespacing=1.15,
    )
    return patch


def text(ax, x, y, s, fs=10, ha="center", va="center", color=None, weight=None):
    ax.text(
        x,
        y,
        s,
        ha=ha,
        va=va,
        fontsize=fs,
        color=color or COLORS["text"],
        fontweight=weight,
        linespacing=1.15,
    )


def arrow(ax, x1, y1, x2, y2, color="#374151", lw=1.25, dashed=False, rad=0.0):
    patch = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=lw,
        color=color,
        linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(patch)
    return patch


def main():
    fig, ax = plt.subplots(figsize=(15.5, 6.2), dpi=230)
    ax.set_xlim(0, 14.8)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    text(ax, 0.35, 5.88, "Filter-Guided Hybrid Cascade (FGHC)", fs=15, ha="left", weight="bold")
    text(
        ax,
        0.35,
        5.55,
        "Deterministic structural prior extraction followed by lightweight neural restoration",
        fs=9.8,
        ha="left",
        color=COLORS["muted"],
    )

    # Highlight proposed region.
    shade = FancyBboxPatch(
        (3.05, 0.58),
        8.55,
        4.85,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        linewidth=1.4,
        edgecolor=COLORS["new"],
        facecolor=COLORS["shade"],
        linestyle="--",
    )
    ax.add_patch(shade)
    text(ax, 3.22, 5.22, "proposed FGHC module", fs=9.5, ha="left", color=COLORS["new"], weight="bold")

    # Main signal flow.
    box(ax, 0.45, 2.65, 1.55, 0.82, "Noisy image\n$X_{noisy}$", COLORS["input"], fs=10.5)
    box(
        ax,
        3.25,
        3.72,
        2.10,
        0.90,
        "Deterministic\npre-filter $\\mathcal{F}$\nPRBF / GGB / AD Zhang",
        COLORS["filter"],
        fs=9.2,
    )
    box(ax, 6.00, 3.72, 1.80, 0.90, "Structural prior\n$X_{filtered}$", COLORS["prior"], fs=10.0)
    box(
        ax,
        7.20,
        1.85,
        2.55,
        1.28,
        "Filter-guided\nfusion strategy\n(new comparison target)",
        COLORS["fusion"],
        fs=10.0,
        lw=1.5,
        edge=COLORS["new"],
    )
    box(ax, 10.25, 1.92, 2.05, 1.14, "Lightweight\nbackbone\nN2D / NAFNet Light", COLORS["backbone"], fs=10.0)
    box(ax, 13.00, 2.03, 1.50, 0.92, "Denoised image\n$\\hat{Y}$", COLORS["output"], fs=10.2)

    # Arrows.
    arrow(ax, 2.0, 3.05, 3.25, 4.17)
    arrow(ax, 5.35, 4.17, 6.0, 4.17)
    arrow(ax, 6.9, 3.72, 7.7, 3.13)
    arrow(ax, 2.0, 3.05, 7.2, 2.43)
    arrow(ax, 9.75, 2.49, 10.25, 2.49)
    arrow(ax, 12.3, 2.49, 13.0, 2.49)

    text(ax, 4.05, 3.27, "prior branch", fs=8.5, color=COLORS["muted"])
    text(ax, 4.95, 2.52, "noisy branch", fs=8.5, color=COLORS["muted"])

    # Fusion details panel.
    box(ax, 3.45, 0.88, 6.10, 0.88, "Integration variants evaluated in the same cascade", "#FFFFFF", fs=10.0, lw=0.9)
    variants = [
        ("Early Fusion", "$[X_{noisy} \\parallel X_{filtered}]$"),
        ("Residual", "$\\mathcal{N}_{\\theta}(X_{filtered}) + X_{noisy}$"),
        ("Attention", "$X_{noisy} \\odot M(X_{filtered})$"),
        ("SFT", "$\\gamma(X_{filtered}) \\cdot norm(X_{noisy}) + \\beta(X_{filtered})$"),
    ]
    x0 = 3.55
    for i, (name, eq) in enumerate(variants):
        bx = x0 + i * 2.42
        box(ax, bx, 0.18, 2.22, 0.52, f"{name}\n{eq}", "#FFF1F2", fs=7.2, lw=0.8, edge="#BE123C")

    # Baseline comparison references.
    box(ax, 0.45, 0.70, 2.25, 0.70, "Standalone baselines\nclassical filters / bare networks", "#F3F4F6", fs=8.7, dashed=True, edge="#6B7280")
    arrow(ax, 1.58, 2.65, 1.58, 1.40, color="#6B7280", lw=1.0, dashed=True)
    text(ax, 1.85, 1.88, "reported separately\nin Table 2", fs=8.0, ha="left", color="#6B7280")

    text(
        ax,
        10.12,
        4.75,
        "What is new here:\nfiltered structural priors are used to guide compact neural backbones\nthrough alternative fusion mechanisms under resource constraints.",
        fs=9.2,
        ha="left",
        color=COLORS["muted"],
    )

    fig.savefig(PNG_PATH, bbox_inches="tight", pad_inches=0.16)
    fig.savefig(PDF_PATH, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)
    print(f"saved {PNG_PATH}")
    print(f"saved {PDF_PATH}")


if __name__ == "__main__":
    main()
