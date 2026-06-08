#!/usr/bin/env python3
"""Generate Figure 2: three representative examples with edge- and quality-oriented FGHC variants."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ARTICLE_DIR = PROJECT_ROOT / "paper"
FIG_DIR = ARTICLE_DIR / "fig"
STAT_SCRIPT = PROJECT_ROOT / "src" / "statistical_evaluation.py"
OUT = FIG_DIR / "figure2_three_examples_variant_b.png"

EXAMPLES = [
    {
        "tag": "A",
        "title": "TwoPhoton BPAE G, avg4",
        "mod_dir_contains": "TwoPhoton_BPAE_G",
        "noise_level": "avg4",
        "fov": "12",
        "image_id": "FOV12_WL780_P05_HV130_0110005",
    },
    {
        "tag": "B",
        "title": "Confocal BPAE G, avg4",
        "mod_dir_contains": "Confocal_BPAE_G",
        "noise_level": "avg4",
        "fov": "12",
        "image_id": "FOV12_HV110_P0500510000",
    },
    {
        "tag": "C",
        "title": "Confocal FISH, avg4",
        "mod_dir_contains": "Confocal_FISH",
        "noise_level": "avg4",
        "fov": "1",
        "image_id": "FOV1_HV140_P100510000",
    },
]

METHOD_IDS = [
    "BM3D",
    "Bare_N2D",
    "N2D_GGB_SFT_Modulation",
    "Bare_NAFNet",
    "NAFNet_AD_Zhang_SFT_Modulation",
]

METHOD_LABELS = {
    "BM3D": "BM3D",
    "Bare_N2D": "Bare N2D",
    "N2D_GGB_SFT_Modulation": "GGB + N2D SFT\n(edge FGHC)",
    "Bare_NAFNet": "Bare NAFNet\nLight",
    "NAFNet_AD_Zhang_SFT_Modulation": "AD Zhang + NAFNet\nSFT (quality FGHC)",
}

COL_LABELS = [
    "Full reference\nwith ROI",
    "Noisy\nROI",
    "Reference\nROI",
    METHOD_LABELS["BM3D"],
    METHOD_LABELS["Bare_N2D"],
    METHOD_LABELS["N2D_GGB_SFT_Modulation"],
    METHOD_LABELS["Bare_NAFNet"],
    METHOD_LABELS["NAFNet_AD_Zhang_SFT_Modulation"],
]


def load_stat_module():
    spec = importlib.util.spec_from_file_location("statistical_evaluation", STAT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {STAT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["statistical_evaluation"] = module
    spec.loader.exec_module(module)
    return module


def to_gray_float(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)


def to_rgb_float(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def error_map(gt: np.ndarray, img: np.ndarray) -> np.ndarray:
    return np.abs(to_gray_float(img) - to_gray_float(gt))


def crop(img: np.ndarray, roi):
    x, y, w, h = roi
    return img[y : y + h, x : x + w]


def metric_short(gt: np.ndarray, img: np.ndarray) -> str:
    p = psnr(gt, img, data_range=255)
    s = ssim(gt, img, data_range=255, channel_axis=2, win_size=7)
    return f"{p:.1f} dB / {s:.3f}"


def choose_roi(gt: np.ndarray, outputs: dict[str, np.ndarray], size: int = 128, stride: int = 16):
    gt_g = to_gray_float(gt)
    grad = cv2.magnitude(cv2.Sobel(gt_g, cv2.CV_32F, 1, 0), cv2.Sobel(gt_g, cv2.CV_32F, 0, 1))
    errors = {name: error_map(gt, img) for name, img in outputs.items()}
    baseline_names = ["BM3D", "Bare_N2D", "Bare_NAFNet"]
    proposed_names = ["N2D_GGB_SFT_Modulation", "NAFNet_AD_Zhang_SFT_Modulation"]

    h, w = gt_g.shape
    margin = max(8, size // 10)
    best = None
    for y in range(margin, max(margin + 1, h - size - margin + 1), stride):
        for x in range(margin, max(margin + 1, w - size - margin + 1), stride):
            ys = slice(y, y + size)
            xs = slice(x, x + size)
            baseline_error = np.mean([errors[name][ys, xs].mean() for name in baseline_names])
            proposed_error = np.mean([errors[name][ys, xs].mean() for name in proposed_names])
            detail = grad[ys, xs].mean()
            contrast = gt_g[ys, xs].std()
            score = (baseline_error - proposed_error) + 0.025 * detail + 0.015 * contrast
            candidate = (score, x, y, baseline_error, proposed_error, detail)
            if best is None or candidate > best:
                best = candidate
    if best is None:
        return (w // 2 - size // 2, h // 2 - size // 2, size, size)
    return (int(best[1]), int(best[2]), size, size)


def find_pair(stat, example):
    pairs = list(
        stat.parse_aimd_extracted(
            stat.CONFIG["data"]["path"],
            mode="test",
            mod_dir_contains=example["mod_dir_contains"],
            noise_level=example["noise_level"],
            fovs={example["fov"]},
        )
    )
    pair = next((p for p in pairs if p[2] == example["image_id"]), None)
    if pair is None:
        raise RuntimeError(f"Could not find {example['image_id']} for {example['title']}")
    return pair


def render_method(stat, specs, method_id: str, noisy: np.ndarray, device: str):
    spec = specs[method_id]
    if "model" in spec:
        model = stat.load_model(spec, device)
        result, _latency, _vram = stat.evaluate_torch_model(model, noisy, spec, device)
        del model
        stat.hard_cleanup()
        return result
    result, _latency, _vram = stat.evaluate_classical(spec["classical_func"], noisy)
    stat.hard_cleanup()
    return result


def set_axis_style(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#666666")


def annotate_metric(ax, text: str):
    ax.text(
        0.03,
        0.04,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.8,
        color="white",
        bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=1.5),
    )


def main(args):
    stat = load_stat_module()
    stat.FORCE_GGB_FALLBACK = True
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    specs = stat.method_specs()

    example_data = []
    all_errors = []
    for example in EXAMPLES:
        noisy_path, gt_path, label = find_pair(stat, example)
        noisy = stat.center_crop(cv2.imread(noisy_path), stat.CONFIG["eval"]["crop_size"])
        gt = stat.center_crop(cv2.imread(gt_path), stat.CONFIG["eval"]["crop_size"])
        if noisy is None or gt is None:
            raise RuntimeError(f"Unreadable pair for {label}")
        print(f"[*] Example {example['tag']}: {example['title']} / {label}")
        outputs = {}
        for method_id in METHOD_IDS:
            print(f"    rendering {method_id} on {device}")
            outputs[method_id] = render_method(stat, specs, method_id, noisy, device)
        roi = choose_roi(gt, outputs, size=args.roi_size)
        print(f"    ROI x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
        for img in [noisy, *outputs.values()]:
            all_errors.append(crop(error_map(gt, img), roi).ravel())
        example_data.append({"example": example, "label": label, "noisy": noisy, "gt": gt, "outputs": outputs, "roi": roi})

    err_vmax = float(np.percentile(np.concatenate(all_errors), 98))
    err_vmax = max(8.0, min(err_vmax, 90.0))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
    n_examples = len(example_data)
    fig = plt.figure(figsize=(14.4, 9.6))
    gs = GridSpec(
        nrows=n_examples * 2,
        ncols=10,
        figure=fig,
        # Columns 0--7 contain image panels; column 8 is a spacer;
        # column 9 is reserved for the shared error-map colour bar.
        width_ratios=[1.25, 1, 1, 1, 1, 1, 1, 1, 0.18, 0.10],
        height_ratios=[1, 1] * n_examples,
        wspace=0.07,
        hspace=0.16,
    )

    axes_for_cbar = []
    im_err = None
    for row_pair, item in enumerate(example_data):
        r0 = row_pair * 2
        r1 = r0 + 1
        example = item["example"]
        noisy = item["noisy"]
        gt = item["gt"]
        outputs = item["outputs"]
        roi = item["roi"]
        x, y, w, h = roi

        full_ax = fig.add_subplot(gs[r0:r1 + 1, 0])
        full_ax.imshow(to_rgb_float(gt))
        full_ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="#ffd23f", linewidth=1.8))
        full_ax.set_title(COL_LABELS[0] if row_pair == 0 else "", fontsize=7, pad=4)
        full_ax.set_ylabel(
            f"{example['tag']}. {example['title']}\nROI crops / absolute error maps",
            fontsize=7,
            rotation=90,
            labelpad=20,
        )
        set_axis_style(full_ax)

        panels = [
            ("Noisy", noisy, None),
            ("Reference", gt, None),
            ("BM3D", outputs["BM3D"], "BM3D"),
            ("Bare N2D", outputs["Bare_N2D"], "Bare_N2D"),
            ("GGB + N2D SFT", outputs["N2D_GGB_SFT_Modulation"], "N2D_GGB_SFT_Modulation"),
            ("Bare NAFNet", outputs["Bare_NAFNet"], "Bare_NAFNet"),
            ("AD + NAFNet SFT", outputs["NAFNet_AD_Zhang_SFT_Modulation"], "NAFNet_AD_Zhang_SFT_Modulation"),
        ]

        for j, (_short, img, method_id) in enumerate(panels, start=1):
            crop_ax = fig.add_subplot(gs[r0, j])
            crop_ax.imshow(to_rgb_float(crop(img, roi)), interpolation="nearest")
            if row_pair == 0:
                crop_ax.set_title(COL_LABELS[j], fontsize=7, pad=4)
            if method_id is not None:
                annotate_metric(crop_ax, metric_short(gt, img))
            elif _short == "Noisy":
                annotate_metric(crop_ax, metric_short(gt, img))
            set_axis_style(crop_ax)

            err_ax = fig.add_subplot(gs[r1, j])
            if _short == "Reference":
                im_err = err_ax.imshow(np.zeros((h, w)), cmap="magma", vmin=0, vmax=err_vmax)
                err_ax.text(0.5, 0.5, "zero\nerror", ha="center", va="center", color="white", fontsize=6, transform=err_ax.transAxes)
            else:
                im_err = err_ax.imshow(crop(error_map(gt, img), roi), cmap="magma", vmin=0, vmax=err_vmax, interpolation="nearest")
            set_axis_style(err_ax)
            axes_for_cbar.append(err_ax)

    # Visual group cues: edge and quality FGHC columns get subtle colored labels at top.
    fig.text(0.57, 0.972, "edge-oriented comparison", ha="center", va="center", fontsize=8, color="#185a9d")
    fig.text(0.80, 0.972, "quality-oriented comparison", ha="center", va="center", fontsize=8, color="#8a3ffc")

    cax = fig.add_subplot(gs[:, 9])
    cbar = fig.colorbar(im_err, cax=cax)
    cbar.ax.set_ylabel("absolute error, intensity levels", fontsize=7, labelpad=8)
    cbar.ax.tick_params(labelsize=6, pad=2)

    fig.suptitle(
        "Representative qualitative comparison across three FMD examples",
        fontsize=10,
        y=0.995,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"[*] Wrote {args.output}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--roi-size", type=int, default=128)
    parser.add_argument("--dpi", type=int, default=350)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
