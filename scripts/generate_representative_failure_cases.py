#!/usr/bin/env python3
"""Generate representative failure/limitation cases with ROI crops and error maps."""

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
OUT = FIG_DIR / "failure_case_representative_roi_error.png"

CASES = [
    {
        "tag": "A",
        "title": "Standalone deterministic filtering leaves residual photon noise",
        "mod_dir_contains": "TwoPhoton_BPAE_G",
        "noise_level": "avg4",
        "fov": "12",
        "image_id": "FOV12_WL780_P05_HV130_0110005",
        "failure": "PRBF_Standalone",
        "comparator": "N2D_GGB_SFT_Modulation",
    },
    {
        "tag": "B",
        "title": "Standalone compact neural denoising can soften fine structures",
        "mod_dir_contains": "Confocal_BPAE_G",
        "noise_level": "avg4",
        "fov": "12",
        "image_id": "FOV12_HV110_P0500510000",
        "failure": "Bare_N2D",
        "comparator": "N2D_GGB_SFT_Modulation",
    },
    {
        "tag": "C",
        "title": "Residual fusion can reintroduce noisy high-frequency components",
        "mod_dir_contains": "Confocal_FISH",
        "noise_level": "avg4",
        "fov": "1",
        "image_id": "FOV1_HV140_P100510000",
        "failure": "N2D_PRBF_Residual",
        "comparator": "Bare_N2D",
    },
    {
        "tag": "D",
        "title": "More complex classical processing is not always safer",
        "mod_dir_contains": "TwoPhoton_MICE",
        "noise_level": "avg4",
        "fov": "12",
        "image_id": "FOV12_WL780_P05_HV130_0310005",
        "failure": "BM3D_Improved",
        "comparator": "BM3D",
    },
]

METHOD_LABELS = {
    "PRBF_Standalone": "PRBF\nstandalone",
    "N2D_GGB_SFT_Modulation": "GGB + N2D SFT\nFGHC comparator",
    "Bare_N2D": "Bare N2D",
    "N2D_PRBF_Residual": "PRBF + N2D\nResidual",
    "BM3D_Improved": "Improved\nBM3D",
    "BM3D": "BM3D",
}


def load_stat_module():
    spec = importlib.util.spec_from_file_location("statistical_evaluation", STAT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {STAT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["statistical_evaluation"] = module
    spec.loader.exec_module(module)
    return module


def gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


def crop(img: np.ndarray, roi):
    x, y, w, h = roi
    return img[y : y + h, x : x + w]


def error_map(gt: np.ndarray, img: np.ndarray) -> np.ndarray:
    return np.abs(gray(img) - gray(gt))


def metric_short(gt: np.ndarray, img: np.ndarray) -> str:
    p = psnr(gt, img, data_range=255)
    s = ssim(gt, img, data_range=255, channel_axis=2, win_size=7)
    return f"{p:.1f} dB / {s:.3f}"


def choose_failure_roi(gt: np.ndarray, failure_img: np.ndarray, comparator_img: np.ndarray, size: int = 128, stride: int = 16):
    gt_g = gray(gt)
    failure_e = error_map(gt, failure_img)
    comparator_e = error_map(gt, comparator_img)
    grad = cv2.magnitude(cv2.Sobel(gt_g, cv2.CV_32F, 1, 0), cv2.Sobel(gt_g, cv2.CV_32F, 0, 1))
    h, w = gt_g.shape
    margin = max(8, size // 10)
    best = None
    for y in range(margin, max(margin + 1, h - size - margin + 1), stride):
        for x in range(margin, max(margin + 1, w - size - margin + 1), stride):
            ys = slice(y, y + size)
            xs = slice(x, x + size)
            diff = failure_e[ys, xs].mean() - comparator_e[ys, xs].mean()
            detail = grad[ys, xs].mean()
            contrast = gt_g[ys, xs].std()
            score = diff + 0.025 * detail + 0.015 * contrast
            candidate = (score, x, y, diff, detail, contrast)
            if best is None or candidate > best:
                best = candidate
    if best is None:
        return (w // 2 - size // 2, h // 2 - size // 2, size, size)
    return (int(best[1]), int(best[2]), size, size)


def find_pair(stat, case):
    pairs = list(
        stat.parse_aimd_extracted(
            stat.CONFIG["data"]["path"],
            mode="test",
            mod_dir_contains=case["mod_dir_contains"],
            noise_level=case["noise_level"],
            fovs={case["fov"]},
        )
    )
    pair = next((p for p in pairs if p[2] == case["image_id"]), None)
    if pair is None:
        raise RuntimeError(f"Could not find {case['image_id']}")
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


def add_metric(ax, text: str):
    ax.text(
        0.03,
        0.04,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6,
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

    rendered = []
    all_error_values = []
    for case in CASES:
        noisy_path, gt_path, label = find_pair(stat, case)
        noisy = stat.center_crop(cv2.imread(noisy_path), stat.CONFIG["eval"]["crop_size"])
        gt = stat.center_crop(cv2.imread(gt_path), stat.CONFIG["eval"]["crop_size"])
        if noisy is None or gt is None:
            raise RuntimeError(f"Unreadable pair for {label}")
        print(f"[*] Case {case['tag']}: {case['title']} / {label}")
        failure_img = render_method(stat, specs, case["failure"], noisy, device)
        comparator_img = render_method(stat, specs, case["comparator"], noisy, device)
        roi = choose_failure_roi(gt, failure_img, comparator_img, size=args.roi_size)
        print(f"    ROI x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
        for img in [noisy, failure_img, comparator_img]:
            all_error_values.append(crop(error_map(gt, img), roi).ravel())
        rendered.append({
            "case": case,
            "label": label,
            "noisy": noisy,
            "gt": gt,
            "failure": failure_img,
            "comparator": comparator_img,
            "roi": roi,
        })

    err_vmax = float(np.percentile(np.concatenate(all_error_values), 98))
    err_vmax = max(8.0, min(err_vmax, 90.0))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
    fig = plt.figure(figsize=(13.8, 8.8))
    gs = GridSpec(
        nrows=len(rendered),
        ncols=9,
        figure=fig,
        width_ratios=[1.18, 1, 1, 1, 1, 1, 1, 0.18, 0.10],
        wspace=0.07,
        hspace=0.22,
    )

    col_titles = [
        "Reference\nwith ROI",
        "Noisy\nROI",
        "Reference\nROI",
        "Limitation\ncase",
        "Comparator",
        "Limitation\nerror",
        "Comparator\nerror",
    ]
    axes_for_cbar = []
    im_err = None
    for row, item in enumerate(rendered):
        case = item["case"]
        noisy = item["noisy"]
        gt = item["gt"]
        failure_img = item["failure"]
        comparator_img = item["comparator"]
        roi = item["roi"]
        x, y, w, h = roi

        panels = [
            ("full", gt, None),
            ("noisy", noisy, metric_short(gt, noisy)),
            ("reference", gt, None),
            (case["failure"], failure_img, metric_short(gt, failure_img)),
            (case["comparator"], comparator_img, metric_short(gt, comparator_img)),
            ("failure_error", failure_img, None),
            ("comparator_error", comparator_img, None),
        ]

        for col, (kind, img, metric) in enumerate(panels):
            ax = fig.add_subplot(gs[row, col])
            if kind == "full":
                ax.imshow(gray(img), cmap="gray", vmin=0, vmax=255)
                ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="#ffd23f", linewidth=1.8))
            elif kind.endswith("_error"):
                im_err = ax.imshow(crop(error_map(gt, img), roi), cmap="magma", vmin=0, vmax=err_vmax, interpolation="nearest")
                axes_for_cbar.append(ax)
            else:
                ax.imshow(crop(gray(img), roi), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
                if metric:
                    add_metric(ax, metric)
            if row == 0:
                title = col_titles[col]
                if col == 3:
                    title += f"\n{METHOD_LABELS[case['failure']]}"
                elif col == 4:
                    title += f"\n{METHOD_LABELS[case['comparator']]}"
                ax.set_title(title, fontsize=7, pad=4)
            elif col in (3, 4):
                ax.set_title(METHOD_LABELS[case["failure" if col == 3 else "comparator"]], fontsize=6.5, pad=3)
            if col == 0:
                ax.set_ylabel(f"{case['tag']}. {case['title']}", fontsize=7, rotation=90, labelpad=18)
            set_axis_style(ax)

    cax = fig.add_subplot(gs[:, 8])
    cbar = fig.colorbar(im_err, cax=cax)
    cbar.ax.set_ylabel("absolute error, intensity levels", fontsize=7, labelpad=8)
    cbar.ax.tick_params(labelsize=6, pad=2)

    fig.suptitle("Representative failure and limitation cases", fontsize=10, y=0.995)
    args.output.parent.mkdir(parents=True, exist_ok=True)
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
