#!/usr/bin/env python3
"""Per-image statistical evaluation for the denoising experiments.

This script recomputes test-set metrics for saved checkpoints and exports:
- per-image metrics
- mean +/- std summaries
- paired Wilcoxon tests for key comparisons
- a compact LaTeX table for the article

Run from either the project root or the 8/ directory.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path


REQUIRED = {
    "numpy": "numpy",
    "cv2": "opencv-contrib-python",
    "skimage": "scikit-image",
    "torch": "torch",
    "scipy": "scipy",
    "bm3d": "bm3d",
}


def check_dependencies() -> None:
    missing = []
    for module, package in REQUIRED.items():
        if importlib.util.find_spec(module) is None:
            missing.append(package)
    if missing:
        print("[!] Missing Python packages:")
        for package in missing:
            print(f"    - {package}")
        print("\nInstall the non-Torch dependencies with:")
        print("    python3 -m pip install opencv-contrib-python scikit-image scipy bm3d pandas openpyxl tqdm")
        print("\nInstall torch separately for your CPU/CUDA setup from https://pytorch.org/get-started/locally/")
        sys.exit(1)


check_dependencies()

import bm3d  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from skimage.metrics import peak_signal_noise_ratio as psnr  # noqa: E402
from skimage.metrics import structural_similarity as ssim  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


CONFIG = {
    "data": {
        "path": PROJECT_ROOT / "datasets" / "AIMD",
        "test_limit": 50,
        "splits_file": SCRIPT_DIR / "fixed_splits.json",
    },
    "eval": {
        "crop_size": 512,
        "ssim_win_size": 7,
        "data_range": 255,
    },
    "paths": {
        "stats": SCRIPT_DIR / "Statistical_Results",
    },
    "seeds": {
        "split": 42,
    },
}


BM3D_SIGMA = 0.1


def hard_cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def apply_bilateral(image_np: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(image_np, d=9, sigmaColor=75, sigmaSpace=75)


def apply_prbf(image_np: np.ndarray) -> np.ndarray:
    img_f = image_np.astype(np.float32)
    img_anscombe = 2.0 * np.sqrt(img_f + (3.8 / 8.0))
    filtered_anscombe = cv2.bilateralFilter(img_anscombe, d=5, sigmaColor=1.5, sigmaSpace=50)
    img_inv = (filtered_anscombe / 2.0) ** 2 - (3.8 / 8.0)
    return np.clip(img_inv, 0, 255).astype(np.uint8)


FORCE_GGB_FALLBACK = False


def apply_gradient_guided_bilateral(image_np: np.ndarray) -> np.ndarray:
    if FORCE_GGB_FALLBACK:
        return cv2.bilateralFilter(image_np, d=5, sigmaColor=50, sigmaSpace=50)
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad_mag = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1),
    )
    grad_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    try:
        return cv2.ximgproc.jointBilateralFilter(
            joint=grad_norm, src=image_np, d=5, sigmaColor=50, sigmaSpace=50
        )
    except AttributeError:
        return cv2.bilateralFilter(image_np, d=5, sigmaColor=50, sigmaSpace=50)


def apply_ad_zhang(image_np: np.ndarray, num_iter: int = 5, k: float = 50.0, lambda_: float = 0.1) -> np.ndarray:
    img = image_np.astype(np.float32)

    def g4(grad: np.ndarray) -> np.ndarray:
        val = (grad / k) ** 2
        return np.exp(-val) * (1.0 - np.tanh(val))

    def _ad_single_channel(channel: np.ndarray) -> np.ndarray:
        I = channel.copy()
        for _ in range(num_iter):
            Ip = np.pad(I, 1, mode="edge")
            N = Ip[:-2, 1:-1] - I
            S = Ip[2:, 1:-1] - I
            E = Ip[1:-1, 2:] - I
            W = Ip[1:-1, :-2] - I
            NE = Ip[:-2, 2:] - I
            NW = Ip[:-2, :-2] - I
            SE = Ip[2:, 2:] - I
            SW = Ip[2:, :-2] - I
            update = (g4(N) * N + g4(S) * S + g4(E) * E + g4(W) * W) + 0.5 * (
                g4(NE) * NE + g4(NW) * NW + g4(SE) * SE + g4(SW) * SW
            )
            I = I + lambda_ * update
        return I

    if len(img.shape) == 3:
        res = np.zeros_like(img)
        for c in range(img.shape[2]):
            res[:, :, c] = _ad_single_channel(img[:, :, c])
        return np.clip(res, 0, 255).astype(np.uint8)
    return np.clip(_ad_single_channel(img), 0, 255).astype(np.uint8)


def apply_bm3d(image_np: np.ndarray) -> np.ndarray:
    res = bm3d.bm3d(image_np.astype(np.float32) / 255.0, BM3D_SIGMA)
    return (np.clip(res, 0, 1) * 255).astype(np.uint8)


def apply_improved_bm3d(image_np: np.ndarray) -> np.ndarray:
    ad_filtered = apply_ad_zhang(image_np)
    res = bm3d.bm3d(ad_filtered.astype(np.float32) / 255.0, BM3D_SIGMA)
    return (np.clip(res, 0, 1) * 255).astype(np.uint8)


class HybridEarlyFusion(nn.Module):
    def __init__(self, base_model_fn):
        super().__init__()
        self.model = base_model_fn(in_channels=6)

    def forward(self, noisy, filtered):
        return self.model(torch.cat([noisy, filtered], dim=1))


class HybridResidual(nn.Module):
    def __init__(self, base_model_fn):
        super().__init__()
        self.model = base_model_fn(in_channels=3)

    def forward(self, noisy, filtered):
        return self.model(filtered) + noisy


class HybridAttention(nn.Module):
    def __init__(self, base_model_fn):
        super().__init__()
        self.mask_gen = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid(),
        )
        self.model = base_model_fn(in_channels=3)

    def forward(self, noisy, filtered):
        return self.model(noisy * self.mask_gen(filtered))


class HybridSFT(nn.Module):
    def __init__(self, base_model_fn):
        super().__init__()
        self.sft_gen = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 6, 3, padding=1),
        )
        self.norm = nn.InstanceNorm2d(3, affine=False)
        self.model = base_model_fn(in_channels=3)

    def forward(self, noisy, filtered):
        gamma, beta = torch.split(self.sft_gen(filtered), 3, dim=1)
        return self.model(self.norm(noisy) * gamma + beta)


class Noise2Detail(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_features=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, out_channels, 3, 1, 1),
        )

    def forward(self, x):
        return self.model(x)


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, channels, ffn_ratio=2):
        super().__init__()
        dw_ch = channels * 2
        self.norm1 = LayerNorm2d(channels)
        self.dw = nn.Conv2d(channels, dw_ch, 3, 1, 1, groups=channels)
        self.gate = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1))
        self.pw_out = nn.Conv2d(channels, channels, 1)
        self.norm2 = LayerNorm2d(channels)
        ffn_ch = int(channels * ffn_ratio) * 2
        self.ffn_in = nn.Conv2d(channels, ffn_ch, 1)
        self.ffn_out = nn.Conv2d(ffn_ch // 2, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        h = self.gate(self.dw(self.norm1(x)))
        h = self.pw_out(h * self.sca(h))
        x = x + h * self.beta
        h = self.ffn_out(self.gate(self.ffn_in(self.norm2(x))))
        return x + h * self.gamma


class NAFNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        width=16,
        enc_blocks=(1, 1, 1, 1),
        dec_blocks=(1, 1, 1, 1),
        middle_blocks=1,
    ):
        super().__init__()
        self.intro = nn.Conv2d(in_channels, width, 3, 1, 1)
        self.outro = nn.Conv2d(width, out_channels, 3, 1, 1)
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        ch = width
        for n in enc_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(n)]))
            self.downs.append(nn.Conv2d(ch, ch * 2, 2, 2))
            ch *= 2
        self.middle = nn.Sequential(*[NAFBlock(ch) for _ in range(middle_blocks)])
        for n in dec_blocks:
            self.ups.append(nn.Sequential(nn.Conv2d(ch, ch * 2, 1), nn.PixelShuffle(2)))
            ch //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(n)]))
        self.padder = 2 ** len(enc_blocks)

    def _pad(self, x):
        _, _, h, w = x.shape
        ph = (self.padder - h % self.padder) % self.padder
        pw = (self.padder - w % self.padder) % self.padder
        return nn.functional.pad(x, (0, pw, 0, ph), mode="reflect"), h, w

    def forward(self, x):
        x, h, w = self._pad(x)
        x = self.intro(x)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)
        x = self.middle(x)
        for dec, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            x = dec(up(x) + skip)
        return self.outro(x)[:, :, :h, :w]


def NAFNet_Light(in_channels=3, out_channels=3):
    return NAFNet(
        in_channels=in_channels,
        out_channels=out_channels,
        width=16,
        enc_blocks=(1, 1, 1, 1),
        dec_blocks=(1, 1, 1, 1),
        middle_blocks=1,
    )


FILTERS = {
    "PRBF": apply_prbf,
    "GGB": apply_gradient_guided_bilateral,
    "AD_Zhang": apply_ad_zhang,
    "BM3D": apply_bm3d,
    "Bilateral_Classic": apply_bilateral,
    "PRBF_Standalone": apply_prbf,
    "GGB_Standalone": apply_gradient_guided_bilateral,
    "Anisotropic_Zhang": apply_ad_zhang,
    "BM3D_Improved": apply_improved_bm3d,
}


def method_specs() -> OrderedDict[str, dict]:
    specs = OrderedDict()

    def add_neural(method_id, category, pre_filter, display, model, weight, is_hybrid=False, filter_name=None):
        specs[method_id] = {
            "category": category,
            "pre_filter": pre_filter,
            "display": display,
            "model": model,
            "weight": PROJECT_ROOT / "weights" / weight,
            "is_hybrid": is_hybrid,
            "filter_func": FILTERS.get(filter_name) if filter_name else None,
        }

    add_neural("Bare_N2D", "Standalone neural baseline", "None", "Bare N2D", Noise2Detail, "best_Bare_N2D.pth")
    add_neural(
        "Bare_NAFNet",
        "Standalone neural baseline",
        "None",
        "Bare NAFNet Light",
        NAFNet_Light,
        "best_Bare_NAFNet.pth",
    )

    combos = [
        ("N2D_PRBF_EarlyFusion", "PRBF", "N2D Early Fusion", lambda: HybridEarlyFusion(Noise2Detail), "best_N2D_PRBF_EarlyFusion.pth", "PRBF"),
        ("N2D_PRBF_Residual", "PRBF", "N2D Residual", lambda: HybridResidual(Noise2Detail), "best_N2D_PRBF_Residual.pth", "PRBF"),
        ("N2D_GGB_Attention", "GGB", "N2D Attention", lambda: HybridAttention(Noise2Detail), "best_N2D_GGB_Attention.pth", "GGB"),
        ("N2D_GGB_SFT_Modulation", "GGB", "N2D SFT", lambda: HybridSFT(Noise2Detail), "best_N2D_GGB_SFT_Modulation.pth", "GGB"),
        ("N2D_AD_Zhang_Attention", "AD Zhang", "N2D Attention", lambda: HybridAttention(Noise2Detail), "best_N2D_AD_Zhang_Attention.pth", "AD_Zhang"),
        ("N2D_AD_Zhang_SFT_Modulation", "AD Zhang", "N2D SFT", lambda: HybridSFT(Noise2Detail), "best_N2D_AD_Zhang_SFT_Modulation.pth", "AD_Zhang"),
        ("NAFNet_PRBF_EarlyFusion", "PRBF", "NAFNet Light Early Fusion", lambda: HybridEarlyFusion(NAFNet_Light), "best_NAFNet_PRBF_EarlyFusion.pth", "PRBF"),
        ("NAFNet_PRBF_Residual", "PRBF", "NAFNet Light Residual", lambda: HybridResidual(NAFNet_Light), "best_NAFNet_PRBF_Residual.pth", "PRBF"),
        ("NAFNet_GGB_Attention", "GGB", "NAFNet Light Attention", lambda: HybridAttention(NAFNet_Light), "best_NAFNet_GGB_Attention.pth", "GGB"),
        ("NAFNet_GGB_SFT_Modulation", "GGB", "NAFNet Light SFT", lambda: HybridSFT(NAFNet_Light), "best_NAFNet_GGB_SFT_Modulation.pth", "GGB"),
        ("NAFNet_AD_Zhang_Attention", "AD Zhang", "NAFNet Light Attention", lambda: HybridAttention(NAFNet_Light), "best_NAFNet_AD_Zhang_Attention.pth", "AD_Zhang"),
        ("NAFNet_AD_Zhang_SFT_Modulation", "AD Zhang", "NAFNet Light SFT", lambda: HybridSFT(NAFNet_Light), "best_NAFNet_AD_Zhang_SFT_Modulation.pth", "AD_Zhang"),
    ]
    for method_id, pre_filter, display, model, weight, filter_name in combos:
        specs[method_id] = {
            "category": "Proposed FGHC variant",
            "pre_filter": pre_filter,
            "display": display,
            "model": model,
            "weight": PROJECT_ROOT / "weights" / weight,
            "is_hybrid": True,
            "filter_func": FILTERS[filter_name],
        }

    for method_id, display in [
        ("BM3D", "BM3D"),
        ("Bilateral_Classic", "Bilateral Classic"),
        ("PRBF_Standalone", "PRBF Standalone"),
        ("GGB_Standalone", "GGB Standalone"),
        ("Anisotropic_Zhang", "Aniso Zhang"),
        ("BM3D_Improved", "BM3D Improved"),
    ]:
        specs[method_id] = {
            "category": "Classical baseline",
            "pre_filter": "Classical",
            "display": display,
            "classical_func": FILTERS[method_id],
        }
    return specs


KEY_METHODS = [
    "Bare_N2D",
    "N2D_GGB_SFT_Modulation",
    "Bare_NAFNet",
    "NAFNet_AD_Zhang_SFT_Modulation",
    "BM3D",
]


TESTS = [
    ("N2D_GGB_SFT_Modulation", "Bare_N2D", "GGB + N2D SFT vs Bare N2D"),
    ("NAFNet_AD_Zhang_SFT_Modulation", "Bare_NAFNet", "AD Zhang + NAFNet Light SFT vs Bare NAFNet Light"),
    ("N2D_GGB_SFT_Modulation", "BM3D", "GGB + N2D SFT vs BM3D"),
]


def natural_key(value: str):
    return (0, int(value)) if value.isdigit() else (1, value)


def get_or_create_splits(root_path: Path, seed: int, splits_file: Path) -> dict:
    if splits_file.exists():
        return json.loads(splits_file.read_text())
    valid_exts = (".png", ".bmp", ".tif", ".tiff")
    all_fovs = set()
    for root, _dirs, files in os.walk(root_path):
        if any(f.lower().endswith(valid_exts) for f in files):
            candidate = os.path.basename(os.path.dirname(root))
            if candidate in ["gt", "clean", "avg50", "avg16", "16", "50"]:
                all_fovs.add(os.path.basename(root))
    all_fovs = sorted(all_fovs)
    rng = np.random.RandomState(seed)
    rng.shuffle(all_fovs)
    t = int(len(all_fovs) * 0.7)
    v = t + int(len(all_fovs) * 0.15)
    splits = {"train": all_fovs[:t], "val": all_fovs[t:v], "test": all_fovs[v:]}
    splits_file.write_text(json.dumps(splits, indent=4))
    return splits


def parse_aimd_extracted(
    root_path: Path,
    mode: str = "test",
    mod_dir_contains: str | None = None,
    noise_level: str | None = None,
    fovs: set[str] | None = None,
):
    splits = get_or_create_splits(root_path, CONFIG["seeds"]["split"], CONFIG["data"]["splits_file"])
    allowed = set(splits["train"] + splits["val"] + splits["test"]) if mode == "all" else set(splits.get(mode, []))
    valid_exts = (".png", ".bmp", ".tif", ".tiff")
    mod_dirs = set()
    for root, _dirs, files in os.walk(root_path):
        if any(f.lower().endswith(valid_exts) for f in files):
            mod_dirs.add(os.path.dirname(os.path.dirname(root)))
    for mod_dir in sorted(mod_dirs):
        if mod_dir_contains and mod_dir_contains not in mod_dir:
            continue
        levels = [d for d in os.listdir(mod_dir) if os.path.isdir(os.path.join(mod_dir, d))]
        gt_folder = next((c for c in ["gt", "clean", "avg50", "avg16", "16", "50"] if c in levels), None)
        if not gt_folder:
            continue
        levels.remove(gt_folder)
        if noise_level:
            levels = [lvl for lvl in levels if lvl == noise_level]
        for fov in sorted(os.listdir(os.path.join(mod_dir, gt_folder)), key=natural_key):
            if fov not in allowed:
                continue
            if fovs and fov not in fovs:
                continue
            gt_dir = os.path.join(mod_dir, gt_folder, fov)
            gt_path = sorted(
                [os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.lower().endswith(valid_exts)]
            )
            for noise_lvl in sorted(levels):
                n_fov_p = os.path.join(mod_dir, noise_lvl, fov)
                if not os.path.exists(n_fov_p):
                    continue
                n_paths = sorted(
                    [os.path.join(n_fov_p, f) for f in os.listdir(n_fov_p) if f.lower().endswith(valid_exts)]
                )
                if len(gt_path) == 1:
                    for np_ in n_paths:
                        yield np_, gt_path[0], f"FOV{fov}_{os.path.basename(np_).split('.')[0]}"
                elif len(gt_path) == len(n_paths):
                    for np_, gp_ in zip(n_paths, gt_path):
                        yield np_, gp_, f"FOV{fov}_{os.path.basename(np_).split('.')[0]}"


def center_crop(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    ch, cw = min(size, h), min(size, w)
    sy, sx = h // 2 - ch // 2, w // 2 - cw // 2
    return img[sy : sy + ch, sx : sx + cw]


def tensor_from_bgr(img_np: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0


def load_model(spec: dict, device: str):
    model = spec["model"]() if callable(spec["model"]) else spec["model"]
    if not spec["weight"].exists():
        raise FileNotFoundError(spec["weight"])
    state = torch.load(spec["weight"], map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def evaluate_torch_model(model, img_np: np.ndarray, spec: dict, device: str):
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    with torch.no_grad():
        t_n = tensor_from_bgr(img_np, device)
        if spec["is_hybrid"]:
            filtered = spec["filter_func"](img_np)
            t_f = tensor_from_bgr(filtered, device)
            out = model(t_n, t_f)
        else:
            out = model(t_n)
    res = (np.clip(out.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)
    latency = time.time() - start
    vram = torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" and torch.cuda.is_available() else 0.0
    return res, latency, vram


def evaluate_classical(func, img_np: np.ndarray):
    start = time.time()
    res = func(img_np)
    latency = time.time() - start
    return res, latency, 0.0


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {sec:.0f}s"


def evaluate_method(method_id: str, spec: dict, pairs: list[tuple[str, str, str]], device: str) -> list[dict]:
    print(f"[*] Evaluating {method_id} ({spec['display']})")
    method_start = time.time()
    rows = []
    model = None
    if "model" in spec:
        model = load_model(spec, device)

    ws = CONFIG["eval"]["ssim_win_size"]
    dr = CONFIG["eval"]["data_range"]
    crop_size = CONFIG["eval"]["crop_size"]

    for idx, (npth, gpth, label) in enumerate(pairs, start=1):
        n_img = cv2.imread(npth)
        g_img = cv2.imread(gpth)
        if n_img is None or g_img is None:
            print(f"[!] Skipping unreadable pair: {npth} | {gpth}")
            continue
        n_img = center_crop(n_img, crop_size)
        g_img = center_crop(g_img, crop_size)

        n_psnr = psnr(g_img, n_img, data_range=dr)
        n_ssim = ssim(g_img, n_img, data_range=dr, channel_axis=2, win_size=ws)

        try:
            if model is not None:
                res, latency, vram = evaluate_torch_model(model, n_img, spec, device)
            else:
                res, latency, vram = evaluate_classical(spec["classical_func"], n_img)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(f"[!] OOM on {method_id} / {label}; skipped")
                hard_cleanup()
                continue
            raise

        r_psnr = psnr(g_img, res, data_range=dr)
        r_ssim = ssim(g_img, res, data_range=dr, channel_axis=2, win_size=ws)
        fps = 1.0 / latency if latency > 0 else 0.0

        rows.append(
            {
                "method_id": method_id,
                "category": spec["category"],
                "pre_filter": spec["pre_filter"],
                "method": spec["display"],
                "image_id": label,
                "noisy_psnr": n_psnr,
                "result_psnr": r_psnr,
                "psnr_gain": r_psnr - n_psnr,
                "noisy_ssim": n_ssim,
                "result_ssim": r_ssim,
                "ssim_gain": r_ssim - n_ssim,
                "latency": latency,
                "fps": fps,
                "vram_mb": vram,
            }
        )
        elapsed = time.time() - method_start
        avg_per_image = elapsed / max(idx, 1)
        remaining = avg_per_image * max(len(pairs) - idx, 0)
        print(
            f"    {idx}/{len(pairs)} {label}: "
            f"step={format_seconds(latency)}, "
            f"avg={format_seconds(avg_per_image)}/image, "
            f"elapsed={format_seconds(elapsed)}, "
            f"eta={format_seconds(remaining)}"
        )
        hard_cleanup()

    method_elapsed = time.time() - method_start
    print(
        f"[+] Finished {method_id}: "
        f"{len(rows)} images, total={format_seconds(method_elapsed)}, "
        f"avg={format_seconds(method_elapsed / max(len(rows), 1))}/image"
    )
    del model
    hard_cleanup()
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0


def summarize(rows: list[dict]) -> list[dict]:
    grouped = OrderedDict()
    for row in rows:
        grouped.setdefault(row["method_id"], []).append(row)
    summaries = []
    for method_id, items in grouped.items():
        base = items[0]
        summary = {
            "method_id": method_id,
            "category": base["category"],
            "pre_filter": base["pre_filter"],
            "method": base["method"],
            "n": len(items),
        }
        for metric in [
            "noisy_psnr",
            "result_psnr",
            "psnr_gain",
            "noisy_ssim",
            "result_ssim",
            "ssim_gain",
            "latency",
            "fps",
            "vram_mb",
        ]:
            m, s = mean_std([float(x[metric]) for x in items])
            summary[f"{metric}_mean"] = m
            summary[f"{metric}_std"] = s
        summaries.append(summary)
    return summaries


def paired_tests(rows: list[dict]) -> list[dict]:
    by_method = OrderedDict()
    for row in rows:
        by_method.setdefault(row["method_id"], {})[row["image_id"]] = row

    out = []
    for method_id, baseline_id, label in TESTS:
        if method_id not in by_method or baseline_id not in by_method:
            continue
        common = sorted(set(by_method[method_id]) & set(by_method[baseline_id]))
        if len(common) < 2:
            continue
        result = {"comparison": label, "method_id": method_id, "baseline_id": baseline_id, "n": len(common)}
        for metric in ["psnr_gain", "ssim_gain"]:
            diff = np.asarray(
                [by_method[method_id][image][metric] - by_method[baseline_id][image][metric] for image in common],
                dtype=float,
            )
            result[f"{metric}_delta_mean"] = float(np.mean(diff))
            try:
                result[f"{metric}_p_value"] = float(wilcoxon(diff, zero_method="wilcox").pvalue)
            except ValueError:
                result[f"{metric}_p_value"] = math.nan
        out.append(result)
    return out


def fmt_pm(mean: float, std: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def fmt_p(value: float) -> str:
    if value is None or math.isnan(value):
        return "--"
    if value < 0.001:
        return "$<0.001$"
    return f"{value:.3f}"


def write_latex_table(path: Path, summaries: list[dict], tests: list[dict], method_order: list[str]) -> None:
    summary_by_id = {row["method_id"]: row for row in summaries}
    tests_by_method = {row["method_id"]: row for row in tests}
    lines = [
        r"\begin{table}[htbp]",
        r"    \centering",
        r"    \caption{Statistical variability of key denoising results on the test set. Metrics are reported as mean~$\pm$~standard deviation across test images. The $p$-values are computed with the paired Wilcoxon signed-rank test against the corresponding baseline.}",
        r"    \label{tab:statistical_variability}",
        r"    \small",
        r"    \renewcommand{\arraystretch}{1.15}",
        r"    \begin{tabular}{llccc}",
        r"        \toprule",
        r"        \textbf{Pre-filter} & \textbf{Method} & \textbf{PSNR Gain} & \textbf{SSIM Gain} & \textbf{$p$-value} \\",
        r"        \midrule",
    ]
    for method_id in method_order:
        if method_id not in summary_by_id:
            continue
        row = summary_by_id[method_id]
        test = tests_by_method.get(method_id)
        p = fmt_p(test["psnr_gain_p_value"]) if test else "--"
        lines.append(
            "        "
            + f"{row['pre_filter']} & {row['method']} & "
            + f"{fmt_pm(row['psnr_gain_mean'], row['psnr_gain_std'])} & "
            + f"{fmt_pm(row['ssim_gain_mean'], row['ssim_gain_std'], digits=4)} & "
            + f"{p} \\\\"
        )
    lines += [
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
        "",
    ]
    path.write_text("\n".join(lines))


def maybe_write_xlsx(out_dir: Path, rows: list[dict], summaries: list[dict], tests: list[dict]) -> None:
    if importlib.util.find_spec("pandas") is None or importlib.util.find_spec("openpyxl") is None:
        print("[*] pandas/openpyxl not available; skipped XLSX export.")
        return
    import pandas as pd

    xlsx_path = out_dir / "statistical_evaluation.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="per_image_metrics", index=False)
        pd.DataFrame(summaries).to_excel(writer, sheet_name="summary", index=False)
        pd.DataFrame(tests).to_excel(writer, sheet_name="paired_tests", index=False)
    print(f"[*] Wrote {xlsx_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", choices=["key", "all"], default="key")
    parser.add_argument(
        "--method-ids",
        default=None,
        help="Comma-separated explicit method IDs. Overrides --methods. Example: N2D_GGB_SFT_Modulation,GGB_Standalone",
    )
    parser.add_argument("--test-limit", type=int, default=CONFIG["data"]["test_limit"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=CONFIG["paths"]["stats"])
    parser.add_argument(
        "--paper-table2-subset",
        action="store_true",
        help="Use the explicit WideField_BPAE_R/avg4/FOV1 subset that reproduces Table 2 Noisy PSNR.",
    )
    parser.add_argument("--mod-dir-contains", default=None)
    parser.add_argument("--noise-level", default=None)
    parser.add_argument("--fovs", default=None, help="Comma-separated FOV ids, for example: 1,12,17")
    parser.add_argument(
        "--force-ggb-fallback",
        action="store_true",
        help="Use cv2.bilateralFilter fallback for GGB to match runs made without cv2.ximgproc.",
    )
    return parser.parse_args()


def main() -> None:
    global FORCE_GGB_FALLBACK
    args = parse_args()
    FORCE_GGB_FALLBACK = args.force_ggb_fallback
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = method_specs()
    if args.method_ids:
        selected = [x.strip() for x in args.method_ids.split(",") if x.strip()]
        unknown = [x for x in selected if x not in specs]
        if unknown:
            raise ValueError(f"Unknown method IDs: {unknown}. Available: {list(specs.keys())}")
    else:
        selected = list(specs.keys()) if args.methods == "all" else KEY_METHODS

    mod_dir_contains = args.mod_dir_contains
    noise_level = args.noise_level
    fovs = set(x.strip() for x in args.fovs.split(",")) if args.fovs else None
    if args.paper_table2_subset:
        mod_dir_contains = "WideField_BPAE_R"
        noise_level = "avg4"
        fovs = {"1"}

    pairs = list(
        parse_aimd_extracted(
            CONFIG["data"]["path"],
            mode="test",
            mod_dir_contains=mod_dir_contains,
            noise_level=noise_level,
            fovs=fovs,
        )
    )[: args.test_limit]
    if not pairs:
        raise RuntimeError(f"No test pairs found under {CONFIG['data']['path']}")

    print(f"[*] Device: {args.device}")
    print(f"[*] Test pairs: {len(pairs)}")
    if mod_dir_contains or noise_level or fovs:
        print(f"[*] Pair filter: mod_dir_contains={mod_dir_contains}, noise_level={noise_level}, fovs={sorted(fovs) if fovs else None}")
    print(f"[*] GGB fallback mode: {FORCE_GGB_FALLBACK}")
    print(f"[*] Methods: {', '.join(selected)}")

    all_rows = []
    run_start = time.time()
    completed_methods = 0
    for method_id in selected:
        rows = evaluate_method(method_id, specs[method_id], pairs, args.device)
        all_rows.extend(rows)
        completed_methods += 1
        run_elapsed = time.time() - run_start
        avg_method = run_elapsed / max(completed_methods, 1)
        run_eta = avg_method * max(len(selected) - completed_methods, 0)
        print(
            f"[=] Progress: {completed_methods}/{len(selected)} methods, "
            f"elapsed={format_seconds(run_elapsed)}, "
            f"avg_method={format_seconds(avg_method)}, "
            f"eta={format_seconds(run_eta)}"
        )

    per_image_fields = [
        "method_id",
        "category",
        "pre_filter",
        "method",
        "image_id",
        "noisy_psnr",
        "result_psnr",
        "psnr_gain",
        "noisy_ssim",
        "result_ssim",
        "ssim_gain",
        "latency",
        "fps",
        "vram_mb",
    ]
    write_csv(out_dir / "per_image_metrics.csv", all_rows, per_image_fields)

    summaries = summarize(all_rows)
    summary_fields = list(summaries[0].keys()) if summaries else []
    write_csv(out_dir / "statistical_summary.csv", summaries, summary_fields)

    tests = paired_tests(all_rows)
    test_fields = list(tests[0].keys()) if tests else ["comparison", "method_id", "baseline_id", "n"]
    write_csv(out_dir / "paired_wilcoxon_tests.csv", tests, test_fields)
    write_latex_table(out_dir / "statistical_variability_table.tex", summaries, tests, selected)
    maybe_write_xlsx(out_dir, all_rows, summaries, tests)

    print(f"[*] Wrote {out_dir / 'per_image_metrics.csv'}")
    print(f"[*] Wrote {out_dir / 'statistical_summary.csv'}")
    print(f"[*] Wrote {out_dir / 'paired_wilcoxon_tests.csv'}")
    print(f"[*] Wrote {out_dir / 'statistical_variability_table.tex'}")


if __name__ == "__main__":
    main()
