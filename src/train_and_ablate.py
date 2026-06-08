#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('jupyter nbconvert --to script *.ipynb')


# In[2]:


import os
import time
import random
import ctypes
import gc
import json
import psutil
import traceback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import cv2
cv2.setNumThreads(1)  # Детерминированность OpenCV

import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import bm3d



# In[3]:


# ==========================================
# 0. CONFIGURATION
# ==========================================
CONFIG = {
    "data": {
        "path": "datasets/AIMD",
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "patch_size": 256,
        "train_limit": 200,
        "val_limit": 50,
        "test_limit": 50,
        "splits_file": "splits/fixed_splits.json"
    },
    "train": {
        "epochs": 50,
        "batch_size": 8,
        "patience": 5,
        "num_runs": 3,
    },
    "eval": {
        "mode": "crop",
        "crop_size": 512,
        "ssim_win_size": 7,
        "data_range": 255,
        "images_to_save": 5
    },
    "seeds": {
        "global": 13,
        "split": 42,
        "loader": 100,
        "patch": 777
    },
    "paths": {
        "visuals": "results/Experiment_Visuals",
        "baselines": "results/Final_Baselines_Visuals",
        "results_excel": "results/ablation_study_results.xlsx",
        "filter_cache": "results/.filter_cache",
    }
}


# In[4]:


# ==========================================
# UTILITY
# ==========================================
def hard_cleanup():
    """Агрессивная очистка GPU и CPU памяти."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


set_global_seed(CONFIG["seeds"]["global"])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[*] Device: {device}")

for p in CONFIG["paths"].values():
    if not p.endswith(('.xlsx', '.json')):
        os.makedirs(p, exist_ok=True)




# In[21]:


# bm3d improved
def apply_improved_bm3d(image_np):
    """
    Improved BM3D (Zhang Y. et al., 2020)
    Шаг 1: Предварительная фильтрация анизотропной диффузией (AD Zhang)
    Шаг 2: Традиционный BM3D поверх отфильтрованного изображения
    """
    ad_filtered = apply_ad_zhang(image_np)
    res = bm3d.bm3d(ad_filtered.astype(np.float32) / 255.0, BM3D_SIGMA)
    return (np.clip(res, 0, 1) * 255).astype(np.uint8)


# In[5]:


# ==========================================
# 1. КЛАССИЧЕСКИЕ ФИЛЬТРЫ
# ==========================================
def apply_bilateral(image_np):
    return cv2.bilateralFilter(image_np, d=9, sigmaColor=75, sigmaSpace=75)


def apply_prbf(image_np):
    img_f = image_np.astype(np.float32)
    img_anscombe = 2.0 * np.sqrt(img_f + (3.8 / 8.0))
    filtered_anscombe = cv2.bilateralFilter(img_anscombe, d=5, sigmaColor=1.5, sigmaSpace=50)
    img_inv = (filtered_anscombe / 2.0) ** 2 - (3.8 / 8.0)
    return np.clip(img_inv, 0, 255).astype(np.uint8)


def apply_gradient_guided_bilateral(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad_mag = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    )
    grad_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    try:
        return cv2.ximgproc.jointBilateralFilter(
            joint=grad_norm, src=image_np, d=5, sigmaColor=50, sigmaSpace=50
        )
    except AttributeError:
        return cv2.bilateralFilter(image_np, d=5, sigmaColor=50, sigmaSpace=50)


def apply_ad_zhang(image_np, num_iter=5, k=50.0, lambda_=0.1):
    """
    Анизотропная диффузия Zhang Y. et al. (2020).
    g4 + 8 направлений. Паддинг пересчитывается каждую итерацию.
    """
    img = image_np.astype(np.float32)

    def g4(grad):
        val = (grad / k) ** 2
        return np.exp(-val) * (1.0 - np.tanh(val))

    def _ad_single_channel(channel):
        I = channel.copy()
        for _ in range(num_iter):
            Ip = np.pad(I, 1, mode='edge')
            N  = Ip[:-2, 1:-1] - I;  S  = Ip[2:,  1:-1] - I
            E  = Ip[1:-1, 2:]  - I;  W  = Ip[1:-1, :-2]  - I
            NE = Ip[:-2, 2:]   - I;  NW = Ip[:-2, :-2]   - I
            SE = Ip[2:,  2:]   - I;  SW = Ip[2:,  :-2]   - I
            update = (g4(N)*N + g4(S)*S + g4(E)*E + g4(W)*W) + \
                     0.5 * (g4(NE)*NE + g4(NW)*NW + g4(SE)*SE + g4(SW)*SW)
            I = I + lambda_ * update
        return I

    if len(img.shape) == 3:
        res = np.zeros_like(img)
        for c in range(img.shape[2]):
            res[:, :, c] = _ad_single_channel(img[:, :, c])
        return np.clip(res, 0, 255).astype(np.uint8)
    else:
        return np.clip(_ad_single_channel(img), 0, 255).astype(np.uint8)



# In[6]:


# ==========================================
# 2. СТРАТЕГИИ СЛИЯНИЯ
# ==========================================
class HybridEarlyFusion(nn.Module):
    """Конкатенация noisy + filtered по каналам перед сетью."""
    def __init__(self, base_model_fn):
        super().__init__()
        self.model = base_model_fn(in_channels=6)

    def forward(self, noisy, filtered):
        return self.model(torch.cat([noisy, filtered], dim=1))


class HybridResidual(nn.Module):
    """filtered → сеть, noisy добавляется как skip на выход."""
    def __init__(self, base_model_fn):
        super().__init__()
        self.model = base_model_fn(in_channels=3)

    def forward(self, noisy, filtered):
        return self.model(filtered) + noisy


class HybridAttention(nn.Module):
    """filtered генерирует spatial-маску, которая взвешивает noisy."""
    def __init__(self, base_model_fn):
        super().__init__()
        self.mask_gen = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 3, padding=1), nn.Sigmoid()
        )
        self.model = base_model_fn(in_channels=3)

    def forward(self, noisy, filtered):
        return self.model(noisy * self.mask_gen(filtered))


class HybridSFT(nn.Module):
    """Spatial Feature Transform: filtered управляет гамма/бета нормализации."""
    def __init__(self, base_model_fn):
        super().__init__()
        self.sft_gen = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 6, 3, padding=1)
        )
        self.norm  = nn.InstanceNorm2d(3, affine=False)
        self.model = base_model_fn(in_channels=3)

    def forward(self, noisy, filtered):
        gamma, beta = torch.split(self.sft_gen(filtered), 3, dim=1)
        return self.model(self.norm(noisy) * gamma + beta)




# In[7]:


# ==========================================
# 3. БАЗОВЫЕ МОДЕЛИ
# ==========================================
class Noise2Detail(nn.Module):
    """Лёгкая 3-слойная свёрточная сеть."""
    def __init__(self, in_channels=3, out_channels=3, num_features=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(num_features, out_channels, 3, 1, 1)
        )

    def forward(self, x):
        return self.model(x)



def init_weights_kaiming(m):
    if isinstance(m, nn.Conv2d):
        # Инициализация Кайминга (He) для слоев с ReLU
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class DnCNN(nn.Module):
    """17-слойная DnCNN с residual learning."""
    def __init__(self, in_channels=3, out_channels=3, num_features=64):
        super().__init__()
        layers = [nn.Conv2d(in_channels, num_features, 3, 1, 1), nn.ReLU(inplace=True)]
        for _ in range(15):
            layers += [nn.Conv2d(num_features, num_features, 3, 1, 1),
                       nn.BatchNorm2d(num_features), nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(num_features, out_channels, 3, 1, 1))
        self.dncnn = nn.Sequential(*layers)

        self.apply(init_weights_kaiming)

    def forward(self, x):
        return x - self.dncnn(x)


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias   = nn.Parameter(torch.zeros(num_channels))
        self.eps    = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var  = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        x    = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, channels, ffn_ratio=2):
        super().__init__()
        dw_ch        = channels * 2
        self.norm1   = LayerNorm2d(channels)
        self.dw      = nn.Conv2d(channels, dw_ch, 3, 1, 1, groups=channels)
        self.gate    = SimpleGate()
        self.sca     = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1))
        self.pw_out  = nn.Conv2d(channels, channels, 1)
        self.norm2   = LayerNorm2d(channels)
        ffn_ch       = int(channels * ffn_ratio) * 2
        self.ffn_in  = nn.Conv2d(channels, ffn_ch, 1)
        self.ffn_out = nn.Conv2d(ffn_ch // 2, channels, 1)
        self.beta    = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma   = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        h = self.gate(self.dw(self.norm1(x)))
        h = self.pw_out(h * self.sca(h))
        x = x + h * self.beta
        h = self.ffn_out(self.gate(self.ffn_in(self.norm2(x))))
        return x + h * self.gamma


class NAFNet(nn.Module):
    """
    NAFNet (ECCV 2022). U-Net с NAFBlock'ами и pixel-shuffle апсемплингом.
    Параметры по умолчанию = NAFNet-32 из оригинальной статьи.
    """
    def __init__(self, in_channels=3, out_channels=3,
                 width=32, enc_blocks=(1, 1, 1, 28),
                 dec_blocks=(1, 1, 1, 1), middle_blocks=1):
        super().__init__()
        self.intro    = nn.Conv2d(in_channels, width, 3, 1, 1)
        self.outro    = nn.Conv2d(width, out_channels, 3, 1, 1)
        self.encoders = nn.ModuleList()
        self.downs    = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups      = nn.ModuleList()
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
        return nn.functional.pad(x, (0, pw, 0, ph), mode='reflect'), h, w

    def forward(self, x):
        x, h, w = self._pad(x)
        x = self.intro(x)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x); skips.append(x); x = down(x)
        x = self.middle(x)
        for dec, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            x = dec(up(x) + skip)
        return self.outro(x)[:, :, :h, :w]



# In[8]:


# ==========================================
# 4. ДАТАСЕТ: ЛЕНИВЫЙ + ДИСКОВЫЙ КЕШ ФИЛЬТРОВ
# ==========================================
def get_cached_filtered(n_path, filter_func, filter_name):
    """
    Считает filter_func(img) один раз и кеширует результат на диск как .npy.
    При повторных запусках — просто отдаёт путь, без пересчёта.
    Критично для медленных фильтров (BM3D, AD Zhang).
    """
    cache_dir  = CONFIG["paths"]["filter_cache"]
    key        = f"{os.path.basename(n_path)}_{filter_name}.npy"
    cache_path = os.path.join(cache_dir, key)
    if os.path.exists(cache_path):
        return cache_path
    img = cv2.imread(n_path)
    if img is None:
        return None
    np.save(cache_path, filter_func(img))
    del img
    return cache_path


class HybridDataset(Dataset):
    """
    Ленивый датасет:
    - В RAM хранятся только пути + координаты патчей (минимум).
    - Изображения читаются с диска в __getitem__ и сразу освобождаются.
    - Отфильтрованные изображения кешируются на диск (.npy) через
      get_cached_filtered — не живут в RAM, не пересчитываются при рестарте.
    - Flip-аугментация синхронна для всей тройки noisy/filtered/gt.
    """
    def __init__(self, pairs, patch_size, mode, filter_func=None, filter_name="none"):
        self.mode       = mode
        self.patch_size = patch_size
        self.has_filter = filter_func is not None
        self.patch_refs = []  # (n_path, g_path, f_cache_or_None, y, x, flip_v, flip_h)

        rng = np.random.RandomState(CONFIG["seeds"]["patch"])

        for n_p, g_p, _ in pairs:
            img = cv2.imread(n_p)
            if img is None:
                continue
            h, w = img.shape[:2]
            del img  # shape получили — освобождаем

            f_cache = None
            if filter_func is not None:
                f_cache = get_cached_filtered(n_p, filter_func, filter_name)
                if f_cache is None:
                    continue  # битый файл — пропускаем

            for _ in range(4):
                y      = rng.randint(0, max(1, h - patch_size))
                x      = rng.randint(0, max(1, w - patch_size))
                flip_v = (mode == 'train') and (rng.rand() > 0.5)
                flip_h = (mode == 'train') and (rng.rand() > 0.5)
                self.patch_refs.append((n_p, g_p, f_cache, y, x, flip_v, flip_h))

    def _extract(self, img, y, x, flip_v, flip_h):
        ps = self.patch_size
        p  = img[y:y + ps, x:x + ps].copy()
        if flip_v: p = p[::-1]
        if flip_h: p = p[:, ::-1]
        return p.copy()

    def __len__(self):
        return len(self.patch_refs)

    def __getitem__(self, idx):
        n_p, g_p, f_cache, y, x, flip_v, flip_h = self.patch_refs[idx]
        dummy = torch.zeros(3, self.patch_size, self.patch_size)

        n_img = cv2.imread(n_p)
        g_img = cv2.imread(g_p)
        if n_img is None or g_img is None:
            return (dummy, dummy, dummy) if self.has_filter else (dummy, dummy)

        n = self._extract(n_img, y, x, flip_v, flip_h)
        g = self._extract(g_img, y, x, flip_v, flip_h)
        del n_img, g_img

        t_n = torch.from_numpy(n).permute(2, 0, 1).float() / 255
        t_g = torch.from_numpy(g).permute(2, 0, 1).float() / 255

        if f_cache is not None:
            f_img = np.load(f_cache)
            f     = self._extract(f_img, y, x, flip_v, flip_h)
            del f_img
            t_f = torch.from_numpy(f).permute(2, 0, 1).float() / 255
            return t_n, t_f, t_g
        return t_n, t_g



# In[9]:


# ==========================================
# 5. СПЛИТЫ
# ==========================================
def get_or_create_splits(root_path, seed, splits_file):
    if os.path.exists(splits_file):
        with open(splits_file, 'r') as f:
            return json.load(f)
    valid_exts = ('.png', '.bmp', '.tif', '.tiff')
    all_fovs   = set()
    for root, dirs, files in os.walk(root_path):
        if any(f.lower().endswith(valid_exts) for f in files):
            candidate = os.path.basename(os.path.dirname(root))
            if candidate in ['gt', 'clean', 'avg50', 'avg16', '16', '50']:
                all_fovs.add(os.path.basename(root))
    all_fovs = sorted(all_fovs)
    rng = random.Random(seed)
    rng.shuffle(all_fovs)
    t = int(len(all_fovs) * CONFIG["data"]["train_ratio"])
    v = t + int(len(all_fovs) * CONFIG["data"]["val_ratio"])
    splits = {"train": all_fovs[:t], "val": all_fovs[t:v], "test": all_fovs[v:]}
    with open(splits_file, 'w') as f:
        json.dump(splits, f, indent=4)
    return splits


def parse_aimd_extracted(root_path, mode='all'):
    splits  = get_or_create_splits(root_path, CONFIG["seeds"]["split"], CONFIG["data"]["splits_file"])
    allowed = set(splits["train"] + splits["val"] + splits["test"]) \
              if mode == 'all' else set(splits.get(mode, []))
    valid_exts = ('.png', '.bmp', '.tif', '.tiff')
    mod_dirs   = set()
    for root, dirs, files in os.walk(root_path):
        if any(f.lower().endswith(valid_exts) for f in files):
            mod_dirs.add(os.path.dirname(os.path.dirname(root)))
    for mod_dir in mod_dirs:
        levels    = [d for d in os.listdir(mod_dir) if os.path.isdir(os.path.join(mod_dir, d))]
        gt_folder = next((c for c in ['gt', 'clean', 'avg50', 'avg16', '16', '50'] if c in levels), None)
        if not gt_folder:
            continue
        levels.remove(gt_folder)
        for fov in os.listdir(os.path.join(mod_dir, gt_folder)):
            if fov not in allowed:
                continue
            gt_dir  = os.path.join(mod_dir, gt_folder, fov)
            gt_path = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir)
                               if f.lower().endswith(valid_exts)])
            for noise_lvl in levels:
                n_fov_p = os.path.join(mod_dir, noise_lvl, fov)
                if not os.path.exists(n_fov_p):
                    continue
                n_paths = sorted([os.path.join(n_fov_p, f) for f in os.listdir(n_fov_p)
                                   if f.lower().endswith(valid_exts)])
                if len(gt_path) == 1:
                    for np_ in n_paths:
                        yield np_, gt_path[0], f"FOV{fov}_{os.path.basename(np_).split('.')[0]}"
                elif len(gt_path) == len(n_paths):
                    for np_, gp_ in zip(n_paths, gt_path):
                        yield np_, gp_, f"FOV{fov}_{os.path.basename(np_).split('.')[0]}"




# In[10]:


# ==========================================
# 6. ОБУЧЕНИЕ
# ==========================================
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        return torch.mean(torch.sqrt((x - y) ** 2 + self.eps ** 2))


def train_with_validation(model, optimizer, train_loader, val_loader,
                           epochs, save_path, is_hybrid=False):
    criterion              = CharbonnierLoss()
    best_val, patience_cnt = float('inf'), 0
    t_hist, v_hist         = [], []

    for epoch in range(epochs):
        model.train()
        tr_loss = 0.0
        pbar    = tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False)

        for batch in pbar:
            # set_to_none=True освобождает память градиентов сразу,
            # а не обнуляет — экономит RAM по сравнению с zero_grad()
            optimizer.zero_grad(set_to_none=True)
            try:
                if is_hybrid:
                    n, f, g = [b.to(device) for b in batch]
                    loss = criterion(model(n, f), g)
                else:
                    n, g = [b.to(device) for b in batch]
                    loss = criterion(model(n), g)
                loss.backward()
                optimizer.step()
                tr_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    tqdm.write(f"[!] OOM на train батче epoch={epoch+1}, пропускаем.")
                    hard_cleanup()
                    continue
                raise

        model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                try:
                    if is_hybrid:
                        n, f, g = [b.to(device) for b in batch]
                        vl_loss += criterion(model(n, f), g).item()
                    else:
                        n, g = [b.to(device) for b in batch]
                        vl_loss += criterion(model(n), g).item()
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        tqdm.write(f"[!] OOM на val батче, пропускаем.")
                        hard_cleanup()
                        continue
                    raise

        # Чистим после каждой эпохи — не даём фрагментации накапливаться
        hard_cleanup()

        avg_tr = tr_loss / max(len(train_loader), 1)
        avg_vl = vl_loss / max(len(val_loader), 1)
        t_hist.append(avg_tr)
        v_hist.append(avg_vl)
        tqdm.write(f"Epoch {epoch + 1}: Train={avg_tr:.5f} | Val={avg_vl:.5f}")

        if avg_vl < best_val:
            best_val = avg_vl
            torch.save(model.state_dict(), save_path)
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= CONFIG["train"]["patience"]:
                tqdm.write("Early Stopping!")
                break

    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, weights_only=True))
    return model, t_hist, v_hist



# In[11]:


# ==========================================
# 7. ВИЗУАЛИЗАЦИЯ
# ==========================================
def save_loss_plot(train_h, val_h, name, save_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(train_h, label='Train')
    plt.plot(val_h,   label='Val')
    plt.title(f"Loss: {name}")
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"loss_{name}.png"))
    plt.close()


def save_visual_report_plot(noisy, denoised, gt, label, n_psnr, n_ssim,
                             r_psnr, r_ssim, ssim_map, save_dir, prefix):
    diff      = cv2.absdiff(gt, denoised)
    error_map = cv2.applyColorMap(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)
    ssim_gray = np.mean(ssim_map, axis=2)
    fig, axes = plt.subplots(1, 5, figsize=(30, 6))
    imgs   = [cv2.cvtColor(noisy,     cv2.COLOR_BGR2RGB),
               cv2.cvtColor(denoised,  cv2.COLOR_BGR2RGB),
               cv2.cvtColor(gt,        cv2.COLOR_BGR2RGB),
               cv2.cvtColor(error_map, cv2.COLOR_BGR2RGB),
               ssim_gray]
    titles = [f"Noisy\nPSNR:{n_psnr:.2f} SSIM:{n_ssim:.4f}",
               f"Denoised\nPSNR:{r_psnr:.2f} SSIM:{r_ssim:.4f}",
               "Ground Truth", "Error (Jet)", "SSIM Heatmap"]
    for i, (ax, img, title) in enumerate(zip(axes, imgs, titles)):
        if i == 4:
            im = ax.imshow(img, cmap='magma', vmin=0, vmax=1)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_{label}_report.jpg"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


def generate_summary_heatmap(excel_path, save_dir):
    try:
        df    = pd.read_excel(excel_path)
        pivot = df.pivot_table(index="Integration", columns="Pre-filter", values="PSNR Gain")
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".3f",
                    cbar_kws={'label': 'PSNR Gain (dB)'}, linewidths=.5)
        plt.title("Ablation: PSNR Gain Matrix", fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "Global_Summary_Heatmap.png"), dpi=300)
        plt.close()
        print(f"[*] Хитмап → {save_dir}/Global_Summary_Heatmap.png")
    except Exception as e:
        print(f"[!] Хитмап не удался: {e}")



# In[12]:


# ==========================================
# 8. INFERENCE И EVALUATION
# ==========================================
class InferencePipeline:
    def __init__(self, model, is_torch=True, is_hybrid=False, filter_func=None):
        self.model       = model
        self.is_torch    = is_torch
        self.is_hybrid   = is_hybrid
        self.filter_func = filter_func
        self.process     = psutil.Process(os.getpid())

    def __call__(self, img_np):
        if self.is_torch and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start = time.time()
        if self.is_torch:
            t_n = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255
            with torch.no_grad():
                if self.is_hybrid:
                    t_f = torch.from_numpy(self.filter_func(img_np)).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255
                    out = self.model(t_n, t_f)
                else:
                    out = self.model(t_n)
            res  = (np.clip(out.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)
            vram = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
        else:
            res  = self.model(img_np)
            vram = 0.0
        lat = time.time() - start
        ram = self.process.memory_info().rss / (1024 ** 2)
        return res, lat, vram, ram


def run_evaluation(pipeline, limit, mode='test', save_dir=None, prefix="Test"):
    results = []
    pairs   = list(parse_aimd_extracted(CONFIG["data"]["path"], mode=mode))[:limit]
    ws, dr  = CONFIG["eval"]["ssim_win_size"], CONFIG["eval"]["data_range"]

    for i, (npth, gpth, lbl) in enumerate(pairs):
        n_img = cv2.imread(npth)
        g_img = cv2.imread(gpth)
        if n_img is None or g_img is None:
            continue

        if CONFIG["eval"]["mode"] == "crop":
            h, w   = n_img.shape[:2]
            ch, cw = min(512, h), min(512, w)
            sy, sx = h // 2 - ch // 2, w // 2 - cw // 2
            n_img  = n_img[sy:sy + ch, sx:sx + cw]
            g_img  = g_img[sy:sy + ch, sx:sx + cw]

        n_psnr = psnr(g_img, n_img, data_range=dr)
        n_ssim = ssim(g_img, n_img, data_range=dr, channel_axis=2, win_size=ws)

        try:
            res, lat, vrm, ram = pipeline(n_img)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[!] OOM при инференсе {lbl}, пропускаем.")
                hard_cleanup()
                continue
            raise

        r_psnr           = psnr(g_img, res, data_range=dr)
        r_ssim, ssim_map = ssim(g_img, res, data_range=dr, channel_axis=2, win_size=ws, full=True)

        fps = 1.0 / lat if lat > 0 else 0.0
        results.append({
            "Noisy PSNR": n_psnr, "Result PSNR": r_psnr, "PSNR Gain": r_psnr - n_psnr,
            "Noisy SSIM": n_ssim, "Result SSIM": r_ssim, "SSIM Gain": r_ssim - n_ssim,
            "Latency": lat, "FPS": fps, "VRAM (MB)": vrm, "RAM (MB)": ram
        })

        if save_dir and i < CONFIG["eval"]["images_to_save"]:
            save_visual_report_plot(n_img, res, g_img, lbl,
                                    n_psnr, n_ssim, r_psnr, r_ssim,
                                    ssim_map, save_dir, prefix)

        # Чистим после каждого изображения — инференс на полном разрешении тяжёлый
        hard_cleanup()

    if not results:
        return {}
    df = pd.DataFrame(results)
    return {k: df[k].mean() for k in df.columns}


def save_formatted_excel(results_list, save_path):
    if not results_list:
        return
    df    = pd.DataFrame(results_list)
    order = ["Pre-filter", "Integration",
             "Noisy PSNR", "Result PSNR", "PSNR Gain",
             "Noisy SSIM", "Result SSIM", "SSIM Gain",
             "Latency", "VRAM (MB)", "RAM (MB)"]
    cols  = [c for c in order if c in df.columns]
    df[cols].to_excel(save_path, index=False)



# In[22]:


# ==========================================
# 9. RUNNER
# ==========================================
def train_and_evaluate_model(model, optimizer, exp_name, tr_loader, vl_loader,
                              is_hybrid, filter_func, save_dir, all_res,
                              pre_filter_name, integration_name):
    model, t_h, v_h = train_with_validation(
        model, optimizer, tr_loader, vl_loader,
        epochs=CONFIG["train"]["epochs"],
        save_path=f"best_{exp_name}.pth",
        is_hybrid=is_hybrid
    )
    save_loss_plot(t_h, v_h, exp_name, save_dir)

    pipe    = InferencePipeline(model, is_torch=True, is_hybrid=is_hybrid, filter_func=filter_func)
    metrics = run_evaluation(pipe, CONFIG["data"]["test_limit"],
                              save_dir=save_dir, prefix=exp_name)
    if metrics:
        metrics.update({"Integration": integration_name, "Pre-filter": pre_filter_name})
        all_res.append(metrics)

    save_formatted_excel(all_res, CONFIG["paths"]["results_excel"])

    # Явно удаляем модель из памяти после каждого эксперимента
    del model, pipe
    hard_cleanup()


ABLATION_COMBOS = [
    {"f": apply_prbf,                      "f_name": "PRBF",     "strat": HybridEarlyFusion, "s_name": "EarlyFusion"},
    {"f": apply_prbf,                      "f_name": "PRBF",     "strat": HybridResidual,    "s_name": "Residual"},
    {"f": apply_gradient_guided_bilateral,  "f_name": "GGB",      "strat": HybridAttention,   "s_name": "Attention"},
    {"f": apply_gradient_guided_bilateral,  "f_name": "GGB",      "strat": HybridSFT,         "s_name": "SFT_Modulation"},
    {"f": apply_ad_zhang,                  "f_name": "AD_Zhang", "strat": HybridAttention,   "s_name": "Attention"},
    {"f": apply_ad_zhang,                  "f_name": "AD_Zhang", "strat": HybridSFT,         "s_name": "SFT_Modulation"},
]

BM3D_SIGMA = 0.1  # ≈ 25.5/255 — типичный уровень шума для микроскопии AIMD
CLASSICAL_FILTERS = {
    "BM3D":              lambda x: (np.clip(bm3d.bm3d(x.astype(np.float32) / 255, BM3D_SIGMA), 0, 1) * 255).astype(np.uint8),
    "Bilateral_Classic": apply_bilateral,
    "PRBF_Standalone":   apply_prbf,
    "GGB_Standalone":    apply_gradient_guided_bilateral,
    "Anisotropic_Zhang": apply_ad_zhang,
    "BM3D_Improved":     apply_improved_bm3d,  # <--- Твоя новая строчка
}


def run_phase(base_cls, base_tag, all_res, tr_pairs, vl_pairs,
              tr_loader_bare, vl_loader_bare, run_classics=False):
    """
    Bare + все гибриды для одной базовой сети.
    Каждый эксперимент в try/except — падение одного не убивает фазу,
    всё что успело — сохраняется в Excel.
    """
    print(f"\n{'='*60}")
    print(f"ФАЗА: {base_tag}" + (" + классика" if run_classics else ""))
    print(f"{'='*60}")

    # --- Bare ---
    print(f"\n[*] Training Bare_{base_tag}...")
    try:
        bare = base_cls(in_channels=3, out_channels=3).to(device)
        train_and_evaluate_model(
            bare, optim.Adam(bare.parameters(), lr=1e-3),
            exp_name=f"Bare_{base_tag}",
            tr_loader=tr_loader_bare, vl_loader=vl_loader_bare,
            is_hybrid=False, filter_func=None,
            save_dir=CONFIG["paths"]["visuals"], all_res=all_res,
            pre_filter_name="None", integration_name=f"Bare_{base_tag}"
        )
    except Exception:
        print(f"[!] Bare_{base_tag} упал:\n{traceback.format_exc()}")
        hard_cleanup()

    # --- Гибриды ---
    print(f"\n>>> Hybrid {base_tag} models...")
    for exp in ABLATION_COMBOS:
        exp_name = f"{base_tag}_{exp['f_name']}_{exp['s_name']}"
        print(f"\n[*] Training {exp_name}...")
        try:
            tr_loader_hyb = DataLoader(
                HybridDataset(tr_pairs, 256, 'train',
                              filter_func=exp['f'], filter_name=exp['f_name']),
                batch_size=CONFIG["train"]["batch_size"], shuffle=True,
                num_workers=2, pin_memory=(device == 'cuda')
            )
            vl_loader_hyb = DataLoader(
                HybridDataset(vl_pairs, 256, 'val',
                              filter_func=exp['f'], filter_name=exp['f_name']),
                batch_size=CONFIG["train"]["batch_size"],
                num_workers=2, pin_memory=(device == 'cuda')
            )
            hybrid = exp['strat'](base_cls).to(device)
            train_and_evaluate_model(
                hybrid, optim.Adam(hybrid.parameters(), lr=1e-3),
                exp_name=exp_name,
                tr_loader=tr_loader_hyb, vl_loader=vl_loader_hyb,
                is_hybrid=True, filter_func=exp['f'],
                save_dir=CONFIG["paths"]["visuals"], all_res=all_res,
                pre_filter_name=exp['f_name'],
                integration_name=f"{base_tag}_{exp['s_name']}"
            )
            del tr_loader_hyb, vl_loader_hyb
            hard_cleanup()
        except Exception:
            print(f"[!] {exp_name} упал:\n{traceback.format_exc()}")
            hard_cleanup()

    # --- Классика ---
    if run_classics:
        print("\n>>> BASELINES: Classical Filters")
        for f_name, f_func in CLASSICAL_FILTERS.items():
            print(f"[*] Testing {f_name}...")
            try:
                m = run_evaluation(
                    InferencePipeline(f_func, is_torch=False),
                    CONFIG["data"]["test_limit"],
                    prefix=f_name,
                    save_dir=CONFIG["paths"]["baselines"]
                )
                if m:
                    m.update({"Integration": f_name, "Pre-filter": "Classical"})
                    all_res.append(m)
                save_formatted_excel(all_res, CONFIG["paths"]["results_excel"])
            except Exception:
                print(f"[!] {f_name} упал:\n{traceback.format_exc()}")

    save_formatted_excel(all_res, CONFIG["paths"]["results_excel"])
    print(f"\n[✔] {base_tag} завершён → {CONFIG['paths']['results_excel']}")
    generate_summary_heatmap(CONFIG["paths"]["results_excel"], CONFIG["paths"]["visuals"])



# In[14]:


# ==========================================
# 10. ПОДГОТОВКА ДАННЫХ
# ==========================================
print("[*] Подготовка списков файлов...")
tr_pairs = list(parse_aimd_extracted(CONFIG["data"]["path"], mode='train'))[:CONFIG["data"]["train_limit"]]
vl_pairs = list(parse_aimd_extracted(CONFIG["data"]["path"], mode='val'))[:CONFIG["data"]["val_limit"]]

if os.path.exists(CONFIG["paths"]["results_excel"]):
    all_res = pd.read_excel(CONFIG["paths"]["results_excel"]).to_dict('records')
else:
    all_res = []

tr_loader_bare = DataLoader(
    HybridDataset(tr_pairs, 256, 'train', filter_func=None),
    batch_size=CONFIG["train"]["batch_size"], shuffle=True,
    num_workers=2, pin_memory=(device == 'cuda')
)
vl_loader_bare = DataLoader(
    HybridDataset(vl_pairs, 256, 'val', filter_func=None),
    batch_size=CONFIG["train"]["batch_size"],
    num_workers=2, pin_memory=(device == 'cuda')
)


# In[15]:


# ==========================================
# 11. ЗАПУСК ФАЗ
# ========================================== run_classics=True
# Фаза 1: N2D классика (классика запускается в конце NAFNet)
run_phase(Noise2Detail, "N2D",    all_res, tr_pairs, vl_pairs, tr_loader_bare, vl_loader_bare, run_classics=True)


# In[16]:


# вот это тяжёлая nafnet
#run_phase(NAFNet,       "NAFNet", all_res, tr_pairs, vl_pairs, tr_loader_bare, vl_loader_bare, run_classics=False)



# In[17]:


# Создаем легкую версию NAFNet (16 каналов, 4 блока энкодера)
def NAFNet_Light(in_channels=3, out_channels=3):
    return NAFNet(
        in_channels=in_channels, 
        out_channels=out_channels, 
        width=16,                   # Количество каналов: 16
        enc_blocks=(1, 1, 1, 1),    # Убираем тяжелый блок из 28 слоев, оставляем везде по 1
        dec_blocks=(1, 1, 1, 1),    
        middle_blocks=1
    )


# In[18]:


# nafnet полегче
run_phase(NAFNet_Light, "NAFNet", all_res, tr_pairs, vl_pairs, tr_loader_bare, vl_loader_bare, run_classics=True)


# In[19]:


save_formatted_excel(all_res, CONFIG["paths"]["results_excel"])
print(f"\n[✔] Всё готово. Финальные результаты: {CONFIG['paths']['results_excel']}")
generate_summary_heatmap(CONFIG["paths"]["results_excel"], CONFIG["paths"]["visuals"])


# In[20]:


# Фаза 2: DnCNN
# run_phase(DnCNN, "DnCNN", all_res, tr_pairs, vl_pairs, tr_loader_bare, vl_loader_bare, run_classics=False)



# In[23]:


# ==========================================
# ДОПОЛНИТЕЛЬНЫЙ ТЕСТ: Только классические фильтры
# ==========================================
print("\n>>> ЗАПУСК ДОПОЛНИТЕЛЬНЫХ ТЕСТОВ: Classical Filters (включая BM3D_Improved)")

# Загружаем существующие результаты, чтобы не затереть их
if os.path.exists(CONFIG["paths"]["results_excel"]):
    all_res = pd.read_excel(CONFIG["paths"]["results_excel"]).to_dict('records')
else:
    all_res = []

# Ищем, какие классические фильтры мы уже тестировали, чтобы не гонять их дважды
# (если хочешь прогнать все заново, можешь удалить эту проверку)
tested_integrations = [res.get("Integration") for res in all_res]

for f_name, f_func in CLASSICAL_FILTERS.items():
    if f_name in tested_integrations:
        print(f"[*] {f_name} уже протестирован, пропускаем...")
        continue

    print(f"[*] Testing {f_name}...")
    try:
        pipe = InferencePipeline(f_func, is_torch=False)
        metrics = run_evaluation(
            pipe, 
            CONFIG["data"]["test_limit"],
            prefix=f_name,
            save_dir=CONFIG["paths"]["baselines"]
        )
        if metrics:
            metrics.update({"Integration": f_name, "Pre-filter": "Classical"})
            all_res.append(metrics)

        # Сразу сохраняем промежуточный результат
        save_formatted_excel(all_res, CONFIG["paths"]["results_excel"])
    except Exception as e:
        print(f"[!] {f_name} упал:\n{traceback.format_exc()}")
        hard_cleanup()

# Финальное сохранение и перерисовка
save_formatted_excel(all_res, CONFIG["paths"]["results_excel"])
print(f"\n[✔] Тестирование классики завершено. Обновляем файлы: {CONFIG['paths']['results_excel']}")
generate_summary_heatmap(CONFIG["paths"]["results_excel"], CONFIG["paths"]["visuals"])


# In[ ]:




