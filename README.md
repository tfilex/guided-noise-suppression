# Filter-Guided Hybrid Cascades for Non-Stationary Noise Suppression

This repository contains the code, checkpoints, paper sources, and reproducibility artifacts for the article **Comparative Analysis of Baseline and Filter-Guided Hybrid Architectures for Non-Stationary Noise Suppression**.

The study compares classical filtering, lightweight neural denoisers, and filter-guided hybrid cascades under Poisson-dominated non-stationary noise. Experiments are based on fluorescence microscopy images, with the robotics connection treated as a constrained sensing use case rather than a completed ROS/onboard deployment benchmark.

![Filter-guided hybrid cascade architecture](paper/fig/fghc_architecture.png)

## Short Summary

The best reconstruction quality is obtained by the AD Zhang + NAFNet Light SFT configuration, while GGB + N2D SFT offers the strongest lightweight trade-off between quality and deployment cost. In the reported comparison, AD Zhang + NAFNet Light SFT reaches a PSNR gain of 5.46 dB and SSIM of 0.941, and GGB + N2D SFT reaches a PSNR gain of 5.32 dB, SSIM of 0.938, 10.6 FPS, and 79.4 MB VRAM.

![Qualitative comparison from the paper](paper/fig/figure2_three_examples_variant_b.png)

## Repository Layout

```text
paper/        LaTeX source, final PDF, and selected figures
src/          Training, ablation, and statistical evaluation code
scripts/      Figure-generation helper scripts
weights/      Released model checkpoints
splits/       Fixed experimental train/validation/test split
results/      Exported tables and statistical summaries
datasets/     Dataset placeholder and download instructions
.old/         Local archived working files; ignored by Git
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For CUDA systems, install the PyTorch build that matches the local driver from the official PyTorch installation page.

## Data

Download the fluorescence microscopy denoising dataset and place it at:

```text
datasets/AIMD/
```

The fixed split used by the experiments is included at `splits/fixed_splits.json`.

## Main Commands

Run the full training and ablation script:

```bash
python src/train_and_ablate.py
```

Recompute statistical tables from the released checkpoints:

```bash
python src/statistical_evaluation.py --out-dir results/Statistical_Results
```

Build the paper PDF:

```bash
cd paper
pdflatex -interaction=nonstopmode paper.tex
bibtex paper
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
```

## Notes

The included checkpoints are the available trained weights from the article experiments. The full image dataset is excluded because of size and licensing/distribution constraints.
