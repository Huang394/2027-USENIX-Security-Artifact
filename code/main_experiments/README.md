# Main Experiment Code

This directory contains the main purification and evaluation implementation used to produce the paper-level full-split logs released under `../../logs/`.

This code is built on the public ZIP implementation:

```text
https://github.com/sycny/ZIP
```

ZIP is the code release for the NeurIPS 2023 paper `Black-box Backdoor Defense via Zero-shot Image Purification` by Yucheng Shi, Mengnan Du, Xuansheng Wu, Zihan Guan, Jin Sun, and Ninghao Liu. This artifact preserves the relevant ZIP-derived structure and adds/adapts restoration backend routing, ConvIR-based purification paths, logging, scheduling, and adaptive stress-test code used for this paper.

For artifact review, the primary audit path is log-based:

```powershell
conda run -n tf python ..\..\scripts\parse_logs.py
conda run -n tf python ..\..\scripts\classify_outcomes.py
```

Full experiment reruns require external datasets, victim checkpoints, diffusion assets, and restoration checkpoints documented in `../../REPRODUCIBILITY.md`.

## Included Files

```text
main.py
settings.py
run_logging.py
experiment_schedule.json
configs/
attack/
preprocess/
scripts/
stress_tests/adaptive_pipeline/
PURIFICATION_BACKEND_GUIDELINE.md
UPSTREAM_ZIP_README.md
LICENSE
```

## What This Code Supports

- Full-split CA/ASR/PA evaluation with the same denominator semantics used by the released logs.
- Restoration and purification backends used in the main experiment matrix.
- Adaptive pipeline stress-test scripts under `stress_tests/adaptive_pipeline/`.
- CIFAR10 ImageFolder export helper under `scripts/`.

The released paper-level logs are in:

```text
../../logs/imagenette2/
../../logs/cifar10/
```

Adaptive stress-test JSON evidence is in:

```text
../../logs/adaptive_pipeline/
```

## Important Scope Boundaries

- `../../results/imagenette2_full_matrix.csv` is derived from full logs, not sampled mechanism-analysis outputs.
- CIFAR10 is an external-validity check and is not part of the 192-setting Imagenette2 accounting matrix.
- `diffusionzip` is a legacy implementation name. In paper-facing documentation and tables, use `I2I-Diffusion`.
- Lite-BD source code is not redistributed in this artifact. Lite-BD appears only through released logs and parsed result rows.
- `attack/BackdoorBox/` is third-party reference code preserved for compatibility with this experiment implementation.

## Full Rerun Sketch

The normal evaluation entry point is `main.py`. Example shape:

```powershell
conda run -n tf python main.py `
  --dataset Imagenette2 `
  --attack_method BadNet `
  --img_size 256 `
  --deg haze `
  --deg_scale 0.5 `
  --at_threshold 1 `
  -pctes -pptes -upctes -upptes
```

This command shape is illustrative. A real full rerun requires the expected dataset/checkpoint layout and backend assets described in `PURIFICATION_BACKEND_GUIDELINE.md` and `../../REPRODUCIBILITY.md`.

## Provenance

`UPSTREAM_ZIP_README.md` preserves the original ZIP project README for provenance. The artifact-facing instructions in this file supersede the upstream demo instructions for review.
