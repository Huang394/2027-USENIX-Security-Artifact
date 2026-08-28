# Artifact Manifest

This manifest records the release contents and how each item supports the paper.

## Top-Level Files

```text
README.md
REPRODUCIBILITY.md
MANIFEST.md
LICENSES.md
requirements.txt
```

`requirements.txt` lists dependencies for full reruns, mechanism-analysis reruns, and validation. The paper-number audit scripts are standard-library-only.

## Main Experiment Code

`code/main_experiments/` preserves the main purification backend and evaluation code needed to understand and rerun the non-Lite-BD pipelines:

This component is ZIP-derived. The upstream project is `https://github.com/sycny/ZIP`, the code release for the NeurIPS 2023 paper `Black-box Backdoor Defense via Zero-shot Image Purification`.

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
README.md
UPSTREAM_ZIP_README.md
PURIFICATION_BACKEND_GUIDELINE.md
LICENSE
```

Large datasets, generated image folders, pretraining folders, checkpoints, caches, and working-directory-only outputs are not part of this release path.

## Main Experiment Logs

`logs/imagenette2/` and `logs/cifar10/` preserve the full evaluation logs used for paper-level CA, ASR, and PA parsing.

Observed log families include:

```text
attack_log
diffusionzip
haze
litebd
litebd_bicubic
litebd_swinir
motion_blur
zip
```

Derived files:

| File | Purpose |
| --- | --- |
| `results/imagenette2_full_matrix.csv` | Paper-level Imagenette2 baselines plus 192 defense settings. |
| `results/cifar10_external_check.csv` | CIFAR10 external-validity check with backend input-handling labels. |
| `results/threshold_sensitivity.csv` | Outcome-accounting counts under multiple threshold profiles. |

## Adaptive Stress Tests

`logs/adaptive_pipeline/` preserves adaptive pipeline result JSON files.

Derived files:

| File | Purpose |
| --- | --- |
| `results/adaptive_badnet_seeds.csv` | BadNet adaptive seed table. |
| `results/adaptive_blended_observation.csv` | Supplementary Blended adaptive observation. |

## Mechanism Analysis

`code/analysis_tool/` preserves the mechanism-analysis implementation, including:

```text
README.md
configs/mechanism_representative_cases.yaml
mechanism-analysis package directory
tools/
tests/
mypy.ini
```

`figures/mechanism_raw/` preserves released evidence for each representative mechanism case:

```text
mechanism_summary.md
case_config.json
behavior_decision.json
metrics_summary.csv
model_output/
feature_space/
input_space/
trigger_representation/
visual_evidence/
figures/
```

Derived files:

| File | Purpose |
| --- | --- |
| `results/mechanism_summary.csv` | Quantitative mechanism summary from released mechanism-analysis outputs. |
| `results/mechanism_visual_evidence_index.csv` | Index of released visual and tabular evidence files. |

CA/ASR/PA values in mechanism `metrics_summary.csv` are sampled diagnostic values only.

## Lite-BD Handling

Lite-BD source code is intentionally not included. The artifact includes Lite-BD evidence only through:

```text
logs/imagenette2/litebd*
logs/cifar10/litebd*
results/imagenette2_full_matrix.csv rows for Lite-BD
results/cifar10_external_check.csv rows for Lite-BD
```

Reviewers who want to rerun Lite-BD should use the official Lite-BD repository and the external-asset notes in `REPRODUCIBILITY.md`.

## Demo Subset

`demo_data/README.md` documents the smoke-test policy. Current smoke tests operate on released logs and mechanism-analysis outputs, not on redistributed images or checkpoints.
