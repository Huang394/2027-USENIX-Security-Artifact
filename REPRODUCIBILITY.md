# Reproducibility Guide

This artifact supports three reproducibility levels.

## Environment

The paper-number audit scripts under `scripts/` use only the Python standard library. The bundled `requirements.txt` is intended for full main-experiment reruns, mechanism-analysis reruns, and validation tooling:

```powershell
conda create -n backdoor-artifact python=3.9
conda activate backdoor-artifact
python -m pip install -r requirements.txt
```

If the review machine has a specific CUDA version, install the matching PyTorch and torchvision wheels following the PyTorch selector, then install the remaining packages from `requirements.txt`.

## Level 1: Paper-Number Audit

Goal: regenerate paper-level CA, ASR, PA, outcome accounting, and adaptive stress-test tables from released logs and JSON outputs.

Run:

```powershell
conda run -n tf python scripts\parse_logs.py
conda run -n tf python scripts\classify_outcomes.py
conda run -n tf python scripts\reproduce_tables.py
```

Checks:

- `results/imagenette2_full_matrix.csv` has 196 rows: 4 origin baselines and 192 defense settings.
- `results/cifar10_external_check.csv` is separate from Imagenette2 accounting and includes an `input_handling` column.
- `results/adaptive_badnet_seeds.csv` and `results/adaptive_blended_observation.csv` are derived from `logs/adaptive_pipeline/*.json`.
- `results/threshold_sensitivity.csv` recomputes outcome counts under strict/default/permissive threshold profiles.

This level does not require full image datasets or checkpoints because it audits frozen experiment logs.

## Level 2: Mechanism-Evidence Audit

Goal: inspect and regenerate the mechanism-analysis summary table and visual-evidence index from released mechanism-analysis outputs.

Run:

```powershell
conda run -n tf python scripts\reproduce_tables.py
conda run -n tf python scripts\reproduce_figures.py
```

Checks:

- `results/mechanism_summary.csv` summarizes the released cases under `figures/mechanism_raw/`.
- `results/mechanism_visual_evidence_index.csv` indexes prediction-flow, confusion-matrix, embedding, Grad-CAM, frequency, residual, trigger, and case-figure evidence where present.
- `figures/mechanism_raw/*/metrics_summary.csv` values are sampled diagnostics only and are not paper-level metrics.

## Level 3: Lightweight Execution Smoke Test

Goal: verify that the artifact parser, classifier, adaptive summarizer, and mechanism-evidence summarizer run in the review environment.

Run:

```powershell
conda run -n tf python scripts\run_demo_metric_check.py
conda run -n tf python scripts\run_demo_mechanism.py
```

The smoke tests use released logs and summaries. They do not download datasets, load checkpoints, or rerun expensive purification pipelines.

## Full Reproduction Requirements

Full reruns require external assets that are not redistributed in this artifact:

| Asset | Expected use |
| --- | --- |
| Imagenette2 | Main paper accounting dataset. |
| CIFAR-10 | External-validity check. |
| Backdoored victim classifiers | CA/ASR/PA evaluation. |
| ZIP/guided-diffusion assets | I2I-Diffusion backend reruns. |
| ConvIR checkpoints | Dehazing/deblurring restoration backends. |
| Lite-BD/SwinIR assets | Third-party Lite-BD baseline reruns. |

For released audit paths, these assets are not needed because the paper-level numbers are parsed from logs.

## Path And Anonymization Notes

Release-facing files should not contain user-local absolute paths, account names, local machine names, or commit-history details. Mechanism case configs in `figures/mechanism_raw/*/case_config.json` have been sanitized so dataset and checkpoint fields document non-redistributed external assets rather than local source-machine locations.

Some implementation paths and logs use `diffusionzip`; the paper-facing name is `I2I-Diffusion`.
