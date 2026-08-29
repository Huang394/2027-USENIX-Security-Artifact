# USENIX Security 2027 Artifact

Artifact for:

> Beyond Attack Success Rate: Rethinking Restoration-Based Backdoor Purification through Prediction Recovery

This release is designed for auditability. It does not package every working-directory file, full datasets, large checkpoints, or all expensive purification outputs. Instead, it provides the logs, summaries, code, and lightweight scripts needed to verify the paper's empirical claims and inspect the mechanism evidence.

## What To Verify

1. Rebuild paper-level CA, ASR, and poisoned accuracy (PA) tables from full evaluation logs.
2. Rebuild the 192-setting Imagenette2 outcome-accounting matrix.
3. Rebuild adaptive BadNet and supplementary Blended stress-test tables from JSON outputs.
4. Rebuild the mechanism-analysis summary and inspect released visual evidence.
5. Run a lightweight smoke test for the parser/classifier/summary path.

## Layout

```text
code/
  main_experiments/      Main purification/evaluation implementation.
  analysis_tool/         Mechanism-analysis implementation.
configs/
  main_experiments/      Main experiment configs.
  mechanism_representative_cases.yaml
logs/
  imagenette2/           Full paper-level Imagenette2 evaluation logs.
  cifar10/               Full CIFAR10 external-validity logs.
  adaptive_pipeline/     Adaptive stress-test JSON outputs.
results/                 Regenerated CSV tables.
figures/
  mechanism_raw/         Released mechanism evidence.
demo_data/               Smoke-test notes.
scripts/                 Artifact parsers and reproducers.
requirements.txt         Runtime/development dependency list.
```

## Metric Authority

Paper-level CA/ASR/PA values come from `logs/imagenette2/` and `logs/cifar10/`, which are copied from the main experiment log tree.

The mechanism-analysis files under `figures/mechanism_raw/*/metrics_summary.csv` use sampled diagnostic subsets. They are included to explain representative mechanisms and visual evidence. They must not be used as full-split paper-level CA/ASR/PA values.

CIFAR10 is an external-validity check. It is intentionally separated in `results/cifar10_external_check.csv` and is not counted in the 192-setting Imagenette2 accounting matrix.

## Environment

The top-level log-audit scripts use only the Python standard library. Install `requirements.txt` when running the full experiment code, mechanism-analysis code, or validation tools:

```powershell
conda create -n backdoor-artifact python=3.9
conda activate backdoor-artifact
python -m pip install -r requirements.txt
```

For GPU reruns, install a `torch`/`torchvision` build matching the review machine's CUDA setup if the default pip wheels are not appropriate.

## Rebuild Results

Run from the artifact root:

```powershell
python scripts\parse_logs.py
python scripts\classify_outcomes.py
python scripts\reproduce_tables.py
python scripts\reproduce_figures.py
```

Expected outputs include:

```text
results/imagenette2_full_matrix.csv
results/cifar10_external_check.csv
results/adaptive_badnet_seeds.csv
results/adaptive_blended_observation.csv
results/mechanism_summary.csv
results/threshold_sensitivity.csv
results/mechanism_visual_evidence_index.csv
```

`results/imagenette2_full_matrix.csv` contains 196 rows: 4 origin attack baselines plus 192 completed defense settings. The paper's accounting matrix uses the 192 rows where `condition=defense`.

## Smoke Tests

```powershell
conda run -n tf python scripts\run_demo_metric_check.py
conda run -n tf python scripts\run_demo_mechanism.py
```

These checks parse released logs and already-generated mechanism evidence. They do not require full datasets or model checkpoints, and they do not establish empirical claims beyond confirming that the artifact tooling runs end to end.

## Backend Naming

Some files and logs retain the legacy implementation name `diffusionzip`. In the paper-facing text and tables, this backend is named `I2I-Diffusion`.

## Main-Code Provenance

`code/main_experiments/` is built on the public ZIP repository, `https://github.com/sycny/ZIP`, which accompanies the NeurIPS 2023 paper `Black-box Backdoor Defense via Zero-shot Image Purification` by Yucheng Shi, Mengnan Du, Xuansheng Wu, Zihan Guan, Jin Sun, and Ninghao Liu. The original ZIP README is preserved as `code/main_experiments/UPSTREAM_ZIP_README.md`; the artifact-facing README in `code/main_experiments/README.md` describes the modifications and review path for this submission.

## Lite-BD

Lite-BD is a third-party baseline. This artifact does not redistribute Lite-BD source code. It includes only the Lite-BD logs and parsed result rows needed to audit numbers reported in the paper. Reviewers who want to rerun the full Lite-BD baseline should use the official Lite-BD repository and the provenance notes in `REPRODUCIBILITY.md`.
