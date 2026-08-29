# Mechanism Analysis Tool

This directory contains the mechanism-analysis implementation used to generate the released evidence under `../../figures/mechanism_raw/`.

The analysis compares paired clean, poisoned, degraded, restored, and final-defense outputs. It generates tabular and visual evidence for prediction flow, confusion matrices, feature embeddings, model-output trajectories, input residuals, frequency spectra, Grad-CAM views, and trigger-specific diagnostics when available.

## Artifact Review Path

For artifact review, use the top-level scripts:

```powershell
python ..\..\scripts\reproduce_tables.py
python ..\..\scripts\reproduce_figures.py
python ..\..\scripts\run_demo_mechanism.py
```

These commands summarize and index already-released mechanism evidence. They do not require full datasets or checkpoints.

## Metric Boundary

The CA/ASR/PA values in each case's `metrics_summary.csv` are sampled diagnostic values. They support mechanism interpretation only. Paper-level full-split CA/ASR/PA values come from `../../logs/` and are reconstructed by `../../scripts/parse_logs.py`.

## Released Evidence

Each released case under `../../figures/mechanism_raw/` can include:

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

`case_config.json` files have been sanitized so non-redistributed datasets and checkpoints are documented without source-machine paths.

## Full Reruns

Full mechanism-analysis reruns require the external datasets, model checkpoints, restored image folders, and trigger masks described in `../../REPRODUCIBILITY.md`. The copied source code retains some legacy internal module and script names for compatibility with the original experiment workflow.
