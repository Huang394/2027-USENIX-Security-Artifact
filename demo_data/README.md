# Demo Data Policy

The current lightweight demo path uses released logs and mechanism-analysis summary outputs rather than redistributed images or checkpoints.

Run from the artifact root:

```powershell
python scripts\run_demo_metric_check.py
python scripts\run_demo_mechanism.py
```

These smoke tests verify that metric parsing, outcome classification, adaptive JSON summarization, and mechanism-evidence indexing run end to end. They are not empirical evidence for the paper claims and should not be treated as a substitute for the full evaluation logs.

Full image-level reruns require external datasets and checkpoints documented in `REPRODUCIBILITY.md`.
