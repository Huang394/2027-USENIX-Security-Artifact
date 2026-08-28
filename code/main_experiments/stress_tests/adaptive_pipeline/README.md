# Adaptive Pipeline Stress Test

This folder contains standalone stress tests for adaptive triggers against a
fixed purifier pipeline. These tests are intentionally separate from the
project's CA / ASR / PA evaluation path.

## Metric

The first metric is `PABR`:

```text
Pipeline Adaptive Break Rate =
  fraction of held-out non-target images classified as the target class after
  adaptive trigger -> degradation -> restoration -> fixed classifier
```

This is not ordinary ASR. ASR evaluates the original attack trigger. PABR
evaluates a new trigger optimized with knowledge of the purifier pipeline.
The original-trigger baseline recreates the BackdoorBox Blended pattern used by
`attack/backdoorattack.py`: a single-channel uint8 random pattern generated from
the run seed and applied with `alpha`.

Adaptive search runs the purifier in memory because the trigger changes at
every optimization step. To cross-check against image folders produced by the
main CA / ASR / PA pipeline, pass existing purified outputs with:

```text
--saved_original_purified_pois_root <path-to-purified-val_pois>
--saved_clean_purified_root <path-to-purified-val>
```

Those optional baselines run classifier-only evaluation on already-purified
images and are reported separately in the JSON.

For paper-facing runs, use:

```text
--require_main_asr_baseline
```

This fails unless `--saved_original_purified_pois_root` is provided. The JSON
then marks the canonical comparison baseline as:

```text
canonical_original_baseline_source = saved_main_pipeline_output
comparable_to_main_asr = true
```

Without a saved main-pipeline baseline, the script still runs, but the
after-pipeline original-trigger result is only an in-memory diagnostic and is
reported as:

```text
canonical_original_baseline_source = in_memory_unverified
comparable_to_main_asr = false
```

Before running the purifier pipeline, the script also reports the original
Blended trigger target rate directly on the fixed classifier:

```text
original_asr_before_pipeline
original_trigger_target_rate_before_pipeline
```

Use `-1` to evaluate all eligible samples:

```powershell
--search_samples 5000 `
--eval_samples -1
```

`search_samples=-1` is also supported, but full-search optimization is usually
much slower than using a fixed search subset and evaluating on the full
held-out split.

## Implemented Cases

The first implementation targets a minimal case study:

```text
Blended trigger + deterministic ConvIR purifier pipeline
```

Typical command:

```powershell
scripts\use-tf.cmd python stress_tests\adaptive_pipeline\adaptive_blended.py `
  --dataset Imagenette2 `
  --classes 20 `
  --img_size 256 `
  --attack_target 1 `
  --degradation_type haze `
  --degradation_strength 0.5 `
  --use_conv_ir `
  --restorer_ckpt pretrain\ots-base.pkl `
  --classifier_ckpt Imagenette2_pretrain\Blended\Res_34_256\ckpt.pth `
  --search_samples 1000 `
  --eval_samples 1000 `
  --eval_offset 0 `
  --steps 300 `
  --alpha 0.2 `
  --adaptive_init original_blended
```

Use this as a Section 7 / limitation case study. A high PABR means the purifier
should not be claimed as adaptively robust. A low PABR only means this one
simple adaptive trigger search did not break the fixed pipeline.

The second implementation targets the higher-value adaptive robustness check:

```text
BadNet trigger + deterministic ConvIR purifier pipeline
```

This is the preferred case when the original BadNet result has low post-pipeline
ASR but acceptable PA. The adaptive search only updates the BadNet patch pixels;
the classifier and purifier are fixed. The final evaluation can export both the
original BadNet poisoned images and the learned adaptive BadNet poisoned images
through the same `Purify(...).pur()` path used by the main CA / ASR / PA flow.

Typical BadNet command:

```powershell
scripts\use-tf.cmd python stress_tests\adaptive_pipeline\adaptive_badnet.py `
  --dataset Imagenette2 `
  --classes 20 `
  --img_size 256 `
  --attack_target 1 `
  --poisoned_rate 0.05 `
  --degradation_type haze `
  --degradation_strength 0.5 `
  --use_conv_ir `
  --restorer_ckpt pretrain\its-base.pkl `
  --classifier_ckpt Imagenette2_pretrain\BadNet\Res_34_256\ckpt.pth `
  --search_samples 5000 `
  --eval_samples -1 `
  --eval_offset 0 `
  --lr 0.05 `
  --steps 300 `
  --batch_size 16 `
  --num_workers 2 `
  --adaptive_init original_badnet `
  --best_eval_samples 512 `
  --best_eval_interval 30 `
  --final_eval_via_main_pipeline `
  --main_pipeline_output_dir stress_tests\adaptive_pipeline\main_pipeline_outputs\full_badnet `
  --output_dir stress_tests\adaptive_pipeline\results\full_badnet
```

For BadNet, `--alpha` is ignored because the patch is applied with the
BackdoorBox BadNets binary weight mask. Use `--attack_seed` to match the original
BadNet patch generation and `--optim_seed` only for adaptive search sampling and
random initialization. If those are omitted, both fall back to `--seed`.
