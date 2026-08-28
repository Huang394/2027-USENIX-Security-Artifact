# Mechanism Summary: gopro_wanet_blur3

## Case

- Attack: WaNet
- Trigger metadata: geometric_warp
- Degradation: motion_blur (3.0)
- Restoration prior: ConvIR-GoPro
- Defense output setting: degradation_restoration_poisoned
- Expected behavior label: attack_effect_recovery_or_counterproductive
- Analysis modules: model_output, geometry_structure, feature_space, input_space
- Target class: 0

## Metric-Level Observation

- Origin ASR: 0.9760
- Origin PA: 0.7617
- Final CA: 0.7949
- Final ASR: 0.5600
- Final PA: 0.8203
- ASR delta: -0.4160
- PA delta: 0.0586

## Prediction Distribution

| setting | predicted_label | prediction_relation | count | rate | is_final_output |
| --- | --- | --- | --- | --- | --- |
| degradation_restoration_poisoned | 0 | attacker_target | 445 | 0.869140625 | True |
| degradation_restoration_poisoned | 1 | other_wrong | 46 | 0.08984375 | True |
| degradation_restoration_poisoned | 2 | other_wrong | 1 | 0.001953125 | True |
| degradation_restoration_poisoned | 3 | other_wrong | 8 | 0.015625 | True |
| degradation_restoration_poisoned | 4 | other_wrong | 7 | 0.013671875 | True |
| degradation_restoration_poisoned | 5 | other_wrong | 2 | 0.00390625 | True |
| degradation_restoration_poisoned | 6 | other_wrong | 1 | 0.001953125 | True |
| degradation_restoration_poisoned | 8 | other_wrong | 2 | 0.00390625 | True |

## Prediction Flip Evidence

See `model_output/prediction_flip_summary.csv` for per-sample transitions.

## Pairwise Pipeline Evidence

| comparison | left_setting | right_setting | ca_delta | asr_delta | pa_delta | target_logit_delta | target_probability_delta | true_class_logit_delta | true_class_probability_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D_vs_origin | origin_poisoned | degradation_poisoned | None | -0.9359999999999999 | 0.091796875 | -11.664050350140315 | -0.3743364706192158 | -6.78125795465894 | 0.05148740212298153 |
| R_vs_origin | origin_poisoned | restoration_poisoned | None | -0.248 | 0.052734375 | -7.818065184401348 | -0.08984722725392658 | -5.208975294139236 | 0.04311334611134632 |
| RD_vs_origin | origin_poisoned | degradation_restoration_poisoned | None | -0.4159999999999999 | 0.05859375 | -9.101017082342878 | -0.13771232669397193 | -6.353534013964236 | 0.030912594217775813 |
| RD_vs_D | degradation_poisoned | degradation_restoration_poisoned | None | 0.52 | -0.033203125 | 2.5630332677974366 | 0.2366241439252439 | 0.42772394069470465 | -0.020574807905205716 |
| RD_vs_R | restoration_poisoned | degradation_restoration_poisoned | None | -0.16799999999999993 | 0.005859375 | -1.2829518979415298 | -0.047865099440045356 | -1.1445587198249996 | -0.012200751893570505 |

## Model-Output Evidence

| setting | domain | defense_output_setting | is_final_output | num_samples | ca | asr | asr_num_samples | asr_excluded_target_origin_samples | pa | mean_target_logit | mean_true_class_logit | mean_target_margin | mean_target_probability | mean_true_class_probability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| degradation_poisoned | poisoned | degradation_restoration_poisoned | False | 512 | nan | 0.04 | 125 | 387 | 0.853515625 | 6.885528961021919 | 8.654000178677961 | -1.7684712132904679 | 0.6180609416528962 | 0.8140630300425755 |
| degradation_restoration_poisoned | poisoned | degradation_restoration_poisoned | True | 512 | nan | 0.56 | 125 | 387 | 0.8203125 | 9.448562228819355 | 9.081724119372666 | 0.36683810828253627 | 0.85468508557814 | 0.7934882221373698 |
| origin_poisoned | poisoned | degradation_restoration_poisoned | False | 512 | nan | 0.976 | 125 | 387 | 0.76171875 | 18.549579311162233 | 15.435258133336902 | 3.11432116990909 | 0.992397412272112 | 0.762575627919594 |
| restoration_poisoned | poisoned | degradation_restoration_poisoned | False | 512 | nan | 0.728 | 125 | 387 | 0.814453125 | 10.731514126760885 | 10.226282839197665 | 0.5052312863990664 | 0.9025501850181854 | 0.8056889740309403 |

## Feature-Space Evidence

- Mean poison-to-restored feature shift: 197.561410
- Mean restored-poison distance to clean centroid: 205.241321
- Mean restored-clean distance to clean centroid: 213.172413

## Visual Evidence

- `confusion_matrix/poisoned_pipeline_confusion.png`
- `confusion_matrix/final_output_confusion.png`
- `prediction_flow/origin_to_final_prediction_flow.csv`
- `prediction_flow/origin_to_final_prediction_flow.png`
- `frequency/spectrum_grid.png`
- `embedding/feature_embedding_points.csv`
- `embedding/feature_embedding.png`
- `gradcam/origin_poisoned_gradcam.png`
- `gradcam/degradation_poisoned_gradcam.png`
- `gradcam/degradation_restoration_poisoned_gradcam.png`

Skipped visual evidence:
- None

## Failure-Case Evidence

| failure_case | num_samples | rate | requires_target_class | target_class_available |
| --- | --- | --- | --- | --- |
| clean_misclassified | 57 | 0.111328125 | False | True |
| final_clean_broken | 74 | 0.14453125 | False | True |
| attack_failed | 3 | 0.005859375 | True | True |
| final_poisoned_defense_failed_asr | 444 | 0.8671875 | True | True |
| final_poisoned_target_to_other_wrong | 22 | 0.04296875 | True | True |
| final_poisoned_recovered | 418 | 0.81640625 | True | True |
| restoration_poisoned_defense_failed_asr | 476 | 0.9296875 | True | True |
| restoration_poisoned_overcorrected | 4 | 0.0078125 | True | True |
| restoration_poisoned_recovered | 414 | 0.80859375 | True | True |
| degradation_restoration_poisoned_defense_failed_asr | 444 | 0.8671875 | True | True |
| degradation_restoration_poisoned_overcorrected | 22 | 0.04296875 | True | True |
| degradation_restoration_poisoned_recovered | 418 | 0.81640625 | True | True |

## Prediction Outcome Decision

Final outcome label: `attack_effect_recovery_or_counterproductive`

Rationale:
- Restoration after degradation increases ASR relative to degradation-only by 0.5200.

## Caveats

- This is an attack-effect recovery signal, not trigger recovery proof without geometry, feature, or residual evidence.
- DeBackdoor-style re-synthesis and restoration-prior integrity tests are future hooks, not part of this v3.0 core run.