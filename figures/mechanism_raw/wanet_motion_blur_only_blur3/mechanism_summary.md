# Mechanism Summary: wanet_motion_blur_only_blur3

## Case

- Attack: WaNet
- Trigger metadata: geometric_warp
- Degradation: motion_blur (3.0)
- Restoration prior: none
- Defense output setting: degradation_poisoned
- Expected behavior label: clean_preserving_suppress
- Analysis modules: geometry_structure, model_output, feature_space, input_space
- Target class: 0

## Metric-Level Observation

- Origin ASR: 0.9760
- Origin PA: 0.7617
- Final CA: 0.8379
- Final ASR: 0.0400
- Final PA: 0.8535
- ASR delta: -0.9360
- PA delta: 0.0918

## Prediction Distribution

| setting | predicted_label | prediction_relation | count | rate | is_final_output |
| --- | --- | --- | --- | --- | --- |
| degradation_poisoned | 0 | attacker_target | 330 | 0.64453125 | True |
| degradation_poisoned | 1 | other_wrong | 125 | 0.244140625 | True |
| degradation_poisoned | 2 | other_wrong | 2 | 0.00390625 | True |
| degradation_poisoned | 3 | other_wrong | 2 | 0.00390625 | True |
| degradation_poisoned | 5 | other_wrong | 1 | 0.001953125 | True |
| degradation_poisoned | 8 | other_wrong | 49 | 0.095703125 | True |
| degradation_poisoned | 9 | other_wrong | 3 | 0.005859375 | True |

## Prediction Flip Evidence

See `model_output/prediction_flip_summary.csv` for per-sample transitions.

## Pairwise Pipeline Evidence

| comparison | left_setting | right_setting | ca_delta | asr_delta | pa_delta | target_logit_delta | target_probability_delta | true_class_logit_delta | true_class_probability_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D_vs_origin | origin_poisoned | degradation_poisoned | None | -0.9359999999999999 | 0.091796875 | -11.664050350140315 | -0.3743364706192158 | -6.78125795465894 | 0.05148740212298153 |
| RD_vs_origin | origin_poisoned | degradation_restoration_poisoned | None | -0.9359999999999999 | 0.09375 | -11.665374360221904 | -0.3748853796065077 | -6.7959541799500585 | 0.05064377206708459 |
| RD_vs_D | degradation_poisoned | degradation_restoration_poisoned | None | 0.0 | 0.001953125 | -0.001324010081589222 | -0.000548908987291874 | -0.014696225291118026 | -0.0008436300558969378 |

## Model-Output Evidence

| setting | domain | defense_output_setting | is_final_output | num_samples | ca | asr | asr_num_samples | asr_excluded_target_origin_samples | pa | mean_target_logit | mean_true_class_logit | mean_target_margin | mean_target_probability | mean_true_class_probability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| degradation_poisoned | poisoned | degradation_poisoned | True | 512 | nan | 0.04 | 125 | 387 | 0.853515625 | 6.885528961021919 | 8.654000178677961 | -1.7684712132904679 | 0.6180609416528962 | 0.8140630300425755 |
| degradation_restoration_poisoned | poisoned | degradation_poisoned | False | 512 | nan | 0.04 | 125 | 387 | 0.85546875 | 6.88420495094033 | 8.639303953386843 | -1.7550989992450923 | 0.6175120326656043 | 0.8132193999866786 |
| origin_poisoned | poisoned | degradation_poisoned | False | 512 | nan | 0.976 | 125 | 387 | 0.76171875 | 18.549579311162233 | 15.435258133336902 | 3.11432116990909 | 0.992397412272112 | 0.762575627919594 |

## Feature-Space Evidence

- Mean poison-to-restored feature shift: 3.615708
- Mean restored-poison distance to clean centroid: 1.946652
- Mean restored-clean distance to clean centroid: 1.919017

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

Skipped visual evidence:
- None

## Failure-Case Evidence

| failure_case | num_samples | rate | requires_target_class | target_class_available |
| --- | --- | --- | --- | --- |
| clean_misclassified | 57 | 0.111328125 | False | True |
| final_clean_broken | 39 | 0.076171875 | False | True |
| attack_failed | 3 | 0.005859375 | True | True |
| final_poisoned_defense_failed_asr | 330 | 0.64453125 | True | True |
| final_poisoned_target_to_other_wrong | 70 | 0.13671875 | True | True |
| final_poisoned_recovered | 434 | 0.84765625 | True | True |
| restoration_poisoned_defense_failed_asr | 0 | 0.0 | True | True |
| restoration_poisoned_overcorrected | 0 | 0.0 | True | True |
| restoration_poisoned_recovered | 0 | 0.0 | True | True |
| degradation_restoration_poisoned_defense_failed_asr | 331 | 0.646484375 | True | True |
| degradation_restoration_poisoned_overcorrected | 69 | 0.134765625 | True | True |
| degradation_restoration_poisoned_recovered | 435 | 0.849609375 | True | True |

## Prediction Outcome Decision

Final outcome label: `clean_preserving_suppress`

Rationale:
- ASR drops by 0.9360 and poisoned accuracy recovers or reaches the high-PA threshold.

## Caveats

- DeBackdoor-style re-synthesis and restoration-prior integrity tests are future hooks, not part of this v3.0 core run.