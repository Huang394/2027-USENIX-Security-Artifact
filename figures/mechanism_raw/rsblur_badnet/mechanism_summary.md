# Mechanism Summary: rsblur_badnet

## Case

- Attack: BadNet
- Trigger metadata: localized_patch
- Degradation: none (0.0)
- Restoration prior: RSBlur
- Defense output setting: restoration_poisoned
- Expected behavior label: destroy_or_low_utility_suppression
- Analysis modules: input_space, feature_space, model_output, local_patch
- Target class: 1

## Metric-Level Observation

- Origin ASR: 0.9742
- Origin PA: 0.2637
- Final CA: 0.4004
- Final ASR: 0.1318
- Final PA: 0.6191
- ASR delta: -0.8424
- PA delta: 0.3555

## Prediction Distribution

| setting | predicted_label | prediction_relation | count | rate | is_final_output |
| --- | --- | --- | --- | --- | --- |
| restoration_poisoned | 0 | other_wrong | 213 | 0.416015625 | True |
| restoration_poisoned | 1 | attacker_target | 156 | 0.3046875 | True |
| restoration_poisoned | 3 | other_wrong | 5 | 0.009765625 | True |
| restoration_poisoned | 4 | other_wrong | 16 | 0.03125 | True |
| restoration_poisoned | 6 | other_wrong | 12 | 0.0234375 | True |
| restoration_poisoned | 7 | other_wrong | 1 | 0.001953125 | True |
| restoration_poisoned | 8 | other_wrong | 38 | 0.07421875 | True |
| restoration_poisoned | 9 | other_wrong | 71 | 0.138671875 | True |

## Prediction Flip Evidence

See `model_output/prediction_flip_summary.csv` for per-sample transitions.

## Pairwise Pipeline Evidence

| comparison | left_setting | right_setting | ca_delta | asr_delta | pa_delta | target_logit_delta | target_probability_delta | true_class_logit_delta | true_class_probability_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D_vs_origin | origin_poisoned | degradation_poisoned | None | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| R_vs_origin | origin_poisoned | restoration_poisoned | None | -0.8423772609819121 | 0.35546875 | -14.960334988194518 | -0.6656898658090409 | -7.264666197414044 | 0.27200213372961113 |
| RD_vs_origin | origin_poisoned | degradation_restoration_poisoned | None | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| RD_vs_D | degradation_poisoned | degradation_restoration_poisoned | None | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| RD_vs_R | restoration_poisoned | degradation_restoration_poisoned | None | 0.8423772609819121 | -0.35546875 | 14.960334988194518 | 0.6656898658090409 | 7.264666197414044 | -0.27200213372961113 |

## Model-Output Evidence

| setting | domain | defense_output_setting | is_final_output | num_samples | ca | asr | asr_num_samples | asr_excluded_target_origin_samples | pa | mean_target_logit | mean_true_class_logit | mean_target_margin | mean_target_probability | mean_true_class_probability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| degradation_poisoned | poisoned | restoration_poisoned | False | 512 | nan | 0.9741602067183462 | 387 | 125 | 0.263671875 | 20.05422130227089 | 13.865027196297888 | 6.189194110222161 | 0.9663945122320001 | 0.27773465396793184 |
| degradation_restoration_poisoned | poisoned | restoration_poisoned | False | 512 | nan | 0.9741602067183462 | 387 | 125 | 0.263671875 | 20.05422130227089 | 13.865027196297888 | 6.189194110222161 | 0.9663945122320001 | 0.27773465396793184 |
| origin_poisoned | poisoned | restoration_poisoned | False | 512 | nan | 0.9741602067183462 | 387 | 125 | 0.263671875 | 20.05422130227089 | 13.865027196297888 | 6.189194110222161 | 0.9663945122320001 | 0.27773465396793184 |
| restoration_poisoned | poisoned | restoration_poisoned | True | 512 | nan | 0.13178294573643412 | 387 | 125 | 0.619140625 | 5.0938863140763715 | 6.600360998883843 | -1.506474684807472 | 0.30070464642295924 | 0.549736787697543 |

## Feature-Space Evidence

- Mean poison-to-restored feature shift: 197.889642
- Mean restored-poison distance to clean centroid: 178.985161
- Mean restored-clean distance to clean centroid: 186.194849

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
- `gradcam/restoration_poisoned_gradcam.png`

Skipped visual evidence:
- None

## Failure-Case Evidence

| failure_case | num_samples | rate | requires_target_class | target_class_available |
| --- | --- | --- | --- | --- |
| clean_misclassified | 59 | 0.115234375 | False | True |
| final_clean_broken | 254 | 0.49609375 | False | True |
| attack_failed | 10 | 0.01953125 | True | True |
| final_poisoned_defense_failed_asr | 156 | 0.3046875 | True | True |
| final_poisoned_target_to_other_wrong | 144 | 0.28125 | True | True |
| final_poisoned_recovered | 307 | 0.599609375 | True | True |
| restoration_poisoned_defense_failed_asr | 156 | 0.3046875 | True | True |
| restoration_poisoned_overcorrected | 144 | 0.28125 | True | True |
| restoration_poisoned_recovered | 307 | 0.599609375 | True | True |
| degradation_restoration_poisoned_defense_failed_asr | 502 | 0.98046875 | True | True |
| degradation_restoration_poisoned_overcorrected | 0 | 0.0 | True | True |
| degradation_restoration_poisoned_recovered | 125 | 0.244140625 | True | True |

## Prediction Outcome Decision

Final outcome label: `destroy_or_low_utility_suppression`

Rationale:
- Final clean accuracy 0.4004 is below 0.7000.

## Caveats

- DeBackdoor-style re-synthesis and restoration-prior integrity tests are future hooks, not part of this v3.0 core run.