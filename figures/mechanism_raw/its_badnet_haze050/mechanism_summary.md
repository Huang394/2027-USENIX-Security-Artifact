# Mechanism Summary: its_badnet_haze050

## Case

- Attack: BadNet
- Trigger metadata: localized_patch
- Degradation: haze (0.5)
- Restoration prior: ConvIR-ITS
- Defense output setting: degradation_restoration_poisoned
- Expected behavior label: clean_preserving_suppress
- Analysis modules: local_patch, model_output, input_space, feature_space
- Target class: 1

## Metric-Level Observation

- Origin ASR: 0.9742
- Origin PA: 0.2637
- Final CA: 0.8301
- Final ASR: 0.1292
- Final PA: 0.8047
- ASR delta: -0.8450
- PA delta: 0.5410

## Prediction Distribution

| setting | predicted_label | prediction_relation | count | rate | is_final_output |
| --- | --- | --- | --- | --- | --- |
| degradation_restoration_poisoned | 0 | other_wrong | 300 | 0.5859375 | True |
| degradation_restoration_poisoned | 1 | attacker_target | 163 | 0.318359375 | True |
| degradation_restoration_poisoned | 3 | other_wrong | 25 | 0.048828125 | True |
| degradation_restoration_poisoned | 4 | other_wrong | 4 | 0.0078125 | True |
| degradation_restoration_poisoned | 5 | other_wrong | 11 | 0.021484375 | True |
| degradation_restoration_poisoned | 7 | other_wrong | 2 | 0.00390625 | True |
| degradation_restoration_poisoned | 8 | other_wrong | 3 | 0.005859375 | True |
| degradation_restoration_poisoned | 9 | other_wrong | 4 | 0.0078125 | True |

## Prediction Flip Evidence

See `model_output/prediction_flip_summary.csv` for per-sample transitions.

## Pairwise Pipeline Evidence

| comparison | left_setting | right_setting | ca_delta | asr_delta | pa_delta | target_logit_delta | target_probability_delta | true_class_logit_delta | true_class_probability_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D_vs_origin | origin_poisoned | degradation_poisoned | None | -0.15762273901808788 | 0.119140625 | -8.49089327082038 | -0.11812766678383468 | -4.598040776414564 | 0.11561283610314377 |
| R_vs_origin | origin_poisoned | restoration_poisoned | None | -0.20155038759689925 | 0.119140625 | -5.731731414794922 | -0.15366236456740978 | -2.5126447668881156 | 0.10875801087053044 |
| RD_vs_origin | origin_poisoned | degradation_restoration_poisoned | None | -0.8449612403100775 | 0.541015625 | -13.006478106835857 | -0.6462552902577543 | -3.919420891732443 | 0.484311734612768 |
| RD_vs_D | degradation_poisoned | degradation_restoration_poisoned | None | -0.6873385012919897 | 0.421875 | -4.515584836015478 | -0.5281276234739196 | 0.6786198846821208 | 0.36869889850962423 |
| RD_vs_R | restoration_poisoned | degradation_restoration_poisoned | None | -0.6434108527131783 | 0.421875 | -7.274746692040935 | -0.4925929256903445 | -1.4067761248443276 | 0.37555372374223756 |

## Model-Output Evidence

| setting | domain | defense_output_setting | is_final_output | num_samples | ca | asr | asr_num_samples | asr_excluded_target_origin_samples | pa | mean_target_logit | mean_true_class_logit | mean_target_margin | mean_target_probability | mean_true_class_probability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| degradation_poisoned | poisoned | degradation_restoration_poisoned | False | 512 | nan | 0.8165374677002584 | 387 | 125 | 0.3828125 | 11.56332803145051 | 9.266986419883324 | 2.296341613866389 | 0.8482668454481654 | 0.3933474900710756 |
| degradation_restoration_poisoned | poisoned | degradation_restoration_poisoned | True | 512 | nan | 0.12919896640826872 | 387 | 125 | 0.8046875 | 7.047743195435032 | 9.945606304565445 | -2.897863114718348 | 0.3201392219742458 | 0.7620463885806998 |
| origin_poisoned | poisoned | degradation_restoration_poisoned | False | 512 | nan | 0.9741602067183462 | 387 | 125 | 0.263671875 | 20.05422130227089 | 13.865027196297888 | 6.189194110222161 | 0.9663945122320001 | 0.27773465396793184 |
| restoration_poisoned | poisoned | degradation_restoration_poisoned | False | 512 | nan | 0.772609819121447 | 387 | 125 | 0.3828125 | 14.322489887475967 | 11.352382429409772 | 2.9701074649346992 | 0.8127321476645903 | 0.3864926648384623 |

## Feature-Space Evidence

- Mean poison-to-restored feature shift: 3.434719
- Mean restored-poison distance to clean centroid: 2.013713
- Mean restored-clean distance to clean centroid: 1.921805

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
| clean_misclassified | 59 | 0.115234375 | False | True |
| final_clean_broken | 36 | 0.0703125 | False | True |
| attack_failed | 10 | 0.01953125 | True | True |
| final_poisoned_defense_failed_asr | 163 | 0.318359375 | True | True |
| final_poisoned_target_to_other_wrong | 50 | 0.09765625 | True | True |
| final_poisoned_recovered | 402 | 0.78515625 | True | True |
| restoration_poisoned_defense_failed_asr | 419 | 0.818359375 | True | True |
| restoration_poisoned_overcorrected | 17 | 0.033203125 | True | True |
| restoration_poisoned_recovered | 187 | 0.365234375 | True | True |
| degradation_restoration_poisoned_defense_failed_asr | 163 | 0.318359375 | True | True |
| degradation_restoration_poisoned_overcorrected | 50 | 0.09765625 | True | True |
| degradation_restoration_poisoned_recovered | 402 | 0.78515625 | True | True |

## Prediction Outcome Decision

Final outcome label: `clean_preserving_suppress`

Rationale:
- ASR drops by 0.8450 and poisoned accuracy recovers or reaches the high-PA threshold.

## Caveats

- DeBackdoor-style re-synthesis and restoration-prior integrity tests are future hooks, not part of this v3.0 core run.