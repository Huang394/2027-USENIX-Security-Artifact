# Mechanism Summary: gopro_badnet_blur3

## Case

- Attack: BadNet
- Trigger metadata: localized_patch
- Degradation: motion_blur (3.0)
- Restoration prior: ConvIR-GoPro
- Defense output setting: degradation_restoration_poisoned
- Expected behavior label: clean_preserving_suppress
- Analysis modules: local_patch, model_output, input_space, feature_space
- Target class: 1

## Metric-Level Observation

- Origin ASR: 0.9742
- Origin PA: 0.2637
- Final CA: 0.8379
- Final ASR: 0.0491
- Final PA: 0.8320
- ASR delta: -0.9251
- PA delta: 0.5684

## Prediction Distribution

| setting | predicted_label | prediction_relation | count | rate | is_final_output |
| --- | --- | --- | --- | --- | --- |
| degradation_restoration_poisoned | 0 | ground_truth | 313 | 0.611328125 | True |
| degradation_restoration_poisoned | 1 | attacker_target | 132 | 0.2578125 | True |
| degradation_restoration_poisoned | 3 | other_wrong | 35 | 0.068359375 | True |
| degradation_restoration_poisoned | 4 | other_wrong | 2 | 0.00390625 | True |
| degradation_restoration_poisoned | 5 | other_wrong | 15 | 0.029296875 | True |
| degradation_restoration_poisoned | 8 | other_wrong | 10 | 0.01953125 | True |
| degradation_restoration_poisoned | 9 | other_wrong | 5 | 0.009765625 | True |

## Prediction Flip Evidence

See `model_output/prediction_flip_summary.csv` for per-sample transitions.

## Pairwise Pipeline Evidence

| comparison | left_setting | right_setting | ca_delta | asr_delta | pa_delta | target_logit_delta | target_probability_delta | true_class_logit_delta | true_class_probability_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D_vs_origin | origin_poisoned | degradation_poisoned | None | -0.5142118863049095 | 0.380859375 | -9.15191661939025 | -0.3907579555619265 | -2.523710599693004 | 0.36610432369412627 |
| R_vs_origin | origin_poisoned | restoration_poisoned | None | -0.958656330749354 | 0.5625 | -15.063049554359168 | -0.7425474069403402 | -4.168532814423088 | 0.5029771256185082 |
| RD_vs_origin | origin_poisoned | degradation_restoration_poisoned | None | -0.9250645994832041 | 0.568359375 | -14.116767366649583 | -0.699719479265547 | -3.776757995481603 | 0.5114743562758791 |
| RD_vs_D | degradation_poisoned | degradation_restoration_poisoned | None | -0.4108527131782946 | 0.1875 | -4.964850747259334 | -0.3089615237036205 | -1.2530473957885988 | 0.14537003258175285 |
| RD_vs_R | restoration_poisoned | degradation_restoration_poisoned | None | 0.03359173126614987 | 0.005859375 | 0.9462821877095848 | 0.04282792767479318 | 0.39177481894148514 | 0.008497230657370958 |

## Model-Output Evidence

| setting | domain | defense_output_setting | is_final_output | num_samples | ca | asr | asr_num_samples | asr_excluded_target_origin_samples | pa | mean_target_logit | mean_true_class_logit | mean_target_margin | mean_target_probability | mean_true_class_probability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| degradation_poisoned | poisoned | degradation_restoration_poisoned | False | 512 | nan | 0.4599483204134367 | 387 | 125 | 0.64453125 | 10.90230468288064 | 11.341316596604884 | -0.4390119151212275 | 0.5756365566700736 | 0.6438389776620581 |
| degradation_restoration_poisoned | poisoned | degradation_restoration_poisoned | True | 512 | nan | 0.04909560723514212 | 387 | 125 | 0.83203125 | 5.937453935621306 | 10.088269200816285 | -4.150815261993557 | 0.2666750329664531 | 0.789209010243811 |
| origin_poisoned | poisoned | degradation_restoration_poisoned | False | 512 | nan | 0.9741602067183462 | 387 | 125 | 0.263671875 | 20.05422130227089 | 13.865027196297888 | 6.189194110222161 | 0.9663945122320001 | 0.27773465396793184 |
| restoration_poisoned | poisoned | degradation_restoration_poisoned | False | 512 | nan | 0.015503875968992248 | 387 | 125 | 0.826171875 | 4.9911717479117215 | 9.6964943818748 | -4.705322623485699 | 0.22384710529165994 | 0.78071177958644 |

## Feature-Space Evidence

- Mean poison-to-restored feature shift: 3.833245
- Mean restored-poison distance to clean centroid: 1.967882
- Mean restored-clean distance to clean centroid: 1.945555

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
| final_clean_broken | 38 | 0.07421875 | False | True |
| attack_failed | 10 | 0.01953125 | True | True |
| final_poisoned_defense_failed_asr | 132 | 0.2578125 | True | True |
| final_poisoned_target_to_other_wrong | 67 | 0.130859375 | True | True |
| final_poisoned_recovered | 416 | 0.8125 | True | True |
| restoration_poisoned_defense_failed_asr | 110 | 0.21484375 | True | True |
| restoration_poisoned_overcorrected | 83 | 0.162109375 | True | True |
| restoration_poisoned_recovered | 413 | 0.806640625 | True | True |
| degradation_restoration_poisoned_defense_failed_asr | 132 | 0.2578125 | True | True |
| degradation_restoration_poisoned_overcorrected | 67 | 0.130859375 | True | True |
| degradation_restoration_poisoned_recovered | 416 | 0.8125 | True | True |

## Prediction Outcome Decision

Final outcome label: `clean_preserving_suppress`

Rationale:
- ASR drops by 0.9251 and poisoned accuracy recovers or reaches the high-PA threshold.

## Caveats

- DeBackdoor-style re-synthesis and restoration-prior integrity tests are future hooks, not part of this v3.0 core run.