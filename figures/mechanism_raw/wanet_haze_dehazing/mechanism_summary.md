# Mechanism Summary: wanet_haze_dehazing

## Case

- Attack: WaNet
- Trigger metadata: geometric_warp
- Degradation: haze (0.5)
- Restoration prior: ConvIR-ITS
- Defense output setting: degradation_restoration_poisoned
- Expected behavior label: preserve_or_weak_effect
- Analysis modules: geometry_structure, model_output, feature_space, input_space
- Target class: 0

## Metric-Level Observation

- Origin ASR: 0.9760
- Origin PA: 0.7617
- Final CA: 0.8281
- Final ASR: 0.8720
- Final PA: 0.7422
- ASR delta: -0.1040
- PA delta: -0.0195

## Prediction Distribution

| setting | predicted_label | prediction_relation | count | rate | is_final_output |
| --- | --- | --- | --- | --- | --- |
| degradation_restoration_poisoned | 0 | attacker_target | 481 | 0.939453125 | True |
| degradation_restoration_poisoned | 1 | ground_truth | 8 | 0.015625 | True |
| degradation_restoration_poisoned | 4 | other_wrong | 1 | 0.001953125 | True |
| degradation_restoration_poisoned | 5 | other_wrong | 1 | 0.001953125 | True |
| degradation_restoration_poisoned | 8 | other_wrong | 1 | 0.001953125 | True |
| degradation_restoration_poisoned | 9 | other_wrong | 20 | 0.0390625 | True |

## Prediction Flip Evidence

See `model_output/prediction_flip_summary.csv` for per-sample transitions.

## Pairwise Pipeline Evidence

| comparison | left_setting | right_setting | ca_delta | asr_delta | pa_delta | target_logit_delta | target_probability_delta | true_class_logit_delta | true_class_probability_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D_vs_origin | origin_poisoned | degradation_poisoned | None | 0.02400000000000002 | -0.005859375 | -3.50782404653728 | 0.0021736552371294238 | -3.482361821399536 | -0.008663928488920614 |
| R_vs_origin | origin_poisoned | restoration_poisoned | None | -0.19999999999999996 | -0.05078125 | -5.133917833591113 | -0.12439283597208117 | -3.6118426191969775 | -0.05472213951929772 |
| RD_vs_origin | origin_poisoned | degradation_restoration_poisoned | None | -0.10399999999999998 | -0.01953125 | -4.867345992592163 | -0.06406606831637873 | -3.399330871005077 | -0.022096593858938363 |
| RD_vs_D | degradation_poisoned | degradation_restoration_poisoned | None | -0.128 | -0.013671875 | -1.3595219460548833 | -0.06623972355350816 | 0.08303095039445907 | -0.013432665370017749 |
| RD_vs_R | restoration_poisoned | degradation_restoration_poisoned | None | 0.09599999999999997 | 0.03125 | 0.2665718409989495 | 0.06032676765570244 | 0.21251174819190055 | 0.032625545660359356 |

## Model-Output Evidence

| setting | domain | defense_output_setting | is_final_output | num_samples | ca | asr | asr_num_samples | asr_excluded_target_origin_samples | pa | mean_target_logit | mean_true_class_logit | mean_target_margin | mean_target_probability | mean_true_class_probability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| degradation_poisoned | poisoned | degradation_restoration_poisoned | False | 512 | nan | 1.0 | 125 | 387 | 0.755859375 | 15.041755264624953 | 11.952896311937366 | 3.0888589592650533 | 0.9945710675092414 | 0.7539116994306734 |
| degradation_restoration_poisoned | poisoned | degradation_restoration_poisoned | True | 512 | nan | 0.872 | 125 | 387 | 0.7421875 | 13.68223331857007 | 12.035927262331825 | 1.6463060528039932 | 0.9283313439557332 | 0.7404790340606556 |
| origin_poisoned | poisoned | degradation_restoration_poisoned | False | 512 | nan | 0.976 | 125 | 387 | 0.76171875 | 18.549579311162233 | 15.435258133336902 | 3.11432116990909 | 0.992397412272112 | 0.762575627919594 |
| restoration_poisoned | poisoned | degradation_restoration_poisoned | False | 512 | nan | 0.776 | 125 | 387 | 0.7109375 | 13.41566147757112 | 11.823415514139924 | 1.5922459734138101 | 0.8680045763000308 | 0.7078534884002963 |

## Feature-Space Evidence

- Mean poison-to-restored feature shift: 114.259835
- Mean restored-poison distance to clean centroid: 211.874493
- Mean restored-clean distance to clean centroid: 221.137500

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
| final_clean_broken | 38 | 0.07421875 | False | True |
| attack_failed | 3 | 0.005859375 | True | True |
| final_poisoned_defense_failed_asr | 481 | 0.939453125 | True | True |
| final_poisoned_target_to_other_wrong | 23 | 0.044921875 | True | True |
| final_poisoned_recovered | 377 | 0.736328125 | True | True |
| restoration_poisoned_defense_failed_asr | 451 | 0.880859375 | True | True |
| restoration_poisoned_overcorrected | 51 | 0.099609375 | True | True |
| restoration_poisoned_recovered | 362 | 0.70703125 | True | True |
| degradation_restoration_poisoned_defense_failed_asr | 481 | 0.939453125 | True | True |
| degradation_restoration_poisoned_overcorrected | 23 | 0.044921875 | True | True |
| degradation_restoration_poisoned_recovered | 377 | 0.736328125 | True | True |

## Prediction Outcome Decision

Final outcome label: `preserve_or_weak_effect`

Rationale:
- Final ASR 0.8720 remains high while final CA 0.8281 is acceptable.

## Caveats

- DeBackdoor-style re-synthesis and restoration-prior integrity tests are future hooks, not part of this v3.0 core run.