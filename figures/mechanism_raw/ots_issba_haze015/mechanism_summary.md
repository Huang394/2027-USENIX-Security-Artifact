# Mechanism Summary: ots_issba_haze015

## Case

- Attack: ISSBA
- Trigger metadata: invisible_sample_specific
- Degradation: haze (0.15)
- Restoration prior: ConvIR-OTS
- Defense output setting: degradation_restoration_poisoned
- Expected behavior label: clean_preserving_suppress
- Analysis modules: input_space, frequency_global, feature_invisible, model_output, feature_space
- Target class: 1

## Metric-Level Observation

- Origin ASR: 1.0000
- Origin PA: 0.2441
- Final CA: 0.8887
- Final ASR: 0.0155
- Final PA: 0.8887
- ASR delta: -0.9845
- PA delta: 0.6445

## Prediction Distribution

| setting | predicted_label | prediction_relation | count | rate | is_final_output |
| --- | --- | --- | --- | --- | --- |
| degradation_restoration_poisoned | 0 | other_wrong | 351 | 0.685546875 | True |
| degradation_restoration_poisoned | 1 | attacker_target | 111 | 0.216796875 | True |
| degradation_restoration_poisoned | 2 | other_wrong | 4 | 0.0078125 | True |
| degradation_restoration_poisoned | 3 | other_wrong | 24 | 0.046875 | True |
| degradation_restoration_poisoned | 4 | other_wrong | 2 | 0.00390625 | True |
| degradation_restoration_poisoned | 5 | other_wrong | 4 | 0.0078125 | True |
| degradation_restoration_poisoned | 6 | other_wrong | 3 | 0.005859375 | True |
| degradation_restoration_poisoned | 7 | other_wrong | 2 | 0.00390625 | True |
| degradation_restoration_poisoned | 8 | other_wrong | 7 | 0.013671875 | True |
| degradation_restoration_poisoned | 9 | other_wrong | 4 | 0.0078125 | True |

## Prediction Flip Evidence

See `model_output/prediction_flip_summary.csv` for per-sample transitions.

## Pairwise Pipeline Evidence

| comparison | left_setting | right_setting | ca_delta | asr_delta | pa_delta | target_logit_delta | target_probability_delta | true_class_logit_delta | true_class_probability_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D_vs_origin | origin_poisoned | degradation_poisoned | None | 0.0 | 0.0 | -0.5440411008894444 | -0.0006506805075332522 | 0.5171991673560115 | 0.0006564565729572092 |
| R_vs_origin | origin_poisoned | restoration_poisoned | None | -0.9844961240310077 | 0.63671875 | -14.988205207308056 | -0.7846129484586032 | 1.4539103135757614 | 0.5843460543434903 |
| RD_vs_origin | origin_poisoned | degradation_restoration_poisoned | None | -0.9844961240310077 | 0.64453125 | -15.00488852665876 | -0.7856006885391742 | 1.4990426491131075 | 0.587942624978942 |
| RD_vs_D | degradation_poisoned | degradation_restoration_poisoned | None | -0.9844961240310077 | 0.64453125 | -14.460847425769316 | -0.784950008031641 | 0.981843481757096 | 0.5872861684059848 |
| RD_vs_R | restoration_poisoned | degradation_restoration_poisoned | None | 0.0 | 0.0078125 | -0.01668331935070455 | -0.000987740080571009 | 0.04513233553734608 | 0.0035965706354517124 |

## Model-Output Evidence

| setting | domain | defense_output_setting | is_final_output | num_samples | ca | asr | asr_num_samples | asr_excluded_target_origin_samples | pa | mean_target_logit | mean_true_class_logit | mean_target_margin | mean_target_probability | mean_true_class_probability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| degradation_poisoned | poisoned | degradation_restoration_poisoned | False | 512 | nan | 1.0 | 387 | 125 | 0.244140625 | 17.8062417563051 | 7.953078807724523 | 9.853162968531251 | 0.9992466422263533 | 0.24488080229457065 |
| degradation_restoration_poisoned | poisoned | degradation_restoration_poisoned | True | 512 | nan | 0.015503875968992248 | 387 | 125 | 0.888671875 | 3.3453943305357825 | 8.93492228948162 | -5.589527958887629 | 0.21429663419471232 | 0.8321669707005555 |
| origin_poisoned | poisoned | degradation_restoration_poisoned | False | 512 | nan | 1.0 | 387 | 125 | 0.244140625 | 18.350282857194543 | 7.435879640368512 | 10.914403211325407 | 0.9998973227338865 | 0.24422434572161345 |
| restoration_poisoned | poisoned | degradation_restoration_poisoned | False | 512 | nan | 0.015503875968992248 | 387 | 125 | 0.880859375 | 3.362077649886487 | 8.889789953944273 | -5.5277123026316985 | 0.21528437427528332 | 0.8285704000651037 |

## Feature-Space Evidence

- Mean poison-to-restored feature shift: 5.538711
- Mean restored-poison distance to clean centroid: 1.724728
- Mean restored-clean distance to clean centroid: 1.720627

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
| clean_misclassified | 48 | 0.09375 | False | True |
| final_clean_broken | 11 | 0.021484375 | False | True |
| attack_failed | 0 | 0.0 | True | True |
| final_poisoned_defense_failed_asr | 111 | 0.216796875 | True | True |
| final_poisoned_target_to_other_wrong | 51 | 0.099609375 | True | True |
| final_poisoned_recovered | 455 | 0.888671875 | True | True |
| restoration_poisoned_defense_failed_asr | 110 | 0.21484375 | True | True |
| restoration_poisoned_overcorrected | 55 | 0.107421875 | True | True |
| restoration_poisoned_recovered | 451 | 0.880859375 | True | True |
| degradation_restoration_poisoned_defense_failed_asr | 111 | 0.216796875 | True | True |
| degradation_restoration_poisoned_overcorrected | 51 | 0.099609375 | True | True |
| degradation_restoration_poisoned_recovered | 455 | 0.888671875 | True | True |

## Prediction Outcome Decision

Final outcome label: `clean_preserving_suppress`

Rationale:
- ASR drops by 0.9845 and poisoned accuracy recovers or reaches the high-PA threshold.

## Caveats

- DeBackdoor-style re-synthesis and restoration-prior integrity tests are future hooks, not part of this v3.0 core run.