# Mechanism Summary: its_blended_haze050

## Case

- Attack: Blended
- Trigger metadata: distributed_global
- Degradation: haze (0.5)
- Restoration prior: ConvIR-ITS
- Defense output setting: degradation_restoration_poisoned
- Expected behavior label: target_disruption_without_clean_recovery
- Analysis modules: frequency_global, model_output, feature_space, input_space
- Target class: 1

## Metric-Level Observation

- Origin ASR: 1.0000
- Origin PA: 0.2441
- Final CA: 0.8555
- Final ASR: 0.0362
- Final PA: 0.0059
- ASR delta: -0.9638
- PA delta: -0.2383

## Prediction Distribution

| setting | predicted_label | prediction_relation | count | rate | is_final_output |
| --- | --- | --- | --- | --- | --- |
| degradation_restoration_poisoned | 1 | attacker_target | 17 | 0.033203125 | True |
| degradation_restoration_poisoned | 3 | other_wrong | 5 | 0.009765625 | True |
| degradation_restoration_poisoned | 4 | other_wrong | 44 | 0.0859375 | True |
| degradation_restoration_poisoned | 7 | other_wrong | 3 | 0.005859375 | True |
| degradation_restoration_poisoned | 8 | other_wrong | 3 | 0.005859375 | True |
| degradation_restoration_poisoned | 9 | other_wrong | 440 | 0.859375 | True |

## Prediction Flip Evidence

See `model_output/prediction_flip_summary.csv` for per-sample transitions.

## Pairwise Pipeline Evidence

| comparison | left_setting | right_setting | ca_delta | asr_delta | pa_delta | target_logit_delta | target_probability_delta | true_class_logit_delta | true_class_probability_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D_vs_origin | origin_poisoned | degradation_poisoned | None | -0.01033591731266148 | 0.0078125 | -6.077602160163224 | -0.1530241525033489 | -1.2146603969013086 | 0.023719658885494904 |
| R_vs_origin | origin_poisoned | restoration_poisoned | None | -0.8294573643410853 | -0.19921875 | -10.333725501797744 | -0.8215189116382362 | -3.8929345970827853 | -0.19618274757635276 |
| RD_vs_origin | origin_poisoned | degradation_restoration_poisoned | None | -0.9638242894056848 | -0.23828125 | -11.904234102446935 | -0.9637942299386522 | -4.487591540069843 | -0.2352787151587807 |
| RD_vs_D | degradation_poisoned | degradation_restoration_poisoned | None | -0.9534883720930233 | -0.24609375 | -5.826631942283711 | -0.8107700774353033 | -3.2729311431685346 | -0.2589983740442756 |
| RD_vs_R | restoration_poisoned | degradation_restoration_poisoned | None | -0.1343669250645995 | -0.0390625 | -1.570508600649191 | -0.14227531830041595 | -0.5946569429870578 | -0.03909596758242795 |

## Model-Output Evidence

| setting | domain | defense_output_setting | is_final_output | num_samples | ca | asr | asr_num_samples | asr_excluded_target_origin_samples | pa | mean_target_logit | mean_true_class_logit | mean_target_margin | mean_target_probability | mean_true_class_probability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| degradation_poisoned | poisoned | degradation_restoration_poisoned | False | 512 | nan | 0.9896640826873385 | 387 | 125 | 0.251953125 | 5.776346494443715 | 3.002199261725764 | 2.774147230666131 | 0.8453168220585212 | 0.2682059450834231 |
| degradation_restoration_poisoned | poisoned | degradation_restoration_poisoned | True | 512 | nan | 0.03617571059431524 | 387 | 125 | 0.005859375 | -0.05028544783999678 | -0.2707318814427708 | 0.2204464357419056 | 0.03454674462321794 | 0.009207571039147489 |
| origin_poisoned | poisoned | degradation_restoration_poisoned | False | 512 | nan | 1.0 | 387 | 125 | 0.244140625 | 11.853948654606938 | 4.216859658627072 | 7.637088989838958 | 0.9983409745618701 | 0.2444862861979282 |
| restoration_poisoned | poisoned | degradation_restoration_poisoned | False | 512 | nan | 0.17054263565891473 | 387 | 125 | 0.044921875 | 1.5202231528091943 | 0.323925061544287 | 1.1962980928365141 | 0.1768220629236339 | 0.04830353862157544 |

## Feature-Space Evidence

- Mean poison-to-restored feature shift: 4.979204
- Mean restored-poison distance to clean centroid: 4.676771
- Mean restored-clean distance to clean centroid: 1.798799

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
| clean_misclassified | 44 | 0.0859375 | False | True |
| final_clean_broken | 33 | 0.064453125 | False | True |
| attack_failed | 0 | 0.0 | True | True |
| final_poisoned_defense_failed_asr | 17 | 0.033203125 | True | True |
| final_poisoned_target_to_other_wrong | 495 | 0.966796875 | True | True |
| final_poisoned_recovered | 3 | 0.005859375 | True | True |
| restoration_poisoned_defense_failed_asr | 89 | 0.173828125 | True | True |
| restoration_poisoned_overcorrected | 423 | 0.826171875 | True | True |
| restoration_poisoned_recovered | 23 | 0.044921875 | True | True |
| degradation_restoration_poisoned_defense_failed_asr | 17 | 0.033203125 | True | True |
| degradation_restoration_poisoned_overcorrected | 495 | 0.966796875 | True | True |
| degradation_restoration_poisoned_recovered | 3 | 0.005859375 | True | True |

## Prediction Outcome Decision

Final outcome label: `target_disruption_without_clean_recovery`

Rationale:
- ASR drops by 0.9638, but final poisoned accuracy 0.0059 remains below 0.5000.

## Caveats

- DeBackdoor-style re-synthesis and restoration-prior integrity tests are future hooks, not part of this v3.0 core run.