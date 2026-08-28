"""Main mechanism analysis runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch import nn

from stage4_analysis.behavior import BehaviorDecision, decide_behavior
from stage4_analysis.config import CaseConfig, resolve_analysis_modules, thresholds_from_config
from stage4_analysis.data import load_image_dataset, sample_dataset_by_indices, sample_paired_batch
from stage4_analysis.metrics import compute_residual_metrics, cosine_similarity, l2_distance
from stage4_analysis.models import FeatureExtractor, load_classifier, resolve_device
from stage4_analysis.pipeline_settings import (
    PIPELINE_SETTINGS,
    resolve_defense_output_setting,
    validate_required_inputs,
)
from stage4_analysis.pipelines import build_pipeline_images, load_restorer
from stage4_analysis.visualization import save_bar_plot, save_line_plot, save_tensor_grid
from stage4_analysis.visual_evidence import VisualEvidenceResult, write_visual_evidence


class Stage4Analyzer:
    def __init__(self, config: CaseConfig) -> None:
        self.config = config
        self.case_dir = config.output_dir / config.name
        self.device = resolve_device(config.device)
        modules = resolve_analysis_modules(config.analysis_modules, config.trigger_metadata)
        self.config = CaseConfig(**{**config.__dict__, "analysis_modules": modules})
        self.defense_output = resolve_defense_output_setting(self.config)
        validate_required_inputs(self.config, self.defense_output)

    def run(self) -> None:
        self._prepare_dirs()

        clean_dataset = load_image_dataset(self.config.clean_dataset_path, self.config.image_size)
        poisoned_dataset = load_image_dataset(self.config.poisoned_dataset_path, self.config.image_size)
        batch = sample_paired_batch(
            clean_dataset,
            poisoned_dataset,
            self.config.num_samples,
            self.config.true_class_filter,
        )
        clean = batch.clean.to(self.device)
        poisoned = batch.poisoned.to(self.device)
        labels = batch.labels

        classifier = load_classifier(
            self.config.backdoored_model_path,
            self.device,
            self.config.model_arch,
            self.config.num_classes,
        )
        restorer = load_restorer(self.config.restoration_model_path, self.device)
        images = build_pipeline_images(
            clean,
            poisoned,
            self.config.degradation,
            self.config.strength,
            restorer,
            self.config.degradation_seed,
            self.config.batch_size,
        )
        self._override_pipeline_images(images, batch.indices, batch.labels)
        self._write_case_config()

        if self.config.save_images:
            self._write_input_space(images)

        logit_df = self._write_model_output(classifier, images, labels, batch.indices, batch.paths)
        failure_summary = self._write_failure_cases(logit_df, images)
        feature_df = self._write_feature_space(classifier, images, labels)
        self._write_trigger_modules(images, logit_df, feature_df)
        metrics_summary = self._write_metrics_summary(logit_df)
        pairwise = self._write_pairwise_pipeline_comparison(metrics_summary)
        decision = self._write_behavior_decision(metrics_summary, pairwise)
        visual_result = self._write_visual_evidence(classifier, images, labels, logit_df)
        self._write_summary(
            feature_df,
            failure_summary,
            metrics_summary,
            pairwise,
            decision,
            visual_result,
        )

    def _prepare_dirs(self) -> None:
        for name in (
            "input_space",
            "trigger_representation",
            "model_output",
            "feature_space",
            "failure_cases",
            "figures",
            "visual_evidence",
        ):
            (self.case_dir / name).mkdir(parents=True, exist_ok=True)

    def _write_case_config(self) -> None:
        path = self.case_dir / "case_config.json"
        data = self.config.to_json_dict()
        data["resolved_defense_output_setting"] = self.defense_output.poisoned_setting
        data["resolved_final_clean_setting"] = self.defense_output.clean_setting
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)

    def _override_pipeline_images(
        self,
        images: dict[str, torch.Tensor],
        indices: list[int],
        labels: torch.Tensor,
    ) -> None:
        overrides = {
            "degraded_clean": self.config.degraded_clean_dataset_path,
            "degraded_poisoned": self.config.degraded_poisoned_dataset_path,
            "restored_clean": self.config.restored_clean_dataset_path,
            "restored_poisoned": self.config.restored_poisoned_dataset_path,
            "restored_degraded_clean": self.config.restored_degraded_clean_dataset_path,
            "restored_degraded_poisoned": self.config.restored_degraded_poisoned_dataset_path,
        }
        for image_key, dataset_path in overrides.items():
            if dataset_path is None:
                continue
            dataset = load_image_dataset(dataset_path, self.config.image_size)
            images[image_key] = sample_dataset_by_indices(
                dataset,
                indices,
                labels,
                image_key,
            )

    def _final_clean_key(self) -> str:
        return self.defense_output.clean_image_key

    def _final_poisoned_key(self) -> str:
        return self.defense_output.poisoned_image_key

    def _write_input_space(self, images: dict[str, torch.Tensor]) -> None:
        output_dir = self.case_dir / "input_space"
        comparisons = [
            ("poisoned_to_final_defense_poisoned", "poisoned", self._final_poisoned_key()),
            ("clean_to_final_defense_clean", "clean", self._final_clean_key()),
            ("poisoned_to_restored_poisoned", "poisoned", "restored_poisoned"),
            (
                "degraded_poisoned_to_restored_degraded_poisoned",
                "degraded_poisoned",
                "restored_degraded_poisoned",
            ),
            ("poisoned_to_restored_degraded_poisoned", "poisoned", "restored_degraded_poisoned"),
            ("restored_poisoned_to_restored_degraded_poisoned", "restored_poisoned", "restored_degraded_poisoned"),
        ]
        rows = []
        for name, left, right in comparisons:
            if left not in images or right not in images:
                continue
            metrics = compute_residual_metrics(images[left], images[right])
            residual = torch.abs(images[left] - images[right])
            rows.append({"comparison": name, **metrics.__dict__})
            save_tensor_grid(
                [images[left], images[right], residual],
                [left, right, "absolute residual"],
                output_dir / f"{name}_grid.png",
            )
        pd.DataFrame(rows).to_csv(output_dir / "input_residual_metrics.csv", index=False)
        save_tensor_grid(
            [
                images["clean"],
                images["poisoned"],
                images["degraded_poisoned"],
                images[self._final_poisoned_key()],
                torch.abs(images["poisoned"] - images[self._final_poisoned_key()]),
            ],
            ["clean", "poisoned", "D(poisoned)", "final poisoned", "residual"],
            self.case_dir / "figures" / "sample_grid.png",
        )

    @torch.inference_mode()
    def _write_model_output(
        self,
        classifier: nn.Module,
        images: dict[str, torch.Tensor],
        labels: torch.Tensor,
        original_indices: list[int],
        paths: list[str],
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        target_class = self.config.target_class
        for setting, image_key in PIPELINE_SETTINGS.items():
            if image_key not in images:
                continue
            for start, end in _batch_slices(labels.shape[0], self.config.batch_size):
                batch_images = images[image_key][start:end].to(self.device)
                batch_labels = labels[start:end].to(self.device)
                logits = classifier(batch_images)
                probabilities = torch.softmax(logits, dim=1)
                predicted = torch.argmax(logits, dim=1)
                if target_class is None:
                    target_indices = predicted
                else:
                    target_indices = torch.full_like(batch_labels, target_class)
                true_logits = logits.gather(1, batch_labels[:, None]).squeeze(1)
                target_logits = logits.gather(1, target_indices[:, None]).squeeze(1)
                true_probs = probabilities.gather(1, batch_labels[:, None]).squeeze(1)
                target_probs = probabilities.gather(1, target_indices[:, None]).squeeze(1)
                for batch_idx in range(batch_labels.shape[0]):
                    idx = start + batch_idx
                    rows.append(
                        {
                            "sample_index": idx,
                            "original_index": int(original_indices[idx]),
                            "source_path": paths[idx],
                            "setting": setting,
                            "domain": "poisoned" if "poisoned" in setting else "clean",
                            "true_label": int(batch_labels[batch_idx].item()),
                            "target_class": None if target_class is None else int(target_class),
                            "predicted_label": int(predicted[batch_idx].item()),
                            "target_logit": float(target_logits[batch_idx].item()),
                            "true_class_logit": float(true_logits[batch_idx].item()),
                            "target_margin": float(
                                (target_logits[batch_idx] - true_logits[batch_idx]).item()
                            ),
                            "target_probability": float(target_probs[batch_idx].item()),
                            "true_class_probability": float(true_probs[batch_idx].item()),
                        }
                    )
        df = pd.DataFrame(rows)
        output_dir = self.case_dir / "model_output"
        df.to_csv(output_dir / "logit_trajectory.csv", index=False)
        if self.config.prediction_distribution:
            self._prediction_distribution(df).to_csv(
                output_dir / "prediction_distribution.csv",
                index=False,
            )
        self._prediction_flip_summary(df).to_csv(
            output_dir / "prediction_flip_summary.csv",
            index=False,
        )
        aggregated = (
            df.groupby("setting", as_index=False)[
                ["target_logit", "true_class_logit", "target_margin", "target_probability"]
            ]
            .mean()
            .sort_values("setting")
        )
        save_line_plot(
            aggregated,
            "setting",
            ["target_logit", "true_class_logit", "target_margin"],
            output_dir / "logit_margin_boxplot.png",
            "Mean logit trajectory",
        )
        save_line_plot(
            aggregated,
            "setting",
            ["target_probability"],
            output_dir / "target_probability_curve.png",
            "Mean target probability",
        )
        return df

    def _prediction_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        origin = df[df["setting"] == "origin_poisoned"][["sample_index", "predicted_label"]]
        origin = origin.rename(columns={"predicted_label": "origin_predicted_label"})
        merged = df.merge(origin, on="sample_index")
        grouped = merged.groupby("setting")
        return grouped.apply(
            lambda group: pd.Series(
                {
                    "num_samples": int(len(group)),
                    "prediction_change_rate": float(
                        (group["predicted_label"] != group["origin_predicted_label"]).mean()
                    ),
                    "clean_accuracy_or_pa": float(
                        (group["predicted_label"] == group["true_label"]).mean()
                    ),
                    "asr": self._asr(group),
                    "mean_target_logit": float(group["target_logit"].mean()),
                    "mean_true_class_logit": float(group["true_class_logit"].mean()),
                    "mean_target_margin": float(group["target_margin"].mean()),
                }
            )
        ).reset_index()

    def _metrics_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for setting in sorted(df["setting"].unique()):
            group = df[df["setting"] == setting]
            clean = group[group["domain"] == "clean"]
            poisoned = group[group["domain"] == "poisoned"]
            active = poisoned if not poisoned.empty else clean
            rows.append(
                {
                    "setting": setting,
                    "domain": "poisoned" if not poisoned.empty else "clean",
                    "defense_output_setting": self.defense_output.poisoned_setting,
                    "is_final_output": setting
                    in {self.defense_output.poisoned_setting, self.defense_output.clean_setting},
                    "num_samples": int(len(active)),
                    "ca": _accuracy(clean),
                    "asr": self._asr(group),
                    "asr_num_samples": self._asr_num_samples(group),
                    "asr_excluded_target_origin_samples": self._asr_excluded_count(group),
                    "pa": _accuracy(poisoned),
                    "mean_target_logit": _mean_or_none(active, "target_logit"),
                    "mean_true_class_logit": _mean_or_none(active, "true_class_logit"),
                    "mean_target_margin": _mean_or_none(active, "target_margin"),
                    "mean_target_probability": _mean_or_none(active, "target_probability"),
                    "mean_true_class_probability": _mean_or_none(
                        active,
                        "true_class_probability",
                    ),
                }
            )
        return pd.DataFrame(rows)

    def _prediction_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        poisoned = df[df["domain"] == "poisoned"]
        rows = []
        for (setting, predicted_label), group in poisoned.groupby(
            ["setting", "predicted_label"],
        ):
            relation = self._prediction_relation(int(predicted_label), group["true_label"])
            total = len(poisoned[poisoned["setting"] == setting])
            rows.append(
                {
                    "setting": setting,
                    "predicted_label": int(predicted_label),
                    "prediction_relation": relation,
                    "count": int(len(group)),
                    "rate": _safe_rate(len(group), total),
                    "is_final_output": setting == self.defense_output.poisoned_setting,
                }
            )
        return pd.DataFrame(rows)

    def _prediction_flip_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        origin = df[df["setting"] == "origin_poisoned"][
            ["sample_index", "predicted_label"]
        ].rename(columns={"predicted_label": "origin_poisoned_predicted_label"})
        poisoned = df[df["domain"] == "poisoned"].merge(origin, on="sample_index", how="left")
        rows = []
        for row in poisoned.itertuples(index=False):
            origin_pred = int(row.origin_poisoned_predicted_label)
            final_pred = int(row.predicted_label)
            true_label = int(row.true_label)
            rows.append(
                {
                    "sample_index": int(row.sample_index),
                    "original_index": int(row.original_index),
                    "setting": str(row.setting),
                    "is_final_output": row.setting == self.defense_output.poisoned_setting,
                    "true_label": true_label,
                    "target_class": self.config.target_class,
                    "origin_poisoned_predicted_label": origin_pred,
                    "predicted_label": final_pred,
                    "transition": self._flip_transition(origin_pred, final_pred, true_label),
                }
            )
        return pd.DataFrame(rows)

    def _prediction_relation(self, predicted_label: int, true_labels: pd.Series) -> str:
        if self.config.target_class is not None and predicted_label == self.config.target_class:
            return "attacker_target"
        if bool((true_labels == predicted_label).all()):
            return "ground_truth"
        return "other_wrong"

    def _flip_transition(self, origin_pred: int, final_pred: int, true_label: int) -> str:
        if self.config.target_class is None:
            return "target_unknown"
        target_class = self.config.target_class
        if origin_pred == target_class and final_pred == true_label:
            return "target_to_true"
        if origin_pred == target_class and final_pred == target_class:
            return "target_unchanged"
        if origin_pred == target_class and final_pred != true_label:
            return "target_to_other_wrong"
        if origin_pred == true_label and final_pred != true_label:
            return "true_to_wrong"
        if final_pred == true_label:
            return "recovered_or_true_preserved"
        return "wrong_to_wrong"

    def _asr(self, group: pd.DataFrame) -> float | None:
        asr_population = self._asr_population(group)
        if asr_population.empty:
            return None
        return float((asr_population["predicted_label"] == self.config.target_class).mean())

    def _asr_num_samples(self, group: pd.DataFrame) -> int | None:
        if self.config.target_class is None:
            return None
        return int(len(self._asr_population(group)))

    def _asr_excluded_count(self, group: pd.DataFrame) -> int | None:
        if self.config.target_class is None:
            return None
        poisoned = group[group["domain"] == "poisoned"]
        return int((poisoned["true_label"] == self.config.target_class).sum())

    def _asr_population(self, group: pd.DataFrame) -> pd.DataFrame:
        if self.config.target_class is None:
            return pd.DataFrame()
        poisoned = group[group["domain"] == "poisoned"]
        return poisoned[poisoned["true_label"] != self.config.target_class]

    def _write_failure_cases(
        self,
        logit_df: pd.DataFrame,
        images: dict[str, torch.Tensor],
    ) -> pd.DataFrame:
        output_dir = self.case_dir / "failure_cases"
        output_dir.mkdir(parents=True, exist_ok=True)
        matrix = _build_outcome_matrix(logit_df)
        matrix.to_csv(output_dir / "sample_outcome_matrix.csv", index=False)

        cases = {
            "clean_misclassified": matrix[
                matrix["origin_clean_predicted_label"] != matrix["true_label"]
            ],
            "final_clean_broken": _clean_broken_case(
                matrix,
                self.defense_output.clean_setting,
            ),
            **_target_dependent_failure_cases(
                matrix,
                self.config.target_class,
                self.defense_output.poisoned_setting,
            ),
        }

        target_dependent_names = {
            "attack_failed",
            "final_poisoned_defense_failed_asr",
            "final_poisoned_target_to_other_wrong",
            "final_poisoned_recovered",
            "restoration_poisoned_defense_failed_asr",
            "restoration_poisoned_overcorrected",
            "restoration_poisoned_recovered",
            "degradation_restoration_poisoned_defense_failed_asr",
            "degradation_restoration_poisoned_overcorrected",
            "degradation_restoration_poisoned_recovered",
        }
        rows = []
        for name, case_df in cases.items():
            case_df.to_csv(output_dir / f"{name}.csv", index=False)
            rows.append(
                {
                    "failure_case": name,
                    "num_samples": int(len(case_df)),
                    "rate": _safe_rate(len(case_df), len(matrix)),
                    "requires_target_class": name in target_dependent_names,
                    "target_class_available": self.config.target_class is not None,
                }
            )
            if self.config.save_images and not case_df.empty:
                self._write_failure_case_grid(name, case_df, images, output_dir / "example_grids")

        summary = pd.DataFrame(rows)
        summary.to_csv(output_dir / "failure_case_summary.csv", index=False)
        return summary

    def _write_failure_case_grid(
        self,
        name: str,
        case_df: pd.DataFrame,
        images: dict[str, torch.Tensor],
        output_dir: Path,
    ) -> None:
        selected = [int(index) for index in case_df["sample_index"].head(8).tolist()]
        image_keys = [
            "clean",
            "poisoned",
            self._final_clean_key(),
            self._final_poisoned_key(),
        ]
        available_keys = [key for key in image_keys if key in images]
        grid_tensors = [
            torch.stack([images[key][index] for index in selected])
            for key in available_keys
        ]
        save_tensor_grid(grid_tensors, available_keys, output_dir / f"{name}.png")

    def _write_feature_space(
        self,
        classifier: nn.Module,
        images: dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> pd.DataFrame:
        extractor = FeatureExtractor(classifier, self.config.layers)
        rows: list[dict[str, Any]] = []
        try:
            final_clean_key = self._final_clean_key()
            final_poisoned_key = self._final_poisoned_key()
            feature_count = min(
                labels.shape[0],
                max(1, self.config.centroid_sample_count),
            )
            feature_labels = labels[:feature_count]
            desired_keys = {
                "clean",
                "poisoned",
                "degraded_poisoned",
                "restored_poisoned",
                "restored_degraded_poisoned",
                final_poisoned_key,
                final_clean_key,
            }
            features = {}
            for name, tensor in images.items():
                if name in desired_keys:
                    features[name] = _extract_features_batched(
                        extractor,
                        tensor[:feature_count],
                        self.config.layers,
                        self.config.batch_size,
                        self.device,
                    )
        finally:
            extractor.close()
        labels_cpu = feature_labels.detach().cpu()
        for layer in self.config.layers:
            clean_features = features["clean"][layer]
            final_clean_features = features[self._final_clean_key()][layer]
            for label in labels_cpu.unique():
                mask = labels_cpu == label
                clean_centroid = clean_features[mask].mean(dim=0, keepdim=True)
                if self.config.target_class is None:
                    target_centroid = clean_features.mean(dim=0, keepdim=True)
                else:
                    target_mask = labels_cpu == self.config.target_class
                    target_centroid = (
                        clean_features[target_mask].mean(dim=0, keepdim=True)
                        if bool(target_mask.any())
                        else clean_features.mean(dim=0, keepdim=True)
                    )
                for sample_index in torch.where(mask)[0]:
                    idx = int(sample_index.item())
                    poison_f = features["poisoned"][layer][idx : idx + 1]
                    final_poison_f = features[self._final_poisoned_key()][layer][idx : idx + 1]
                    final_clean_f = final_clean_features[idx : idx + 1]
                    rows.append(
                        {
                            "sample_index": idx,
                            "layer": layer,
                            "true_label": int(label.item()),
                            "feature_shift_poison_to_restored_poison": float(
                                l2_distance(poison_f, final_poison_f).item()
                            ),
                            "cosine_poison_to_restored_poison": float(
                                cosine_similarity(poison_f, final_poison_f).item()
                            ),
                            "dist_restored_poison_to_clean_centroid": float(
                                l2_distance(final_poison_f, clean_centroid).item()
                            ),
                            "dist_poison_to_clean_centroid": float(
                                l2_distance(poison_f, clean_centroid).item()
                            ),
                            "dist_restored_poison_to_target_centroid": float(
                                l2_distance(final_poison_f, target_centroid).item()
                            ),
                            "dist_poison_to_target_centroid": float(
                                l2_distance(poison_f, target_centroid).item()
                            ),
                            "dist_restored_clean_to_clean_centroid": float(
                                l2_distance(final_clean_f, clean_centroid).item()
                            ),
                        }
                    )
        df = pd.DataFrame(rows)
        output_dir = self.case_dir / "feature_space"
        df.to_csv(output_dir / "feature_distance.csv", index=False)
        self._write_feature_pairwise_delta(features, output_dir)
        if self.config.save_features:
            torch.save(features, output_dir / "raw_features.pt")
        summary = df.groupby("layer", as_index=False).mean(numeric_only=True)
        save_bar_plot(
            summary,
            "layer",
            "feature_shift_poison_to_restored_poison",
            output_dir / "feature_distance_bar.png",
            "Feature shift after restoration",
        )
        return df

    def _write_feature_pairwise_delta(
        self,
        features: dict[str, dict[str, torch.Tensor]],
        output_dir: Path,
    ) -> None:
        rows = []
        pairs = [
            ("RD_vs_D", "restored_degraded_poisoned", "degraded_poisoned"),
            ("RD_vs_R", "restored_degraded_poisoned", "restored_poisoned"),
        ]
        for comparison, left, right in pairs:
            if left not in features or right not in features:
                continue
            for layer in self.config.layers:
                rows.append(
                    {
                        "comparison": comparison,
                        "layer": layer,
                        "mean_l2_delta": float(
                            l2_distance(features[left][layer], features[right][layer])
                            .mean()
                            .item()
                        ),
                        "mean_cosine_similarity": float(
                            cosine_similarity(features[left][layer], features[right][layer])
                            .mean()
                            .item()
                        ),
                    }
                )
        pd.DataFrame(rows).to_csv(output_dir / "feature_pairwise_delta.csv", index=False)

    def _write_trigger_modules(
        self,
        images: dict[str, torch.Tensor],
        logit_df: pd.DataFrame,
        feature_df: pd.DataFrame,
    ) -> None:
        del logit_df, feature_df
        for module in self.config.analysis_modules:
            if module.name in {"input_space", "model_output", "feature_space"}:
                continue
            if module.name == "local_patch":
                self._write_local_patch(images)
            elif module.name == "frequency_global":
                self._write_frequency_global(images)
            elif module.name == "geometry_structure":
                self._write_geometry_structure(images)
            elif module.name == "feature_invisible":
                self._write_feature_invisible(images)
            elif module.name == "semantic_natural":
                self._write_semantic_natural()
            else:
                self._write_module_note(module.name, f"Unknown module '{module.name}' was skipped.")

    def _write_local_patch(self, images: dict[str, torch.Tensor]) -> None:
        if self.config.trigger_mask_path is None:
            self._write_module_note("local_patch", "Skipped: --trigger-mask-path is required.")
            return
        height, width = images["poisoned"].shape[-2:]
        mask = _load_mask(self.config.trigger_mask_path, (int(height), int(width)))
        restored_key = self._final_poisoned_key()
        diff = torch.abs(images["poisoned"] - images[restored_key])
        mask = mask.to(device=diff.device, dtype=diff.dtype)
        trigger = mask
        background = 1.0 - mask
        trigger_change = _masked_mean(diff, trigger)
        background_change = _masked_mean(diff, background)
        rows = [
            {
                "trigger_region_change": trigger_change,
                "background_change": background_change,
                "trigger_specificity": trigger_change / max(background_change, 1e-8),
            }
        ]
        out_dir = self.case_dir / "trigger_representation" / "local_patch"
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_dir / "local_patch_metrics.csv", index=False)
        save_tensor_grid(
            [images["poisoned"], images[restored_key], diff * mask],
            ["poisoned", restored_key, "masked residual"],
            out_dir / "patch_residual_heatmap.png",
        )

    def _write_frequency_global(self, images: dict[str, torch.Tensor]) -> None:
        out_dir = self.case_dir / "trigger_representation" / "frequency_global"
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for setting in ("clean", "poisoned", self._final_poisoned_key(), self._final_clean_key()):
            if setting not in images:
                continue
            energies = _frequency_band_energy(images[setting])
            rows.append({"setting": setting, **energies})
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / "frequency_band_energy.csv", index=False)
        save_bar_plot(df, "setting", "high_frequency_energy", out_dir / "frequency_band_energy_bar.png", "High-frequency energy")

    def _write_geometry_structure(self, images: dict[str, torch.Tensor]) -> None:
        out_dir = self.case_dir / "trigger_representation" / "geometry_structure"
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for name, key in {
            "poisoned_to_degraded_poisoned": "degraded_poisoned",
            "poisoned_to_restored_degraded_poisoned": "restored_degraded_poisoned",
        }.items():
            if key not in images:
                continue
            metrics = compute_residual_metrics(images["poisoned"], images[key])
            rows.append({"comparison": name, **metrics.__dict__})
        pd.DataFrame(rows).to_csv(out_dir / "geometry_residual_metrics.csv", index=False)
        if "restored_degraded_poisoned" not in images:
            return
        save_tensor_grid(
            [
                images["poisoned"],
                images["degraded_poisoned"],
                images["restored_degraded_poisoned"],
                torch.abs(images["poisoned"] - images["restored_degraded_poisoned"]),
            ],
            ["poisoned", "D(poisoned)", "R(D(poisoned))", "geometry residual"],
            out_dir / "geometry_residual_heatmap.png",
        )

    def _write_feature_invisible(self, images: dict[str, torch.Tensor]) -> None:
        out_dir = self.case_dir / "trigger_representation" / "feature_invisible"
        out_dir.mkdir(parents=True, exist_ok=True)
        restored_key = self._final_poisoned_key()
        residual = torch.abs(images["poisoned"] - images[restored_key])
        high_residual = _high_frequency_residual(residual)
        pd.DataFrame(
            [{"high_frequency_residual": float(high_residual.mean().item())}]
        ).to_csv(out_dir / "invisible_metrics.csv", index=False)
        save_tensor_grid(
            [images["poisoned"], images[restored_key], high_residual],
            ["poisoned", restored_key, "high-frequency residual"],
            out_dir / "high_frequency_residual_map.png",
        )

    def _write_semantic_natural(self) -> None:
        self._write_module_note(
            "semantic_natural",
            "Semantic natural trigger analysis requires detector outputs or annotations; no detector was configured.",
        )

    def _write_module_note(self, module_name: str, message: str) -> None:
        out_dir = self.case_dir / "trigger_representation" / module_name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "README.md").write_text(message + "\n", encoding="utf-8")

    def _write_metrics_summary(self, logit_df: pd.DataFrame) -> pd.DataFrame:
        summary = self._metrics_summary(logit_df)
        summary.to_csv(self.case_dir / "metrics_summary.csv", index=False)
        return summary

    def _write_pairwise_pipeline_comparison(self, metrics_summary: pd.DataFrame) -> pd.DataFrame:
        pairs = [
            ("D_vs_origin", "origin_poisoned", "degradation_poisoned"),
            ("R_vs_origin", "origin_poisoned", "restoration_poisoned"),
            ("RD_vs_origin", "origin_poisoned", "degradation_restoration_poisoned"),
            ("RD_vs_D", "degradation_poisoned", "degradation_restoration_poisoned"),
            ("RD_vs_R", "restoration_poisoned", "degradation_restoration_poisoned"),
        ]
        rows = []
        for comparison, left, right in pairs:
            left_row = _summary_row(metrics_summary, left)
            right_row = _summary_row(metrics_summary, right)
            if left_row is None or right_row is None:
                continue
            rows.append(
                {
                    "comparison": comparison,
                    "left_setting": left,
                    "right_setting": right,
                    "ca_delta": _delta(left_row, right_row, "ca"),
                    "asr_delta": _delta(left_row, right_row, "asr"),
                    "pa_delta": _delta(left_row, right_row, "pa"),
                    "target_logit_delta": _delta(
                        left_row,
                        right_row,
                        "mean_target_logit",
                    ),
                    "target_probability_delta": _delta(
                        left_row,
                        right_row,
                        "mean_target_probability",
                    ),
                    "true_class_logit_delta": _delta(
                        left_row,
                        right_row,
                        "mean_true_class_logit",
                    ),
                    "true_class_probability_delta": _delta(
                        left_row,
                        right_row,
                        "mean_true_class_probability",
                    ),
                }
            )
        pairwise = pd.DataFrame(rows)
        pairwise.to_csv(
            self.case_dir / "model_output" / "pairwise_pipeline_comparison.csv",
            index=False,
        )
        return pairwise

    def _write_behavior_decision(
        self,
        metrics_summary: pd.DataFrame,
        pairwise: pd.DataFrame,
    ) -> BehaviorDecision:
        decision = decide_behavior(
            metrics_summary=metrics_summary,
            pairwise_comparison=pairwise,
            defense_output_setting=self.defense_output.poisoned_setting,
            final_clean_setting=self.defense_output.clean_setting,
            thresholds=thresholds_from_config(self.config),
            target_class=self.config.target_class,
            expected_behavior=self.config.expected_behavior,
        )
        with (self.case_dir / "behavior_decision.json").open("w", encoding="utf-8") as handle:
            json.dump(decision.to_json_dict(), handle, indent=2, ensure_ascii=False)
        return decision

    def _write_visual_evidence(
        self,
        classifier: nn.Module,
        images: dict[str, torch.Tensor],
        labels: torch.Tensor,
        logit_df: pd.DataFrame,
    ) -> VisualEvidenceResult:
        if not self.config.visual_evidence:
            return VisualEvidenceResult(generated=[], skipped=["visual evidence disabled"])
        return write_visual_evidence(
            case_dir=self.case_dir,
            config=self.config,
            defense_output=self.defense_output,
            classifier=classifier,
            images=images,
            labels=labels,
            logit_df=logit_df,
        )

    def _write_summary(
        self,
        feature_df: pd.DataFrame,
        failure_summary: pd.DataFrame,
        metrics_summary: pd.DataFrame,
        pairwise: pd.DataFrame,
        decision: BehaviorDecision,
        visual_result: VisualEvidenceResult,
    ) -> None:
        target = self.config.target_class
        poison_rows = metrics_summary[metrics_summary["setting"].str.contains("poisoned")]
        feature_summary = feature_df.mean(numeric_only=True).to_dict()
        final_poisoned_row = _summary_row(
            metrics_summary,
            self.defense_output.poisoned_setting,
        )
        final_clean_row = _summary_row(metrics_summary, self.defense_output.clean_setting)
        origin_row = _summary_row(metrics_summary, "origin_poisoned")
        distribution_path = self.case_dir / "model_output" / "prediction_distribution.csv"
        distribution = (
            pd.read_csv(distribution_path)
            if distribution_path.exists()
            else pd.DataFrame()
        )
        final_distribution = (
            distribution[distribution["is_final_output"]]
            if "is_final_output" in distribution
            else pd.DataFrame()
        )
        lines = [
            f"# Mechanism Summary: {self.config.name}",
            "",
            "## Case",
            "",
            f"- Attack: {self.config.attack}",
            f"- Trigger metadata: {self.config.trigger_metadata or 'not specified'}",
            f"- Degradation: {self.config.degradation} ({self.config.strength})",
            f"- Restoration prior: {self.config.restoration_prior}",
            f"- Defense output setting: {self.defense_output.poisoned_setting}",
            f"- Expected behavior label: {self.config.expected_behavior or 'not specified'}",
            f"- Analysis modules: {', '.join(module.name for module in self.config.analysis_modules)}",
            f"- Target class: {target if target is not None else 'not specified'}",
            "",
            "## Metric-Level Observation",
            "",
            f"- Origin ASR: {_format_metric(origin_row, 'asr')}",
            f"- Origin PA: {_format_metric(origin_row, 'pa')}",
            f"- Final CA: {_format_metric(final_clean_row, 'ca')}",
            f"- Final ASR: {_format_metric(final_poisoned_row, 'asr')}",
            f"- Final PA: {_format_metric(final_poisoned_row, 'pa')}",
            f"- ASR delta: {_format_delta(origin_row, final_poisoned_row, 'asr')}",
            f"- PA delta: {_format_delta(origin_row, final_poisoned_row, 'pa')}",
            "",
            "## Prediction Distribution",
            "",
            _dataframe_to_markdown(final_distribution),
            "",
            "## Prediction Flip Evidence",
            "",
            "See `model_output/prediction_flip_summary.csv` for per-sample transitions.",
            "",
            "## Pairwise Pipeline Evidence",
            "",
            _dataframe_to_markdown(pairwise),
            "",
            "## Model-Output Evidence",
            "",
            _dataframe_to_markdown(poison_rows),
            "",
            "## Feature-Space Evidence",
            "",
            f"- Mean poison-to-restored feature shift: {feature_summary.get('feature_shift_poison_to_restored_poison', float('nan')):.6f}",
            f"- Mean restored-poison distance to clean centroid: {feature_summary.get('dist_restored_poison_to_clean_centroid', float('nan')):.6f}",
            f"- Mean restored-clean distance to clean centroid: {feature_summary.get('dist_restored_clean_to_clean_centroid', float('nan')):.6f}",
            "",
            "## Visual Evidence",
            "",
            *(
                [f"- `{item}`" for item in visual_result.generated]
                if visual_result.generated
                else ["- No visual evidence generated."]
            ),
            "",
            "Skipped visual evidence:",
            *(
                [f"- {item}" for item in visual_result.skipped]
                if visual_result.skipped
                else ["- None"]
            ),
            "",
            "## Failure-Case Evidence",
            "",
            _dataframe_to_markdown(failure_summary),
            "",
            "## Prediction Outcome Decision",
            "",
            f"Final outcome label: `{decision.label}`",
            "",
            "Rationale:",
            *[f"- {item}" for item in decision.rationale],
            "",
            "## Caveats",
            "",
            *[f"- {item}" for item in decision.caveats],
            "- DeBackdoor-style re-synthesis and restoration-prior integrity tests are future hooks, not part of this v3.0 core run.",
        ]
        (self.case_dir / "stage4_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _build_outcome_matrix(logit_df: pd.DataFrame) -> pd.DataFrame:
    metadata = (
        logit_df[
            [
                "sample_index",
                "original_index",
                "source_path",
                "true_label",
                "target_class",
            ]
        ]
        .drop_duplicates("sample_index")
        .sort_values("sample_index")
        .reset_index(drop=True)
    )
    value_columns = [
        "predicted_label",
        "target_probability",
        "true_class_probability",
        "target_margin",
    ]
    wide = logit_df.pivot(
        index="sample_index",
        columns="setting",
        values=value_columns,
    )
    wide.columns = [f"{setting}_{value}" for value, setting in wide.columns]
    wide = wide.reset_index()
    return metadata.merge(wide, on="sample_index", how="left")


def _target_dependent_failure_cases(
    matrix: pd.DataFrame,
    target_class: int | None,
    final_poisoned_setting: str,
) -> dict[str, pd.DataFrame]:
    names = (
        "attack_failed",
        "final_poisoned_defense_failed_asr",
        "final_poisoned_target_to_other_wrong",
        "final_poisoned_recovered",
        "restoration_poisoned_defense_failed_asr",
        "restoration_poisoned_overcorrected",
        "restoration_poisoned_recovered",
        "degradation_restoration_poisoned_defense_failed_asr",
        "degradation_restoration_poisoned_overcorrected",
        "degradation_restoration_poisoned_recovered",
    )
    if target_class is None:
        return {name: matrix.iloc[0:0] for name in names}

    true_label = matrix["true_label"]
    origin_attack_success = matrix["origin_poisoned_predicted_label"] == target_class
    final_pred = _prediction_column(matrix, final_poisoned_setting)
    restored_poisoned_pred = _prediction_column(matrix, "restoration_poisoned")
    restored_degraded_poisoned_pred = _prediction_column(
        matrix,
        "degradation_restoration_poisoned",
    )
    empty = matrix.iloc[0:0]
    return {
        "attack_failed": matrix[~origin_attack_success],
        "final_poisoned_defense_failed_asr": empty
        if final_pred is None
        else matrix[origin_attack_success & (final_pred == target_class)],
        "final_poisoned_target_to_other_wrong": empty
        if final_pred is None
        else matrix[
            origin_attack_success & (final_pred != target_class) & (final_pred != true_label)
        ],
        "final_poisoned_recovered": empty
        if final_pred is None
        else matrix[origin_attack_success & (final_pred == true_label)],
        "restoration_poisoned_defense_failed_asr": matrix[
            origin_attack_success & (restored_poisoned_pred == target_class)
        ]
        if restored_poisoned_pred is not None
        else empty,
        "restoration_poisoned_overcorrected": matrix[
            origin_attack_success
            & (restored_poisoned_pred != target_class)
            & (restored_poisoned_pred != true_label)
        ]
        if restored_poisoned_pred is not None
        else empty,
        "restoration_poisoned_recovered": matrix[
            origin_attack_success & (restored_poisoned_pred == true_label)
        ]
        if restored_poisoned_pred is not None
        else empty,
        "degradation_restoration_poisoned_defense_failed_asr": matrix[
            origin_attack_success & (restored_degraded_poisoned_pred == target_class)
        ]
        if restored_degraded_poisoned_pred is not None
        else empty,
        "degradation_restoration_poisoned_overcorrected": matrix[
            origin_attack_success
            & (restored_degraded_poisoned_pred != target_class)
            & (restored_degraded_poisoned_pred != true_label)
        ]
        if restored_degraded_poisoned_pred is not None
        else empty,
        "degradation_restoration_poisoned_recovered": matrix[
            origin_attack_success & (restored_degraded_poisoned_pred == true_label)
        ]
        if restored_degraded_poisoned_pred is not None
        else empty,
    }


def _clean_broken_case(matrix: pd.DataFrame, final_clean_setting: str) -> pd.DataFrame:
    final_pred = _prediction_column(matrix, final_clean_setting)
    if final_pred is None:
        return matrix.iloc[0:0]
    return matrix[
        (matrix["origin_clean_predicted_label"] == matrix["true_label"])
        & (final_pred != matrix["true_label"])
    ]


def _prediction_column(matrix: pd.DataFrame, setting: str) -> pd.Series | None:
    column = f"{setting}_predicted_label"
    if column not in matrix:
        return None
    return matrix[column]


def _batch_slices(total: int, batch_size: int) -> list[tuple[int, int]]:
    step = max(1, batch_size)
    return [(start, min(start + step, total)) for start in range(0, total, step)]


def _extract_features_batched(
    extractor: FeatureExtractor,
    tensor: torch.Tensor,
    layers: tuple[str, ...],
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    blocks: dict[str, list[torch.Tensor]] = {layer: [] for layer in layers}
    for start, end in _batch_slices(tensor.shape[0], batch_size):
        extracted = extractor.extract(tensor[start:end].to(device))
        for layer in layers:
            blocks[layer].append(extracted[layer].detach().cpu())
    return {layer: torch.cat(parts, dim=0) for layer, parts in blocks.items()}


def _safe_rate(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return float(count / total)


def _accuracy(group: pd.DataFrame) -> float | None:
    if group.empty:
        return None
    return float((group["predicted_label"] == group["true_label"]).mean())


def _mean_or_none(group: pd.DataFrame, column: str) -> float | None:
    if group.empty or column not in group:
        return None
    return float(group[column].mean())


def _summary_row(df: pd.DataFrame, setting: str) -> pd.Series | None:
    rows = df[df["setting"] == setting]
    if rows.empty:
        return None
    return rows.iloc[0]


def _delta(left: pd.Series, right: pd.Series, column: str) -> float | None:
    if column not in left or column not in right:
        return None
    if pd.isna(left[column]) or pd.isna(right[column]):
        return None
    return float(right[column] - left[column])


def _format_metric(row: pd.Series | None, column: str) -> str:
    if row is None or column not in row or pd.isna(row[column]):
        return "n/a"
    return f"{float(row[column]):.4f}"


def _format_delta(
    left: pd.Series | None,
    right: pd.Series | None,
    column: str,
) -> str:
    if left is None or right is None:
        return "n/a"
    value = _delta(left, right, column)
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _load_mask(path: Path, size: tuple[int, int]) -> torch.Tensor:
    image = Image.open(path).convert("L").resize((size[1], size[0]))
    tensor = torch.tensor(list(image.getdata()), dtype=torch.float32).reshape(1, 1, size[0], size[1])
    return tensor / 255.0


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    mask = mask.to(device=values.device, dtype=values.dtype)
    return float((values * mask).sum().item() / max((mask.sum() * values.shape[0] * values.shape[1]).item(), 1e-8))


def _frequency_band_energy(images: torch.Tensor) -> dict[str, float]:
    gray = images.mean(dim=1)
    spectrum = torch.fft.fftshift(torch.fft.fft2(gray), dim=(-2, -1)).abs()
    height, width = spectrum.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=images.device),
        torch.linspace(-1.0, 1.0, width, device=images.device),
        indexing="ij",
    )
    radius = torch.sqrt(xx**2 + yy**2)
    bands = {
        "low_frequency_energy": radius <= 0.15,
        "mid_frequency_energy": (radius > 0.15) & (radius <= 0.45),
        "high_frequency_energy": radius > 0.45,
    }
    return {name: float(spectrum[:, mask].mean().item()) for name, mask in bands.items()}


def _high_frequency_residual(residual: torch.Tensor) -> torch.Tensor:
    blurred = torch.nn.functional.avg_pool2d(residual, kernel_size=5, stride=1, padding=2)
    return torch.clamp(residual - blurred, min=0.0)


def _interpret_case(metric_summary: pd.DataFrame, feature_summary: dict[str, float]) -> str:
    if metric_summary.empty:
        return "Insufficient evidence: no model-output metrics were produced."
    clean = metric_summary[metric_summary["setting"] == "restoration_clean"]
    poisoned = metric_summary[metric_summary["setting"] == "restoration_poisoned"]
    if not clean.empty and float(clean["clean_accuracy_or_pa"].iloc[0]) < 0.5:
        return (
            "Restored clean accuracy/PA is low, so prioritize a destroy or distribution-shift "
            "explanation before claiming backdoor suppression."
        )
    if not poisoned.empty and "asr" in poisoned and pd.notna(poisoned["asr"].iloc[0]):
        asr = float(poisoned["asr"].iloc[0])
        if asr < 0.5:
            return (
                "Target prediction rate is reduced after restoration. Treat this as an "
                "attack-effect suppression signal and corroborate it with residual and feature evidence."
            )
    shift = feature_summary.get("feature_shift_poison_to_restored_poison", 0.0)
    if shift > 0.0:
        return (
            "Feature movement is measurable. Compare clean-centroid and target-centroid deltas "
            "to decide whether the movement supports suppress, preserve, recover, or destroy."
        )
    return "No strong mechanism conclusion from the minimal automatic summary."


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    columns = [str(column) for column in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in df.itertuples(index=False):
        values = [str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)
