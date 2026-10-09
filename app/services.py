import pandas as pd

EXPORT_COMMAND = "python -m src.analysis.export_{module} --csv data/raw/cdc_diabetes.csv"

def get_dashboard_summary(data: pd.DataFrame, target_column: str) -> dict:
    """Generate dashboard summary from analytics functions."""
    from src.data._checks import require_binary_target
    from src.data.quality import generate_data_quality_report
    from src.statistics.descriptive import summarize_categorical_variable
    require_binary_target(data, target_column)
    return {"status": "success", "quality": generate_data_quality_report(data),
            "target_distribution": summarize_categorical_variable(data, target_column),
            "dataset_variant": data.attrs.get("dataset_variant", "unspecified")}

def get_eda_summary(data: pd.DataFrame, feature: str) -> dict:
    """Generate EDA results."""
    from src.common.config import PROJECT_CONFIG, VARIABLE_METADATA, TARGET_DEFINITION
    from src.statistics.descriptive import (
        summarize_numeric_variable, summarize_categorical_variable,
        compare_by_target, calculate_prevalence_table,
    )
    from src.data._checks import require_frame
    target = PROJECT_CONFIG["target_column"]
    require_frame(data, [feature, target])
    metadata = VARIABLE_METADATA.get(feature, {})
    categorical = metadata.get("type") in {"binary", "ordinal"} or not pd.api.types.is_numeric_dtype(data[feature])
    summary = summarize_categorical_variable(data, feature) if categorical else summarize_numeric_variable(data, feature)
    result = {"status": "success", "summary": summary, "feature_metadata": metadata,
              "dataset_variant": data.attrs.get("dataset_variant", "unspecified"),
              "target_definition": TARGET_DEFINITION,
              "warnings": ["Descriptive, unadjusted sample associations; no causal conclusions."]}
    if feature != target:
        result["by_target"] = compare_by_target(data, feature, target)
        if categorical:
            result["prevalence"] = calculate_prevalence_table(data, feature, target)
    return result

def get_probability_analysis(data: pd.DataFrame, target_column: str, target_value, conditions: dict) -> dict:
    """Coordinate conditional probability analysis."""
    pass

def get_statistical_test_result(data: pd.DataFrame, test_type: str, feature: str, target_column: str, parameters: dict) -> dict:
    """Dispatch supported statistical tests."""
    pass

def ensure_model_files(models_directory: str, csv_path: str) -> bool:
    """Retrain the saved pipelines if metadata.json exists but its .pkl files do not (they are not in Git).

    Downloads the verified UCI dataset to csv_path first when it is missing. Returns True if it retrained.
    """
    import json
    from pathlib import Path
    from src.common.exceptions import DataLoadError
    folder = Path(models_directory)
    try:
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        artifacts = [folder / Path(record["artifact"]).name for record in metadata["models"].values()]
    except (OSError, KeyError, ValueError):
        return False
    if all(path.is_file() for path in artifacts):
        return False
    from src.analysis.export_modeling import save_baseline_models
    from src.data.loader import download_uci_dataset
    try:
        save_baseline_models(download_uci_dataset(csv_path), str(folder))
    except Exception as exc:
        raise DataLoadError(f"Saved models are missing and retraining failed: {exc}. "
                            f"Run {EXPORT_COMMAND.format(module='modeling')} first.") from exc
    return True

def load_model_artifacts(models_directory: str = "outputs/models") -> dict:
    """Load exported models, metadata and metrics. Only for trusted, project-generated files (pickle)."""
    import json
    from pathlib import Path
    import joblib
    from src.common.exceptions import DataLoadError
    folder = Path(models_directory)
    try:
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        models = {name: joblib.load(folder / Path(record["artifact"]).name)
                  for name, record in metadata["models"].items()}
    except (OSError, KeyError, ValueError) as exc:
        raise DataLoadError(f"Cannot load model artifacts from {folder}: {exc}. "
                            f"Run {EXPORT_COMMAND.format(module='modeling')} first.") from exc
    return {"models": models, "metadata": metadata, "metrics": metrics}

def _read_outputs(output_directory: str, manifest_name: str, tables: dict) -> tuple[dict, dict]:
    """Read a manifest and {key: tables/<file>.csv}; raise DataLoadError with the command that creates them."""
    import json
    from pathlib import Path
    from src.common.exceptions import DataLoadError
    folder = Path(output_directory)
    try:
        manifest = json.loads((folder / manifest_name).read_text(encoding="utf-8"))
        frames = {key: pd.read_csv(folder / "tables" / f"{name}.csv") for key, name in tables.items()}
    except (OSError, ValueError) as exc:
        module = manifest_name.replace("_manifest.json", "")
        raise DataLoadError(f"Cannot load {module} outputs from {folder}: {exc}. "
                            f"Run {EXPORT_COMMAND.format(module=module)} first.") from exc
    return manifest, frames

def load_modeling_outputs(output_directory: str = "outputs") -> dict:
    """Load exported test-set tables for every saved model (comparison, imbalance, thresholds, curves...)."""
    import json
    from pathlib import Path
    from src.common.config import TARGET_DEFINITION
    from src.common.exceptions import DataLoadError
    models_folder = Path(output_directory) / "models"
    try:
        metadata = json.loads((models_folder / "metadata.json").read_text(encoding="utf-8"))
        calibration = json.loads((models_folder / "metrics.json").read_text(encoding="utf-8"))["calibration"]
        names = list(metadata["models"])
    except (OSError, KeyError, ValueError) as exc:
        raise DataLoadError(f"Cannot load model metadata from {output_directory}: {exc}. "
                            f"Run {EXPORT_COMMAND.format(module='modeling')} first.") from exc
    per_model = {"thresholds": "thresholds_analysis", "calibration": "calibration_table",
                 "importance": "feature_importance", "roc": "roc_curve", "precision_recall": "precision_recall_curve"}
    tables = {"comparison": "model_comparison", "imbalance": "imbalance_experiments"}
    tables |= {f"{kind}/{name}": f"{prefix}_{name}" for kind, prefix in per_model.items() for name in names}
    manifest, frames = _read_outputs(output_directory, "modeling_manifest.json", tables)
    results = {"comparison": frames["comparison"].set_index("model"), "imbalance": frames["imbalance"],
               "calibration_summary": pd.DataFrame(calibration).T.rename_axis("model")}
    results |= {kind: {name: frames[f"{kind}/{name}"] for name in names} for kind in per_model}
    prevalence = (results["comparison"]["false_negatives"] + results["comparison"]["true_positives"]) \
        / results["comparison"]["n"]
    return {"status": "success", "results": results,
            "warnings": ["Metrics are from the held-out test set; no model is ranked best on every metric."],
            "dataset_variant": manifest["dataset_variant"], "target_definition": TARGET_DEFINITION,
            "metadata": {**{k: manifest[k] for k in ("timestamp", "dataset_version_sha256", "rows",
                                                     "train_rows", "test_rows", "leakage_check")},
                         "models": names, "test_prevalence": float(prevalence.iloc[0]),
                         "threshold": metadata["threshold"], "threshold_grid": metadata["threshold_grid"],
                         "threshold_selection": metadata["threshold_selection"],
                         "importance": metadata["importance"], "random_state": metadata["random_state"]}}

def get_threshold_comparison(threshold_tables: dict, threshold: float) -> pd.DataFrame:
    """One row per model with the test-set metrics at a threshold from the exported grid."""
    import math
    from src.common.exceptions import InvalidParameterError
    rows = []
    for name, table in threshold_tables.items():
        match = table[[math.isclose(t, threshold, abs_tol=1e-9) for t in table["threshold"]]]
        if match.empty:
            raise InvalidParameterError(f"Threshold {threshold} is not in the exported grid "
                                        f"{table['threshold'].tolist()}.")
        rows.append(match.iloc[[0]].assign(model=name))
    return pd.concat(rows).set_index("model")

def load_regression_outputs(output_directory: str = "outputs") -> dict:
    """Load exported logistic-regression tables and the run manifest."""
    names = ["baseline_coefficients", "interaction_coefficients", "baseline_odds_ratios",
             "model_comparison", "bmi_odds_by_age"]
    manifest, frames = _read_outputs(output_directory, "regression_manifest.json",
                                     {name: f"full_clean_v1_regression_{name}" for name in names})
    return {"status": "success", "results": {**frames, "interaction_term": manifest["interaction_term"],
                                             "model_comparison_test": manifest["model_comparison"]},
            "warnings": manifest["limitations"],
            "dataset_variant": manifest["dataset_variant"], "target_definition": manifest["target_definition"],
            "metadata": {k: manifest[k] for k in ("timestamp", "dataset_version_sha256", "n", "feature_treatment",
                                                  "centering_means", "confidence_level", "converged")}}

def load_duplicate_sensitivity(output_directory: str = "outputs") -> dict:
    """Load the duplicate-handling sensitivity experiment (two dataset variants x two split types)."""
    manifest, frames = _read_outputs(output_directory, "duplicate_sensitivity_manifest.json",
                                     {"table": "duplicate_sensitivity"})
    table = frames["table"]
    scenarios = pd.DataFrame.from_dict(manifest["scenarios"], orient="index").rename_axis("scenario")
    return {"status": "success",
            "results": {"all_test_rows": table[table["test_subset"] == "all"].reset_index(drop=True),
                        "subsets": table, "scenarios": scenarios},
            "warnings": [f"Sensitivity analysis: the primary results use only the {manifest['primary_scenario']} "
                         "scenario (grouped split on unique_profile_v1).",
                         "Each scenario uses one seed and one split; AUC differences of about 0.005 or less "
                         "are within sampling noise.",
                         "full_clean_v1 test sets contain repeated, mostly low-risk profiles and a lower "
                         "prevalence, so their metrics describe a different test population."],
            "dataset_variant": "full_clean_v1 and unique_profile_v1",
            "target_definition": manifest["target_definition"],
            "metadata": {k: manifest[k] for k in ("timestamp", "dataset_version_sha256", "primary_scenario",
                                                  "random_state", "test_size", "threshold", "seen_definition")}}

def get_model_comparison(model_results: dict) -> pd.DataFrame:
    """Prepare model metrics for UI presentation; do not rank universally."""
    from src.modeling.classification import compare_models
    table = compare_models(model_results)
    first = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc", "brier_score"]
    return table[first + [c for c in table.columns if c not in first]]

def get_prediction_result(model, user_inputs: dict, threshold: float, model_metadata: dict | None = None) -> dict:
    """Coordinate prediction-profile validation and prediction."""
    from src.common.config import TARGET_DEFINITION, VARIABLE_METADATA
    from src.analysis.profiles import build_risk_profile, predict_risk_profile
    model_metadata = model_metadata or {}
    profile = build_risk_profile(user_inputs, VARIABLE_METADATA, model_metadata.get("default_profile"))
    prediction = predict_risk_profile(model, profile, threshold)
    warnings = [prediction["disclaimer"]]
    if profile["defaulted_features"]:
        warnings.append("Inputs not provided were set to typical training-set values: "
                        + ", ".join(profile["defaulted_features"]) + ".")
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    return {"status": "success",
            "results": {**prediction, "profile": profile["display"],
                        "defaulted_features": profile["defaulted_features"]},
            "warnings": warnings,
            "dataset_variant": model_metadata.get("dataset_variant", "unspecified"),
            "target_definition": TARGET_DEFINITION,
            "metadata": {"model_type": type(estimator).__name__,
                         "trained_at": model_metadata.get("timestamp"),
                         "dataset_version_sha256": model_metadata.get("dataset_version_sha256")}}
