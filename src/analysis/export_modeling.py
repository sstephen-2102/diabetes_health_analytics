"""Run from the project root: python -m src.analysis.export_modeling --csv PATH."""
import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
import sklearn

from src.common.config import (ANALYSIS_CONFIG, EXPECTED_COLUMNS, FEATURE_COLUMNS, MODEL_CONFIG,
                               PROJECT_CONFIG, TARGET_DEFINITION, VARIABLE_METADATA)
from src.common.exceptions import DataValidationError
from src.data.loader import load_dataset
from src.data.preprocessing import prepare_analysis_data
from src.data.validation import validate_dataset
from src.modeling.calibration import calculate_calibration_metrics
from src.modeling.classification import (compare_models, evaluate_classifier, generate_predictions, split_data,
                                         train_gradient_boosting, train_logistic_classifier, train_random_forest)
from src.modeling.imbalance import balance_training_data, inspect_class_distribution, train_class_weighted_model
from src.modeling.interpretation import calculate_feature_importance
from src.modeling.thresholds import evaluate_thresholds
from src.analysis.model_figures import (calibration_figure, importance_figure, precision_recall_figure,
                                        roc_figure, threshold_figure)
from src.analysis.visualization import prepare_precision_recall_data, prepare_roc_curve_data

TARGET = PROJECT_CONFIG["target_column"]
SEED = PROJECT_CONFIG["random_state"]
VARIANT = PROJECT_CONFIG["primary_ml_dataset_variant"]
THRESHOLD = MODEL_CONFIG["default_threshold"]
THRESHOLD_GRID = [round(t, 2) for t in np.arange(0.10, 0.91, 0.05)]
CALIBRATION_BINS = ANALYSIS_CONFIG["calibration_bins"]
BASELINES = {
    "logistic_regression": train_logistic_classifier,
    "random_forest": train_random_forest,
    "gradient_boosting": train_gradient_boosting
}

def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")

def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")

def _class_counts(y):
    counts = inspect_class_distribution(y)["counts"]
    return {"negatives": int(counts.get(0,0)), "positives": int(counts.get(1,0))}

def _default_profile(X_train):
    """Typical training-set value per feature, used for Prediction Explorer inputs the user leaves out."""
    return {column: int(X_train[column].mode().iloc[0]) if VARIABLE_METADATA[column]["allowed_values"]
            else float(X_train[column].median())
            for column in FEATURE_COLUMNS}

def _primary_split(csv_path: str, check_reference: bool) -> tuple[pd.DataFrame, dict]:
    """Load and validate the CSV, build the primary ML dataset and split it."""
    raw = load_dataset(csv_path)
    validation = validate_dataset(raw, EXPECTED_COLUMNS, TARGET)
    if not validation["is_valid"]:
        raise DataValidationError("; ".join(validation["errors"]))
    if check_reference and not validation["matches_reference"]:
        raise DataValidationError("Dataset does not match reference dataset.")
    data = prepare_analysis_data(raw, TARGET, FEATURE_COLUMNS, dataset_variant=VARIANT)
    return data, split_data(data, TARGET, test_size=PROJECT_CONFIG["test_size"], random_state=SEED)

def save_baseline_models(csv_path: str, models_directory: str = "outputs/models",
                         check_reference: bool = True) -> list[str]:
    """Retrain and save only the baseline pipelines, identical to those export_modeling saves.

    Used where the .pkl files are not in Git (for example a fresh clone or a cloud deployment).
    """
    _, split = _primary_split(csv_path, check_reference)
    folder = Path(models_directory)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, train in BASELINES.items():
        path = folder / f"{name}.pkl"
        joblib.dump(train(split["X_train"], split["y_train"], random_state=SEED), path, compress=3)
        paths.append(str(path))
    return paths

def export_modeling(csv_path: str, output_directory: str = "outputs", check_reference: bool = True,
                    importance_sample: int = 10_000) -> dict:
    """Export modeling artifacts for a CSV dataset."""
    data, split = _primary_split(csv_path, check_reference)
    output = Path(output_directory)

    tables_dir, models_dir, figures_dir = output / "tables", output / "models", output / "figures"
    for folder in (tables_dir, models_dir, figures_dir):
        folder.mkdir(parents=True, exist_ok=True)
    files = []
    def save_table(table, name):
        path = tables_dir / f"{name}.csv"
        table.to_csv(path, index=not isinstance(table.index, pd.RangeIndex))
        files.append(str(path.relative_to(output)))

    def save_figure(fig, name):
        fig.text(0.99, 0, f"dataset: {VARIANT}", ha="right", va="top", fontsize=8, color="dimgray")
        path = figures_dir / f"{VARIANT}_{name}.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        files.append(str(path.relative_to(output)))

    X_train, X_test = split["X_train"], split["X_test"]
    y_train, y_test = split["y_train"], split["y_test"]

    # Baselines on natural training distributions
    models, probabilities, metrics = {}, {}, {}
    for name, train in BASELINES.items():
        models[name] = train(X_train, y_train, random_state=SEED)
        predicted = generate_predictions(models[name], X_test, THRESHOLD)
        probabilities[name] = predicted["probabilities"]
        metrics[name] = evaluate_classifier(y_test, predicted["predictions"], predicted["probabilities"])
    save_table(compare_models(metrics), "model_comparison")

    # Imbalance strategies (training data only: test set keeps its natural distribution)
    undersampled = balance_training_data(X_train, y_train, method="undersample", random_state=SEED)
    strategies = {
        "original": (models["logistic_regression"], y_train),
        "class_weighted": (train_class_weighted_model("logistic_regression", X_train, y_train, random_state=SEED), y_train),
        "undersampling": (train_logistic_classifier(undersampled["X_train"], undersampled["y_train"], random_state=SEED), undersampled["y_train"]),
    }
    before = _class_counts(y_train)
    rows = []
    for strategy, (model, y_fitted) in strategies.items():
        predicted = generate_predictions(model, X_test, THRESHOLD)
        result = evaluate_classifier(y_test, predicted["predictions"], predicted["probabilities"])
        after = _class_counts(y_fitted)
        rows.append({
            "strategy": strategy,
            "model": "logistic_regression", "dataset_variant": VARIANT,
            "seed": SEED,
            "threshold": THRESHOLD,
            "train_negatives_before": before["negatives"],
            "train_positives_before": before["positives"],
            "train_negatives_after": after["negatives"],
            "train_positives_after": after["positives"],
            **{k: v for k, v in result.items() if k != "confusion_matrix"},
            **result["confusion_matrix"]
        })
    save_table(pd.DataFrame(rows), "imbalance_experiments")

    # Thresholds, calibration and importance of each baseline
    sample = X_test.sample(min(importance_sample, len(X_test)), random_state=SEED)
    calibration = {}
    threshold_tables, calibration_tables, importance_tables, roc_curves, pr_curves = {}, {}, {}, {}, {}
    for name, model in models.items():
        threshold_tables[name] = evaluate_thresholds(y_test, probabilities[name], THRESHOLD_GRID)
        save_table(threshold_tables[name], f"thresholds_analysis_{name}")
        summary = calculate_calibration_metrics(y_test, probabilities[name], CALIBRATION_BINS)
        calibration_tables[name] = summary.pop("calibration_table")
        save_table(calibration_tables[name], f"calibration_table_{name}")
        calibration[name] = summary
        importance_tables[name] = calculate_feature_importance(model, sample, y_test.loc[sample.index], n_repeats=10, random_state=SEED)
        save_table(importance_tables[name], f"feature_importance_{name}")
        roc_curves[name] = prepare_roc_curve_data(y_test, probabilities[name])
        save_table(roc_curves[name], f"roc_curve_{name}")
        pr_curves[name] = prepare_precision_recall_data(y_test, probabilities[name])
        save_table(pr_curves[name], f"precision_recall_curve_{name}")

    # Figures
    save_figure(roc_figure(roc_curves, {n: m["roc_auc"] for n, m in metrics.items()}), "roc_curves")
    save_figure(precision_recall_figure(pr_curves, {n: m["pr_auc"] for n, m in metrics.items()}, float(y_test.mean())),
                "precision_recall_curves")
    save_figure(calibration_figure(calibration_tables), "calibration_curves")
    save_figure(threshold_figure(threshold_tables, THRESHOLD), "threshold_tradeoffs")
    save_figure(importance_figure(importance_tables), "feature_importance")

    # Model artifacts and metrics
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    dataset_version = hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()
    model_records = {}
    for name, model in models.items():
        path = models_dir / f"{name}.pkl"
        joblib.dump(model, path, compress=3)
        files.append(str(path.relative_to(output)))
        model_records[name] = {
            "experiment_id": f"{name}_{VARIANT}_{SEED}",
            "model_type": type(model.named_steps["classifier"]).__name__,
            "parameters": model.named_steps["classifier"].get_params(),
            "training_strategy": "original training distribution",
            "artifact": str(path.relative_to(output))
        }
    metadata = {
        "timestamp": timestamp,
        "source_file": Path(csv_path).name,
        "dataset_version_sha256": dataset_version,
        "dataset_variant": VARIANT,
        "target_definition": TARGET_DEFINITION,
        "features": FEATURE_COLUMNS,
        "random_state": SEED,
        "split": split["metadata"],
        "default_profile": _default_profile(X_train),
        "preprocessing": "complete-row deduplication; StandardScaler inside each Pipeline",
        "threshold": THRESHOLD,
        "threshold_grid": THRESHOLD_GRID,
        "threshold_selection": "none; full grid reported on the test set as a trade-off (no validation set)",
        "calibration_bins": CALIBRATION_BINS,
        "importance": {
            "method": "permutation", 
            "scoring": "roc_auc",
            "n_repeats": 10,
            "sample_rows": len(sample),
            "sample_source": "test set"
        },
        "models": model_records,
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__
        }
    }
    _write_json(models_dir / "metadata.json", metadata)
    _write_json(models_dir / "metrics.json", {"test_metrics": metrics, "calibration": calibration, "imbalance_experiments": rows})
    files += ["models/metadata.json", "models/metrics.json"]

    # Manifest
    manifest = {
        "timestamp": timestamp,
        "source_file": Path(csv_path).name,
        "dataset_version_sha256": dataset_version,
        "dataset_variant": VARIANT,
        "rows": len(data),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "leakage_check": split["metadata"]["leakage_check"]["status"],
        "files": files
    }
    _write_json(output / "modeling_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Path to the CSV file")
    parser.add_argument("--output", default="outputs", help="Output directory")
    args = parser.parse_args()
    manifest = export_modeling(args.csv, args.output)
    print(f"Exported {len(manifest['files'])} modeling artifacts to {args.output} (leakage check: {manifest['leakage_check']})")


