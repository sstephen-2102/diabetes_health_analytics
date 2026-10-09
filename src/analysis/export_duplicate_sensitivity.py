"""Run from the project root: python -m src.analysis.export_duplicate_sensitivity --csv PATH.

Duplicate-handling sensitivity (RQ13, H11): the three baselines are trained on each
combination of dataset variant (full_clean_v1, unique_profile_v1) and split type
(grouped by feature vector, plain random). The grouped unique_profile_v1 scenario
is the primary experiment; the random splits deliberately allow identical feature
vectors on both sides to show how much that changes evaluation.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.analysis.export_modeling import BASELINES
from src.common.config import EXPECTED_COLUMNS, FEATURE_COLUMNS, MODEL_CONFIG, PROJECT_CONFIG, TARGET_DEFINITION
from src.common.exceptions import DataValidationError
from src.data.loader import load_dataset
from src.data.preprocessing import prepare_analysis_data
from src.data.validation import validate_dataset
from src.modeling.classification import evaluate_classifier, generate_predictions, split_data

TARGET = PROJECT_CONFIG["target_column"]
SEED = PROJECT_CONFIG["random_state"]
TEST_SIZE = PROJECT_CONFIG["test_size"]
THRESHOLD = MODEL_CONFIG["default_threshold"]
SCENARIOS = {
    "unique_grouped": ("unique_profile_v1", "grouped"),
    "unique_random": ("unique_profile_v1", "random"),
    "full_grouped": ("full_clean_v1", "grouped"),
    "full_random": ("full_clean_v1", "random"),
}
MODEL_PARAMETERS = {"random_forest": {"n_jobs": -1}}


def _split(data, split_type):
    if split_type == "grouped":
        split = split_data(data, TARGET, test_size=TEST_SIZE, random_state=SEED)
        return split["X_train"], split["X_test"], split["y_train"], split["y_test"]
    return train_test_split(data[FEATURE_COLUMNS], data[TARGET], test_size=TEST_SIZE,
                            stratify=data[TARGET], random_state=SEED)


def _seen_in_training(X_train, X_test):
    """True for test rows whose exact feature vector also appears in the training set."""
    train_hashes = set(pd.util.hash_pandas_object(X_train, index=False))
    return pd.util.hash_pandas_object(X_test, index=False).isin(train_hashes).to_numpy()


def _metric_row(scenario, model, subset, y_true, probabilities):
    predictions = (probabilities >= THRESHOLD).astype(int)
    result = evaluate_classifier(y_true, predictions, probabilities)
    return {"scenario": scenario, "model": model, "test_subset": subset,
            "test_prevalence": float(y_true.mean()),
            **{k: v for k, v in result.items() if k != "confusion_matrix"},
            **result["confusion_matrix"]}


def export_duplicate_sensitivity(csv_path: str, output_directory: str = "outputs",
                                 check_reference: bool = True) -> dict:
    """Train every baseline on each scenario and save metrics on the full test set and on
    test rows seen or unseen in training."""
    raw = load_dataset(csv_path)
    validation = validate_dataset(raw, EXPECTED_COLUMNS, TARGET)
    if not validation["is_valid"]:
        raise DataValidationError("; ".join(validation["errors"]))
    if check_reference and not validation["matches_reference"]:
        raise DataValidationError("Dataset does not match reference dataset.")

    variants = {v: prepare_analysis_data(raw, TARGET, FEATURE_COLUMNS, dataset_variant=v)
                for v in ("full_clean_v1", "unique_profile_v1")}
    rows, scenarios = [], {}
    for scenario, (variant, split_type) in SCENARIOS.items():
        X_train, X_test, y_train, y_test = _split(variants[variant], split_type)
        seen = _seen_in_training(X_train, X_test)
        scenarios[scenario] = {
            "dataset_variant": variant, "split": split_type,
            "train_rows": len(X_train), "test_rows": len(X_test),
            "seen_test_rows": int(seen.sum()), "seen_test_share": float(seen.mean()),
        }
        subsets = {"all": slice(None), "seen_in_training": seen, "unseen_in_training": ~seen}
        for model, train in BASELINES.items():
            fitted = train(X_train, y_train, random_state=SEED, **MODEL_PARAMETERS.get(model, {}))
            probabilities = generate_predictions(fitted, X_test, THRESHOLD)["probabilities"]
            for subset, mask in subsets.items():
                y_subset = y_test[mask]
                if y_subset.nunique() == 2:
                    rows.append(_metric_row(scenario, model, subset, y_subset, probabilities[mask]))

    table = pd.DataFrame(rows)
    details = pd.DataFrame.from_dict(scenarios, orient="index")
    table = details.join(table.set_index("scenario"), how="right").rename_axis("scenario").reset_index()

    output = Path(output_directory)
    (output / "tables").mkdir(parents=True, exist_ok=True)
    table_path = output / "tables" / "duplicate_sensitivity.csv"
    table.to_csv(table_path, index=False)

    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_file": Path(csv_path).name,
        "dataset_version_sha256": hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
        "target_definition": TARGET_DEFINITION,
        "primary_scenario": "unique_grouped",
        "random_state": SEED,
        "test_size": TEST_SIZE,
        "threshold": THRESHOLD,
        "seen_definition": "test row whose exact feature vector also appears in the training set",
        "scenarios": scenarios,
        "files": [str(table_path.relative_to(output))],
    }
    (output / "duplicate_sensitivity_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Path to the CSV file")
    parser.add_argument("--output", default="outputs", help="Output directory")
    args = parser.parse_args()
    manifest = export_duplicate_sensitivity(args.csv, args.output)
    shares = ", ".join(f"{name} {s['seen_test_share']:.1%}" for name, s in manifest["scenarios"].items())
    print(f"Exported duplicate sensitivity to {args.output} (test rows seen in training: {shares})")
