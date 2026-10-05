"""Run from the project root: python -m src.analysis.export_eda --csv PATH."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import pandas as pd
from src.common.config import VARIABLE_METADATA, EXPECTED_COLUMNS, TARGET_DEFINITION
from src.common.exceptions import DataValidationError
from src.data.loader import load_dataset
from src.data.validation import validate_dataset
from src.analysis.eda import compare_dataset_variants


def export_eda(csv_path: str, output_directory: str = "outputs", check_reference: bool = True) -> dict:
    """Export variant-labeled tables, quality JSON, figures and a run manifest.

    Full-source runs check project reference dimensions/counts before any export.
    check_reference=False is explicit opt-out for fixtures/new dataset versions.
    No raw data or models are exported. Existing generated outputs are regenerated.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = load_dataset(csv_path)
    validation = validate_dataset(data, EXPECTED_COLUMNS, "Diabetes_binary")
    if not validation["is_valid"]:
        raise DataValidationError("; ".join(validation["errors"]))
    if check_reference and not validation["matches_reference"]:
        raise DataValidationError("Full source differs from reference counts; review data before export.")
    result = compare_dataset_variants(data)
    output = Path(output_directory)
    (output / "tables").mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    files = []

    def save_table(table, name, variant=None):
        path = output / "tables" / f"{name}.csv"
        table = table.copy()
        if not isinstance(table.index, pd.RangeIndex):
            table = table.rename_axis(table.index.name or "variable").reset_index()
        if variant:
            table["dataset_variant"] = variant
        table.to_csv(path, index=False)
        files.append(str(path.relative_to(output)))

    def save_figure(fig, name, variant):
        fig.suptitle(variant, fontsize=10, color="dimgray")
        fig.tight_layout(rect=[0, 0, 1, .95])
        path = output / "figures" / f"{variant}_{name}.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        files.append(str(path.relative_to(output)))

    for variant, bundle in result["variants"].items():
        frame, tables = bundle["data"], bundle["tables"]
        for name, table in tables.items():
            save_table(table, f"{variant}_{name}", variant)
        path = output / "tables" / f"{variant}_data_quality.json"
        path.write_text(json.dumps(bundle["quality"], indent=2, allow_nan=False), encoding="utf-8")
        files.append(str(path.relative_to(output)))
        path = output / "tables" / f"{variant}_validation.json"
        path.write_text(json.dumps(bundle["validation"], indent=2), encoding="utf-8")
        files.append(str(path.relative_to(output)))

        fig, ax = plt.subplots(figsize=(8, 5))
        target = tables["target_distribution"]
        bars = ax.bar(["Category 0", "Category 1"], target["count"], color="steelblue")
        ax.bar_label(bars, labels=[f"{r.count:,} ({r.percentage:.1f}%)" for r in target.itertuples()], padding=4)
        ax.set(title="Target distribution", ylabel="Respondents")
        ax.margins(y=.2)
        save_figure(fig, "target_distribution", variant)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(frame.BMI, bins=30, color="steelblue", edgecolor="white")
        ax.axvline(frame.BMI.mean(), color="darkorange", linestyle="--", label=f"Mean {frame.BMI.mean():.2f}")
        ax.axvline(frame.BMI.median(), color="darkred", linestyle=":", label=f"Median {frame.BMI.median():.0f}")
        ax.set(title="BMI distribution — all values retained", xlabel="BMI", ylabel="Respondents")
        ax.legend()
        save_figure(fig, "bmi_distribution", variant)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot([frame.loc[frame.Diabetes_binary == c, "BMI"] for c in [0, 1]], flierprops={"marker": ".", "markersize": 2, "alpha": .15})
        ax.set_xticks([1, 2], ["Category 0", "Category 1"])
        ax.set(title="BMI by target — all values retained", ylabel="BMI")
        save_figure(fig, "bmi_by_target", variant)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, column in zip(axes, ["MentHlth", "PhysHlth"]):
            counts = frame[column].value_counts().sort_index()
            ax.bar(counts.index, counts.values, color="steelblue")
            ax.set(title=VARIABLE_METADATA[column]["label"], xlabel="Reported days", ylabel="Respondents")
        save_figure(fig, "health_day_distributions", variant)

        for feature in ["Age", "Sex", "Education", "Income", "GenHlth", "HighBP", "HighChol", "PhysActivity"]:
            table = tables[f"prevalence_{feature}"]
            fig, ax = plt.subplots(figsize=(9, 5))
            bars = ax.barh(table.label, table.target_percentage, color="steelblue")
            ax.invert_yaxis()
            ax.bar_label(bars, fmt="%.1f%%", padding=3)
            ax.set(title=f"Target category 1 by {VARIABLE_METADATA[feature]['label']}", xlabel="Category 1 percentage within group")
            ax.margins(x=.2)
            save_figure(fig, f"prevalence_{feature}", variant)

        corr = tables["spearman_correlation"]
        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr)), corr.index)
        ax.set_title("Spearman correlation — exploratory, unadjusted")
        for i in range(len(corr)):
            for j in range(len(corr)):
                ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, label="Spearman coefficient")
        save_figure(fig, "spearman_correlation", variant)

    save_table(result["sensitivity"], "full_vs_unique_sensitivity")
    full = result["variants"]["full_clean_v1"]
    unique = result["variants"]["unique_profile_v1"]
    q = full["quality"]
    full_pct = full["data"].Diabetes_binary.mean() * 100
    unique_pct = unique["data"].Diabetes_binary.mean() * 100
    summary = f"""# EDA results and handoff

Source: UCI dataset 891. SHA-256 is recorded in eda_manifest.json.
Target definition: {TARGET_DEFINITION}. This follows the project contract and
imported UCI dictionary; the original binary preprocessing was not independently audited.

## Verification
- full_clean_v1: {q['rows']:,} rows, {q['columns']} columns, {q['missing_cells']} missing cells.
- Repeated extra rows: {q['duplicate_rows']:,} ({q['duplicate_rate_percent']:.4f}%).
- unique_profile_v1: {len(unique['data']):,} complete-row distinct profiles.
- BMI IQR flags: {q['potential_outliers']['BMI']['flagged_count']:,}; all retained.

## Duplicate sensitivity
Category 1 represents {full_pct:.4f}% of full_clean_v1 and {unique_pct:.4f}% of
unique_profile_v1, a change of {unique_pct-full_pct:+.4f} percentage points.
The full_vs_unique_sensitivity.csv file compares all categorical conditional
percentages and target-group means. Deduplication changes the represented sample;
it does not establish which respondents were duplicated.

## Main patterns
The full sample is imbalanced. Category 1 has higher typical BMI and more reported
poor physical-health days. Its share generally increases through age group 11,
then declines. Higher education/income categories generally have lower shares.
Reported high BP, high cholesterol, stroke, heart disease/attack and walking
difficulty have higher category-1 shares; activity has a lower share.
General-health category-1 shares increase from Excellent to Poor.
Consult labeled CSVs for exact estimates; these are unadjusted comparisons.

## Correlation interpretation
The matrix uses Spearman correlation for numeric counts, BMI, ordered categories
and the binary target. Age/Education/Income are ordered codes, not exact quantities.
The matrix is descriptive, not an importance ranking or adjusted causal model.

## Decisions and limitations
Raw data are preserved. Main EDA uses full_clean_v1; unique_profile_v1 is a labeled
sensitivity sample and the project-specified primary ML input. No balancing,
outlier removal, imputation or model fitting occurs here. Survey weights/design
are not applied; results describe the sample, not national prevalence. Reporting
error, unequal group sizes and confounding remain limitations. BMI extremes are
unverified. Complete-row deduplication does not guarantee distinct feature vectors
across train/test: the modeling team must check that separately.

## Reproduce
From the repository root, install requirements then run:
`python -m src.analysis.export_eda --csv data/raw/cdc_diabetes.csv`
See docs/MEMBER1_HANDOFF.md for contracts, changes, test scope and next steps.
"""
    (output / "EDA_RESULTS.md").write_text(summary, encoding="utf-8")
    files.append("EDA_RESULTS.md")
    manifest = {"source_file": Path(csv_path).name,
                "source_sha256": hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
                "target_definition": TARGET_DEFINITION, "dataset_variants": list(result["variants"]),
                "python_version": platform.python_version(), "pandas_version": pd.__version__,
                "matplotlib_version": matplotlib.__version__, "spearman_method": "average ranks",
                "reference_checks": validation["reference_checks"], "files": files}
    (output / "eda_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()
    manifest = export_eda(args.csv, args.output)
    print(f"Exported {len(manifest['files'])} artifacts for both dataset variants.")
