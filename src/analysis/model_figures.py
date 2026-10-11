"""Static report figures for modeling results. Inputs are prepared tables; outputs are matplotlib Figures.

Uses matplotlib.figure.Figure directly (no pyplot), so no global state, backend or plt.close() is needed.
"""
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import NullFormatter

COLORS = {"logistic_regression": "tab:blue", "random_forest": "tab:green", "gradient_boosting": "tab:orange"}
MIN_CALIBRATION_COUNT = 50


def _label(name: str) -> str:
    return name.replace("_", " ").title()


def _new_figure(width: float, height: float) -> Figure:
    return Figure(figsize=(width, height), layout="constrained")


def roc_figure(curves: dict, roc_auc: dict) -> Figure:
    fig = _new_figure(6, 5)
    ax = fig.subplots()
    for name, curve in curves.items():
        ax.plot(curve["false_positive_rate"], curve["true_positive_rate"], color=COLORS.get(name),
                label=f"{_label(name)} (AUC {roc_auc[name]:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="No skill")
    ax.set(xlabel="False-positive rate (1 - specificity)", ylabel="True-positive rate (recall)",
           title="ROC curves - test set", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="lower right")
    return fig


def precision_recall_figure(curves: dict, pr_auc: dict, prevalence: float) -> Figure:
    fig = _new_figure(6, 5)
    ax = fig.subplots()
    for name, curve in curves.items():
        ax.plot(curve["recall"], curve["precision"], color=COLORS.get(name),
                label=f"{_label(name)} (AP {pr_auc[name]:.3f})")
    ax.axhline(prevalence, ls="--", color="gray", label=f"No skill ({prevalence:.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves - test set", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper right")
    return fig


def calibration_figure(tables: dict, min_count: int = MIN_CALIBRATION_COUNT) -> Figure:
    fig = _new_figure(6, 5)
    ax = fig.subplots()
    for name, table in tables.items():
        shown = table[table["count"] >= min_count]
        ax.plot(shown["mean_predicted"], shown["observed_frequency"], marker="o", color=COLORS.get(name),
                label=_label(name))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed frequency", xlim=(0, 1), ylim=(0, 1),
           title=f"Calibration - test set (bins with at least {min_count} rows)")
    ax.legend(loc="upper left")
    return fig


def threshold_figure(tables: dict, default_threshold: float) -> Figure:
    fig = _new_figure(5 * len(tables), 4.8)
    axes = np.atleast_1d(fig.subplots(1, len(tables), sharey=True))
    for ax, (name, table) in zip(axes, tables.items()):
        predicted_positive = table["true_positives"] + table["false_positives"] > 0
        ax.plot(table["threshold"], table["precision"].where(predicted_positive), marker=".", label="Precision")
        for metric, label in [("recall", "Recall"), ("specificity", "Specificity"), ("f1", "F1")]:
            ax.plot(table["threshold"], table[metric], marker=".", label=label)
        ax.axvline(default_threshold, ls=":", color="gray")
        ax.set(title=_label(name), xlabel="Threshold", ylim=(0, 1))
    axes[0].set_ylabel("Score")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=4)
    fig.suptitle("Threshold trade-offs - test set (no threshold selected; dotted line = default; "
                 "precision omitted where nothing is predicted positive)")
    return fig


def importance_figure(tables: dict, top_n: int = 10) -> Figure:
    fig = _new_figure(5.5 * len(tables), 5)
    axes = np.atleast_1d(fig.subplots(1, len(tables)))
    for ax, (name, table) in zip(axes, tables.items()):
        top = table.head(top_n).iloc[::-1]
        ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color=COLORS.get(name))
        ax.set(title=_label(name), xlabel="Mean drop in ROC-AUC when shuffled")
    fig.suptitle(f"Permutation importance, top {top_n} - test-set sample (model reliance, not causation)")
    return fig


def odds_ratio_figure(table) -> Figure:
    table = table.sort_values("odds_ratio")
    fig = _new_figure(7, 0.35 * len(table) + 1.5)
    ax = fig.subplots()
    rows = np.arange(len(table))
    ax.errorbar(table["odds_ratio"], rows, fmt="o", color="steelblue", capsize=3,
                xerr=[table["odds_ratio"] - table["ci_lower"], table["ci_upper"] - table["odds_ratio"]])
    ax.axvline(1, ls="--", color="gray")
    ax.set_xscale("log")
    ticks = [t for t in [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8]
             if table["ci_lower"].min() / 1.2 <= t <= table["ci_upper"].max() * 1.2]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(rows, table["feature"])
    ax.set(xlabel="Adjusted odds ratio per unit (95% CI, log scale)",
           title="Baseline logistic regression - adjusted associations")
    return fig


def bmi_by_age_figure(table) -> Figure:
    fig = _new_figure(8, 4.5)
    ax = fig.subplots()
    ax.errorbar(table["age_group"], table["bmi_odds_ratio"], fmt="o-", color="steelblue", capsize=3,
                yerr=[table["bmi_odds_ratio"] - table["ci_lower"], table["ci_upper"] - table["bmi_odds_ratio"]])
    ax.axhline(1, ls="--", color="gray")
    ax.set(xlabel="Age group", ylabel="BMI odds ratio per 1 unit (95% CI)",
           title="BMI association by age band (BMI x Age model)")
    ax.tick_params(axis="x", rotation=45)
    return fig