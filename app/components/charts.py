"""Presentation components: charts. Inputs are exported or analytics tables; outputs are Plotly figures."""
import plotly.graph_objects as go

from app.components.tables import model_label

COLORS = {"logistic_regression": "#1f77b4", "random_forest": "#2ca02c", "gradient_boosting": "#ff7f0e"}
METRIC_COLORS = {"precision": "#1f77b4", "recall": "#d62728", "specificity": "#2ca02c", "f1": "#9467bd"}
MIN_CALIBRATION_COUNT = 50


def _layout(fig: go.Figure, title: str, x: str, y: str, **overrides) -> go.Figure:
    fig.update_layout({"height": 420, "margin": dict(t=50, b=40), **overrides},
                      title=title, xaxis_title=x, yaxis_title=y)
    return fig


def roc_chart(curves: dict, roc_auc: dict) -> go.Figure:
    fig = go.Figure()
    for name, curve in curves.items():
        fig.add_scatter(x=curve["false_positive_rate"], y=curve["true_positive_rate"], mode="lines",
                        name=f"{model_label(name)} (AUC {roc_auc[name]:.3f})", line_color=COLORS.get(name))
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="No skill", line=dict(dash="dash", color="gray"))
    return _layout(fig, "ROC curves (test set)", "False-positive rate (1 - specificity)",
                   "True-positive rate (recall)")


def precision_recall_chart(curves: dict, pr_auc: dict, prevalence: float) -> go.Figure:
    fig = go.Figure()
    for name, curve in curves.items():
        fig.add_scatter(x=curve["recall"], y=curve["precision"], mode="lines",
                        name=f"{model_label(name)} (AP {pr_auc[name]:.3f})", line_color=COLORS.get(name))
    fig.add_hline(prevalence, line_dash="dash", line_color="gray",
                  annotation_text=f"No skill = prevalence {prevalence:.3f}")
    return _layout(fig, "Precision-recall curves (test set)", "Recall", "Precision", yaxis_range=[0, 1])


def confusion_matrix_chart(counts) -> go.Figure:
    z = [[int(counts["true_negatives"]), int(counts["false_positives"])],
         [int(counts["false_negatives"]), int(counts["true_positives"])]]
    fig = go.Figure(go.Heatmap(z=z, x=["Predicted 0", "Predicted 1"], y=["Actual 0", "Actual 1"],
                               text=[[f"{v:,}" for v in row] for row in z], texttemplate="%{text}",
                               colorscale="Blues", showscale=False))
    return _layout(fig, "Confusion matrix", "", "", yaxis_autorange="reversed", height=320)


def threshold_chart(table, threshold: float) -> go.Figure:
    fig = go.Figure()
    for metric, color in METRIC_COLORS.items():
        fig.add_scatter(x=table["threshold"], y=table[metric], mode="lines+markers", name=metric.capitalize(),
                        line_color=color)
    fig.add_vline(threshold, line_dash="dot", line_color="black", annotation_text=f"selected {threshold:.2f}")
    return _layout(fig, "Metric trade-offs across thresholds (test set)", "Classification threshold",
                   "Metric value", yaxis_range=[0, 1])


def imbalance_chart(table, metrics: list[str]) -> go.Figure:
    fig = go.Figure()
    models = table["model"].unique()
    for _, row in table.iterrows():
        name = row["strategy"].replace("_", " ") + ("" if len(models) == 1 else f" ({model_label(row['model'])})")
        fig.add_bar(x=[m.replace("_", " ") for m in metrics], y=row[metrics], name=name,
                    text=[f"{v:.2f}" for v in row[metrics]], textposition="outside")
    title = f"{', '.join(model_label(m) for m in models)}: test-set metrics by training strategy (threshold 0.5)"
    return _layout(fig, title, "", "Metric value", barmode="group", yaxis_range=[0, 1.05])


def calibration_chart(tables: dict, min_count: int = MIN_CALIBRATION_COUNT) -> go.Figure:
    fig = go.Figure()
    for name, table in tables.items():
        shown = table[table["count"] >= min_count]
        fig.add_scatter(x=shown["mean_predicted"], y=shown["observed_frequency"], mode="lines+markers",
                        name=model_label(name), line_color=COLORS.get(name), customdata=shown["count"],
                        hovertemplate="predicted %{x:.3f}<br>observed %{y:.3f}<br>n = %{customdata:,}")
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                    line=dict(dash="dash", color="gray"))
    return _layout(fig, f"Calibration (test set; bins with fewer than {min_count} rows hidden)",
                   "Mean predicted probability", "Observed positive fraction",
                   xaxis_range=[0, 1], yaxis_range=[0, 1])


def importance_chart(table, top_n: int = 10, title: str = "Permutation importance") -> go.Figure:
    top = table.nlargest(top_n, "importance_mean").iloc[::-1]
    fig = go.Figure(go.Bar(x=top["importance_mean"], y=top["feature"], orientation="h",
                           error_x=dict(type="data", array=top["importance_std"])))
    return _layout(fig, title, "Mean drop in ROC AUC when shuffled", "", height=max(320, 28 * top_n))


def odds_ratio_chart(table, title: str) -> go.Figure:
    ordered = table.sort_values("odds_ratio")
    fig = go.Figure(go.Scatter(
        x=ordered["odds_ratio"], y=ordered["feature"], mode="markers",
        error_x=dict(type="data", symmetric=False, array=ordered["ci_upper"] - ordered["odds_ratio"],
                     arrayminus=ordered["odds_ratio"] - ordered["ci_lower"])))
    fig.add_vline(1, line_dash="dash", line_color="gray")
    return _layout(fig, title, "Odds ratio (log scale, 95% CI)", "", xaxis_type="log",
                   height=max(360, 24 * len(ordered)), showlegend=False)


def bmi_by_age_chart(table) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=table["age_group"], y=table["bmi_odds_ratio"], mode="lines+markers",
        error_y=dict(type="data", symmetric=False, array=table["ci_upper"] - table["bmi_odds_ratio"],
                     arrayminus=table["bmi_odds_ratio"] - table["ci_lower"])))
    fig.add_hline(1, line_dash="dash", line_color="gray")
    return _layout(fig, "Odds ratio per 1-unit BMI increase, by age group", "Age group",
                   "BMI odds ratio (95% CI)", showlegend=False)


TARGET_COLORS = {0: "#7f9fbf", 1: "#d62728"}


def target_distribution_chart(summary) -> go.Figure:
    labels = [f"{code}: {label}" for code, label in zip(summary["category"], summary["label"])]
    fig = go.Figure(go.Bar(x=labels, y=summary["count"],
                           marker_color=[TARGET_COLORS[c] for c in summary["category"]],
                           text=[f"{n:,} ({p:.1f}%)" for n, p in zip(summary["count"], summary["percentage"])],
                           textposition="outside"))
    return _layout(fig, "Target distribution", "", "Respondents", height=360, showlegend=False,
                   yaxis_range=[0, summary["count"].max() * 1.15])


def category_share_chart(summary, title: str) -> go.Figure:
    fig = go.Figure(go.Bar(x=summary["percentage"], y=summary["label"].astype(str), orientation="h",
                           text=[f"{p:.1f}%" for p in summary["percentage"]], textposition="outside",
                           customdata=summary["count"],
                           hovertemplate="%{y}: %{x:.2f}% (n = %{customdata:,})<extra></extra>"))
    return _layout(fig, title, "Share of respondents (%)", "", yaxis_autorange="reversed", showlegend=False,
                   height=max(320, 32 * len(summary)), xaxis_range=[0, summary["percentage"].max() * 1.2])


def prevalence_chart(prevalence, overall: float, title: str) -> go.Figure:
    fig = go.Figure(go.Bar(x=prevalence["target_percentage"], y=prevalence["label"].astype(str), orientation="h",
                           marker_color=TARGET_COLORS[1],
                           text=[f"{p:.1f}%" for p in prevalence["target_percentage"]], textposition="outside",
                           customdata=prevalence[["target_count", "total_observations"]],
                           hovertemplate="%{y}: %{x:.2f}% (%{customdata[0]:,} of %{customdata[1]:,})<extra></extra>"))
    fig.add_vline(overall, line_dash="dash", line_color="gray", annotation_text=f"overall {overall:.1f}%")
    return _layout(fig, title, "Category 1 share within group (%)", "", yaxis_autorange="reversed",
                   showlegend=False, height=max(320, 32 * len(prevalence)),
                   xaxis_range=[0, prevalence["target_percentage"].max() * 1.2])


def value_distribution_chart(table, target_labels: dict, title: str, x_title: str) -> go.Figure:
    fig = go.Figure()
    for code in table.columns:
        fig.add_scatter(x=table.index, y=table[code], mode="lines+markers", name=target_labels[code],
                        line_color=TARGET_COLORS[code], marker_size=4)
    return _layout(fig, title, x_title, "Share of target group (%)", legend=dict(orientation="h", y=-0.2))


def binary_comparison_chart(table, labels: dict) -> go.Figure:
    ordered = table.sort_values("difference_percentage_points")
    names = [labels.get(f, f) for f in ordered["feature"]]
    fig = go.Figure()
    fig.add_bar(y=names, x=ordered["category_1_percent_no"], orientation="h", name="Code 0 (No; Female for Sex)",
                marker_color=TARGET_COLORS[0])
    fig.add_bar(y=names, x=ordered["category_1_percent_yes"], orientation="h", name="Code 1 (Yes; Male for Sex)",
                marker_color=TARGET_COLORS[1])
    return _layout(fig, "Category 1 share by answer to each binary indicator", "Category 1 share (%)", "",
                   barmode="group", height=max(420, 40 * len(ordered)), legend=dict(orientation="h", y=-0.1))


def interval_chart(table, title: str, x_title: str, reference: float | None = None,
                   from_zero: bool = False) -> go.Figure:
    """Point estimates with confidence intervals; table columns: group, estimate, lower, upper, n."""
    fig = go.Figure(go.Scatter(
        x=table["estimate"], y=table["group"].astype(str), mode="markers", marker=dict(size=9, color=TARGET_COLORS[1]),
        error_x=dict(type="data", symmetric=False, array=table["upper"] - table["estimate"],
                     arrayminus=table["estimate"] - table["lower"]),
        customdata=table[["lower", "upper", "n"]],
        hovertemplate="%{y}: %{x:.2f} [%{customdata[0]:.2f}, %{customdata[1]:.2f}], n = %{customdata[2]:,}"
                      "<extra></extra>"))
    if reference is not None:
        fig.add_vline(reference, line_dash="dash", line_color="gray")
    return _layout(fig, title, x_title, "", yaxis_autorange="reversed", showlegend=False,
                   height=max(220, 36 * len(table) + 120), xaxis_rangemode="tozero" if from_zero else "normal")


def cramers_v_chart(table, labels: dict) -> go.Figure:
    """Cramér's V per chi-square hypothesis; table columns: hypothesis, feature, effect_value."""
    names = [f"{h}: {labels.get(f, f)}" for h, f in zip(table["hypothesis"], table["feature"])]
    fig = go.Figure(go.Bar(x=table["effect_value"], y=names, orientation="h", marker_color=TARGET_COLORS[1],
                           text=[f"{v:.3f}" for v in table["effect_value"]], textposition="outside"))
    return _layout(fig, "Effect size of each chi-square association (Cramér's V)", "Cramér's V", "",
                   yaxis_autorange="reversed", showlegend=False, height=max(300, 44 * len(table)),
                   xaxis_range=[0, max(table["effect_value"].max() * 1.25, 0.1)])


def correlation_heatmap(matrix, labels: dict) -> go.Figure:
    names = [labels.get(c, c) for c in matrix.columns]
    fig = go.Figure(go.Heatmap(z=matrix.values, x=names, y=names, zmin=-1, zmax=1, colorscale="RdBu_r",
                               text=[[f"{v:.2f}" for v in row] for row in matrix.values], texttemplate="%{text}",
                               colorbar_title="rho"))
    return _layout(fig, "Spearman rank correlation", "", "", height=560, yaxis_autorange="reversed")