"""Presentation components: charts. Inputs are exported tables; outputs are Plotly figures."""
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