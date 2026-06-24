from __future__ import annotations

from io import BytesIO
from typing import List, Optional

import pandas as pd


def aggregate_dfs(dfs: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    try:
        return pd.concat(dfs, ignore_index=True)
    except ValueError:
        return None


def plot_comparison(df: pd.DataFrame):
    """Create a matplotlib/seaborn grouped bar plot with CIs and bottom labels.

    Returns: (fig, ax)
    """
    if df is None or df.empty:
        return None, None

    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
    except Exception:
        return None, None

    # Order categories to make plot stable
    x_labels = list(pd.unique(df["TaskLabel"]))
    models = list(pd.unique(df["Model"]))

    fig_width = 16
    fig_height = max(6, len(x_labels) * 0.6)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=600)
    sns.set_style("whitegrid")
    palette = sns.color_palette("colorblind", n_colors=len(models))

    # Ensure order
    plot_df = df.copy()
    plot_df["TaskLabel"] = pd.Categorical(plot_df["TaskLabel"], categories=x_labels, ordered=True)
    plot_df["Model"] = pd.Categorical(plot_df["Model"], categories=models, ordered=True)

    sns.barplot(
        data=plot_df,
        x="TaskLabel",
        y="Mean",
        hue="Model",
        capsize=0.2,
        err_kws={"linewidth": 1},
        alpha=0.85,
        palette=palette,
        ax=ax,
    )

    # Customize plot
    plt.title('Performance Comparison Across Tasks')
    plt.xlabel('Task', fontsize=16)
    plt.ylabel('Score', fontsize=16)

    # Rotate x-axis labels for better readability 
    plt.xticks(rotation=45, ha='right', fontsize=14)

    # Set yticks fontsize
    plt.yticks(fontsize=16)

    # Add manual CI whiskers to match autoeval style and numbers at bottom
    n_models = len(models)
    for i, model in enumerate(models):
        m_df = plot_df[plot_df["Model"] == model]
        for j, label in enumerate(x_labels):
            row = m_df[m_df["TaskLabel"] == label]
            if row.empty:
                continue
            y_mean = float(row["Mean"].iloc[0])
            low = float(row["Lower"].iloc[0])
            up = float(row["Upper"].iloc[0])
            # Compute bar position:
            x = j + (i - n_models / 2 + 0.5) * (0.8 / max(n_models, 1))
            color = palette[i]
            ax.vlines(x=x, ymin=low, ymax=up, color=color, linewidth=2, alpha=0.7)
            # value at bottom
            try:
                x_text = x + 0.02
                ax.text(x_text, 0.01, f"{y_mean:.3f}", ha="center", va="bottom", fontsize=8, rotation=90,
                        color="black", fontweight="bold")
            except Exception:
                pass

    # Adjust legend position
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout to prevent label cutoff
    fig.tight_layout()
    return fig, ax


def compute_paired_delta_stats(reports_by_model: dict, baseline_model: str, z: float = 1.96) -> dict:
    """Confidence interval of the *paired* per-dataset score difference (model - baseline).

    For zero-shot frameworks (e.g. PGYM) every model is scored on the same datasets, so the
    honest uncertainty of a model-vs-baseline comparison is the CI of the per-dataset deltas,
    not the spread of either model's scores across datasets. This compares the two models on
    the *same* dataset, one at a time, which cancels per-dataset difficulty.

    ``reports_by_model`` maps model name -> a report exposing ``aggregated_members``
    (combined_task_name -> [dataset_name]) and ``individual_results`` (dataset_name ->
    RankingResult with ``.scc.mean`` / ``.ndcg.mean``).

    Returns ``{(model, task, metric): {"mean_pp": signed_mean_delta_pp,
    "ci_pp": half_width_pp, "n": n_shared}}`` (95% CI, normal approximation z=1.96).
    Entries are only produced where >= 2 datasets are shared with the baseline; everything
    else is omitted and the plot falls back to its default whisker.
    """
    try:
        import numpy as np
    except Exception:
        return {}

    stats: dict = {}
    baseline = reports_by_model.get(baseline_model)
    if baseline is None:
        return stats
    base_members = getattr(baseline, "aggregated_members", {}) or {}
    base_individual = getattr(baseline, "individual_results", {}) or {}

    for model_name, report in reports_by_model.items():
        if model_name == baseline_model:
            continue
        members_by_task = getattr(report, "aggregated_members", {}) or {}
        individual = getattr(report, "individual_results", {}) or {}
        for task, members in members_by_task.items():
            base_task_members = set(base_members.get(task, []))
            shared = [d for d in members
                      if d in base_task_members and d in individual and d in base_individual]
            if len(shared) < 2:
                continue
            for metric in ("scc", "ndcg"):
                try:
                    m_scores = np.array([getattr(individual[d], metric).mean for d in shared], dtype=float)
                    b_scores = np.array([getattr(base_individual[d], metric).mean for d in shared], dtype=float)
                except Exception:
                    continue
                deltas = m_scores - b_scores
                n = len(deltas)
                se = float(np.std(deltas, ddof=1)) / (n ** 0.5)
                stats[(model_name, task, metric)] = {
                    "mean_pp": float(deltas.mean()) * 100.0,
                    "ci_pp": z * se * 100.0,
                    "n": n,
                }
    return stats


def plot_delta_comparison(df: pd.DataFrame, baseline_model: str, paired_stats: Optional[dict] = None):
    """Create a matplotlib/seaborn grouped bar plot showing deltas relative to a baseline model.

    When ``paired_stats`` is provided (see :func:`compute_paired_delta_stats`), the whisker for a
    matching (model, task, metric) is the 95% CI of the paired per-dataset difference centred on
    the mean delta, and a '*' marks comparisons whose CI excludes zero. Bars without paired stats
    keep the default whisker (the model's own band re-centred on the baseline mean).

    Returns: (fig, ax)
    """
    if df is None or df.empty or baseline_model not in df["Model"].unique():
        return None, None

    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return None, None

    # Calculate deltas
    # We need to match tasks across models.
    # The df has columns: Task, Metric, TaskLabel, Model, Mean, Lower, Upper
    # Supervised also has: Test Set, Protocol
    # Some rows might be missing for some models.

    # Check if this is supervised (has Test Set column) or zero-shot
    has_test_set = "Test Set" in df.columns

    # Identify the baseline rows
    baseline_df = df[df["Model"] == baseline_model].copy()
    # Create a key for matching: Task | Test Set (if exists) | Metric
    df_key = baseline_df["Task"].astype(str)
    if has_test_set:
        df_key += "|" + baseline_df["Test Set"].fillna("").astype(str)
    df_key += "|" + baseline_df["Metric"].astype(str)
    baseline_df["key"] = df_key
    baseline_map = baseline_df.set_index("key")

    # Models to plot (excluding baseline)
    other_models = [m for m in df["Model"].unique() if m != baseline_model]
    if not other_models:
        return None, None

    delta_rows = []
    baseline_ci_rows = []
    used_paired = False

    for _, row in df.iterrows():
        # Build the same key structure
        key = str(row["Task"])
        if has_test_set:
            key += "|" + str(row.get("Test Set", ""))
        key += "|" + str(row["Metric"])

        if key not in baseline_map.index:
            continue

        base_row = baseline_map.loc[key]
        if isinstance(base_row, pd.DataFrame):
            base_row = base_row.iloc[0]

        # Delta in percentage points (assuming metrics are 0-1)
        # If the metric is already in percentage, this is still fine.
        mean_delta = (row["Mean"] - base_row["Mean"]) * 100.0
        lower_delta = (row["Lower"] - base_row["Mean"]) * 100.0
        upper_delta = (row["Upper"] - base_row["Mean"]) * 100.0

        # If a paired CI is available for this (model, task, metric), replace the whisker
        # with the 95% CI of the per-dataset paired difference, centred on the mean delta.
        significant = None
        if paired_stats:
            pstat = paired_stats.get((row["Model"], str(row["Task"]), str(row["Metric"])))
            if pstat is not None:
                # Use the raw signed paired mean for the bar + CI centre so they stay
                # consistent. The aggregated row["Mean"] is abs()'d for scc/Spearman
                # (_maybe_metric_abs), which would mis-place the whisker if a mean is < 0.
                mean_delta = float(pstat["mean_pp"])
                ci = float(pstat["ci_pp"])
                lower_delta = mean_delta - ci
                upper_delta = mean_delta + ci
                significant = bool(abs(mean_delta) > ci)
                used_paired = True

        if row["Model"] == baseline_model:
            baseline_ci_rows.append({
                "TaskLabel": row["TaskLabel"],
                "Lower": lower_delta,
                "Upper": upper_delta,
                "key": key
            })
        else:
            delta_rows.append({
                "Model": row["Model"],
                "TaskLabel": row["TaskLabel"],
                "Mean": mean_delta,
                "Lower": lower_delta,
                "Upper": upper_delta,
                "Significant": significant,
                "key": key
            })

    if not delta_rows:
        return None, None

    delta_df = pd.DataFrame(delta_rows)

    x_labels = list(pd.unique(delta_df["TaskLabel"]))
    models = list(pd.unique(delta_df["Model"]))

    fig_width = 16
    fig_height = max(6, len(x_labels) * 0.6)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=600)
    sns.set_style("whitegrid")
    palette = sns.color_palette("colorblind", n_colors=len(models))

    # Ensure order
    delta_df["TaskLabel"] = pd.Categorical(delta_df["TaskLabel"], categories=x_labels, ordered=True)
    delta_df["Model"] = pd.Categorical(delta_df["Model"], categories=models, ordered=True)

    sns.barplot(
        data=delta_df,
        x="TaskLabel",
        y="Mean",
        hue="Model",
        palette=palette,
        alpha=0.85,
        ax=ax
    )

    plt.axhline(0, color='black', linewidth=1.5)
    title = f'Performance Delta relative to {baseline_model}'
    if used_paired:
        title += '\n(whiskers: 95% CI of paired per-dataset difference;  * = significant)'
    plt.title(title, fontsize=16 if used_paired else 18)
    plt.xlabel('Task', fontsize=16)
    plt.ylabel('Delta (pp)', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(fontsize=16)

    n_models = len(models)
    has_sig = "Significant" in delta_df.columns
    # Label offsets: small fraction of the data range for the (tight) paired axis,
    # absolute pp for the default (+/-20) axis so existing plots are unchanged.
    if used_paired:
        _span = max(1.0, float(delta_df[["Lower", "Upper"]].abs().to_numpy().max()))
        off_up, off_dn = _span * 0.05, _span * 0.10
    else:
        off_up, off_dn = 2.0, 6.0
    for i, model in enumerate(models):
        m_df = delta_df[delta_df["Model"] == model]
        for j, label in enumerate(x_labels):
            row = m_df[m_df["TaskLabel"] == label]
            if row.empty:
                continue
            y_mean = float(row["Mean"].iloc[0])
            low = float(row["Lower"].iloc[0])
            up = float(row["Upper"].iloc[0])
            sig = row["Significant"].iloc[0] if has_sig else None

            x = j + (i - n_models / 2 + 0.5) * (0.8 / max(n_models, 1))
            color = palette[i]
            # Grey out the whisker + label when the paired CI overlaps zero (not significant).
            # ``sig`` may be None (no paired stats), or a numpy bool, so avoid `is False`.
            not_significant = sig is not None and not bool(sig)
            ebar_color = "#9e9e9e" if not_significant else color

            # Error bars
            ax.vlines(x=x, ymin=low, ymax=up, color=ebar_color, linewidth=2)
            # Cap lines
            ax.hlines(y=[low, up], xmin=x - 0.05, xmax=x + 0.05, color=ebar_color, linewidth=2)

            # Value text; '*' marks a statistically significant difference (CI excludes 0).
            try:
                star = " *" if bool(sig) else ""
                text_color = "#9e9e9e" if not_significant else "black"
                text_y = up + off_up if y_mean >= 0 else low - off_dn
                ax.text(x, text_y, f"{y_mean:+.1f}{star}", ha="center", va="bottom", fontsize=9,
                        color=text_color, fontweight="bold")
            except Exception:
                pass

    if used_paired:
        # Paired CIs are small; autoscale tightly so the bars/whiskers are visible
        # (the fixed +/-20 floor below would squash them).
        span = max(1.0, float(delta_df[["Lower", "Upper", "Mean"]].abs().to_numpy().max())) * 1.35
        ax.set_ylim(-span, span)
    else:
        # Set y-axis limits to at least -20 to +20
        current_ylim = ax.get_ylim()
        new_ylim = (min(current_ylim[0], -20), max(current_ylim[1], 20))
        ax.set_ylim(new_ylim)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()

    return fig, ax


def fig_to_png_bytes(fig) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
    buf.seek(0)
    return buf.getvalue()


def fig_to_pdf_bytes(fig) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="pdf", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()
