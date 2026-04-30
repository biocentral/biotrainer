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


def plot_delta_comparison(df: pd.DataFrame, baseline_model: str):
    """Create a matplotlib/seaborn grouped bar plot showing deltas relative to a baseline model.

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
    plt.title(f'Performance Delta relative to {baseline_model}', fontsize=18)
    plt.xlabel('Task', fontsize=16)
    plt.ylabel('Delta (pp)', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(fontsize=16)

    n_models = len(models)
    for i, model in enumerate(models):
        m_df = delta_df[delta_df["Model"] == model]
        for j, label in enumerate(x_labels):
            row = m_df[m_df["TaskLabel"] == label]
            if row.empty:
                continue
            y_mean = float(row["Mean"].iloc[0])
            low = float(row["Lower"].iloc[0])
            up = float(row["Upper"].iloc[0])

            x = j + (i - n_models / 2 + 0.5) * (0.8 / max(n_models, 1))
            color = palette[i]

            # Error bars
            ax.vlines(x=x, ymin=low, ymax=up, color=color, linewidth=2)
            # Cap lines
            ax.hlines(y=[low, up], xmin=x - 0.05, xmax=x + 0.05, color=color, linewidth=2)

            # Value text
            try:
                text_y = up + 2.0 if y_mean >= 0 else low - 6.0
                ax.text(x, text_y, f"{y_mean:+.1f}", ha="center", va="bottom", fontsize=9,
                        color="black", fontweight="bold")
            except Exception:
                pass

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
