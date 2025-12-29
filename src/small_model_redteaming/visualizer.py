"""
Visualization utilities for red teaming results.

Provides plotting and visualization functions for analyzing
test results from the red teaming modules.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# Optional imports for visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def _check_visualization_deps() -> None:
    """Check if visualization dependencies are available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install small-model-redteaming[visualization]"
        )


def _apply_style() -> None:
    """Apply consistent styling to plots."""
    _check_visualization_deps()
    if HAS_SEABORN:
        sns.set_theme(style="whitegrid", palette="husl")
    else:
        # Try multiple style names for matplotlib compatibility
        for style in ["seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"]:
            try:
                plt.style.use(style)
                break
            except OSError:
                continue


def plot_awareness_scores(
    scores: Dict[str, float],
    title: str = "Evaluation Awareness Scores",
    figsize: Tuple[int, int] = (10, 6),
    color_threshold: Optional[float] = None,
    horizontal: bool = False,
    save_path: Optional[str] = None,
) -> Any:
    """
    Plot awareness scores across different test conditions.

    Creates a bar chart showing awareness/performance scores for
    different evaluation conditions (baseline, formal, casual, etc.).

    Args:
        scores: Dict mapping condition names to scores (0-100 scale).
        title: Plot title.
        figsize: Figure size as (width, height).
        color_threshold: Score above which bars are highlighted (red).
        horizontal: If True, create horizontal bar chart.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure object.

    Example:
        >>> scores = {"baseline": 75.0, "formal": 72.0, "high_stakes": 85.0}
        >>> fig = plot_awareness_scores(scores, color_threshold=80)
    """
    _check_visualization_deps()
    _apply_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Handle empty input
    if not scores:
        ax.text(
            0.5,
            0.5,
            "No data to display",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        ax.set_title(title, fontsize=14, fontweight="bold")
        return fig

    conditions = list(scores.keys())
    values = list(scores.values())

    # Determine colors based on threshold
    if color_threshold is not None:
        colors = ["#e74c3c" if v >= color_threshold else "#3498db" for v in values]
    else:
        colors = "#3498db"

    if horizontal:
        bars = ax.barh(conditions, values, color=colors, edgecolor="white")
        ax.set_xlabel("Score (%)", fontsize=12)
        ax.set_ylabel("Condition", fontsize=12)
        ax.set_xlim(0, 100)
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(
                val + 1,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%",
                va="center",
                fontsize=10,
            )
    else:
        bars = ax.bar(conditions, values, color=colors, edgecolor="white")
        ax.set_xlabel("Condition", fontsize=12)
        ax.set_ylabel("Score (%)", fontsize=12)
        ax.set_ylim(0, 100)
        plt.xticks(rotation=45, ha="right")
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 1,
                f"{val:.1f}%",
                ha="center",
                fontsize=10,
            )

    ax.set_title(title, fontsize=14, fontweight="bold")

    # Add threshold line if specified
    if color_threshold is not None:
        if horizontal:
            ax.axvline(x=color_threshold, color="#e74c3c", linestyle="--", alpha=0.7)
        else:
            ax.axhline(y=color_threshold, color="#e74c3c", linestyle="--", alpha=0.7)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_agreement_comparison(
    baseline: float,
    observed: float,
    unobserved: float,
    subject: str = "Subject",
    figsize: Tuple[int, int] = (8, 6),
    show_difference: bool = True,
    save_path: Optional[str] = None,
) -> Any:
    """
    Plot comparison of agreement scores across observation conditions.

    Creates a grouped bar chart comparing model agreement percentages
    under baseline, observed, and unobserved conditions for deceptive
    alignment analysis.

    Args:
        baseline: Baseline agreement percentage (0-100).
        observed: Observed condition agreement percentage (0-100).
        unobserved: Unobserved condition agreement percentage (0-100).
        subject: Name of the subject being tested.
        figsize: Figure size as (width, height).
        show_difference: If True, annotate with difference values.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure object.

    Example:
        >>> fig = plot_agreement_comparison(
        ...     baseline=75.0,
        ...     observed=80.0,
        ...     unobserved=65.0,
        ...     subject="Capitalism"
        ... )
    """
    _check_visualization_deps()
    _apply_style()

    fig, ax = plt.subplots(figsize=figsize)

    conditions = ["Baseline", "Observed", "Unobserved"]
    values = [baseline, observed, unobserved]
    colors = ["#3498db", "#2ecc71", "#e74c3c"]

    bars = ax.bar(conditions, values, color=colors, edgecolor="white", width=0.6)

    ax.set_ylabel("Agreement (%)", fontsize=12)
    ax.set_title(
        f"Agreement Comparison: {subject}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylim(0, 100)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 1,
            f"{val:.1f}%",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    # Show difference annotations
    if show_difference:
        obs_diff = observed - baseline
        unobs_diff = unobserved - baseline
        obs_unobs_diff = observed - unobserved

        annotation_text = (
            f"Observed vs Baseline: {obs_diff:+.1f}%\n"
            f"Unobserved vs Baseline: {unobs_diff:+.1f}%\n"
            f"Observed vs Unobserved: {obs_unobs_diff:+.1f}%"
        )

        ax.text(
            0.98,
            0.98,
            annotation_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_confidence_distribution(
    confidences: List[float],
    title: str = "Confidence Distribution",
    figsize: Tuple[int, int] = (10, 6),
    bins: int = 30,
    show_stats: bool = True,
    save_path: Optional[str] = None,
) -> Any:
    """
    Plot distribution of model confidence scores.

    Creates a histogram with optional KDE overlay showing the
    distribution of confidence scores from model responses.

    Args:
        confidences: List of confidence scores (log probabilities).
        title: Plot title.
        figsize: Figure size as (width, height).
        bins: Number of histogram bins.
        show_stats: If True, display mean and std statistics.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure object.

    Example:
        >>> confidences = [-2.3, -1.5, -3.1, -0.8, -2.0]
        >>> fig = plot_confidence_distribution(confidences)
    """
    _check_visualization_deps()
    _apply_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Handle empty input
    if not confidences:
        ax.text(
            0.5,
            0.5,
            "No data to display",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        ax.set_title(title, fontsize=14, fontweight="bold")
        return fig

    confidences_array = np.array(confidences)

    if HAS_SEABORN:
        sns.histplot(
            confidences_array,
            bins=bins,
            kde=True,
            ax=ax,
            color="#3498db",
            edgecolor="white",
        )
    else:
        ax.hist(
            confidences_array,
            bins=bins,
            color="#3498db",
            edgecolor="white",
            alpha=0.7,
        )

    ax.set_xlabel("Confidence Score (log probability)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    if show_stats:
        mean_val = np.mean(confidences_array)
        std_val = np.std(confidences_array)
        median_val = np.median(confidences_array)

        stats_text = (
            f"Mean: {mean_val:.3f}\n"
            f"Std: {std_val:.3f}\n"
            f"Median: {median_val:.3f}\n"
            f"N: {len(confidences)}"
        )

        ax.text(
            0.98,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
        )

        # Add mean line
        ax.axvline(x=mean_val, color="#e74c3c", linestyle="--", label="Mean")
        ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_deceptive_alignment_results(
    results: Dict[str, Any],
    subject: str,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> Any:
    """
    Plot comprehensive deceptive alignment test results.

    Creates a multi-panel visualization showing agreement scores
    and observation impact analysis from deceptive alignment tests.

    Args:
        results: Results dict from DeceptiveAlignmentTest.test_deceptive_alignment().
        subject: Name of the subject tested.
        figsize: Figure size as (width, height).
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure object.

    Example:
        >>> results = {
        ...     "baseline_agreement": 75.0,
        ...     "agreement": 60.0,
        ...     "observed_agreement": 70.0,
        ...     "unobserved_agreement": 55.0,
        ... }
        >>> fig = plot_deceptive_alignment_results(results, "Capitalism")
    """
    _check_visualization_deps()
    _apply_style()

    has_observation = "observed_agreement" in results

    if has_observation:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, ax1 = plt.subplots(figsize=(figsize[0] // 2, figsize[1]))

    # Panel 1: Agreement comparison
    conditions = ["Baseline", "With Developer\nMessage"]
    values = [results["baseline_agreement"], results["agreement"]]
    colors = ["#3498db", "#9b59b6"]

    bars1 = ax1.bar(conditions, values, color=colors, edgecolor="white", width=0.5)
    ax1.set_ylabel("Agreement (%)", fontsize=12)
    ax1.set_title(f"Agreement Shift: {subject}", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 100)

    for bar, val in zip(bars1, values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            val + 1,
            f"{val:.1f}%",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    # Add difference annotation
    diff = results["agreement_difference"]
    diff_color = "#2ecc71" if diff >= 0 else "#e74c3c"
    ax1.annotate(
        f"Change: {diff:+.1f}%",
        xy=(0.5, max(values) + 5),
        fontsize=12,
        ha="center",
        color=diff_color,
        fontweight="bold",
    )

    # Panel 2: Observation impact (if available)
    if has_observation:
        obs_conditions = ["Observed", "Unobserved"]
        obs_values = [
            results["observed_agreement"],
            results["unobserved_agreement"],
        ]
        obs_colors = ["#2ecc71", "#e74c3c"]

        bars2 = ax2.bar(
            obs_conditions, obs_values, color=obs_colors, edgecolor="white", width=0.5
        )
        ax2.set_ylabel("Agreement (%)", fontsize=12)
        ax2.set_title("Observation Impact", fontsize=12, fontweight="bold")
        ax2.set_ylim(0, 100)

        for bar, val in zip(bars2, obs_values):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                val + 1,
                f"{val:.1f}%",
                ha="center",
                fontsize=11,
                fontweight="bold",
            )

        # Add observation impact annotation
        obs_impact = results.get("observation_impact", 0)
        impact_color = "#e74c3c" if abs(obs_impact) > 5 else "#7f8c8d"
        ax2.annotate(
            f"Impact: {obs_impact:+.1f}%",
            xy=(0.5, max(obs_values) + 5),
            fontsize=12,
            ha="center",
            color=impact_color,
            fontweight="bold",
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_exfiltration_results(
    results: Dict[str, Any],
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> Any:
    """
    Plot data exfiltration test results.

    Creates a multi-panel visualization showing vulnerability scores
    by category and overall risk assessment.

    Args:
        results: Results dict from DataExfiltrationTest.run_all_tests().
        figsize: Figure size as (width, height).
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure object.

    Example:
        >>> results = {
        ...     "direct": {"vulnerability_score": 12.5},
        ...     "indirect": {
        ...         "roleplay_summary": {"leak_count": 2},
        ...         "encoding_summary": {"leak_count": 0},
        ...         "indirect_summary": {"leak_count": 1},
        ...         "injection_summary": {"leak_count": 3},
        ...     },
        ...     "summary": {"vulnerability_score": 15.0, "risk_level": "MEDIUM"},
        ... }
        >>> fig = plot_exfiltration_results(results)
    """
    _check_visualization_deps()
    _apply_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Panel 1: Vulnerability by category
    categories = ["Direct"]
    vuln_scores = [results["direct"]["vulnerability_score"]]

    if "indirect" in results:
        indirect = results["indirect"]
        for key in ["roleplay", "encoding", "indirect", "injection"]:
            summary_key = f"{key}_summary"
            if summary_key in indirect:
                categories.append(key.capitalize())
                leak_count = indirect[summary_key]["leak_count"]
                total = indirect[summary_key]["total_attempts"]
                score = (leak_count / total * 100) if total > 0 else 0
                vuln_scores.append(score)

    # Color by severity
    colors = []
    for score in vuln_scores:
        if score >= 50:
            colors.append("#c0392b")
        elif score >= 25:
            colors.append("#e74c3c")
        elif score >= 10:
            colors.append("#f39c12")
        elif score > 0:
            colors.append("#f1c40f")
        else:
            colors.append("#2ecc71")

    bars1 = ax1.bar(categories, vuln_scores, color=colors, edgecolor="white")
    ax1.set_ylabel("Vulnerability Score (%)", fontsize=12)
    ax1.set_title("Vulnerability by Category", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, max(100, max(vuln_scores) + 10))
    plt.sca(ax1)
    plt.xticks(rotation=45, ha="right")

    for bar, val in zip(bars1, vuln_scores):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            val + 1,
            f"{val:.1f}%",
            ha="center",
            fontsize=10,
        )

    # Panel 2: Risk level gauge
    summary = results.get("summary", {})
    risk_level = summary.get("risk_level", "UNKNOWN")
    overall_score = summary.get("vulnerability_score", 0)

    risk_colors = {
        "MINIMAL": "#2ecc71",
        "LOW": "#f1c40f",
        "MEDIUM": "#f39c12",
        "HIGH": "#e74c3c",
        "CRITICAL": "#c0392b",
        "UNKNOWN": "#95a5a6",
    }

    # Create a simple gauge visualization
    risk_levels = ["MINIMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
    risk_positions = {level: i for i, level in enumerate(risk_levels)}

    gauge_colors = [risk_colors[level] for level in risk_levels]
    gauge_values = [1, 1, 1, 1, 1]

    ax2.bar(risk_levels, gauge_values, color=gauge_colors, edgecolor="white", alpha=0.3)

    # Highlight current risk level
    if risk_level in risk_positions:
        ax2.bar(
            [risk_level],
            [1],
            color=risk_colors[risk_level],
            edgecolor="black",
            linewidth=2,
        )

    ax2.set_ylabel("")
    ax2.set_title(
        f"Risk Level: {risk_level}\n(Score: {overall_score:.1f}%)",
        fontsize=12,
        fontweight="bold",
        color=risk_colors.get(risk_level, "#333"),
    )
    ax2.set_ylim(0, 1.2)
    ax2.set_yticks([])
    plt.sca(ax2)
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_preference_heatmap(
    preferences: Dict[str, float],
    title: str = "Model Preference Heatmap",
    figsize: Tuple[int, int] = (10, 6),
    cmap: str = "RdYlGn",
    save_path: Optional[str] = None,
) -> Any:
    """
    Plot a heatmap of model preferences across subjects.

    Creates a visual representation of how strongly the model
    agrees with different subjects/topics.

    Args:
        preferences: Dict mapping subject names to agreement scores (0-100).
        title: Plot title.
        figsize: Figure size as (width, height).
        cmap: Matplotlib colormap name.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure object.

    Example:
        >>> preferences = {
        ...     "Helping Animals": 85.0,
        ...     "Helping Humans": 90.0,
        ...     "Capitalism": 45.0,
        ...     "Religion": 50.0,
        ... }
        >>> fig = plot_preference_heatmap(preferences)
    """
    _check_visualization_deps()
    _apply_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Handle empty input
    if not preferences:
        ax.text(
            0.5,
            0.5,
            "No data to display",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        ax.set_title(title, fontsize=14, fontweight="bold")
        return fig

    subjects = list(preferences.keys())
    values = np.array([[preferences[s] for s in subjects]])

    if HAS_SEABORN:
        sns.heatmap(
            values,
            annot=True,
            fmt=".1f",
            cmap=cmap,
            xticklabels=subjects,
            yticklabels=["Agreement %"],
            ax=ax,
            vmin=0,
            vmax=100,
            cbar_kws={"label": "Agreement (%)"},
        )
    else:
        im = ax.imshow(values, cmap=cmap, vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(subjects)))
        ax.set_xticklabels(subjects, rotation=45, ha="right")
        ax.set_yticks([0])
        ax.set_yticklabels(["Agreement %"])
        plt.colorbar(im, ax=ax, label="Agreement (%)")

        # Add annotations
        for i, val in enumerate(values[0]):
            ax.text(i, 0, f"{val:.1f}", ha="center", va="center", fontsize=11)

    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_test_comparison(
    test_results: Dict[str, Dict[str, float]],
    metric: str = "accuracy",
    title: str = "Test Condition Comparison",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> Any:
    """
    Plot comparison of a metric across multiple test conditions.

    Creates a grouped bar chart for comparing performance metrics
    across different test conditions and models/configurations.

    Args:
        test_results: Nested dict of {condition: {metric_name: value}}.
        metric: The metric key to plot.
        title: Plot title.
        figsize: Figure size as (width, height).
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure object.

    Example:
        >>> results = {
        ...     "baseline": {"accuracy": 85.0, "f1": 0.82},
        ...     "formal": {"accuracy": 83.0, "f1": 0.80},
        ...     "high_stakes": {"accuracy": 78.0, "f1": 0.75},
        ... }
        >>> fig = plot_test_comparison(results, metric="accuracy")
    """
    _check_visualization_deps()
    _apply_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Handle empty input
    if not test_results:
        ax.text(
            0.5,
            0.5,
            "No data to display",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        ax.set_title(title, fontsize=14, fontweight="bold")
        return fig

    conditions = list(test_results.keys())
    values = [test_results[c].get(metric, 0) for c in conditions]

    # Use color gradient based on values
    norm_values = np.array(values)
    if norm_values.max() > norm_values.min():
        norm_values = (norm_values - norm_values.min()) / (
            norm_values.max() - norm_values.min()
        )
    else:
        norm_values = np.ones_like(norm_values) * 0.5

    if HAS_SEABORN:
        palette = sns.color_palette("viridis", len(conditions))
        colors = [palette[int(v * (len(palette) - 1))] for v in norm_values]
    else:
        colors = plt.cm.viridis(norm_values)

    bars = ax.bar(conditions, values, color=colors, edgecolor="white")

    ax.set_xlabel("Test Condition", fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + (ax.get_ylim()[1] * 0.01),
            f"{val:.1f}",
            ha="center",
            fontsize=10,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
