#!/usr/bin/env python3

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns


def plot_paired_bootstrap(
    result,
    name_a="Model A",
    name_b="Model B",
    alpha=0.05,
    bins=60,
    ax=None,
    title=None,
):
    """
    Visualize a paired_bootstrap_test() result.

    Parameters
    ----------
    result : dict
        Output of paired_bootstrap_test() (must contain the 'diff' array).
    name_a, name_b : str
        Model names, used for axis labels and win/loss annotation.
    alpha : float
        Significance level used only to color the verdict text
        (p_value < alpha -> green "significant", else red "not significant").
    bins : int
        Histogram bin count.
    ax : matplotlib.axes.Axes, optional
        Draw into an existing axis instead of creating a new figure.
    title : str, optional
        Custom title. Defaults to "{name_a} vs {name_b}".

    Returns
    -------
    fig, ax
    """
    diff = result["diff"]
    mean_diff = result["mean_diff"]
    ci_low, ci_high = result["ci_low"], result["ci_high"]
    p_value = result["p_value"]
    p_gt, p_lt = result["p_gt"], result["p_lt"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5.5))
    else:
        fig = ax.figure

    # --- histogram of the bootstrap difference distribution ---
    counts, edges, patches = ax.hist(
        diff,
        bins=bins,
        color="#4C72B0",
        alpha=0.75,
        edgecolor="white",
        linewidth=0.3,
        label=f"bootstrap diffs ({name_a} \u2212 {name_b})",
    )

    # --- shade the tail that determines the p-value ---
    # the smaller tail (min(p_gt, p_lt)) is the one that "disagrees" with
    # the observed direction; we shade both the disagreeing tail (hatched,
    # this is what feeds the p-value) and, lightly, the whole opposite side.
    losing_side = "left" if p_gt >= p_lt else "right"  # side with less mass
    for patch, left_edge in zip(patches, edges[:-1]):
        is_tail = (left_edge < 0) if losing_side == "left" else (left_edge >= 0)
        if is_tail:
            patch.set_facecolor("#C44E52")
            patch.set_hatch("//")
            patch.set_alpha(0.85)

    # --- null hypothesis line ---
    ax.axvline(0, color="black", linestyle="-", linewidth=1.5, zorder=5)
    ax.text(
        0,
        ax.get_ylim()[1] * 1.03,
        "H\u2080: diff = 0",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )

    # --- observed effect (mean) ---
    ax.axvline(mean_diff, color="#2A2A2A", linestyle="--", linewidth=1.8, zorder=5)

    # --- 95% CI band ---
    ymax = ax.get_ylim()[1]
    ax.axvspan(ci_low, ci_high, color="#55A868", alpha=0.12, zorder=0)
    ax.axvline(ci_low, color="#55A868", linestyle=":", linewidth=1.5)
    ax.axvline(ci_high, color="#55A868", linestyle=":", linewidth=1.5)
    ax.annotate(
        "",
        xy=(ci_high, ymax * 0.92),
        xytext=(ci_low, ymax * 0.92),
        arrowprops=dict(arrowstyle="<->", color="#55A868", lw=1.5),
    )
    ax.text(
        (ci_low + ci_high) / 2,
        ymax * 0.95,
        "95% CI",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#2E6E43",
        fontweight="bold",
    )

    # --- stats box ---
    sig = p_value < alpha
    verdict_color = "#2E7D32" if sig else "#B71C1C"
    verdict = (
        f"significant (p < {alpha})" if sig else f"not significant (p \u2265 {alpha})"
    )
    stats_text = (
        f"mean diff = {mean_diff:+.4f}\n"
        f"95% CI = [{ci_low:+.4f}, {ci_high:+.4f}]\n"
        f"P(diff>0) = {p_gt:.3f}   P(diff<0) = {p_lt:.3f}\n"
        f"p-value = {p_value:.4g}\n"
        f"{verdict}"
    )
    ax.text(
        0.02,
        0.97,
        stats_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor=verdict_color,
            linewidth=1.5,
        ),
        color=verdict_color if False else "#1a1a1a",
    )

    ax.set_xlabel(f"Accuracy difference  ({name_a} \u2212 {name_b})", fontsize=10)
    ax.set_ylabel("Bootstrap replicate count", fontsize=10)
    ax.set_title(
        title or f"Paired bootstrap: {name_a} vs {name_b}",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(
        handles=[
            plt.Rectangle(
                (0, 0),
                1,
                1,
                fc="#4C72B0",
                alpha=0.75,
                label="replicates favoring the observed direction",
            ),
            plt.Rectangle(
                (0, 0),
                1,
                1,
                fc="#C44E52",
                alpha=0.85,
                hatch="//",
                label="replicates disagreeing (feeds p-value)",
            ),
        ],
        loc="upper right",
        fontsize=8,
        framealpha=0.9,
    )
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.margins(x=0.02)
    fig.tight_layout()
    # plot and close
    plt.show()
    plt.close()
    return


def plot_pvalue_heatmap(
    df,
    model_a_col="model_a",
    model_b_col="model_b",
    pvalue_col="p_adj",
    fill_diagonal=np.nan,
    cmap="viridis_r",
    annotate=True,
    figsize=(8, 6),
):
    """
    Plot a heatmap of adjusted p-values for pairwise model comparisons.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing pairwise comparisons.
    model_a_col : str
        Column containing the first model.
    model_b_col : str
        Column containing the second model.
    pvalue_col : str
        Column containing the p-values to plot (default: 'p_adj').
    fill_diagonal : float
        Value placed on the diagonal (default: np.nan).
    cmap : str
        Matplotlib colormap.
    annotate : bool
        Whether to annotate cells with p-values.
    figsize : tuple
        Figure size.
    """

    # Get all unique models
    models = sorted(set(df[model_a_col]).union(df[model_b_col]))

    # Initialize matrix
    mat = pd.DataFrame(
        fill_diagonal,
        index=models,
        columns=models,
        dtype=float,
    )

    # Fill matrix symmetrically
    for _, row in df.iterrows():
        a = row[model_a_col]
        b = row[model_b_col]
        p = row[pvalue_col]

        mat.loc[a, b] = p
        mat.loc[b, a] = p

    # Plot
    plt.figure(figsize=figsize)
    sns.heatmap(
        mat,
        cmap=cmap,
        annot=annotate,
        fmt=".3g",
        linewidths=0.5,
        square=True,
        cbar_kws={"label": "Adjusted p-value"},
        vmin=0,
        vmax=1,
    )

    plt.title("Pairwise Adjusted p-values")
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()
    return mat


def paired_bootstrap_test(scores_a, scores_b):
    """One pairwise comparison from paired bootstrap replicates."""
    diff = scores_a - scores_b
    mean_diff = diff.mean()
    ci_low, ci_high = np.percentile(diff, [2.5, 97.5])
    p_gt = np.mean(diff > 0)
    p_lt = np.mean(diff < 0)
    # fraction of bootstrap replicates that "disagree"
    # with the observed sign, doubled for a two-sided test.
    p_value = 2 * min(p_gt, p_lt)
    p_value = min(p_value, 1.0)
    results = {
        "diff": diff,  # kept for plotting; drop this key before exporting to a table/df
        "mean_diff": mean_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_gt": p_gt,
        "p_lt": p_lt,
        "p_value": p_value,
    }
    # plot_paired_bootstrap(results)
    return results


def compare_all_models(data: dict, correction="holm"):
    """
    data: {model_name: np.ndarray of shape (n_bootstrap,)}
    Returns a DataFrame with pairwise comparisons, multiple-comparison corrected.
    """
    models = list(data.keys())
    rows = []
    for a, b in combinations(models, 2):
        res = paired_bootstrap_test(data[a], data[b])
        rows.append({"model_a": a, "model_b": b, **res})

    df = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)

    # Holm-Bonferroni correction
    if correction == "holm":
        m = len(df)
        df["p_adj"] = [min(1.0, p * (m - i)) for i, p in enumerate(df["p_value"])]
        df["p_adj"] = df["p_adj"].cummax()  # enforce monotonicity
    elif correction == "bonferroni":
        df["p_adj"] = (df["p_value"] * len(df)).clip(upper=1.0)
    else:
        df["p_adj"] = df["p_value"]

    df["significant_0.05"] = df["p_adj"] < 0.05
    return df


def load_bootstrap_scores(root_dir, dataset, seed, task):
    """
    Load bootstrap*scores.npy files for all models.

    Returns
    -------
    dict
        {
            model_name: np.ndarray,
            ...
        }
    """

    data = {}

    root = Path(root_dir)

    pattern = f"{dataset}_{seed}/{task}/bootstrap*scores.npy"

    for model_dir in root.iterdir():
        if not model_dir.is_dir():
            continue

        matches = list(model_dir.glob(pattern))

        if len(matches) == 0:
            print(f"Skipping {model_dir.name}: no bootstrap scores found.")
            continue

        if len(matches) > 1:
            print(
                f"Warning: multiple bootstrap files found for {model_dir.name}, using {matches[0]}"
            )

        data[model_dir.name] = np.load(matches[0])

        print(
            f"Loaded {len(data[model_dir.name])} bootstrap scores for {model_dir.name}"
        )
        print(f"Mean score for {model_dir.name}: {np.mean(data[model_dir.name]):.4f}")
        print(
            f"95% CI for {model_dir.name}: [{np.percentile(data[model_dir.name], 2.5):.4f}, {np.percentile(data[model_dir.name], 97.5):.4f}]"
        )
        exit()
    return data


def summarize_scores(data, descending=False):
    """
    Compute mean, 5th percentile and 95th percentile for each model.
    """

    results = []

    for model, scores in data.items():
        results.append(
            {
                "model": model,
                "mean": np.mean(scores),
                "q05": np.quantile(scores, 0.05),  # for 90% CI
                "q95": np.quantile(scores, 0.95),
                # "q02.5": np.quantile(scores, 0.025),  # for 95% CI
                # "q97.5": np.quantile(scores, 0.975),
            }
        )

    results.sort(key=lambda x: x["mean"], reverse=descending)

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default=".",
        help="Root directory containing results folders of the models.",
    )

    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. adni_cn)")

    # add default seed 37
    parser.add_argument("--seed", default=37, help="Seed (e.g. 37)")

    parser.add_argument("--task", required=True, help="Task name (e.g. sex)")

    parser.add_argument(
        "--save", default=None, help="Path to save the t-test heatmap image."
    )

    parser.add_argument(
        "--descending",
        action="store_true",
        help="Sort by descending mean instead of ascending.",
    )

    args = parser.parse_args()
    save_path = Path(args.save) if args.save is not None else None

    data = load_bootstrap_scores(
        args.root,
        args.dataset,
        args.seed,
        args.task,
    )

    # ==========================
    # STATISTICAL TESTS
    # ==========================

    print("\n Stat tests:\n")
    comparison_df = compare_all_models(data, correction=None)
    print(comparison_df)
    # print proportion of significant comparisons
    n_significant = comparison_df["significant_0.05"].sum()
    n_total = len(comparison_df)
    print(
        f"\nSignificant tests: {n_significant}/{n_total} = {n_significant / n_total:.2%}"
    )
    print(comparison_df.columns)

    _ = plot_pvalue_heatmap(comparison_df, pvalue_col="p_adj", fill_diagonal=np.nan)

    # ==========================
    # MEAN METRICS
    # ==========================

    results = summarize_scores(data, descending=args.descending)
    print("\nSummary table:\n")
    summary_df = pd.DataFrame(results).sort_values(by="mean", ascending=args.descending)
    print(summary_df.to_string())

    # if save_path does not exist, create it
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(
            save_path / f"{args.dataset}_{args.seed}_{args.task}_summary.csv",
            index=False,
        )


if __name__ == "__main__":
    main()
