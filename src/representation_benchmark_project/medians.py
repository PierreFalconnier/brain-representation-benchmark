#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd


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
                "q05": np.quantile(scores, 0.05),
                "q95": np.quantile(scores, 0.95),
            }
        )

    results.sort(key=lambda x: x["mean"], reverse=descending)

    return results


def main():
    root = "/home/falconnier/Documents/pfe/mri-representation-learning/results"
    datasets = ["adni", "ppmi"]
    seed = 37
    tasks = ["sex", "age_at_visit", "diagnosis"]

    all_ranks = []

    for dataset in datasets:
        for task in tasks:
            # if dataset == "ppmi" and task == "diagnosis":
            #     continue  # Skip ppmi diagnosis task
            print(f"Processing {dataset} - {task}")

            data = load_bootstrap_scores(root, dataset, seed, task)

            rows = []
            for model, scores in data.items():
                rows.append(
                    {
                        "model": model,
                        "mean": np.mean(scores),
                    }
                )

            df = pd.DataFrame(rows)

            # Higher is better except for age prediction
            ascending = task == "age_at_visit"

            df = df.sort_values("mean", ascending=ascending).reset_index(drop=True)
            df["rank"] = np.arange(1, len(df) + 1)
            df["dataset"] = dataset
            df["task"] = task

            print(df[["model", "mean", "rank"]].to_string(index=False))

            all_ranks.append(df[["model", "dataset", "task", "rank"]])

    all_ranks = pd.concat(all_ranks, ignore_index=True)

    print("\n==============================")
    print("Ranks across all datasets/tasks")
    print("==============================\n")
    print(all_ranks.to_string(index=False))

    median_ranks = (
        all_ranks.groupby("model")["rank"]
        .median()
        .sort_values()
        .reset_index(name="median_rank")
    )

    median_ranks["mean_rank"] = (
        all_ranks.groupby("model")["rank"].mean().reindex(median_ranks["model"]).values
    )

    print("\n==============================")
    print("Median rank across all datasets/tasks")
    print("==============================\n")
    print(median_ranks.to_string(index=False))


if __name__ == "__main__":
    main()
