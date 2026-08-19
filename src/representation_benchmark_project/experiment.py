import hashlib
import json
import os
from itertools import product

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from mlxtend.evaluate import BootstrapOutOfBag
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    make_scorer,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    ShuffleSplit,
    StratifiedGroupKFold,
    StratifiedShuffleSplit,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib


def split_dataset(
    data_dict,
    image_key,
    target_key,
    task_type,
    test_size=0.2,
    n_splits=5,
    random_state=42,
):
    X = data_dict[image_key]
    y = data_dict[target_key]
    subjects = data_dict["subject"]
    unique_subjects = np.unique(subjects)

    # --- Train/Test split at subject level ---
    if task_type == "classification":
        subject_labels = np.array([y[subjects == s][0] for s in unique_subjects])
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        train_subj_idx, test_subj_idx = next(
            splitter.split(unique_subjects, subject_labels)
        )
    else:
        splitter = ShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        train_subj_idx, test_subj_idx = next(splitter.split(unique_subjects))

    train_mask = np.isin(subjects, unique_subjects[train_subj_idx])
    test_mask = np.isin(subjects, unique_subjects[test_subj_idx])

    X_train, y_train, groups_train = X[train_mask], y[train_mask], subjects[train_mask]
    X_test, y_test, groups_test = X[test_mask], y[test_mask], subjects[test_mask]

    # --- Cross-validation on train split ---
    if task_type == "classification":
        cv = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
    else:
        cv = GroupKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )  # ← updated

    cv_splits = list(cv.split(X_train, y_train, groups=groups_train))

    return {
        "X_train": X_train.astype(np.float32),
        "y_train": y_train,
        "groups_train": groups_train,
        "X_test": X_test.astype(np.float32),
        "y_test": y_test,
        "groups_test": groups_test,
        "cv_splits": cv_splits,
    }


def to_float32_if_overflow(x):
    if np.isinf(np.std(x)) or np.isinf(np.mean(x)):
        print("Overflow detected. Converting features to float32.")
        x = x.astype(np.float32)
        print("Mean:", np.mean(x))
        print("Std:", np.std(x))
    return x


def convert_weights_to_str(data):
    if "classifier__weights" in data:
        value = data["classifier__weights"]
        if not isinstance(value, (str, int, float)):
            # Replace with the class name as a string
            data["classifier__weights"] = (
                f"{value.__class__.__module__}.{value.__class__.__name__} - temp={value.temperature}"
            )
    return data


# --- bootstrap code ---


def _check_arrays(X, y=None):
    if isinstance(X, list):
        raise ValueError("X must be a numpy array")
    if not len(X.shape) == 2:
        raise ValueError("X must be a 2D array. Try X[:, numpy.newaxis]")
    try:
        if y is None:
            return
    except AttributeError:
        if not len(y.shape) == 1:
            raise ValueError("y must be a 1D array.")

    if not len(y) == X.shape[0]:
        raise ValueError("X and y must contain thesame number of samples")

    # if contain inf or nan, raise error
    if np.isinf(X).any() or np.isnan(X).any():
        raise ValueError(
            "X contains inf or nan values. Please handle them before bootstrapping."
        )


def no_information_rate(targets, predictions, loss_fn):
    combinations = np.array(list(product(targets, predictions)))
    return loss_fn(combinations[:, 0], combinations[:, 1])


def accuracy(targets, predictions):
    return np.mean(np.array(targets) == np.array(predictions))


def mse(targets, predictions):
    return np.mean((np.array(targets) - np.array(predictions)) ** 2)


# modified from mlxtend's implementation to handle groups to avoid
# data leakage when multiple samples belong to the same subject (group)
class GroupBootstrapOutOfBag:
    """
    Bootstrap Out-of-Bag splitter that prevents data leakage when multiple
    samples belong to the same subject (group).

    Bootstrapping is performed at the group level: all samples from a group
    are either entirely in the training set or entirely in the OOB test set.

    Parameters
    ----------
    n_splits : int (default=200)
    random_seed : int or None (default=None)
    """

    def __init__(self, n_splits=200, random_seed=None):
        if not isinstance(n_splits, int) or n_splits < 1:
            raise ValueError("Number of splits must be greater than 1.")
        self.n_splits = n_splits
        self.random_seed = random_seed

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("groups must be provided.")

        groups = np.asarray(groups)
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)

        rng = np.random.RandomState(self.random_seed)

        for _ in range(self.n_splits):
            # bootstrap at group level
            sampled_groups = rng.choice(unique_groups, size=n_groups, replace=True)
            oob_groups = set(unique_groups) - set(sampled_groups)

            # expand to sample indices
            train_idx = np.where(np.isin(groups, sampled_groups))[0]
            test_idx = np.where(np.isin(groups, list(oob_groups)))[0]

            if len(test_idx) == 0:
                continue  # unlucky draw, all groups in train — skip

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def bootstrap_point632_score(
    estimator,
    X,
    y,
    groups=None,
    n_splits=200,
    method=".632",
    scoring_func=None,
    predict_proba=False,
    random_seed=None,
    clone_estimator=True,
    n_jobs=-1,
    **fit_params,
):
    if not isinstance(n_splits, int) or n_splits < 1:
        raise ValueError("Number of splits must be greater than 1. Got %s." % n_splits)

    allowed_methods = (".632", ".632+", "oob")
    if not isinstance(method, str) or method not in allowed_methods:
        raise ValueError(
            "The `method` must be in %s. Got %s." % (allowed_methods, method)
        )

    # Pandas compatibility
    if hasattr(X, "values"):
        X = X.values
    if hasattr(y, "values"):
        y = y.values

    _check_arrays(X, y)

    X = np.where(np.isfinite(X), X, np.nan)
    X = X.astype(np.float32)

    if scoring_func is None:
        if estimator._estimator_type == "classifier":
            scoring_func = accuracy
        elif estimator._estimator_type == "regressor":
            scoring_func = mse
        else:
            raise AttributeError(
                "Estimator type undefined. Please provide a scoring_func argument."
            )

    def bootstrap_iteration(split_id, train_idx, test_idx):
        # Clone the estimator for parallelization safety
        est = clone(estimator) if clone_estimator else estimator

        est.fit(X[train_idx], y[train_idx], **fit_params)

        predict_func = est.predict_proba if predict_proba else est.predict
        predicted_test_val = predict_func(X[test_idx])

        if method in (".632", ".632+"):
            predicted_train_val = predict_func(X)

        if predict_proba:
            len_uniq = np.unique(y)
            if len(len_uniq) == 2:
                predicted_train_val = predicted_train_val[:, 1]
                predicted_test_val = predicted_test_val[:, 1]

        test_acc = scoring_func(y[test_idx], predicted_test_val)

        if method == "oob":
            acc = test_acc
        else:
            test_err = 1 - test_acc
            train_err = 1 - scoring_func(y, predicted_train_val)

            if method == ".632+":
                gamma = 1 - no_information_rate(y, est.predict(X), scoring_func)
                R = (test_err - train_err) / (gamma - train_err)
                weight = 0.632 / (1 - 0.368 * R)
            else:
                weight = 0.632

            acc = 1 - (weight * test_err + (1.0 - weight) * train_err)

        return split_id, acc

    # generate bootstrap splits
    if groups is not None:
        oob = GroupBootstrapOutOfBag(n_splits=n_splits, random_seed=random_seed)
        splits = list(oob.split(X, groups=groups))
        print(f"Generated {len(splits)} bootstrap splits with groups.")
    else:
        oob = BootstrapOutOfBag(n_splits=n_splits, random_seed=random_seed)
        splits = list(oob.split(X))
        print(f"Generated {len(splits)} bootstrap splits without groups.")

    # --- reproducibility checks ---
    print(f"random_seed used: {random_seed}")
    print(f"Splits hash: {hash_splits(splits)}")
    print(f"First split: train[:10]={splits[0][0][:10]}, test[:10]={splits[0][1][:10]}")
    print(
        f"Last split:  train[:10]={splits[-1][0][:10]}, test[:10]={splits[-1][1][:10]}"
    )
    print(f"Sum of all train indices: {sum(t.sum() for t, _ in splits)}")
    print(f"Sum of all test indices:  {sum(t.sum() for _, t in splits)}")

    # # run bootstrap iterations in parallel
    # with tqdm_joblib(tqdm(desc="Bootstrapping", total=len(splits))):
    #     scores = Parallel(n_jobs=n_jobs)(
    #         delayed(bootstrap_iteration)(train, test) for train, test in splits
    #     )
    # return np.array(scores)

    # run with explicit ids
    with tqdm_joblib(tqdm(desc="Bootstrapping", total=len(splits))):
        results = Parallel(n_jobs=n_jobs)(
            delayed(bootstrap_iteration)(i, train, test)
            for i, (train, test) in enumerate(splits)
        )

    # --- verification ---
    returned_ids = [r[0] for r in results]
    expected_ids = list(range(len(splits)))
    assert returned_ids == expected_ids, (
        "Returned order does not match submission order"
    )
    if returned_ids != expected_ids:
        print(
            f"First mismatch at: {[i for i, (a, b) in enumerate(zip(returned_ids, expected_ids)) if a != b][:5]}"
        )

    # --- fix: sort by split_id regardless, so you're safe even if order ever changes ---
    results_sorted = sorted(results, key=lambda r: r[0])
    scores = np.array([r[1] for r in results_sorted])
    return scores


def hash_splits(splits):
    """Create a deterministic hash of all train/test indices in the splits."""
    hasher = hashlib.sha256()
    for train_idx, test_idx in splits:
        hasher.update(np.asarray(train_idx).tobytes())
        hasher.update(np.asarray(test_idx).tobytes())
    return hasher.hexdigest()


# --- \bootstrap code ---
class Experiment(object):
    def __init__(
        self,
        x,
        y,
        task_type,
        pipeline,
        param_grid,
        cv,
        seed=None,
        verbose=0,
    ):
        self.x = x
        self.y = y
        self.task_type = task_type
        self.pipeline = pipeline
        self.param_grid = param_grid
        self.cv = cv
        self.seed = seed
        self.verbose = verbose

        # scoring functions and refit metric for grid search
        if task_type == "classification":
            self.refit = "balanced_accuracy"
            self.scoring = {
                "f1": make_scorer(f1_score, average="binary"),
                "balanced_accuracy": make_scorer(balanced_accuracy_score),
            }
        elif task_type == "regression":
            self.refit = "mean_absolute_error"
            self.scoring = {
                "mean_absolute_error": make_scorer(
                    mean_absolute_error, greater_is_better=False
                ),  #  greater_is_better is False so the grid search will find the lowest mae
                "r2_score": make_scorer(r2_score),
            }
        else:
            raise NotImplementedError("Given task type not implemented")

        self.grid_search = None
        self.best_model = None

    def run(self, n_jobs=-1, groups=None):
        self.grid_search = GridSearchCV(
            estimator=self.pipeline,
            param_grid=self.param_grid,
            cv=self.cv,
            scoring=self.scoring,
            refit=self.refit,
            n_jobs=n_jobs,
            verbose=self.verbose,
        )

        self.grid_search.fit(self.x, self.y)
        self.best_model = self.grid_search.best_estimator_

    def save(self, result_folder_path):
        os.makedirs(result_folder_path, exist_ok=True)

        # grid search results to CSV file
        results_file = os.path.join(result_folder_path, "grid_search_results.csv")
        results_df = pd.DataFrame(self.grid_search.cv_results_)
        results_df.to_csv(results_file, index=False)

        # Save the full GridSearchCV object
        grid_search_file = os.path.join(result_folder_path, "grid_search_results.pkl")
        joblib.dump(self.grid_search.cv_results_, grid_search_file)

        # best mean scores and associated stds
        best_index = int(self.grid_search.best_index_)

        # just modify the value for 'classifier__weights' if exist
        best_params = convert_weights_to_str(self.grid_search.best_params_)

        best_scores = {
            "best_index": best_index,
            "best_params": best_params,
            "best_score": self.grid_search.best_score_,  # metric used for refit
        }

        for key in self.scoring.keys():
            best_scores[f"best_{key}_score_std"] = self._get_best_score_std(key)
            best_scores[f"best_{key}_score"] = self._get_best_score(key)

        best_scores_file = os.path.join(result_folder_path, "best_scores.json")
        with open(best_scores_file, "w") as f:
            json.dump(best_scores, f, indent=4)

        # balanced accuracies of best model that will be used for
        # 5x2cv f test
        N = 5
        acc = [
            self.grid_search.cv_results_[f"split{i}_test_{self.refit}"][best_index]
            for i in range(N)
        ]
        np.save(
            os.path.join(result_folder_path, "scores_best_model.npy"), np.array(acc)
        )

        # best model
        model_file = os.path.join(result_folder_path, "best_model.pkl")
        joblib.dump(self.best_model, model_file)

    def _get_best_score(self, metric):
        return max(self.grid_search.cv_results_[f"mean_test_{metric}"])

    def _get_best_score_std(self, metric):
        best_index = np.argmax(self.grid_search.cv_results_[f"mean_test_{metric}"])
        return self.grid_search.cv_results_[f"std_test_{metric}"][best_index]


def build_pipeline(task_type, seed):
    if task_type == "classification":
        model = LogisticRegression(
            class_weight="balanced",
            random_state=seed,
            # solver="saga",
            solver="lbfgs",
            max_iter=5000,
        )
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )

        param_grid = {
            "model__C": [np.inf] + [10**i for i in range(-6, 5)],
            # "model__l1_ratio": [0, 0.5, 1.0],
        }

        return (
            pipeline,
            param_grid,
            balanced_accuracy_score,
        )

    else:
        model = Ridge(solver="cholesky", random_state=seed)
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )
        param_grid = {"model__alpha": [10**i for i in range(-6, 5)] + [0]}

        return (pipeline, param_grid, mean_absolute_error)


def perform_boostrap(
    model,
    x,
    y,
    groups=None,
    n_splits=1000,
    method=".632",
    scoring_func=None,
    random_seed=None,
    result_folder_path=None,
    show_plot=False,
    n_jobs=-1,
):
    if scoring_func is None:
        raise ValueError("A scoring function must be provided for the boostrap")

    print(f"Performing bootstrapping ({scoring_func.__name__})")
    scores = bootstrap_point632_score(
        estimator=model,
        X=x,
        y=y,
        groups=groups,
        n_splits=n_splits,
        method=method,
        scoring_func=scoring_func,
        random_seed=random_seed,
        n_jobs=-1,
    )
    print(f"Bootstrap completed. Mean {scoring_func.__name__}: {np.mean(scores):.4f}")

    # save the score in npy file
    if result_folder_path is not None:
        np.save(
            os.path.join(
                result_folder_path, f"bootstrap_{scoring_func.__name__}_scores.npy"
            ),
            scores,
        )

    mean_metric = np.mean(scores)
    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)

    if result_folder_path is not None:
        results = {
            "metric": scoring_func.__name__,
            "mean": mean_metric,
            "confidence_interval": {"lower": lower, "upper": upper},
        }

        json_file_path = os.path.join(
            result_folder_path, f"bootstrap_{scoring_func.__name__}_results.json"
        )
        with open(json_file_path, "w") as json_file:
            json.dump(results, json_file, indent=4)

        plt.figure(figsize=(8, 5))
        plt.hist(scores, bins=60, color="skyblue", alpha=0.7, edgecolor="black")
        plt.axvline(
            mean_metric,
            color="red",
            linestyle="--",
            label=f"Mean = {mean_metric:.4f}",
        )
        plt.axvline(
            lower,
            color="green",
            linestyle="--",
            label=f"95% CI Lower = {lower:.4f}",
        )
        plt.axvline(
            upper,
            color="green",
            linestyle="--",
            label=f"95% CI Upper = {upper:.4f}",
        )

        plt.title(
            f"Bootstrap {scoring_func.__name__} Distribution - 95% Confidence Interval"
        )
        plt.xlabel(f"{scoring_func.__name__} Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        if show_plot:
            plt.show()

        plt.savefig(
            os.path.join(result_folder_path, f"boostrap_{scoring_func.__name__}.svg"),
            format="svg",
        )
        plt.close()

        print("Bootstrap done. Results saved to:", result_folder_path)
