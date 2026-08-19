import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rootutils
import torch
from omegaconf import DictConfig
from sklearn.utils import resample
from tqdm import tqdm

from representation_benchmark_project.configs import register_omegaconf_resolvers
from representation_benchmark_project.utils import pylogger, rich_utils

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


# ------- HYDRA UTILS -------
def pre_hydra_routine() -> None:
    """Configure environment and variables that must be set before running Hydra."""
    rootutils.setup_root(__file__, indicator="pyproject.toml")
    # the setup_root above is equivalent to:
    # - setting up PROJECT_ROOT environment variable
    #       (which is used as a base for paths in "configs/paths/default.yaml")
    #       (this way all filepaths are the same no matter where you run the code)
    # - loading environment variables from ".env" in root dir
    #
    # you can remove it if you:
    # 1. set `root_dir` to "." in "configs/paths/default.yaml"
    #
    # more info: https://github.com/ashleve/rootutils

    # Register custom OmegaConf resolvers
    register_omegaconf_resolvers()


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    Args:
        cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


# ------- /HYDRA UTILS -------


# ------- CACHE KEY UTILS ----


# @torch.no_grad()
@torch.inference_mode()
def get_features(
    encoder, dataloader, features_folder_path, image_key="x0", force_compute=False
):
    if features_folder_path is None:
        raise ValueError("features_folder_path is None")

    features_folder_path = Path(features_folder_path)
    data_dict_path = features_folder_path / "data_dict.pkl"

    if data_dict_path.is_file() and not force_compute:
        # loading existing dict
        print("Existing saved features found (data_dict.pkl file). Loading...")
        with open(data_dict_path, "rb") as f:
            data_dict = pickle.load(f)

    else:
        print(
            f"No saved features found (no data_dict.pkl file). Computing and saving them to {features_folder_path}"
        )

        data_dict = {}

        device = next(encoder.parameters()).device
        encoder.eval()
        # computing features
        for batch in tqdm(dataloader, desc="Computing features"):
            for k, v in batch.items():
                if k not in data_dict:
                    data_dict[k] = []  # init list if necessary

                if k == image_key:
                    x = v.to(device)
                    features = encoder(x)
                    data_dict[k].append(features.detach().cpu().numpy())
                else:
                    if isinstance(v, torch.Tensor):
                        data_dict[k].append(v.detach().cpu().numpy())
                    else:
                        data_dict[k].append(np.array(v))

        for k in data_dict:
            data_dict[k] = np.concatenate(data_dict[k], axis=0)

        # saving
        features_folder_path.mkdir(parents=True, exist_ok=True)
        with open(data_dict_path, "wb") as f:
            pickle.dump(data_dict, f)

    return data_dict


def bootstrap_metric(
    y_val_pred,
    y_val,
    n_bootstrap,
    metric_func,
    seed=None,
    figure_path=None,
    show_plot=False,
):
    bootstrap_metrics = []

    for _ in range(n_bootstrap):
        y_val_bootstrap, y_val_pred_bootstrap = resample(
            y_val, y_val_pred, replace=True, random_state=seed
        )

        metric_value = metric_func(y_val_bootstrap, y_val_pred_bootstrap)
        bootstrap_metrics.append(metric_value)

    mean_metric = np.mean(bootstrap_metrics)
    std_metric = np.std(bootstrap_metrics, ddof=1)  # unbiased std estimator

    # distribution of metrics

    ci_lower = np.percentile(bootstrap_metrics, 2.5)
    ci_upper = np.percentile(bootstrap_metrics, 97.5)

    if figure_path is not None:
        plt.figure(figsize=(8, 5))
        plt.hist(
            bootstrap_metrics, bins=40, color="skyblue", alpha=0.7, edgecolor="black"
        )
        plt.axvline(
            mean_metric, color="red", linestyle="--", label=f"Mean = {mean_metric:.4f}"
        )
        plt.axvline(
            ci_lower,
            color="green",
            linestyle="--",
            label=f"95% CI Lower = {ci_lower:.4f}",
        )
        plt.axvline(
            ci_upper,
            color="green",
            linestyle="--",
            label=f"95% CI Upper = {ci_upper:.4f}",
        )
        plt.title("Bootstrap metric distribution - 95% confidence interval")
        plt.xlabel("Metric Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.savefig(figure_path, format="svg")

        if show_plot:
            plt.show()
        plt.close()

    return mean_metric, std_metric, ci_lower, ci_upper
