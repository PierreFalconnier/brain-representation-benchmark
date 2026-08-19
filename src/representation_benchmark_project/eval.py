import pickle
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder

from representation_benchmark_project.data.datamodule import BIDSDataModule
from representation_benchmark_project.experiment import (
    Experiment,
    build_pipeline,
    perform_boostrap,
    split_dataset,
)
from representation_benchmark_project.models.encoder import ImageEncoder
from representation_benchmark_project.utils import (
    RankedLogger,
    extras,
    get_features,
    pre_hydra_routine,
)

torch.set_float32_matmul_precision("high")

log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base=None, config_path="configs", config_name="task.yaml")
def main(cfg: DictConfig) -> float | None:
    pre_hydra_routine()
    extras(cfg)

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}> and setting it up")
    datamodule: BIDSDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    dataloader = datamodule.test_dataloader()
    image_key = list(cfg.data.modalities.keys())[0]
    target_key = cfg.target_key  # "age_at_visit", "research group"/"primdiag", "sex"

    # check if data)dict exist before instantiating the model, if not, compute features and save them in a pickle file
    features_folder_path = Path(cfg.output_dir).parent / "features"
    data_dict_path = features_folder_path / "data_dict.pkl"

    if data_dict_path.is_file() and not cfg.force_compute:
        log.info("Existing saved features found (data_dict.pkl file). Loading...")
        with open(data_dict_path, "rb") as f:
            data_dict = pickle.load(f)

    else:
        log.info(f"Instantiating backbone <{cfg.model._target_}>")
        model = hydra.utils.instantiate(cfg.model)

        log.info("Instantiating image encoder wrapper:")
        if cfg.get("device") == "cuda" and not torch.cuda.is_available():
            cfg.device = "cpu"
        encoder = ImageEncoder(
            backbone=model,
            adapt_method=cfg.adapt_method,
            seed=cfg.seed,
            device=cfg.device,
        )

        if compile_cfg := cfg.get("compile"):  # to test
            # try to compile the encoder, if error, catch and log it but continue without compiling
            try:
                compile_kwargs = (
                    compile_cfg if isinstance(compile_cfg, DictConfig) else {}
                )
                encoder = torch.compile(encoder, **compile_kwargs)
                log.info("Encoder compiled!")
            except Exception:
                log.warning("Failed to compile the encoder. Continue")

        # compute / retrieve features for all images in the dataloader
        data_dict = get_features(
            encoder=encoder,
            dataloader=dataloader,
            features_folder_path=features_folder_path,
            image_key=image_key,
            force_compute=cfg.force_compute,
        )

    # define task type and metrics
    if isinstance(data_dict[target_key][0], (int, float, np.number)):
        task_type = "regression"
        # filter out patient, keep controls "CN" or "Control")
        mask = np.isin(data_dict["diagnosis"], ["CN", "Control"])
        data_dict = {k: v[mask] for k, v in data_dict.items()}
        log.info(
            f"Filtered out patients, kept {len(data_dict[image_key])} controls for regression task"
        )
        values, counts = np.unique(data_dict["sex"], return_counts=True)
        log.info(f"Sex distribution: {dict(zip(values, counts))}")
    else:
        task_type = "classification"
        # print repartition of classes
        values, counts = np.unique(data_dict[target_key], return_counts=True)
        log.info(f"Class distribution for {target_key}: {dict(zip(values, counts))}")
        # encode target labels with sklearn's LabelEncoder
        le = LabelEncoder()
        data_dict[target_key] = le.fit_transform(data_dict[target_key])

    # split dataset and create flods
    data_split_dict = split_dataset(
        data_dict,
        image_key,
        target_key,
        task_type,
        test_size=0.8,
        n_splits=5,
        random_state=cfg.seed,
    )

    # pipeline, hyperparameter grid and metric for the task
    pipeline, param_grid, metric = build_pipeline(task_type, cfg.seed)

    # grid search with cross-validation

    experiment = Experiment(
        x=data_split_dict["X_train"],
        y=data_split_dict["y_train"],
        task_type=task_type,
        pipeline=pipeline,
        param_grid=param_grid,
        cv=data_split_dict["cv_splits"],
        seed=cfg.seed,
        verbose=10,
    )
    experiment.run(
        n_jobs=cfg.n_jobs, groups=data_dict.get("subject")
    )  # subject-aware CV if subject is available, else regular CV

    experiment.save(cfg.output_dir)

    # boostrap on test set with best model selected on val set

    perform_boostrap(
        model=experiment.best_model,
        x=data_split_dict["X_test"],
        y=data_split_dict["y_test"],
        groups=data_split_dict["groups_test"],
        n_splits=cfg.n_splits,
        method=".632",
        scoring_func=metric,
        random_seed=cfg.seed,
        result_folder_path=cfg.output_dir,
        n_jobs=cfg.n_jobs,
    )
    print("Done!")


if __name__ == "__main__":
    main()
