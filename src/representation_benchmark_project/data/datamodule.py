from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Union

import lightning as L
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from representation_benchmark_project.data.dataset import BIDSDataset


class TransformSubset(torch.utils.data.Dataset):
    """
    lightweight wrapper to create a subset with the indices
    and a transform sepcific to that subset
    """

    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample = self.dataset[self.indices[idx]]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class BIDSDataModule(L.LightningDataModule):
    def __init__(
        self,
        root_dir: Union[str, Path],
        modalities: dict,
        csv_path: Optional[Union[str, Path]] = None,
        metadata_columns: Optional[List[str]] = None,
        stratify_columns: Optional[List[str]] = None,
        sub_prefix: str = "sub-",
        session_prefix: str = "ses-",
        train_transform=None,
        val_transform=None,
        test_transform=None,
        train_split=0.8,
        val_split=0.2,
        batch_size=2,
        num_workers=4,
        pin_memory=True,
        shuffle_train=True,
        shuffle_val=False,
        shuffle_test=False,
        drop_last_train=False,
        drop_last_val=False,
        drop_last_test=False,
        random_state=None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.num_workers = (
            max(0, int(torch.get_num_threads() - 1))
            if num_workers is None
            else num_workers
        )
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.batch_size_per_device = batch_size
        self.random_state = random_state
        self.stratify_columns = stratify_columns or []
        self.stratify_columns = [col.lower() for col in self.stratify_columns]
        self.train_split = train_split
        self.val_split = val_split
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.sub_prefix = sub_prefix
        self.session_prefix = session_prefix
        self.modalities = modalities

        self.test_split = 1 - (self.train_split + self.val_split)
        total = round(self.train_split + self.val_split + self.test_split, 5)
        if not (0 <= total <= 1.0):
            raise ValueError(
                "train_split + val_split + test_split must be in the range [0, 1.0]"
            )

    def _random_split(self, n: int) -> tuple[list, list, list]:
        """Return (train, val, test) index lists from a random permutation of range(n)."""
        if self.random_state is not None:
            generator = torch.Generator().manual_seed(self.random_state)
            perm = torch.randperm(n, generator=generator).tolist()
        else:
            perm = torch.randperm(n).tolist()

        n_train = round(n * self.train_split)
        n_val = round(n * self.val_split)
        n_test = n - n_train - n_val

        # print(
        #     f"Random split: {n_train} train / {n_val} val / {n_test} test samples (total: {n})"
        # )
        # exit()

        return (
            perm[:n_train],
            perm[n_train : n_train + n_val],
            perm[n_train + n_val :] if n_test > 0 else [],
        )

    def setup(self, stage=None):
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of "
                    f"devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )

        if not self.train_dataset and not self.val_dataset and not self.test_dataset:
            self.full_dataset = BIDSDataset(
                root_dir=self.hparams.root_dir,
                modalities=self.hparams.modalities,
                csv_path=self.hparams.csv_path,
                metadata_columns=self.hparams.metadata_columns,
                sub_prefix=self.hparams.sub_prefix,
                session_prefix=self.hparams.session_prefix,
            )

            if self.full_dataset.df_metadata is None and (
                self.hparams.metadata_columns or self.stratify_columns
            ):
                raise RuntimeError(
                    "Metadata columns were specified but metadata could not be loaded. Check the CSV path and format / access right."
                )

            stratify_df = None

            if self.full_dataset.dataset_is_flat:
                # No subject structure — split sample indices directly, no stratification possible.
                n_total = len(self.full_dataset)
                train_indices, val_indices, test_indices = self._random_split(n_total)
                train_subjects = val_subjects = test_subjects = set()

            else:
                # Structured layout — split at subject level, then map back to sample indices.
                subjects = list({s["subject"] for s in self.full_dataset.samples})
                n_total = len(subjects)
                test_split = 1.0 - (self.train_split + self.val_split)

                train_subjects = val_subjects = test_subjects = set()

                if self.stratify_columns:
                    # Build a per-subject stratification key
                    subject_metadata = defaultdict(dict)
                    for sample in self.full_dataset.samples:
                        for key in self.stratify_columns:
                            if key in sample:
                                subject_metadata[sample["subject"]][key] = sample[key]

                    stratify_df = pd.DataFrame(
                        [
                            {"subject": subj, **meta}
                            for subj, meta in subject_metadata.items()
                        ]
                    )
                    stratify_col = (
                        "stratify_key"
                        if len(self.stratify_columns) > 1
                        else self.stratify_columns[0]
                    )
                    if len(self.stratify_columns) > 1:
                        stratify_df["stratify_key"] = (
                            stratify_df[self.stratify_columns]
                            .astype(str)
                            .agg("_".join, axis=1)
                        )

                    # Two-step stratified split
                    if test_split == 1:
                        test_subjects = set(subjects)
                        train_val_subjects = set()
                    elif test_split > 0:
                        train_val_subjects, test_subjects_list = train_test_split(
                            subjects,
                            test_size=test_split,
                            stratify=stratify_df[stratify_col],
                            random_state=self.random_state,
                        )
                        test_subjects = set(test_subjects_list)
                    else:
                        train_val_subjects = subjects

                    if self.train_split == 0 and self.val_split > 0:
                        train_subjects = set()
                        val_subjects = set(train_val_subjects)
                    elif self.val_split == 0 and self.train_split > 0:
                        train_subjects = set(train_val_subjects)
                        val_subjects = set()
                    elif self.train_split == 0 and self.val_split == 0:
                        train_subjects = set()
                        val_subjects = set()
                    else:
                        train_subjects_list, val_subjects_list = train_test_split(
                            train_val_subjects,
                            test_size=self.val_split
                            / (self.train_split + self.val_split),
                            stratify=stratify_df.loc[
                                stratify_df["subject"].isin(train_val_subjects),
                                stratify_col,
                            ],
                            random_state=self.random_state,
                        )
                        train_subjects = set(train_subjects_list)
                        val_subjects = set(val_subjects_list)

                else:
                    # No stratification — reuse the same helper, but on subjects
                    train_pos, val_pos, test_pos = self._random_split(n_total)
                    train_subjects = {subjects[i] for i in train_pos}
                    val_subjects = {subjects[i] for i in val_pos}
                    test_subjects = {subjects[i] for i in test_pos}

                # Map subject sets back to sample indices
                train_indices = [
                    i
                    for i, s in enumerate(self.full_dataset.samples)
                    if s["subject"] in train_subjects
                ]
                val_indices = [
                    i
                    for i, s in enumerate(self.full_dataset.samples)
                    if s["subject"] in val_subjects
                ]
                test_indices = [
                    i
                    for i, s in enumerate(self.full_dataset.samples)
                    if s["subject"] in test_subjects
                ]

            self.train_dataset = TransformSubset(
                self.full_dataset, train_indices, transform=self.train_transform
            )
            self.val_dataset = TransformSubset(
                self.full_dataset, val_indices, transform=self.val_transform
            )
            self.test_dataset = TransformSubset(
                self.full_dataset, test_indices, transform=self.test_transform
            )

            # Logging
            print("\n--- Data split ---")
            if self.full_dataset.dataset_is_flat:
                print(f"Layout: flat  |  total samples: {n_total}")
            else:
                print(f"Layout: structured  |  total subjects: {n_total}")
                print(
                    f"  Train / val / test subjects: {len(train_subjects)} / {len(val_subjects)} / {len(test_subjects)}"
                )
            print(
                f"Train / val / test samples: {len(train_indices)} / {len(val_indices)} / {len(test_indices)}"
            )

            # Log stratification distribution if applicable
            if stratify_df is not None and stratify_col is not None:
                print("\n--- Stratification ---")
                for split_name, split_subjects in [
                    ("Train", train_subjects),
                    ("Val", val_subjects),
                    ("Test", test_subjects),
                ]:
                    if split_subjects:
                        split_stratify = stratify_df[
                            stratify_df["subject"].isin(split_subjects)
                        ]
                        print(f"{split_name} stratification distribution:")
                        print(split_stratify[stratify_col].value_counts(normalize=True))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size_per_device,
            shuffle=self.hparams.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.num_workers > 0,
            drop_last=self.hparams.drop_last_train,  # drop last batch if it's smaller than batch size
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size_per_device,
            shuffle=self.hparams.shuffle_val,
            num_workers=self.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.num_workers > 0,
            drop_last=self.hparams.drop_last_val,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size_per_device,
            shuffle=self.hparams.shuffle_test,
            num_workers=self.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.num_workers > 0,
            drop_last=self.hparams.drop_last_test,
        )


if __name__ == "__main__":
    from monai.transforms import Compose, LoadImaged, ScaleIntensityRangePercentilesd
    from tqdm import tqdm

    transforms = None

    # MONAI transforms
    keys = ["x0"]
    transforms = Compose(
        [
            LoadImaged(
                keys=keys,
                ensure_channel_first=True,
                # reader="numpyreader",
                # mmap_mode="r",
            ),
            ScaleIntensityRangePercentilesd(
                keys=keys, lower=0.1, upper=99.9, b_min=0.0, b_max=1.0, clip=True
            ),
            # ResizeWithPadOrCropd(keys=keys, spatial_size=[160, 160]),
            # ResizeWithPadOrCropd(keys=keys, spatial_size=[160, 192, 160]),
        ]
    )

    # DataModule
    dm = BIDSDataModule(
        # root_dir="/run/media/falconnier/bb9ecfb7-b58f-41e9-a37d-fda12951eb4e/PPMI_anat_dwi_BIDS",
        # root_dir="/run/user/11501/gvfs/sftp:host=linux1.dg.creatis.insa-lyon.fr,user=falconnier/misc/raid/falconnier/Documents/data/PPMI_T1w_bids_processed_reorganized_curated",
        # root_dir="/run/user/11501/gvfs/sftp:host=linux1.dg.creatis.insa-lyon.fr,user=falconnier/misc/raid/falconnier/Documents/data/2D_PPMI_T1w_bids_processed_reorganized_curated",
        # root_dir="/home/falconnier/Documents/data/2D_PPMI_T1w_bids_processed_reorganized_curated",
        # root_dir="/run/media/falconnier/bb9ecfb7-b58f-41e9-a37d-fda12951eb4e/IXI_bids",
        # root_dir="/home/falconnier/Documents/data/ADNI_T1w_curated",
        # root_dir="/home/falconnier/Documents/data/PPMI_T1w_curated",
        # root_dir="/run/media/falconnier/bb9ecfb7-b58f-41e9-a37d-fda12951eb4e/PPMI_T1w_curated",
        root_dir="/run/media/falconnier/bb9ecfb7-b58f-41e9-a37d-fda12951eb4e/ADNI_T1w_curated",
        # root_dir="/run/media/falconnier/bb9ecfb7-b58f-41e9-a37d-fda12951eb4e/debug_dataset",
        modalities={"x0": "*T1w_middle_coronal*.npy"},
        # modalities={"x0": "*T1w_normalized.nii.gz"},
        # csv_path="/run/media/falconnier/bb9ecfb7-b58f-41e9-a37d-fda12951eb4e/PPMI_T1w_curated/ppmi_clinical_imaging_merged_beforefiltering_20260529.csv",
        csv_path="/run/media/falconnier/bb9ecfb7-b58f-41e9-a37d-fda12951eb4e/ADNI_T1w_curated/ADNI_T1w_idaSearch_5_29_2026_upgraded_CN_noageduplicates.csv",
        # csv_path="/run/user/11501/gvfs/sftp:host=linux1.dg.creatis.insa-lyon.fr,user=falconnier/misc/raid/falconnier/Documents/data/ADNI_T1w_curated/ADNI_T1w_idaSearch_5_29_2026_CN_AD.csv",
        # csv_path="/run/media/falconnier/bb9ecfb7-b58f-41e9-a37d-fda12951eb4e/ADNI_T1w_curated/ADNI_T1w_idaSearch_5_29_2026_upgraded_CN_noageduplicates.csv",
        # csv_path="/home/falconnier/Documents/data/ADNI_T1w_curated/ADNI_T1w_idaSearch_5_29_2026_upgraded_CN_noageduplicates.csv",
        # csv_path="/home/falconnier/Documents/data/PPMI_T1w_curated/ppmi_clinical_imaging_merged_beforefiltering_20260529.csv",
        # csv_path="/home/falconnier/Documents/mri-preprocessing/csv_exploration/PPMI_explo/ppmi_clinical_imaging_merged_beforefiltering_20260529.csv",
        # csv_path="/home/falconnier/Documents/mri-preprocessing/csv_exploration/PPMI_explo/ppmi_clinical_imaging_merged_beforefiltering_20260528.csv",
        # csv_path="/home/falconnier/Documents/mri-preprocessing/csv_exploration/PPMI_explo/ppmi_clinical_imaging_merged_20260330.csv",
        # csv_path="/run/user/11501/gvfs/sftp:host=linux1.dg.creatis.insa-lyon.fr,user=falconnier/misc/raid/falconnier/Documents/data/2D_PPMI_T1w_bids_processed_reorganized_curated/ppmi_full_data.csv",
        # csv_path="/home/falconnier/Documents/mri-preprocessing/csv_exploration/PPMI_explo/ppmi_clinical_imaging_merged_20260415_181552.csv",
        # metadata_columns=["age_at_visit", "research group", "sex"],  # adni
        # stratify_columns=["research group", "sex"],  # adni
        # metadata_columns=["age_at_visit"],  # ppmi
        train_transform=transforms,
        val_transform=transforms,
        test_transform=transforms,
        train_split=0.8,
        val_split=0.2,
        batch_size=32,
        num_workers=8,
        # sub_prefix="",
        # session_prefix="",
        # sub_prefix="sub-",
        # session_prefix="ses-",
    )

    dm.setup()

    # CHECK DATASET
    print(f"\nTrain samples: {len(dm.train_dataset)}")
    sample = dm.train_dataset[0]
    print("\nAvailable keys:")
    print(sample.keys())

    print(f"Subject: {sample['subject']}")
    print(f"Session: {sample['session']}")
    print(f"age at visit: {sample['metadata'].get('age_at_visit', 'N/A')}")
    # # print(f"Metadata keys: {list(sample['metadata'].keys())}")
    # print(
    #     f"Metadata sample: { {k: sample['metadata'][k] for k in list(sample['metadata'].keys())} }"
    # )

    print("\nModalities tensor shapes")
    for mod in dm.modalities.keys():
        if mod in sample:
            if isinstance(sample[mod], torch.Tensor):
                print(f"{mod}: shape={tuple(sample[mod].shape)}")
                print(
                    sample[mod].min(),
                    sample[mod].max(),
                    sample[mod].mean(),
                    sample[mod].std(),
                )
            else:
                print(f"{mod}: {sample[mod]}")
                print(f"{mod}: type={type(sample[mod])}")
        else:
            print(f"{mod}: MISSING")

    # CHECK DATALOADER

    train_loader = dm.train_dataloader()

    # Inspect one batch
    print("\n--- Train batch ---")
    batch = next(iter(train_loader))
    print("Batch keys:", batch.keys())

    # inter full dataloader to check for any issues
    print("\nIterating through train dataloader...")
    for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        print(f"Batch {i}:")
        print("First batch keys:", batch.keys())
        for k in batch.keys():
            if isinstance(batch[k], torch.Tensor):
                print(f"  {k}: shape={tuple(batch[k].shape)}, dtype={batch[k].dtype}")
            else:
                print(f"  {k}: {batch[k]}")
                print(f"  {k}: type={type(batch[k])}")

        # Just iterate to check for errors, no need to print everything
    print("Finished iterating through train dataloader without errors.")
