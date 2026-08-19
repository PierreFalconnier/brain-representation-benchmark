from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


# =========================
# Dataset
# =========================
class BIDSDataset(Dataset):
    """
    MONAI-compatible dataset for BIDS-like folder structures.

    Supports:
    - Strict BIDS layout:   <root>/sub-<id>/ses-<id>/<files>
    - Session-less layout:  <root>/sub-<id>/<files>
    - Flat layout:          <root>/<files>  (no subject/session grouping; all samples are independent)

    Args:
        root_dir:         Root directory of the dataset.
        modalities:       Dict mapping a modality name to a glob pattern.
                          Example: {"T1w": "*T1_brain*.npy", "dwi": "*dwi*.nii.gz"}
                          Patterns are matched recursively under each subject/session folder.
        csv_path:         Optional CSV with metadata.  Must contain a "subject" column
                          and optionally a "session" column.  Used as the join key.
        metadata_columns: Optional list of column names to extract from the CSV.
                          If None, all columns are kept.  Ignored when csv_path is None.
        transform:        Optional callable applied to the output dict of __getitem__.
        sub_prefix:       Prefix for subject folders (default "sub-").
        session_prefix:   Prefix for session folders (default "ses-").

    Returns (per sample):
        {
            "subject":  str,
            "session":  str | None,   # None when dataset has no session level
            "metadata": dict,         # empty dict when no CSV was supplied
            "<modality>": str,        # absolute path for each requested modality
            ...
        }
    """

    def __init__(
        self,
        root_dir: str | Path,
        modalities: dict[str, str],
        csv_path: str | Path | None = None,
        metadata_columns: list[str] | None = None,
        transform: Callable | None = None,
        sub_prefix: str = "sub-",
        session_prefix: str = "ses-",
    ):
        self.root_dir = Path(root_dir)
        self.modalities = modalities  # {name: glob_pattern}
        self.transform = transform
        self.sub_prefix = sub_prefix
        self.session_prefix = session_prefix
        self.dataset_is_flat = None  # set inside _build_index

        self._metadata_join_cols: list[str] = []  # set inside _load_csv
        self.df_metadata = self._load_csv(csv_path, metadata_columns)
        self.metadata_columns: list[str] = (
            [c for c in self.df_metadata.columns if c not in self._metadata_join_cols]
            if self.df_metadata is not None
            else []
        )

        print(f"Building index from {self.root_dir}...")
        self.index_df, self.has_sessions = self._build_index()

        print("Building samples...")
        self.samples = self._build_samples()

        print(f"Dataset ready: {len(self)} samples.")

    # ------------------------------------------------------------------
    # CSV / metadata
    # ------------------------------------------------------------------

    def _load_csv(
        self,
        csv_path: str | Path | None,
        metadata_columns: list[str] | None,
    ) -> pd.DataFrame | None:
        if csv_path is None:
            return None

        df = pd.read_csv(csv_path)
        df["subject"] = df["subject"].astype(str)

        has_session_col = "session" in df.columns
        if has_session_col:
            df["session"] = df["session"].apply(
                lambda x: str(int(x)) if pd.notna(x) else None
            )
            index_cols = ["subject", "session"]
        else:
            index_cols = ["subject"]

        df = df.drop_duplicates(subset=index_cols, keep="first")

        # Select only the requested columns (index cols are always kept)
        # standardise column names to lowercase for easier matching with metadata
        df.columns = [col.lower() for col in df.columns]

        if metadata_columns is not None:
            metadata_columns = [col.lower() for col in metadata_columns]
            missing = set(metadata_columns) - set(df.columns)
            if missing:
                raise ValueError(
                    f"Requested metadata columns not found in CSV: {missing}"
                )
            df = df[index_cols + metadata_columns]

        # store join cols so _build_samples knows how to merge
        self._metadata_join_cols = index_cols

        return df  # flat DataFrame, not indexed

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _build_index(self) -> tuple[pd.DataFrame, bool]:
        """
        Scan the root directory and return a long-format DataFrame with columns
        (subject, session, modality, path) and a flag indicating whether a session
        level was detected.

        Three layouts are supported, detected automatically:
        - sub + ses:  <root>/<sub_prefix>-<id>/<ses_prefix>-<id>/<files>
        - sub only:   <root>/<sub_prefix>-<id>/<files>
        - flat:       <root>/<files>   (subject=None, session=None for all rows)
        """
        rows = []
        has_sessions = False

        sub_dirs = sorted(
            d for d in self.root_dir.glob(f"{self.sub_prefix}*") if d.is_dir()
        )

        if sub_dirs:
            self.dataset_is_flat = False
            # --- structured layout: sub[-ses] folders ---
            for sub_dir in sub_dirs:
                subject = sub_dir.name.split("-", 1)[1]
                ses_dirs = sorted(
                    d for d in sub_dir.glob(f"{self.session_prefix}*") if d.is_dir()
                )

                if ses_dirs:
                    has_sessions = True
                    search_dirs = [(ses.name.split("-", 1)[1], ses) for ses in ses_dirs]
                else:
                    search_dirs = [(None, sub_dir)]

                for session, search_root in search_dirs:
                    for mod_name, pattern in self.modalities.items():
                        files = sorted(search_root.rglob(pattern))
                        if files:
                            rows.append(
                                {
                                    "subject": subject,
                                    "session": session,
                                    "modality": mod_name,
                                    "path": str(files[0]),
                                }
                            )
        else:
            # --- flat layout: files directly under root_dir, no sub/ses structure ---
            self.dataset_is_flat = True
            for mod_name, pattern in self.modalities.items():
                for f in sorted(self.root_dir.rglob(pattern)):
                    rows.append(
                        {
                            "subject": None,
                            "session": None,
                            "modality": mod_name,
                            "path": str(f),
                        }
                    )

        df = pd.DataFrame(rows, columns=["subject", "session", "modality", "path"])

        if df.empty:
            raise RuntimeError(
                f"No data found in '{self.root_dir}' for the requested modalities: "
                f"{list(self.modalities.keys())}"
            )

        return df, has_sessions

    # ------------------------------------------------------------------
    # Sample building
    # ------------------------------------------------------------------

    def _build_samples(self) -> list[dict]:
        """
        Pivot the long-format index into one row per (subject [, session]),
        keep only complete samples, merge metadata if available, and
        normalise numpy scalar types for safe batching.
        """
        # --- flat layout: subject and session are all None, no grouping key ---
        # pivot_table would collapse everything into one row; instead just
        # keep the long-format rows as-is and verify all modalities are present.
        if not self.has_sessions and self.index_df["subject"].isna().all():
            present = set(self.index_df["modality"].unique())
            missing = set(self.modalities.keys()) - present
            if missing:
                raise RuntimeError(
                    f"Flat layout is missing required modalities: {missing}"
                )
            # Number files within each modality, then pivot on that counter
            self.index_df["_sample_idx"] = self.index_df.groupby("modality").cumcount()
            df = self.index_df.pivot_table(
                index="_sample_idx",
                columns="modality",
                values="path",
                aggfunc="first",
            )
            df.columns.name = None
            df = df.reset_index(drop=True)
            df["subject"] = None
            df["session"] = None

        else:
            # --- structured layout: group by subject [+ session] ---
            index_cols = ["subject", "session"] if self.has_sessions else ["subject"]

            df = self.index_df.pivot_table(
                index=index_cols,
                columns="modality",
                values="path",
                aggfunc="first",
            )
            df.columns.name = None
            df = df.reset_index()

        # Drop samples missing any required modality
        df = df.dropna(subset=list(self.modalities.keys()))

        # Merge with metadata (inner join — drops samples with no metadata row)
        if self.df_metadata is not None:
            # Ensure consistent types on join keys
            for col in self._metadata_join_cols:
                df[col] = df[col].astype(str)
                self.df_metadata[col] = self.df_metadata[col].astype(str)

            df = df.merge(self.df_metadata, on=self._metadata_join_cols, how="inner")

            # Drop rows where any requested metadata column is NaN
            if self.metadata_columns:
                df = df.dropna(subset=self.metadata_columns)

        # Normalise numpy scalars and strip NaN/None for safe DataLoader batching
        cleaned = []
        for record in df.to_dict("records"):
            sample: dict = {}
            for k, v in record.items():
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                if isinstance(v, np.integer):
                    v = v.item()
                elif isinstance(v, np.floating):
                    v = v.item()
                elif isinstance(v, np.bool_):
                    v = bool(v)
                elif isinstance(v, np.ndarray):
                    v = v.tolist()
                sample[k] = v
            cleaned.append(sample)

        return cleaned

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        # number of samples
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        out = {
            "subject": sample.get("subject", "unknown"),  # None cause batching issues
            "session": sample.get("session", "unknown"),
            **{k: sample[k] for k in self.metadata_columns if k in sample},
            **{m: sample[m] for m in self.modalities},
        }

        return self.transform(out) if self.transform else out
