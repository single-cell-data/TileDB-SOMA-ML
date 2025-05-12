from __future__ import annotations

from tiledbsoma import ExperimentAxisQuery

from tiledbsoma_ml import ExperimentDataset, experiment_dataloader
from tiledbsoma_ml._common import MiniBatch

import numpy as np
import pandas as pd
import torch
from functools import lru_cache
import numba
from typing import Any, Sequence, Dict
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder


# Numba-accelerated string joining
@numba.jit(nopython=False)
def _fast_join_strings(str_arrays, separator):
    result = np.empty(len(str_arrays[0]), dtype=object)
    for i in range(len(result)):
        parts = [str_arrays[j][i] for j in range(len(str_arrays))]
        result[i] = separator.join(parts)
    return result


class SCVIDataModule(LightningDataModule):  # type: ignore[misc]
    """PyTorch Lightning DataModule for training scVI models from SOMA data."""

    def __init__(
            self,
            query: ExperimentAxisQuery,
            *args: Any,
            batch_column_names: Sequence[str] | None = None,
            batch_labels: Sequence[str] | None = None,
            dataloader_kwargs: dict[str, Any] | None = None,
            **kwargs: Any,
    ):
        """Initialize the SCVIDataModule."""
        super().__init__()
        self.query = query
        self.dataset_args = args
        self.dataset_kwargs = kwargs
        self.dataloader_kwargs = dataloader_kwargs or {}
        self.batch_column_names = batch_column_names or ["dataset_id", "assay", "suspension_type", "donor_id"]
        self.batch_colsep = "//"
        self.batch_colname = "scvi_batch"

        # Prepare LabelEncoder for the scVI batch label
        if batch_labels is None:
            obs_df = self.query.obs(column_names=self.batch_column_names).concat().to_pandas()
            self._add_batch_col(obs_df, inplace=True)
            batch_labels = obs_df[self.batch_colname].unique()

        self.batch_labels = batch_labels
        self.batch_encoder = LabelEncoder().fit(self.batch_labels)

        # Pre-compute batch value to index mapping
        self._batch_value_to_index_map = {
            val: idx for idx, val in enumerate(self.batch_labels)
        }

        # Initialize cache for transformed batch encodings
        self._encoding_cache = {}

    def _add_batch_col(self, obs_df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """Add batch column to obs DataFrame efficiently."""
        # Skip if column already exists
        if self.batch_colname in obs_df.columns:
            return obs_df

        if not inplace:
            obs_df = obs_df.copy()

        # Fast path for single column case
        if len(self.batch_column_names) == 1:
            obs_df[self.batch_colname] = obs_df[self.batch_column_names[0]].astype(str)
            return obs_df

        # Extract string arrays
        str_arrays = [obs_df[col].astype(str).values for col in self.batch_column_names]

        # Use numba-accelerated function
        result = _fast_join_strings(str_arrays, self.batch_colsep)

        obs_df[self.batch_colname] = result
        return obs_df

    def setup(self, stage: str | None = None) -> None:
        """Set up the data module."""
        # Instantiate the ExperimentDataset
        self.train_dataset = ExperimentDataset(
            self.query,
            *self.dataset_args,
            obs_column_names=self.batch_column_names,
            **self.dataset_kwargs,
        )

        # Pre-fetch a small batch to warm up caches
        if hasattr(self, 'train_dataset') and len(self.train_dataset) > 0:
            try:
                self.train_dataset[0]
            except:
                pass

    def train_dataloader(self) -> DataLoader:
        """Return optimized DataLoader for training."""
        default_kwargs = {
            'num_workers': 4,
            'pin_memory': torch.cuda.is_available(),
            'prefetch_factor': 2,
            'persistent_workers': True,
        }

        dataloader_kwargs = {**default_kwargs, **self.dataloader_kwargs}

        return experiment_dataloader(
            self.train_dataset,
            **dataloader_kwargs,
        )

    def on_before_batch_transfer(
            self,
            batch: MiniBatch,
            dataloader_idx: int,
    ) -> dict[str, torch.Tensor | None]:
        """Transform batch data efficiently before transfer to device."""
        batch_X, batch_obs = batch

        # Only add batch column if needed
        if self.batch_colname not in batch_obs.columns:
            self._add_batch_col(batch_obs, inplace=True)

        # Ensure contiguous memory layout for efficient GPU transfer
        X_tensor = torch.from_numpy(batch_X).float().contiguous()

        # Get batch values
        batch_values = batch_obs[self.batch_colname].values

        # Transform batch values to indices efficiently using our pre-computed mapping
        batch_indices = np.empty(len(batch_values), dtype=np.int64)

        # Use mapping for fast lookup
        for i, val in enumerate(batch_values):
            batch_indices[i] = self._batch_value_to_index_map.get(val, 0)

        # Convert to tensor efficiently with proper memory layout
        batch_tensor = torch.from_numpy(batch_indices).unsqueeze(1).contiguous()

        return {
            "X": X_tensor,
            "batch": batch_tensor,
            "labels": torch.empty(0),
        }

    # scVI expected properties
    @property
    def n_obs(self) -> int:
        return len(self.query.obs_joinids())

    @property
    def n_vars(self) -> int:
        return len(self.query.var_joinids())

    @property
    def n_batch(self) -> int:
        return len(self.batch_encoder.classes_)