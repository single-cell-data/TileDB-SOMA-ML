from __future__ import annotations

import os
from enum import Enum
from typing import Any, Sequence

import pandas as pd
import torch
from lightning import LightningDataModule
from sklearn.preprocessing import LabelEncoder
from tiledbsoma import ExperimentAxisQuery
from torch.utils.data import DataLoader

from tiledbsoma_ml import ExperimentDataset, experiment_dataloader
from tiledbsoma_ml._common import MiniBatch
from tiledbsoma_ml._query_ids import QueryIDs
from tiledbsoma_ml.x_locator import XLocator

DEFAULT_DATALOADER_KWARGS: dict[str, Any] = {
    "pin_memory": torch.cuda.is_available(),
    "persistent_workers": True,
    "num_workers": max(((os.cpu_count() or 1) // 2), 1),
}


class DatasetSplit(Enum):
    """Enum for dataset splits."""

    TRAIN = "train"
    VAL = "val"


class SCVIDataModule(LightningDataModule):  # type: ignore[misc]
    """PyTorch Lightning DataModule for training scVI models from SOMA data.

    Wraps a |ExperimentDataset| to stream the results of a SOMA |ExperimentAxisQuery|,
    exposing a |DataLoader| to generate tensors ready for scVI model training. Also handles deriving
    the scVI batch label as a tuple of obs columns.

    Lifecycle:
        Experimental.
    """

    def __init__(
        self,
        query: ExperimentAxisQuery,
        *args: Any,
        batch_column_names: Sequence[str] | None = None,
        batch_labels: Sequence[str] | None = None,
        dataloader_kwargs: dict[str, Any] | None = None,
        train_size: float = 1.0,
        seed: int = 42,
        **kwargs: Any,
    ):
        """Args:

        query: |ExperimentAxisQuery|
        Defines the desired result set from a SOMA |Experiment|.
        *args, **kwargs:
        Additional arguments passed through to |ExperimentDataset|.

        batch_column_names: Sequence[str], optional
        List of obs column names, the tuple of which defines the scVI batch label (not to to be confused with
                        a batch of training data). Defaults to
                        `["dataset_id", "assay", "suspension_type", "donor_id"]`.

        batch_labels: Sequence[str], optional
        List of possible values of the batch label, for mapping to label tensors. By default,
                        this will be derived from the unique labels in the given query results (given
                        `batch_column_names`), making the label mapping depend on the query. The `batch_labels`
                        attribute in the `SCVIDataModule` used for training may be saved and here restored in
                        another instance for a different query. That ensures the label mapping will be correct
                        for the trained model, even if the second query doesn't return examples of every
                        training batch label.

        dataloader_kwargs: dict, optional
        Keyword arguments passed to `tiledbsoma_ml.experiment_dataloader()`, e.g. `num_workers`.

        train_size: float, optional
        Fraction of data to use for training (between 0 and 1). Default is 1.0 (use all data for training).
        If less than 1.0, the remaining data will be used for validation.

        seed: int, optional
        Random seed for deterministic train/validation split. Default is 42.
        """
        super().__init__()
        self.query = query
        self.dataset_args = args
        self.dataset_kwargs = kwargs
        self.dataloader_kwargs = {
            **DEFAULT_DATALOADER_KWARGS,
            **(dataloader_kwargs or {}),
        }
        self.batch_column_names = (
            batch_column_names
            if batch_column_names is not None
            else ["dataset_id", "assay", "suspension_type", "donor_id"]
        )
        self.batch_colsep = "//"
        self.batch_colname = "scvi_batch"
        # prepare LabelEncoder for the scVI batch label:
        #   1. read obs DataFrame for the whole query result set
        #   2. add scvi_batch column
        #   3. fit LabelEncoder to the scvi_batch column's unique values
        if batch_labels is None:
            obs_df = (
                self.query.obs(column_names=self.batch_column_names)
                .concat()
                .to_pandas()
            )
            self._add_batch_col(obs_df, inplace=True)
            batch_labels = obs_df[self.batch_colname].unique()
        self.batch_labels = batch_labels
        self.batch_encoder = LabelEncoder().fit(self.batch_labels)
        self.train_size = train_size
        self.seed = seed
        self.train_query_ids: QueryIDs | None = None
        self.val_query_ids: QueryIDs | None = None
        self.x_locator: XLocator | None = None
        self.layer_name = kwargs.get("layer_name", "raw")

    def setup(self, stage: str | None = None) -> None:
        # Create QueryIDs and XLocator from the query
        query_ids = QueryIDs.create(self.query)
        self.x_locator = XLocator.create(
            self.query.experiment,
            measurement_name=self.query.measurement_name,
            layer_name=self.layer_name,
        )

        # Split data into train and validation sets if train_size < 1.0
        if self.train_size < 1.0:
            # Use QueryIDs.random_split() for efficient splitting
            val_size = 1.0 - self.train_size
            train_ids, val_ids = query_ids.random_split(
                self.train_size, val_size, seed=self.seed
            )
            self.train_query_ids = train_ids
            self.val_query_ids = val_ids
        else:
            # Use all data for training
            self.train_query_ids = query_ids
            self.val_query_ids = None

    def _create_dataloader(self, split: DatasetSplit) -> DataLoader | None:
        """Create a dataloader for the specified dataset split.

        Args:
            split: The dataset split (TRAIN or VAL)

        Returns:
            DataLoader for the specified split, or None if the split doesn't exist
        """
        # Get the appropriate query_ids based on split
        query_ids_map = {
            DatasetSplit.TRAIN: self.train_query_ids,
            DatasetSplit.VAL: self.val_query_ids,
        }

        query_ids = query_ids_map.get(split)
        if query_ids is None or self.x_locator is None:
            return None

        # Filter out query and layer_name from dataset_kwargs since we're using x_locator and query_ids
        filtered_kwargs = {
            k: v
            for k, v in self.dataset_kwargs.items()
            if k not in ("query", "layer_name")
        }

        # Create dataset with appropriate query_ids
        dataset = ExperimentDataset(
            x_locator=self.x_locator,
            query_ids=query_ids,
            obs_column_names=list(self.batch_column_names),
            **filtered_kwargs,
        )
        return experiment_dataloader(
            dataset,
            **self.dataloader_kwargs,
        )

    def train_dataloader(self) -> DataLoader:
        """Create the training dataloader.

        Returns:
            DataLoader for training data

        Raises:
            AssertionError: If setup() hasn't been called
        """
        loader = self._create_dataloader(DatasetSplit.TRAIN)
        assert loader is not None, "setup() must be called before train_dataloader()"
        return loader

    def val_dataloader(self) -> DataLoader | None:
        """Create the validation dataloader.

        Returns:
            DataLoader for validation data, or None if no validation split exists
        """
        return self._create_dataloader(DatasetSplit.VAL)

    def _add_batch_col(
        self, obs_df: pd.DataFrame, inplace: bool = False
    ) -> pd.DataFrame:
        # synthesize a new column for obs_df by concatenating the self.batch_column_names columns
        if not inplace:
            obs_df = obs_df.copy()
        obs_df[self.batch_colname] = (
            obs_df[self.batch_column_names]
            .astype(str)
            .agg(self.batch_colsep.join, axis=1)
        )
        return obs_df

    def on_before_batch_transfer(
        self,
        batch: MiniBatch,
        dataloader_idx: int,
    ) -> dict[str, torch.Tensor | None]:
        # DataModule hook: transform the ExperimentDataset data batch (X: ndarray, obs_df: DataFrame)
        # into X & batch variable tensors for scVI (using batch_encoder on scvi_batch)
        batch_X, batch_obs = batch
        self._add_batch_col(batch_obs, inplace=True)
        return {
            "X": torch.from_numpy(batch_X).float(),
            "batch": torch.from_numpy(
                self.batch_encoder.transform(batch_obs[self.batch_colname])
            ).unsqueeze(1),
            "labels": torch.empty(0),
        }

    # scVI expects these properties on the DataModule:

    @property
    def n_obs(self) -> int:
        return len(self.query.obs_joinids())

    @property
    def n_vars(self) -> int:
        return len(self.query.var_joinids())

    @property
    def n_batch(self) -> int:
        return len(self.batch_encoder.classes_)
