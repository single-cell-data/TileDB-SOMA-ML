from __future__ import annotations

from typing import Any, Sequence

import pandas as pd
import torch
from lightning import LightningDataModule
from sklearn.preprocessing import LabelEncoder
from tiledbsoma import ExperimentAxisQuery, SOMATileDBContext
from torch.utils.data import DataLoader

from tiledbsoma_ml import ExperimentDataset, experiment_dataloader
from tiledbsoma_ml.dataloader import optimized_experiment_dataloader
from tiledbsoma_ml._common import MiniBatch


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
        use_optimized_dataloader: bool = False,
        context: SOMATileDBContext | None = None,
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

        use_optimized_dataloader: bool, optional
        Whether to use the optimized dataloader with threading and GPU optimizations. 
        Defaults to False for backward compatibility. When True, uses 
        `optimized_experiment_dataloader` which returns dictionary format batches.

        context: SOMATileDBContext, optional
        Custom TileDB context for S3 optimizations. If provided, this context will be used
        for all TileDB-SOMA operations. Useful for configuring S3 performance settings.
        """
        super().__init__()
        self.query = query
        self.dataset_args = args
        self.dataset_kwargs = kwargs
        if context is not None:
            self.dataset_kwargs["context"] = context
        self.dataloader_kwargs = (
            dataloader_kwargs if dataloader_kwargs is not None else {}
        )
        self.use_optimized_dataloader = use_optimized_dataloader
        self.context = context
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
            # Use custom context if provided
            if self.context is not None:
                with self.query.experiment.open(context=self.context) as exp:
                    obs_df = (
                        exp.obs(column_names=self.batch_column_names)
                        .concat()
                        .to_pandas()
                    )
            else:
                obs_df = (
                    self.query.obs(column_names=self.batch_column_names)
                    .concat()
                    .to_pandas()
                )
            obs_df = self._add_batch_col(obs_df, inplace=False)
            batch_labels = obs_df[self.batch_colname].unique()
        self.batch_labels = batch_labels
        self.batch_encoder = LabelEncoder().fit(self.batch_labels)

    def setup(self, stage: str | None = None) -> None:
        # Instantiate the ExperimentDataset with the provided args and kwargs.
        self.train_dataset = ExperimentDataset(
            self.query,
            *self.dataset_args,
            obs_column_names=self.batch_column_names,  # type: ignore[arg-type]
            **self.dataset_kwargs,  # type: ignore[misc]
        )

    def train_dataloader(self) -> DataLoader:
        if self.use_optimized_dataloader:
            return optimized_experiment_dataloader(
                self.train_dataset,
                **self.dataloader_kwargs,
            )
        else:
            return experiment_dataloader(
                self.train_dataset,
                **self.dataloader_kwargs,
            )

    def _add_batch_col(
        self, obs_df: pd.DataFrame, inplace: bool = False
    ) -> pd.DataFrame:
        # synthesize a new column for obs_df by concatenating the self.batch_column_names columns
        if not inplace:
            obs_df = obs_df.copy()
        else:
            # Always make a copy to avoid SettingWithCopyWarning
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
        # DataModule hook: transform the ExperimentDataset data batch into X & batch variable tensors for scVI
        # Supports both tuple format (standard dataloader) and dictionary format (optimized dataloader)
        
        if isinstance(batch, dict):
            # Optimized dataloader format: {'X': tensor, 'obs': dataframe, ...}
            batch_X = batch['X']
            batch_obs = batch.get('obs', None)
            
            # Handle case where obs data might not be present in optimized format
            if batch_obs is None:
                # If no obs data, create a minimal batch encoding
                # This assumes all samples belong to the same batch for this mini-batch
                batch_size = batch_X.shape[0] if hasattr(batch_X, 'shape') else len(batch_X)
                default_batch_label = self.batch_colsep.join(['unknown'] * len(self.batch_column_names))
                batch_encoded = torch.zeros(batch_size, 1, dtype=torch.long)
                
                return {
                    "X": torch.from_numpy(batch_X).float() if not torch.is_tensor(batch_X) else batch_X.float(),
                    "batch": batch_encoded,
                    "labels": torch.empty(0),
                }
            else:
                batch_obs = self._add_batch_col(batch_obs, inplace=False)
                return {
                    "X": torch.from_numpy(batch_X).float() if not torch.is_tensor(batch_X) else batch_X.float(),
                    "batch": torch.from_numpy(
                        self.batch_encoder.transform(batch_obs[self.batch_colname])
                    ).unsqueeze(1),
                    "labels": torch.empty(0),
                }
        else:
            # Standard dataloader format: (X_batch, obs_batch) tuple
            batch_X, batch_obs = batch
            batch_obs = self._add_batch_col(batch_obs, inplace=False)
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
