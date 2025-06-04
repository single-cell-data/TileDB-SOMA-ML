from __future__ import annotations

import lightning
from tiledbsoma import ExperimentAxisQuery

from tiledbsoma_ml import ExperimentDataset, experiment_dataloader
from tiledbsoma_ml._common import MiniBatch

import numpy as np
import pandas as pd
import torch
from functools import lru_cache
import numba
from typing import Any, Sequence, Dict
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
import os


# Numba-accelerated string joining
# def _fast_join_strings(str_arrays, separator):
#     result = np.empty(len(str_arrays[0]), dtype=object)
#     for i in range(len(result)):
#         parts = [str_arrays[j][i] for j in range(len(str_arrays))]
#         result[i] = separator.join(parts)
#     return result

def _ultra_fast_join_strings(str_arrays, separator):
    # Convert to 2D NumPy array of str if not already
    arr = np.array(str_arrays, dtype=str)
    return np.char.add.reduce(np.char.add(arr, separator)[:-1], axis=0)

class CUDAStreamPrefetchCallback(lightning.Callback):
    def __init__(self, num_streams=3):
        super().__init__()
        self.num_streams = num_streams
        self.streams = None
        self.current_stream = 0

    def on_fit_start(self, trainer, pl_module):
        # Create CUDA streams for prefetching
        self.streams = [torch.cuda.Stream() for _ in range(self.num_streams)]

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Set current stream for computing this batch
        torch.cuda.current_stream().wait_stream(self.streams[self.current_stream])

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Queue next batch loading on the next stream
        self.current_stream = (self.current_stream + 1) % self.num_streams
        with torch.cuda.stream(self.streams[self.current_stream]):
            # Signal to start loading the next batch
            pass


class SCVIDataModule(LightningDataModule):  # type: ignore[misc]
    """PyTorch Lightning DataModule for training scVI models from SOMA data."""

    def __init__(
            self,
            query: ExperimentAxisQuery,
            *args: Any,
            batch_column_names: Sequence[str] | None = None,
            batch_labels: Sequence[str] | None = None,
            dataloader_kwargs: dict[str, Any] | None = None,
            cache_size: int = 1000,  # Number of batches to cache
            **kwargs: Any,
    ):
        """Initialize the SCVIDataModule."""
        super().__init__()
        self.train_dataset = None
        self.query = query
        self.dataset_args = args
        self.dataset_kwargs = kwargs
        self.dataloader_kwargs = dataloader_kwargs or {}
        self.batch_column_names = batch_column_names or ["dataset_id", "assay", "suspension_type", "donor_id"]
        self.batch_colsep = "//"
        self.batch_colname = "scvi_batch"
        self.cache_size = cache_size
        self._batch_cache = {}
        self._performance_metrics = {
            'batch_load_times': [],
            'batch_transfer_times': [],
            'memory_usage': []
        }

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

    def _manage_memory(self):
        """Manage memory usage by clearing caches if needed."""
        if torch.cuda.is_available():
            # Clear CUDA cache if memory usage is high
            if torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() > 0.9:
                torch.cuda.empty_cache()
                self._batch_cache.clear()
                self._encoding_cache.clear()

    def _get_optimal_batch_size(self) -> int:
        """Suggest optimal batch size based on available memory."""
        if torch.cuda.is_available():
            # Get available GPU memory
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            # Estimate memory per sample (rough estimate)
            memory_per_sample = 4 * self.n_vars * 4  # 4 bytes per float32
            # Calculate safe batch size (using 80% of available memory)
            optimal_batch_size = int((free_memory * 0.8) / memory_per_sample)
            return max(32, min(optimal_batch_size, 1024))  # Keep between 32 and 1024
        return 128  # Default batch size for CPU

    def setup(self, stage: str | None = None) -> None:
        """Set up the data module."""
        # Instantiate the ExperimentDataset
        self.train_dataset = ExperimentDataset(
            self.query,
            *self.dataset_args,
            obs_column_names=self.batch_column_names, # type: ignore[arg-type]
            **self.dataset_kwargs, # type: ignore[misc]
        )

        # Pre-fetch a small batch to warm up caches
        if hasattr(self, 'train_dataset') and len(self.train_dataset) > 0:
            try:
                self.train_dataset[0]
            except:
                pass

        # Set optimal batch size if not specified
        if 'batch_size' not in self.dataloader_kwargs:
            self.dataloader_kwargs['batch_size'] = self._get_optimal_batch_size()

    def train_dataloader(self) -> DataLoader:
        """Return optimized DataLoader for training."""
        # Optimize DataLoader settings based on system resources
        num_workers = min(8, os.cpu_count() or 4)  # Use up to 8 workers or available CPU cores
        default_kwargs = {
            'num_workers': num_workers,
            'pin_memory': torch.cuda.is_available(),
            'prefetch_factor': 3,  # Increased from 2
            'persistent_workers': True,  # Changed to True for better performance
            'shuffle': True,
            'drop_last': True,  # Drop incomplete batches for better performance
        }

        # Override with user settings but keep our optimizations
        dataloader_kwargs = {**default_kwargs, **self.dataloader_kwargs}
        
        # Ensure use_eager_fetch is enabled for better performance
        if 'use_eager_fetch' not in self.dataset_kwargs:
            self.dataset_kwargs['use_eager_fetch'] = 2  # Increased from 1 for better prefetching

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
        import time
        start_time = time.time()

        batch_X, batch_obs = batch

        # Check cache first
        cache_key = hash(str(batch_obs.index))
        if cache_key in self._batch_cache:
            return self._batch_cache[cache_key]

        # Only add batch column if needed
        if self.batch_colname not in batch_obs.columns:
            self._add_batch_col(batch_obs, inplace=True)

        # Ensure contiguous memory layout for efficient GPU transfer
        X_tensor = torch.from_numpy(batch_X).float().contiguous()

        # Get batch values and convert to indices using vectorized operations
        batch_values = batch_obs[self.batch_colname].values
        batch_indices = np.vectorize(self._batch_value_to_index_map.get)(batch_values)
        
        # Convert to tensor efficiently with proper memory layout
        batch_tensor = torch.from_numpy(batch_indices).unsqueeze(1).contiguous()

        result = {
            "X": X_tensor,
            "batch": batch_tensor,
            "labels": torch.empty(0),
        }

        # Cache the result
        if len(self._batch_cache) < self.cache_size:
            self._batch_cache[cache_key] = result

        # Record performance metrics
        transfer_time = time.time() - start_time
        self._performance_metrics['batch_transfer_times'].append(transfer_time)
        if torch.cuda.is_available():
            self._performance_metrics['memory_usage'].append(torch.cuda.memory_allocated())

        # Manage memory periodically
        if len(self._performance_metrics['batch_transfer_times']) % 100 == 0:
            self._manage_memory()

        return result

    def get_performance_metrics(self) -> dict:
        """Get performance metrics for monitoring."""
        return {
            'avg_batch_transfer_time': np.mean(self._performance_metrics['batch_transfer_times']),
            'max_memory_usage': max(self._performance_metrics['memory_usage']) if self._performance_metrics['memory_usage'] else 0,
            'cache_hits': len(self._batch_cache),
            'optimal_batch_size': self._get_optimal_batch_size()
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

    def get_cuda_prefetch_callback(self) -> CUDAStreamPrefetchCallback:
        """Get the CUDA prefetch callback for overlapping data loading with computation."""
        return CUDAStreamPrefetchCallback(num_streams=3)