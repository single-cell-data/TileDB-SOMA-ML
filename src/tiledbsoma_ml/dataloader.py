# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
#
# Licensed under the MIT License.

from __future__ import annotations

from typing import Any, TypeVar

from torch.utils.data import DataLoader

from tiledbsoma_ml._distributed import init_multiprocessing
from tiledbsoma_ml.dataset import ExperimentDataset

_T = TypeVar("_T")


def experiment_dataloader(
    ds: ExperimentDataset,
    **dataloader_kwargs: Any,
) -> DataLoader:
    """|DataLoader| factory method for safely wrapping an |ExperimentDataset|.

    Several |DataLoader| constructor parameters are not applicable, or are non-performant,
    when using loaders from this module, including ``shuffle``, ``batch_size``, ``sampler``, and ``batch_sampler``.
    Specifying any of these parameters will result in an error.

    Refer to ``https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader`` for more information on
    |DataLoader| parameters.

    Args:
        ds:
            A |IterableDataset|. May include chained data pipes.
        **dataloader_kwargs:
            Additional keyword arguments to pass to the |DataLoader| constructor,
            except for ``shuffle``, ``batch_size``, ``sampler``, and ``batch_sampler``, which are not
            supported when using data loaders in this module.

    Returns:
        |DataLoader|

    Raises:
        ValueError: if any of the ``shuffle``, ``batch_size``, ``sampler``, or ``batch_sampler`` params
            are passed as keyword arguments.

    Lifecycle:
        experimental
    """
    unsupported_dataloader_args = [
        "shuffle",
        "batch_size",
        "sampler",
        "batch_sampler",
    ]
    if set(unsupported_dataloader_args).intersection(dataloader_kwargs.keys()):
        raise ValueError(
            f"The {','.join(unsupported_dataloader_args)} DataLoader parameters are not supported"
        )

    if dataloader_kwargs.get("num_workers", 0) > 0:
        init_multiprocessing()

    if "collate_fn" not in dataloader_kwargs:
        dataloader_kwargs["collate_fn"] = _collate_noop

    return DataLoader(
        ds,
        batch_size=None,  # batching is handled by upstream iterator
        shuffle=False,  # shuffling is handled by upstream iterator
        **dataloader_kwargs,
    )


def _collate_noop(datum: _T) -> _T:
    """Noop collation used by |experiment_dataloader|.

    Private.
    """
    return datum
