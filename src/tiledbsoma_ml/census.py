from typing import Tuple

from sklearn.preprocessing import LabelEncoder
from somacore import AxisQuery
from tiledbsoma import Experiment, SOMATileDBContext
from torch.utils.data import DataLoader

from tiledbsoma_ml import ExperimentAxisQueryIterableDataset, experiment_dataloader, ExperimentAxisQueryIterDataPipe
from tiledbsoma_ml.dataloader import ExperimentAxisQueryIterableWrapper, ExperimentAxisQueryIterableWrapperType


def census_dataloader(
    tissue: str,
    batch_size: int,
    io_batch_size: int,
    shuffle_chunk_size: int,
    census_version: str,
    num_workers: int | None,
    seed: int | None,
    use_iterdatapipe: bool,
    **dataloader_kwargs,
) -> Tuple[ExperimentAxisQueryIterableWrapper, DataLoader, LabelEncoder]:
    census_url = f"s3://cellxgene-census-public-us-west-2/cell-census/{census_version}/soma/census_data/homo_sapiens/"
    experiment = Experiment.open(
        census_url,
        context=SOMATileDBContext(tiledb_config={"vfs.s3.region": "us-west-2"}),
    )
    obs_value_filter = f"tissue_general == '{tissue}' and is_primary_data == True"
    with experiment.axis_query(
            measurement_name="RNA",
            obs_query=AxisQuery(value_filter=obs_value_filter)
    ) as query:
        obs_df = query.obs(column_names=["cell_type"]).concat().to_pandas()
        cell_type_encoder = LabelEncoder().fit(obs_df["cell_type"].unique())

        cls: ExperimentAxisQueryIterableWrapperType = (
            ExperimentAxisQueryIterDataPipe if use_iterdatapipe else ExperimentAxisQueryIterableDataset
        )
        ds = cls(
            query,
            X_name="raw",
            obs_column_names=["cell_type"],
            batch_size=batch_size,
            io_batch_size=io_batch_size,
            shuffle_chunk_size=shuffle_chunk_size,
            seed=seed,
        )

    dl = experiment_dataloader(
        ds,
        num_workers=num_workers,
        persistent_workers=True,
        **dataloader_kwargs,
    )
    return ds, dl, cell_type_encoder
