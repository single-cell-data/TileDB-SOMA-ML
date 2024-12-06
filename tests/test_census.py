import tiledbsoma as soma
import torch
from sklearn.preprocessing import LabelEncoder

from tiledbsoma_ml import ExperimentAxisQueryIterDataPipe, ExperimentAxisQueryIterableDataset, experiment_dataloader

CZI_Census_Homo_Sapiens_URL = f"s3://cellxgene-census-public-us-west-2/cell-census/{census_version}/soma/census_data/homo_sapiens/"

def test_census_training():
    workers = None
    tissue = "ovary"       # "tissue_general" obs filter
    census_version = "2024-07-01"
    batch_size = 128
    learning_rate = 1e-5
    n_epochs = 20

    experiment = soma.open(
        CZI_Census_Homo_Sapiens_URL,
        context=soma.SOMATileDBContext(tiledb_config={"vfs.s3.region": "us-west-2"}),
    )
    obs_value_filter = f"tissue_general == '{tissue}' and is_primary_data == True"
    with experiment.axis_query(
        measurement_name="RNA",
        obs_query=soma.AxisQuery(value_filter=obs_value_filter)
    ) as query:
        obs_df = query.obs(column_names=["cell_type"]).concat().to_pandas()
        cell_type_encoder = LabelEncoder().fit(obs_df["cell_type"].unique())

        experiment_dataset = ExperimentAxisQueryIterDataPipe(
            query,
            X_name="raw",
            obs_column_names=["cell_type"],
            batch_size=batch_size,
        )

    assert len(obs_df) == 53751
    assert len(experiment_dataset) == 1420

    ei = experiment_dataset._exp_iter
    ois = ei._obs_joinids
    assert ois.shape == (53751,)

    opi = list(ei._create_obs_joinids_partition())
    assert len(opi) == 1

    assert (sorted(opi[0]) == ois).all()
