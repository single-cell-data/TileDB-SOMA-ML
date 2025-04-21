import pandas as pd
import torch
from click import option, argument

from . import tdbsml
from .train import dataloader
from .. import experiment_dataloader
from ..model import LogisticRegression


from .base import DEFAULT_CENSUS_VERSION, DEFAULT_BATCH_SIZE, DEFAULT_N_EPOCHS, DEFAULT_LEARNING_RATE, DEFAULT_IO_CHUNK_SIZE, DEFAULT_SHUFFLE_CHUNK_SIZE


@tdbsml.command
@option('-b', '--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
@option('-i', '--io-batch-size', type=int, default=DEFAULT_IO_CHUNK_SIZE)
@option('-n', '--n-batches', type=int, default=1, help='Number of batches to test')
@option('-s', '--shuffle-chunk-size', type=int, default=DEFAULT_SHUFFLE_CHUNK_SIZE)
@option('-S', '--seed', type=int)
@option('-t', '--tissue', help='"tissue_general" obs filter')
@option('-v', '--census-version', default=DEFAULT_CENSUS_VERSION)
@option('-w', '--num-workers', type=int, default=2)
@argument('model_path')
def test(
    batch_size: int,
    io_batch_size: int,
    n_batches: int,
    shuffle_chunk_size: int,
    seed: int | None,
    tissue: str,
    census_version: str,
    num_workers: int | None,
    model_path: str,
):
    ds, dl, cell_type_encoder = dataloader(
        tissue=tissue,
        batch_size=batch_size,
        io_batch_size=io_batch_size,
        shuffle_chunk_size=shuffle_chunk_size,
        census_version=census_version,
        num_workers=num_workers,
        seed=seed,
    )

    # The size of the input dimension is the number of genes
    input_dim = ds.shape[1]

    # The size of the output dimension is the number of distinct cell_type values
    output_dim = len(cell_type_encoder.classes_)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = LogisticRegression(input_dim, output_dim).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)

    test_dataloader = experiment_dataloader(ds)
    batch_dfs = []
    for batch_idx in range(n_batches):
        X_batch, y_batch = next(iter(test_dataloader))
        X_batch = torch.from_numpy(X_batch)
        y_batch = torch.from_numpy(cell_type_encoder.transform(y_batch['cell_type']))
        outputs = model(X_batch.to(device))
        probabilities = torch.nn.functional.softmax(outputs, 1)
        predictions = torch.argmax(probabilities, axis=1)
        predicted = pd.Series(cell_type_encoder.inverse_transform(predictions.cpu()))
        actual = pd.Series(cell_type_encoder.inverse_transform(y_batch.ravel().numpy()))
        batch_df = pd.DataFrame(dict(actual=actual, predicted=predicted))
        batch_dfs.append(batch_df)

    df = pd.concat(batch_dfs)
    # ct = pd.crosstab(df.actual, df.predicted).replace(0, "")
    # print(ct)
    n = len(df)
    n_eq = (df.actual == df.predicted).sum()
    print(f"{n_eq}/{n} = {n_eq/n:.2%}")
