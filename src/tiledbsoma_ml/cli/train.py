import os

import tiledbsoma as soma
import torch
from click import option
from sklearn.preprocessing import LabelEncoder

from tiledbsoma_ml import ExperimentAxisQueryIterableDataset, experiment_dataloader
from . import tdbsml

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ..model import LogisticRegression
from .base import DEFAULT_CENSUS_VERSION, DEFAULT_BATCH_SIZE, DEFAULT_N_EPOCHS, DEFAULT_LEARNING_RATE, \
    DEFAULT_IO_CHUNK_SIZE, DEFAULT_SHUFFLE_CHUNK_SIZE


def train_epoch(model, train_dataloader, loss_fn, optimizer, device, cell_type_encoder):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for X_batch, y_batch in train_dataloader:
        optimizer.zero_grad()

        X_batch = torch.from_numpy(X_batch).float().to(device)

        # Perform prediction
        outputs = model(X_batch)

        # Determine the predicted label
        probabilities = torch.nn.functional.softmax(outputs, 1)
        predictions = torch.argmax(probabilities, axis=1)

        # Compute the loss and perform back propagation
        y_batch = torch.from_numpy(cell_type_encoder.transform(y_batch['cell_type'])).to(device)
        train_correct += (predictions == y_batch).sum().item()
        train_total += len(predictions)

        loss = loss_fn(outputs, y_batch.long())
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    train_loss /= train_total
    train_accuracy = train_correct / train_total
    return train_loss, train_accuracy


def dataloader(
    tissue: str,
    batch_size: int,
    io_batch_size: int,
    shuffle_chunk_size: int,
    census_version: str,
    num_workers: int | None,
    seed: int | None,
):
    census_url = f"s3://cellxgene-census-public-us-west-2/cell-census/{census_version}/soma/census_data/homo_sapiens/"
    experiment = soma.open(
        census_url,
        context=soma.SOMATileDBContext(tiledb_config={"vfs.s3.region": "us-west-2"}),
    )
    obs_value_filter = f"tissue_general == '{tissue}' and is_primary_data == True"
    with experiment.axis_query(
            measurement_name="RNA",
            obs_query=soma.AxisQuery(value_filter=obs_value_filter)
    ) as query:
        obs_df = query.obs(column_names=["cell_type"]).concat().to_pandas()
        cell_type_encoder = LabelEncoder().fit(obs_df["cell_type"].unique())

        ds = ExperimentAxisQueryIterableDataset(
            query,
            X_name="raw",
            obs_column_names=["cell_type"],
            batch_size=batch_size,
            io_batch_size=io_batch_size,
            shuffle_chunk_size=shuffle_chunk_size,
            seed=seed,
        )

    dl = experiment_dataloader(ds, num_workers=num_workers)
    return ds, dl, cell_type_encoder


@tdbsml.command
@option('-b', '--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
@option('-i', '--io-batch-size', type=int, default=DEFAULT_IO_CHUNK_SIZE)
@option('-I', '--model-in-path', help='Load initial model from this path')
@option('-l', '--learning-rate', type=float, default=DEFAULT_LEARNING_RATE)
@option('-m', '--model-path', help='Load model from this path, and save it back to this path; equivalent to setting both `-I/--model-in-path` and `-O/--model-out-path` to this value')
@option('-n', '--n-epochs', type=int, default=DEFAULT_N_EPOCHS)
@option('-O', '--model-out-path', help='Save trained model to this path')
@option('-s', '--shuffle-chunk-size', type=int, default=DEFAULT_SHUFFLE_CHUNK_SIZE)
@option('-S', '--seed', type=int)
@option('-t', '--tissue', help='"tissue_general" obs filter')
@option('-v', '--census-version', default=DEFAULT_CENSUS_VERSION)
@option('-w', '--num-workers', type=int)
def train(
    batch_size: int,
    io_batch_size: int,
    model_in_path: str | None,
    learning_rate: float,
    model_path: str | None,
    n_epochs: int,
    model_out_path: str | None,
    shuffle_chunk_size: int,
    seed: int | None,
    tissue: str,
    census_version: str,
    num_workers: int | None,
):
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if num_workers is None:
        num_workers = torch.cuda.device_count()

    ds, dl, cell_type_encoder = dataloader(
        tissue=tissue,
        batch_size=batch_size,
        io_batch_size=io_batch_size,
        shuffle_chunk_size=shuffle_chunk_size,
        census_version=census_version,
        num_workers=num_workers,
        seed=seed,
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # The size of the input dimension is the number of genes
    input_dim = ds.shape[1]

    # The size of the output dimension is the number of distinct cell_type values
    output_dim = len(cell_type_encoder.classes_)

    model = LogisticRegression(input_dim, output_dim).to(device)
    model_in_path = model_in_path or model_path
    if model_in_path is not None:
        model.load_state_dict(torch.load(model_in_path))
    model = DDP(model, device_ids=[local_rank])
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epochs = []
    for epoch in range(n_epochs):
        ds.set_epoch(epoch)
        train_loss, train_accuracy = train_epoch(
            model=model,
            train_dataloader=dl,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            cell_type_encoder=cell_type_encoder,
        )
        print(
            f"Epoch {epoch + 1}: Train Loss: {train_loss:.7f} Accuracy {train_accuracy:.4f}"
        )
        epochs.append({
            'tissue': tissue,
            'seed': seed,
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
        })

    if local_rank == 0:
        model_out_path = model_out_path or model_path
        if model_out_path is not None:
            # Remove DDP wrapper before saving
            torch.save(model.module.state_dict(), model_out_path)

    dist.destroy_process_group()
