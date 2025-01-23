import logging
from functools import partial
from warnings import filterwarnings

import pytorch_lightning as pl
import torch
from sklearn.preprocessing import LabelEncoder
from somacore import AxisQuery
from tiledbsoma import SOMATileDBContext, Experiment

from tiledbsoma_ml import ExperimentAxisQueryIterDataPipe, experiment_dataloader
from tiledbsoma_ml.census import census_dataloader
from tiledbsoma_ml.model.lightning import LogisticRegression


def setup_logging(worker_id: int | None = None, verbose: bool = True):
    logging.basicConfig(
        format="%(asctime)s %(process)-7s %(levelname)-8s %(message)s",
        level=logging.DEBUG if verbose else logging.WARNING,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.captureWarnings(True)
    logging.getLogger("numba").setLevel(logging.WARNING)
    if worker_id is not None:
        logging.debug(f"DataLoader worker {worker_id} started")


def train(
    batch_size: int,
    use_gpu: bool | None,
    io_batch_size: int,
    learning_rate: float,
    n_epochs: int,
    shuffle_chunk_size: int,
    seed: int | None,
    tissue: str,
    census_version: str,
    num_workers: int | None,
    use_iterdatapipe: bool,
    verbose: bool,
    model_in_path: str | None,
    model_out_path: str | None,
):
    torch.set_float32_matmul_precision("high")
    if use_gpu is True:
        if not torch.cuda.is_available():
            raise ValueError("GPU is not available")
        accelerator = "gpu"
    elif use_gpu is False:
        accelerator = "cpu"
    else:
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    if accelerator == "cpu":
        filterwarnings(
            "ignore", category=UserWarning, module="torch.cuda", lineno=654
        )

    ds, dl, cell_type_encoder = census_dataloader(
        tissue=tissue,
        batch_size=batch_size,
        io_batch_size=io_batch_size,
        shuffle_chunk_size=shuffle_chunk_size,
        census_version=census_version,
        num_workers=num_workers,
        seed=seed,
        use_iterdatapipe=use_iterdatapipe,
        worker_init_fn=partial(setup_logging, verbose=verbose)
    )

    # The size of the input dimension is the number of genes
    input_dim = ds.shape[1]

    # The size of the output dimension is the number of distinct cell_type values
    output_dim = len(cell_type_encoder.classes_)

    # Initialize the PyTorch Lightning model
    model = LogisticRegression(
        input_dim,
        output_dim,
        cell_type_encoder=cell_type_encoder,
        datapipe=ds,
        learning_rate=learning_rate,
    )
    if model_in_path is not None:
        model.load_state_dict(torch.load(model_in_path))

    # Define the PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=n_epochs,
        accelerator=accelerator,
        strategy="ddp",
    )

    # Train the model
    trainer.fit(model, train_dataloaders=dl)
