"""
mamba create -n toymodel python=3.11
mamba activate toymodel
mamba install lightning pytorch torchvision torchaudio pytorch-cuda=12.4 'numpy<2.0' ipython  -c pytorch -c nvidia
pip install scikit-learn tiledbsoma typed-argument-parser
"""

from __future__ import annotations

import logging
import sys
import warnings
from functools import partial
from typing import Literal

import pytorch_lightning as pl
import torch
from sklearn.preprocessing import LabelEncoder
from tap import Tap

import tiledbsoma as soma

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    import tiledbsoma_ml as soma_ml


class Arguments(Tap):
    n_workers: int = 0
    accelerator: Literal["gpu", "cpu"] = "gpu"
    max_epochs: int = 10
    verbose: bool = False


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


class LogisticRegressionLightning(pl.LightningModule):
    def __init__(self, input_dim, output_dim, datapipe, cell_type_encoder, learning_rate=1e-5):
        super(LogisticRegressionLightning, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.cell_type_encoder = cell_type_encoder
        self.learning_rate = learning_rate
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.epoch = 0
        self.datapipe = datapipe

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs

    def training_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        # X_batch = X_batch.float()
        X_batch = torch.from_numpy(X_batch).float().to(self.device)

        # Perform prediction
        outputs = self(X_batch)

        # Determine the predicted label
        probabilities = torch.nn.functional.softmax(outputs, 1)
        predictions = torch.argmax(probabilities, axis=1)

        # Compute loss
        # y_batch = y_batch.flatten()
        y_batch = torch.from_numpy(
            self.cell_type_encoder.transform(y_batch["cell_type"])
        ).to(self.device)
        loss = self.loss_fn(outputs, y_batch.long())

        # Compute accuracy
        train_correct = (predictions == y_batch).sum().item()
        train_accuracy = train_correct / len(predictions)

        # Log loss and accuracy
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_accuracy", train_accuracy, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def on_train_epoch_start(self):
        self.datapipe.set_epoch(self.epoch)

    def on_train_epoch_end(self):
        self.epoch += 1


def train(args):

    torch.set_float32_matmul_precision("high")

    if args.accelerator == "cpu":
        warnings.filterwarnings(
            "ignore", category=UserWarning, module="torch.cuda", lineno=654
        )

    CZI_Census_Homo_Sapiens_URL = (
        "s3://tiledb-bruce/cell-census/2024-07-01/soma/census_data/homo_sapiens/"
    )
    experiment = soma.open(
        CZI_Census_Homo_Sapiens_URL,
        context=soma.SOMATileDBContext(tiledb_config={"vfs.s3.region": "us-west-2"}),
    )
    obs_value_filter = "tissue_general == 'bone marrow' and is_primary_data == True"
    # obs_value_filter = "tissue_general == 'lung' and is_primary_data == True"

    with experiment.axis_query(
        measurement_name="RNA", obs_query=soma.AxisQuery(value_filter=obs_value_filter)
    ) as query:
        obs_df = query.obs(column_names=["cell_type"]).concat().to_pandas()
        cell_type_encoder = LabelEncoder().fit(obs_df["cell_type"].unique())

        experiment_datapipe = soma_ml.ExperimentAxisQueryIterDataPipe(
            query,
            X_name="raw",
            obs_column_names=["soma_joinid", "cell_type"],
            batch_size=128,
            shuffle=True,
        )

    # train_datapipe, _test_datapipe = experiment_datapipe.random_split(
    #     weights={"train": 0.8, "test": 0.2}, seed=1
    # )
    train_datapipe = experiment_datapipe
    train_dataloader = soma_ml.experiment_dataloader(
        train_datapipe,
        num_workers=args.n_workers,
        persistent_workers=True,
        worker_init_fn=partial(setup_logging, verbose=args.verbose),
    )
    # val_dataloader = soma_ml.experiment_dataloader(test_datapipe, num_workers=2)

    # The size of the input dimension is the number of genes
    input_dim = experiment_datapipe.shape[1]

    # The size of the output dimension is the number of distinct cell_type values
    output_dim = len(cell_type_encoder.classes_)

    # Initialize the PyTorch Lightning model
    model = LogisticRegressionLightning(
        input_dim, output_dim, cell_type_encoder=cell_type_encoder, datapipe=train_datapipe
    )

    # Define the PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        strategy="ddp",
    )

    # Train the model
    trainer.fit(model, train_dataloaders=train_dataloader)


if __name__ == "__main__":
    args = Arguments().parse_args()
    setup_logging(worker_id=None, verbose=args.verbose)
    torch.multiprocessing.set_start_method("spawn", force=True)
    sys.exit(train(args))
