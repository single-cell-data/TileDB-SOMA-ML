import pytorch_lightning as pl
import torch
from sklearn.preprocessing import LabelEncoder

from tiledbsoma_ml import ExperimentDataset


class LogisticRegression(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dataset: ExperimentDataset,
        cell_type_encoder: LabelEncoder,
        learning_rate: float = 1e-5,
    ):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.cell_type_encoder = cell_type_encoder
        self.learning_rate = learning_rate
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.epoch = 0
        self.dataset = dataset

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
        self.dataset.set_epoch(self.epoch)

    def on_train_epoch_end(self):
        self.epoch += 1
