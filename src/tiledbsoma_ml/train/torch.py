import os

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from tiledbsoma_ml.census import census_dataloader
from ..model.torch import LogisticRegression
from ..utils import err
import torch


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


def train(
    batch_size: int,
    use_gpu: bool | None,
    io_batch_size: int,
    learning_rate: float,
    n_epochs: int,
    model_in_path: str | None,
    model_out_path: str | None,
    shuffle_chunk_size: int,
    seed: int | None,
    tissue: str,
    census_version: str,
    num_workers: int | None,
    verbose: bool,  # TODO
):
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        ddp = True
    else:
        ddp = False
        local_rank = None

    ds, dl, cell_type_encoder = census_dataloader(
        tissue=tissue,
        batch_size=batch_size,
        io_batch_size=io_batch_size,
        shuffle_chunk_size=shuffle_chunk_size,
        census_version=census_version,
        num_workers=num_workers,
        seed=seed,
    )

    if use_gpu is True:
        if not torch.cuda.is_available():
            raise ValueError("GPU is not available")
        device = torch.device("cuda")
    elif use_gpu is False:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # The size of the input dimension is the number of genes
    input_dim = ds.shape[1]

    # The size of the output dimension is the number of distinct cell_type values
    output_dim = len(cell_type_encoder.classes_)

    if seed is not None:
        torch.manual_seed(seed)

    model = LogisticRegression(input_dim, output_dim).to(device)
    if model_in_path is not None:
        model.load_state_dict(torch.load(model_in_path, weights_only=True))
    if ddp:
        device_ids = None if local_rank is None else [local_rank]
        model = DDP(model, device_ids=device_ids)
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
        err(f"Epoch {epoch + 1}: Train Loss: {train_loss:.7f} Accuracy {train_accuracy:.4f}")
        epochs.append({
            'tissue': tissue,
            'seed': seed,
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
        })

    if not local_rank:  # 0 or None
        if model_out_path is not None:
            # Remove DDP wrapper before saving
            state_dict = model.state_dict() if local_rank is None else model.module.state_dict()
            torch.save(state_dict, model_out_path)

    if ddp:
        dist.destroy_process_group()
