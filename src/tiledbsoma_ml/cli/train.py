import logging
from os.path import exists
from typing import Tuple

from click import option
from torch.ao.quantization.fx.utils import node_arg_is_bias

from . import sml
from .base import (
    DEFAULT_N_EPOCHS,
    DEFAULT_LEARNING_RATE,
    shuffle_chunk_size_opt,
    seed_opt,
    tissue_opt,
    verbose_opt,
    census_version_opt,
    num_workers_opt
)
from .base import batch_size_opt, io_batch_size_opt
from ..train import lightning, torch

LOG_LEVEL_ABBREVS = {
    'd': 'debug',
    'i': 'info',
    'w': 'warn',
    'e': 'error',
}


@sml.command(no_args_is_help=True)
@batch_size_opt
@option('-g/-G', '--use-gpu/--no-use-gpu', default=None, help="Specify using GPU vs. CPU; default: infer from `torch.cuda.is_available()`")
@io_batch_size_opt
@option('-I', '--model-in-path', help='Load initial model from this path')
@option('-l', '--use-lightning', is_flag=True, help="Use PyTorch Lightning")
@option('-L', '--log', 'log_strs', multiple=True, help="Logging configs, e.g. io_batches=debug")
@option('-M', '--model-path', help="Load model from this path (if one exists), and save it back to this path; equivalent to setting both `-I/--model-in-path` and `-O/--model-out-path` to this value, but will create a new model if one doesn't exist")
@option('-n', '--n-epochs', type=int, default=DEFAULT_N_EPOCHS, help=f"Number of epochs to train for (default: {DEFAULT_N_EPOCHS})")
@option('-O', '--model-out-path', help='Save trained model to this path')
@option('-r', '--learning-rate', type=float, default=DEFAULT_LEARNING_RATE, help=f"Training learning rate (default: {DEFAULT_LEARNING_RATE})")
@shuffle_chunk_size_opt
@seed_opt
@tissue_opt
@verbose_opt
@census_version_opt
@num_workers_opt
def train(
    *args,
    log_strs: Tuple[str, ...],
    model_in_path: str | None,
    model_path: str | None,
    model_out_path: str | None,
    use_lightning: bool,
    **kwargs,
):
    for log_str in log_strs:
        k, v = log_str.split('=', 1)
        v = LOG_LEVEL_ABBREVS.get(v.lower(), v)
        logging.getLogger(f'tiledbsoma_ml.{k}').setLevel(v.upper())
    kwargs['model_in_path'] = model_in_path or (model_path if model_path and exists(model_path) else None)
    kwargs['model_out_path'] = model_out_path or model_path
    if use_lightning:
        lightning.train(*args, **kwargs)
    else:
        torch.train(*args, **kwargs)
