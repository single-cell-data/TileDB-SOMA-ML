from click import group, option

from tiledbsoma_ml.dataset import DEFAULT_IO_BATCH_SIZE, DEFAULT_SHUFFLE_CHUNK_SIZE

DEFAULT_CENSUS_VERSION = '2024-07-01'
DEFAULT_BATCH_SIZE = 128
DEFAULT_N_EPOCHS = 20
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_N_WORKERS = 2


batch_size_opt = option('-b', '--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help=f"Size (in rows) of batches sent to GPU (default: {DEFAULT_BATCH_SIZE})")
io_batch_size_opt = option('-i', '--io-batch-size', type=int, default=DEFAULT_IO_BATCH_SIZE, help=f"Batch sizes to fetch from Census (default: {DEFAULT_IO_BATCH_SIZE})")
shuffle_chunk_size_opt = option('-s', '--shuffle-chunk-size', type=int, default=DEFAULT_SHUFFLE_CHUNK_SIZE, help=f"Chunk sizes used for shuffling (default: {DEFAULT_SHUFFLE_CHUNK_SIZE})")
seed_opt = option('-S', '--seed', type=int, help="Seed for various RNGs")
tissue_opt = option('-t', '--tissue', required=True, help='"tissue_general" obs filter')
verbose_opt = option('-v', '--verbose', is_flag=True, help="Verbose logging")
census_version_opt = option('-V', '--census-version', default=DEFAULT_CENSUS_VERSION, help=f"Census version to use (default: {DEFAULT_CENSUS_VERSION})")
num_workers_opt = option('-w', '--num-workers', type=int, default=DEFAULT_N_WORKERS, help=f"Number of DataLoader workers to use (default: {DEFAULT_N_WORKERS})")


@group
def sml():
    pass
