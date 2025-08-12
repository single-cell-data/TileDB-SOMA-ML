import argparse
from lightning.pytorch.callbacks import Callback
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(
        description="Train SCVI with custom IO/shuffle settings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--io-batch-size", type=int, default=65536,
                        help="Rows read from disk per IO batch")
    parser.add_argument("--shuffle-chunk-size", type=int, default=64,
                        help="Row‑chunk size the loader shuffles")
    # you can expose any other hyper‑param the same way:
    return parser.parse_args()

def main():
    import warnings
    from typing import Any, Dict, List
    import torch
    import cellxgene_census
    import pandas as pd
    import scanpy as sc
    import scvi
    import tiledbsoma as soma
    from tiledbsoma_ml import SCVIDataModule
    import torch
    from cellxgene_census.experimental.pp import highly_variable_genes
    from lightning import LightningDataModule
    from sklearn.preprocessing import LabelEncoder
    from torch.utils.data import DataLoader
    

    from poller import GPUUtilSampler
    import uuid, time, json

    

    # import logging
    # logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(message)s")
    # logging.getLogger("tiledbsoma_ml.dataset").setLevel(logging.DEBUG)

    import time
    from torch.profiler import schedule, ProfilerActivity, tensorboard_trace_handler
    from pytorch_lightning.profilers import PyTorchProfiler
    from lightning.pytorch.callbacks import DeviceStatsMonitor
    from lightning.pytorch.loggers import TensorBoardLogger


    import os, torch

    # trace_dir = "./log/pt_prof"          # folder per run
    # os.makedirs(trace_dir, exist_ok=True)

    # prof_sched = schedule(
    # wait=2,         # warm‑up steps (skipped)
    # warmup=3,       # record but don't report
    # active=5,       # report these steps
    # repeat=1,       # how many cycles
    # )
    
    args = get_args() 

    run_id = f"run_{uuid.uuid4().hex[:6]}"
    run_dir = f"./runs/{args.io_batch_size}_{args.shuffle_chunk_size}_{run_id}"

    t0 = time.perf_counter()
    warnings.filterwarnings("ignore")

    experiment_name = "mus_musculus"
    obs_value_filter = 'is_primary_data == True and tissue_general in ["pancreas", "kidney"] and nnz >= 300'
    top_n_hvg = 8000
    hvg_batch = ["assay", "suspension_type"]

    LOCAL_URI = "/home/ec2-user/new/mm_pan_kidney_subset_soma"       # same path you wrote above
    # census = cellxgene_census.open_soma(uri=LOCAL_URI)
    exp = soma.open(LOCAL_URI, mode="r")


    query = exp.axis_query(
        measurement_name="RNA",
        obs_query=soma.AxisQuery(value_filter=obs_value_filter),
    )
    print(f"{query.n_obs} obs")
    query.obs(column_names=['dataset_id']).concat().to_pandas().dataset_id.astype(str).value_counts()
    print(exp.ms["RNA"].X.keys()) 

    hvgs_df = highly_variable_genes(
        query,
        n_top_genes=top_n_hvg,
        batch_key=hvg_batch,
        layer="data"
    )
    hv = hvgs_df.highly_variable
    hv_idx = hv[hv].index


    hvg_query = exp.axis_query(
        measurement_name="RNA",
        obs_query=soma.AxisQuery(value_filter=obs_value_filter),
        var_query=soma.AxisQuery(coords=(list(hv_idx),)),
    )

    datamodule = SCVIDataModule(
        hvg_query,
        layer_name="data",
        batch_size=1024,
        shuffle=False,
        shuffle_on_gpu=True,
        device=torch.device('cuda', 0),
        use_eager_fetch=True,
        seed=42,
        data_on_disc=True,
        dataloader_kwargs={"num_workers": 8, "persistent_workers": True, "pin_memory": True},
        io_batch_size=args.io_batch_size, 
        shuffle_chunk_size=args.shuffle_chunk_size,
    )

    (datamodule.n_obs, datamodule.n_vars, datamodule.n_batch)


    n_layers = 1
    n_latent = 50

    model = scvi.model.SCVI(
        n_layers=n_layers,
        n_latent=n_latent,
        gene_likelihood="nb",
        encode_covariates=False,
    )

    sampler = GPUUtilSampler(
        run_meta = {
            "run_id": run_id,
            "batch_size": 1024,
            "dataset_uri": LOCAL_URI,
            "n_layers": 1,
            "n_latent": 50,
            "io_batch_size":args.io_batch_size, 
            "shuffle_chunk_size":args.shuffle_chunk_size,
        },
        out_dir = run_dir,
        interval = 0.1,
    )
    sampler.start()
    
    model.train(
        datamodule=datamodule,
        max_epochs=2,
        early_stopping=False,
        devices=1,
        strategy="ddp_find_unused_parameters_true",
    )

    sampler.flush()


    print("The time: ", time.perf_counter() - t0)


if __name__ == "__main__":
    main()