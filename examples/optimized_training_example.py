#!/usr/bin/env python3
"""
Example: High-Performance GPU Training with TileDB-SOMA-ML

This example demonstrates how to use the optimized TileDB-SOMA-ML features
for maximum performance when training on GPU with S3-backed data.

Note: The optimized dataloader returns batches as dictionaries with keys like 'X',
not as tuples (X_batch, obs_batch) like the standard implementation.
"""

import numpy as np
import torch
import time
from tiledbsoma import Experiment, AxisQuery
from tiledbsoma_ml import ExperimentDataset, optimized_experiment_dataloader
import cellxgene_census
from cellxgene_census.experimental.pp import highly_variable_genes
import tiledbsoma as soma


def setup_optimized_dataset(experiment_path: str, measurement_name: str = "RNA") -> ExperimentDataset:
    """Create an optimized ExperimentDataset for high-performance training."""
    
    # Note: The experiment context should be managed by the caller
    # This function assumes the experiment_path is accessible
    
    try:
        # Test if we can open the experiment
        
        census = cellxgene_census.open_soma(census_version="2025-01-30")
        experiment_name = "mus_musculus"
        obs_value_filter = 'is_primary_data == True and tissue_general in ["spleen", "kidney"] and nnz > 1000'
        top_n_hvg = 8000
        hvg_batch = ["assay", "suspension_type"]

        hvgs_df = highly_variable_genes(
            census["census_data"][experiment_name].axis_query(
                measurement_name="RNA", obs_query=soma.AxisQuery(value_filter=obs_value_filter)
            ),
            n_top_genes=top_n_hvg,
            batch_key=hvg_batch,
        )
        hv = hvgs_df.highly_variable
        hv_idx = hv[hv].index

        hvg_query = census["census_data"][experiment_name].axis_query(
            measurement_name="RNA",
            obs_query=soma.AxisQuery(value_filter=obs_value_filter),
            var_query=soma.AxisQuery(coords=(list(hv_idx),)),
        )

        # Open experiment and create query
        query = census["census_data"][experiment_name].axis_query(
            measurement_name=measurement_name,
            obs_query=AxisQuery(value_filter="is_primary_data == True")
        )
        
        # Create dataset with optimizations
        dataset = ExperimentDataset(
                    query=query,
                    layer_name="raw",
                    batch_size=512,                    # Larger batches for GPU efficiency
                    io_batch_size=65536,              # Reduced from 131072 for better memory management
                    shuffle=True,
                    shuffle_chunk_size=128,           # Balance randomness vs. locality
                    
                    # GPU optimizations (reduced for stability)
                    pin_memory=True,                  # Faster CPU-GPU transfers
                    prefetch_factor=2,                # Reduced prefetch for stability
                    use_cuda_streams=True,            # Overlap computation and transfer
                    tensor_cache_size=4,              # Reduced cache size
                    
                    # Enable optimized fetching
                    use_eager_fetch=True,
                )

        return dataset
            
    except Exception as e:
        print(f"Error opening experiment: {e}")
        print("Make sure the experiment path is correct and accessible.")
        print("For CELLxGENE Census data, you may need to install cellxgene-census:")
        print("pip install cellxgene-census")
        raise


def create_optimized_dataloader(dataset: ExperimentDataset, num_workers: int = 4):
    """Create an optimized DataLoader for maximum throughput."""
    
    return optimized_experiment_dataloader(
        dataset,
        num_workers=num_workers,          # Multiple workers for S3 parallelism
        persistent_workers=True,          # Keep workers alive between epochs
        prefetch_factor=2,                # PyTorch-level prefetching
        pin_memory=None,                  # Auto-detect based on CUDA availability
    )


def benchmark_performance(dataloader, num_batches: int = 100):
    """Benchmark the performance of the dataloader."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Warmup
    print("Warming up...")
    for i, batch in enumerate(dataloader):
        if i >= 5:  # 5 warmup batches
            break
        X_batch = batch['X']
        if torch.cuda.is_available() and torch.is_tensor(X_batch):
            X_tensor = X_batch.to(device)
    
    # Benchmark
    print(f"Benchmarking {num_batches} batches...")
    start_time = time.time()
    total_samples = 0
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        
        X_batch = batch['X']
        
        # Simulate GPU training
        if torch.cuda.is_available() and torch.is_tensor(X_batch):
            # Transfer to GPU (may already be on GPU with pinned memory)
            if X_batch.device != device:
                X_tensor = X_batch.to(device, non_blocking=True)
            else:
                X_tensor = X_batch
            
            # Simulate computation
            with torch.cuda.stream(torch.cuda.current_stream()):
                result = torch.sum(X_tensor)  # Simple computation
                torch.cuda.synchronize()  # Wait for completion
        elif isinstance(X_batch, np.ndarray):
            # Handle numpy arrays
            X_tensor = torch.from_numpy(X_batch).to(device)
            result = torch.sum(X_tensor)
        
        total_samples += X_batch.shape[0]
        
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            throughput = total_samples / elapsed
            print(f"Batch {i+1}: {throughput:.1f} samples/sec")
    
    total_time = time.time() - start_time
    final_throughput = total_samples / total_time
    
    print(f"\nFinal Results:")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Total samples: {total_samples}")
    print(f"Average throughput: {final_throughput:.1f} samples/sec")
    
    return final_throughput


def main():
    """Main training loop example."""
    
    # Use a public CELLxGENE Census example
    print("Loading data from CELLxGENE Census...")
    
    try:
        # Import cellxgene_census if available
        import cellxgene_census
        
        # Open the census data
        with cellxgene_census.open_soma(census_version="2025-01-30") as census:
            # Get mouse data
            experiment_name = "mus_musculus"
            obs_value_filter = 'is_primary_data == True and tissue_general in ["spleen", "kidney"] and nnz > 1000'
            top_n_hvg = 8000
            hvg_batch = ["assay", "suspension_type"]
            
            hvgs_df = highly_variable_genes(
                census["census_data"][experiment_name].axis_query(
                    measurement_name="RNA", obs_query=soma.AxisQuery(value_filter=obs_value_filter)
                ),
                n_top_genes=top_n_hvg,
                batch_key=hvg_batch,
            )
            hv = hvgs_df.highly_variable
            hv_idx = hv[hv].index

            # Create a smaller query for testing
            hvg_query = census["census_data"][experiment_name].axis_query(
                measurement_name="RNA",
                obs_query=soma.AxisQuery(value_filter=obs_value_filter),
                var_query=soma.AxisQuery(coords=(list(hv_idx),)),
            )
            # Create dataset with optimizations
            dataset = ExperimentDataset(
                query=hvg_query,
                layer_name="raw", 
                batch_size=128,              # Smaller batch size for stability
                io_batch_size=16384,         # Smaller IO batch size for safer memory usage
                shuffle=True,
                
                # GPU optimizations (conservative settings)
                pin_memory=torch.cuda.is_available(),
                prefetch_factor=1,           # Reduced to 1 for stability
                use_cuda_streams=torch.cuda.is_available(),
                tensor_cache_size=2,         # Reduced cache size
                use_eager_fetch=True,
            )
            
            print("Creating optimized dataloader...")
            dataloader = create_optimized_dataloader(dataset, num_workers=1)  # Conservative setting
            
            print("Starting performance benchmark...")
            throughput = benchmark_performance(dataloader, num_batches=10)  # Reduced batches
            
            print(f"\nOptimized throughput: {throughput:.1f} samples/sec")
            
            # Example training loop
            print("\nStarting training...")
            for epoch in range(2):  # Reduced epochs for testing
                dataset.set_epoch(epoch)
                
                epoch_start = time.time()
                batch_count = 0
                
                for batch in dataloader:
                    batch_count += 1
                    X_batch = batch['X']
                    
                    # You can access other data in the batch dict if needed
                    # obs_data = batch.get('soma_joinid', None)  # If available
                    
                    if batch_count >= 10:  # Limit for example
                        break
                
                epoch_time = time.time() - epoch_start
                print(f"Epoch {epoch + 1}: {batch_count} batches in {epoch_time:.2f}s")
    
    except ImportError:
        print("cellxgene-census not available. Using local experiment example.")
        print("To run with CELLxGENE Census data, install: pip install cellxgene-census")
        
        # Fall back to local experiment example
        experiment_path = "s3://your-bucket/experiment.soma"
        try:
            dataset = setup_optimized_dataset(experiment_path)
            dataloader = create_optimized_dataloader(dataset, num_workers=1)  # Conservative setting
            throughput = benchmark_performance(dataloader, num_batches=10)  # Reduced batches
            print(f"Optimized throughput: {throughput:.1f} samples/sec")
        except Exception as e:
            print(f"Error with local experiment: {e}")
            print("Please update experiment_path with your actual SOMA experiment.")
    
    except Exception as e:
        print(f"Error: {e}")
        print("There was an issue with the data loading. This could be due to:")
        print("1. Network connectivity issues")
        print("2. Missing dependencies")
        print("3. Authentication issues with S3")


if __name__ == "__main__":
    # Environment setup recommendations
    import os
    
    print("Setting optimal environment variables...")
    
    # TileDB optimizations
    os.environ.setdefault("TILEDB_VFS_S3_MAX_PARALLEL_OPS", "4")  # Reduced for stability
    os.environ.setdefault("TILEDB_VFS_S3_MULTIPART_PART_SIZE", "33554432")  # 32MB
    
    # PyTorch optimizations
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
        torch.backends.cudnn.benchmark = True
    else:
        print("CUDA not available, running on CPU")
    
    main()
 