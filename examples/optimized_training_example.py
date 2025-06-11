#!/usr/bin/env python3
"""
Example: High-Performance GPU Training with TileDB-SOMA-ML

This example demonstrates how to use the optimized TileDB-SOMA-ML features
for maximum performance when training on GPU with S3-backed data.
"""

import torch
import time
from tiledbsoma import Experiment, AxisQuery
from tiledbsoma_ml import ExperimentDataset, optimized_experiment_dataloader


def setup_optimized_dataset(experiment_path: str, measurement_name: str = "RNA") -> ExperimentDataset:
    """Create an optimized ExperimentDataset for high-performance training."""
    
    # Open experiment and create query
    exp = Experiment.open(experiment_path)
    query = exp.axis_query(
        measurement_name=measurement_name,
        obs_query=AxisQuery(value_filter="is_primary_data == True")
    )
    
    # Create dataset with performance optimizations
    dataset = ExperimentDataset(
        query=query,
        layer_name="raw",
        batch_size=512,                    # Larger batches for GPU efficiency
        io_batch_size=131072,             # Large IO batches for S3 performance
        shuffle=True,
        shuffle_chunk_size=128,           # Balance randomness vs. locality
        
        # GPU optimizations
        pin_memory=True,                  # Faster CPU-GPU transfers
        prefetch_factor=3,                # Prefetch 3 batches ahead
        use_cuda_streams=True,            # Overlap computation and transfer
        tensor_cache_size=6,              # Cache 6 different tensor sizes
        
        # Enable optimized fetching
        use_eager_fetch=True,
    )
    
    return dataset


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
    for i, (X_batch, obs_batch) in enumerate(dataloader):
        if i >= 5:  # 5 warmup batches
            break
        if torch.cuda.is_available():
            X_tensor = torch.from_numpy(X_batch).to(device)
    
    # Benchmark
    print(f"Benchmarking {num_batches} batches...")
    start_time = time.time()
    total_samples = 0
    
    for i, (X_batch, obs_batch) in enumerate(dataloader):
        if i >= num_batches:
            break
            
        # Simulate GPU training
        if torch.cuda.is_available():
            # Transfer to GPU
            X_tensor = torch.from_numpy(X_batch).to(device, non_blocking=True)
            
            # Simulate computation
            with torch.cuda.stream(torch.cuda.current_stream()):
                result = torch.sum(X_tensor)  # Simple computation
                torch.cuda.synchronize()  # Wait for completion
        
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
    
    # Example experiment path (replace with your actual path)
    experiment_path = "s3://your-bucket/experiment.soma"
    
    try:
        print("Setting up optimized dataset...")
        dataset = setup_optimized_dataset(experiment_path)
        
        print("Creating optimized dataloader...")
        dataloader = create_optimized_dataloader(dataset, num_workers=4)
        
        print("Starting performance benchmark...")
        throughput = benchmark_performance(dataloader, num_batches=50)
        
        print(f"\nOptimized throughput: {throughput:.1f} samples/sec")
        
        # Example training loop
        print("\nStarting training...")
        for epoch in range(3):
            dataset.set_epoch(epoch)  # Important for proper shuffling
            
            epoch_start = time.time()
            batch_count = 0
            
            for X_batch, obs_batch in dataloader:
                # Your training code here
                batch_count += 1
                
                if batch_count >= 20:  # Limit for example
                    break
            
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch + 1}: {batch_count} batches in {epoch_time:.2f}s")
    
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to update experiment_path with your actual SOMA experiment.")


if __name__ == "__main__":
    # Environment setup recommendations
    import os
    
    print("Setting optimal environment variables...")
    
    # TileDB optimizations
    os.environ.setdefault("TILEDB_VFS_S3_MAX_PARALLEL_OPS", "8")
    os.environ.setdefault("TILEDB_VFS_S3_MULTIPART_PART_SIZE", "67108864")  # 64MB
    
    # PyTorch optimizations
    if torch.cuda.is_available():
        os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
 