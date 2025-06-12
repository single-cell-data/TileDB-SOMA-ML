#!/usr/bin/env python3
"""
Performance Comparison: Standard vs Optimized TileDB-SOMA-ML DataLoaders

This script runs both the standard (pre-optimization) and optimized dataloaders
side by side to compare their performance characteristics.

Key Differences:
- Standard: Returns tuples (X_batch, obs_batch), no threading optimizations
- Optimized: Returns dictionaries {'X': tensor, ...}, with threading optimizations
"""

import numpy as np
import torch
import time
import sys
import os
from typing import Dict, Any, Tuple
from tiledbsoma import Experiment, AxisQuery
from tiledbsoma_ml import ExperimentDataset, experiment_dataloader
from tiledbsoma_ml.dataloader import optimized_experiment_dataloader
import cellxgene_census
from cellxgene_census.experimental.pp import highly_variable_genes
import tiledbsoma as soma


def setup_environment_for_test(test_type: str):
    """Configure environment variables for each test type."""
    
    if test_type == "standard":
        print("Setting up environment for STANDARD dataloader test...")
        # Conservative settings for standard dataloader
        os.environ["TILEDB_VFS_S3_MAX_PARALLEL_OPS"] = "1"
        os.environ["TILEDB_VFS_S3_MULTIPART_PART_SIZE"] = "16777216"  # 16MB
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["NUMBA_NUM_THREADS"] = "1"
        if torch.cuda.is_available():
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
            torch.backends.cudnn.benchmark = False
            
    elif test_type == "optimized":
        print("Setting up environment for OPTIMIZED dataloader test...")
        # Optimized settings
        os.environ["TILEDB_VFS_S3_MAX_PARALLEL_OPS"] = "4"
        os.environ["TILEDB_VFS_S3_MULTIPART_PART_SIZE"] = "33554432"  # 32MB
        os.environ["OMP_NUM_THREADS"] = str(min(4, os.cpu_count() or 1))
        os.environ["MKL_NUM_THREADS"] = str(min(4, os.cpu_count() or 1))
        os.environ["NUMBA_NUM_THREADS"] = "1"  # Keep conservative for stability
        if torch.cuda.is_available():
            os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
            torch.backends.cudnn.benchmark = True


def create_standard_dataset(query) -> ExperimentDataset:
    """Create a dataset with standard (pre-optimization) settings."""
    return ExperimentDataset(
        query=query,
        layer_name="raw",
        batch_size=128,                   # Standard batch size
        io_batch_size=8192,               # Standard IO batch size
        shuffle=True,
        shuffle_chunk_size=64,            # Standard chunk size
        
        # No GPU optimizations
        pin_memory=False,
        prefetch_factor=1,
        use_cuda_streams=False,
        tensor_cache_size=1,
        use_eager_fetch=False,
        
        # No threading optimizations
        max_concurrent_requests=1,
        prefetch_queue_size=1,
        enable_pinned_memory=False,
        enable_tensor_cache=False,
        max_concurrent_batch_processing=1,
    )


def create_optimized_dataset(query) -> ExperimentDataset:
    """Create a dataset with optimized settings."""
    return ExperimentDataset(
        query=query,
        layer_name="raw",
        batch_size=128,                   # Same batch size for fair comparison
        io_batch_size=16384,              # Larger IO batch size
        shuffle=True,
        shuffle_chunk_size=128,           # Larger chunk size for better locality
        
        # GPU optimizations
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2,
        use_cuda_streams=torch.cuda.is_available(),
        tensor_cache_size=4,
        use_eager_fetch=True,
        
        # Threading optimizations
        max_concurrent_requests=2,
        prefetch_queue_size=2,
        enable_pinned_memory=torch.cuda.is_available(),
        enable_tensor_cache=True,
        max_concurrent_batch_processing=2,
    )


def benchmark_standard_dataloader(dataset: ExperimentDataset, num_batches: int = 50) -> Dict[str, Any]:
    """Benchmark the standard dataloader performance."""
    
    dataloader = experiment_dataloader(
        dataset,
        batch_size=64,
        num_workers=0,
        shuffle=False,
        pin_memory=False,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Warmup
    for i, (X_batch, obs_batch) in enumerate(dataloader):
        if i >= 3:
            break
        if torch.cuda.is_available() and torch.is_tensor(X_batch):
            X_tensor = X_batch.to(device)
    
    # Benchmark
    start_time = time.time()
    total_samples = 0
    batch_times = []
    
    for i, (X_batch, obs_batch) in enumerate(dataloader):
        if i >= num_batches:
            break
        
        batch_start = time.time()
        
        # Standard data processing
        if torch.cuda.is_available():
            if torch.is_tensor(X_batch):
                X_tensor = X_batch.to(device)
            elif isinstance(X_batch, np.ndarray):
                X_tensor = torch.from_numpy(X_batch).to(device)
            else:
                X_tensor = torch.from_numpy(X_batch.toarray() if hasattr(X_batch, 'toarray') else X_batch).to(device)
            
            result = torch.sum(X_tensor)
            torch.cuda.synchronize()
        else:
            if isinstance(X_batch, np.ndarray):
                X_tensor = torch.from_numpy(X_batch)
            elif hasattr(X_batch, 'toarray'):
                X_tensor = torch.from_numpy(X_batch.toarray())
            else:
                X_tensor = X_batch if torch.is_tensor(X_batch) else torch.tensor(X_batch)
            result = torch.sum(X_tensor)
        
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        total_samples += X_batch.shape[0]
    
    total_time = time.time() - start_time
    
    return {
        'type': 'standard',
        'total_time': total_time,
        'total_samples': total_samples,
        'throughput': total_samples / total_time,
        'avg_batch_time': np.mean(batch_times),
        'std_batch_time': np.std(batch_times),
        'batches_processed': len(batch_times)
    }


def benchmark_optimized_dataloader(dataset: ExperimentDataset, num_batches: int = 50) -> Dict[str, Any]:
    """Benchmark the optimized dataloader performance."""
    
    dataloader = optimized_experiment_dataloader(
        dataset,
        batch_size=64,
        max_concurrent_requests=2,
        max_concurrent_batch_processing=2,
        enable_pinned_memory=torch.cuda.is_available(),
        enable_tensor_cache=True,
        num_workers=0
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Warmup
    for i, batch in enumerate(dataloader):
        if i >= 3:
            break
        X_batch = batch['X']
        if torch.cuda.is_available() and torch.is_tensor(X_batch):
            X_tensor = X_batch.to(device)
    
    # Benchmark
    start_time = time.time()
    total_samples = 0
    batch_times = []
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        
        batch_start = time.time()
        
        # Optimized data processing
        X_batch = batch['X']
        
        if torch.cuda.is_available():
            if torch.is_tensor(X_batch):
                if X_batch.device != device:
                    X_tensor = X_batch.to(device, non_blocking=True)
                else:
                    X_tensor = X_batch
            elif isinstance(X_batch, np.ndarray):
                X_tensor = torch.from_numpy(X_batch).to(device)
            else:
                X_tensor = torch.from_numpy(X_batch.toarray() if hasattr(X_batch, 'toarray') else X_batch).to(device)
            
            with torch.cuda.stream(torch.cuda.current_stream()):
                result = torch.sum(X_tensor)
                torch.cuda.synchronize()
        else:
            if isinstance(X_batch, np.ndarray):
                X_tensor = torch.from_numpy(X_batch)
            elif hasattr(X_batch, 'toarray'):
                X_tensor = torch.from_numpy(X_batch.toarray())
            else:
                X_tensor = X_batch if torch.is_tensor(X_batch) else torch.tensor(X_batch)
            result = torch.sum(X_tensor)
        
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        total_samples += X_batch.shape[0]
    
    total_time = time.time() - start_time
    
    return {
        'type': 'optimized',
        'total_time': total_time,
        'total_samples': total_samples,
        'throughput': total_samples / total_time,
        'avg_batch_time': np.mean(batch_times),
        'std_batch_time': np.std(batch_times),
        'batches_processed': len(batch_times)
    }


def print_comparison_results(standard_results: Dict[str, Any], optimized_results: Dict[str, Any]):
    """Print a detailed comparison of the results."""
    
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON RESULTS")
    print("="*80)
    
    print(f"\n{'Metric':<30} {'Standard':<15} {'Optimized':<15} {'Improvement':<15}")
    print("-" * 75)
    
    # Throughput comparison
    standard_throughput = standard_results['throughput']
    optimized_throughput = optimized_results['throughput']
    throughput_improvement = optimized_throughput / standard_throughput
    
    print(f"{'Throughput (samples/sec)':<30} {standard_throughput:<15.1f} {optimized_throughput:<15.1f} {throughput_improvement:<15.2f}x")
    
    # Total time comparison
    standard_time = standard_results['total_time']
    optimized_time = optimized_results['total_time']
    time_improvement = standard_time / optimized_time
    
    print(f"{'Total Time (sec)':<30} {standard_time:<15.2f} {optimized_time:<15.2f} {time_improvement:<15.2f}x")
    
    # Average batch time comparison
    standard_batch_time = standard_results['avg_batch_time']
    optimized_batch_time = optimized_results['avg_batch_time']
    batch_time_improvement = standard_batch_time / optimized_batch_time
    
    print(f"{'Avg Batch Time (sec)':<30} {standard_batch_time:<15.4f} {optimized_batch_time:<15.4f} {batch_time_improvement:<15.2f}x")
    
    # Batch time variability
    standard_std = standard_results['std_batch_time']
    optimized_std = optimized_results['std_batch_time']
    
    print(f"{'Batch Time Std Dev':<30} {standard_std:<15.4f} {optimized_std:<15.4f} {'-':<15}")
    
    print(f"{'Samples Processed':<30} {standard_results['total_samples']:<15} {optimized_results['total_samples']:<15} {'-':<15}")
    print(f"{'Batches Processed':<30} {standard_results['batches_processed']:<15} {optimized_results['batches_processed']:<15} {'-':<15}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if throughput_improvement > 1.1:
        print(f"✅ Optimized dataloader is {throughput_improvement:.2f}x FASTER than standard dataloader")
    elif throughput_improvement > 0.9:
        print(f"⚖️  Performance is similar between standard and optimized dataloaders")
    else:
        print(f"⚠️  Standard dataloader is faster than optimized dataloader")
    
    print(f"\nKey Differences:")
    print(f"- Standard dataloader:  Single-threaded, no GPU optimizations, tuple format")
    print(f"- Optimized dataloader: Multi-threaded, GPU optimizations, dictionary format")
    print(f"- Hardware: {'GPU' if torch.cuda.is_available() else 'CPU'} processing")


def main():
    """Run the performance comparison."""
    
    print("TileDB-SOMA-ML DataLoader Performance Comparison")
    print("=" * 60)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU")
    
    try:
        import cellxgene_census
        
        # Open the census data
        with cellxgene_census.open_soma(census_version="2025-01-30") as census:
            experiment_name = "mus_musculus"
            obs_value_filter = 'is_primary_data == True and tissue_general in ["spleen", "kidney"] and nnz > 1000'
            top_n_hvg = 2000  # Smaller for faster testing
            hvg_batch = ["assay", "suspension_type"]
            
            print("Loading highly variable genes...")
            hvgs_df = highly_variable_genes(
                census["census_data"][experiment_name].axis_query(
                    measurement_name="RNA", obs_query=soma.AxisQuery(value_filter=obs_value_filter)
                ),
                n_top_genes=top_n_hvg,
                batch_key=hvg_batch,
            )
            hv = hvgs_df.highly_variable
            hv_idx = hv[hv].index
            
            # Create query
            hvg_query = census["census_data"][experiment_name].axis_query(
                measurement_name="RNA",
                obs_query=soma.AxisQuery(value_filter=obs_value_filter),
                var_query=soma.AxisQuery(coords=(list(hv_idx),)),
            )
            
            # Test 1: Standard Dataloader
            print("\n" + "-" * 60)
            print("TESTING STANDARD DATALOADER")
            print("-" * 60)
            
            setup_environment_for_test("standard")
            standard_dataset = create_standard_dataset(hvg_query)
            standard_results = benchmark_standard_dataloader(standard_dataset, num_batches=20)
            
            print(f"Standard throughput: {standard_results['throughput']:.1f} samples/sec")
            
            # Test 2: Optimized Dataloader
            print("\n" + "-" * 60)
            print("TESTING OPTIMIZED DATALOADER")
            print("-" * 60)
            
            setup_environment_for_test("optimized")
            optimized_dataset = create_optimized_dataset(hvg_query)
            optimized_results = benchmark_optimized_dataloader(optimized_dataset, num_batches=20)
            
            print(f"Optimized throughput: {optimized_results['throughput']:.1f} samples/sec")
            
            # Print comparison
            print_comparison_results(standard_results, optimized_results)
    
    except ImportError:
        print("\nERROR: cellxgene-census not available.")
        print("To run this comparison, install: pip install cellxgene-census")
        sys.exit(1)
    
    except Exception as e:
        print(f"\nERROR: {e}")
        print("There was an issue with the data loading. This could be due to:")
        print("1. Network connectivity issues")
        print("2. Missing dependencies")
        print("3. Authentication issues with S3")
        sys.exit(1)


if __name__ == "__main__":
    main() 