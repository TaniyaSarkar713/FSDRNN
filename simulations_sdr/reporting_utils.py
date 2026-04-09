"""
Utility functions for setup reporting: GPU info, timing, aggregate statistics.
Used by all setup scripts for consistent formatting and analysis.
"""

import os
import socket
import subprocess
import numpy as np
import torch
import time
from datetime import datetime
from collections import defaultdict


def get_gpu_info():
    """Get GPU and CUDA information."""
    info = {}
    
    # Host info
    info['hostname'] = socket.gethostname()
    
    # CUDA/GPU info
    info['cuda_available'] = torch.cuda.is_available()
    info['cuda_version'] = torch.version.cuda if torch.cuda.is_available() else 'N/A'
    info['torch_version'] = torch.__version__
    info['cuda_visible_devices'] = os.environ.get('CUDA_VISIBLE_DEVICES', 'all')
    
    if torch.cuda.is_available():
        info['num_gpus'] = torch.cuda.device_count()
        info['current_device'] = torch.cuda.current_device()
        
        # GPU names
        gpu_names = []
        for i in range(info['num_gpus']):
            gpu_names.append(torch.cuda.get_device_name(i))
        info['gpu_names'] = gpu_names
    else:
        info['gpu_names'] = []
    
    return info


def print_system_info(setup_name, task_id=None, base_seed=None, n_reps=None):
    """Print header with system, GPU, and setup information."""
    gpu_info = get_gpu_info()
    
    print("=" * 60)
    print("=== host ===")
    print(gpu_info['hostname'])
    
    if gpu_info['cuda_available'] and gpu_info.get('gpu_names'):
        print("=== visible gpu(s) ===")
        print(f"CUDA_VISIBLE_DEVICES={gpu_info['cuda_visible_devices']}")
        for i, name in enumerate(gpu_info['gpu_names']):
            print(f"GPU {i}: {name}")
    
    print(f"torch: {gpu_info['torch_version']}")
    print(f"cuda available: {gpu_info['cuda_available']}")
    if gpu_info['cuda_available']:
        print(f"CUDA_VISIBLE_DEVICES: {gpu_info['cuda_visible_devices']}")
        if gpu_info.get('gpu_names'):
            print(f"gpu: {gpu_info['gpu_names'][gpu_info['current_device']]}")
    print("=" * 60)
    
    # Setup info
    print(f"Setup: {setup_name}")
    if task_id is not None:
        print(f"Task ID: {task_id}")
    if base_seed is not None:
        print(f"Base seed: {base_seed}")
    if n_reps is not None:
        print(f"Number of repetitions: {n_reps}")
    print("=" * 60)
    print(f"Running: simulations_sdr/{setup_name}.py with {n_reps} independent runs")
    print()


def aggregate_results(all_results, n_reps, metrics=None):
    """
    Aggregate results across repetitions.
    
    Args:
        all_results: list of dicts from each repetition
        n_reps: number of repetitions
        metrics: list of metric keys to aggregate (default: all in first result)
    
    Returns:
        dict with aggregated statistics
    """
    if not all_results:
        return {}
    
    if metrics is None:
        # Get all method names from first result
        metrics = list(all_results[0].get('methods', {}).keys())
    
    aggregated = {}
    
    for metric in metrics:
        values = []
        train_values = []
        gap_values = []
        times = []
        
        for rep_result in all_results:
            methods = rep_result.get('methods', {})
            if metric in methods:
                method_data = methods[metric]
                
                # Find MSE-like key (various names: mse, kl_divergence, angular_error, etc.)
                loss_key = None
                for key in method_data.keys():
                    if key not in ['oracle_efficiency_ratio', 'time_seconds', 'train_mse', 'gap']:
                        loss_key = key
                        break
                
                if loss_key and loss_key in method_data:
                    values.append(method_data[loss_key])
                
                # Also collect train_mse and gap if available
                if 'train_mse' in method_data:
                    train_values.append(method_data['train_mse'])
                if 'gap' in method_data:
                    gap_values.append(method_data['gap'])
                
                if 'time_seconds' in method_data:
                    times.append(method_data['time_seconds'])
        
        if values:
            aggregated[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'n': len(values)
            }
            
            # Add train_mse stats if available
            if train_values:
                aggregated[metric]['train_mean'] = float(np.mean(train_values))
                aggregated[metric]['train_std'] = float(np.std(train_values))
                aggregated[metric]['train_min'] = float(np.min(train_values))
                aggregated[metric]['train_max'] = float(np.max(train_values))
            
            # Add gap stats if available
            if gap_values:
                aggregated[metric]['gap_mean'] = float(np.mean(gap_values))
                aggregated[metric]['gap_std'] = float(np.std(gap_values))
                aggregated[metric]['gap_min'] = float(np.min(gap_values))
                aggregated[metric]['gap_max'] = float(np.max(gap_values))
            
            if times:
                aggregated[metric]['time_mean'] = float(np.mean(times))
                aggregated[metric]['time_std'] = float(np.std(times))
                aggregated[metric]['time_min'] = float(np.min(times))
                aggregated[metric]['time_max'] = float(np.max(times))
    
    return aggregated


def print_aggregate_statistics(aggregated, loss_metric_name='mse'):
    """
    Print formatted aggregate statistics table.
    
    Args:
        aggregated: dict from aggregate_results()
        loss_metric_name: name to display (e.g., 'mse', 'kl_divergence', 'angular_error')
    """
    print("\n" + "=" * 130)
    print("AGGREGATE STATISTICS OVER ALL REPETITIONS")
    print("=" * 130)
    
    # Check if we have train_mse and gap metrics
    has_train_data = any('train_mean' in stats for stats in aggregated.values())
    
    if has_train_data:
        # Expanded table format with train MSE, gap, and time
        method_width = 25
        metric_width = 12
        train_width = 12
        gap_width = 12
        time_width = 10
        std_width = 10
        
        header = f"{'Method':<{method_width}} | {loss_metric_name.upper():>{metric_width}} | {'Train':>{train_width}} | {'Gap':>{gap_width}} | {'Time':>{time_width}} | {'±std':>{std_width}}"
        print(header)
        print("-" * len(header))
        
        # Sort by mean (ascending)
        sorted_methods = sorted(aggregated.items(), key=lambda x: x[1]['mean'])
        
        for method, stats in sorted_methods:
            mean = stats['mean']
            std = stats['std']
            train_mean = stats.get('train_mean', np.nan)
            gap_mean = stats.get('gap_mean', np.nan)
            time_mean = stats.get('time_mean', np.nan)
            
            print(f"{method:<{method_width}} | {mean:>{metric_width}.6f} | {train_mean:>{train_width}.6f} | {gap_mean:>{gap_width}.6f} | {time_mean:>{time_width}.3f} | ±{std:>{std_width-1}.6f}")
    else:
        # Original table format (for backwards compatibility)
        method_width = 25
        metric_width = 12
        std_width = 12
        min_width = 12
        max_width = 12
        
        header = f"{'Method':<{method_width}} | {loss_metric_name.upper():>{metric_width}} | {'± std':>{std_width}} | {'Min':>{min_width}} | {'Max':>{max_width}}"
        print(header)
        print("-" * len(header))
        
        # Sort by mean (ascending)
        sorted_methods = sorted(aggregated.items(), key=lambda x: x[1]['mean'])
        
        for method, stats in sorted_methods:
            mean = stats['mean']
            std = stats['std']
            min_val = stats['min']
            max_val = stats['max']
            
            print(f"{method:<{method_width}} | {mean:>{metric_width}.6f} | ±{std:>{std_width-1}.6f} | {min_val:>{min_width}.6f} | {max_val:>{max_width}.6f}")
    
    print("=" * 130)


def print_time_comparison(aggregated):
    """Print timing comparison table for all methods."""
    
    # Filter methods that have timing info
    methods_with_time = {m: s for m, s in aggregated.items() if 'time_mean' in s}
    
    if not methods_with_time:
        return
    
    print("\n" + "=" * 90)
    print("TIME COMPARISON (seconds per run)")
    print("=" * 90)
    
    method_width = 25
    time_width = 12
    std_width = 12
    min_width = 12
    max_width = 12
    
    header = f"{'Method':<{method_width}} | {'Mean':>{time_width}} | {'± std':>{std_width}} | {'Min':>{min_width}} | {'Max':>{max_width}}"
    print(header)
    print("-" * len(header))
    
    # Sort by mean time (ascending)
    sorted_methods = sorted(methods_with_time.items(), key=lambda x: x[1]['time_mean'])
    
    total_times = []
    for method, stats in sorted_methods:
        mean_time = stats['time_mean']
        std_time = stats['time_std']
        min_time = stats['time_min']
        max_time = stats['time_max']
        
        print(f"{method:<{method_width}} | {mean_time:>{time_width}.4f} | ±{std_time:>{std_width-1}.4f} | {min_time:>{min_width}.4f} | {max_time:>{max_width}.4f}")
        total_times.append(mean_time)
    
    print("-" * len(header))
    print(f"{'TOTAL':<{method_width}} | {sum(total_times):>{time_width}.4f} seconds")
    print("=" * 90)


def print_subspace_metrics(all_results):
    """Print subspace distance metrics if available."""
    
    distances = []
    ratios = []
    
    for rep_result in all_results:
        if 'subspace_metrics' in rep_result:
            metrics = rep_result['subspace_metrics']
            if 'projection_distance' in metrics:
                distances.append(metrics['projection_distance'])
            if 'oracle_efficiency_ratio' in metrics:
                ratios.append(metrics['oracle_efficiency_ratio'])
    
    if distances:
        print("\n" + "=" * 90)
        print("SUBSPACE DISTANCE (Projection Distance to True Subspace)")
        print("=" * 90)
        print(f"Mean projection distance: {np.mean(distances):.6f}")
        print(f"Std: ±{np.std(distances):.6f}")
        print(f"Range: [{np.min(distances):.6f}, {np.max(distances):.6f}]")
        print("(Lower is better, range is [0, 1])")
        print("=" * 90)


def print_final_ranking(aggregated, loss_metric_name='mse'):
    """Print final top-k ranking."""
    
    print("\n" + "=" * 100)
    print("METHOD RANKING (by average Test MSE)")
    print("=" * 100)
    
    # Check if we have train_mse and gap metrics
    has_train_data = any('train_mean' in stats for stats in aggregated.values())
    
    sorted_methods = sorted(aggregated.items(), key=lambda x: x[1]['mean'])
    medals = ['🥇', '🥈', '🥉']
    
    if has_train_data:
        for rank, (method, stats) in enumerate(sorted_methods):
            mean = stats['mean']
            train_mean = stats.get('train_mean', np.nan)
            gap_mean = stats.get('gap_mean', np.nan)
            time_mean = stats.get('time_mean', np.nan)
            medal = medals[rank] if rank < 3 else f"  {rank + 1}. "
            print(f"{medal} {method:<20} Test={mean:.6f} | Train={train_mean:.6f} | Gap={gap_mean:.6f} | Time={time_mean:.3f}s")
    else:
        for rank, (method, stats) in enumerate(sorted_methods):
            mean = stats['mean']
            medal = medals[rank] if rank < 3 else f"  {rank + 1}. "
            print(f"{medal} {method:<20} {loss_metric_name.upper()} = {mean:.6f}")
    
    print("=" * 100)


class MethodTimer:
    """Context manager for timing method execution."""
    
    def __init__(self, method_name):
        self.method_name = method_name
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
