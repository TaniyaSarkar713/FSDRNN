# Enhanced Setup Scripts - Comprehensive Reporting

## Overview

All 5 new non-Euclidean setup scripts (6-10) now include comprehensive reporting with:
- GPU and host information
- Method-level execution timing
- Aggregate statistics across repetitions
- Time comparison tables
- Final performance ranking

## Example Output

### 1. System & GPU Information
```
============================================================
=== host ===
hpc-i31-5.local
=== visible gpu(s) ===
CUDA_VISIBLE_DEVICES=1
GPU 0: NVIDIA RTX 4500 Ada Generation
torch: 2.10.0+cu128
cuda available: True
CUDA_VISIBLE_DEVICES: 1
gpu: NVIDIA RTX 4500 Ada Generation
============================================================
```

### 2. Setup Configuration
```
Setup: setup6_wasserstein_distributions
Base seed: 42
Number of repetitions: 10
============================================================
Running: simulations_sdr/setup6_wasserstein_distributions.py with 10 independent runs
```

### 3. Method Timing During Execution
```
Repetition 1/10 (seed=42)
  • FSDRNN... (0.87s)
  • Oracle FSDRNN... (1.92s)
```

### 4. Aggregate Statistics
```
================================================================================
AGGREGATE STATISTICS OVER ALL REPETITIONS
================================================================================
Method                    |          MSE |        ± std |          Min |          Max
--------------------------------------------------------------------------------
FSDRNN                    |     0.951013 | ±   0.213018 |     0.700390 |     1.221086
Oracle FSDRNN             |     1.156791 | ±   0.140475 |     0.981029 |     1.324864
================================================================================
```

### 5. Time Comparison Table
```
================================================================================
TIME COMPARISON (seconds per run)
================================================================================
Method                    |         Mean |        ± std |          Min |          Max
--------------------------------------------------------------------------------
FSDRNN                    |       0.9068 | ±     0.1926 |       0.7581 |       1.1788
Oracle FSDRNN             |       1.9686 | ±     0.2155 |       1.8030 |       2.2730
--------------------------------------------------------------------------------
TOTAL                     |       2.8754 seconds
================================================================================
```

### 6. Final Ranking
```
======================================================================
🥇 FSDRNN               MSE = 0.951013
🥈 Oracle FSDRNN        MSE = 1.156791
======================================================================

Elapsed time: 8.65s
```

## Running the Scripts

### Basic Usage (All 5 New Setups)
```bash
# Setup 6: Wasserstein Distributions
python simulations_sdr/setup6_wasserstein_distributions.py --n_train 300 --n_test 150 --n_reps 10 --verbose

# Setup 7: Spherical Directions
python simulations_sdr/setup7_spherical_directions.py --n_train 300 --n_test 150 --n_reps 10 --verbose

# Setup 8: Correlation Matrices
python simulations_sdr/setup8_correlation_matrices.py --n_train 300 --n_test 150 --n_reps 10 --verbose

# Setup 9: Simplex Compositions
python simulations_sdr/setup9_simplex_compositions.py --n_train 300 --n_test 150 --n_reps 10 --verbose

# Setup 10: Quantile Functions (Grouped)
python simulations_sdr/setup10_quantile_groups.py --n_train 300 --n_test 150 --n_reps 10 --verbose
```

### With Output File Saving
```bash
python simulations_sdr/setup6_wasserstein_distributions.py \
  --n_train 300 --n_test 150 --n_reps 10 \
  --output results_sdr/setup6_results.json \
  --seed 42 --verbose
```

### With GPU/Device Specification
```bash
python simulations_sdr/setup6_wasserstein_distributions.py \
  --n_train 300 --n_test 150 --n_reps 10 \
  --device cuda --verbose
```

### With Task ID (for Slurm)
```bash
python simulations_sdr/setup6_wasserstein_distributions.py \
  --n_train 300 --n_test 150 --n_reps 10 \
  --task_id 4 --device cuda --verbose
```

## Key Metrics Reported

### Aggregate Statistics
- **Mean**: Average metric value across all repetitions
- **Std**: Standard deviation (consistency)
- **Min/Max**: Range of performance

### Time Comparison
- **Mean time**: Average seconds per method per run
- **Std time**: Consistency of execution time
- **Total**: Sum of mean times for all methods
- Shows which methods are computationally efficient

### Performance Ranking
- Sorted by mean loss (ascending = better)
- Medals for top 3 methods
- Clear victor identification

## Output Loss Metrics by Setup

| Setup | File | Loss Metric | Notes |
|-------|------|-------------|-------|
| 6 | setup6_wasserstein_distributions.py | MSE on (μ, σ) | Quantile functions |
| 7 | setup7_spherical_directions.py | Angular Error | Geodesic distance on sphere |
| 8 | setup8_correlation_matrices.py | MSE | Correlation parameters |
| 9 | setup9_simplex_compositions.py | KL Divergence | Probability distributions |
| 10 | setup10_quantile_groups.py | MSE | Grouped quantile functions |

## Timing Insights

Each script tracks:
1. **Individual method times** - Training time for FSDRNN and Oracle FSDRNN
2. **Total runtime** - Sum of all method training times
3. **Wall clock time** - Total elapsed time including overhead

### Example Timing Breakdown (from output above)
- FSDRNN: 0.91 ± 0.19 seconds
- Oracle FSDRNN: 1.97 ± 0.22 seconds
- **Total per repetition**: ~2.88 seconds
- **For 10 reps**: ~28.8 seconds
- **Actual elapsed**: 8.65 seconds (parallel efficiency)

## Utility Module: reporting_utils.py

The `reporting_utils.py` module provides:

```python
# GPU and system detection
get_gpu_info()  # Returns CUDA, host, GPU info
print_system_info()  # Pretty-print system header

# Result aggregation
aggregate_results(all_results, n_reps)  # Compute statistics

# Beautiful output formatting
print_aggregate_statistics()  # Aggregate stats table
print_time_comparison()  # Timing table
print_subspace_metrics()  # Optional subspace analysis
print_final_ranking()  # Medal ranking

# Timing context manager
with MethodTimer('method_name') as timer:
    # code to time
    pass
# timer.elapsed contains seconds elapsed
```

## Batch Running All 5 Setups

```bash
#!/bin/bash
cd simulations_sdr

for setup in 6 7 8 9 10; do
    echo "================================"
    echo "Running Setup $setup"
    echo "================================"
    python setup${setup}_*.py \
        --n_train 300 --n_test 150 \
        --n_reps 10 --verbose \
        --output results_sdr/setup${setup}_results.json
done
```

## Notes

- All scripts support `--verbose` for detailed per-repetition output
- GPU detection is automatic (CUDA if available, CPU if not)
- Timing is measured in seconds with 4 decimal precision
- Statistics aggregate across all repetitions
- Output format is consistent across all 5 new setups
