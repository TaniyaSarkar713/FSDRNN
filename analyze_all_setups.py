"""
Aggregate and analyze results from all 9 SDR setups.
Usage: python analyze_all_setups.py
"""

import json
import numpy as np
from pathlib import Path

def load_results(filename):
    """Load results from a JSON file."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def analyze_setup(setup_num):
    """Analyze results from a single setup."""
    filename = f"setup{setup_num}_results.json"
    results = load_results(filename)
    
    if results is None:
        print(f"  ⚠ {filename} not found - run the setup first")
        return None
    
    methods = list(results[0]['methods'].keys())
    setup_info = results[0]
    
    print(f"\n{'='*70}")
    print(f"Setup {setup_num}: {setup_info.get('setup', 'Unknown')}")
    print(f"{'='*70}")
    print(f"  Configuration: d₀={setup_info.get('d0', '?')}, V={setup_info.get('V', '?')}, " +
          f"p={setup_info.get('p', '?')}, n_train={setup_info.get('n_train', '?')}")
    print(f"  Model: {setup_info.get('model', 'linear+nonlinear')}")
    print()
    
    # Aggregate metrics
    all_results = {method: [] for method in methods}
    for rep in results:
        for method in methods:
            all_results[method].append(rep['methods'][method]['mse'])
    
    # Print comparison table
    print(f"{'Method':<15} | {'MSE (mean)':<12} | {'± std':<10} | {'Min':<8} | {'Max':<8}")
    print("-" * 75)
    
    # Rank by mean MSE
    ranking = sorted(
        [(m, np.mean(all_results[m])) for m in methods],
        key=lambda x: x[1]
    )
    
    for rank, (method, mean_mse) in enumerate(ranking, 1):
        std_mse = np.std(all_results[method])
        min_mse = np.min(all_results[method])
        max_mse = np.max(all_results[method])
        marker = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        print(f"{marker} {method:<12} | {mean_mse:12.6f} | ±{std_mse:8.6f} | {min_mse:8.6f} | {max_mse:8.6f}")
    
    return all_results

def main():
    print("\n" + "="*70)
    print("SDR SIMULATION RESULTS SUMMARY")
    print("="*70)
    
    all_setups = {}
    for i in range(1, 10):
        all_setups[i] = analyze_setup(i)
    
    # Overall summary
    print("\n\n" + "="*70)
    print("OVERALL SUMMARY ACROSS ALL SETUPS")
    print("="*70)
    
    methods = ['Global Mean', 'GFR', 'DFR', 'E2M', 'FSDRNN']
    method_ranks = {m: [] for m in methods}
    
    for setup_num, results in all_setups.items():
        if results is None:
            continue
        
        # Rank methods by MSE in this setup
        ranking = sorted(
            [(m, np.mean(results[m])) for m in methods],
            key=lambda x: x[1]
        )
        
        for rank, (method, _) in enumerate(ranking, 1):
            method_ranks[method].append(rank)
    
    # Compute average rank
    print(f"\n{'Method':<15} | {'Avg Rank':<10} | {'Ranking Distribution':<40}")
    print("-" * 70)
    
    for method in sorted(method_ranks.keys(), key=lambda m: np.mean(method_ranks[m])):
        avg_rank = np.mean(method_ranks[method])
        ranks = method_ranks[method]
        distribution = {r: ranks.count(r) for r in sorted(set(ranks))}
        dist_str = ", ".join([f"#{r}:{c}" for r, c in sorted(distribution.items())])
        
        print(f"{method:<15} | {avg_rank:10.2f} | {dist_str:<40}")
    
    # Winner
    winner = min(method_ranks.keys(), key=lambda m: np.mean(method_ranks[m]))
    print(f"\n🏆 Best overall method: {winner} (avg rank: {np.mean(method_ranks[winner]):.2f})")

if __name__ == '__main__':
    main()
