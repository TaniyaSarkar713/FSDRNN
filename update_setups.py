#!/usr/bin/env python3
"""Update setups 4-10 with validation split grid search."""

import re
import os

os.chdir("simulations_sdr")

# Template for new grid_search function
GRID_SEARCH_TEMPLATE = '''def grid_search_fsdrnn_d(X_train, Y_train, {params}, val_split=0.2):
    """
    Grid search for optimal latent dimension d using validation split (not test set).
    
    Args:
        X_train, Y_train: Full training data
        val_split: Fraction of training data to use for validation
    
    Returns:
        best_method: FSdrnnWrapper with best d, trained on full X_train
        best_d: the best d value
        results_per_d: dict with validation error for each d
    """
    # Split training data into train and validation  
    n = X_train.shape[0]
    val_size = max(int(n * val_split), 10)
    idx = np.arange(n)
    np.random.shuffle(idx)
    train_idx = idx[:-val_size]
    val_idx = idx[-val_size:]
    
    X_tr = X_train[train_idx]
    Y_tr = Y_train[train_idx]
    X_val = X_train[val_idx]
    Y_val = Y_train[val_idx]
    
    results_per_d = {{}}
    
    for d in d_values:
        if verbose:
            print(f"    Testing d={{d}}...")
        
        method = FSdrnnWrapper(p, V, d=d, lr=lr, epochs=epochs, dropout=dropout, device=device, verbose=False)
        method.fit(X_tr, Y_tr)
        
        # Evaluate on validation set
        Y_pred = method.predict(X_val)
        val_error = evaluate_prediction(Y_val, Y_pred)['mse']
        results_per_d[d] = val_error
        
        if verbose:
            print(f"      d={{d}}: Val MSE = {{val_error:.6f}}")
    
    # Pick best d and train final model on full training data
    best_d = min(results_per_d, key=results_per_d.get)
    best_method = FSdrnnWrapper(p, V, d=best_d, lr=lr, epochs=epochs, dropout=dropout, device=device, verbose=False)
    best_method.fit(X_train, Y_train)
    
    return best_method, best_d, results_per_d
'''

setups = {
    4: ("setup4_nonlinear_z.py", "p=100, V=40, d_values=[2, 3, 5], lr=3e-4, epochs=1000, dropout=0.2, device='cpu', verbose=False"),
    5: ("setup5_correlated_responses.py", "p=100, V=40, d_values=[2, 3, 5], lr=5e-4, epochs=1000, dropout=0.2, device='cpu', verbose=False"),
    6: ("setup6_wasserstein_distributions.py", "p=100, V=20, d_values=[2, 3, 5], lr=3e-4, epochs=1000, dropout=0.1, device='cpu', verbose=False"),
    7: ("setup7_spherical_directions.py", "p=20, V=15, d_values=[2, 3, 5], lr=5e-4, epochs=1000, dropout=0.1, device='cpu', verbose=False"),
    8: ("setup8_correlation_matrices.py", "p=20, V=15, d_values=[2, 3, 5], lr=5e-4, epochs=1000, dropout=0.1, device='cpu', verbose=False"),
    9: ("setup9_simplex_compositions.py", "p=10, V=8, d_values=[2, 3, 5], lr=5e-4, epochs=1000, dropout=0.1, device='cpu', verbose=False"),
    10: ("setup10_quantile_groups.py", "p=20, V=12, d_values=[2, 3, 5], lr=5e-4, epochs=1000, dropout=0.1, device='cpu', verbose=False"),
}

for setup_num, (filename, params) in setups.items():
    try:
        with open(filename, 'r') as f:
            content = f.read()
        
        # Find grid_search function
        start_idx = content.find("def grid_search_fsdrnn_d")
        if start_idx == -1:
            print(f"✗ {filename}: function not found")
            continue
        
        # Find the end (next "def " at line start)
        end_idx = content.find("\ndef ", start_idx + 10)
        if end_idx == -1:
            end_idx = len(content)
        
        # Replace the function
        new_func = GRID_SEARCH_TEMPLATE.format(params=params) + "\n\n"
        updated_content = content[:start_idx] + new_func + content[end_idx:]
        
        with open(filename, 'w') as f:
            f.write(updated_content)
        
        print(f"✓ Updated {filename}")
    
    except Exception as e:
        print(f"✗ {filename}: {e}")

print("\nDone!")
