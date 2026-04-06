#!/usr/bin/env python3
"""Verify all setups have validation split implemented."""

import os

os.chdir("simulations_sdr")

setup_files = {
    1: "setup1_linear.py",
    2: "setup2_linear_p50.py",
    3: "setup3_linear_v20.py",
    4: "setup4_nonlinear_z.py",
    5: "setup5_correlated_responses.py",
    6: "setup6_wasserstein_distributions.py",
    7: "setup7_spherical_directions.py",
    8: "setup8_correlation_matrices.py",
    9: "setup9_simplex_compositions.py",
    10: "setup10_quantile_groups.py",
}

print("VALIDATION SPLIT VERIFICATION")
print("=" * 70)

all_complete = True
for setup_num, filename in setup_files.items():
    if not os.path.exists(filename):
        print(f"Setup {setup_num:2d}: ✗ File not found")
        all_complete = False
        continue
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # Extract grid_search function
    func_start = content.find("def grid_search_fsdrnn_d")
    func_end = content.find("\ndef ", func_start + 10) if func_start >= 0 else -1
    
    if func_start < 0:
        print(f"Setup {setup_num:2d}: ✗ grid_search_fsdrnn_d not found")
        all_complete = False
        continue
    
    if func_end < 0:
        func_end = len(content)
    
    func_content = content[func_start:func_end]
    
    # Check for validation split indicators
    has_train_val_split = ("train_idx = idx[:-val_size]" in func_content or 
                           "train_idx = idx[: int(n * val_split)]" in func_content)
    has_full_retrain = "best_method.fit(X_train, Y_train)" in func_content
    no_test_in_func = "X_test" not in func_content and "Y_test" not in func_content
    has_val_param = "val_split=" in func_content
    
    is_complete = has_train_val_split and has_full_retrain and no_test_in_func
    status = "✓" if is_complete else "✗"
    
    print(f"Setup {setup_num:2d}: {status} ", end="")
    if is_complete:
        print("✓ Validation split complete")
    else:
        issues = []
        if not has_train_val_split:
            issues.append("no train/val split")
        if not has_full_retrain:
            issues.append("no full retrain")
        if not no_test_in_func:
            issues.append("uses test set")
        print("✗ " + ", ".join(issues))
        all_complete = False

print("=" * 70)
if all_complete:
    print("\n✅ ALL 10 SETUPS COMPLETE WITH VALIDATION SPLIT!\n")
else:
    print("\n❌ Some setups need fixes\n")
