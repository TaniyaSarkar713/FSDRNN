# Tuned Simulation Script

This is a copy of `test_spd_frechet_adaptive.py` with pre-tuned hyperparameters hardcoded to skip the tuning phase.

## Tuned Parameters

- **DFR**: {'hidden': 16, 'layer': 3, 'lr': 0.0005, 'manifold_dim': 2}
- **FSDRNN**: {'dropout': 0.3, 'entropy_reg': 0.0, 'lr': 0.001, 'nuclear_reg': 0.0001, 'reduction_dim': 2, 'reduction_type': 'nonlinear', 'response_rank': 5}

## Default Arguments

The script has the following defaults set:
- `--metric power`
- `--n_reps 20`
- `--n_test 200`
- `--epochs 1000`
- `--batch_size 32`
- `--lr 5e-4`
- `--dropout 0.3`

## Usage

Run the script directly (tuning is skipped by default):

```bash
python test_spd_frechet_adaptive.py
```

Or with custom arguments:

```bash
python test_spd_frechet_adaptive.py --metric power --n_reps 20 --n_test 200 --epochs 1000 --batch_size 32 --lr 5e-4 --dropout 0.3
```

This will use the pre-tuned parameters for fast execution.