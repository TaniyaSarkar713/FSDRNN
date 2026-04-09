# SDR Simulation Setups

This folder contains 9 different simulation setups for comparing **5 methods** on multivariate Sufficient Dimension Reduction (SDR) models:

- **Global Mean**: Baseline - predicts the sample mean for all inputs
- **GFR**: Global Fréchet Regression - linear regression on each response
- **DFR**: Deep Fréchet Regression - neural network for each response independently
- **E2M**: Embedding to Manifold - shared latent representation
- **FSDRNN**: Fréchet SDR Neural Network with Adaptive LoRA - joint SDR with dimension reduction

## Setup Descriptions

### Setup 1: Linear-Nonlinear Model
**File**: `setup1_linear_nonlinear.py`

- **d₀**: 2 (true structural dimension)
- **V**: 8 (responses)
- **p**: 20 (input dimension)
- **Model**: $Y_v = a_{v1} z_1 + a_{v2} z_2 + a_{v3} z_1 z_2 + \varepsilon_v$
- **Sparse directions**: Two disjoint blocks of 5 coordinates each
- **Noise**: Gaussian, σ = 0.25
- **Purpose**: Clean linear + interaction model testing baseline performance

```bash
python setup1_linear_nonlinear.py --n_train 300 --n_test 150 --n_reps 3
```

---

### Setup 2: Pure Nonlinear Model
**File**: `setup2_pure_nonlinear.py`

- **d₀**: 2
- **V**: 8
- **p**: 20
- **Model**: $Y_v = \exp(a_v z_1 \times 0.3) + b_v \sin(z_2) + \varepsilon_v$
- **No linear terms** - purely exponential and sinusoidal functions
- **Noise**: Gaussian, σ = 0.25
- **Purpose**: Test performance on highly nonlinear structure

```bash
python setup2_pure_nonlinear.py --n_train 300 --n_test 150 --n_reps 3
```

---

### Setup 3: High-Dimensional Response
**File**: `setup3_high_response_dim.py`

- **d₀**: 2
- **V**: 20 (high response dimension!)
- **p**: 20
- **Model**: Linear + nonlinear with 20 response variables
- **Noise**: Gaussian, σ = 0.25
- **Purpose**: Test scaling to many response variables

```bash
python setup3_high_response_dim.py
```

---

### Setup 4: Three-Index Model
**File**: `setup4_three_index.py`

- **d₀**: 3 (three sufficient directions!)
- **V**: 8
- **p**: 25 (increased to accommodate 3 directions)
- **Model**: $Y_v = \sum_{j=1}^3 a_{vj} z_j + \varepsilon_v$
- **Noise**: Gaussian, σ = 0.25
- **Purpose**: Test when true dimension > 2

```bash
python setup4_three_index.py
```

---

### Setup 5: High Noise
**File**: `setup5_high_noise.py`

- **d₀**: 2
- **V**: 8
- **p**: 20
- **Model**: Same as Setup 1 (linear + nonlinear)
- **Noise**: Gaussian, σ = 1.0 (4× higher noise!)
- **Purpose**: Test robustness under high noise

```bash
python setup5_high_noise.py
```

---

### Setup 6: Small Sample Size
**File**: `setup6_small_sample.py`

- **d₀**: 2
- **V**: 8
- **p**: 20
- **n_train**: 100 (small training set!)
- **n_test**: 75
- **Model**: Linear + nonlinear
- **Noise**: Gaussian, σ = 0.25
- **Default n_reps**: 5 (more repetitions for stability)
- **Purpose**: Test performance with limited data

```bash
python setup6_small_sample.py
```

---

### Setup 7: Heteroscedastic Noise
**File**: `setup7_heteroscedastic_noise.py`

- **d₀**: 2
- **V**: 8
- **p**: 20
- **Noise**: σ(z₁) = 0.1 + 0.15|z₁| (depends on latent variables!)
- **Purpose**: Test when noise variance is non-constant

```bash
python setup7_heteroscedastic_noise.py
```

---

### Setup 8: High Input Dimension
**File**: `setup8_high_input_dim.py`

- **d₀**: 2
- **V**: 8
- **p**: 50 (high input dimension!)
- **Model**: Linear + nonlinear
- **True structure**: Still sparse (first 5 + next 5 coordinates)
- **Noise**: Gaussian, σ = 0.25
- **Purpose**: Test dimension reduction with p >> d₀

```bash
python setup8_high_input_dim.py
```

---

### Setup 9: Polynomial Interactions
**File**: `setup9_polynomial.py`

- **d₀**: 2
- **V**: 8
- **p**: 20
- **Model**: $Y_v = a_{v1} z_1^2 + a_{v2} z_2^2 + a_{v3} z_1 z_2 + \varepsilon_v$
- **Higher-order polynomial terms** (quadratic)
- **Noise**: Gaussian, σ = 0.25
- **Purpose**: Test on more complex polynomial structure

```bash
python setup9_polynomial.py
```

---

## Running Experiments

### Individual Setup
Run a single setup with custom parameters:
```bash
python setup1_linear_nonlinear.py --n_train 400 --n_test 200 --n_reps 5
```

### All Setups at Once
```bash
for i in {1..9}; do
  echo "Running Setup $i..."
  python setup${i}_*.py --n_train 300 --n_test 150 --n_reps 3
done
```

### With Display of Methods
Each script prints:
- Method fitting progress
- MSE and RMSE for each method per repetition
- Aggregate statistics (mean ± std) over repetitions

### Output
Each setup generates a **JSON results file**:
- `setup1_results.json`
- `setup2_results.json`
- ... etc

## Shared Utilities

**File**: `sdr_utils.py`

Contains reusable implementations:
- `GlobalMean`: Baseline prediction
- `GFR`: Linear regression per response
- `DFR`: Neural network per response
- `E2M`: Shared encoder + response heads
- `FSDRNN`: Joint SDR with Adaptive LoRA
- `run_experiment()`: Generic experiment harness

All method implementations support both CPU and GPU (auto-detected).

## Performance Expectations

**Typical Results Summary** (Setup 1-2):
```
Global Mean:  MSE ≈ 2.3   (baseline)
GFR:          MSE ≈ 0.44  (good, linear)
DFR:          MSE ≈ 0.32  (best on linear component)
E2M:          MSE ≈ 1.88  (requires more tuning)
FSDRNN:       MSE ≈ 0.59  (competitive with GFR)
```

- **Setup 1** (linear): DFR > GFR > FSDRNN
- **Setup 2** (nonlinear): Methods diverge - FSDRNN/DFR advantage expected
- **Setup 5** (high noise): All methods struggle; relative ranking persists
- **Setup 6** (small n): Higher variance; more repetitions helpful

## Customization

### Modify a Setup
Edit any `setupX_*.py` file to change:
- Data generator function (replace `data_gen()`)
- Sample sizes: `run_experiment(..., n_train=500, n_test=250)`
- Repetitions: `n_reps=10`
- Training epochs/learning rates in `sdr_utils.py`

### Add New Methods
Add to `sdr_utils.py`:
1. Implement method class or wrapper
2. Add to `methods` dict in `run_experiment()`
3. Results automatically saved as JSON

## Files

```
simulations_sdr/
├── sdr_utils.py                      # Shared utilities & implementations
├── setup1_linear_nonlinear.py        # Original template (more verbose)
├── setup2_pure_nonlinear.py
├── setup3_high_response_dim.py
├── setup4_three_index.py
├── setup5_high_noise.py
├── setup6_small_sample.py
├── setup7_heteroscedastic_noise.py
├── setup8_high_input_dim.py
├── setup9_polynomial.py
├── README.md                         # This file
├── setup1_results.json
├── setup2_results.json
├── ... (result files after running)
```

## Notes

- All scripts compare the **same 5 methods** for consistency
- Results saved as **JSON** for easy post-processing
- Methods use same hyperparameters across setups (learning rate 1e-3, epochs 200)
- CUDA GPU is used if available (auto-detection)
- Reproducible: set `seed` parameter for deterministic results

## Next Steps

1. Run a few setups to understand method behavior
2. Aggregate results across setups for summary statistics
3. Identify which setups favor FSDRNN vs baselines
4. Optional: Add your own custom setups following the template
