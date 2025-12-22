# Adduct Generalization Hyperparameter Sweep

## Quick Start

Run the focused hyperparameter sweep for adduct-related features:

```bash
# Activate your environment
pyenv activate GXS_torch

# Run sweep with 50 iterations
python run_adduct_sweep.py --count 50

# Or with custom project name
python run_adduct_sweep.py --count 100 --project my-adduct-sweep
```

## What This Sweeps

This sweep focuses **only** on parameters related to the new generalized adduct descriptor features:

### 🎯 Dense Head Architecture
- `final_activation`: Output activation function (relu, selu, elu, none)
- `num_dense_layers`: Number of dense layers (2, 3, 4, 6)
- `dense_units`: Units per dense layer (128, 256, 384, 512)

### 🔧 Global Feature Encoder (Adduct Descriptors)
- `global_emb_dim`: Embedding dimension for adduct descriptors (8, 16, 24, 32)
- `num_global_encoder_layers`: MLP depth for encoding descriptors (1, 2, 3, 4)

### 📈 Training Dynamics
- `learning_rate`: Initial learning rate (log-uniform: 0.0001 to 0.01)
- `weight_decay`: L2 regularization (log-uniform: 0.00001 to 0.001)
- `dropout`: Dropout probability (0.0, 0.1, 0.2, 0.3)

### 📉 Learning Rate Scheduler
- `lr_scheduler.factor`: LR reduction factor (0.1, 0.2, 0.5)
- `lr_scheduler.patience`: Epochs before LR reduction (10, 15, 20, 25)
- `lr_scheduler.min_lr`: Minimum learning rate (log-uniform: 1e-7 to 1e-6)

## What's Fixed

All core model architecture parameters are kept constant:
- Model: Sigma with ECC layers
- ECC channels: 16
- ECC layers: 3
- Kernel network: [64, 64, 64, 64]
- Batch size: 16
- Readout: median
- Feature encoding: Descriptor-based (not OHE)

## Files Created

1. **`sweep_configs/sigma_adduct_generalization_sweep.yaml`**
   - W&B sweep configuration in YAML format
   - Defines all hyperparameters and their ranges

2. **`run_adduct_sweep.py`**
   - Convenient Python script to run the sweep
   - Handles environment checks and W&B setup
   - Provides progress monitoring

## Usage Examples

### Basic Usage
```bash
# Run 50 iterations
python run_adduct_sweep.py --count 50
```

### Custom Configuration
```bash
# Use different project name
python run_adduct_sweep.py --count 100 --project adduct-optimization-v2

# Quick test with 5 runs
python run_adduct_sweep.py --count 5 --project test-sweep
```

### Dry Run (Create Sweep Without Running)
```bash
# Create sweep but don't run agent
python run_adduct_sweep.py --dry-run

# Then run agent manually (allows running on multiple machines)
wandb agent <SWEEP_ID>
```

## Monitoring Results

After starting the sweep, you can:

1. **View in W&B Dashboard**: The script will print the URL
2. **Monitor in Terminal**: Real-time progress shown in console
3. **Track Best Model**: W&B tracks `best_val_mae` automatically

## Expected Compute Time

- **Per run**: ~15-30 minutes (200 epochs, SigmaCCS dataset)
- **50 runs**: ~12-25 hours total
- **100 runs**: ~25-50 hours total

Early termination (Hyperband) will stop poorly performing runs early, saving compute time.

## Parameter Space Size

- **Discrete parameters**: ~1,500 combinations
- **Continuous parameters**: Infinite (sampled via Bayesian optimization)
- **Recommended runs**: 50-100 for good coverage

## Tips

1. **Start Small**: Try 10 runs first to verify everything works
2. **Use Bayesian Optimization**: Already configured in sweep config
3. **Monitor Early**: Check first few runs to catch any issues
4. **Multiple Agents**: You can run multiple agents in parallel:
   ```bash
   # Terminal 1
   python run_adduct_sweep.py --count 25
   
   # Terminal 2 (after sweep is created)
   wandb agent <SWEEP_ID>
   ```

## Results Analysis

After the sweep completes:

1. Check W&B parallel coordinates plot to see parameter relationships
2. Look at the importance plot to see which parameters matter most
3. Export top-5 configurations for final evaluation
4. Update `configs/sigma_config.json` with best parameters

## Troubleshooting

**If W&B is not authenticated:**
```bash
wandb login
```

**If GraphXSection_torch import fails:**
```bash
# Make sure you're in the right environment
pyenv activate GXS_torch

# Verify installation
python -c "import GraphXSection_torch; print('OK')"
```

**If sweep gets stuck:**
- Check W&B dashboard for error messages
- Look at `outputs/` directory for model logs
- Cancel sweep and restart with `--count 5` for debugging
