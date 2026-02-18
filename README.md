# hybrid-rl: Robustness and Recovery in Offline-to-Online RL

CS 234 Project — Phase I: Characterizing Failure Modes in Offline RL

## Project Structure

```
hybrid-rl/
├── run_phase1.py            # Entry point – runs experiment grid & generates plots
├── smoke_test.py            # Quick end-to-end sanity check
├── src/
│   ├── data/
│   │   ├── loader.py        # D4RL HDF5 download & Gymnasium env creation
│   │   ├── corruption.py    # Top-k% removal, Gaussian reward noise
│   │   └── dataset.py       # OfflineDataset & ReplayBuffer
│   ├── algos/
│   │   ├── networks.py      # Shared NN blocks (Q-net, V-net, policy)
│   │   ├── cql.py           # Conservative Q-Learning (CQL)
│   │   └── iql.py           # Implicit Q-Learning (IQL)
│   ├── eval/
│   │   ├── metrics.py       # Normalized returns, Q-stats, loss variance
│   │   └── plotting.py      # Learning curves, bar charts, diagnostics
│   └── train.py             # Unified offline training loop
├── results/                 # Auto-created: checkpoints + metrics JSON
└── plots/                   # Auto-created: figures
```

## Setup

```bash
# Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

No legacy MuJoCo 210 or `mujoco-py` required — datasets are downloaded as
HDF5 files directly from the D4RL servers and environments use Gymnasium v4.

### Dependencies

See [requirements.txt](requirements.txt) for the full list. Key packages:

| Package | Purpose |
|---------|---------|
| `torch` | Neural networks & optimisation |
| `gymnasium` + `mujoco` | MuJoCo simulation environments (v4) |
| `h5py` | Loading D4RL offline datasets (HDF5) |
| `matplotlib` | Plotting learning curves & diagnostics |
| `tensorboard` | Optional live training dashboards |
| `tqdm` | Progress bars |

## Quick Start

```bash
# Verify everything works (downloads halfcheetah-medium-v2 on first run)
python smoke_test.py

# Single quick run (20k steps, one seed)
python run_phase1.py --quick --algo cql --env halfcheetah-medium-v2

# Full Phase I grid (CQL × IQL × clean/k=30/k=60 × 3 seeds × 2 envs)
python run_phase1.py --device auto

# Re-generate plots from existing results
python run_phase1.py --plot_only
```

## Key CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--algo` | both | `cql` or `iql` |
| `--env` | both | D4RL env id, e.g. `halfcheetah-medium-v2` |
| `--remove_top_k` | 0,30,60 | % of top trajectories to remove |
| `--seed` | 0,1,2 | Random seed |
| `--total_steps` | 500k | Gradient steps |
| `--device` | `auto` | `cpu`, `cuda`, `mps`, or `auto` |
| `--quick` | — | 20k steps, single seed |

## Corruption Strategies

1. **Top-k% trajectory removal** – removes the highest-return trajectories
   (k ∈ {30, 60}) to simulate missing high-quality demonstrations.
2. **Gaussian reward noise** (stretch goal) – adds N(0, σ²) noise to reward
   signals to test robustness to noisy supervision.

## Metrics & Plots

- **Normalized return** (D4RL convention): 100 × (score − random) / (expert − random)
- **Q-value diagnostics**: mean Q ± std over training
- **Rolling critic loss variance**: detects value collapse / instability
- **Corruption comparison bar chart**: final performance vs. corruption level