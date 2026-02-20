# Phase I: Characterizing Failure Modes in Offline RL

**Goal (from proposal):** Characterize how severely existing offline RL algorithms degrade under realistic dataset flaws (missing high-quality trajectories, noisy rewards), and how different forms and degrees of corruption affect offline learning and stability.

---

## 1. What We Are Doing

1. **Train** two representative offline RL algorithms—**CQL** and **IQL**—on D4RL MuJoCo datasets.
2. **Corrupt** those datasets in controlled ways (top-*k*% trajectory removal; optionally Gaussian reward noise).
3. **Evaluate** performance (normalized returns) and **stability** (loss variance, Q-value fluctuations) across clean vs. corrupted conditions.
4. **Analyze and report** how corruption type and severity affect learning curves and final performance.

No online fine-tuning yet; Phase I is purely offline training and diagnostics.

---

## 2. Environment and Data

| Item | Choice |
|------|--------|
| **Benchmark** | D4RL MuJoCo locomotion |
| **Environments** | HalfCheetah, Hopper (e.g. `halfcheetah-medium-v2`, `hopper-medium-v2`) |
| **Data source** | D4RL HDF5 datasets, loaded via project data loader (no legacy MuJoCo 210 / `mujoco-py`) |

Datasets are fetched from D4RL servers; reference scores (random, expert) are used for normalized-return computation.

---

## 3. Corruption Strategies

| Strategy | Description | Implementation |
|----------|-------------|----------------|
| **Top-*k*% trajectory removal** | Remove the highest-return *k*% of trajectories to simulate missing high-quality data. | `k ∈ {0, 30, 60}` — 0 = clean baseline. |
| **Gaussian reward noise** (stretch) | Add N(0, σ²) noise to rewards to test robustness to noisy supervision. | Optional; can be combined with top-*k* for additional experiments. |

Corruption is applied in `src/data/corruption.py`; the loader and training pipeline support `remove_top_k` and reward-noise parameters.

---

## 4. Algorithms

| Algorithm | Role |
|-----------|------|
| **CQL** (Conservative Q-Learning) | Pessimistic value estimation; penalizes Q-values on OOD actions. |
| **IQL** (Implicit Q-Learning) | Expectile regression for value; avoids explicit maximization over actions. |

Both are implemented in `src/algos/` and trained via the unified loop in `src/train.py`.

---

## 5. Experiment Grid

**Full grid:**

- **Algorithms:** CQL, IQL  
- **Environments:** HalfCheetah-medium-v2, Hopper-medium-v2  
- **Corruption:** clean (k=0), k=30, k=60  
- **Seeds:** 0, 1, 2  

**Total runs:** 2 algos × 2 envs × 3 corruption levels × 3 seeds = **36 training runs**.

Each run: offline training for a fixed number of gradient steps (e.g. 500k full, 20k for quick sanity checks), with periodic evaluation in the environment.

**Outputs per run:** checkpoint(s), `metrics.json` (steps, normalized returns, Q-stats, critic loss, rolling loss variance, etc.).

---

## 6. Metrics

| Metric | Purpose |
|--------|---------|
| **Normalized return** | D4RL convention: 100 × (score − random) / (expert − random). Primary performance measure. |
| **Learning curves** | Normalized return vs. training step (mean ± std over seeds) to see degradation and stability over time. |
| **Q-value diagnostics** | Mean Q ± std over training; helps detect value collapse or instability. |
| **Rolling critic loss variance** | Variance of critic loss over a window; highlights training instability. |
| **Final performance vs. corruption** | Bar chart of final normalized return by algorithm and corruption level (clean, k=30, k=60). |

Implemented in `src/eval/metrics.py` and `src/eval/plotting.py`; training loop in `src/train.py` logs and saves these.

---

## 7. Deliverables (Artifacts)

1. **Results directory** (`results/` by default): one folder per run (e.g. `cql_halfcheetah-medium-v2_k30_noise0.0_s0`) containing:
   - `metrics.json` (steps, normalized_return, Q-stats, critic_loss_rolling_var, etc.)
   - Checkpoints if saved
2. **Plots** (in `plots/`):
   - Learning curves per environment (all algos × corruption levels, mean ± std over seeds).
   - Corruption comparison bar chart: final normalized return by algo and corruption level, per env.
   - Q-value diagnostics: Q mean ± std over training (e.g. clean data, one per algo per env).
   - Loss variance: rolling critic loss variance over training, per algo and corruption level, per env.
3. **Summary** (for the report): qualitative and quantitative description of how corruption affects learning curves, final performance, and stability (loss variance, Q behavior).

---

## 8. Execution Checklist

- [ ] **Environment:** Venv created and dependencies installed (`pip install -r requirements.txt`); `python smoke_test.py` passes.
- [ ] **Quick sanity check:**  
  `python run_phase1.py --quick --algo cql --env halfcheetah-medium-v2`  
  (short run, single seed; confirms data load, corruption, training, eval, and logging.)
- [ ] **Single full run (optional):** e.g.  
  `python run_phase1.py --algo cql --env halfcheetah-medium-v2 --remove_top_k 30 --seed 0`  
  to validate one full-length run and metrics/checkpoints.
- [ ] **Full Phase I grid:**  
  `python run_phase1.py --device auto`  
  (runs all 36 experiments; can be parallelized or batched by env/algo if needed.)
- [ ] **Plots from existing results:**  
  `python run_phase1.py --plot_only`  
  (regenerate all Phase I figures from current `results/`.)
- [ ] **Report:** Summarize learning curves, bar charts, Q and loss-variance plots; describe how CQL and IQL degrade with k=30 and k=60 and any stability issues.

---

## 9. Success Criteria for Phase I

- All 36 runs complete and produce valid `metrics.json`.
- Learning curves and bar charts clearly show:
  - Performance drop as corruption increases (clean → k=30 → k=60).
  - Differences between CQL and IQL under the same corruption.
- Q-value and loss-variance plots reveal any value collapse or training instability (e.g. high variance, divergence) under corruption.
- Findings are documented for use in Phase II (uncertainty-driven online recovery).

---

## 10. References to Code

| Component | Location |
|-----------|----------|
| Entry point / experiment grid | `run_phase1.py` |
| Data loading, D4RL HDF5 | `src/data/loader.py` |
| Corruption (top-*k*, reward noise) | `src/data/corruption.py` |
| Offline dataset / replay | `src/data/dataset.py` |
| CQL / IQL / networks | `src/algos/` |
| Training loop | `src/train.py` |
| Normalized return, Q-stats | `src/eval/metrics.py` |
| Learning curves, bar chart, Q diag, loss variance | `src/eval/plotting.py` |
