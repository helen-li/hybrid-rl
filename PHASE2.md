# Phase II: Uncertainty-Driven Online Recovery

**Goal (from proposal):** Investigate whether uncertainty-aware exploration can improve online fine-tuning from corrupted offline models, enabling faster performance recovery compared to naive online fine-tuning.

---

## 1. What We Are Doing

1. **Load** Phase I offline checkpoints (CQL and IQL trained on clean and corrupted D4RL datasets).
2. **Extend** the critic to an **ensemble of Q-networks** (default 3, configurable up to 5) for uncertainty estimation.
3. **Fine-tune online** with environment interaction, using **ensemble disagreement** as an exploration bonus added to rewards.
4. **Compare** ensemble-bonus fine-tuning against vanilla fine-tuning (no bonus) to measure the benefit of uncertainty-driven exploration.
5. **Evaluate** sample efficiency, asymptotic performance, and training stability during the offline-to-online transition.

---

## 2. Environment and Data

| Item | Choice |
|------|--------|
| **Benchmark** | D4RL MuJoCo locomotion (same as Phase I) |
| **Environments** | HalfCheetah, Hopper (`halfcheetah-medium-v2`, `hopper-medium-v2`) |
| **Offline data** | Same corrupted D4RL datasets from Phase I, used for hybrid replay |
| **Online data** | New transitions collected by interacting with the environment during fine-tuning |

---

## 3. Ensemble Q-Networks

| Component | Description | Implementation |
|-----------|-------------|----------------|
| **EnsembleQNetwork** | N independent Q-network MLPs (same architecture as Phase I: [256, 256]) | `src/algos/networks.py` |
| **Initialization** | First 2 members loaded from Phase I `DoubleQNetwork` checkpoint (q1, q2); remaining members randomly initialized for diversity | `load_double_q()` in `EnsembleQNetwork` |
| **Pessimistic estimate** | `q_min`: minimum Q-value across all ensemble members, used for Bellman targets and actor loss | Same role as `min(q1, q2)` in Phase I |
| **Disagreement signal** | `disagreement`: standard deviation of Q-values across members — high std means the agent is uncertain | Used as the exploration bonus |

The number of ensemble members is configurable via `--n_ensemble` (default 3, supports 3–5).

---

## 4. Exploration Bonus

During online data collection, an exploration bonus is added to the environment reward before storing in the replay buffer:

**r' = r + λ × std(Q₁(s, a), Q₂(s, a), ..., Qₙ(s, a))**

| Parameter | Description | Default |
|-----------|-------------|---------|
| **λ** (`bonus_coeff`) | Scaling coefficient for the exploration bonus | 1.0 |
| **bonus_type** | `"ensemble"` (disagreement bonus) or `"none"` (vanilla baseline) | `"ensemble"` |

The offline dataset retains its original rewards — only online transitions get the bonus. This incentivizes the agent to visit states where the ensemble disagrees (i.e., where the offline data provided insufficient coverage).

---

## 5. Online Fine-Tuning Loop

| Step | Description |
|------|-------------|
| **1. Load checkpoint** | Load Phase I offline checkpoint into `OnlineCQL` or `OnlineIQL` agent with ensemble critic |
| **2. Collect transitions** | Agent interacts with the environment using its current policy (stochastic, not deterministic) |
| **3. Compute bonus** | For each transition, compute ensemble disagreement and add `λ × disagreement` to the reward |
| **4. Store in replay buffer** | Online transitions (with bonus-augmented rewards) are stored in a `ReplayBuffer` |
| **5. Hybrid replay** | Each training batch is split: `online_ratio` (default 50%) from the online buffer, the rest from the offline dataset |
| **6. Update agent** | Standard CQL or IQL update using the mixed batch |
| **7. Evaluate periodically** | Evaluate policy in the environment every `eval_interval` steps |

**CQL penalty reduction:** The CQL conservative penalty α is reduced from 5.0 (offline) to 1.0 (online) to allow the agent to explore beyond the offline data distribution.

---

## 6. Experiment Grid

**Full grid:**

- **Algorithms:** CQL, IQL
- **Environments:** HalfCheetah-medium-v2, Hopper-medium-v2
- **Corruption:** clean (k=0), k=30, k=60 (loads corresponding Phase I checkpoint)
- **Bonus type:** ensemble, none (vanilla baseline)
- **Seeds:** 0, 1, 2

**Total runs:** 2 algos × 2 envs × 3 corruption levels × 2 bonus types × 3 seeds = **72 fine-tuning runs**.

Each run: 250,000 online steps (5,000 for quick smoke tests), with periodic evaluation every 5,000 steps.

**Outputs per run:** `config.json`, `metrics.json` (steps, normalized returns, ensemble disagreement, critic loss), `checkpoint.pt`.

---

## 7. Metrics

| Metric | Purpose |
|--------|---------|
| **Normalized return** | D4RL convention: 100 × (score − random) / (expert − random). Primary performance measure. |
| **Fine-tuning curves** | Normalized return vs. online steps, comparing ensemble bonus vs. vanilla for each corruption level. Solid lines = ensemble, dashed = vanilla. |
| **Sample efficiency** | Number of online steps required to reach a fixed performance threshold (80% of clean offline performance). Lower is better. |
| **Ensemble disagreement** | Mean std across ensemble members over training. Should start high (diverse init) and decrease as members converge. |
| **Offline checkpoint performance** | Step-0 evaluation before any online training — baseline to measure recovery from. |

Implemented in `src/eval/metrics.py` and `src/eval/plotting.py`; fine-tuning loop in `src/finetune.py` logs and saves these.

---

## 8. Deliverables (Artifacts)

1. **Results directory** (`results_phase2/` by default): one folder per run (e.g. `ft_cql_halfcheetah-medium-v2_k30_noise0.0_ensemble_s0`) containing:
   - `config.json` (full fine-tuning configuration)
   - `metrics.json` (steps, normalized_return, ensemble_disagreement, critic_loss_avg, etc.)
   - `checkpoint.pt` (fine-tuned model weights)
2. **Plots** (in `plots_phase2/`):
   - Fine-tuning curves per algo+env (ensemble vs. vanilla, across corruption levels, mean ± std over seeds).
   - Sample efficiency bar chart: online steps to threshold by method and corruption level, per env.
3. **Summary** (for the report): quantitative and qualitative description of how ensemble-driven exploration affects recovery speed, asymptotic performance, and stability compared to vanilla fine-tuning.

---

## 9. Execution Checklist

- [ ] **Phase I checkpoints ready:** Ensure all required Phase I runs are complete (at minimum seed 0 for all 12 algo × env × corruption combinations in `results/`).
- [ ] **Quick sanity check:**
  `conda run -n hybrid-rl python run_phase2.py --quick --algo cql --env halfcheetah-medium-v2 --remove_top_k 0`
  (5k online steps, single seed; confirms checkpoint loading, ensemble construction, online collection, hybrid replay, and evaluation.)
- [ ] **Single full run (optional):** e.g.
  `conda run -n hybrid-rl python run_phase2.py --algo cql --env halfcheetah-medium-v2 --remove_top_k 30 --bonus_type ensemble --seed 0`
  to validate one full-length run and metrics.
- [ ] **Full Phase II grid:**
  `conda run -n hybrid-rl python run_phase2.py --device auto`
  (runs all 72 experiments; can be batched by algo/env if needed.)
- [ ] **Plots from existing results:**
  `conda run -n hybrid-rl python run_phase2.py --plot_only`
  (regenerate all Phase II figures from current `results_phase2/`.)
- [ ] **Report:** Summarize fine-tuning curves, sample efficiency, and disagreement trends; describe how ensemble bonus compares to vanilla fine-tuning across corruption levels.

---

## 10. Success Criteria for Phase II

- All 72 runs complete and produce valid `metrics.json`.
- Fine-tuning curves clearly show:
  - Performance recovery from corrupted offline checkpoints over online steps.
  - Ensemble bonus runs recover faster (steeper curves) than vanilla, especially at high corruption (k=60).
- Sample efficiency plots show fewer online steps needed with ensemble bonus vs. vanilla.
- Ensemble disagreement decreases over training as members converge (sanity check).
- Findings are documented for the final report with comparison to Phase I offline-only baselines.

---

## 11. References to Code

| Component | Location |
|-----------|----------|
| Entry point / experiment grid | `run_phase2.py` |
| Online fine-tuning loop | `src/finetune.py` |
| Ensemble Q-network | `src/algos/networks.py` (`EnsembleQNetwork`) |
| Online CQL (ensemble critic) | `src/algos/online_cql.py` |
| Online IQL (ensemble critic) | `src/algos/online_iql.py` |
| Offline dataset / replay buffer | `src/data/dataset.py` |
| Data loading, corruption | `src/data/loader.py`, `src/data/corruption.py` |
| Normalized return, evaluation | `src/eval/metrics.py` |
| Fine-tuning curves, sample efficiency plots | `src/eval/plotting.py` |
| Phase I checkpoints (input) | `results/` |
| Phase II results (output) | `results_phase2/` |
| Phase II plots (output) | `plots_phase2/` |