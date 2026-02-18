#!/usr/bin/env python3
"""
Minimal smoke test: verifies imports, dataset loading, corruption,
one gradient step for each algorithm, and evaluation.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import torch

# ---- 1. Dataset loading -------------------------------------------------- #
print("=" * 60)
print("1. Loading D4RL dataset ...")
from src.data.loader import load_d4rl_dataset, get_d4rl_normalization_scores

dataset, env = load_d4rl_dataset("halfcheetah-medium-v2")
print(f"   {dataset.summary()}")

random_score, expert_score = get_d4rl_normalization_scores("halfcheetah-medium-v2")
print(f"   Reference scores: random={random_score}, expert={expert_score}")

# ---- 2. Corruption ------------------------------------------------------ #
print("\n" + "=" * 60)
print("2. Applying corruption ...")
from src.data.corruption import remove_top_k_trajectories, add_reward_noise

ds_30 = remove_top_k_trajectories(dataset, k_pct=30)
print(f"   After removing top 30%: {ds_30.summary()}")

ds_noisy = add_reward_noise(dataset, noise_std=1.0, seed=42)
print(f"   After Gaussian noise: {ds_noisy.summary()}")

# ---- 3. Sampling -------------------------------------------------------- #
print("\n" + "=" * 60)
print("3. Sampling a batch ...")
batch = dataset.sample(batch_size=64, device="cpu")
for k, v in batch.items():
    print(f"   {k}: shape={v.shape}, dtype={v.dtype}")

# ---- 4. CQL gradient step ----------------------------------------------- #
print("\n" + "=" * 60)
print("4. CQL: one gradient step ...")
from src.algos.cql import CQL, CQLConfig

cql_cfg = CQLConfig(batch_size=64, cql_n_samples=4)  # small for speed
cql_agent = CQL(dataset.obs_dim, dataset.act_dim, cql_cfg, device="cpu")
metrics = cql_agent.update(batch)
print(f"   CQL metrics: { {k: f'{v:.4f}' for k, v in metrics.items()} }")

# ---- 5. IQL gradient step ----------------------------------------------- #
print("\n" + "=" * 60)
print("5. IQL: one gradient step ...")
from src.algos.iql import IQL, IQLConfig

iql_cfg = IQLConfig(batch_size=64)
iql_agent = IQL(dataset.obs_dim, dataset.act_dim, iql_cfg, device="cpu")
metrics = iql_agent.update(batch)
print(f"   IQL metrics: { {k: f'{v:.4f}' for k, v in metrics.items()} }")

# ---- 6. Policy evaluation ----------------------------------------------- #
print("\n" + "=" * 60)
print("6. Evaluating random CQL policy (1 episode) ...")
from src.eval.metrics import evaluate_policy, compute_normalized_return

eval_res = evaluate_policy(cql_agent, env, n_episodes=1)
norm = compute_normalized_return(eval_res["return_mean"], random_score, expert_score)
print(f"   Raw return: {eval_res['return_mean']:.1f}")
print(f"   Normalized return: {norm:.1f}")

print("\n" + "=" * 60)
print("OK  All smoke tests passed!")
