"""
D4RL dataset loading utilities.

Downloads the HDF5 files directly from the D4RL repository and creates
Gymnasium environments using the modern ``mujoco`` backend (no legacy
``mujoco-py`` / MuJoCo 210 required).
"""

from __future__ import annotations
import os
import urllib.request
from pathlib import Path
from typing import Tuple

import gymnasium as gym
import h5py
import numpy as np

from src.data.dataset import OfflineDataset


# --------------------------------------------------------------------------- #
#  Reference scores for normalized-return computation
#  Source: D4RL paper Table 1 (Fu et al., 2020).
# --------------------------------------------------------------------------- #

_D4RL_REF_SCORES = {
    "halfcheetah": {"random": -280.18, "expert": 12135.0},
    "hopper":      {"random": -20.27,  "expert": 3234.3},
    "walker2d":    {"random": 1.63,    "expert": 4592.3},
}


def get_d4rl_normalization_scores(env_name: str) -> Tuple[float, float]:
    """Return (random_score, expert_score) for normalized-return calc.

    *env_name* can be a full D4RL id like ``halfcheetah-medium-v2`` or just
    the task prefix ``halfcheetah``.
    """
    prefix = env_name.split("-")[0].lower()
    scores = _D4RL_REF_SCORES.get(prefix)
    if scores is None:
        raise ValueError(
            f"No reference scores for '{prefix}'. "
            f"Available: {list(_D4RL_REF_SCORES.keys())}"
        )
    return scores["random"], scores["expert"]


# --------------------------------------------------------------------------- #
#  D4RL HDF5 download + parsing
# --------------------------------------------------------------------------- #

# Map D4RL dataset name -> (download URL, Gymnasium env id).
_DATASET_URLS = {
    "halfcheetah-medium-v2":        "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_medium-v2.hdf5",
    "halfcheetah-medium-replay-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_medium_replay-v2.hdf5",
    "halfcheetah-medium-expert-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_medium_expert-v2.hdf5",
    "hopper-medium-v2":             "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_medium-v2.hdf5",
    "hopper-medium-replay-v2":      "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_medium_replay-v2.hdf5",
    "hopper-medium-expert-v2":      "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_medium_expert-v2.hdf5",
}

_GYM_ENV_MAP = {
    "halfcheetah": "HalfCheetah-v4",
    "hopper":      "Hopper-v4",
    "walker2d":    "Walker2d-v4",
}

SUPPORTED_DATASETS = list(_DATASET_URLS.keys())

DATA_DIR = Path(os.environ.get("D4RL_DATA_DIR", Path.home() / ".d4rl_datasets"))


def _download_dataset(env_name: str) -> Path:
    """Download the HDF5 file if not already cached."""
    url = _DATASET_URLS.get(env_name)
    if url is None:
        raise ValueError(
            f"Unknown dataset '{env_name}'. Supported: {SUPPORTED_DATASETS}"
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    local_path = DATA_DIR / f"{env_name}.hdf5"

    if local_path.exists():
        return local_path

    print(f"[loader] Downloading {env_name} -> {local_path} ...")
    urllib.request.urlretrieve(url, str(local_path))
    print(f"[loader] Download complete ({local_path.stat().st_size / 1e6:.1f} MB)")
    return local_path


def _load_hdf5(path: Path) -> dict:
    """Read an HDF5 D4RL file into a flat dict of numpy arrays.

    Only reads top-level datasets (ignores nested groups like metadata/).
    """
    data = {}
    with h5py.File(str(path), "r") as f:
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                data[key] = f[key][:]
    return data


def _make_gymnasium_env(env_name: str) -> gym.Env:
    """Create the corresponding Gymnasium (v4) environment."""
    prefix = env_name.split("-")[0].lower()
    gym_id = _GYM_ENV_MAP.get(prefix)
    if gym_id is None:
        raise ValueError(f"No Gymnasium env for prefix '{prefix}'")
    return gym.make(gym_id)


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #

def load_d4rl_dataset(env_name: str) -> Tuple[OfflineDataset, gym.Env]:
    """Load a D4RL dataset and return an ``OfflineDataset`` + the Gym env.

    The HDF5 file is downloaded from the D4RL servers on first use and
    cached under ``~/.d4rl_datasets/``.

    Parameters
    ----------
    env_name : str
        A valid D4RL environment id, e.g. ``"halfcheetah-medium-v2"``.

    Returns
    -------
    dataset : OfflineDataset
    env     : gymnasium.Env
    """
    hdf5_path = _download_dataset(env_name)
    raw = _load_hdf5(hdf5_path)

    # HDF5 keys: observations, actions, rewards, terminals, (timeouts, next_observations ...)
    observations = raw["observations"].astype(np.float32)
    actions = raw["actions"].astype(np.float32)
    rewards = raw["rewards"].astype(np.float32).flatten()
    terminals = raw["terminals"].astype(np.float32).flatten()
    timeouts = raw.get("timeouts", np.zeros_like(terminals)).astype(np.float32).flatten()

    # Use next_observations from file if available; otherwise shift.
    if "next_observations" in raw:
        next_observations = raw["next_observations"].astype(np.float32)
    else:
        next_observations = np.roll(observations, -1, axis=0).copy()
        ends = np.where((terminals + timeouts) > 0)[0]
        if len(observations) - 1 not in ends:
            ends = np.append(ends, len(observations) - 1)
        next_observations[ends] = observations[ends]

    dataset = OfflineDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=terminals,
        timeouts=timeouts,
    )

    env = _make_gymnasium_env(env_name)
    return dataset, env
