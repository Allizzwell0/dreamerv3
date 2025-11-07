#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a trained DreamerV3 policy on the AUVEnv and export trajectories to CSV,
along with summary metrics.

Usage:
  python eval_auv.py --ckpt ~/logdir/auv/20251106T161609 --episodes 20 --out_csv eval_trajectories.csv

Notes:
- If DreamerV3 checkpoint loading fails or isn't provided, a RandomFallbackPolicy is used
  so the pipeline still runs end-to-end.
- Works with the AUVEnv you shared (8-dim vector obs). If your obs includes x,y,goal_x,goal_y
  (12-dim), this script will auto-detect and record them into the CSV.
"""
import os
import csv
import math
import json
import time
import argparse
from pathlib import Path
import numpy as np

# ===== MODIFY THIS IMPORT TO MATCH YOUR ENV PATH =====
# Example: from auv_env import AUVEnv
try:
    from embodied.envs.AUV_Env import AUVEnv  
except Exception as e:
    raise ImportError("Failed to import AUVEnv. Please edit the import path in eval_auv.py "
                      "to point to your environment class.\n"
                      f"Original error: {e}")


class RandomFallbackPolicy:
    """Fallback policy for testing the evaluation pipeline if a real policy isn't available."""
    def reset(self):
        pass
    def __call__(self, obs):
        a = np.random.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)
        return {'reset': False, 'action': a}


def load_trained_policy(checkpoint_dir=None):
    """
    Try to load a DreamerV3 policy from a checkpoint directory.
    Expected structure (from your logdir):
      ckpt/ , config.yaml , metrics.jsonl , ...
    Returns an object with .reset() and __call__(obs)->action_dict
    If loading fails, returns RandomFallbackPolicy.
    """
    if checkpoint_dir is None:
        print("[eval_auv] No checkpoint provided. Using RandomFallbackPolicy.")
        return RandomFallbackPolicy()

    checkpoint_dir = str(Path(checkpoint_dir).expanduser())
    config_path = Path(checkpoint_dir) / "config.yaml"
    ckpt_dir = Path(checkpoint_dir) / "ckpt"
    if not config_path.exists():
        # maybe the user passed the parent directory
        parent_config = Path(checkpoint_dir).parent / "config.yaml"
        if parent_config.exists():
            config_path = parent_config

    try:
        import embodied
        from embodied.core import checkpoint as ckpt_core
        # DreamerV3 agent import may differ by install; try common paths:
        try:
            from dreamerv3.agent import Agent as DreamerAgent
        except Exception:
            # Fallback older/newer layouts
            from dreamerv3 import agent as dreamer_agent
            DreamerAgent = dreamer_agent.Agent

        # Load config; embodied.Config.load_yaml usually expects a path string
        if config_path.exists():
            cfg = embodied.Config(embodied.Config.load_yaml(str(config_path)))
        else:
            print(f"[eval_auv] Warning: config.yaml not found near {checkpoint_dir}. Using defaults.")
            cfg = embodied.Config()

        # logdir should be the run folder (the directory with ckpt/ etc.)
        cfg.logdir = checkpoint_dir

        agent = DreamerAgent(cfg)

        # Load weights
        # Depending on version, you either load from run dir or ckpt dir
        if ckpt_dir.exists():
            ck = ckpt_core.Checkpoint(str(ckpt_dir))
        else:
            ck = ckpt_core.Checkpoint(str(checkpoint_dir))
        ck.load(agent)
        print(f"[eval_auv] Loaded DreamerV3 agent from {checkpoint_dir}")

        class Policy:
            def __init__(self, agent):
                self.agent = agent
                self.state = None
            def reset(self):
                self.state = None
            def __call__(self, obs):
                # Expect obs dict with 'vector' etc.
                action, self.state = self.agent.policy(obs, self.state, mode='eval')
                # Ensure correct shape/keys for AUVEnv
                if isinstance(action, dict):
                    # passthrough if already in expected dict format
                    return action
                else:
                    # assume continuous action array of size 2
                    arr = np.asarray(action, dtype=np.float32).reshape(-1)
                    if arr.size < 2:
                        # pad if needed
                        arr = np.pad(arr, (0, 2 - arr.size))
                    return {'reset': False, 'action': arr[:2]}

        return Policy(agent)

    except Exception as e:
        print(f"[eval_auv] Could not load Dreamer policy from '{checkpoint_dir}': {e}")
        print("[eval_auv] Falling back to RandomFallbackPolicy.")
        return RandomFallbackPolicy()


def evaluate_auv(
    policy,
    episodes=20,
    dt=0.05,
    max_steps=500,
    out_csv="eval_trajectories.csv",
    seed=0,
    verbose=True,
):
    env = AUVEnv(dt=dt, max_steps=max_steps)
    np.random.seed(seed)

    header = [
        "episode", "t", "reward", "discount",
        "x", "y", "theta", "u", "v", "r",
        "goal_x", "goal_y",
        "xe", "ye", "dist",
        "is_terminal", "is_last"
    ]
    out_csv = str(Path(out_csv).expanduser())
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        ep_returns = []
        ep_lengths = []
        successes = 0
        final_dists = []

        for ep in range(episodes):
            traj = env.step({'reset': True})
            policy.reset()
            done = False
            ep_return = 0.0
            t = 0

            vec = traj['vector']  # [xe, ye, dist, cosθ, sinθ, u, v, r, (maybe x,y,goal_x,goal_y)]
            # Base fields from 8-dim obs
            theta = float(math.atan2(vec[4], vec[3]))
            u, v, r = map(float, vec[5:8])
            xe, ye, dist = float(vec[0]), float(vec[1]), float(vec[2])
            # Optional world/goal if provided (12-dim)
            x = y = goal_x = goal_y = float('nan')
            if len(vec) >= 12:
                x, y = float(vec[8]), float(vec[9])
                goal_x, goal_y = float(vec[10]), float(vec[11])

            writer.writerow([ep, t, 0.0, 1.0, x, y, theta, u, v, r,
                             goal_x, goal_y, xe, ye, dist, False, False])

            for t in range(1, max_steps + 1):
                action = policy(traj)
                traj = env.step(action)
                vec = traj['vector']

                theta = float(math.atan2(vec[4], vec[3]))
                u, v, r = map(float, vec[5:8])
                xe, ye, dist = float(vec[0]), float(vec[1]), float(vec[2])

                # Optional world/goal
                if len(vec) >= 12:
                    x, y = float(vec[8]), float(vec[9])
                    goal_x, goal_y = float(vec[10]), float(vec[11])
                else:
                    x = y = goal_x = goal_y = float('nan')

                reward = float(traj['reward'])
                discount = float(traj.get('discount', 1.0))
                is_last = bool(traj['is_last'])
                is_terminal = bool(traj['is_terminal'])

                ep_return += reward

                writer.writerow([ep, t, reward, discount, x, y, theta, u, v, r,
                                 goal_x, goal_y, xe, ye, dist, is_terminal, is_last])

                if is_last:
                    done = True
                    final_dists.append(dist)
                    if is_terminal:
                        successes += 1
                    break

            ep_returns.append(ep_return)
            ep_lengths.append(t)

            if verbose:
                print(f"[Episode {ep:03d}] return={ep_return:.2f} steps={t} "
                      f"success={'Y' if done and traj['is_terminal'] else 'N'} "
                      f"final_dist={final_dists[-1] if final_dists else float('nan'):.3f}")

    metrics = {
        "episodes": episodes,
        "success_rate": successes / episodes if episodes else 0.0,
        "avg_return": float(np.mean(ep_returns)) if ep_returns else 0.0,
        "std_return": float(np.std(ep_returns)) if ep_returns else 0.0,
        "avg_ep_len": float(np.mean(ep_lengths)) if ep_lengths else 0.0,
        "final_dist_mean": float(np.mean(final_dists)) if final_dists else float("nan"),
        "final_dist_std": float(np.std(final_dists)) if final_dists else float("nan"),
        "csv_path": os.path.abspath(out_csv),
    }
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_csv", type=str, default="eval_trajectories.csv")
    p.add_argument("--ckpt", type=str, default=None, help="DreamerV3 run dir (contains ckpt/ and config.yaml)")
    args = p.parse_args()

    policy = load_trained_policy(args.ckpt)
    m = evaluate_auv(policy,
                     episodes=args.episodes,
                     dt=args.dt,
                     max_steps=args.max_steps,
                     out_csv=args.out_csv,
                     seed=args.seed,
                     verbose=True)
    print("\n=== Evaluation Summary ===")
    for k, v in m.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
