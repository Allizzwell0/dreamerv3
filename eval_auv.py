#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a trained DreamerV3 policy on the (CONTINUOUS-ACTION) AUVEnv and export trajectories to CSV,
along with summary metrics.

Usage:
  python eval_auv.py --ckpt ~/logdir/auv/20251106T161609 \
    --episodes 200 --out_dir ~/logdir/auv/20251106T161609/eval_output

Notes:
- This version assumes your AUVEnv uses CONTINUOUS actions (e.g. action ∈ [-1,1]^2).
- If DreamerV3 checkpoint loading fails or isn't provided, a RandomContinuousPolicy is used.
- If obs includes x,y,goal_x,goal_y (12-dim), they are auto-recorded into the CSV.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np

# ===== MODIFY THIS IMPORT TO MATCH YOUR ENV PATH =====
try:
    from embodied.envs.AUV_Env import AUVEnv
except Exception as e:
    raise ImportError(
        "Failed to import AUVEnv. Please edit the import path in eval_auv.py "
        "to point to your environment class.\n"
        f"Original error: {e}"
    )


# ------------ 连续动作 fallback 策略 ------------

class RandomContinuousPolicy:
    """连续动作的 fallback 策略：action ∈ [low, high]^n."""
    def __init__(self, act_low, act_high, seed: int = 0):
        self.low = np.array(act_low, dtype=np.float32)
        self.high = np.array(act_high, dtype=np.float32)
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def __call__(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        _ = obs
        a = self.rng.uniform(self.low, self.high)
        return {"reset": False, "action": a.astype(np.float32)}


# ------------ 按 main + eval_only 风格加载 DreamerV3 agent 权重（连续动作） ------------

def load_trained_policy(
    checkpoint_dir: Optional[str],
    act_shape,
    act_low,
    act_high,
    seed: int = 0,
):
    """
    连续动作版：
    - 用 dreamerv3/main.py 里的 make_agent(config) 构造 Agent（和训练完全一致）
    - 用 elements.Checkpoint() 加载 agent 权重（风格参照 embodied/run/eval_only.py）
    - 输出连续动作向量（直接传给 AUVEnv）
    """
    if checkpoint_dir is None:
        print("[eval_auv] No checkpoint provided. Using RandomContinuousPolicy.")
        return RandomContinuousPolicy(act_low, act_high, seed)

    run_dir = Path(checkpoint_dir).expanduser()
    if not run_dir.exists():
        print(f"[eval_auv] Checkpoint dir '{run_dir}' does not exist. Using RandomContinuousPolicy.")
        return RandomContinuousPolicy(act_low, act_high, seed)

    # 优先在 run_dir/config.yaml 找 config；找不到再尝试父目录
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        parent_config = run_dir.parent / "config.yaml"
        if parent_config.exists():
            config_path = parent_config

    if not config_path.exists():
        print(f"[eval_auv] config.yaml not found near '{run_dir}'. Using RandomContinuousPolicy.")
        return RandomContinuousPolicy(act_low, act_high, seed)

    try:
        import elements
        import ruamel.yaml as yaml
        from dreamerv3 import main as dv3_main

        # -------- 1) 读取 config.yaml -> elements.Config --------
        cfg_text = config_path.read_text(encoding="utf-8")
        raw_cfg = yaml.YAML(typ="safe").load(cfg_text)
        config = elements.Config(raw_cfg)
        # 确保 logdir 指向当前 run_dir
        config = config.update(logdir=str(run_dir))
        # print(f"[eval_auv] Loaded DreamerV3 config from '{config_path}'.")

        # -------- 2) 用 main 里的 make_agent 构造 Agent --------
        # 和训练时完全一致（内部会调用 make_env + wrap_env）
        agent = dv3_main.make_agent(config)
        # print(f"[eval_auv] Created DreamerV3 agent from config at '{config_path}'.")

        # -------- 3) 用 elements.Checkpoint 加载 agent 权重 --------
        # 参照 embodied/run/eval_only.py:
        #   cp = elements.Checkpoint()
        #   cp.agent = agent
        #   cp.load(args.from_checkpoint, keys=['agent'])
        import elements as _elements  # 复用同一个 elements 模块
        cp = _elements.Checkpoint()
        cp.agent = agent
        # print(f"[eval_auv] Loading DreamerV3 agent weights from checkpoint...")

        ckpt_root = run_dir / "ckpt"
        # 如果传进来的是 run_dir（包含 ckpt/），优先从 ckpt/ 加载；否则尝试直接从 run_dir 加载
        load_root = ckpt_root if ckpt_root.exists() else run_dir

        cp.load(str(load_root), keys=["agent"])
        print(f"[eval_auv] Loaded DreamerV3 agent weights from {load_root}")

        # -------- 4) 连续动作封装（直接把 Dreamer 的动作传给 env） --------
        act_shape = tuple(act_shape)

        class ContinuousPolicyWrapper:
            def __init__(self, agent_, act_shape_, seed_):
                self.agent = agent_
                self.act_shape = act_shape_
                self.state = None
                self.rng = np.random.default_rng(seed_)

            def reset(self) -> None:
                self.state = None

            def __call__(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
                # Dreamer policy 输出 raw_action，可能是 dict / array / tensor
                raw_action, self.state = self.agent.policy(obs, self.state, mode="eval")

                if isinstance(raw_action, dict):
                    act = raw_action.get("action", raw_action)
                else:
                    act = raw_action

                act = np.asarray(act, dtype=np.float32)

                # 如果是 batched (1,2) 之类，拆成 (2,)
                if act.ndim > 1:
                    act = act.reshape(-1)[: np.prod(self.act_shape)]

                act = act.reshape(self.act_shape)

                # 通常已经在 [-1,1]，保险起见裁一下
                act = np.clip(act, -1.0, 1.0)

                return {"reset": False, "action": act}

        return ContinuousPolicyWrapper(agent, act_shape, seed)

    except Exception as e:
        print(f"[eval_auv] Could not load DreamerV3 policy from '{run_dir}': {e}")
        print("[eval_auv] Falling back to RandomContinuousPolicy.")
        return RandomContinuousPolicy(act_low, act_high, seed)


# ------------ 评估循环：使用连续动作 policy 在 AUVEnv 上 rollout ------------

def evaluate_auv(
    ckpt_dir: Optional[str],
    *,
    episodes: int = 20,
    dt: float = 0.05,
    max_steps: int = 500,
    success_threshold: float = 0.3,
    out_csv: Path,
    seed: int = 0,
    verbose: bool = True,
) -> Dict[str, float]:
    """Roll out multiple episodes, compute success rate, and write trajectories."""

    rng = np.random.default_rng(seed)
    env = AUVEnv(dt=dt, max_steps=max_steps)

    # for reproducibility
    env.np_random.seed(seed)
    np.random.seed(seed)

    # ---- 连续动作信息，从 env.act_space['action'] 中读取 ----
    act_space = env.act_space["action"]
    act_shape = act_space.shape              # 例如 (2,)
    act_low = getattr(act_space, "low", -1.0)
    act_high = getattr(act_space, "high", 1.0)

    # build policy AFTER we know action shape / range
    policy = load_trained_policy(ckpt_dir, act_shape, act_low, act_high, seed)

    header = [
        "episode",
        "t",
        "reward",
        "discount",
        "x",
        "y",
        "theta",
        "u",
        "v",
        "r",
        "goal_x",
        "goal_y",
        "xe",
        "ye",
        "dist",
        "is_terminal",
        "is_last",
    ]

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    ep_returns: list[float] = []
    ep_lengths: list[int] = []
    final_dists: list[float] = []
    successes = 0

    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for ep in range(episodes):
            policy.reset()
            # resample env target/initialization for diversity
            env_seed = int(rng.integers(0, 2**31 - 1))
            env.np_random.seed(env_seed)
            traj = env.step({"reset": True})

            vec = traj["vector"]
            theta = float(math.atan2(vec[4], vec[3]))
            u, v, r_val = map(float, vec[5:8])
            xe, ye, dist = map(float, vec[:3])
            x = y = goal_x = goal_y = float("nan")
            if len(vec) >= 12:
                x, y = float(vec[8]), float(vec[9])
                goal_x, goal_y = float(vec[10]), float(vec[11])

            writer.writerow(
                [ep, 0, 0.0, 1.0, x, y, theta, u, v, r_val, goal_x, goal_y, xe, ye, dist, False, False]
            )

            ep_return = 0.0
            final_dist = dist
            success_flag = dist <= success_threshold
            steps = 0

            for t in range(1, max_steps + 1):
                steps = t
                # 连续动作策略：返回 {"reset": False, "action": np.array(shape=act_shape)}
                action = policy(traj)
                traj = env.step(action)
                vec = traj["vector"]

                theta = float(math.atan2(vec[4], vec[3]))
                u, v, r_val = map(float, vec[5:8])
                xe, ye, dist = map(float, vec[:3])
                if len(vec) >= 12:
                    x, y = float(vec[8]), float(vec[9])
                    goal_x, goal_y = float(vec[10]), float(vec[11])
                else:
                    x = y = goal_x = goal_y = float("nan")

                reward = float(traj["reward"])
                discount = float(traj.get("discount", 1.0))
                is_last = bool(traj["is_last"])
                is_terminal = bool(traj["is_terminal"])

                ep_return += reward
                final_dist = dist
                if dist <= success_threshold:
                    success_flag = True

                writer.writerow(
                    [ep, t, reward, discount, x, y, theta, u, v, r_val, goal_x, goal_y, xe, ye, dist, is_terminal, is_last]
                )

                if is_last:
                    break

            ep_returns.append(ep_return)
            ep_lengths.append(steps)
            final_dists.append(final_dist)
            if success_flag:
                successes += 1

            if verbose:
                status = "SUCCESS" if success_flag else "FAIL"
                print(
                    f"[Episode {ep:03d}] return={ep_return:.2f} steps={steps} "
                    f"status={status} final_dist={final_dist:.3f}"
                )

    metrics = {
        "episodes": episodes,
        "success_rate": successes / episodes if episodes else 0.0,
        "success_count": successes,
        "avg_return": float(np.mean(ep_returns)) if ep_returns else 0.0,
        "std_return": float(np.std(ep_returns)) if ep_returns else 0.0,
        "avg_ep_len": float(np.mean(ep_lengths)) if ep_lengths else 0.0,
        "final_dist_mean": float(np.mean(final_dists)) if final_dists else float("nan"),
        "final_dist_std": float(np.std(final_dists)) if final_dists else float("nan"),
        "csv_path": os.path.abspath(out_csv),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a CONTINUOUS-action policy in the AUV environment")
    parser.add_argument("--ckpt", type=str, default=None, help="DreamerV3 run directory (contains ckpt/)")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes")
    parser.add_argument("--dt", type=float, default=0.05, help="Environment integration step")
    parser.add_argument("--max_steps", type=int, default=500, help="Maximum steps per episode")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for evaluation")
    parser.add_argument(
        "--success_threshold",
        type=float,
        default=0.1,  # match your env's success_radius default (0.1)
        help="Distance (m) regarded as a successful reach (independent of is_terminal).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="eval_outputs",
        help="Directory to store evaluation CSV and summary",
    )
    parser.add_argument(
        "--summary_json",
        type=str,
        default=None,
        help="Optional path to save the aggregated metrics as JSON",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "trajectories.csv"

    metrics = evaluate_auv(
        args.ckpt,
        episodes=args.episodes,
        dt=args.dt,
        max_steps=args.max_steps,
        success_threshold=args.success_threshold,
        out_csv=csv_path,
        seed=args.seed,
        verbose=True,
    )

    if args.summary_json:
        import json
        summary_path = Path(args.summary_json).expanduser()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    print("\n=== Evaluation Summary ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
