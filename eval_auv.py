#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a trained DreamerV3 policy on the (CONTINUOUS-ACTION) 6-DOF AUVEnv
and export 3D trajectories to CSV, along with summary metrics.

Usage example:
  python eval_auv.py --ckpt ~/logdir/auv/20251106T161609 \
    --episodes 50 --out_dir ~/logdir/auv/20251106T161609/eval_output

Notes:
- This version assumes your AUVEnv uses CONTINUOUS actions (action ∈ [-1,1]^n).
- If DreamerV3 checkpoint loading fails or isn't provided, a RandomContinuousPolicy is used.
- 现在假定环境为 6DOF REMUS 环境，obs['vector'] 为 30 维：
    [0:4]   e_b = [xb, yb, zb, dist]          目标在船体系误差
    [4:10]  姿态编码 [cos φ, sin φ, cos θ, sin θ, cos ψ, sin ψ]
    [10:16] 速度 ν = [u, v, w, p, q, r]
    [16:22] 位置 [x, y, z] + 目标位置 [gx, gy, gz] （世界系）
    [22:25] phase_cos_xyz = [cos φx, cos φy, cos φz]
    [25:28] phase_sin_xyz = [sin φx, sin φy, sin φz]
    [28]    t_norm   （全局归一化时间）
    [29]    seg_phase（当前轨迹段相位，0~1）
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Dict, Optional, Any, List

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


# ------------ 加载 DreamerV3 连续动作策略 ------------

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
    - 用 elements.Checkpoint() 加载 agent 权重（完全仿照 embodied/run/eval_only.py）
    """

    if checkpoint_dir is None:
        print("[eval_auv] No checkpoint provided. Using RandomContinuousPolicy.")
        return RandomContinuousPolicy(act_low, act_high, seed)

    run_path = Path(checkpoint_dir).expanduser().resolve()
    if not run_path.exists():
        print(f"[eval_auv] Checkpoint dir '{run_path}' does not exist. Using RandomContinuousPolicy.")
        return RandomContinuousPolicy(act_low, act_high, seed)

    def _looks_like_step_dir(p: Path) -> bool:
        return (p / "manifest").exists() or (p / "checkpoint").exists()

    load_path: Optional[Path] = None

    if run_path.is_dir() and _looks_like_step_dir(run_path):
        load_path = run_path
    else:
        ckpt_root = run_path / "ckpt"
        if not ckpt_root.exists() and run_path.name == "ckpt":
            ckpt_root = run_path

        if ckpt_root.exists():
            subdirs = [d for d in ckpt_root.iterdir() if d.is_dir()]
            if not subdirs:
                print(f"[eval_auv] No checkpoint subdirectories found in '{ckpt_root}'. Using RandomContinuousPolicy.")
                return RandomContinuousPolicy(act_low, act_high, seed)
            subdirs = sorted(subdirs)
            load_path = subdirs[-1]

    if load_path is None:
        print(f"[eval_auv] Could not resolve a checkpoint step dir from '{run_path}'. Using RandomContinuousPolicy.")
        return RandomContinuousPolicy(act_low, act_high, seed)

    print(f"[eval_auv] Using checkpoint step dir: {load_path}")

    # -------- 找 config.yaml --------
    config_path = None
    probe = load_path
    for _ in range(4):
        cand = probe / "config.yaml"
        if cand.exists():
            config_path = cand
            break
        probe = probe.parent

    if config_path is None:
        print(f"[eval_auv] config.yaml not found near '{load_path}'. Using RandomContinuousPolicy.")
        return RandomContinuousPolicy(act_low, act_high, seed)

    try:
        import elements
        import ruamel.yaml as yaml
        from dreamerv3 import main as dv3_main

        cfg_text = config_path.read_text(encoding="utf-8")
        raw_cfg = yaml.YAML(typ="safe").load(cfg_text)
        config = elements.Config(raw_cfg)

        config_root = config_path.parent
        config = config.update(logdir=str(config_root))

        agent = dv3_main.make_agent(config)

        cp = elements.Checkpoint()
        cp.agent = agent
        print(f"Loading checkpoint: {load_path}")
        cp.load(str(load_path), keys=['agent'])
        print(f"[eval_auv] Loaded DreamerV3 agent weights from {load_path}")

        act_shape = tuple(act_shape)

        class ContinuousPolicyWrapper:
            def __init__(self, agent_, act_shape_, seed_):
                self.agent = agent_
                self.act_shape = tuple(act_shape_)
                self.carry = None
                self.rng = np.random.default_rng(seed_)

            def reset(self) -> None:
                try:
                    self.carry = self.agent.init_policy(batch_size=1)
                except TypeError:
                    self.carry = self.agent.init_policy()

            def __call__(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
                if self.carry is None:
                    self.reset()

                obs_batched = {k: np.asarray(v)[None] for k, v in obs.items()}
                self.carry, acts, outs = self.agent.policy(
                    self.carry, obs_batched, mode="eval"
                )

                if not isinstance(acts, dict):
                    raise RuntimeError(f"agent.policy() returned non-dict acts: {type(acts)}")
                if "action" not in acts:
                    raise RuntimeError(f"'action' not found in policy acts keys: {list(acts.keys())}")

                act_arr = np.asarray(acts["action"], dtype=np.float32)

                if act_arr.ndim == 1:
                    act_vec = act_arr
                elif act_arr.ndim >= 2:
                    act_vec = act_arr[0]
                else:
                    raise RuntimeError(f"Unexpected action shape from policy: {act_arr.shape}")

                act_vec = act_vec.reshape(self.act_shape)
                act_vec = np.clip(act_vec, -1.0, 1.0)
                return {"reset": False, "action": act_vec}

        return ContinuousPolicyWrapper(agent, act_shape, seed)

    except Exception as e:
        print(f"[eval_auv] Could not load DreamerV3 policy from '{run_path}': {e}")
        print("[eval_auv] Falling back to RandomContinuousPolicy.")
        return RandomContinuousPolicy(act_low, act_high, seed)


# ------------ 评估循环：3D 轨迹跟踪 ------------

def evaluate_auv(
    ckpt_dir: Optional[str],
    *,
    episodes: int = 20,
    dt: float = 0.05,
    max_steps: int = 500,
    success_threshold: float = 1.0,
    track_success_ratio: float = 0.8,
    out_csv: Path,
    seed: int = 0,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Roll out multiple episodes in 3D tracking mode.

    对于“轨迹跟踪”任务：
      - 不再因为 dist <= success_threshold 提前结束 episode
      - 每个 episode 统计：
          mean_dist: 平均距离
          max_dist:  最大距离
          track_ratio: 有多少比例时间 dist <= success_threshold
      - success_flag: track_ratio >= track_success_ratio 视为“成功 episode”
    """

    rng = np.random.default_rng(seed)
    env = AUVEnv(
        dt=dt,
        max_steps=max_steps,
        moving_goal=True,
    )

    env.np_random.seed(seed)
    np.random.seed(seed)

    act_space = env.act_space["action"]
    act_shape = act_space.shape
    act_low = getattr(act_space, "low", -1.0)
    act_high = getattr(act_space, "high", 1.0)

    policy = load_trained_policy(ckpt_dir, act_shape, act_low, act_high, seed)

    # 这里的 header 保持和之前版本兼容，方便 plot_auv.py 直接用
    header = [
        "episode",
        "t",
        "reward",
        "discount",
        "x",
        "y",
        "z",
        "psi",
        "theta",
        "u",
        "v",
        "w",
        "p",
        "q",
        "r",
        "goal_x",
        "goal_y",
        "goal_z",
        "xb",
        "yb",
        "zb",
        "dist",
        "phase_cos",   # 现在写入的是 phase_x 的 cos
        "phase_sin",   # 对应 phase_x 的 sin
        "t_norm",
        "is_terminal",
        "is_last",
    ]

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    ep_returns: List[float] = []
    ep_lengths: List[int] = []
    final_dists: List[float] = []
    mean_dists: List[float] = []
    max_dists: List[float] = []
    track_ratios: List[float] = []
    successes = 0

    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for ep in range(episodes):
            policy.reset()
            env_seed = int(rng.integers(0, 2**31 - 1))
            env.np_random.seed(env_seed)

            traj = env.step({"reset": True})
            vec = traj["vector"]

            # ------- 按 30 维新结构解析 obs -------
            xb, yb, zb, dist = map(float, vec[0:4])
            cos_phi, sin_phi = float(vec[4]), float(vec[5])
            cos_th, sin_th = float(vec[6]), float(vec[7])
            cos_psi, sin_psi = float(vec[8]), float(vec[9])
            u, v, w, p_ang, q_ang, r_val = map(float, vec[10:16])
            x, y, z = map(float, vec[16:19])
            goal_x, goal_y, goal_z = map(float, vec[19:22])
            phase_cos_xyz = vec[22:25]   # 3 个 phase 的 cos
            phase_sin_xyz = vec[25:28]   # 3 个 phase 的 sin
            t_norm = float(vec[28])
            seg_phase = float(vec[29])   # 目前没写入 CSV，如有需要可以扩展

            # 只写入第一个 phase 分量，保持和旧 header 对齐
            phase_cos = float(phase_cos_xyz[0])
            phase_sin = float(phase_sin_xyz[0])

            # 由 cos/sin 还原姿态角
            psi = math.atan2(sin_psi, cos_psi)
            theta = math.atan2(sin_th, cos_th)
            # phi = math.atan2(sin_phi, cos_phi)  # 如需也存，可以改 CSV header

            writer.writerow(
                [
                    ep, 0, 0.0, 1.0,
                    x, y, z,
                    psi, theta,
                    u, v, w, p_ang, q_ang, r_val,
                    goal_x, goal_y, goal_z,
                    xb, yb, zb, dist,
                    phase_cos, phase_sin, t_norm,
                    False, False,
                ]
            )

            ep_return = 0.0
            final_dist = dist
            steps = 0
            ep_dists_step: List[float] = [dist]

            for t in range(1, max_steps + 1):
                steps = t
                action = policy(traj)
                traj = env.step(action)
                vec = traj["vector"]

                xb, yb, zb, dist = map(float, vec[0:4])
                cos_phi, sin_phi = float(vec[4]), float(vec[5])
                cos_th, sin_th = float(vec[6]), float(vec[7])
                cos_psi, sin_psi = float(vec[8]), float(vec[9])
                u, v, w, p_ang, q_ang, r_val = map(float, vec[10:16])
                x, y, z = map(float, vec[16:19])
                goal_x, goal_y, goal_z = map(float, vec[19:22])
                phase_cos_xyz = vec[22:25]
                phase_sin_xyz = vec[25:28]
                t_norm = float(vec[28])
                seg_phase = float(vec[29])

                phase_cos = float(phase_cos_xyz[0])
                phase_sin = float(phase_sin_xyz[0])

                psi = math.atan2(sin_psi, cos_psi)
                theta = math.atan2(sin_th, cos_th)

                reward = float(traj["reward"])
                discount = float(traj.get("discount", 1.0))
                is_last = bool(traj["is_last"])
                is_terminal = bool(traj["is_terminal"])

                ep_return += reward
                final_dist = dist
                ep_dists_step.append(dist)

                writer.writerow(
                    [
                        ep, t, reward, discount,
                        x, y, z,
                        psi, theta,
                        u, v, w, p_ang, q_ang, r_val,
                        goal_x, goal_y, goal_z,
                        xb, yb, zb, dist,
                        phase_cos, phase_sin, t_norm,
                        is_terminal, is_last,
                    ]
                )

                if is_last:
                    break

            ep_returns.append(ep_return)
            ep_lengths.append(steps)
            final_dists.append(final_dist)

            ep_dists_arr = np.array(ep_dists_step, dtype=float)
            mean_dist = float(np.mean(ep_dists_arr))
            max_dist = float(np.max(ep_dists_arr))
            track_ratio = float(np.mean(ep_dists_arr <= success_threshold))

            mean_dists.append(mean_dist)
            max_dists.append(max_dist)
            track_ratios.append(track_ratio)

            success_flag = track_ratio >= track_success_ratio
            if success_flag:
                successes += 1

            if verbose:
                status = "SUCCESS" if success_flag else "FAIL"
                print(
                    f"[Episode {ep:03d}] return={ep_return:.2f} steps={steps} "
                    f"status={status} final_dist={final_dist:.3f} "
                    f"mean_dist={mean_dist:.3f} track_ratio={track_ratio:.2f}"
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
        "mean_dist_mean": float(np.mean(mean_dists)) if mean_dists else float("nan"),
        "mean_dist_std": float(np.std(mean_dists)) if mean_dists else float("nan"),
        "max_dist_mean": float(np.mean(max_dists)) if max_dists else float("nan"),
        "track_ratio_mean": float(np.mean(track_ratios)) if track_ratios else float("nan"),
        "track_ratio_std": float(np.std(track_ratios)) if track_ratios else float("nan"),
        "csv_path": os.path.abspath(out_csv),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a CONTINUOUS-action policy in 6-DOF AUV environment")
    parser.add_argument("--ckpt", type=str, default=None, help="DreamerV3 run directory (contains ckpt/)")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes")
    parser.add_argument("--dt", type=float, default=0.05, help="Environment integration step")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum steps per episode")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for evaluation")
    parser.add_argument(
        "--success_threshold",
        type=float,
        default=1.0,
        help="Distance (m) regarded as 'good tracking' for ratio/statistics.",
    )
    parser.add_argument(
        "--track_success_ratio",
        type=float,
        default=0.3,
        help="Episode is counted as SUCCESS if fraction of steps with dist <= success_threshold "
             "is at least this value (e.g. 0.8).",
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
        track_success_ratio=args.track_success_ratio,
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
