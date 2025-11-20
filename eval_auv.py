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
    - 用 elements.Checkpoint() 加载 agent 权重（完全仿照 embodied/run/eval_only.py）
    - 输出连续动作向量（直接传给 AUVEnv）

    支持三种传参形式：
      1) --ckpt 指向一个具体 step 目录（含 manifest/checkpoint 等）
      2) --ckpt 指向 run_dir（里面有 ckpt/ 子目录）
      3) --ckpt 指向 ckpt 目录本身
    """
    if checkpoint_dir is None:
        print("[eval_auv] No checkpoint provided. Using RandomContinuousPolicy.")
        return RandomContinuousPolicy(act_low, act_high, seed)

    run_path = Path(checkpoint_dir).expanduser().resolve()
    if not run_path.exists():
        print(f"[eval_auv] Checkpoint dir '{run_path}' does not exist. Using RandomContinuousPolicy.")
        return RandomContinuousPolicy(act_low, act_high, seed)

    # -------- 0) 判断本身是否就是一个 step 目录（含 manifest/checkpoint） --------
    def _looks_like_step_dir(p: Path) -> bool:
        return (p / "manifest").exists() or (p / "checkpoint").exists()

    load_path: Optional[Path] = None

    if run_path.is_dir() and _looks_like_step_dir(run_path):
        # 直接是某个 step 目录
        load_path = run_path
    else:
        # -------- 1) 如果传的是 run_dir，优先找 run_dir/ckpt 下的子目录 --------
        ckpt_root = run_path / "ckpt"
        if not ckpt_root.exists():
            # 也可能直接传的是 ckpt 目录本身
            if run_path.name == "ckpt":
                ckpt_root = run_path

        if ckpt_root.exists():
            subdirs = [d for d in ckpt_root.iterdir() if d.is_dir()]
            if not subdirs:
                print(f"[eval_auv] No checkpoint subdirectories found in '{ckpt_root}'. Using RandomContinuousPolicy.")
                return RandomContinuousPolicy(act_low, act_high, seed)
            subdirs = sorted(subdirs)
            load_path = subdirs[-1]   # 按名字排序，最后一个一般是最新的 step

    if load_path is None:
        print(f"[eval_auv] Could not resolve a checkpoint step dir from '{run_path}'. Using RandomContinuousPolicy.")
        return RandomContinuousPolicy(act_low, act_high, seed)

    print(f"[eval_auv] Using checkpoint step dir: {load_path}")

    # -------- 找 config.yaml：从 load_path 往上爬，仿照 main.py 的结构 --------
    config_path = None
    probe = load_path
    for _ in range(4):  # step -> ckpt -> run_dir -> 上一层
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

        # -------- 1) 读取 config.yaml -> elements.Config（与 main.py 一致）--------
        cfg_text = config_path.read_text(encoding="utf-8")
        raw_cfg = yaml.YAML(typ="safe").load(cfg_text)
        config = elements.Config(raw_cfg)

        # logdir 用 config.yaml 所在目录（一般就是 run_dir）
        config_root = config_path.parent
        config = config.update(logdir=str(config_root))

        # -------- 2) 用 main 里的 make_agent 构造 Agent --------
        agent = dv3_main.make_agent(config)

        # -------- 3) 用 elements.Checkpoint 加载 agent 权重（仿照 eval_only）--------
        cp = elements.Checkpoint()
        cp.agent = agent
        print(f"Loading checkpoint: {load_path}")
        # 这里的 load_path 就相当于 eval_only 里的 args.from_checkpoint
        cp.load(str(load_path), keys=['agent'])
        print(f"[eval_auv] Loaded DreamerV3 agent weights from {load_path}")

        # -------- 4) 连续动作封装：模仿 Driver 的调用方式 --------
        act_shape = tuple(act_shape)

        class ContinuousPolicyWrapper:
            """
            完全模仿 embodied.Driver 的用法：
            carry, acts, outs = agent.policy(carry, obs, mode='eval')
            只不过我们在这里自己维护 carry，而不是用 Driver。
            """
            def __init__(self, agent_, act_shape_, seed_):
                self.agent = agent_
                self.act_shape = tuple(act_shape_)
                self.carry = None
                self.rng = np.random.default_rng(seed_)

            def reset(self) -> None:
                """
                等价于 Driver.reset(init_policy)，只不过我们自己调一遍。
                Driver 里是：
                self.carry = init_policy and init_policy(self.length)
                这里 length=1（单环境），所以 batch_size=1。
                """
                try:
                    self.carry = self.agent.init_policy(batch_size=1)
                except TypeError:
                    # 如果这个 Agent.init_policy 不要参数，就退化成无参
                    self.carry = self.agent.init_policy()

            def __call__(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
                if self.carry is None:
                    self.reset()

                # Driver._step 里会把每个 env 的 obs 堆成 batch：
                #   obs = {k: np.stack([x[k] for x in obs_list])}
                # 我们只有1个 env，所以手动加 batch 维：(1, ...)
                obs_batched = {k: np.asarray(v)[None] for k, v in obs.items()}

                # 对齐 Driver 的调用方式：
                #   self.carry, acts, outs = policy(self.carry, obs, ...)
                self.carry, acts, outs = self.agent.policy(
                    self.carry, obs_batched, mode="eval"
                )

                # acts 是一个 dict，key 和 env.act_space 对齐：
                #  比如 {'action': (B, act_dim)}，这里 B=1
                if not isinstance(acts, dict):
                    raise RuntimeError(f"agent.policy() returned non-dict acts: {type(acts)}")

                if "action" not in acts:
                    # 如果你的 action 名字不是 'action'，这里可以改成别的 key
                    raise RuntimeError(f"'action' not found in policy acts keys: {list(acts.keys())}")

                act_arr = np.asarray(acts["action"], dtype=np.float32)

                # 期望形状是 (1, act_dim) 或 (act_dim,)；统一取第 0 个 env
                if act_arr.ndim == 1:
                    # 形如 (act_dim,) 的情况，直接用
                    act_vec = act_arr
                elif act_arr.ndim >= 2:
                    # 形如 (B, act_dim, ...) 的情况，取第一个 env
                    act_vec = act_arr[0]
                else:
                    raise RuntimeError(f"Unexpected action shape from policy: {act_arr.shape}")

                # reshape 成期望形状，比如 (2,)
                act_vec = act_vec.reshape(self.act_shape)
                # 通常 Dreamer 的动作已经在 [-1, 1]，这里再裁一次保险
                act_vec = np.clip(act_vec, -1.0, 1.0)

                return {"reset": False, "action": act_vec}



        return ContinuousPolicyWrapper(agent, act_shape, seed)

    except Exception as e:
        print(f"[eval_auv] Could not load DreamerV3 policy from '{run_path}': {e}")
        print("[eval_auv] Falling back to RandomContinuousPolicy.")
        return RandomContinuousPolicy(act_low, act_high, seed)



# ------------ 评估循环：使用连续动作 policy 在 AUVEnv 上 rollout ------------

def evaluate_auv(
    ckpt_dir: Optional[str],
    *,
    episodes: int = 20,
    dt: float = 0.05,
    max_steps: int = 500,
    success_threshold: float = 1.0,
    out_csv: Path,
    seed: int = 0,
    verbose: bool = True,
) -> Dict[str, float]:
    """Roll out multiple episodes, compute success rate, and write trajectories."""

    rng = np.random.default_rng(seed)
    env = AUVEnv(
    dt=dt,
    max_steps=max_steps,
    moving_goal=True,             # ⭐ 开启移动目标
    # goal_trajectory_type="circle", # 'circle' / 'line' / 'lemniscate'
    # goal_center=(10.0, 10.0),
    # goal_radius=3.0,
    # goal_speed=0.3,
)


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
    "phase_cos",      # ⭐ 新增
    "phase_sin",      # ⭐ 新增
    "t_norm",         # ⭐ 新增
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
            phase_cos = phase_sin = t_norm = float("nan")

            if len(vec) >= 12:
                x, y = float(vec[8]), float(vec[9])
                goal_x, goal_y = float(vec[10]), float(vec[11])
            if len(vec) >= 15:
                phase_cos = float(vec[12])
                phase_sin = float(vec[13])
                t_norm = float(vec[14])

            writer.writerow(
                [
                    ep, 0, 0.0, 1.0,
                    x, y, theta, u, v, r_val,
                    goal_x, goal_y, xe, ye, dist,
                    phase_cos, phase_sin, t_norm,
                    False, False,
                ]
            )


            ep_return = 0.0
            final_dist = dist
            success_flag = dist <= success_threshold
            steps = 0

            for t in range(1, max_steps + 1):
                steps = t
                # 连续动作策略：返回 {"reset": False, "action": np.array(shape=act_shape)}
                action = policy(traj)
                # print("Action at step", t, ":", action["action"])
                traj = env.step(action)
                vec = traj["vector"]

                theta = float(math.atan2(vec[4], vec[3]))
                u, v, r_val = map(float, vec[5:8])
                xe, ye, dist = map(float, vec[:3])

                x = y = goal_x = goal_y = float("nan")
                phase_cos = phase_sin = t_norm = float("nan")

                if len(vec) >= 12:
                    x, y = float(vec[8]), float(vec[9])
                    goal_x, goal_y = float(vec[10]), float(vec[11])
                if len(vec) >= 15:
                    phase_cos = float(vec[12])
                    phase_sin = float(vec[13])
                    t_norm = float(vec[14])

                reward = float(traj["reward"])
                discount = float(traj.get("discount", 1.0))
                is_last = bool(traj["is_last"])
                is_terminal = bool(traj["is_terminal"])

                ep_return += reward
                final_dist = dist

                writer.writerow(
                    [
                        ep, t, reward, discount,
                        x, y, theta, u, v, r_val,
                        goal_x, goal_y, xe, ye, dist,
                        phase_cos, phase_sin, t_norm,
                        is_terminal, is_last,
                    ]
                )

                EPS = 1e-6
                if dist <= success_threshold + EPS:
                    success_flag = True
                    break

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
        default=1.0,  # match your env's success_radius default (1.0)
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
