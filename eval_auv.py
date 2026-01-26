#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a trained DreamerV3 policy on the (CONTINUOUS-ACTION) AUVEnv and export trajectories to CSV,
along with summary metrics + overshoot detection.

Usage:
  python eval_auv.py --ckpt ~/logdir/auv/20251106T161609 \
    --episodes 200 --out_dir ~/logdir/auv/20251106T161609/eval_output

New:
- If checkpoint is provided, optionally build env from the training config.yaml (recommended).
- Export TD distance diagnostics if env provides: log/dist_td, log/dist_dot_td
- Detect first overshoot point per episode and save overshoots.json

Overshoot definition (default):
- Maintain min_dist_td so far; if after at least overshoot_min_steps since min,
  dist_td rises above min_dist_td + overshoot_eps AND (optionally) dist_dot_td > 0,
  then the first such step is marked as overshoot point.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple

import numpy as np

# ----------------- path bootstrap (make `dreamerv3` importable when running as script) -----------------
THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent if (THIS.parent.name == "dreamerv3") else THIS.parent
# Typical layout:
#   ROOT/
#     dreamerv3/
#       main.py
#       eval_auv.py  (this file)
#     embodied/
sys.path.insert(0, str(ROOT))

# ===== MODIFY THIS IMPORT TO MATCH YOUR ENV PATH (fallback mode) =====
try:
    from embodied.envs.AUV_Env import AUVEnv
except Exception as e:
    AUVEnv = None
    _AUV_IMPORT_ERR = e


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


# ------------ continuous-action fallback policy ------------

class RandomContinuousPolicy:
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


# ----------------- checkpoint helpers -----------------

def _looks_like_step_dir(p: Path) -> bool:
    return (p / "manifest").exists() or (p / "checkpoint").exists()


def resolve_ckpt_step_dir(ckpt_dir: str) -> Optional[Path]:
    run_path = Path(ckpt_dir).expanduser().resolve()
    if not run_path.exists():
        return None

    if run_path.is_dir() and _looks_like_step_dir(run_path):
        return run_path

    ckpt_root = run_path / "ckpt"
    if not ckpt_root.exists() and run_path.name == "ckpt":
        ckpt_root = run_path

    if not ckpt_root.exists():
        return None

    subdirs = [d for d in ckpt_root.iterdir() if d.is_dir()]
    if not subdirs:
        return None
    subdirs = sorted(subdirs)
    return subdirs[-1]


def find_config_yaml_near(step_dir: Path, max_up: int = 6) -> Optional[Path]:
    probe = step_dir
    for _ in range(max_up):
        cand = probe / "config.yaml"
        if cand.exists():
            return cand
        probe = probe.parent
    return None


def load_elements_config(config_path: Path):
    import elements
    import ruamel.yaml as yaml
    raw_cfg = yaml.YAML(typ="safe").load(config_path.read_text(encoding="utf-8"))
    cfg = elements.Config(raw_cfg)
    # Use run root as logdir
    cfg = cfg.update(logdir=str(config_path.parent))
    return cfg


# ----------------- build env (recommended) -----------------

def build_env(
    ckpt_dir: Optional[str],
    *,
    dt: float,
    max_steps: int,
    seed: int,
    use_env_from_ckpt: bool,
):
    """
    Recommended: if ckpt_dir provided and use_env_from_ckpt=True:
      - read config.yaml
      - use dreamerv3.main.make_env(config, index=0) to match training exactly
    Fallback:
      - construct AUVEnv directly
    """
    rng = np.random.default_rng(seed)
    env_seed = int(rng.integers(0, 2**31 - 1))

    if ckpt_dir and use_env_from_ckpt:
        try:
            step_dir = resolve_ckpt_step_dir(ckpt_dir)
            if step_dir is None:
                raise RuntimeError("Could not resolve checkpoint step dir.")
            cfg_path = find_config_yaml_near(step_dir)
            if cfg_path is None:
                raise RuntimeError("config.yaml not found near checkpoint.")
            cfg = load_elements_config(cfg_path)

            from dreamerv3 import main as dv3_main
            # Ensure eval uses same task/env config; override dt/max_steps if you want:
            # NOTE: your AUVEnv __init__ must accept dt/max_steps as kwargs for this to work.
            # If not, delete the overrides.
            cfg = cfg.update(env=cfg.env)  # keep structure
            # Make env with wrappers exactly like training
            env = dv3_main.make_env(cfg, index=0, dt=dt, max_steps=max_steps)
            # Seed: training env seeding uses use_seed; here we seed the underlying env if possible
            if hasattr(env, "np_random"):
                env.np_random.seed(env_seed)
            np.random.seed(env_seed)
            return env
        except Exception as e:
            print(f"[eval_auv] build_env_from_ckpt failed: {e}")
            print("[eval_auv] Falling back to direct AUVEnv construction.")

    # Fallback: direct env
    if AUVEnv is None:
        raise ImportError(
            "AUVEnv import failed and env-from-ckpt also failed.\n"
            "Please fix AUVEnv import path.\n"
            f"Original import error: {_AUV_IMPORT_ERR}"
        )

    env = AUVEnv(dt=dt, max_steps=max_steps, moving_goal=True)
    if hasattr(env, "np_random"):
        env.np_random.seed(env_seed)
    np.random.seed(env_seed)
    return env


# ----------------- load trained policy -----------------

def load_trained_policy(
    checkpoint_dir: Optional[str],
    act_shape,
    act_low,
    act_high,
    seed: int = 0,
):
    if checkpoint_dir is None:
        print("[eval_auv] No checkpoint provided. Using RandomContinuousPolicy.")
        return RandomContinuousPolicy(act_low, act_high, seed)

    step_dir = resolve_ckpt_step_dir(checkpoint_dir)
    if step_dir is None:
        print(f"[eval_auv] Could not resolve checkpoint from '{checkpoint_dir}'. Using RandomContinuousPolicy.")
        return RandomContinuousPolicy(act_low, act_high, seed)

    cfg_path = find_config_yaml_near(step_dir)
    if cfg_path is None:
        print(f"[eval_auv] config.yaml not found near '{step_dir}'. Using RandomContinuousPolicy.")
        return RandomContinuousPolicy(act_low, act_high, seed)

    print(f"[eval_auv] Using checkpoint step dir: {step_dir}")
    print(f"[eval_auv] Using config: {cfg_path}")

    try:
        import elements
        from dreamerv3 import main as dv3_main

        config = load_elements_config(cfg_path)

        # Build agent same as training
        agent = dv3_main.make_agent(config)

        # Load weights
        cp = elements.Checkpoint()
        cp.agent = agent
        cp.load(str(step_dir), keys=["agent"])
        print(f"[eval_auv] Loaded DreamerV3 agent weights from {step_dir}")

        act_shape = tuple(act_shape)

        class ContinuousPolicyWrapper:
            def __init__(self, agent_, act_shape_, seed_):
                self.agent = agent_
                self.act_shape = tuple(act_shape_)
                self.carry = None
                self.rng = np.random.default_rng(seed_)

            def reset(self) -> None:
                self.carry = self.agent.init_policy(batch_size=1)

            def __call__(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
                if self.carry is None:
                    self.reset()

                # IMPORTANT: filter obs keys to what agent expects
                obs_filtered = {k: obs[k] for k in self.agent.obs_space if k in obs}
                obs_batched = {k: np.asarray(v)[None] for k, v in obs_filtered.items()}

                self.carry, acts, outs = self.agent.policy(self.carry, obs_batched, mode="eval")

                if "action" not in acts:
                    raise RuntimeError(f"'action' not found in policy acts keys: {list(acts.keys())}")

                act_arr = np.asarray(acts["action"], dtype=np.float32)
                act_vec = act_arr[0] if act_arr.ndim >= 2 else act_arr
                act_vec = act_vec.reshape(self.act_shape)
                act_vec = np.clip(act_vec, -1.0, 1.0)
                return {"reset": False, "action": act_vec}

        return ContinuousPolicyWrapper(agent, act_shape, seed)

    except Exception as e:
        print(f"[eval_auv] Could not load DreamerV3 policy: {e}")
        print("[eval_auv] Falling back to RandomContinuousPolicy.")
        return RandomContinuousPolicy(act_low, act_high, seed)


# ----------------- overshoot detection -----------------

def detect_first_overshoot(
    *,
    t: int,
    dist_td: float,
    dist_dot_td: float,
    min_dist_td: float,
    min_step: int,
    overshoot_found: bool,
    overshoot_eps: float,
    overshoot_min_steps: int,
    overshoot_require_distdot: bool,
) -> bool:
    if overshoot_found:
        return False
    if t - min_step < overshoot_min_steps:
        return False
    if dist_td <= min_dist_td + overshoot_eps:
        return False
    if overshoot_require_distdot and not (dist_dot_td > 0.0):
        return False
    return True


# ----------------- main evaluate -----------------

def evaluate_auv(
    ckpt_dir: Optional[str],
    *,
    episodes: int = 20,
    dt: float = 0.05,
    max_steps: int = 800,
    success_threshold: float = 1.0,
    track_success_ratio: float = 0.8,
    out_dir: Path,
    seed: int = 0,
    verbose: bool = True,
    use_env_from_ckpt: bool = True,
    # overshoot params
    overshoot_eps: float = 0.2,
    overshoot_min_steps: int = 20,
    overshoot_require_distdot: bool = True,
) -> Dict[str, float]:

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "trajectories.csv"
    overshoot_json = out_dir / "overshoots.json"

    rng = np.random.default_rng(seed)

    # Build env
    env = build_env(
        ckpt_dir,
        dt=dt,
        max_steps=max_steps,
        seed=seed,
        use_env_from_ckpt=use_env_from_ckpt,
    )

    # Read action space from env (works with wrappers)
    act_space = env.act_space["action"]
    act_shape = act_space.shape
    act_low = getattr(act_space, "low", -1.0)
    act_high = getattr(act_space, "high", 1.0)

    policy = load_trained_policy(ckpt_dir, act_shape, act_low, act_high, seed)

    header = [
        "episode", "t",
        "reward", "discount",
        "x", "y", "theta",
        "u", "v", "r",
        "goal_x", "goal_y",
        "xe", "ye",
        "dist",
        "dist_td", "dist_dot_td",
        "min_dist_td_sofar",
        "is_overshoot_step",
        "err_x", "err_y", "err_heading",
        "phase_cos", "phase_sin", "t_norm",
        "action_0", "action_1",
        "is_terminal", "is_last",
    ]

    ep_returns: List[float] = []
    ep_lengths: List[int] = []
    final_dists: List[float] = []
    mean_dists: List[float] = []
    max_dists: List[float] = []
    track_ratios: List[float] = []
    rmse_x_list: List[float] = []
    rmse_y_list: List[float] = []
    rmse_heading_list: List[float] = []
    successes = 0

    overshoot_records: List[Dict[str, Any]] = []

    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for ep in range(episodes):
            policy.reset()

            # resample env randomness per episode
            env_seed = int(rng.integers(0, 2**31 - 1))
            if hasattr(env, "np_random"):
                env.np_random.seed(env_seed)
            np.random.seed(env_seed)

            zero_act = np.zeros(act_shape, dtype=np.float32)
            traj = env.step({"reset": True, "action": zero_act})  


            ep_return = 0.0
            steps = 0

            # stats buffers
            ep_dists_step: List[float] = []
            sum_ex2 = sum_ey2 = sum_hd2 = 0.0
            count_err = 0

            # overshoot trackers
            overshoot_found = False
            overshoot_step = None
            overshoot_payload = None

            min_dist_td = float("inf")
            min_step = 0
            prev_dist_td = None

            for t in range(0, max_steps + 1):
                steps = t

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

                # TD diagnostics (prefer env log keys)
                dist_td = float(traj.get("log/dist_td", dist))
                if "log/dist_dot_td" in traj:
                    dist_dot_td = float(traj["log/dist_dot_td"])
                else:
                    if prev_dist_td is None:
                        dist_dot_td = 0.0
                    else:
                        dist_dot_td = (dist_td - prev_dist_td) / max(dt, 1e-8)

                prev_dist_td = dist_td

                # update min
                if dist_td < min_dist_td:
                    min_dist_td = dist_td
                    min_step = t

                # errors (world frame)
                if not (math.isnan(x) or math.isnan(y) or math.isnan(goal_x) or math.isnan(goal_y)):
                    err_x = goal_x - x
                    err_y = goal_y - y
                    desired_heading = math.atan2(goal_y - y, goal_x - x)
                    err_heading = _wrap_pi(desired_heading - theta)
                    sum_ex2 += err_x * err_x
                    sum_ey2 += err_y * err_y
                    sum_hd2 += err_heading * err_heading
                    count_err += 1
                else:
                    err_x = err_y = err_heading = float("nan")

                reward = float(traj.get("reward", 0.0))
                discount = float(traj.get("discount", 1.0))
                is_last = bool(traj.get("is_last", False))
                is_terminal = bool(traj.get("is_terminal", False))

                # record distance stats (use raw dist for your original metrics)
                ep_dists_step.append(dist)

                # overshoot detection
                is_overshoot_step = detect_first_overshoot(
                    t=t,
                    dist_td=dist_td,
                    dist_dot_td=dist_dot_td,
                    min_dist_td=min_dist_td,
                    min_step=min_step,
                    overshoot_found=overshoot_found,
                    overshoot_eps=overshoot_eps,
                    overshoot_min_steps=overshoot_min_steps,
                    overshoot_require_distdot=overshoot_require_distdot,
                )
                if is_overshoot_step and not overshoot_found:
                    overshoot_found = True
                    overshoot_step = t
                    overshoot_payload = dict(
                        episode=ep,
                        overshoot_step=t,
                        min_step=min_step,
                        min_dist_td=float(min_dist_td),
                        dist_td=float(dist_td),
                        dist_dot_td=float(dist_dot_td),
                        x=float(x), y=float(y), theta=float(theta),
                        goal_x=float(goal_x), goal_y=float(goal_y),
                        u=float(u), v=float(v), r=float(r_val),
                        xe=float(xe), ye=float(ye), dist=float(dist),
                    )

                # action: at t=0, no action yet (after reset), so write NaN
                if t == 0:
                    a0 = a1 = float("nan")
                else:
                    # action was produced for this transition in previous loop
                    # we store it in `last_raw_act`
                    a0, a1 = last_raw_act  # noqa

                writer.writerow(
                    [
                        ep, t,
                        reward, discount,
                        x, y, theta,
                        u, v, r_val,
                        goal_x, goal_y,
                        xe, ye,
                        dist,
                        dist_td, dist_dot_td,
                        float(min_dist_td),
                        1 if is_overshoot_step else 0,
                        err_x, err_y, err_heading,
                        phase_cos, phase_sin, t_norm,
                        a0, a1,
                        is_terminal, is_last,
                    ]
                )

                # end episode?
                if is_last:
                    ep_return += reward
                    break

                # step env with policy action (except at t=0 we haven't acted yet)
                action = policy(traj)
                raw_act = np.asarray(action.get("action", np.zeros(act_shape)), dtype=float).reshape(-1)
                if raw_act.size == 1:
                    raw_act = np.array([raw_act.item(), 0.0], dtype=float)
                else:
                    raw_act = raw_act[:2]
                last_raw_act = (float(raw_act[0]), float(raw_act[1]))  # for CSV row at next t
                traj = env.step(action)
                ep_return += reward

            # episode summary
            ep_returns.append(ep_return)
            ep_lengths.append(steps)
            final_dists.append(float(ep_dists_step[-1]) if ep_dists_step else float("nan"))

            ep_dists_arr = np.array(ep_dists_step, dtype=float) if ep_dists_step else np.array([np.nan])
            mean_dist = float(np.nanmean(ep_dists_arr))
            max_dist = float(np.nanmax(ep_dists_arr))
            track_ratio = float(np.nanmean(ep_dists_arr <= success_threshold))

            mean_dists.append(mean_dist)
            max_dists.append(max_dist)
            track_ratios.append(track_ratio)

            if count_err > 0:
                rmse_x = float(math.sqrt(sum_ex2 / count_err))
                rmse_y = float(math.sqrt(sum_ey2 / count_err))
                rmse_heading = float(math.sqrt(sum_hd2 / count_err))
            else:
                rmse_x = rmse_y = rmse_heading = float("nan")
            rmse_x_list.append(rmse_x)
            rmse_y_list.append(rmse_y)
            rmse_heading_list.append(rmse_heading)

            success_flag = track_ratio >= track_success_ratio
            if success_flag:
                successes += 1

            # store overshoot record
            if overshoot_payload is None:
                overshoot_payload = dict(
                    episode=ep,
                    overshoot_step=None,
                    min_step=int(min_step),
                    min_dist_td=float(min_dist_td) if min_dist_td != float("inf") else None,
                    note="no overshoot detected with current thresholds",
                )
            overshoot_records.append(overshoot_payload)

            if verbose:
                status = "SUCCESS" if success_flag else "FAIL"
                os_str = f"overshoot_step={overshoot_step}" if overshoot_step is not None else "overshoot_step=None"
                print(
                    f"[Episode {ep:03d}] return={ep_return:.2f} steps={steps} "
                    f"status={status} final_dist={final_dists[-1]:.3f} "
                    f"mean_dist={mean_dist:.3f} track_ratio={track_ratio:.2f} "
                    f"rmse_x={rmse_x:.3f} rmse_y={rmse_y:.3f} rmse_heading={rmse_heading:.3f} "
                    f"{os_str}"
                )

    # save overshoots.json
    overshoot_json.write_text(json.dumps(overshoot_records, indent=2, ensure_ascii=False), encoding="utf-8")

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
        "rmse_x_mean": float(np.mean(rmse_x_list)) if rmse_x_list else float("nan"),
        "rmse_x_std": float(np.std(rmse_x_list)) if rmse_x_list else float("nan"),
        "rmse_y_mean": float(np.mean(rmse_y_list)) if rmse_y_list else float("nan"),
        "rmse_y_std": float(np.std(rmse_y_list)) if rmse_y_list else float("nan"),
        "rmse_heading_mean": float(np.mean(rmse_heading_list)) if rmse_heading_list else float("nan"),
        "rmse_heading_std": float(np.std(rmse_heading_list)) if rmse_heading_list else float("nan"),
        "csv_path": os.path.abspath(out_csv),
        "overshoot_json": os.path.abspath(overshoot_json),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a CONTINUOUS-action policy in the AUV environment")
    parser.add_argument("--ckpt", type=str, default=None, help="DreamerV3 run directory (contains ckpt/) or ckpt step dir")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--max_steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--success_threshold", type=float, default=0.5)
    parser.add_argument("--track_success_ratio", type=float, default=0.8)

    parser.add_argument("--out_dir", type=str, default="eval_outputs")
    parser.add_argument("--summary_json", type=str, default=None)

    # recommended: build env from ckpt config.yaml
    parser.add_argument("--use_env_from_ckpt", action="store_true", help="Build env via dreamerv3.main.make_env(config) (recommended)")
    parser.add_argument("--no_use_env_from_ckpt", action="store_false", dest="use_env_from_ckpt")
    parser.set_defaults(use_env_from_ckpt=True)

    # overshoot params
    parser.add_argument("--overshoot_eps", type=float, default=0.2)
    parser.add_argument("--overshoot_min_steps", type=int, default=20)
    parser.add_argument("--overshoot_require_distdot", action="store_true")
    parser.add_argument("--no_overshoot_require_distdot", action="store_false", dest="overshoot_require_distdot")
    parser.set_defaults(overshoot_require_distdot=True)

    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser()

    metrics = evaluate_auv(
        args.ckpt,
        episodes=args.episodes,
        dt=args.dt,
        max_steps=args.max_steps,
        success_threshold=args.success_threshold,
        track_success_ratio=args.track_success_ratio,
        out_dir=out_dir,
        seed=args.seed,
        verbose=True,
        use_env_from_ckpt=args.use_env_from_ckpt,
        overshoot_eps=args.overshoot_eps,
        overshoot_min_steps=args.overshoot_min_steps,
        overshoot_require_distdot=args.overshoot_require_distdot,
    )

    if args.summary_json:
        summary_path = Path(args.summary_json).expanduser()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Evaluation Summary ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
