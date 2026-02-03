#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run LOS + MPC baseline on AUVEnv.
"""

# from __future__ import annotations

# ---- path bootstrap MUST be after __future__ ----
import sys
from pathlib import Path

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
sys.path.insert(0, str(ROOT))

# ---- normal imports below ----
import argparse
import csv
import json
import math
from typing import Any, Dict, List

import numpy as np

from embodied.envs.AUV_Env import AUVEnv
from los_guidance import ConstrainedLOS, LOSConfig
from mpc_model import AUVModelConfig, SimpleAUVErrorModel
from mpc_controller import MPCConfig, MPCController



def parse_kv_list(kvs: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in kvs:
        if "=" not in item:
            raise ValueError(f"Bad --env item: {item}, expected key=value")
        k, v = item.split("=", 1)
        v = v.strip()
        if v.lower() in ("true", "false"):
            out[k] = (v.lower() == "true")
        else:
            try:
                if "." in v or "e" in v.lower():
                    out[k] = float(v)
                else:
                    out[k] = int(v)
            except Exception:
                out[k] = v
    return out


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--max_steps", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)

    # metrics
    p.add_argument("--success_threshold", type=float, default=0.5)
    p.add_argument("--track_success_ratio", type=float, default=0.8)

    # env overrides (AUVEnv __init__ kwargs)
    p.add_argument("--env", action="append", default=[], help="AUVEnv kw override: key=value (repeatable)")

    # LOS params
    p.add_argument("--los_lookahead", type=float, default=2.0)
    p.add_argument("--los_yaw_rate_limit", type=float, default=1.2)
    p.add_argument("--los_speed_limit", type=float, default=2.0)
    p.add_argument("--los_speed_min", type=float, default=0.2)
    p.add_argument("--los_k_speed", type=float, default=0.6)
    p.add_argument("--los_slow_heading_gain", type=float, default=0.6)

    # MPC params
    p.add_argument("--mpc_horizon", type=int, default=20)
    p.add_argument("--mpc_u_limit", type=float, default=5.0)
    p.add_argument("--mpc_r_limit", type=float, default=3.0)
    p.add_argument("--mpc_dthrust_limit", type=float, default=0.3)
    p.add_argument("--mpc_ddelta_limit", type=float, default=0.3)

    # rough model params
    p.add_argument("--tau_u", type=float, default=0.8)
    p.add_argument("--k_u", type=float, default=3.0)
    p.add_argument("--tau_r", type=float, default=0.4)
    p.add_argument("--k_r", type=float, default=6.0)

    args = p.parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    env_kwargs = parse_kv_list(args.env)
    env_kwargs.setdefault("dt", args.dt)
    env_kwargs.setdefault("max_steps", args.max_steps)
    env_kwargs.setdefault("seed", args.seed)

    env = AUVEnv(**env_kwargs)

    # LOS
    los = ConstrainedLOS(LOSConfig(
        lookahead=args.los_lookahead,
        yaw_rate_limit=args.los_yaw_rate_limit,
        speed_limit=args.los_speed_limit,
        speed_min=args.los_speed_min,
        k_speed=args.los_k_speed,
        slow_heading_gain=args.los_slow_heading_gain,
    ))

    # MPC
    model = SimpleAUVErrorModel(AUVModelConfig(
        dt=args.dt,
        tau_u=args.tau_u,
        k_u=args.k_u,
        tau_r=args.tau_r,
        k_r=args.k_r,
        rudder_max=float(getattr(env, "rudder_max", 0.6)),
    ))
    mpc = MPCController(model, MPCConfig(
        horizon=args.mpc_horizon,
        u_limit=args.mpc_u_limit,
        r_limit=args.mpc_r_limit,
        dthrust_limit=args.mpc_dthrust_limit,
        ddelta_limit=args.mpc_ddelta_limit,
    ))

    header = [
        "episode", "t",
        "reward", "discount",
        "x", "y", "theta",
        "u", "v", "r",
        "goal_x", "goal_y",
        "xb", "yb", "dist",
        "dist_td", "dist_dot_td",
        "psi_ref", "u_ref", "epsi",
        "action_0", "action_1",
        "mpc_status",
        "is_terminal", "is_last",
    ]
    csv_path = out_dir / "trajectories.csv"

    ep_returns: List[float] = []
    ep_lengths: List[int] = []
    mean_dists: List[float] = []
    max_dists: List[float] = []
    track_ratios: List[float] = []
    successes = 0

    rng = np.random.default_rng(args.seed)

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for ep in range(args.episodes):
            # reseed env randomness per episode
            env_seed = int(rng.integers(0, 2**31 - 1))
            if hasattr(env, "np_random"):
                try:
                    env.np_random = np.random.RandomState(env_seed)
                except Exception:
                    pass

            los.reset()
            prev_action = np.zeros((2,), dtype=np.float32)

            traj = env.step({"reset": True, "action": prev_action})

            ep_ret = 0.0
            ep_dists: List[float] = []
            prev_dist_td = None

            for t in range(args.max_steps + 1):
                vec = np.asarray(traj["vector"], dtype=float).reshape(-1)

                xb, yb = float(vec[0]), float(vec[1])
                dist = float(math.hypot(xb, yb))

                theta = float(math.atan2(vec[4], vec[3]))
                u, v, r = map(float, vec[5:8])

                x = y = gx = gy = float("nan")
                if vec.size >= 12:
                    x, y, gx, gy = map(float, vec[8:12])

                dist_td = float(traj.get("log/dist_td", dist))
                if "log/dist_dot_td" in traj:
                    dist_dot_td = float(traj["log/dist_dot_td"])
                else:
                    if prev_dist_td is None:
                        dist_dot_td = 0.0
                    else:
                        dist_dot_td = (dist_td - prev_dist_td) / max(args.dt, 1e-8)
                prev_dist_td = dist_td

                # LOS reference
                psi_ref, u_ref = los.compute(xb=xb, yb=yb, theta=theta, dt=args.dt, dist=dist_td)
                epsi = wrap_pi(theta - psi_ref)

                # MPC state
                x0 = np.array([xb, yb, epsi, u, r], dtype=float)
                act, info = mpc.act(x0=x0, u_ref=u_ref, u_prev=prev_action)

                # step
                traj2 = env.step({"reset": False, "action": act.astype(np.float32)})

                reward = float(traj.get("reward", 0.0))
                discount = float(traj.get("discount", 1.0))
                is_last = bool(traj.get("is_last", False))
                is_terminal = bool(traj.get("is_terminal", False))

                ep_ret += reward
                ep_dists.append(dist)

                writer.writerow([
                    ep, t,
                    reward, discount,
                    x, y, theta,
                    u, v, r,
                    gx, gy,
                    xb, yb, dist,
                    dist_td, dist_dot_td,
                    psi_ref, u_ref, epsi,
                    float(act[0]), float(act[1]),
                    str(info.get("status", "")),
                    int(is_terminal), int(is_last),
                ])

                prev_action = act.astype(np.float32)
                traj = traj2

                if is_last:
                    break

            ep_returns.append(ep_ret)
            ep_lengths.append(t)

            d = np.asarray(ep_dists, dtype=float) if ep_dists else np.asarray([np.nan])
            mean_dist = float(np.nanmean(d))
            max_dist = float(np.nanmax(d))

            # track_ratio 公式：每步是否在阈值内的均值
            track_ratio = float(np.nanmean(d <= args.success_threshold))

            mean_dists.append(mean_dist)
            max_dists.append(max_dist)
            track_ratios.append(track_ratio)

            success_flag = track_ratio >= args.track_success_ratio
            successes += int(success_flag)

            print(
                f"[LOS+MPC] ep={ep:03d} return={ep_ret:.2f} steps={t} "
                f"mean_dist={mean_dist:.3f} max_dist={max_dist:.3f} "
                f"track_ratio={track_ratio:.2f} status={'SUCCESS' if success_flag else 'FAIL'}"
            )

    metrics = {
        "episodes": int(args.episodes),
        "success_rate": successes / max(1, args.episodes),
        "success_count": int(successes),
        "avg_return": float(np.mean(ep_returns)) if ep_returns else 0.0,
        "std_return": float(np.std(ep_returns)) if ep_returns else 0.0,
        "avg_ep_len": float(np.mean(ep_lengths)) if ep_lengths else 0.0,
        "mean_dist_mean": float(np.mean(mean_dists)) if mean_dists else float("nan"),
        "mean_dist_std": float(np.std(mean_dists)) if mean_dists else float("nan"),
        "max_dist_mean": float(np.mean(max_dists)) if max_dists else float("nan"),
        "track_ratio_mean": float(np.mean(track_ratios)) if track_ratios else float("nan"),
        "track_ratio_std": float(np.std(track_ratios)) if track_ratios else float("nan"),
        "csv_path": str(csv_path),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== LOS+MPC Summary ===")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
