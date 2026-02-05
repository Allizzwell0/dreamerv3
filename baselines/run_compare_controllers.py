#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare controllers on AUVEnv with the same LOS reference.

Controllers:
- mpc : LOS + QP-MPC (existing baseline)
- pid : LOS + PID
- smc : LOS + SMC

Outputs:
  out_dir/
    mpc/trajectories.csv, mpc/metrics.json
    pid/trajectories.csv, pid/metrics.json
    smc/trajectories.csv, smc/metrics.json
    summary.json

This script tries to reuse the exact logging format from run_los_mpc.py so you can
diff trajectories and metrics easily.

Usage example:
  python run_compare_controllers.py --out_dir runs/compare --episodes 20 \
    --env moving_goal=true --auto_rudder_sign
"""

import sys
from pathlib import Path

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
sys.path.insert(0, str(ROOT))

import argparse
import csv
import json
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---- project imports (keep consistent with run_los_mpc.py) ----
from los_guidance import ConstrainedLOS, LOSConfig
from mpc_model import AUVModelConfig, SimpleAUVErrorModel
from mpc_controller import MPCConfig, MPCController
from pid_controller import PIDConfig, PIDController
from smc_controller import SMCConfig, SMCController


def _import_env():
    # Try the user's original import path first, then fall back to local file.
    try:
        from embodied.envs.AUV_Env import AUVEnv  # type: ignore
        return AUVEnv
    except Exception:
        from AUV_Env import AUVEnv  # type: ignore
        return AUVEnv


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


def detect_rudder_sign(env, dt: float, probe_steps: int = 30) -> float:
    """
    Autodetect mapping from action[1] to yaw direction.
    Returns:
      +1.0 if positive rudder_norm makes theta increase (left turn),
      -1.0 otherwise.

    Note: we run a fresh reset and a short rollout (doesn't affect your eval env if you instantiate separately).
    """
    # warm start to get positive surge
    a = np.array([0.6, 0.0], dtype=np.float32)
    traj = env.step({"reset": True, "action": np.zeros((2,), dtype=np.float32)})

    # run a few steps to build forward speed
    for _ in range(10):
        traj = env.step({"reset": False, "action": a})

    vec = np.asarray(traj["vector"], dtype=float).reshape(-1)
    theta0 = float(math.atan2(vec[4], vec[3]))

    # apply positive rudder
    a = np.array([0.6, 0.3], dtype=np.float32)
    for _ in range(probe_steps):
        traj = env.step({"reset": False, "action": a})

    vec = np.asarray(traj["vector"], dtype=float).reshape(-1)
    theta1 = float(math.atan2(vec[4], vec[3]))

    dtheta = wrap_pi(theta1 - theta0)
    return 1.0 if dtheta > 0.0 else -1.0


def run_one_controller(
    *,
    name: str,
    out_dir: Path,
    args: argparse.Namespace,
    env_kwargs: Dict[str, Any],
    rudder_sign_to_env: float,
) -> Dict[str, Any]:
    """
    Runs episodes for one controller and writes trajectories + metrics.
    Returns metrics dict.
    """
    AUVEnv = _import_env()
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

    # controller init
    mpc: Optional[MPCController] = None
    pid: Optional[PIDController] = None
    smc: Optional[SMCController] = None

    if name == "mpc":
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
            dthrust_limit=args.dthrust_limit,
            ddelta_limit=args.ddelta_limit,
        ))
    elif name == "pid":
        pid = PIDController(PIDConfig(
            kp_u=args.pid_kp_u,
            ki_u=args.pid_ki_u,
            kd_u=args.pid_kd_u,
            kp_psi=args.pid_kp_psi,
            ki_psi=args.pid_ki_psi,
            kd_psi=args.pid_kd_psi,
            k_yb=args.pid_k_yb,
            i_u_limit=args.pid_i_u_limit,
            i_psi_limit=args.pid_i_psi_limit,
        ))
    elif name == "smc":
        smc = SMCController(SMCConfig(
            kp_u=args.smc_kp_u,
            ki_u=args.smc_ki_u,
            i_u_limit=args.smc_i_u_limit,
            c1=args.smc_c1,
            k_eq=args.smc_k_eq,
            k_sw=args.smc_k_sw,
            phi=args.smc_phi,
            k_r_damp=args.smc_k_r_damp,
        ))
    else:
        raise ValueError(f"Unknown controller: {name}")

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
    mean_abs_yb: List[float] = []
    mean_abs_epsi: List[float] = []
    mean_act_energy: List[float] = []
    successes = 0

    rng = np.random.default_rng(args.seed)

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for ep in range(args.episodes):
            # reseed env randomness per episode (if supported)
            env_seed = int(rng.integers(0, 2**31 - 1))
            if hasattr(env, "np_random"):
                try:
                    env.np_random = np.random.RandomState(env_seed)
                except Exception:
                    pass

            los.reset()
            if pid is not None:
                pid.reset()
            if smc is not None:
                smc.reset()

            prev_action = np.zeros((2,), dtype=np.float32)
            traj = env.step({"reset": True, "action": prev_action})

            ep_ret = 0.0
            ep_dists: List[float] = []
            ep_abs_yb: List[float] = []
            ep_abs_epsi: List[float] = []
            ep_act_energy: List[float] = []
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

                # --- control ---
                status = ""
                if mpc is not None:
                    x0 = np.array([xb, yb, epsi, u, r], dtype=float)
                    act, info = mpc.act(x0=x0, u_ref=u_ref, u_prev=prev_action)
                    status = str(info.get("status", ""))
                elif pid is not None:
                    act, info = pid.act(xb=xb, yb=yb, theta=theta, u=u, r=r, psi_ref=psi_ref, u_ref=u_ref, dt=args.dt)
                    status = "pid"
                elif smc is not None:
                    act, info = smc.act(theta=theta, u=u, r=r, psi_ref=psi_ref, u_ref=u_ref, dt=args.dt)
                    status = "smc"
                else:
                    raise RuntimeError("No controller active")

                # rate limiting (normalized actions)
                if args.dthrust_limit > 0:
                    act[0] = float(np.clip(act[0], prev_action[0] - args.dthrust_limit, prev_action[0] + args.dthrust_limit))
                if args.ddelta_limit > 0:
                    act[1] = float(np.clip(act[1], prev_action[1] - args.ddelta_limit, prev_action[1] + args.ddelta_limit))

                # map controller rudder sign to environment
                act_env = act.astype(np.float32).copy()
                act_env[1] = float(act_env[1] * rudder_sign_to_env)

                traj2 = env.step({"reset": False, "action": act_env})

                reward = float(traj.get("reward", 0.0))
                discount = float(traj.get("discount", 1.0))
                is_last = bool(traj.get("is_last", False))
                is_terminal = bool(traj.get("is_terminal", False))

                ep_ret += reward
                ep_dists.append(dist)
                ep_abs_yb.append(abs(yb))
                ep_abs_epsi.append(abs(epsi))
                ep_act_energy.append(float(act_env[0] ** 2 + act_env[1] ** 2))

                writer.writerow([
                    ep, t,
                    reward, discount,
                    x, y, theta,
                    u, v, r,
                    gx, gy,
                    xb, yb, dist,
                    dist_td, dist_dot_td,
                    psi_ref, u_ref, epsi,
                    float(act_env[0]), float(act_env[1]),
                    status,
                    int(is_terminal), int(is_last),
                ])

                prev_action = act_env
                traj = traj2
                if is_last:
                    break

            ep_returns.append(ep_ret)
            ep_lengths.append(t)

            d = np.asarray(ep_dists, dtype=float) if ep_dists else np.asarray([np.nan])
            mean_dist = float(np.nanmean(d))
            max_dist = float(np.nanmax(d))
            track_ratio = float(np.nanmean(d <= args.success_threshold))

            mean_dists.append(mean_dist)
            max_dists.append(max_dist)
            track_ratios.append(track_ratio)
            mean_abs_yb.append(float(np.mean(ep_abs_yb)) if ep_abs_yb else float("nan"))
            mean_abs_epsi.append(float(np.mean(ep_abs_epsi)) if ep_abs_epsi else float("nan"))
            mean_act_energy.append(float(np.mean(ep_act_energy)) if ep_act_energy else float("nan"))

            success_flag = track_ratio >= args.track_success_ratio
            successes += int(success_flag)

            print(
                f"[{name.upper()}] ep={ep:03d} return={ep_ret:.2f} steps={t} "
                f"mean_dist={mean_dist:.3f} max_dist={max_dist:.3f} "
                f"track_ratio={track_ratio:.2f} status={'SUCCESS' if success_flag else 'FAIL'}"
            )

    metrics = {
        "controller": name,
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
        "abs_yb_mean": float(np.mean(mean_abs_yb)) if mean_abs_yb else float("nan"),
        "abs_epsi_mean": float(np.mean(mean_abs_epsi)) if mean_abs_epsi else float("nan"),
        "act_energy_mean": float(np.mean(mean_act_energy)) if mean_act_energy else float("nan"),
        "rudder_sign_to_env": float(rudder_sign_to_env),
        "csv_path": str(csv_path),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--controllers", type=str, nargs="+", default=["mpc", "pid", "smc"])
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--max_steps", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)

    # metrics thresholds
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

    # action rate limits (shared)
    p.add_argument("--dthrust_limit", type=float, default=0.3)
    p.add_argument("--ddelta_limit", type=float, default=0.3)

    # rudder sign mapping to env
    p.add_argument("--rudder_sign_to_env", type=float, default=1.0, help="Multiply rudder by this before env.step (use -1 if sign is flipped).")
    p.add_argument("--auto_rudder_sign", action="store_true", help="Probe env to auto-detect rudder sign mapping.")

    # MPC params (baseline)
    p.add_argument("--mpc_horizon", type=int, default=20)
    p.add_argument("--mpc_u_limit", type=float, default=5.0)
    p.add_argument("--mpc_r_limit", type=float, default=3.0)

    # rough model params
    p.add_argument("--tau_u", type=float, default=0.8)
    p.add_argument("--k_u", type=float, default=3.0)
    p.add_argument("--tau_r", type=float, default=0.4)
    p.add_argument("--k_r", type=float, default=6.0)

    # PID params
    p.add_argument("--pid_kp_u", type=float, default=0.55)
    p.add_argument("--pid_ki_u", type=float, default=0.08)
    p.add_argument("--pid_kd_u", type=float, default=0.00)
    p.add_argument("--pid_kp_psi", type=float, default=1.60)
    p.add_argument("--pid_ki_psi", type=float, default=0.15)
    p.add_argument("--pid_kd_psi", type=float, default=0.35)
    p.add_argument("--pid_k_yb", type=float, default=0.00)
    p.add_argument("--pid_i_u_limit", type=float, default=3.0)
    p.add_argument("--pid_i_psi_limit", type=float, default=2.0)

    # SMC params
    p.add_argument("--smc_kp_u", type=float, default=0.55)
    p.add_argument("--smc_ki_u", type=float, default=0.08)
    p.add_argument("--smc_i_u_limit", type=float, default=3.0)
    p.add_argument("--smc_c1", type=float, default=2.0)
    p.add_argument("--smc_k_eq", type=float, default=0.35)
    p.add_argument("--smc_k_sw", type=float, default=0.75)
    p.add_argument("--smc_phi", type=float, default=0.25)
    p.add_argument("--smc_k_r_damp", type=float, default=0.00)

    args = p.parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    env_kwargs = parse_kv_list(args.env)
    env_kwargs.setdefault("dt", args.dt)
    env_kwargs.setdefault("max_steps", args.max_steps)
    env_kwargs.setdefault("seed", args.seed)

    # rudder sign detection
    rudder_sign_to_env = float(args.rudder_sign_to_env)
    if args.auto_rudder_sign:
        AUVEnv = _import_env()
        env_probe = AUVEnv(**env_kwargs)
        rudder_sign_to_env = detect_rudder_sign(env_probe, dt=args.dt, probe_steps=30)
        print(f"[AutoSign] rudder_sign_to_env = {rudder_sign_to_env:+.0f} ( +1 means +rudder => left turn )")

    summary: Dict[str, Any] = {
        "args": vars(args),
        "controllers": {},
    }

    for name in args.controllers:
        cdir = out_dir / name
        cdir.mkdir(parents=True, exist_ok=True)
        metrics = run_one_controller(
            name=name,
            out_dir=cdir,
            args=args,
            env_kwargs=env_kwargs,
            rudder_sign_to_env=rudder_sign_to_env,
        )
        summary["controllers"][name] = metrics

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Summary (key metrics) ===")
    for name in args.controllers:
        m = summary["controllers"][name]
        print(
            f"{name:>4s}  success_rate={m['success_rate']:.2f}  "
            f"mean_dist={m['mean_dist_mean']:.3f}  track_ratio={m['track_ratio_mean']:.2f}  "
            f"abs_yb={m['abs_yb_mean']:.3f}  abs_epsi={m['abs_epsi_mean']:.3f}  act_energy={m['act_energy_mean']:.3f}"
        )


if __name__ == "__main__":
    main()
