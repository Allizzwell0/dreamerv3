# SAC_base/eval_sac_auv.py
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from auv_gym import AUVGym


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


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


def make_eval_env(dt: float, max_steps: int, seed: int, env_kwargs: Dict[str, Any]):
    def _init():
        return AUVGym(dt=dt, max_steps=max_steps, seed=seed, **env_kwargs)
    return DummyVecEnv([_init])


def evaluate_sac(
    model_path: Path,
    out_dir: Path,
    *,
    episodes: int,
    dt: float,
    max_steps: int,
    seed: int,
    env_kwargs: Dict[str, Any],
    vecnorm_path: Optional[Path] = None,
    success_threshold: float = 1.0,
    track_success_ratio: float = 0.8,
    overshoot_eps: float = 0.2,
    overshoot_min_steps: int = 20,
    overshoot_require_distdot: bool = True,
) -> Dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "trajectories.csv"
    overshoot_json = out_dir / "overshoots.json"

    # env
    venv = make_eval_env(dt, max_steps, seed, env_kwargs)

    vecnorm: Optional[VecNormalize] = None
    # VecNormalize：评估时必须 load + 设为 eval 模式（不更新统计量）
    if vecnorm_path is not None and vecnorm_path.exists():
        venv = VecNormalize.load(str(vecnorm_path), venv)
        assert isinstance(venv, VecNormalize)
        vecnorm = venv
        vecnorm.training = False
        vecnorm.norm_reward = False

    model = SAC.load(str(model_path), env=venv, device="cuda")

    header = [
        "episode", "t",
        "reward",
        "x", "y", "theta",
        "u", "v", "r",
        "goal_x", "goal_y",
        "xb", "yb",
        "dist",
        "dist_td", "dist_dot_td",
        "min_dist_td_sofar",
        "is_overshoot_step",
        "err_x", "err_y", "err_heading",
        "phase_cos", "phase_sin", "t_norm",
        "action_0", "action_1",
        "done",
    ]

    rng = np.random.default_rng(seed)

    ep_returns: List[float] = []
    ep_lengths: List[int] = []
    final_dists: List[float] = []
    mean_dists: List[float] = []
    max_dists: List[float] = []
    track_ratios: List[float] = []
    successes = 0

    rmse_x_list: List[float] = []
    rmse_y_list: List[float] = []
    rmse_heading_list: List[float] = []

    overshoot_records: List[Dict[str, Any]] = []

    def _raw_obs(obs_any):
        if vecnorm is None:
            return obs_any
        # VecNormalize stores the unnormalized observation from the latest reset/step
        return vecnorm.get_original_obs()

    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for ep in range(episodes):
            # reset
            ep_seed = int(rng.integers(0, 2**31 - 1))
            obs = venv.reset(seed=ep_seed)
            obs_raw = _raw_obs(obs)
            obs_vec = np.asarray(obs_raw[0], dtype=float)  # (15,)

            # trackers
            ep_return = 0.0
            ep_dists_step: List[float] = []

            overshoot_found = False
            overshoot_payload: Optional[Dict[str, Any]] = None
            min_dist_td = float("inf")
            min_step = 0
            prev_dist_td: Optional[float] = None

            sum_ex2 = sum_ey2 = sum_hd2 = 0.0
            count_err = 0

            last_act = (float("nan"), float("nan"))

            # --- log t=0 (after reset) ---
            xb, yb, dist_td_vec = map(float, obs_vec[:3])
            theta = float(math.atan2(obs_vec[4], obs_vec[3]))
            u, v, r_val = map(float, obs_vec[5:8])
            x, y = float(obs_vec[8]), float(obs_vec[9])
            goal_x, goal_y = float(obs_vec[10]), float(obs_vec[11])
            phase_cos, phase_sin, t_norm = float(obs_vec[12]), float(obs_vec[13]), float(obs_vec[14])

            dist = float(math.sqrt(xb * xb + yb * yb))
            dist_td = float(dist_td_vec)
            dist_dot_td = 0.0

            # init min
            min_dist_td = dist_td
            min_step = 0
            prev_dist_td = dist_td

            # errors
            err_x = goal_x - x
            err_y = goal_y - y
            desired_heading = math.atan2(goal_y - y, goal_x - x)
            err_heading = _wrap_pi(desired_heading - theta)

            sum_ex2 += err_x * err_x
            sum_ey2 += err_y * err_y
            sum_hd2 += err_heading * err_heading
            count_err += 1

            ep_dists_step.append(dist)

            writer.writerow(
                [
                    ep, 0,
                    0.0,
                    x, y, theta,
                    u, v, r_val,
                    goal_x, goal_y,
                    xb, yb,
                    dist,
                    dist_td, dist_dot_td,
                    float(min_dist_td),
                    0,
                    err_x, err_y, err_heading,
                    phase_cos, phase_sin, t_norm,
                    last_act[0], last_act[1],
                    0,
                ]
            )

            done = False
            step_idx = 0

            # --- rollout ---
            while (not done) and (step_idx < max_steps):
                # choose action from *normalized* obs (what the policy expects)
                action, _ = model.predict(obs, deterministic=True)
                act = np.asarray(action[0], dtype=float).reshape(-1)
                act = np.clip(act[:2], -1.0, 1.0)
                last_act = (float(act[0]), float(act[1]))

                obs, rew, dones, infos = venv.step(action)
                obs_raw = _raw_obs(obs)
                obs_vec = np.asarray(obs_raw[0], dtype=float)

                reward = float(rew[0])
                info = infos[0]
                done = bool(dones[0])

                ep_return += reward
                step_idx += 1

                xb, yb, dist_td_vec = map(float, obs_vec[:3])
                theta = float(math.atan2(obs_vec[4], obs_vec[3]))
                u, v, r_val = map(float, obs_vec[5:8])
                x, y = float(obs_vec[8]), float(obs_vec[9])
                goal_x, goal_y = float(obs_vec[10]), float(obs_vec[11])
                phase_cos, phase_sin, t_norm = float(obs_vec[12]), float(obs_vec[13]), float(obs_vec[14])

                # dist (raw) and dist_td (filtered)
                dist = float(info.get("log/dist", math.sqrt(xb * xb + yb * yb)))
                dist_td = float(info.get("log/dist_td", dist_td_vec))

                # dist_dot
                if "log/dist_dot_td" in info:
                    dist_dot_td = float(info.get("log/dist_dot_td", 0.0))
                else:
                    dist_dot_td = float((dist_td - (prev_dist_td if prev_dist_td is not None else dist_td)) / max(dt, 1e-8))
                prev_dist_td = dist_td

                # update min
                if dist_td < min_dist_td:
                    min_dist_td = dist_td
                    min_step = step_idx

                # errors
                err_x = goal_x - x
                err_y = goal_y - y
                desired_heading = math.atan2(goal_y - y, goal_x - x)
                err_heading = _wrap_pi(desired_heading - theta)

                sum_ex2 += err_x * err_x
                sum_ey2 += err_y * err_y
                sum_hd2 += err_heading * err_heading
                count_err += 1

                ep_dists_step.append(dist)

                # overshoot detection (first time only)
                is_overshoot_step = detect_first_overshoot(
                    t=step_idx,
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
                    overshoot_payload = dict(
                        episode=ep,
                        overshoot_step=step_idx,
                        min_step=int(min_step),
                        min_dist_td=float(min_dist_td),
                        dist_td=float(dist_td),
                        dist_dot_td=float(dist_dot_td),
                        x=float(x), y=float(y), theta=float(theta),
                        goal_x=float(goal_x), goal_y=float(goal_y),
                        u=float(u), v=float(v), r=float(r_val),
                        xb=float(xb), yb=float(yb), dist=float(dist),
                    )

                writer.writerow(
                    [
                        ep, step_idx,
                        reward,
                        x, y, theta,
                        u, v, r_val,
                        goal_x, goal_y,
                        xb, yb,
                        dist,
                        dist_td, dist_dot_td,
                        float(min_dist_td),
                        1 if is_overshoot_step else 0,
                        err_x, err_y, err_heading,
                        phase_cos, phase_sin, t_norm,
                        last_act[0], last_act[1],
                        1 if done else 0,
                    ]
                )

            # episode summary
            ep_returns.append(ep_return)
            ep_lengths.append(step_idx)
            final_dists.append(float(ep_dists_step[-1]) if ep_dists_step else float("nan"))

            ep_dists_arr = np.array(ep_dists_step, dtype=float) if ep_dists_step else np.array([np.nan])
            mean_dist = float(np.nanmean(ep_dists_arr))
            max_dist = float(np.nanmax(ep_dists_arr))
            track_ratio = float(np.nanmean(ep_dists_arr <= success_threshold))

            mean_dists.append(mean_dist)
            max_dists.append(max_dist)
            track_ratios.append(track_ratio)

            rmse_x = float(math.sqrt(sum_ex2 / max(1, count_err)))
            rmse_y = float(math.sqrt(sum_ey2 / max(1, count_err)))
            rmse_heading = float(math.sqrt(sum_hd2 / max(1, count_err)))
            rmse_x_list.append(rmse_x)
            rmse_y_list.append(rmse_y)
            rmse_heading_list.append(rmse_heading)

            success_flag = track_ratio >= track_success_ratio
            if success_flag:
                successes += 1

            if overshoot_payload is None:
                overshoot_payload = dict(
                    episode=ep,
                    overshoot_step=None,
                    min_step=int(min_step),
                    min_dist_td=float(min_dist_td) if min_dist_td != float("inf") else None,
                    note="no overshoot detected with current thresholds",
                )
            overshoot_records.append(overshoot_payload)

    overshoot_json.write_text(json.dumps(overshoot_records, indent=2, ensure_ascii=False), encoding="utf-8")

    metrics = {
        "episodes": float(episodes),
        "success_rate": successes / episodes if episodes else 0.0,
        "success_count": float(successes),

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
    }
    return metrics



def parse_kv_list(kvs: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in kvs:
        k, v = item.split("=", 1)
        v = v.strip()
        if v.lower() in ("true", "false"):
            out[k] = (v.lower() == "true")
        else:
            try:
                out[k] = float(v) if ("." in v or "e" in v.lower()) else int(v)
            except Exception:
                out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="path to sac_final.zip or ckpt model")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--vecnorm", type=str, default=None, help="vecnormalize.pkl if used")

    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--max_steps", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--success_threshold", type=float, default=1.0)
    ap.add_argument("--track_success_ratio", type=float, default=0.8)

    ap.add_argument("--overshoot_eps", type=float, default=0.2)
    ap.add_argument("--overshoot_min_steps", type=int, default=20)
    ap.add_argument("--overshoot_require_distdot", action="store_true")
    ap.add_argument("--no_overshoot_require_distdot", action="store_false", dest="overshoot_require_distdot")
    ap.set_defaults(overshoot_require_distdot=True)

    ap.add_argument("--env", action="append", default=[], help="AUVEnv kw override: key=value (repeatable)")

    args = ap.parse_args()

    env_kwargs = parse_kv_list(args.env)

    metrics = evaluate_sac(
        Path(args.model),
        Path(args.out_dir),
        episodes=args.episodes,
        dt=args.dt,
        max_steps=args.max_steps,
        seed=args.seed,
        env_kwargs=env_kwargs,
        vecnorm_path=Path(args.vecnorm) if args.vecnorm else None,
        success_threshold=args.success_threshold,
        track_success_ratio=args.track_success_ratio,
        overshoot_eps=args.overshoot_eps,
        overshoot_min_steps=args.overshoot_min_steps,
        overshoot_require_distdot=args.overshoot_require_distdot,
    )
    (Path(args.out_dir) / "fitness.json").write_text(
        json.dumps({"metrics": metrics}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
