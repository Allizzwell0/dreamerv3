#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot AUV evaluation results from CSV produced by eval_auv.py

Usage:
  python plot_auv.py --csv eval_trajectories.csv --ep 0
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _to_float(value: str, default: float = np.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


def group_rows_by_episode(rows: Iterable[Dict[str, str]]) -> Dict[int, List[Dict[str, str]]]:
    episodes: Dict[int, List[Dict[str, str]]] = {}
    for row in rows:
        try:
            ep = int(row["episode"])
        except Exception:
            continue
        episodes.setdefault(ep, []).append(row)
    return episodes


def prepare_episode_arrays(rows: List[Dict[str, str]]) -> Dict[str, np.ndarray]:
    rows_sorted = sorted(rows, key=lambda r: int(r["t"]))

    def arr(key: str, default=np.nan, dtype=float):
        return np.asarray([_to_float(r.get(key, ""), default=default) for r in rows_sorted], dtype=dtype)

    data = {
        "t": np.asarray([int(r["t"]) for r in rows_sorted], dtype=int),
        "reward": arr("reward", default=0.0),
        "dist": arr("dist"),
        "x": arr("x"),
        "y": arr("y"),
        "goal_x": arr("goal_x"),
        "goal_y": arr("goal_y"),
        "u": arr("u", default=0.0),
        "v": arr("v", default=0.0),
        "r": arr("r", default=0.0),
        "theta": arr("theta", default=0.0),
        "is_terminal": np.asarray(
            [str(r.get("is_terminal", "")).lower() == "true" for r in rows_sorted], dtype=bool
        ),
        "is_last": np.asarray(
            [str(r.get("is_last", "")).lower() == "true" for r in rows_sorted], dtype=bool
        ),
    }

    # 目标位置在整个 episode 内保持不变，只需最后一个非 NaN
    for key in ("goal_x", "goal_y"):
        valid = data[key][~np.isnan(data[key])]
        if valid.size:
            data[key] = np.full_like(data[key], valid[-1])

    return data


def plot_trajectory(data: Dict[str, np.ndarray], episode: int):
    xs = data["x"]
    ys = data["y"]
    mask = ~np.isnan(xs) & ~np.isnan(ys)
    if not np.any(mask):
        return None

    goal = None
    gx = data["goal_x"][mask]
    gy = data["goal_y"][mask]
    if gx.size and gy.size:
        goal = (gx[-1], gy[-1])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(xs[mask], ys[mask], marker="o", markersize=2, linewidth=1.0, label="trajectory")
    if goal is not None:
        ax.scatter(goal[0], goal[1], marker="*", s=120, c="tab:red", label="goal")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title(f"Episode {episode}: XY trajectory")
    ax.set_aspect("equal", "box")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, f"episode_{episode:03d}_trajectory.png"


def plot_speed_profiles(data: Dict[str, np.ndarray], episode: int):
    t = data["t"]
    u = data["u"]
    v = data["v"]
    r = data["r"]
    speed = np.hypot(u, v)

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(t, u, label="surge u")
    axes[0].plot(t, v, label="sway v")
    axes[0].set_ylabel("velocity (m/s)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    axes[1].plot(t, speed, color="tab:green", label="|velocity|")
    axes[1].set_ylabel("speed (m/s)")
    axes[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    axes[2].plot(t, r, color="tab:orange", label="yaw rate r")
    axes[2].set_ylabel("yaw rate (rad/s)")
    axes[2].set_xlabel("time step")
    axes[2].grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    fig.suptitle(f"Episode {episode}: Velocity profiles", y=0.95)
    fig.tight_layout()
    return fig, f"episode_{episode:03d}_velocity.png"


def plot_distance_reward(data: Dict[str, np.ndarray], episode: int):
    t = data["t"]
    dist = data["dist"]
    reward = data["reward"]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    color1 = "tab:blue"
    ax1.set_xlabel("time step")
    ax1.set_ylabel("distance to goal (m)", color=color1)
    ax1.plot(t, dist, color=color1, label="distance")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("reward", color=color2)
    ax2.plot(t, reward, color=color2, alpha=0.7, label="reward")
    ax2.tick_params(axis="y", labelcolor=color2)

    fig.suptitle(f"Episode {episode}: Distance & reward")
    fig.tight_layout()
    return fig, f"episode_{episode:03d}_distance_reward.png"


def plot_success_histories(
    episodes: Dict[int, Dict[str, np.ndarray]], success_threshold: float
):
    summaries = []
    for ep, data in sorted(episodes.items()):
        if data["dist"].size:
            final_dist = float(data["dist"][-1])
            min_dist = float(np.nanmin(data["dist"]))
        else:
            final_dist = float("nan")
            min_dist = float("nan")
        success = bool(min_dist <= success_threshold)
        summaries.append((ep, final_dist, success))

    if not summaries:
        return None

    eps, dists, successes = zip(*summaries)
    eps = np.asarray(eps)
    dists = np.asarray(dists)
    successes = np.asarray(successes)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = np.where(successes, "tab:green", "tab:red")
    ax.bar(eps, dists, color=colors)
    ax.set_xlabel("episode")
    ax.set_ylabel("final distance (m)")
    ax.set_title(
        f"Episode final distance (green = success, threshold = {success_threshold:.2f} m)"
    )
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    fig.tight_layout()
    return fig, "episode_success_overview.png"


def main():
    parser = argparse.ArgumentParser(description="Plot AUV evaluation trajectories")
    parser.add_argument("--csv", type=str, default="eval_outputs/trajectories.csv")
    parser.add_argument("--episode", type=int, default=0, help="Episode index for per-step plots")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Directory to store generated figures (defaults to <csv_dir>/plots)",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Resolution for saved figures")
    parser.add_argument(
        "--success_threshold",
        type=float,
        default=0.3,
        help="Distance threshold (m) used to judge success in the overview plot",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    rows = load_csv(csv_path)
    grouped = group_rows_by_episode(rows)
    if not grouped:
        raise ValueError("No episode data found in CSV.")

    # 转换为 numpy 数组便于后续计算
    episode_arrays: Dict[int, Dict[str, np.ndarray]] = {
        ep: prepare_episode_arrays(ep_rows) for ep, ep_rows in grouped.items()
    }

    target_episode = args.episode
    if target_episode not in episode_arrays:
        target_episode = min(episode_arrays.keys())
        print(
            f"[plot_auv] Episode {args.episode} not found. "
            f"Fallback to episode {target_episode}."
        )

    out_dir = Path(args.out_dir).expanduser() if args.out_dir else csv_path.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[plot_auv] Saving figures to: {out_dir}")

    figures = []
    traj_fig = plot_trajectory(episode_arrays[target_episode], target_episode)
    if traj_fig is not None:
        figures.append(traj_fig)
    speed_fig = plot_speed_profiles(episode_arrays[target_episode], target_episode)
    figures.append(speed_fig)
    dist_fig = plot_distance_reward(episode_arrays[target_episode], target_episode)
    figures.append(dist_fig)

    overview_fig = plot_success_histories(episode_arrays, args.success_threshold)
    if overview_fig is not None:
        figures.append(overview_fig)

    for fig, name in figures:
        fig.savefig(out_dir / name, dpi=args.dpi)
        plt.close(fig)
        print(f"[plot_auv] Saved {name}")


if __name__ == "__main__":
    main()
