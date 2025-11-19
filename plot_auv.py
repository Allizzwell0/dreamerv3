#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot AUV evaluation results from CSV produced by eval_auv.py

Usage:
  python plot_auv.py --csv eval_outputs/trajectories.csv --episode 0
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ===================== CSV 读取与预处理 =====================

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
    # 按时间步排序
    rows_sorted = sorted(rows, key=lambda r: int(r["t"]))

    def arr(key: str, default=np.nan, dtype=float):
        return np.asarray(
            [_to_float(r.get(key, ""), default=default) for r in rows_sorted],
            dtype=dtype,
        )

    data = {
        "t": np.asarray([int(r["t"]) for r in rows_sorted], dtype=int),
        "reward": arr("reward", default=0.0),
        "dist": arr("dist"),
        "x": arr("x"),
        "y": arr("y"),
        "goal_x": arr("goal_x"),      # <<< 保留整条时间序列
        "goal_y": arr("goal_y"),      # <<<
        "u": arr("u", default=0.0),
        "v": arr("v", default=0.0),
        "r": arr("r", default=0.0),
        "theta": arr("theta", default=0.0),
        "is_terminal": np.asarray(
            [str(r.get("is_terminal", "")).lower() == "true" for r in rows_sorted],
            dtype=bool,
        ),
        "is_last": np.asarray(
            [str(r.get("is_last", "")).lower() == "true" for r in rows_sorted],
            dtype=bool,
        ),
    }

    # 不再把 goal_x/goal_y 覆盖成常数；移动目标需要完整轨迹
    return data


def episode_summary(
    data: Dict[str, np.ndarray],
    success_threshold: float,
) -> Tuple[float, float, float, float, bool]:
    """给单个 episode 生成摘要：return、长度、final_dist、min_dist、success。"""
    rewards = data["reward"]
    dist = data["dist"]

    ep_return = float(np.nansum(rewards)) if rewards.size else 0.0
    ep_len = int(len(rewards))
    final_dist = float(dist[-1]) if dist.size else float("nan")
    min_dist = float(np.nanmin(dist)) if np.isfinite(dist).any() else float("nan")
    success = np.isfinite(min_dist) and (min_dist <= success_threshold)

    return ep_return, ep_len, final_dist, min_dist, success


# ===================== 绘图函数 =====================

def plot_trajectory(
    data: Dict[str, np.ndarray],
    episode: int,
    success: bool,
    success_threshold: float,
):
    xs = data["x"]
    ys = data["y"]
    gxs = data["goal_x"]
    gys = data["goal_y"]

    # 有效点掩码
    auv_mask = ~np.isnan(xs) & ~np.isnan(ys)
    goal_mask = ~np.isnan(gxs) & ~np.isnan(gys)

    if not np.any(auv_mask):
        return None

    fig, ax = plt.subplots(figsize=(6, 6))

    # --- AUV 轨迹 ---
    ax.plot(
        xs[auv_mask],
        ys[auv_mask],
        marker="o",
        markersize=2,
        linewidth=1.0,
        label="AUV trajectory",
    )

    # 起点 / 终点
    ax.scatter(xs[auv_mask][0], ys[auv_mask][0], marker="o", s=50, label="AUV start")
    ax.scatter(xs[auv_mask][-1], ys[auv_mask][-1], marker="x", s=70, label="AUV end")

    # --- 目标轨迹（移动目标） ---
    if np.any(goal_mask):
        ax.plot(
            gxs[goal_mask],
            gys[goal_mask],
            linestyle="--",
            linewidth=1.0,
            label="Goal trajectory",
        )
        # 目标起点 / 终点
        ax.scatter(gxs[goal_mask][0], gys[goal_mask][0], marker="^", s=60, label="Goal start")
        ax.scatter(gxs[goal_mask][-1], gys[goal_mask][-1], marker="*", s=120, label="Goal end")

    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    status = "SUCCESS" if success else "FAIL"
    ax.set_title(f"Episode {episode}: XY trajectories ({status}, thr={success_threshold:.2f} m)")
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

    axes[1].plot(t, speed, label="|velocity|")
    axes[1].set_ylabel("speed (m/s)")
    axes[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    axes[2].plot(t, r, label="yaw rate r")
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
    ax1.set_xlabel("time step")
    ax1.set_ylabel("distance to goal (m)")
    ax1.plot(t, dist, label="distance")
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    ax2 = ax1.twinx()
    ax2.set_ylabel("reward")
    ax2.plot(t, reward, alpha=0.7, label="reward")

    fig.suptitle(f"Episode {episode}: Distance & reward")
    fig.tight_layout()
    return fig, f"episode_{episode:03d}_distance_reward.png"


def plot_success_histories(
    episodes: Dict[int, Dict[str, np.ndarray]],
    success_threshold: float,
):
    summaries = []
    for ep, data in sorted(episodes.items()):
        _, _, final_dist, min_dist, success = episode_summary(data, success_threshold)
        summaries.append((ep, final_dist, min_dist, success))

    if not summaries:
        return None

    eps = np.asarray([s[0] for s in summaries], dtype=int)
    final_dists = np.asarray([s[1] for s in summaries], dtype=float)
    successes = np.asarray([s[3] for s in summaries], dtype=bool)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = np.where(successes, "tab:green", "tab:red")
    ax.bar(eps, final_dists, color=colors)
    ax.axhline(success_threshold, linestyle="--", linewidth=1.0, label="success threshold")
    ax.set_xlabel("episode")
    ax.set_ylabel("final distance (m)")
    ax.set_title(
        f"Episode final distance (green = success, threshold = {success_threshold:.2f} m)"
    )
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, "episode_success_overview.png"


# ===================== 主入口 =====================

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
        default=0.3,  # 记得和 eval_auv.py 里的 success_threshold 对齐
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

    # 打印每个 episode 的简单 summary，方便快速看效果
    print("\n[plot_auv] Episode summaries:")
    for ep, data in sorted(episode_arrays.items()):
        ret, length, final_dist, min_dist, success =episode_summary(data, args.success_threshold)
        status = "SUCCESS" if success else "FAIL"
        print(
            f"  Ep {ep:03d}: return={ret:.2f}, len={length:3d}, "
            f"final_dist={final_dist:.3f}, min_dist={min_dist:.3f}, {status}"
        )

    # 当前目标 episode 的 per-step 图
    ep_data = episode_arrays[target_episode]
    _, _, _, _, ep_success = episode_summary(ep_data, args.success_threshold)

    figures = []
    traj_fig = plot_trajectory(ep_data, target_episode, ep_success, args.success_threshold)
    if traj_fig is not None:
        figures.append(traj_fig)
    speed_fig = plot_speed_profiles(ep_data, target_episode)
    figures.append(speed_fig)
    dist_fig = plot_distance_reward(ep_data, target_episode)
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
