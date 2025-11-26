#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot 3D AUV evaluation results from CSV produced by eval_auv.py

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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


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
        "t":       np.asarray([int(r["t"]) for r in rows_sorted], dtype=int),
        "reward":  arr("reward", default=0.0),
        "dist":    arr("dist"),
        # 位姿
        "x":       arr("x"),
        "y":       arr("y"),
        "z":       arr("z"),
        "psi":     arr("psi"),
        "theta":   arr("theta"),
        # 线速度 & 角速度
        "u":       arr("u", default=0.0),
        "v":       arr("v", default=0.0),
        "w":       arr("w", default=0.0),
        "p":       arr("p", default=0.0),
        "q":       arr("q", default=0.0),
        "r":       arr("r", default=0.0),
        # 目标位置
        "goal_x":  arr("goal_x"),
        "goal_y":  arr("goal_y"),
        "goal_z":  arr("goal_z"),
        # 船体坐标系下误差
        "xb":      arr("xb"),
        "yb":      arr("yb"),
        "zb":      arr("zb"),
        # 终止标记
        "is_terminal": np.asarray(
            [str(r.get("is_terminal", "")).lower() == "true" for r in rows_sorted],
            dtype=bool,
        ),
        "is_last": np.asarray(
            [str(r.get("is_last", "")).lower() == "true" for r in rows_sorted],
            dtype=bool,
        ),
    }

    return data


def episode_summary(
    data: Dict[str, np.ndarray],
    success_threshold: float,
    track_success_ratio: float,
) -> Tuple[float, int, float, float, float, bool]:
    """
    单个 episode 的摘要：
      ep_return      : 回报和
      ep_len         : 步数
      final_dist     : 最后一步距离
      mean_dist      : 平均距离
      track_ratio    : dist <= success_threshold 的时间比例
      success        : track_ratio >= track_success_ratio
    """
    rewards = data["reward"]
    dist = data["dist"]

    ep_return = float(np.nansum(rewards)) if rewards.size else 0.0
    ep_len = int(len(rewards))

    if dist.size:
        final_dist = float(dist[-1])
        mean_dist = float(np.nanmean(dist))
        track_ratio = float(np.mean(dist <= success_threshold))
    else:
        final_dist = float("nan")
        mean_dist = float("nan")
        track_ratio = 0.0

    success = bool(track_ratio >= track_success_ratio)

    return ep_return, ep_len, final_dist, mean_dist, track_ratio, success


# ===================== 绘图函数 =====================

def plot_trajectory_3d(
    data: Dict[str, np.ndarray],
    episode: int,
    success: bool,
    success_threshold: float,
):
    xs = data["x"]
    ys = data["y"]
    zs = data["z"]
    gxs = data["goal_x"]
    gys = data["goal_y"]
    gzs = data["goal_z"]

    auv_mask = ~np.isnan(xs) & ~np.isnan(ys) & ~np.isnan(zs)
    goal_mask = ~np.isnan(gxs) & ~np.isnan(gys) & ~np.isnan(gzs)

    if not np.any(auv_mask):
        return None

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    # --- AUV 轨迹 ---
    ax.plot(xs[auv_mask], ys[auv_mask], zs[auv_mask],
            marker="o", markersize=2, linewidth=1.0, label="AUV trajectory")

    # 起点 / 终点
    ax.scatter(xs[auv_mask][0], ys[auv_mask][0], zs[auv_mask][0],
               marker="o", s=50, label="AUV start")
    ax.scatter(xs[auv_mask][-1], ys[auv_mask][-1], zs[auv_mask][-1],
               marker="x", s=70, label="AUV end")

    # --- 目标轨迹 ---
    if np.any(goal_mask):
        ax.plot(gxs[goal_mask], gys[goal_mask], gzs[goal_mask],
                linestyle="--", linewidth=1.0, label="Goal trajectory")
        ax.scatter(gxs[goal_mask][0], gys[goal_mask][0], gzs[goal_mask][0],
                   marker="^", s=60, label="Goal start")
        ax.scatter(gxs[goal_mask][-1], gys[goal_mask][-1], gzs[goal_mask][-1],
                   marker="*", s=120, label="Goal end")

    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_zlabel("z / m")
    status = "SUCCESS" if success else "FAIL"
    ax.set_title(f"Episode {episode}: 3D trajectories ({status}, thr={success_threshold:.2f} m)")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, f"episode_{episode:03d}_trajectory3d.png"


def plot_projection_xy(data: Dict[str, np.ndarray], episode: int, success: bool, success_threshold: float):
    """俯视图 XY 投影。"""
    xs = data["x"]
    ys = data["y"]
    gxs = data["goal_x"]
    gys = data["goal_y"]

    auv_mask = ~np.isnan(xs) & ~np.isnan(ys)
    goal_mask = ~np.isnan(gxs) & ~np.isnan(gys)

    if not np.any(auv_mask):
        return None

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(xs[auv_mask], ys[auv_mask],
            marker="o", markersize=2, linewidth=1.0, label="AUV trajectory")
    ax.scatter(xs[auv_mask][0], ys[auv_mask][0], marker="o", s=50, label="AUV start")
    ax.scatter(xs[auv_mask][-1], ys[auv_mask][-1], marker="x", s=70, label="AUV end")

    if np.any(goal_mask):
        ax.plot(gxs[goal_mask], gys[goal_mask],
                linestyle="--", linewidth=1.0, label="Goal trajectory")
        ax.scatter(gxs[goal_mask][0], gys[goal_mask][0], marker="^", s=60, label="Goal start")
        # 这里修正：原来是 (gys, gys)，现在是 (gxs, gys)
        ax.scatter(gxs[goal_mask][-1], gys[goal_mask][-1], marker="*", s=120, label="Goal end")

    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    status = "SUCCESS" if success else "FAIL"
    ax.set_title(f"Episode {episode}: XY projection ({status}, thr={success_threshold:.2f} m)")
    ax.set_aspect("equal", "box")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, f"episode_{episode:03d}_trajectory_xy.png"


def plot_speed_profiles(data: Dict[str, np.ndarray], episode: int):
    t = data["t"]
    u = data["u"]
    v = data["v"]
    w = data["w"]
    p = data["p"]
    q = data["q"]
    r = data["r"]
    z = data["z"]

    lin_speed = np.sqrt(u**2 + v**2 + w**2)
    ang_speed = np.sqrt(p**2 + q**2 + r**2)

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    # 线速度
    axes[0].plot(t, u, label="u (surge)")
    axes[0].plot(t, v, label="v (sway)")
    axes[0].plot(t, w, label="w (heave)")
    axes[0].plot(t, lin_speed, label="|v|")
    axes[0].set_ylabel("velocity (m/s)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    # 角速度
    axes[1].plot(t, p, label="p")
    axes[1].plot(t, q, label="q")
    axes[1].plot(t, r, label="r")
    axes[1].plot(t, ang_speed, label="|ω|")
    axes[1].set_ylabel("angular rate (rad/s)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    # 深度
    axes[2].plot(t, z, label="depth z")
    axes[2].set_ylabel("z / m")
    axes[2].set_xlabel("time step")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    fig.suptitle(f"Episode {episode}: Velocity & depth profiles", y=0.96)
    fig.tight_layout()
    return fig, f"episode_{episode:03d}_velocity_depth.png"


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
    track_success_ratio: float,
):
    """
    画两个 overview：
      1) track_ratio vs episode（绿色=成功）
      2) mean_dist vs episode（配合 success_threshold）
    返回 [(fig, name), ...]
    """
    summaries = []
    for ep, data in sorted(episodes.items()):
        ep_return, ep_len, final_dist, mean_dist, track_ratio, success = \
            episode_summary(data, success_threshold, track_success_ratio)
        summaries.append((ep, final_dist, mean_dist, track_ratio, success))

    if not summaries:
        return []

    eps = np.asarray([s[0] for s in summaries], dtype=int)
    mean_dists = np.asarray([s[2] for s in summaries], dtype=float)
    track_ratios = np.asarray([s[3] for s in summaries], dtype=float)
    successes = np.asarray([s[4] for s in summaries], dtype=bool)

    figs: List[Tuple[plt.Figure, str]] = []

    # --- 图 1：track_ratio 概览 ---
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    colors = np.where(successes, "tab:green", "tab:red")
    ax1.bar(eps, track_ratios, color=colors)
    ax1.axhline(track_success_ratio, linestyle="--", linewidth=1.0, label="track_success_ratio")
    ax1.set_xlabel("episode")
    ax1.set_ylabel("track ratio")
    ax1.set_ylim(0.0, 1.0)
    ax1.set_title(
        f"Episode track ratio (green = success, ratio >= {track_success_ratio:.2f})"
    )
    ax1.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax1.legend(loc="best")
    fig1.tight_layout()
    figs.append((fig1, "episode_track_ratio_overview.png"))

    # --- 图 2：mean_dist 概览 ---
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.bar(eps, mean_dists, color="tab:blue")
    ax2.axhline(success_threshold, linestyle="--", linewidth=1.0, label="success_threshold")
    ax2.set_xlabel("episode")
    ax2.set_ylabel("mean distance (m)")
    ax2.set_title(
        f"Episode mean distance to goal (threshold = {success_threshold:.2f} m)"
    )
    ax2.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax2.legend(loc="best")
    fig2.tight_layout()
    figs.append((fig2, "episode_mean_dist_overview.png"))

    return figs


# ===================== 主入口 =====================

def main():
    parser = argparse.ArgumentParser(description="Plot 3D AUV evaluation trajectories")
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
        default=1.0,  # 要和 eval_auv.py 里的 success_threshold 对齐
        help="Distance threshold (m) used to judge 'good tracking' when computing track_ratio",
    )
    parser.add_argument(
        "--track_success_ratio",
        type=float,
        default=0.3,  # 要和 eval_auv.py 里的 track_success_ratio 对齐
        help="Episode is considered SUCCESS if track_ratio >= this value.",
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

    # 打印每个 episode 的 summary
    print("\n[plot_auv] Episode summaries:")
    for ep, data in sorted(episode_arrays.items()):
        ret, length, final_dist, mean_dist, track_ratio, success = episode_summary(
            data, args.success_threshold, args.track_success_ratio
        )
        status = "SUCCESS" if success else "FAIL"
        print(
            f"  Ep {ep:03d}: return={ret:.2f}, len={length:3d}, "
            f"final_dist={final_dist:.3f}, mean_dist={mean_dist:.3f}, "
            f"track_ratio={track_ratio:.2f}, {status}"
        )

    # 当前目标 episode 的 per-step 图
    ep_data = episode_arrays[target_episode]
    _, _, _, _, _, ep_success = episode_summary(
        ep_data, args.success_threshold, args.track_success_ratio
    )

    figures: List[Tuple[plt.Figure, str]] = []
    traj3d_fig = plot_trajectory_3d(ep_data, target_episode, ep_success, args.success_threshold)
    if traj3d_fig is not None:
        figures.append(traj3d_fig)

    proj_fig = plot_projection_xy(ep_data, target_episode, ep_success, args.success_threshold)
    if proj_fig is not None:
        figures.append(proj_fig)

    speed_fig = plot_speed_profiles(ep_data, target_episode)
    figures.append(speed_fig)

    dist_fig = plot_distance_reward(ep_data, target_episode)
    figures.append(dist_fig)

    overview_figs = plot_success_histories(
        episode_arrays, args.success_threshold, args.track_success_ratio
    )
    if overview_figs:
        figures.extend(overview_figs)

    for fig, name in figures:
        fig.savefig(out_dir / name, dpi=args.dpi)
        plt.close(fig)
        print(f"[plot_auv] Saved {name}")


if __name__ == "__main__":
    main()
