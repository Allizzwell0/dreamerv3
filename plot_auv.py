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
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


# ===================== CSV 读取与预处理 =====================

def load_csv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return rows, fieldnames


def _to_float(value: str, default: float = np.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default

def _parse_xy_list(value: str) -> np.ndarray:
    """Parse JSON list-of-pairs into (H,2) array; return empty (0,2) if missing."""
    if value is None:
        return np.zeros((0, 2), dtype=float)
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return np.zeros((0, 2), dtype=float)
    try:
        obj = json.loads(s)
        arr = np.asarray(obj, dtype=float)
    except Exception:
        return np.zeros((0, 2), dtype=float)
    arr = np.squeeze(arr)
    if arr.ndim == 1 and arr.size == 2:
        arr = arr.reshape(1, 2)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr
    return np.zeros((0, 2), dtype=float)



def group_rows_by_episode(rows: Iterable[Dict[str, str]]) -> Dict[int, List[Dict[str, str]]]:
    episodes: Dict[int, List[Dict[str, str]]] = {}
    for row in rows:
        try:
            ep = int(row["episode"])
        except Exception:
            continue
        episodes.setdefault(ep, []).append(row)
    return episodes


def prepare_episode_arrays(rows: List[Dict[str, str]], fieldnames: List[str]) -> Dict[str, np.ndarray]:
    # 按时间步排序
    rows_sorted = sorted(rows, key=lambda r: int(r["t"]))

    def arr(key: str, default=np.nan, dtype=float):
        return np.asarray(
            [_to_float(r.get(key, ""), default=default) for r in rows_sorted],
            dtype=dtype,
        )

    data: Dict[str, np.ndarray] = {
        "t": np.asarray([int(r["t"]) for r in rows_sorted], dtype=int),
        "reward": arr("reward", default=0.0),
        "dist": arr("dist"),
        # 位姿 / 速度
        "x": arr("x"),
        "y": arr("y"),
        "theta": arr("theta", default=0.0),
        "u": arr("u", default=0.0),
        "v": arr("v", default=0.0),
        "r": arr("r", default=0.0),
        # 目标轨迹
        "goal_x": arr("goal_x"),
        "goal_y": arr("goal_y"),
        "goal_vx": arr("goal_vx", default=np.nan) if "goal_vx" in fieldnames else None,
        "goal_vy": arr("goal_vy", default=np.nan) if "goal_vy" in fieldnames else None,
        # 误差（与 eval_auv.py 对齐）
        "err_x": arr("err_x"),
        "err_y": arr("err_y"),
        "err_heading": arr("err_heading"),  # rad
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

    # 可选：动作列（来自扩展版 eval_auv.py）
    act_cols = [name for name in fieldnames if name.startswith("action_")]
    if act_cols:
        T = len(rows_sorted)
        acts = np.zeros((T, len(act_cols)), dtype=float)
        for i, r in enumerate(rows_sorted):
            for j, col in enumerate(act_cols):
                acts[i, j] = _to_float(r.get(col, "nan"), default=np.nan)
        data["acts"] = acts
        data["act_cols"] = np.array(act_cols, dtype=object)


    # Optional: WM prediction columns (from eval_auv_wm30.py)
    if "wm_conf" in fieldnames:
        data["wm_conf"] = arr("wm_conf")
    if "wm_mse_local" in fieldnames:
        data["wm_mse_local"] = arr("wm_mse_local")
    if "wm_pred_goal_xy" in fieldnames:
        data["wm_pred_goal_xy_list"] = np.array(
            [_parse_xy_list(r.get("wm_pred_goal_xy", "")) for r in rows_sorted],
            dtype=object,
        )
    if "cv_pred_goal_xy" in fieldnames:
        data["cv_pred_goal_xy_list"] = np.array(
            [_parse_xy_list(r.get("cv_pred_goal_xy", "")) for r in rows_sorted],
            dtype=object,
        )
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
        final_dist = float(np.nanmax(dist)) if np.isnan(dist[-1]) else float(dist[-1])
        mean_dist = float(np.nanmean(dist))
        track_ratio = float(np.mean(dist <= success_threshold))
    else:
        final_dist = float("nan")
        mean_dist = float("nan")
        track_ratio = 0.0

    success = bool(track_ratio >= track_success_ratio)

    return ep_return, ep_len, final_dist, mean_dist, track_ratio, success


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


def plot_error_profiles(data: Dict[str, np.ndarray], episode: int):
    """
    画不同自由度误差随时间：
      - err_x(t), err_y(t)（世界坐标系误差）
      - err_heading(t)（rad -> deg）
    """
    t = data["t"]
    err_x = data.get("err_x", np.full_like(t, np.nan, dtype=float))
    err_y = data.get("err_y", np.full_like(t, np.nan, dtype=float))
    err_heading = data.get("err_heading", np.full_like(t, np.nan, dtype=float))

    # 如果全是 NaN 说明 CSV 里还没这些列，直接返回 None
    if np.all(np.isnan(err_x)) and np.all(np.isnan(err_y)) and np.all(np.isnan(err_heading)):
        return None

    heading_deg = np.rad2deg(err_heading)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    # 位置误差
    axes[0].plot(t, err_x, label="err_x = gx - x")
    axes[0].plot(t, err_y, label="err_y = gy - y")
    axes[0].set_ylabel("position error (m)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    # 航向误差
    axes[1].plot(t, heading_deg, label="err_heading (deg)")
    axes[1].set_ylabel("heading error (deg)")
    axes[1].set_xlabel("time step")
    axes[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    axes[1].legend(loc="upper right")

    fig.suptitle(f"Episode {episode}: Tracking errors over time", y=0.95)
    fig.tight_layout()
    return fig, f"episode_{episode:03d}_errors.png"


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


def plot_dof_curves(
    data: Dict[str, np.ndarray],
    episode: int,
    dof_name: str,
    err_key: str,
    vel_key: str,
    act_index: Optional[int] = None,
):
    """
    单独为某个自由度画几条曲线并保存：
      - error(t)   : data[err_key]
      - velocity(t): data[vel_key]
      - action(t)  : data['acts'][:, act_index]（如果存在）

    dof_name 只用于文件名和 title，例如 'x'、'y'、'heading' 等。
    """
    t = data["t"]

    if err_key not in data or vel_key not in data:
        return None

    err = data[err_key].astype(float)
    vel = data[vel_key].astype(float)

    # 航向误差改成角度便于看
    if err_key == "err_heading":
        err = np.rad2deg(err)
        err_label = f"{err_key} (deg)"
    else:
        err_label = f"{err_key}"
    vel_label = f"{vel_key}"

    # 取对应的动作
    act = None
    acts = data.get("acts", None)
    if acts is not None and acts.size > 0 and act_index is not None:
        if 0 <= act_index < acts.shape[1]:
            act = acts[:, act_index].astype(float)

    n_rows = 3 if act is not None else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(8, 2.6 * n_rows), sharex=True)

    if n_rows == 2:
        ax_err, ax_vel = axes
        ax_act = None
    else:
        ax_err, ax_vel, ax_act = axes

    # 误差曲线
    ax_err.plot(t, err)
    ax_err.set_ylabel(err_label)
    ax_err.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    # 速度曲线
    ax_vel.plot(t, vel)
    ax_vel.set_ylabel(vel_label)
    ax_vel.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    # 动作曲线
    if ax_act is not None and act is not None:
        ax_act.plot(t, act)
        ax_act.set_ylabel(f"action[{act_index}]")
        ax_act.set_xlabel("time step")
        ax_act.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    else:
        ax_vel.set_xlabel("time step")

    fig.suptitle(f"Episode {episode}: DOF '{dof_name}' error / velocity / action", y=0.95)
    fig.tight_layout()

    filename = f"episode_{episode:03d}_dof_{dof_name}.png"
    return fig, filename


def create_episode_animation(
    data: Dict[str, np.ndarray],
    episode: int,
    out_dir: Path,
    fps: int = 20,
    dpi: int = 150,
) -> None:
    """
    为指定 episode 生成轨迹 + 误差/动作 的 mp4 动图，直接保存到 out_dir，
    不弹出任何窗口。
    """
    t = data["t"]
    x = data["x"]
    y = data["y"]
    gx = data.get("goal_x", np.full_like(x, np.nan, dtype=float))
    gy = data.get("goal_y", np.full_like(y, np.nan, dtype=float))
    dist = data.get("dist", np.full_like(t, np.nan, dtype=float))
    err_x = data.get("err_x", np.full_like(t, np.nan, dtype=float))
    err_y = data.get("err_y", np.full_like(t, np.nan, dtype=float))
    err_h = data.get("err_heading", np.full_like(t, np.nan, dtype=float))
    acts = data.get("acts", None)
    act_cols = data.get("act_cols", None)

    T = len(t)
    if T == 0:
        print(f"[plot_auv] Episode {episode}: no data, skip animation.")
        return

    # 画布：上半部分平面轨迹，下半部分误差 + 动作随时间
    fig = plt.figure(figsize=(8, 8))

    ax_traj = fig.add_subplot(2, 1, 1)
    ax_traj.set_title(f"Episode {episode}: AUV Trajectory (World Frame)")
    ax_traj.set_xlabel("x [m]")
    ax_traj.set_ylabel("y [m]")
    ax_traj.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    # 整体目标轨迹
    goal_mask = ~np.isnan(gx) & ~np.isnan(gy)
    if np.any(goal_mask):
        ax_traj.plot(gx[goal_mask], gy[goal_mask], linestyle="--", alpha=0.5, label="Goal path")

    # AUV 历史轨迹和当前点、当前目标点
    (line_auv,) = ax_traj.plot([], [], linewidth=2.0, label="AUV path")
    (point_auv,) = ax_traj.plot([], [], marker="o", markersize=6)
    (point_goal,) = ax_traj.plot([], [], marker="x", markersize=6)
    (line_pred_goal,) = ax_traj.plot([], [], linestyle=":", linewidth=1.5, alpha=0.9, label="WM pred goal (30)")
    (line_cv_goal,) = ax_traj.plot([], [], linestyle="--", linewidth=1.2, alpha=0.8, label="CV pred goal (30)")

    ax_traj.legend(loc="best")

    # 下半部分：误差 + 动作
    ax_err = fig.add_subplot(2, 1, 2)
    ax_err.set_title("Errors and Actions over Time")
    ax_err.set_xlabel("time step")
    ax_err.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    (line_dist,) = ax_err.plot([], [], label="dist")
    (line_ex,) = ax_err.plot([], [], label="err_x")
    (line_ey,) = ax_err.plot([], [], label="err_y")

    act_lines: List = []
    if acts is not None and acts.size > 0:
        act_dim = acts.shape[1]
        # 如果有 act_cols 就用 act_cols 做标签，否则用 action_0,1,...
        if act_cols is not None and len(act_cols) == act_dim:
            labels = [str(c) for c in act_cols]
        else:
            labels = [f"action_{i}" for i in range(act_dim)]
        for lab in labels:
            (ln,) = ax_err.plot([], [], label=lab)
            act_lines.append(ln)

    ax_err.legend(ncol=2, fontsize=8)

    # 左上角文本：当前 step 的误差 + 动作
    text_info = ax_traj.text(
        0.02,
        0.95,
        "",
        transform=ax_traj.transAxes,
        verticalalignment="top",
        fontsize=9,
    )

    # 设置 traj 图的坐标范围
    valid_x = x[~np.isnan(x)]
    valid_y = y[~np.isnan(y)]
    if np.any(goal_mask):
        valid_x = np.concatenate([valid_x, gx[goal_mask]])
        valid_y = np.concatenate([valid_y, gy[goal_mask]])

    if valid_x.size > 0 and valid_y.size > 0:
        margin = 1.0
        ax_traj.set_xlim(valid_x.min() - margin, valid_x.max() + margin)
        ax_traj.set_ylim(valid_y.min() - margin, valid_y.max() + margin)

    # err 图的 y 轴范围
    y_candidates = []
    for arr in (dist, err_x, err_y):
        if arr.size and not np.all(np.isnan(arr)):
            y_candidates.append(arr[~np.isnan(arr)])
    if acts is not None and acts.size > 0:
        finite_acts = acts[np.isfinite(acts)]
        if finite_acts.size:
            y_candidates.append(finite_acts)
    if y_candidates:
        ymin = min(np.min(c) for c in y_candidates)
        ymax = max(np.max(c) for c in y_candidates)
        if np.isfinite(ymin) and np.isfinite(ymax):
            pad = 0.1 * (ymax - ymin + 1e-6)
            ax_err.set_ylim(ymin - pad, ymax + pad)
    ax_err.set_xlim(t.min(), t.max())

    def init():
        line_auv.set_data([], [])
        point_auv.set_data([], [])
        point_goal.set_data([], [])
        line_pred_goal.set_data([], [])
        line_cv_goal.set_data([], [])
        line_dist.set_data([], [])
        line_ex.set_data([], [])
        line_ey.set_data([], [])
        for ln in act_lines:
            ln.set_data([], [])
        text_info.set_text("")
        return (
            line_auv,
            point_auv,
            point_goal,
            line_pred_goal,
            line_cv_goal,
            line_dist,
            line_ex,
            line_ey,
            *act_lines,
            text_info,
        )

    def update(frame: int):
        i = frame
        # 轨迹（线：前 i+1 个点；当前点：长度1的序列）
        line_auv.set_data(x[: i + 1], y[: i + 1])
        point_auv.set_data([x[i]], [y[i]])  # ★ 必须是序列

        if not np.isnan(gx[i]) and not np.isnan(gy[i]):
            point_goal.set_data([gx[i]], [gy[i]])  # ★ 必须是序列
        else:
            point_goal.set_data([], [])

        # WM predicted future goal trajectory (world frame)
        pred_list = data.get("wm_pred_goal_xy_list", None)
        if pred_list is not None and len(pred_list) > i:
            pred_xy = pred_list[i]
            if isinstance(pred_xy, np.ndarray) and pred_xy.ndim == 2 and pred_xy.shape[1] == 2 and pred_xy.shape[0] > 0:
                line_pred_goal.set_data(pred_xy[:, 0], pred_xy[:, 1])
            else:
                line_pred_goal.set_data([], [])
        else:
            line_pred_goal.set_data([], [])

        # Constant-velocity baseline predicted goal trajectory (world frame)
        cv_list = data.get("cv_pred_goal_xy_list", None)
        if cv_list is not None and len(cv_list) > i:
            cv_xy = cv_list[i]
            if isinstance(cv_xy, np.ndarray) and cv_xy.ndim == 2 and cv_xy.shape[1] == 2 and cv_xy.shape[0] > 0:
                line_cv_goal.set_data(cv_xy[:, 0], cv_xy[:, 1])
            else:
                line_cv_goal.set_data([], [])
        else:
            line_cv_goal.set_data([], [])
# 误差 + 动作
        tt = t[: i + 1]
        line_dist.set_data(tt, dist[: i + 1])
        line_ex.set_data(tt, err_x[: i + 1])
        line_ey.set_data(tt, err_y[: i + 1])

        if acts is not None and acts.size > 0:
            for j, ln in enumerate(act_lines):
                ln.set_data(tt, acts[: i + 1, j])

        # 文本
        msg_lines = [
            f"t = {t[i]:.0f}",
            f"dist = {dist[i]:.3f}",
            f"err_x = {err_x[i]:.3f}",
            f"err_y = {err_y[i]:.3f}",
            f"err_heading = {err_h[i]:.3f} rad",
        ]
        gvx_arr = data.get("goal_vx", None)
        gvy_arr = data.get("goal_vy", None)
        if isinstance(gvx_arr, np.ndarray) and i < len(gvx_arr) and not np.isnan(gvx_arr[i]):
            msg_lines.append(f"goal_vx = {gvx_arr[i]:.3f}")
        if isinstance(gvy_arr, np.ndarray) and i < len(gvy_arr) and not np.isnan(gvy_arr[i]):
            msg_lines.append(f"goal_vy = {gvy_arr[i]:.3f}")
        wm_conf_arr = data.get("wm_conf", None)
        wm_mse_arr = data.get("wm_mse_local", None)
        if wm_conf_arr is not None and i < len(wm_conf_arr) and not np.isnan(wm_conf_arr[i]):
            msg_lines.append(f"wm_conf = {wm_conf_arr[i]:.3f}")
        if wm_mse_arr is not None and i < len(wm_mse_arr) and not np.isnan(wm_mse_arr[i]):
            msg_lines.append(f"wm_mse_local = {wm_mse_arr[i]:.6f}")

        if acts is not None and acts.size > 0:
            act_vals = ", ".join(f"{acts[i, j]:.3f}" for j in range(acts.shape[1]))
            msg_lines.append(f"acts: {act_vals}")
        text_info.set_text("\n".join(msg_lines))

        return (
            line_auv,
            point_auv,
            point_goal,
            line_pred_goal,
            line_cv_goal,
            line_dist,
            line_ex,
            line_ey,
            *act_lines,
            text_info,
        )

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=T,
        init_func=init,
        blit=True,
        interval=1000.0 / float(fps),
    )

    out_path = out_dir / f"episode_{episode:03d}_anim.mp4"
    print(f"[plot_auv] Saving animation to: {out_path}")
    try:
        ani.save(out_path, fps=fps, dpi=dpi)
    except Exception as e:
        print(f"[plot_auv] Failed to save animation: {e}")
    plt.close(fig)


# ===================== 主入口 =====================

def main():
    parser = argparse.ArgumentParser(description="Plot AUV evaluation trajectories")
    parser.add_argument("--csv", type=str, default="eval_outputs/trajectories.csv")
    parser.add_argument("--episode", type=int, default=0, help="Episode index for per-step plots and animation")
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
        default=1.0,
        help="Distance threshold (m) used to judge 'good tracking' when computing track_ratio",
    )
    parser.add_argument(
        "--track_success_ratio",
        type=float,
        default=0.8,
        help="Episode is considered SUCCESS if track_ratio >= this value.",
    )
    parser.add_argument(
        "--anim_fps",
        type=int,
        default=20,
        help="Frames per second of the saved animation video.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    rows, fieldnames = load_csv(csv_path)
    grouped = group_rows_by_episode(rows)
    if not grouped:
        raise ValueError("No episode data found in CSV.")

    # 转换为 numpy 数组便于后续计算
    episode_arrays: Dict[int, Dict[str, np.ndarray]] = {
        ep: prepare_episode_arrays(ep_rows, fieldnames) for ep, ep_rows in grouped.items()
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
    print(f"[plot_auv] Saving figures and animation to: {out_dir}")

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

    traj_fig = plot_trajectory(ep_data, target_episode, ep_success, args.success_threshold)
    if traj_fig is not None:
        figures.append(traj_fig)

    speed_fig = plot_speed_profiles(ep_data, target_episode)
    figures.append(speed_fig)

    dist_fig = plot_distance_reward(ep_data, target_episode)
    figures.append(dist_fig)

    # 误差随时间曲线
    err_fig = plot_error_profiles(ep_data, target_episode)
    if err_fig is not None:
        figures.append(err_fig)

    overview_figs = plot_success_histories(
        episode_arrays, args.success_threshold, args.track_success_ratio
    )
    if overview_figs:
        figures.extend(overview_figs)

        # ===== 单独保存几个自由度的 error + 输入 + 输出速度 曲线 =====
    # DOF 1：x 自由度，err_x + u + action_0
    dof_x_fig = plot_dof_curves(
        ep_data,
        target_episode,
        dof_name="x",
        err_key="err_x",
        vel_key="u",
        act_index=0,   # action[0] 对应纵向推进器
    )
    if dof_x_fig is not None:
        figures.append(dof_x_fig)

    # DOF 2：y 自由度，err_y + v + action_1
    dof_y_fig = plot_dof_curves(
        ep_data,
        target_episode,
        dof_name="y",
        err_key="err_y",
        vel_key="v",
        act_index=1,   # action[1] 主要影响横向/转向（舵）
    )
    if dof_y_fig is not None:
        figures.append(dof_y_fig)

    # DOF 3：航向自由度，err_heading + r + action_1
    dof_heading_fig = plot_dof_curves(
        ep_data,
        target_episode,
        dof_name="heading",
        err_key="err_heading",
        vel_key="r",
        act_index=1,   # 航向同样主要由舵控制
    )
    if dof_heading_fig is not None:
        figures.append(dof_heading_fig)


    for fig, name in figures:
        fig.savefig(out_dir / name, dpi=args.dpi)
        plt.close(fig)
        print(f"[plot_auv] Saved {name}")

    # 生成动画（直接保存 mp4）
    create_episode_animation(
        ep_data,
        target_episode,
        out_dir,
        fps=args.anim_fps,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
