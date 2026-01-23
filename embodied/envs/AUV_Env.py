#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUVEnv (Dreamer/embodied.Env)

当前版本：方案1—— cursor + window + lookahead 的“历史轨迹最近点”跟踪

核心变化（保持外部接口不变）：
1) 目标参考点 self.goal：不再取 delay 队列最老点；改为在历史真实目标轨迹中，使用 cursor+window 搜索最近点，
   再加 lookahead，且窗口会持续向前滑动（带防卡死强制推进）。
2) 修复隐患：
   - phase_cos/sin 不再是伪相位：改为“路径切向向量（在艇体坐标系）”两维，仍占用 vector 的第 12/13 维。
   - progress 不再用 prev_dist - dist_td：改为“沿轨迹推进弧长 delta_s”，稳定且不会被参考点跳变污染。
   - overshoot 不再用 episode 全局最小：改为“分段最小值（随 cursor 前进重置）”，避免误报。
   - vector[2] 从 raw dist 改为 dist_td（平滑距离），使观测与 reward 一致更稳定；log/dist 仍保留 raw dist。

外部接口不变：
- obs_space keys, act_space keys 不变
- vector shape (15,) 不变
- log/* 不新增键、不改键名
- PBT / eval / main 不需要改代码（只需按需新增 env.auv.* 覆盖参数）

"""

import os
import csv
import math
import numpy as np
from pathlib import Path
from collections import deque
from itertools import islice

import elements
import embodied


# ========== Tracking Differentiator ==========

class TrackingDifferentiator:
    """
    Han 型二阶 TD，用于对输入 v(t) 做平滑跟踪并给出带限幅的导数。
    """
    def __init__(self, r=2.0, h=0.05, N=5.0):
        self.r = float(r)
        self.h = float(h)
        self.N = float(N)
        self.x1 = 0.0
        self.x2 = 0.0

    def reset(self, v0=0.0):
        self.x1 = float(v0)
        self.x2 = 0.0

    def step(self, v):
        r = self.r
        h = self.h
        h0 = self.N * h

        x1, x2 = self.x1, self.x2
        v = float(v)

        d = r * h0 * h0
        a0 = x2 * h
        y = x1 - v + a0

        if abs(y) > d:
            a1 = math.sqrt(d * (d + 8.0 * abs(y)))
            a2 = a0 + 0.5 * (a1 - d) * math.copysign(1.0, y)
        else:
            a2 = a0 + y

        x1 = x1 + h * x2
        x2 = x2 - r * a2

        self.x1, self.x2 = x1, x2
        return x1, x2   # x1: 平滑距离, x2: 距离变化率


# ----------------- 动力学 / 运动学模型 -----------------

def update_model_state_dyn(state, input, dt):
    Xuu = -1.62e0
    Nvv = -3.18e0
    Yvv = -1.31e0
    Yrr = 6.32e-1
    Nrr = -9.4e+1
    Xu = -9.4e-1
    Yv = -3.55e+1
    Nv = 1.93e0
    Nr = -4.88e0
    Xvr = 3.55e+1
    Xrr = -1.93e0
    Yur = 5.22e0
    Nur = -2e0
    Yuv = -2.86e1
    Yuudr = 9.64e0
    Nuudr = -6.15e0

    Yr = 1.93e0
    Nuv = -2.4e1

    m = 30.51e0
    Iz = 3.45e0
    xg = 0.0
    yg = 0.0

    Xprop, deltar = input
    u, v, r = state

    # 限幅（防炸）
    u = np.clip(u, -10.0, 10.0)
    v = np.clip(v, -10.0, 10.0)
    r = np.clip(r, -6.0, 6.0)
    deltar = np.clip(deltar, -1.5, 1.5)
    Xprop = np.clip(Xprop, -300.0, 300.0)

    M = np.array([
        [m - Xu, 0.0, -m * yg],
        [0.0, m - Yv, m * xg - Yr],
        [-m * yg, m * xg - Nv, Iz - Nr],
    ])

    tau = np.array([
        [Xuu * abs(u) * u + Xvr * v * r + Xrr * r * r + Xprop + m * v * r + m * xg * r * r],
        [Yvv * abs(v) * v + Yrr * abs(r) * r + Yur * u * r + Yuv * u * v + Yuudr * u * u * deltar - m * (u * r - yg * r * r)],
        [Nvv * abs(v) * v + Nrr * abs(r) * r + Nur * u * r + Nuv * u * v + Nuudr * u * u * deltar - m * (xg * u * r + yg * v * r)],
    ])

    accel = np.linalg.inv(M) @ tau
    accel = np.clip(accel, -100.0, 100.0)

    new_state = np.array([u, v, r]) + accel.flatten() * dt
    new_state = np.clip(new_state, -10.0, 10.0)
    return new_state


def update_model_state_kine(state, input, dt):
    u, v, r = input
    x, y, theta = state

    x += (math.cos(theta) * u - math.sin(theta) * v) * dt
    y += (math.sin(theta) * u + math.cos(theta) * v) * dt
    theta += r * dt
    theta = (theta + math.pi) % (2.0 * math.pi) - math.pi
    return np.array([x, y, theta])


def goal_in_body_frame(state_pos, goal):
    """
    state_pos: [x, y, theta] in world frame
    goal: [gx, gy] in world frame
    return: (x_b, y_b, dist) 目标在船体坐标系下的位置和距离
    """
    x, y, theta = state_pos
    gx, gy = goal
    dx = gx - x
    dy = gy - y
    c = math.cos(theta)
    s = math.sin(theta)
    xb = c * dx + s * dy
    yb = -s * dx + c * dy
    dist = float(math.hypot(xb, yb))
    return xb, yb, dist


def world_vec_to_body(theta, vx, vy):
    c = math.cos(theta)
    s = math.sin(theta)
    bx = c * vx + s * vy
    by = -s * vx + c * vy
    return bx, by


# ----------------- 连续动作 + 移动目标的 AUV 环境 -----------------

class AUVEnv(embodied.Env):
    """
    AUV 3 自由度（x, y, ψ）+ 动力学模型环境（连续动作）
    """

    def __init__(
        self,
        task=None,
        dt=0.05,
        max_steps=800,
        success_radius=0.5,
        w_heading=0.1,
        thrust_scale=50.0,
        rudder_max=0.6,

        # === 目标相关参数 ===
        moving_goal=True,
        goal_center=(10.0, 10.0),
        goal_radius=6.0,
        goal_speed=0.3,          # 保留参数（旧 config 兼容），本版本不再用于 phase
        goal_delay_steps=5,      # 保留：只允许使用“至少 delay 步之前”的轨迹点集合

        # === 轨迹最近点跟随（方案1）参数 ===
        goal_hist_len=800,               # 历史真实目标轨迹缓存长度
        nearest_forward_window=200,      # cursor 向前搜索窗口（索引）
        nearest_backtrack_allow=20,      # 允许回退搜索的窗口（索引）
        nearest_lookahead=10,            # 参考点前视（索引）
        nearest_stall_steps=60,          # 连续多少步 cursor 不推进 -> 触发防卡死判断
        nearest_stall_radius=0.8,        # 若距离参考点 < 该半径且 stall -> 强制推进
        nearest_force_advance=3,         # 防卡死：一次强制推进多少索引

        # overshoot 分段重置：cursor 推进多少索引后重置 segment min
        overshoot_reset_idx_delta=20,

        # AUV 自身最大速度 / 角速度
        max_auv_speed=5.0,
        max_auv_turn_rate=3.0,

        # 目标最大速度 / 角速度
        max_goal_speed=2.0,
        max_goal_turn_rate=1.2,

        # 目标控制尺度和更新策略
        goal_thrust_scale=20.0,
        goal_rudder_max=0.3,
        goal_ctrl_interval=10,
        goal_ctrl_smooth=0.8,

        goal_custom_fn=None,

        # === Reward 中能量 / 平滑项 ===
        energy_thrust_coef=1e-2,
        energy_rudder_coef=1e-2,
        smooth_ctrl_coef=1e-2,
        smooth_vel_coef=1e-1,

        # === Reward 相关可调参数（用于 PBT / 超参搜索） ===
        base_k_progress=2.0,
        k_dist=1.0,
        k_ring=0.4,
        bonus_max=1.5,
        hold_bonus=0.5,
        k_speed_near=0.3,
        gamma_far=0.5,
        k_heading_base=0.4,

        # === TD 参数 ===
        td_r=1.0,
        td_N=8.0,

        # === Debug / overshoot ===
        seed=0,
        debug_trace=False,
        trace_dir="",
        save_trace_on_done=True,
        save_trace_on_overshoot=True,
        overshoot_eps=0.2,
        overshoot_min_steps=20,
        overshoot_require_distdot=True,   # True: 需要 dist_dot_td>0 才算回弹
        **kwargs,
    ):
        del task, kwargs

        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.success_radius = float(success_radius)
        self.w_heading = float(w_heading)
        self.thrust_scale = float(thrust_scale)
        self.rudder_max = float(rudder_max)

        self.moving_goal = bool(moving_goal)
        self.goal_center = tuple(goal_center)
        self.goal_radius = float(goal_radius)
        self.goal_speed = float(goal_speed)
        self.goal_delay_steps = int(goal_delay_steps)

        # ---- scheme1 params ----
        self.goal_hist_len = int(goal_hist_len)
        self.nearest_forward_window = int(nearest_forward_window)
        self.nearest_backtrack_allow = int(nearest_backtrack_allow)
        self.nearest_lookahead = int(nearest_lookahead)
        self.nearest_stall_steps = int(nearest_stall_steps)
        self.nearest_stall_radius = float(nearest_stall_radius)
        self.nearest_force_advance = int(nearest_force_advance)
        self.overshoot_reset_idx_delta = int(overshoot_reset_idx_delta)

        self.max_auv_speed = float(max_auv_speed)
        self.max_auv_turn_rate = float(max_auv_turn_rate)

        self.max_goal_speed = float(max_goal_speed)
        self.max_goal_turn_rate = float(max_goal_turn_rate)

        self.goal_thrust_scale = float(goal_thrust_scale)
        self.goal_rudder_max = float(goal_rudder_max)
        self.goal_ctrl_interval = int(goal_ctrl_interval)
        self.goal_ctrl_smooth = float(goal_ctrl_smooth)
        self.goal_custom_fn = goal_custom_fn

        self.energy_thrust_coef = float(energy_thrust_coef)
        self.energy_rudder_coef = float(energy_rudder_coef)
        self.smooth_ctrl_coef = float(smooth_ctrl_coef)
        self.smooth_vel_coef = float(smooth_vel_coef)

        self.base_k_progress = float(base_k_progress)
        self.k_dist = float(k_dist)
        self.k_ring = float(k_ring)
        self.bonus_max = float(bonus_max)
        self.hold_bonus = float(hold_bonus)
        self.k_speed_near = float(k_speed_near)
        self.gamma_far = float(gamma_far)
        self.k_heading_base = float(k_heading_base)

        self.td_r = float(td_r)
        self.td_N = float(td_N)

        # RNG
        self.seed = int(seed)
        self.np_random = np.random.RandomState(self.seed)

        # 状态缓存
        self.prev_control = np.zeros(2, dtype=float)
        self.prev_vel = np.zeros(3, dtype=float)

        self.steps = 0
        self.done = False

        self.state_pos = np.zeros(3, dtype=float)   # [x, y, theta]
        self.state_vel = np.zeros(3, dtype=float)   # [u, v, r]

        self.goal = np.zeros(2, dtype=float)        # 参考目标（用于追踪）
        self.goal_live = np.zeros(2, dtype=float)   # 真实目标点（轨迹生成）
        self.goal_pos = np.zeros(3, dtype=float)
        self.goal_vel = np.zeros(3, dtype=float)
        self.goal_control = np.zeros(2, dtype=float)
        self.goal_ctrl_step = 0

        self.time = 0.0

        # 距离 TD（对 dist 做平滑）
        self.td_dist = TrackingDifferentiator(r=self.td_r, h=self.dt, N=self.td_N)

        # ===== scheme1 trajectory buffers =====
        self._traj_pts = deque(maxlen=max(8, self.goal_hist_len))
        self._traj_s = deque(maxlen=max(8, self.goal_hist_len))      # cumulative arc length
        self._cursor_idx = 0
        self._prev_cursor_s = 0.0
        self._stall_count = 0

        # tangent in body frame (stored each step)
        self._tan_bx = 1.0
        self._tan_by = 0.0

        # overshoot segment min
        self._seg_min_dist_td = None
        self._seg_cursor0 = 0

        # ===== overshoot / trace =====
        self.debug_trace = bool(debug_trace)
        self.trace_dir = str(trace_dir) if trace_dir else ""
        self.save_trace_on_done = bool(save_trace_on_done)
        self.save_trace_on_overshoot = bool(save_trace_on_overshoot)
        self.overshoot_eps = float(overshoot_eps)
        self.overshoot_min_steps = int(overshoot_min_steps)
        self.overshoot_require_distdot = bool(overshoot_require_distdot)

        self._episode_id = 0
        self._trace = []
        self._overshoot = None       # dict or None
        self._saved_trace = False

        if self.debug_trace and self.trace_dir:
            Path(self.trace_dir).mkdir(parents=True, exist_ok=True)

    # === 目标轨迹（动力学 + 平滑随机控制）===
    def _goal_traj(self, t):
        if self.goal_custom_fn is not None:
            gx, gy = self.goal_custom_fn(t)
            return np.array([gx, gy], dtype=float)

        if self.goal_ctrl_step % self.goal_ctrl_interval == 0:
            noise = self.np_random.uniform(-1.0, 1.0, size=2)
            target_ctrl = np.array(
                [noise[0] * self.goal_thrust_scale, noise[1] * self.goal_rudder_max],
                dtype=float,
            )
            self.goal_control = (
                self.goal_ctrl_smooth * self.goal_control
                + (1.0 - self.goal_ctrl_smooth) * target_ctrl
            )

        self.goal_ctrl_step += 1
        control_g = self.goal_control

        self.goal_vel = update_model_state_dyn(self.goal_vel, control_g, self.dt)

        self.goal_vel[0] = np.clip(self.goal_vel[0], -self.max_goal_speed, self.max_goal_speed)
        self.goal_vel[1] = np.clip(self.goal_vel[1], -self.max_goal_speed, self.max_goal_speed)
        self.goal_vel[2] = np.clip(self.goal_vel[2], -self.max_goal_turn_rate, self.max_goal_turn_rate)

        self.goal_pos = update_model_state_kine(self.goal_pos, self.goal_vel, self.dt)
        return self.goal_pos[:2].copy()

    # ===== trace helpers =====
    def _trace_append(self, row: dict):
        if not self.debug_trace:
            return
        self._trace.append(row)

    def _save_trace(self, reason: str):
        if (not self.debug_trace) or self._saved_trace:
            return
        if not self.trace_dir:
            return
        try:
            outdir = Path(self.trace_dir)
            outdir.mkdir(parents=True, exist_ok=True)
            base = f"ep{self._episode_id:06d}_{reason}"
            csv_path = outdir / f"{base}.csv"
            meta_path = outdir / f"{base}.meta.txt"

            if self._trace:
                keys = list(self._trace[0].keys())
                with open(csv_path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=keys)
                    w.writeheader()
                    for r in self._trace:
                        w.writerow(r)

            with open(meta_path, "w") as f:
                f.write(f"episode_id: {self._episode_id}\n")
                f.write(f"reason: {reason}\n")
                f.write(f"seed: {self.seed}\n")
                f.write(f"overshoot_eps: {self.overshoot_eps}\n")
                f.write(f"overshoot_min_steps: {self.overshoot_min_steps}\n")
                f.write(f"overshoot: {self._overshoot}\n")

            self._saved_trace = True
        except Exception as e:
            print(f"[AUVEnv] WARNING: failed to save trace: {e}")

    # ===== Dreamer 接口定义 =====
    @property
    def obs_space(self):
        scalar_f = elements.Space(np.float32, ())
        return {
            "vector": elements.Space(np.float32, (15,)),
            "reward": scalar_f,
            "is_first": elements.Space(bool, ()),
            "is_last": elements.Space(bool, ()),
            "is_terminal": elements.Space(bool, ()),

            "log/dist": scalar_f,
            "log/dist_td": scalar_f,
            "log/dist_dot_td": scalar_f,
            "log/min_dist_td": scalar_f,
            "log/heading_err": scalar_f,

            "log/Xprop": scalar_f,
            "log/deltar": scalar_f,
            "log/u": scalar_f,
            "log/v": scalar_f,
            "log/r": scalar_f,

            "log/r_progress": scalar_f,
            "log/r_heading": scalar_f,
            "log/r_dist": scalar_f,
            "log/goal_bonus": scalar_f,
            "log/hold_bonus": scalar_f,
            "log/energy_cost": scalar_f,
            "log/smooth_cost_ctrl": scalar_f,
            "log/smooth_cost_vel": scalar_f,
            "log/speed_cost_near": scalar_f,
            "log/ring_cost": scalar_f,

            "log/overshoot": scalar_f,
            "log/overshoot_x": scalar_f,
            "log/overshoot_y": scalar_f,
            "log/overshoot_gx": scalar_f,
            "log/overshoot_gy": scalar_f,
        }

    @property
    def act_space(self):
        return {
            "reset": elements.Space(bool, ()),
            "action": elements.Space(np.float32, (2,), -1.0, 1.0),
        }

    def close(self):
        pass

    def _parse_action(self, action):
        a = action.get("action", action)
        a = np.array(a, dtype=np.float32).reshape(-1)
        if a.size == 1:
            a = np.array([a.item(), 0.0], dtype=np.float32)
        assert a.size == 2, f"Continuous action must have 2 dims, got {a.size}"
        a = np.clip(a, -1.0, 1.0)
        Xprop = float(self.thrust_scale * a[0])
        deltar = float(self.rudder_max * a[1])
        return Xprop, deltar

    # ===== scheme1 internals =====
    def _traj_append(self, pt_xy: np.ndarray):
        """
        Append a new real goal point to trajectory buffer, maintaining cumulative arc length.
        Handles deque maxlen eviction and keeps cursor indices consistent.
        """
        pt_xy = np.asarray(pt_xy, dtype=float).reshape(2,)

        # If deque is full, the leftmost element will be evicted on append.
        # Pre-adjust cursor/segment anchors to remain aligned to new indexing.
        if len(self._traj_pts) == self._traj_pts.maxlen:
            self._cursor_idx = max(0, self._cursor_idx - 1)
            self._seg_cursor0 = max(0, self._seg_cursor0 - 1)

        if len(self._traj_pts) == 0:
            s_new = 0.0
        else:
            last = self._traj_pts[-1]
            ds = float(np.linalg.norm(pt_xy - last))
            s_new = float(self._traj_s[-1] + ds)

        self._traj_pts.append(pt_xy.copy())
        self._traj_s.append(s_new)

    def _select_goal_by_path(self):
        """
        Scheme1: cursor + window + lookahead.
        Uses only points up to avail_last = len(traj)-1-goal_delay_steps (i.e., already produced trajectory).
        Updates:
          - self.goal (reference goal point)
          - self._tan_bx, self._tan_by (path tangent in body frame)
          - self._cursor_idx, self._stall_count
        Returns:
          ref_dist (float): distance from AUV to reference goal (raw dist)
          forced_advance (bool): whether anti-stall forced advance happened
          cursor_s (float): cumulative arc length at cursor
        """
        forced_advance = False

        L = len(self._traj_pts)
        if L == 0:
            self.goal = self.goal_live.copy()
            self._tan_bx, self._tan_by = 1.0, 0.0
            return 0.0, forced_advance, 0.0

        # only allow using points that are at least delay_steps old
        avail_last = L - 1 - max(0, self.goal_delay_steps)
        if avail_last < 0:
            avail_last = 0

        # clamp cursor within available range
        self._cursor_idx = int(np.clip(self._cursor_idx, 0, avail_last))

        # window end needs to include lookahead + possible force advance margin
        margin = max(self.nearest_lookahead, self.nearest_force_advance, 0) + 2
        start = max(0, self._cursor_idx - max(0, self.nearest_backtrack_allow))
        end = min(avail_last, self._cursor_idx + max(0, self.nearest_forward_window) + margin)
        if end < start:
            start = end

        # collect candidate points (window)
        cand_pts = list(islice(self._traj_pts, start, end + 1))
        cand_s = list(islice(self._traj_s, start, end + 1))
        if len(cand_pts) == 0:
            self.goal = self._traj_pts[avail_last].copy()
            self._tan_bx, self._tan_by = 1.0, 0.0
            return 0.0, forced_advance, float(self._traj_s[self._cursor_idx])

        pts = np.asarray(cand_pts, dtype=float)  # [K,2]

        x, y = float(self.state_pos[0]), float(self.state_pos[1])
        dx = pts[:, 0] - x
        dy = pts[:, 1] - y
        d2 = dx * dx + dy * dy
        best_local = int(np.argmin(d2))
        best_idx = start + best_local

        # basic stall detection (cursor not advancing)
        if best_idx <= self._cursor_idx:
            self._stall_count += 1
        else:
            self._stall_count = 0

        # advance cursor (monotonic)
        cursor_next = max(self._cursor_idx, best_idx)

        # anti-stall: if stuck and already close to reference, force advance
        dist_to_best = float(math.sqrt(float(d2[best_local])))
        if (self._stall_count >= self.nearest_stall_steps) and (dist_to_best < self.nearest_stall_radius):
            cursor_next = min(avail_last, cursor_next + max(1, self.nearest_force_advance))
            forced_advance = True
            self._stall_count = 0

        self._cursor_idx = int(np.clip(cursor_next, 0, avail_last))

        # reference index: use cursor-based lookahead (more stable than best-based)
        ref_idx = min(avail_last, self._cursor_idx + max(0, self.nearest_lookahead))

        # ref_idx must be within [start,end] candidate list for tangent computation.
        # If not, build a small local list around ref_idx (rare when cursor jumps a lot).
        if not (start <= ref_idx <= end):
            # rebuild around new cursor
            start = max(0, self._cursor_idx - max(0, self.nearest_backtrack_allow))
            end = min(avail_last, self._cursor_idx + max(0, self.nearest_forward_window) + margin)
            cand_pts = list(islice(self._traj_pts, start, end + 1))
            cand_s = list(islice(self._traj_s, start, end + 1))
            pts = np.asarray(cand_pts, dtype=float)

        local_ref = int(ref_idx - start)
        local_ref = int(np.clip(local_ref, 0, len(cand_pts) - 1))

        self.goal = np.asarray(cand_pts[local_ref], dtype=float).copy()

        # tangent: use neighbor within available range
        if (local_ref + 1) < len(cand_pts):
            p0 = np.asarray(cand_pts[local_ref], dtype=float)
            p1 = np.asarray(cand_pts[local_ref + 1], dtype=float)
        elif local_ref > 0:
            p0 = np.asarray(cand_pts[local_ref - 1], dtype=float)
            p1 = np.asarray(cand_pts[local_ref], dtype=float)
        else:
            p0 = np.asarray(cand_pts[local_ref], dtype=float)
            p1 = p0 + np.array([1.0, 0.0], dtype=float)

        tw = p1 - p0
        n = float(np.linalg.norm(tw))
        if n < 1e-6:
            tw = np.array([1.0, 0.0], dtype=float)
            n = 1.0
        tw = tw / n

        # tangent in BODY frame -> replace phase_cos/sin (two dims) with meaningful feature
        theta = float(self.state_pos[2])
        tbx, tby = world_vec_to_body(theta, float(tw[0]), float(tw[1]))
        self._tan_bx, self._tan_by = float(tbx), float(tby)

        # cursor arc length (for progress reward)
        local_cursor = int(self._cursor_idx - start)
        if 0 <= local_cursor < len(cand_s):
            cursor_s = float(cand_s[local_cursor])
        else:
            # fallback (rare)
            cursor_s = float(list(islice(self._traj_s, self._cursor_idx, self._cursor_idx + 1))[0])

        # distance to reference goal (raw)
        ref_dist = float(math.hypot(float(self.goal[0] - x), float(self.goal[1] - y)))
        return ref_dist, forced_advance, cursor_s

    # === step ===
    def step(self, action):
        if action.get("reset", False) or self.done:
            return self._reset()

        self.time += self.dt

        # --------- 目标动力学推进（真实目标） ----------
        if self.moving_goal:
            self.goal_live = self._goal_traj(self.time)
        else:
            # 静态目标：保持不变
            self.goal_live = self.goal.copy()

        # --------- 写入真实目标轨迹 ----------
        self._traj_append(self.goal_live)

        # --------- AUV 动力学推进 ----------
        Xprop, deltar = self._parse_action(action)
        control = np.array([Xprop, deltar], dtype=float)

        self.state_vel = update_model_state_dyn(self.state_vel, control, self.dt)
        self.state_vel[0] = np.clip(self.state_vel[0], -self.max_auv_speed, self.max_auv_speed)
        self.state_vel[1] = np.clip(self.state_vel[1], -self.max_auv_speed, self.max_auv_speed)
        self.state_vel[2] = np.clip(self.state_vel[2], -self.max_auv_turn_rate, self.max_auv_turn_rate)

        self.state_pos = update_model_state_kine(self.state_pos, self.state_vel, self.dt)

        # --------- scheme1: 从历史轨迹中选 reference goal ----------
        _ref_dist_raw, forced_advance, cursor_s = self._select_goal_by_path()

        # === 误差：在船体坐标系下表示参考目标位置 ===
        xb, yb, dist = goal_in_body_frame(self.state_pos, self.goal)

        # TD 平滑距离（用于 reward 与 obs 的 dist 维度）
        dist_td, dist_dot_td = self.td_dist.step(dist)

        # ---- progress: 沿轨迹推进弧长 delta_s（稳定）----
        delta_s = float(cursor_s - self._prev_cursor_s)
        self._prev_cursor_s = float(cursor_s)

        # “phase”两维不再是伪相位：改用 path tangent in body frame
        phase_cos = float(self._tan_bx)
        phase_sin = float(self._tan_by)

        t_norm = self.time / (self.max_steps * self.dt + 1e-6)

        # ========== Reward 计算开始 ==========
        eps = 1e-6

        # 用 dist_td 做归一化尺度（与旧版一致的量纲）
        base_k_progress = self.base_k_progress

        radius_ref_far = max(2.0 * self.goal_radius, 1e-6)
        x_far = np.clip(dist_td / radius_ref_far, 0.0, 1.0)
        gamma_far = self.gamma_far
        dist_norm_far = x_far ** gamma_far

        radius_ref_near = max(self.success_radius, 1e-6)
        x_near = np.clip(dist_td / radius_ref_near, 0.0, 2.0)
        dist_norm_near = x_near

        heading_err = math.atan2(yb, xb)
        k_heading_base = self.k_heading_base
        heading_scale = 0.4 + 0.6 * dist_norm_far
        k_heading_eff = k_heading_base * heading_scale
        raw_cos = math.cos(heading_err)
        cos_clipped = max(raw_cos, -0.3)
        r_heading = k_heading_eff * cos_clipped

        k_dist = self.k_dist
        r_dist = -k_dist * dist_norm_near

        # progress reward: delta_s (arc length along path)
        dist_norm_clip = np.clip(dist_norm_near, 0.0, 1.0)
        k_progress = base_k_progress * (1.3 - 0.3 * dist_norm_clip)
        s0 = 0.03  # 与旧版一致的 shaping scale；对 delta_s 通常仍有效
        r_progress = k_progress * math.tanh(delta_s / (s0 + 1e-12)) * s0

        norm_thrust = Xprop / (self.thrust_scale + eps)
        norm_rudder = deltar / (self.rudder_max + eps)
        base_energy_cost = (
            self.energy_thrust_coef * norm_thrust**2
            + self.energy_rudder_coef * norm_rudder**2
        )

        dX = Xprop - self.prev_control[0]
        d_delta = deltar - self.prev_control[1]
        norm_dX = dX / (self.thrust_scale + eps)
        norm_d_delta = d_delta / (self.rudder_max + eps)
        base_smooth_cost_ctrl = self.smooth_ctrl_coef * (norm_dX**2 + norm_d_delta**2)

        du = self.state_vel[0] - self.prev_vel[0]
        dv = self.state_vel[1] - self.prev_vel[1]
        dr = self.state_vel[2] - self.prev_vel[2]
        base_smooth_cost_vel = 2.0 * self.smooth_vel_coef * (du**2 + dv**2 + dr**2)

        energy_scale = 1.0 + 2.0 * dist_norm_far
        smooth_ctrl_scale = 1.0 + 3.0 * dist_norm_far

        a_vel_far = 0.5
        b_vel_near = 4.0

        speed2 = self.state_vel[0]**2 + self.state_vel[1]**2 + self.state_vel[2]**2
        near = np.clip(
            (1.5 * self.success_radius - dist_td) / (1.5 * self.success_radius + eps),
            0.0, 1.0
        )
        k_speed_near = self.k_speed_near
        speed_cost_near = k_speed_near * near * speed2

        smooth_vel_scale = 1.0 + a_vel_far * dist_norm_far + b_vel_near * near * (1.0 - dist_norm_far)

        energy_cost = energy_scale * base_energy_cost
        smooth_cost_ctrl = smooth_ctrl_scale * base_smooth_cost_ctrl
        smooth_cost_vel = smooth_vel_scale * base_smooth_cost_vel

        ring_cost = 0.0
        if dist_td > self.success_radius:
            t_ring = np.clip((dist_td - self.success_radius) / self.success_radius, 0.0, 1.0)
            k_ring = self.k_ring
            ring_cost = k_ring * (t_ring**2)

        goal_bonus = 0.0
        hold_bonus = 0.0
        if dist_td < self.success_radius:
            bonus_max = self.bonus_max
            proximity = 1.0 - dist_td / (self.success_radius + eps)
            goal_bonus = bonus_max * proximity
            hold_bonus = self.hold_bonus

            energy_cost *= 0.7
            smooth_cost_ctrl *= 0.7
            smooth_cost_vel *= 0.7

        reward = (
            r_progress + r_heading + r_dist
            + goal_bonus + hold_bonus
            - energy_cost - smooth_cost_ctrl - smooth_cost_vel
            - speed_cost_near - ring_cost
        )
        reward = float(np.clip(reward, -10.0, 10.0))
        # ========== Reward 计算结束 ==========

        # ===== overshoot detector（分段 min，随 cursor 前进重置）=====
        if self._seg_min_dist_td is None:
            self._seg_min_dist_td = float(dist_td)
            self._seg_cursor0 = int(self._cursor_idx)

        if forced_advance or ((int(self._cursor_idx) - int(self._seg_cursor0)) >= self.overshoot_reset_idx_delta):
            self._seg_cursor0 = int(self._cursor_idx)
            self._seg_min_dist_td = float(dist_td)
        else:
            self._seg_min_dist_td = min(float(self._seg_min_dist_td), float(dist_td))

        if (self._overshoot is None) and (self.steps >= self.overshoot_min_steps):
            cond = (dist_td > float(self._seg_min_dist_td) + self.overshoot_eps)
            if self.overshoot_require_distdot:
                cond = cond and (dist_dot_td > 0.0)
            if cond:
                self._overshoot = {
                    "t": float(self.time),
                    "step": int(self.steps),
                    "x": float(self.state_pos[0]),
                    "y": float(self.state_pos[1]),
                    "theta": float(self.state_pos[2]),
                    "u": float(self.state_vel[0]),
                    "v": float(self.state_vel[1]),
                    "r": float(self.state_vel[2]),
                    "gx": float(self.goal[0]),
                    "gy": float(self.goal[1]),
                    "dist_td": float(dist_td),
                    "dist_dot_td": float(dist_dot_td),
                    "Xprop": float(Xprop),
                    "deltar": float(deltar),
                    "heading_err": float(heading_err),
                }
                if self.save_trace_on_overshoot:
                    self._save_trace("overshoot")

        # ===== trace append (every step) =====
        self._trace_append({
            "t": float(self.time),
            "step": int(self.steps),
            "x": float(self.state_pos[0]),
            "y": float(self.state_pos[1]),
            "theta": float(self.state_pos[2]),
            "u": float(self.state_vel[0]),
            "v": float(self.state_vel[1]),
            "r": float(self.state_vel[2]),
            "gx": float(self.goal[0]),
            "gy": float(self.goal[1]),
            "xb": float(xb),
            "yb": float(yb),
            "dist": float(dist),
            "dist_td": float(dist_td),
            "dist_dot_td": float(dist_dot_td),
            "min_dist_td": float(self._seg_min_dist_td if self._seg_min_dist_td is not None else np.nan),
            "heading_err": float(heading_err),
            "Xprop": float(Xprop),
            "deltar": float(deltar),
            "r_progress": float(r_progress),
            "r_heading": float(r_heading),
            "r_dist": float(r_dist),
            "goal_bonus": float(goal_bonus),
            "hold_bonus": float(hold_bonus),
            "energy_cost": float(energy_cost),
            "smooth_cost_ctrl": float(smooth_cost_ctrl),
            "smooth_cost_vel": float(smooth_cost_vel),
            "speed_cost_near": float(speed_cost_near),
            "ring_cost": float(ring_cost),
            "reward": float(reward),
            "overshoot": float(1.0 if self._overshoot is not None else 0.0),
            "delta_s": float(delta_s),
            "cursor_idx": int(self._cursor_idx),
            "tan_bx": float(self._tan_bx),
            "tan_by": float(self._tan_by),
        })

        # 结束条件
        self.steps += 1
        if self.steps >= self.max_steps:
            self.done = True

        if self.done and self.save_trace_on_done:
            self._save_trace("done")

        # 更新 prev_*
        self.prev_control = control.copy()
        self.prev_vel = self.state_vel.copy()

        # ===== vector(15)（接口不变，语义更一致）=====
        # 维度定义保持不变：
        # [xb, yb, dist*, cos(theta), sin(theta), u, v, r, x, y, gx, gy, phase_cos, phase_sin, t_norm]
        # 其中 dist* 改为 dist_td（平滑距离），phase_cos/sin 改为切向量(body)
        obs = np.array(
            [
                xb, yb, float(dist_td),
                math.cos(self.state_pos[2]),
                math.sin(self.state_pos[2]),
                self.state_vel[0],
                self.state_vel[1],
                self.state_vel[2],
                self.state_pos[0],
                self.state_pos[1],
                self.goal[0],
                self.goal[1],
                phase_cos, phase_sin, t_norm,
            ],
            dtype=np.float32,
        )

        # overshoot log fields
        o_flag = 1.0 if (self._overshoot is not None) else 0.0
        if self._overshoot is None:
            ox = oy = ogx = ogy = np.nan
        else:
            ox = self._overshoot["x"]
            oy = self._overshoot["y"]
            ogx = self._overshoot["gx"]
            ogy = self._overshoot["gy"]

        return dict(
            vector=obs,
            reward=np.float32(reward),
            is_first=False,
            is_last=self.done,
            is_terminal=False,

            **{
                "log/dist": np.float32(dist),
                "log/dist_td": np.float32(dist_td),
                "log/dist_dot_td": np.float32(dist_dot_td),
                "log/min_dist_td": np.float32(self._seg_min_dist_td if self._seg_min_dist_td is not None else np.nan),
                "log/heading_err": np.float32(heading_err),

                "log/Xprop": np.float32(Xprop),
                "log/deltar": np.float32(deltar),
                "log/u": np.float32(self.state_vel[0]),
                "log/v": np.float32(self.state_vel[1]),
                "log/r": np.float32(self.state_vel[2]),

                "log/r_progress": np.float32(r_progress),
                "log/r_heading": np.float32(r_heading),
                "log/r_dist": np.float32(r_dist),
                "log/goal_bonus": np.float32(goal_bonus),
                "log/hold_bonus": np.float32(hold_bonus),
                "log/energy_cost": np.float32(energy_cost),
                "log/smooth_cost_ctrl": np.float32(smooth_cost_ctrl),
                "log/smooth_cost_vel": np.float32(smooth_cost_vel),
                "log/speed_cost_near": np.float32(speed_cost_near),
                "log/ring_cost": np.float32(ring_cost),

                "log/overshoot": np.float32(o_flag),
                "log/overshoot_x": np.float32(ox),
                "log/overshoot_y": np.float32(oy),
                "log/overshoot_gx": np.float32(ogx),
                "log/overshoot_gy": np.float32(ogy),
            }
        )

    def _reset(self):
        self.steps = 0
        self.done = False
        self.time = 0.0

        # AUV 初始状态
        self.state_pos = np.array(
            [
                self.np_random.uniform(0.0, 5.0),
                self.np_random.uniform(0.0, 5.0),
                self.np_random.uniform(-math.pi, math.pi),
            ],
            dtype=float,
        )
        self.state_vel = np.zeros(3, dtype=float)

        self.prev_control = np.zeros(2, dtype=float)
        self.prev_vel = self.state_vel.copy()

        # 目标初始位置
        if self.moving_goal:
            self.goal_pos = np.array(
                [
                    self.np_random.uniform(8.0, 12.0),
                    self.np_random.uniform(8.0, 12.0),
                    self.np_random.uniform(-math.pi, math.pi),
                ],
                dtype=float,
            )
            self.goal_vel = np.zeros(3, dtype=float)
            self.goal_live = self.goal_pos[:2].copy()
            self.goal_control = np.zeros(2, dtype=float)
            self.goal_ctrl_step = 0
        else:
            self.goal_live = np.array(
                [
                    self.np_random.uniform(8.0, 12.0),
                    self.np_random.uniform(8.0, 12.0),
                ],
                dtype=float,
            )
            self.goal_pos = np.array([self.goal_live[0], self.goal_live[1], 0.0], dtype=float)
            self.goal_vel = np.zeros(3, dtype=float)
            self.goal_control = np.zeros(2, dtype=float)
            self.goal_ctrl_step = 0

        # ---- init trajectory buffers ----
        self._traj_pts.clear()
        self._traj_s.clear()
        self._cursor_idx = 0
        self._stall_count = 0
        self._tan_bx, self._tan_by = 1.0, 0.0

        # prefill some points so delay gating has something to use
        prefill = max(1, self.goal_delay_steps + 1)
        for _ in range(prefill):
            self._traj_append(self.goal_live)

        # select initial reference goal + tangent
        _ref_dist_raw, _forced, cursor_s = self._select_goal_by_path()
        self._prev_cursor_s = float(cursor_s)

        xb, yb, dist = goal_in_body_frame(self.state_pos, self.goal)

        # TD 初始化
        self.td_dist.reset(dist)
        dist_td = float(dist)

        # overshoot segment init
        self._episode_id += 1
        self._trace = []
        self._overshoot = None
        self._saved_trace = False
        self._seg_min_dist_td = float(dist_td)
        self._seg_cursor0 = int(self._cursor_idx)

        # 初始帧 trace
        self._trace_append({
            "t": float(self.time),
            "step": int(self.steps),
            "x": float(self.state_pos[0]),
            "y": float(self.state_pos[1]),
            "theta": float(self.state_pos[2]),
            "u": float(self.state_vel[0]),
            "v": float(self.state_vel[1]),
            "r": float(self.state_vel[2]),
            "gx": float(self.goal[0]),
            "gy": float(self.goal[1]),
            "xb": float(xb),
            "yb": float(yb),
            "dist": float(dist),
            "dist_td": float(dist_td),
            "dist_dot_td": 0.0,
            "min_dist_td": float(self._seg_min_dist_td),
            "heading_err": float(math.atan2(yb, xb)),
            "Xprop": 0.0,
            "deltar": 0.0,
            "r_progress": 0.0,
            "r_heading": 0.0,
            "r_dist": 0.0,
            "goal_bonus": 0.0,
            "hold_bonus": 0.0,
            "energy_cost": 0.0,
            "smooth_cost_ctrl": 0.0,
            "smooth_cost_vel": 0.0,
            "speed_cost_near": 0.0,
            "ring_cost": 0.0,
            "reward": 0.0,
            "overshoot": 0.0,
            "delta_s": 0.0,
            "cursor_idx": int(self._cursor_idx),
            "tan_bx": float(self._tan_bx),
            "tan_by": float(self._tan_by),
        })

        # vector(15)：dist 用 dist_td；phase_cos/sin 用 tangent(body)
        phase_cos = float(self._tan_bx)
        phase_sin = float(self._tan_by)
        t_norm = 0.0

        obs = np.array(
            [
                xb, yb, float(dist_td),
                math.cos(self.state_pos[2]),
                math.sin(self.state_pos[2]),
                self.state_vel[0],
                self.state_vel[1],
                self.state_vel[2],
                self.state_pos[0],
                self.state_pos[1],
                self.goal[0],
                self.goal[1],
                phase_cos,
                phase_sin,
                t_norm,
            ],
            dtype=np.float32,
        )

        return dict(
            vector=obs,
            reward=np.float32(0.0),
            is_first=True,
            is_last=False,
            is_terminal=False,

            **{
                "log/dist": np.float32(dist),
                "log/dist_td": np.float32(dist_td),
                "log/dist_dot_td": np.float32(0.0),
                "log/min_dist_td": np.float32(self._seg_min_dist_td),
                "log/heading_err": np.float32(math.atan2(yb, xb)),

                "log/Xprop": np.float32(0.0),
                "log/deltar": np.float32(0.0),
                "log/u": np.float32(self.state_vel[0]),
                "log/v": np.float32(self.state_vel[1]),
                "log/r": np.float32(self.state_vel[2]),

                "log/r_progress": np.float32(0.0),
                "log/r_heading": np.float32(0.0),
                "log/r_dist": np.float32(0.0),
                "log/goal_bonus": np.float32(0.0),
                "log/hold_bonus": np.float32(0.0),
                "log/energy_cost": np.float32(0.0),
                "log/smooth_cost_ctrl": np.float32(0.0),
                "log/smooth_cost_vel": np.float32(0.0),
                "log/speed_cost_near": np.float32(0.0),
                "log/ring_cost": np.float32(0.0),

                "log/overshoot": np.float32(0.0),
                "log/overshoot_x": np.float32(np.nan),
                "log/overshoot_y": np.float32(np.nan),
                "log/overshoot_gx": np.float32(np.nan),
                "log/overshoot_gy": np.float32(np.nan),
            }
        )
