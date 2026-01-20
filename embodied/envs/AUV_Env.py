#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUVEnv (Dreamer/embodied.Env)

新增能力（用于定位“过冲发生了什么”）：
1) 过冲检测（overshoot）：基于 dist_td 相对历史最小值回弹
2) 诊断信号输出到 obs 的 log/* 字段（不会进入 world model，因为 make_agent 里过滤了 log/ 前缀）
3) 可选：保存 episode 轨迹到 CSV（包含过冲时刻前后全程信息），方便你画图定位动作/速度/heading_err 等变化

用法（在 configs.yaml 的 env.auv 里加）例如：
  env:
    auv:
      debug_trace: True
      trace_dir: /home/mayue/logdir/auv_debug_traces
      overshoot_eps: 0.2
      overshoot_min_steps: 20
      save_trace_on_done: True
      save_trace_on_overshoot: True
      seed: 0
"""

import os
import csv
import math
import numpy as np
from pathlib import Path

import elements
import embodied
from collections import deque


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
        goal_speed=0.3,
        goal_delay_steps = 5,

        # AUV 自身最大速度 / 角速度
        max_auv_speed=5.0,
        max_auv_turn_rate=3.0,

        # 目标最大速度 / 角速度
        max_goal_speed=0.5,
        max_goal_turn_rate=0.3,

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

        self.max_auv_speed = float(max_auv_speed)
        self.max_auv_turn_rate = float(max_auv_turn_rate)

        self.max_goal_speed = float(max_goal_speed)
        self.max_goal_turn_rate = float(max_goal_turn_rate)

        self.goal_thrust_scale = float(goal_thrust_scale)
        self.goal_rudder_max = float(goal_rudder_max)
        self.goal_ctrl_interval = int(goal_ctrl_interval)
        self.goal_ctrl_smooth = float(goal_ctrl_smooth)
        self.goal_custom_fn = goal_custom_fn
        self.goal_delay_steps = int(goal_delay_steps)
        self._goal_hist = deque(maxlen=max(1, self.goal_delay_steps + 1))

        self.goal_live = np.zeros(2, dtype=float)   # 当前真实移动目标（用于生成轨迹）
        # self.goal 继续保留，但语义改成：给 agent 跟踪的“参考目标”（延迟后的点）


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
        self.prev_dist = None

        self.steps = 0
        self.done = False

        self.state_pos = np.zeros(3, dtype=float)   # [x, y, theta]
        self.state_vel = np.zeros(3, dtype=float)   # [u, v, r]

        self.goal = np.zeros(2, dtype=float)
        self.goal_pos = np.zeros(3, dtype=float)
        self.goal_vel = np.zeros(3, dtype=float)
        self.goal_control = np.zeros(2, dtype=float)
        self.goal_ctrl_step = 0

        self.time = 0.0

        # 距离 TD
        self.td_dist = TrackingDifferentiator(r=self.td_r, h=self.dt, N=self.td_N)

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
        self._min_dist_td = None     # float
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

    # === Dreamer 接口定义 ===
    @property
    def obs_space(self):
        scalar_f = elements.Space(np.float32, ())
        return {
            "vector": elements.Space(np.float32, (15,)),
            "reward": scalar_f,
            "is_first": elements.Space(bool, ()),
            "is_last": elements.Space(bool, ()),
            "is_terminal": elements.Space(bool, ()),

            # ---- per-step diagnostics (won't be used by world model) ----
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

            # ---- overshoot detector output ----
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
        # optional hook
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

            # CSV
            if self._trace:
                keys = list(self._trace[0].keys())
                with open(csv_path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=keys)
                    w.writeheader()
                    for r in self._trace:
                        w.writerow(r)

            # meta
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

    # === step ===
    def step(self, action):
        if action.get("reset", False) or self.done:
            return self._reset()

        self.time += self.dt

        # --------- 目标动力学推进（真实目标） ----------
        if self.moving_goal:
            self.goal_live = self._goal_traj(self.time)
        else:
            # 静态目标：goal_live 就等于当前 goal
            self.goal_live = self.goal.copy()

        # 写入历史并取 delay 步之前的参考点
        self._goal_hist.append(self.goal_live.copy())
        self.goal = self._goal_hist[0].copy()   # 这一步之后，reward/obs 用的都是“延迟目标”


        # --------- AUV 动力学推进 ----------
        Xprop, deltar = self._parse_action(action)
        control = np.array([Xprop, deltar], dtype=float)

        self.state_vel = update_model_state_dyn(self.state_vel, control, self.dt)

        self.state_vel[0] = np.clip(self.state_vel[0], -self.max_auv_speed, self.max_auv_speed)
        self.state_vel[1] = np.clip(self.state_vel[1], -self.max_auv_speed, self.max_auv_speed)
        self.state_vel[2] = np.clip(self.state_vel[2], -self.max_auv_turn_rate, self.max_auv_turn_rate)

        self.state_pos = update_model_state_kine(self.state_pos, self.state_vel, self.dt)

        # === 误差：在船体坐标系下表示目标位置 ===
        xb, yb, dist = goal_in_body_frame(self.state_pos, self.goal)

        # TD 平滑距离
        dist_td, dist_dot_td = self.td_dist.step(dist)

        time_ref = max(0.0, self.time - self.goal_delay_steps * self.dt)
        phase = self.goal_speed * time_ref
        phase_cos = math.cos(phase)
        phase_sin = math.sin(phase)

        t_norm = self.time / (self.max_steps * self.dt + 1e-6)

        # ========== Reward 计算开始 ==========
        eps = 1e-6

        if self.prev_dist is None:
            self.prev_dist = dist_td
        progress = self.prev_dist - dist_td
        self.prev_dist = dist_td

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

        dist_norm_clip = np.clip(dist_norm_near, 0.0, 1.0)
        k_progress = base_k_progress * (1.3 - 0.3 * dist_norm_clip)
        s = 0.03
        r_progress = k_progress * math.tanh(progress / s) * s

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

        # ===== overshoot detector =====
        if self._min_dist_td is None:
            self._min_dist_td = float(dist_td)
        else:
            self._min_dist_td = min(self._min_dist_td, float(dist_td))

        if (self._overshoot is None) and (self.steps >= self.overshoot_min_steps):
            cond = (dist_td > self._min_dist_td + self.overshoot_eps)
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
            "min_dist_td": float(self._min_dist_td if self._min_dist_td is not None else np.nan),
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

        obs = np.array(
            [
                xb, yb, dist,
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
                "log/min_dist_td": np.float32(self._min_dist_td if self._min_dist_td is not None else np.nan),
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
        # 如果上一局 trace 还没保存（比如你关了 done 保存但想保留），这里可按需保存
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
        self.prev_dist = None

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
            self.goal = self.goal_pos[:2].copy()
            self.goal_control = np.zeros(2, dtype=float)
            self.goal_ctrl_step = 0
        else:
            self.goal = np.array(
                [
                    self.np_random.uniform(8.0, 12.0),
                    self.np_random.uniform(8.0, 12.0),
                ],
                dtype=float,
            )
            self.goal_pos = np.array([self.goal[0], self.goal[1], 0.0], dtype=float)
            self.goal_vel = np.zeros(3, dtype=float)
            self.goal_control = np.zeros(2, dtype=float)
            self.goal_ctrl_step = 0
        # --- init goal history for delayed tracking ---
        self.goal_live = self.goal.copy()
        self._goal_hist.clear()
        for _ in range(self._goal_hist.maxlen):
            self._goal_hist.append(self.goal_live.copy())

        # 参考目标：队列最老的那个 = delay 步之前
        self.goal = self._goal_hist[0].copy()


        xb, yb, dist = goal_in_body_frame(self.state_pos, self.goal)

        # TD 初始化
        self.td_dist.reset(dist)
        self.prev_dist = float(dist)

        # ===== init overshoot/trace =====
        self._episode_id += 1
        self._trace = []
        self._overshoot = None
        self._min_dist_td = float(dist)
        self._saved_trace = False

        # 初始帧也记一条
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
            "dist_td": float(dist),
            "dist_dot_td": 0.0,
            "min_dist_td": float(dist),
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
        })

        phase = self.goal_speed * self.time
        phase_cos = math.cos(phase)
        phase_sin = math.sin(phase)
        t_norm = 0.0

        obs = np.array(
            [
                xb, yb, dist,
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

        # reset 时 log/* 也必须返回（否则 CheckSpaces 会报缺字段）
        return dict(
            vector=obs,
            reward=np.float32(0.0),
            is_first=True,
            is_last=False,
            is_terminal=False,

            **{
                "log/dist": np.float32(dist),
                "log/dist_td": np.float32(dist),
                "log/dist_dot_td": np.float32(0.0),
                "log/min_dist_td": np.float32(dist),
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
