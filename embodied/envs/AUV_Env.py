#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6DOF REMUS AUV 环境（DreamerV3 / embodied.Env 风格）

状态:
  位置姿态 η = [x, y, z, phi, theta, psi]   (NED/世界坐标系)
  速度     ν = [u, v, w, p, q, r]           (船体系)

动作（连续）: 3 维
  action[0] : 纵向推进器推力   T_prop   [-1, 1] -> [-T_max, T_max] N
  action[1] : 方向舵偏角       δ_r     [-1, 1] -> [-δ_r_max, δ_r_max] rad
  action[2] : 尾平面舵偏角     δ_s     [-1, 1] -> [-δ_s_max, δ_s_max] rad

观测:
  vector (float32, 1D, 30 维):
    [0:4]   目标在船体系误差 e_b = [x_b, y_b, z_b, dist]
    [4:10]  姿态编码 [cos φ, sin φ, cos θ, sin θ, cos ψ, sin ψ]
    [10:16] 速度 ν = [u, v, w, p, q, r]
    [16:22] 位置 [x, y, z] + 目标位置 [gx, gy, gz] （世界系）
    [22:28] 轨迹相位编码 phase_cos/sin_xyz （3 个相位的 cos/sin）
    [28:30] 归一化时间 [t_norm, seg_phase] （整个 episode + 当前段相位）

说明：
  - 水动力系数按 Prestero(2001) / Table 12 命名。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import elements
import embodied


# ============== 一些小工具 ==============

def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _rot_bn(phi: float, theta: float, psi: float) -> np.ndarray:
    """从 body 到 world 的旋转矩阵 R_bn (Z-Y-X: yaw-pitch-roll)."""
    cphi, sphi = math.cos(phi), math.sin(phi)
    cth,  sth  = math.cos(theta), math.sin(theta)
    cpsi, spsi = math.cos(psi), math.sin(psi)

    # R_bn = R_z(psi) * R_y(theta) * R_x(phi)
    R = np.array([
        [ cpsi*cth,  cpsi*sth*sphi - spsi*cphi,  cpsi*sth*cphi + spsi*sphi],
        [ spsi*cth,  spsi*sth*sphi + cpsi*cphi,  spsi*sth*cphi - cpsi*sphi],
        [   -sth,                cth*sphi,                cth*cphi      ],
    ], dtype=float)
    return R


def _euler_kinematics(phi: float, theta: float, p: float, q: float, r: float) -> np.ndarray:
    """[phi_dot, theta_dot, psi_dot] = T(φ,θ) * [p,q,r]."""
    cphi, sphi = math.cos(phi), math.sin(phi)
    cth,  sth  = math.cos(theta), math.sin(theta)

    if abs(cth) < 1e-6:
        cth = 1e-6

    tth  = sth / cth

    T = np.array([
        [1.0,  sphi * tth,           cphi * tth],
        [0.0,       cphi,               -sphi ],
        [0.0,  sphi / cth,           cphi / cth],
    ], dtype=float)
    return T @ np.array([p, q, r], dtype=float)


def goal_in_body_frame(eta_pos: np.ndarray, goal: np.ndarray) -> Tuple[float, float, float, float]:
    """
    把目标位置从世界系转换到船体系.
    eta_pos: [x, y, z, phi, theta, psi]
    goal:    [gx, gy, gz] (world)
    return: (x_b, y_b, z_b, dist)
    """
    x, y, z, phi, theta, psi = eta_pos
    gx, gy, gz = goal
    dx, dy, dz = gx - x, gy - y, gz - z
    R_bn = _rot_bn(phi, theta, psi)
    e_b = R_bn.T @ np.array([dx, dy, dz], dtype=float)
    dist = float(np.linalg.norm(e_b))
    return float(e_b[0]), float(e_b[1]), float(e_b[2]), dist


# ============== REMUS 6DOF 动力学参数 ==============

@dataclass
class RemusParams:
    """
    系数字段尽量与 Prestero(2001) 保持一致。
    """

    # 基本质量 / 重力 / 浮力
    W: float = 299.0
    B: float = 306.0
    m: float = 30.45

    x_g: float = 0.0
    y_g: float = 0.0
    z_g: float = 0.0196

    x_b: float = 0.0
    y_b: float = 0.0
    z_b: float = 0.0

    Ixx: float = 0.177
    Iyy: float = 3.45
    Izz: float = 3.45

    # --- Added mass (dot)  未知项先置 0 ---
    X_du: float = -0.93   # X_{dot u}  
    Y_dv: float = -35.5   # Y_{dot v}  
    Y_dr: float = 1.93   # Y_{dot r}  
    Z_dw: float = -35.5   # Z_{dot w}  
    Z_dq: float = -1.93 # Z_{dot q}
    K_dp: float = -0.0704   # K_{dot p}  
    M_dw: float = -1.93 # M_{dot w}
    M_dq: float = -4.88 # M_{dot q}
    N_dv: float = 1.93   # N_{dot v} 
    N_dr: float = -4.88   # N_{dot r}  

    # --- Surge 相关导数 ---
    Xu: float    = -0.94   # 线性项，如需可自己加到 X_rhs 里
    Xuabs: float = -3.90   # X_{u|u|}  (表中 Xuu)
    Xwq: float   = -35.5   # X_{wq} (近似来自 Xuvq)
    Xqq: float   = -1.93   # X_{qq}
    Xvr: float   = 35.5    # X_{vr}
    Xrr: float   = -1.93   # X_{rr}

    # --- Sway 相关导数 ---
    Yv: float    = -35.5
    Yvabs: float = -1310.0 # Y_{v|v|}
    Yr: float    = 1.93
    Yrabs: float = 0.632   # Y_{r|r|}
    Yur: float   = 5.22
    Ywp: float   = 35.5    # 由 Yvp 近似
    Ypq: float   = 1.93
    Yuv: float   = -28.6
    Yuudr: float = 9.64    # Y_{u^2 δ_r}

    # --- Heave 相关导数 ---
    Zw: float    = -1310.0
    Zwabs: float = -1310.0 # Z_{w|w|}
    Zqabs: float = -0.632  # Z_{q|q|}
    Zuq: float   = -5.22   # 来自 Zwq
    Zvp: float   = -35.5
    Zrp: float   = 1.93
    Zuw: float   = -28.6
    Zuuds: float = -9.64     # Z_{u^2 δ_s}  参看论文

    # --- Roll 相关导数 ---
    Kpp: float   = -0.13   # K_{p|p|}
    K_prop_gain: float = 0.0  # 螺旋桨扭矩对 K 的贡献 不考虑扭矩设为0

    # --- Pitch 相关导数 ---
    Mww: float   = 3.18
    Mqq: float   = -188.0
    Muq: float   = -2.00     
    Mvp: float   = -1.93
    Mrp: float   = 4.86
    Muw: float   = 24.0
    Muuds: float = -6.15     # M_{u^2 δ_s} 

    # --- Yaw 相关导数 ---
    Nvabs: float = -3.18   # N_{v|v|}
    Nr: float    = -4.88
    Nrr: float   = -94.0
    Nur: float   = -2.0
    Npq: float   = -4.86
    Nuv: float   = -24.0     
    Nwp: float   = -1.93     
    Nuudr: float = -6.15   # N_{u^2 δ_r}

    # --- 推进器 + 舵面控制 ---
    X_prop_gain: float = 1.0  # X_prop = gain * T_prop


# ============== 静水力 ==============

def hydrostatic_forces(p: RemusParams, phi: float, theta: float) -> np.ndarray:
    cphi, sphi = math.cos(phi), math.sin(phi)
    cth, sth = math.cos(theta), math.sin(theta)

    W, B = p.W, p.B

    X_HS = -(W - B) * sth
    Y_HS = (W - B) * cth * sphi
    Z_HS = (W - B) * cth * cphi

    K_HS = -((p.y_g * W - p.y_b * B) * cth * cphi +
             (p.z_g * W - p.z_b * B) * cth * sphi)
    M_HS = -((p.z_g * W - p.z_b * B) * sth +
             (p.x_g * W - p.x_b * B) * cth * cphi)
    N_HS = -((p.x_g * W - p.x_b * B) * cth * sphi +
             (p.y_g * W - p.y_b * B) * sth)

    return np.array([X_HS, Y_HS, Z_HS, K_HS, M_HS, N_HS], dtype=float)


# ============== 6DOF 动力学积分（使用上面公式） ==============

def remus_dynamics_step(
    params: RemusParams,
    eta: np.ndarray,
    nu: np.ndarray,
    control_forces: np.ndarray,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据给出的 6DOF REMUS 标量方程构造：
      A * nu_dot = b
    其中 b 包含水静力 + 阻尼 + 交叉项 + 控制 (control_forces = tau)。
    为了保持数值稳定，A 目前只用了对角项（附加质量未知为 0）。
    """

    p = params
    tau = control_forces
    m = p.m

    # 当前状态
    u, v, w, p_ang, q, r = nu
    phi, theta, psi = eta[3:6]

    # 静水力
    X_HS, Y_HS, Z_HS, K_HS, M_HS, N_HS = hydrostatic_forces(p, phi, theta)

    # --- 右端项 b（对应 6 个标量方程） ---

    # Surge
    X_rhs = (
        X_HS
        + p.Xuabs * u * abs(u)
        + (p.Xwq - m) * w * q
        + (p.Xqq + m * p.x_g) * q * q
        + (p.Xvr + m) * v * r
        + (p.Xrr + m * p.x_g) * r * r
        - m * p.y_g * p_ang * q
        - m * p.z_g * p_ang * r
        + tau[0]
    )

    # Sway
    Y_rhs = (
        Y_HS
        + p.Yvabs * v * abs(v)
        + p.Yrabs * r * abs(r)
        + (p.Yur - m) * u * r
        + (p.Ywp + m) * w * p_ang
        + (p.Ypq - m * p.x_g) * p_ang * q
        + p.Yuv * u * v
        + tau[1]
    )

    # Heave
    Z_rhs = (
        Z_HS
        + p.Zwabs * w * abs(w)
        + p.Zqabs * q * abs(q)
        + (p.Zuq + m) * u * q
        + (p.Zvp - m) * v * p_ang
        + (p.Zrp - m * p.x_g) * r * p_ang
        + p.Zuw * u * w
        + tau[2]
    )

    # Roll
    K_rhs = (
        K_HS
        + p.Kpp * p_ang * abs(p_ang)
        - (p.Izz - p.Iyy) * q * r
        + tau[3]
    )

    # Pitch
    M_rhs = (
        M_HS
        + p.Mww * w * abs(w)
        + p.Mqq * q * abs(q)
        + (p.Muq - m * p.x_g) * u * q
        + (p.Mvp + m * p.x_g) * v * p_ang
        + (p.Mrp - (p.Ixx - p.Izz)) * r * p_ang
        + p.Muw * u * w
        + tau[4]
    )

    # Yaw
    N_rhs = (
        N_HS
        + p.Nvabs * v * abs(v)
        + p.Nrr * r * abs(r)
        + (p.Nur - m * p.x_g) * u * r
        + (p.Npq - (p.Iyy - p.Ixx)) * p_ang * q
        + p.Nuv * u * v
        + tau[5]
    )

    b = np.array([X_rhs, Y_rhs, Z_rhs, K_rhs, M_rhs, N_rhs], dtype=float)

    # --- A 矩阵：加速度系数（目前只保留对角） ---
    A = np.zeros((6, 6), dtype=float)
    A[0, 0] = m - p.X_du
    A[1, 1] = m - p.Y_dv
    A[2, 2] = m - p.Z_dw
    A[3, 3] = p.Ixx - p.K_dp
    A[4, 4] = p.Iyy - p.M_dq
    A[5, 5] = p.Izz - p.N_dr

    # 解 nu_dot
    try:
        nu_dot = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # 如果 A 奇异，就退化成对角除法
        diag = np.diag(A)
        diag[diag == 0.0] = 1.0
        nu_dot = b / diag

    nu_dot = np.clip(nu_dot, -50.0, 50.0)
    nu_new = nu + nu_dot * dt

    # 运动学积分：eta_dot = [R_bn v; T(φ,θ) ω]
    R_bn = _rot_bn(phi, theta, psi)
    vel_world = R_bn @ np.array([nu_new[0], nu_new[1], nu_new[2]], dtype=float)
    euler_dot = _euler_kinematics(phi, theta, nu_new[3], nu_new[4], nu_new[5])

    eta_dot = np.concatenate([vel_world, euler_dot])
    eta_new = eta + eta_dot * dt

    eta_new[3] = _wrap_pi(eta_new[3])
    eta_new[4] = _wrap_pi(eta_new[4])
    eta_new[5] = _wrap_pi(eta_new[5])

    return eta_new, nu_new


class Trajectory3D:
    """
    提供多种 3D 轨迹，但与之前不同的是：
      - 一个 Trajectory3D 实例在构造时随机确定轨迹类型和参数；
      - 整个生命周期都沿着同一条平滑曲线运动（不再分段重采样）；
      - 用一个较小的 speed，使目标点移动更慢、更容易跟踪。
    """

    def __init__(
        self,
        center=(10.0, 10.0, -5.0),
        radius=6.0,
        speed=0.2,  # 比原来的 0.3 慢一些
        rng: np.random.RandomState | None = None,
    ):
        self.center = np.array(center, dtype=float)
        self.radius = float(radius)
        self.speed = float(speed)

        # 让 env 传进来的 rng 控制随机性；如果没有就自己创建一个
        self.rng = rng if rng is not None else np.random.RandomState()

        # 随机决定本 episode 使用哪一类轨迹
        self.current_type = self.rng.choice(
            ["line3d", "circle3d", "helix3d", "lemniscate3d"]
        )

        # 记录“起始时间”和“周期长度”，便于 env 里算 seg_phase
        self.seg_start_t = 0.0

        # 下面这个 seg_duration 只用来给 obs 里面的 seg_phase 归一化，
        # 不再触发重新采样，因此设成一个“轨迹周期”的量级即可。
        a = max(self.radius, 1e-3)
        w = self.speed / a      # 角速度
        T = 2.0 * math.pi / max(w, 1e-6)   # 绕一圈的时间
        self.seg_duration = T             # 一圈时间作为 “一段”的时间

        # 存放该条轨迹需要的参数
        self.seg_params: Dict[str, np.ndarray] = {}
        self._init_params()

    def _init_params(self):
        """根据 current_type 随机初始化一次轨迹参数。"""
        cx, cy, cz = self.center
        a = self.radius

        if self.current_type == "line3d":
            # 起点 & 方向，整条轨迹就是一条无限直线
            p0 = self.center + self.rng.uniform(-a, a, size=3)
            direction = self.rng.normal(size=3)
            direction[2] *= 0.3  # 垂直方向不要太猛
            direction /= (np.linalg.norm(direction) + 1e-6)
            self.seg_params = dict(p0=p0, direction=direction)

        elif self.current_type == "circle3d":
            # 水平圆轨迹，深度固定一个值
            depth = cz + self.rng.uniform(-a / 3, a / 3)
            phase0 = float(self.rng.uniform(0.0, 2.0 * math.pi))
            self.seg_params = dict(depth=depth, phase0=phase0)

        elif self.current_type == "helix3d":
            # 水平圆 + 缓慢上下波动（螺旋）
            depth0 = cz + self.rng.uniform(-a / 3, a / 3)
            depth_amp = a / 3
            phase0 = float(self.rng.uniform(0.0, 2.0 * math.pi))
            self.seg_params = dict(depth0=depth0, depth_amp=depth_amp, phase0=phase0)

        elif self.current_type == "lemniscate3d":
            # 3D "∞" 形轨迹，深度固定
            depth = cz + self.rng.uniform(-a / 3, a / 3)
            phase0 = float(self.rng.uniform(0.0, 2.0 * math.pi))
            self.seg_params = dict(depth=depth, phase0=phase0)

        else:
            # 理论上不会走到这里
            self.seg_params = {}

    def __call__(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        返回 (position, phase_vec)
          position: [gx, gy, gz]
          phase_vec: [phase_x, phase_y, phase_z] 供 obs 编码
        整个 episode 内不会重采样参数，因此轨迹是单条平滑曲线。
        """
        dt = t - self.seg_start_t
        a = self.radius
        w = self.speed / max(a, 1e-3)

        cx, cy, cz = self.center

        if self.current_type == "line3d":
            p0 = self.seg_params["p0"]
            direction = self.seg_params["direction"]
            pos = p0 + direction * self.speed * dt
            phase = np.array([w * dt, 0.0, 0.0], dtype=float)

        elif self.current_type == "circle3d":
            depth = self.seg_params["depth"]
            phase0 = self.seg_params["phase0"]
            ang = w * dt + phase0
            pos = np.array([
                cx + a * math.cos(ang),
                cy + a * math.sin(ang),
                depth,
            ], dtype=float)
            phase = np.array([ang, 0.0, 0.0], dtype=float)

        elif self.current_type == "helix3d":
            depth0 = self.seg_params["depth0"]
            depth_amp = self.seg_params["depth_amp"]
            phase0 = self.seg_params["phase0"]
            ang = w * dt + phase0
            pos = np.array([
                cx + a * math.cos(ang),
                cy + a * math.sin(ang),
                depth0 + depth_amp * math.sin(ang * 0.5),  # 垂向变化再慢一点
            ], dtype=float)
            phase = np.array([ang, ang * 0.5, 0.0], dtype=float)

        elif self.current_type == "lemniscate3d":
            depth = self.seg_params["depth"]
            phase0 = self.seg_params["phase0"]
            ang = w * dt + phase0
            s = math.sin(ang)
            c = math.cos(ang)
            pos = np.array([
                cx + a * s,
                cy + a * s * c,
                depth,
            ], dtype=float)
            phase = np.array([ang, 2.0 * ang, 0.0], dtype=float)

        else:
            pos = self.center.copy()
            phase = np.zeros(3, dtype=float)

        return pos, phase


# ============== 6DOF REMUS 环境 ==============

class AUVEnv(embodied.Env):
    def __init__(
        self,
        task=None,
        dt: float = 0.05,
        max_steps: int = 1000,
        success_radius: float = 1.0,
        w_align: float = 1.0,
        thrust_scale: float = 40.0,     # ⬅ 稍微减小推力尺度
        rudder_max: float = 0.35,       # ⬅ 舵角限制从 0.6rad 降到 ~20°
        stern_max: float = 0.35,
        # ==== 跟踪精度相关系数 ====
        dist_coef: float = 1.5,
        lat_coef: float = 1.0,
        heading_coef: float = 0.8,
        # ==== 新增：速度/角速度惩罚 ====
        vel_cost_coef: float = 0.02,    # 惩罚 |v|^2
        rate_cost_coef: float = 0.1,    # 惩罚 |ω|^2
        # ==== 能量 & 平滑：略小，但仍保留 ====
        energy_thrust_coef: float = 5e-6,
        energy_surface_coef: float = 5e-5,
        smooth_ctrl_coef: float = 2e-4,
        smooth_vel_coef: float = 5e-5,
        # ==== 新增：动作一阶滤波（执行器动态） ====
        actuator_tau: float = 0.2,      # 执行器时间常数（s），越大越慢
        **kwargs,
    ):
        del task, kwargs
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.success_radius = float(success_radius)
        self.w_align = float(w_align)

        self.thrust_scale = float(thrust_scale)
        self.rudder_max = float(rudder_max)
        self.stern_max = float(stern_max)

        self.params = RemusParams()

        # 跟踪精度相关权重
        self.dist_coef = float(dist_coef)
        self.lat_coef = float(lat_coef)
        self.heading_coef = float(heading_coef)

        # 速度/角速度惩罚
        self.vel_cost_coef = float(vel_cost_coef)
        self.rate_cost_coef = float(rate_cost_coef)

        # 能量 & 平滑代价权重
        self.energy_thrust_coef = float(energy_thrust_coef)
        self.energy_surface_coef = float(energy_surface_coef)
        self.smooth_ctrl_coef = float(smooth_ctrl_coef)
        self.smooth_vel_coef = float(smooth_vel_coef)

        self.steps = 0
        self.done = False
        self.time = 0.0

        self.np_random = np.random.RandomState(0)

        self.eta = np.zeros(6, dtype=float)
        self.nu = np.zeros(6, dtype=float)
        self.goal = np.zeros(3, dtype=float)

        self.prev_action = np.zeros(3, dtype=float)  # 3 维动作
        self.prev_nu = np.zeros(6, dtype=float)

        # 执行器一阶滤波：ctrl_filtered = α * ctrl_prev + (1-α) * ctrl_raw
        self.actuator_tau = float(actuator_tau)
        self.actuator_alpha = math.exp(-self.dt / max(self.actuator_tau, 1e-3))
        self.prev_ctrl = np.zeros(3, dtype=float)

        self.traj = Trajectory3D(rng=self.np_random)

    # ---- Dreamer/embodied 接口 ----
    @property
    def obs_space(self):
        return {
            "vector": elements.Space(np.float32, (30,)),
            "reward": elements.Space(np.float32),
            "is_first": elements.Space(bool),
            "is_last": elements.Space(bool),
            "is_terminal": elements.Space(bool),
        }

    @property
    def act_space(self):
        return {
            "reset": elements.Space(bool),
            "action": elements.Space(np.float32, (3,), -1.0, 1.0),
        }

    def _parse_action(self, action) -> Tuple[np.ndarray, np.ndarray]:
        a = action.get("action", action)
        a = np.array(a, dtype=np.float32).reshape(-1)
        if a.size == 1:
            a = np.array([a.item(), 0.0, 0.0], dtype=np.float32)
        assert a.size == 3, f"Expect 3D continuous action, got {a.size}"
        a = np.clip(a, -1.0, 1.0)

        T_prop = float(self.thrust_scale * a[0])
        delta_r = float(self.rudder_max * a[1])
        delta_s = float(self.stern_max * a[2])

        ctrl = np.array([T_prop, delta_r, delta_s], dtype=float)
        return a, ctrl

    def _control_to_forces(self, ctrl: np.ndarray) -> np.ndarray:
        """
        ctrl = [T_prop, δ_r, δ_s]
        输出 body frame 合力/力矩 tau = [X,Y,Z,K,M,N]
        """
        T_prop, delta_r, delta_s = ctrl
        p = self.params
        u = float(self.nu[0])

        # 适当限制 u^2，避免高速下舵力爆炸
        u_eff = max(min(u, 2.0), -2.0)
        u2 = u_eff * u_eff

        # 推进器推力
        X = p.X_prop_gain * T_prop

        # 舵面：缩减系数，降低闭环增益
        rudder_scale = 0.5
        stern_scale = 0.5

        Y = rudder_scale * p.Yuudr * u2 * delta_r
        Z = stern_scale * p.Zuuds * u2 * delta_s
        K = p.K_prop_gain * T_prop  # 仍可为 0
        M = stern_scale * p.Muuds * u2 * delta_s
        N = rudder_scale * p.Nuudr * u2 * delta_r

        return np.array([X, Y, Z, K, M, N], dtype=float)


    def step(self, action: Dict[str, np.ndarray]):
        if action.get("reset", False) or self.done:
            return self._reset()

        self.time += self.dt
        self.steps += 1

        # 目标 3D 轨迹
        self.goal, phase = self.traj(self.time)

        # 解析动作 -> 原始控制量
        a_norm, ctrl_raw = self._parse_action(action)

        # === 执行器一阶滤波，抑制 bang-bang ===
        alpha = self.actuator_alpha
        ctrl = alpha * self.prev_ctrl + (1.0 - alpha) * ctrl_raw
        self.prev_ctrl = ctrl.copy()

        # 控制 -> 力 / 力矩
        tau = self._control_to_forces(ctrl)

        # 6DOF 动力学
        self.eta, self.nu = remus_dynamics_step(
            self.params, self.eta, self.nu, tau, self.dt
        )

        # 目标在船体系下误差
        xb, yb, zb, dist = goal_in_body_frame(self.eta, self.goal)

        # 姿态编码
        phi, theta, psi = self.eta[3:6]
        cos_sin = np.array([
            math.cos(phi), math.sin(phi),
            math.cos(theta), math.sin(theta),
            math.cos(psi), math.sin(psi),
        ], dtype=float)

        # 轨迹 phase 编码 (3 个相位的 cos/sin)
        phase_cos = np.cos(phase)
        phase_sin = np.sin(phase)

        t_norm = self.time / (self.max_steps * self.dt + 1e-6)
        seg_phase = (self.time - self.traj.seg_start_t) / (self.traj.seg_duration + 1e-6)
        seg_phase = float(np.clip(seg_phase, 0.0, 1.0))

        # ---- 奖励 ----
        eps = 1e-6

        # 对齐（前向分量 / 距离）
        align = xb / (dist + eps)

        # 距离 & 侧向误差
        dist_cost = self.dist_coef * (dist ** 2)
        lat_cost = self.lat_coef * (yb ** 2 + zb ** 2)

        # 朝向/俯仰误差
        vx, vy, vz = xb, yb, zb
        norm_v = math.sqrt(vx * vx + vy * vy + vz * vz) + eps
        vx /= norm_v; vy /= norm_v; vz /= norm_v
        heading_err = math.atan2(abs(vy), max(vx, eps))
        pitch_err = math.atan2(-vz, math.sqrt(vx * vx + vy * vy))
        heading_cost = self.heading_coef * (heading_err ** 2 + pitch_err ** 2)

        # === 新增：速度 & 角速度惩罚 ===
        u, v, w, p_ang, q, r = self.nu
        vel_cost = self.vel_cost_coef * (u*u + v*v + w*w)
        rate_cost = self.rate_cost_coef * (p_ang*p_ang + q*q + r*r)

        track_cost = dist_cost + lat_cost + heading_cost + vel_cost + rate_cost
        track_reward = -track_cost + self.w_align * align

        # 能量耗散，用滤波后的 ctrl
        norm_T = ctrl[0] / (self.thrust_scale + eps)
        norm_surfaces = np.array([
            ctrl[1] / (self.rudder_max + eps),
            ctrl[2] / (self.stern_max + eps),
        ])
        energy_cost = (
            self.energy_thrust_coef * norm_T**2
            + self.energy_surface_coef * float(np.sum(norm_surfaces**2))
        )

        # 控制平滑性（对原始动作 a_norm）
        da = a_norm - self.prev_action
        smooth_cost_ctrl = self.smooth_ctrl_coef * float(np.sum(da**2))

        # 速度平滑性
        dnu = self.nu - self.prev_nu
        smooth_cost_vel = self.smooth_vel_coef * float(np.sum(dnu**2))

        reward = track_reward - energy_cost - smooth_cost_ctrl - smooth_cost_vel
        # 统一缩放 reward
        reward = reward * 0.01 

        # 再做一层 clip，保证数值稳定
        reward = float(np.clip(reward, -10.0, 10.0))


        if self.steps >= self.max_steps:
            self.done = True

        self.prev_action = a_norm.copy()
        self.prev_nu = self.nu.copy()

        obs_vec = np.concatenate([
            np.array([xb, yb, zb, dist], dtype=float),
            cos_sin,
            self.nu,
            self.eta[0:3],
            self.goal,
            phase_cos,
            phase_sin,
            np.array([t_norm, seg_phase], dtype=float),
        ]).astype(np.float32)

        assert obs_vec.shape == (30,)

        return dict(
            vector=obs_vec,
            reward=np.float32(reward),
            is_first=False,
            is_last=self.done,
            is_terminal=False,
        )

    def _reset(self):
        self.steps = 0
        self.done = False
        self.time = 0.0

        self.eta[:] = np.array([
            self.np_random.uniform(0.0, 5.0),
            self.np_random.uniform(0.0, 5.0),
            self.np_random.uniform(-5.0, -1.0),
            self.np_random.uniform(-0.1, 0.1),
            self.np_random.uniform(-0.1, 0.1),
            self.np_random.uniform(-math.pi, math.pi),
        ], dtype=float)

        self.nu[:] = 0.0
        self.prev_action[:] = 0.0
        self.prev_nu[:] = 0.0
        self.prev_ctrl[:] = 0.0   # ⬅ 新增

        self.traj = Trajectory3D(rng=self.np_random)
        self.goal, phase = self.traj(self.time)

        xb, yb, zb, dist = goal_in_body_frame(self.eta, self.goal)
        phi, theta, psi = self.eta[3:6]
        cos_sin = np.array([
            math.cos(phi), math.sin(phi),
            math.cos(theta), math.sin(theta),
            math.cos(psi), math.sin(psi),
        ], dtype=float)

        phase_cos = np.cos(phase)
        phase_sin = np.sin(phase)
        t_norm = 0.0
        seg_phase = 0.0

        obs_vec = np.concatenate([
            np.array([xb, yb, zb, dist], dtype=float),
            cos_sin,
            self.nu,
            self.eta[0:3],
            self.goal,
            phase_cos,
            phase_sin,
            np.array([t_norm, seg_phase], dtype=float),
        ]).astype(np.float32)

        return dict(
            vector=obs_vec,
            reward=np.float32(0.0),
            is_first=True,
            is_last=False,
            is_terminal=False,
        )
