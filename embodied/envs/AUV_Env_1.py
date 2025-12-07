import numpy as np
import math
import elements
import embodied

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

    # 限幅
    u = np.clip(u, -5.0, 5.0)
    v = np.clip(v, -5.0, 5.0)
    r = np.clip(r, -3.0, 3.0)
    deltar = np.clip(deltar, -0.6, 0.6)
    Xprop = np.clip(Xprop, -50.0, 50.0)

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


def _wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


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
    # 世界 -> 船体
    xb = c * dx + s * dy
    yb = -s * dx + c * dy
    dist = float(math.hypot(xb, yb))
    return xb, yb, dist


# ----------------- 连续动作 + 移动目标的 AUV 环境 -----------------
class AUVEnv(embodied.Env):
    """
    AUV 3 自由度（x, y, ψ）+ 动力学模型环境（连续动作）

    action: shape=(2,), float32, 范围[-1, 1]
      action[0] -> 相对推力（-1~1），内部映射到 [-thrust_scale, +thrust_scale] N
      action[1] -> 相对舵角（-1~1），内部映射到 [-rudder_max, +rudder_max] rad

    支持静态目标 + 移动目标：
      - moving_goal=False: 目标是随机静止点
      - moving_goal=True :
          * goal_trajectory_type in {'circle','line','lemniscate','lissajous'}: 固定一种轨迹
          * goal_trajectory_type == 'random': 若干秒后随机切换到另一种轨迹类型（本代码中默认只用 circle + line）
    """

    def __init__(
        self,
        task=None,
        dt=0.05,
        max_steps=1000,
        success_radius=1.0,
        w_heading=0.1,
        thrust_scale=50.0,
        rudder_max=0.6,
        # === 移动目标相关参数 ===
        moving_goal=True,
        # 'circle' / 'line' / 'lemniscate' / 'lissajous' / 'random'
        goal_trajectory_type="random",
        goal_center=(10.0, 10.0),
        goal_radius=6.0,
        goal_speed=0.3,

        # ⭐ 限制目标自身的速度 / 角速度 / 分段时长
        max_goal_speed=0.8,               # 目标线速度上限 (m/s)
        max_goal_turn_rate=0.3,           # 目标角速度上限 (rad/s)
        seg_duration_range=(10.0, 30.0),  # 每段轨迹持续时间范围 (s)

        goal_custom_fn=None,              # 自定义：fn(t) -> (gx, gy)
        energy_thrust_coef=1e-4,
        energy_rudder_coef=1e-3,
        smooth_ctrl_coef=5e-3,
        smooth_vel_coef=5e-3,             # 速度惩罚
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
        self.goal_trajectory_type = str(goal_trajectory_type)
        self.goal_center = tuple(goal_center)
        self.goal_radius = float(goal_radius)
        self.goal_speed = float(goal_speed)

        # ⭐ 平滑 & 可跟踪约束
        self.max_goal_speed = float(max_goal_speed)
        self.max_goal_turn_rate = float(max_goal_turn_rate)
        self.seg_min, self.seg_max = map(float, seg_duration_range)

        self.goal_custom_fn = goal_custom_fn
        self.energy_thrust_coef = float(energy_thrust_coef)
        self.energy_rudder_coef = float(energy_rudder_coef)
        self.smooth_ctrl_coef = float(smooth_ctrl_coef)
        self.smooth_vel_coef = float(smooth_vel_coef)

        self.prev_control = np.zeros(2, dtype=float)
        self.prev_vel = np.zeros(3, dtype=float)

        self.steps = 0
        self.done = False
        self.np_random = np.random.RandomState(0)

        self.state_pos = np.zeros(3, dtype=float)   # [x, y, theta]
        self.state_vel = np.zeros(3, dtype=float)   # [u, v, r]
        self.goal = np.zeros(2, dtype=float)

        # 时间，用于移动目标的“动力学/轨迹”
        self.time = 0.0

        # ====== 复杂目标轨迹：分段 + 多种类型 ======
        # 为了保证可跟踪，默认只用 circle + line
        self.traj_types = ["circle", "line"]

        # 当前 segment 的信息（仅在 goal_trajectory_type == 'random' 时使用）
        self.seg_start_t = 0.0
        self.seg_duration = 0.0
        self.current_traj_type = None
        self.seg_params = {}

    # === 采样一个新的 segment：随机轨迹类型 + 参数 ===
    def _sample_new_segment(self, t, start_pos=None):
        """
        仅在 goal_trajectory_type == 'random' 时使用。
        在时间 t 开启一个新的轨迹段：
          - 随机挑选 self.traj_types 中的一种
          - 以当前 goal 位置为起点，采样对应轨迹参数
          - 设定持续时间 seg_duration（秒）
        """
        self.seg_start_t = float(t)
        # 每段持续时间更长，且可配置
        self.seg_duration = float(self.np_random.uniform(self.seg_min, self.seg_max))

        # 起点：优先用当前目标位置，否则用 goal_center
        if start_pos is None:
            start_pos = np.array(self.goal_center, dtype=float)
        else:
            start_pos = np.asarray(start_pos, dtype=float)

        # 当前段轨迹类型
        self.current_traj_type = self.np_random.choice(self.traj_types)

        base_r = self.goal_radius
        # 目标线速度上限
        base_speed = min(self.goal_speed, self.max_goal_speed)

        params = {}

        if self.current_traj_type == "circle":
            # 圆轨迹：保证圆弧通过 start_pos
            radius = float(base_r * self.np_random.uniform(0.8, 1.2))
            # 取一个随机法向方向，作为圆心相对于 start_pos 的方向
            normal_angle = self.np_random.uniform(-math.pi, math.pi)
            normal = np.array([math.cos(normal_angle), math.sin(normal_angle)], dtype=float)
            center = start_pos + radius * normal
            # 令 ang0 对应 start_pos
            ang0 = math.atan2(start_pos[1] - center[1], start_pos[0] - center[0])
            direction = float(self.np_random.choice([-1.0, 1.0]))  # 顺/逆时针
            params.update(center=center, radius=radius, ang0=ang0, direction=direction)

        elif self.current_traj_type == "line":
            # 直线：从 start_pos 出发，朝随机方向匀速运动
            angle = self.np_random.uniform(-math.pi, math.pi)
            direction = np.array([math.cos(angle), math.sin(angle)], dtype=float)
            direction /= (np.linalg.norm(direction) + 1e-6)
            speed = base_speed * self.np_random.uniform(0.5, 1.0)
            params.update(start=start_pos, direction=direction, speed=speed)

        self.seg_params = params

    # === 目标轨迹 ===
    def _goal_traj(self, t):
        """根据时间 t 计算目标位置 (gx, gy)。"""

        # 若用户提供自定义轨迹，优先使用
        if self.goal_custom_fn is not None:
            gx, gy = self.goal_custom_fn(t)
            return np.array([gx, gy], dtype=float)

        cx, cy = self.goal_center

        # ===== 模式一：随机分段切换轨迹类型 =====
        if self.goal_trajectory_type == "random":
            # 如果还没初始化当前段，或当前段结束了，就采样一段新的
            if (
                self.current_traj_type is None
                or (t - self.seg_start_t) > self.seg_duration
            ):
                # 把当前目标位置作为新段起点，保证位置连续
                self._sample_new_segment(t, start_pos=self.goal)

            dt_seg = float(t - self.seg_start_t)

            # 线速度与角速度上限
            base_v = min(self.goal_speed, self.max_goal_speed)
            radius_ref = max(self.goal_radius, 1.0)
            w_nominal = base_v / radius_ref
            w = min(w_nominal, self.max_goal_turn_rate)  # 角速度上限

            if self.current_traj_type == "circle":
                center = self.seg_params["center"]
                radius = self.seg_params["radius"]
                ang0 = self.seg_params["ang0"]
                direction = self.seg_params["direction"]

                # 根据 dt_seg 增加角度，限制 w
                w_eff = min(base_v / max(radius, 1e-3), self.max_goal_turn_rate)
                ang = ang0 + direction * w_eff * dt_seg
                gx = center[0] + radius * math.cos(ang)
                gy = center[1] + radius * math.sin(ang)
                return np.array([gx, gy], dtype=float)

            elif self.current_traj_type == "line":
                start = self.seg_params["start"]
                direction = self.seg_params["direction"]
                speed = self.seg_params["speed"]  # 已经被 base_speed 控制
                gx, gy = start + direction * speed * dt_seg
                return np.array([gx, gy], dtype=float)

            # fallback：万一类型不认识
            return np.array([cx, cy], dtype=float)

        # ===== 模式二：老的固定轨迹模式（与之前兼容） =====
        if self.goal_trajectory_type == "circle":
            ang = self.goal_speed * t
            gx = cx + self.goal_radius * math.cos(ang)
            gy = cy + self.goal_radius * math.sin(ang)
            return np.array([gx, gy], dtype=float)

        elif self.goal_trajectory_type == "line":
            s = self.goal_speed * t
            L = 2.0 * self.goal_radius
            if L <= 0.0:
                gx = cx
            else:
                s_mod = s % (2.0 * L)
                if s_mod < L:
                    offset = -self.goal_radius + s_mod
                else:
                    offset = self.goal_radius - (s_mod - L)
                gx = cx + offset
            gy = cy
            return np.array([gx, gy], dtype=float)

        elif self.goal_trajectory_type == "lemniscate":
            ang = self.goal_speed * t
            a = self.goal_radius
            gx = cx + a * math.sin(ang)
            gy = cy + a * math.sin(ang) * math.cos(ang)
            return np.array([gx, gy], dtype=float)

        # 默认：静止在中心
        return np.array([cx, cy], dtype=float)

    # === Dreamer 接口定义 ===
    @property
    def obs_space(self):
        # 15 维： [xb, yb, dist, cosθ, sinθ, u, v, r, x, y, gx, gy, phase_cos, phase_sin, t_norm]
        return {
            "vector": elements.Space(np.float32, (15,)),
            "reward": elements.Space(np.float32),
            "is_first": elements.Space(bool),
            "is_last": elements.Space(bool),
            "is_terminal": elements.Space(bool),
        }

    @property
    def act_space(self):
        return {
            "reset": elements.Space(bool),
            "action": elements.Space(np.float32, (2,), -1.0, 1.0),
        }

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

    # === step ===
    def step(self, action):
        if action.get("reset", False) or self.done:
            return self._reset()

        # 时间推进
        self.time += self.dt

        # 移动目标：根据轨迹更新目标点
        if self.moving_goal:
            self.goal = self._goal_traj(self.time)

        # 动力学更新
        Xprop, deltar = self._parse_action(action)
        control = np.array([Xprop, deltar], dtype=float)
        self.state_vel = update_model_state_dyn(self.state_vel, control, self.dt)
        self.state_pos = update_model_state_kine(self.state_pos, self.state_vel, self.dt)

        # === 误差：在船体坐标系下表示目标位置 ===
        xb, yb, dist = goal_in_body_frame(self.state_pos, self.goal)

        # 轨迹相位 + 归一化时间
        phase = self.goal_speed * self.time
        phase_cos = math.cos(phase)
        phase_sin = math.sin(phase)
        t_norm = self.time / (self.max_steps * self.dt + 1e-6)

        # ========== Reward 计算开始 ==========
        eps = 1e-6

        # 1) 距离归一化到 [0, 1]，参考距离用 2 * goal_radius
        radius_ref = max(2.0 * self.goal_radius, 1e-6)
        dist_norm = np.clip(dist / radius_ref, 0.0, 1.0)

        # 2) 航向对齐程度：目标在船体 x 轴上的投影比例
        forward_cos = xb / (dist + eps)

        # 3) 距离越远时 heading 权重越大，越近时越小（0.3 ~ 1.0 之间平滑变化）
        heading_scale = 0.3 + 0.7 * dist_norm

        # 4) 跟踪奖励：距离 + 航向
        track_reward = -dist_norm + self.w_heading * heading_scale * forward_cos

        # ---------- 能量和速度的“可变权重” ----------
        # （1）先算基础惩罚
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
        base_smooth_cost_vel = self.smooth_vel_coef * (du**2 + dv**2 + dr**2)

        # （2）根据距离动态调整权重：
        #     远处（dist_norm≈1）：scale ≈ 1.0
        #     近处（dist_norm≈0）：energy ≈ 3.0，smooth ≈ 4.0
        energy_scale       = 1.0 + 2.0 * (1.0 - dist_norm)
        smooth_ctrl_scale  = 1.0 + 3.0 * (1.0 - dist_norm)
        smooth_vel_scale   = 1.0 + 3.0 * (1.0 - dist_norm)

        energy_cost      = energy_scale      * base_energy_cost
        smooth_cost_ctrl = smooth_ctrl_scale * base_smooth_cost_ctrl
        smooth_cost_vel  = smooth_vel_scale  * base_smooth_cost_vel

        # 7) 汇总 reward
        reward = (
            0.5                 # baseline 正奖励
            + track_reward
            - energy_cost
            - smooth_cost_ctrl
            - smooth_cost_vel
        )
        reward = float(np.clip(reward, -1.0, 1.0))
        # ========== Reward 计算结束 ==========

        # 轨迹跟踪任务：只按步数结束
        self.steps += 1
        if self.steps >= self.max_steps:
            self.done = True

        # 更新 prev_* 供下一步使用
        self.prev_control = control.copy()
        self.prev_vel = self.state_vel.copy()

        obs = np.array(
            [
                xb,
                yb,
                dist,
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
            reward=np.float32(reward),
            is_first=False,
            is_last=self.done,
            is_terminal=False,
        )


    def _reset(self):
        self.steps = 0
        self.done = False
        self.time = 0.0

        # 自身初始状态
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

        # 重置随机轨迹段信息（下次 _goal_traj 会自动 sample）
        self.seg_start_t = 0.0
        self.seg_duration = 0.0
        self.current_traj_type = None
        self.seg_params = {}

        # 目标初始位置
        if self.moving_goal:
            self.goal = self._goal_traj(self.time)
        else:
            self.goal = np.array(
                [
                    self.np_random.uniform(8.0, 12.0),
                    self.np_random.uniform(8.0, 12.0),
                ],
                dtype=float,
            )

        xb, yb, dist = goal_in_body_frame(self.state_pos, self.goal)

        phase = self.goal_speed * self.time
        phase_cos = math.cos(phase)
        phase_sin = math.sin(phase)
        t_norm = 0.0

        obs = np.array(
            [
                xb,
                yb,
                dist,
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
        )
