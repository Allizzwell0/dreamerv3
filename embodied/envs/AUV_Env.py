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
    Yuv = -2.86e+1
    Yuudr = 9.64e0
    Nuudr = -6.15e0

    Yr = 1.93e0
    Nuv = -2.4e1

    m = 30.51e0
    Iz = 3.45e0
    xg = 0
    yg = 0

    Xprop, deltar = input
    u, v, r = state

    # 限幅
    u = np.clip(u, -5, 5)
    v = np.clip(v, -5, 5)
    r = np.clip(r, -3, 3)
    deltar = np.clip(deltar, -0.6, 0.6)
    Xprop = np.clip(Xprop, -50, 50)

    M = np.array([
        [m - Xu, 0, -m * yg],
        [0, m - Yv, m * xg - Yr],
        [-m * yg, m * xg - Nv, Iz - Nr],
    ])

    tau = np.array([
        [Xuu * abs(u) * u + Xvr * v * r + Xrr * r * r + Xprop + m * v * r + m * xg * r * r],
        [Yvv * abs(v) * v + Yrr * abs(r) * r + Yur * u * r + Yuv * u * v + Yuudr * u * u * deltar - m * (u * r - yg * r * r)],
        [Nvv * abs(v) * v + Nrr * abs(r) * r + Nur * u * r + Nuv * u * v + Nuudr * u * u * deltar - m * (xg * u * r + yg * v * r)],
    ])

    accel = np.linalg.inv(M) @ tau
    accel = np.clip(accel, -100, 100)

    new_state = np.array([u, v, r]) + accel.flatten() * dt
    new_state = np.clip(new_state, -10, 10)
    return new_state


def update_model_state_kine(state, input, dt):
    u, v, r = input
    x, y, theta = state

    x += (np.cos(theta) * u - np.sin(theta) * v) * dt
    y += (np.sin(theta) * u + np.cos(theta) * v) * dt
    theta += r * dt
    theta = (theta + np.pi) % (2 * np.pi) - np.pi
    return np.array([x, y, theta])


def _wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ----------------- 连续动作 + 移动目标的 AUV 环境 -----------------
class AUVEnv(embodied.Env):
    """
    AUV 3自由度（x, y, ψ）+ 动力学模型环境（连续动作）

    action: shape=(2,), float32, 范围[-1, 1]
      action[0] -> 相对推力（-1~1），内部映射到 [-thrust_scale, +thrust_scale] N
      action[1] -> 相对舵角（-1~1），内部映射到 [-rudder_max, +rudder_max] rad

    支持静态目标 + 移动目标：
      - moving_goal=False: 目标是随机静止点（你原来的设定）
      - moving_goal=True : 目标按指定轨迹移动
    """

    def __init__(
        self,
        task=None,
        dt=0.05,
        max_steps=500,
        success_radius=1.0,
        w_heading=0.2,
        thrust_scale=50.0,
        rudder_max=0.6,
        # === 新增：移动目标相关参数 ===
        moving_goal=True,
        goal_trajectory_type="circle",   # 'circle' / 'line' / 'lemniscate'
        goal_center=(10.0, 10.0),
        goal_radius=6.0,
        goal_speed=0.3,
        goal_custom_fn=None,             # 自定义：fn(t) -> (gx, gy)
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
        # self.goal_trajectory_type = np.random.choice(["circle", "line", "lemniscate"])
        self.goal_trajectory_type = "line"
        self.goal_center = tuple(goal_center)
        self.goal_radius = float(goal_radius)
        self.goal_speed = float(goal_speed)
        self.goal_custom_fn = goal_custom_fn

        self.steps = 0
        self.done = False
        self.np_random = np.random.RandomState(0)

        self.state_pos = np.zeros(3)   # [x, y, theta]
        self.state_vel = np.zeros(3)   # [u, v, r]
        self.goal = np.zeros(2)

        # 时间，用于移动目标的“动力学/轨迹”
        self.time = 0.0

    # === 目标轨迹 ===
    def _goal_traj(self, t):
        """根据时间 t 计算目标位置 (gx, gy)。"""
        # 若用户提供自定义轨迹，优先使用
        if self.goal_custom_fn is not None:
            gx, gy = self.goal_custom_fn(t)
            return np.array([gx, gy], dtype=float)

        cx, cy = self.goal_center
        # 随机改变self.goal_trajectory_type的值以测试不同轨迹
        # self.goal_trajectory_type = self.np_random.choice(["circle", "line", "lemniscate"])

        if self.goal_trajectory_type == "circle":
            # 圆轨迹
            ang = self.goal_speed * t
            gx = cx + self.goal_radius * np.cos(ang)
            gy = cy + self.goal_radius * np.sin(ang)
            return np.array([gx, gy], dtype=float)

        elif self.goal_trajectory_type == "line":
            # 直线往返（沿 x 方向）
            s = self.goal_speed * t
            L = 2 * self.goal_radius
            if L <= 0:
                gx = cx
            else:
                s_mod = s % (2 * L)
                if s_mod < L:
                    offset = -self.goal_radius + s_mod
                else:
                    offset = self.goal_radius - (s_mod - L)
                gx = cx + offset
            gy = cy
            return np.array([gx, gy], dtype=float)

        elif self.goal_trajectory_type == "lemniscate":
            # 8 字形
            ang = self.goal_speed * t
            a = self.goal_radius
            gx = cx + a * np.sin(ang)
            gy = cy + a * np.sin(ang) * np.cos(ang)
            return np.array([gx, gy], dtype=float)

        # 默认：静止在中心
        return np.array([cx, cy], dtype=float)

    # === Dreamer 接口定义 ===
    @property
    def obs_space(self):
        # 原来 12 维： [xe, ye, dist, cosθ, sinθ, u, v, r, x, y, gx, gy]
        # 现在多加 3 维：phase_cos, phase_sin, t_norm -> 共 15 维
        return {
            'vector': elements.Space(np.float32, (15,)),
            'reward': elements.Space(np.float32),
            'is_first': elements.Space(bool),
            'is_last': elements.Space(bool),
            'is_terminal': elements.Space(bool),
        }

    @property
    def act_space(self):
        return {
            'reset': elements.Space(bool),
            'action': elements.Space(np.float32, (2,), -1.0, 1.0),
        }

    def _parse_action(self, action):
        a = action.get('action', action)
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
        if action.get('reset', False) or self.done:
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

        # 误差与朝向
        xe = self.state_pos[0] - self.goal[0]
        ye = self.state_pos[1] - self.goal[1]
        dist = float(np.hypot(xe, ye))
        bearing = math.atan2(self.goal[1] - self.state_pos[1],
                             self.goal[0] - self.state_pos[0])
        heading_err = _wrap_pi(bearing - self.state_pos[2])

        # 轨迹相位 + 归一化时间（给世界模型“节奏感”）
        phase = self.goal_speed * self.time
        phase_cos = math.cos(phase)
        phase_sin = math.sin(phase)
        t_norm = self.time / (self.max_steps * self.dt + 1e-6)

        # 奖励（仍然是拦截型）
        reward = -dist + self.w_heading * math.cos(heading_err)

        success = dist < self.success_radius
        if success:
            reward += 50.0
            self.done = True   # 如果想持续跟踪，可以把这一行注释掉

        self.steps += 1
        if self.steps >= self.max_steps:
            self.done = True

        obs = np.array([
            xe, ye, dist,
            math.cos(self.state_pos[2]), math.sin(self.state_pos[2]),
            self.state_vel[0], self.state_vel[1], self.state_vel[2],
            self.state_pos[0], self.state_pos[1],
            self.goal[0], self.goal[1],
            phase_cos, phase_sin,
            t_norm,
        ], dtype=np.float32)

        return dict(
            vector=obs,
            reward=np.float32(reward),
            is_first=False,
            is_last=self.done,
            is_terminal=self.done,  # 如需区分“成功/超时”，可改为 is_terminal=success
        )

    def _reset(self):
        self.steps = 0
        self.done = False
        self.time = 0.0

        # 自身初始状态
        self.state_pos = np.array([
            self.np_random.uniform(0, 5),
            self.np_random.uniform(0, 5),
            self.np_random.uniform(-math.pi, math.pi),
        ], dtype=float)
        self.state_vel = np.zeros(3, dtype=float)

        # 目标初始位置
        if self.moving_goal:
            self.goal = self._goal_traj(self.time)
        else:
            self.goal = np.array([
                self.np_random.uniform(8, 12),
                self.np_random.uniform(8, 12),
            ], dtype=float)

        xe = self.state_pos[0] - self.goal[0]
        ye = self.state_pos[1] - self.goal[1]
        dist = float(np.hypot(xe, ye))

        phase = self.goal_speed * self.time
        phase_cos = math.cos(phase)
        phase_sin = math.sin(phase)
        t_norm = 0.0

        obs = np.array([
            xe, ye, dist,
            math.cos(self.state_pos[2]), math.sin(self.state_pos[2]),
            self.state_vel[0], self.state_vel[1], self.state_vel[2],
            self.state_pos[0], self.state_pos[1],
            self.goal[0], self.goal[1],
            phase_cos, phase_sin,
            t_norm,
        ], dtype=np.float32)

        return dict(
            vector=obs,
            reward=np.float32(0.0),
            is_first=True,
            is_last=False,
            is_terminal=False,
        )
