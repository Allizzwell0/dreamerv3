import numpy as np
import math
import elements
import embodied


def update_model_state_dyn(state, input, dt):
    # 动力学参数
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

    # 输入: 推力 Xprop, 舵角 deltar
    Xprop, deltar = input

    u, v, r = state

     # --- 限幅防爆 ---
    u = np.clip(u, -5, 5)
    v = np.clip(v, -5, 5)
    r = np.clip(r, -3, 3)
    deltar = np.clip(deltar, -0.6, 0.6)
    Xprop = np.clip(Xprop, -50, 50)


    M = np.array([
        [m - Xu, 0, -m * yg],
        [0, m - Yv, m * xg - Yr],
        [-m * yg, m * xg - Nv, Iz - Nr]
    ])

    tau = np.array([
        [Xuu * abs(u) * u + Xvr * v * r + Xrr * r * r + Xprop + m * v * r + m * xg * r * r],
        [Yvv * abs(v) * v + Yrr * abs(r) * r + Yur * u * r + Yuv * u * v + Yuudr * u * u * deltar - m * (u * r - yg * r * r)],
        [Nvv * abs(v) * v + Nrr * abs(r) * r + Nur * u * r + Nuv * u * v + Nuudr * u * u * deltar - m * (xg * u * r + yg * v * r)]
    ])

    accel = np.linalg.inv(M) @ tau
    accel = np.clip(accel, -100, 100)  # 限制加速度范围

    new_state = np.array([u, v, r]) + accel.flatten() * dt
    new_state = np.clip(new_state, -10, 10)  # 限制最终速度范围

    return new_state


def update_model_state_kine(state, input, dt):
    u, v, r = input
    x, y, theta = state

    x += (np.cos(theta) * u - np.sin(theta) * v) * dt
    y += (np.sin(theta) * u + np.cos(theta) * v) * dt
    theta += r * dt

    theta = (theta + np.pi) % (2 * np.pi) - np.pi

    return np.array([x, y, theta])


class AUVEnv(embodied.Env):
    """
    AUV 3自由度（x, y, ψ）+ 动力学模型环境
    Compatible with DreamerV3
    """

    def __init__(self, task=None, dt=0.05, max_steps=500, **kwargs):
        del task
        self.dt = dt
        self.max_steps = max_steps
        self.steps = 0
        self.done = False
        self.np_random = np.random.RandomState(0)

        # 状态变量
        self.state_pos = np.zeros(3)   # [x, y, theta]
        self.state_vel = np.zeros(3)   # [u, v, r]
        self.goal = np.zeros(2)

    # === Dreamer 接口定义 ===
    @property
    def obs_space(self):
        return {
            'vector': elements.Space(np.float32, (8,)),
            'reward': elements.Space(np.float32),
            'is_first': elements.Space(bool),
            'is_last': elements.Space(bool),
            'is_terminal': elements.Space(bool),
        }

    @property
    def act_space(self):
        # 两个动作：主推进推力 + 舵角
        return {
            'reset': elements.Space(bool),
            'action': elements.Space(np.float32, (2,), -1.0, 1.0),
        }

    # === step ===
    def step(self, action):
        if action.get('reset', False) or self.done:
            return self._reset()

        # 动作缩放
        Xprop = action['action'][0] * 50.0     # 推力范围 [-50, 50] N
        deltar = action['action'][1] * 0.5     # 舵角范围 [-0.5, 0.5] rad
        control = np.array([Xprop, deltar])

        # 更新动力学
        self.state_vel = update_model_state_dyn(self.state_vel, control, self.dt)
        self.state_pos = update_model_state_kine(self.state_pos, self.state_vel, self.dt)

        # 计算误差与奖励
        xe = self.state_pos[0] - self.goal[0]
        ye = self.state_pos[1] - self.goal[1]
        pose = np.sqrt(xe ** 2 + ye ** 2)

        reward = -pose
        if pose < 0.3:
            reward += 50
            self.done = True

        self.steps += 1
        if self.steps >= self.max_steps:
            self.done = True

        # 观测
        obs = np.array([
            xe, ye, pose,
            np.cos(self.state_pos[2]), np.sin(self.state_pos[2]),
            self.state_vel[0], self.state_vel[1], self.state_vel[2],
            self.state_pos[0], self.state_pos[1],
            self.goal[0], self.goal[1]
        ], dtype=np.float32)

        return dict(
            vector=obs,
            reward=np.float32(reward),
            is_first=False,
            is_last=self.done,
            is_terminal=self.done,
        )

    def _reset(self):
        """环境重置"""
        self.steps = 0
        self.done = False

        self.state_pos = np.array([
            self.np_random.uniform(0, 5),
            self.np_random.uniform(0, 5),
            self.np_random.uniform(-math.pi, math.pi)
        ])
        self.state_vel = np.zeros(3)
        self.goal = np.array([
            self.np_random.uniform(8, 12),
            self.np_random.uniform(8, 12)
        ])

        xe = self.state_pos[0] - self.goal[0]
        ye = self.state_pos[1] - self.goal[1]
        pose = np.sqrt(xe ** 2 + ye ** 2)

        obs = np.array([
            xe, ye, pose,
            np.cos(self.state_pos[2]), np.sin(self.state_pos[2]),
            self.state_vel[0], self.state_vel[1], self.state_vel[2],
            self.state_pos[0], self.state_pos[1],
            self.goal[0], self.goal[1]
        ], dtype=np.float32)

        return dict(
            vector=obs,
            reward=np.float32(0.0),
            is_first=True,
            is_last=False,
            is_terminal=False,
        )
