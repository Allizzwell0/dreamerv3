import numpy as np

def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

class ConstrainedLOS:
    def __init__(self, dt, lookahead=3.0, yawrate_max=0.4,
                 u_min=0.2, u_max=2.0, k_speed=0.8):
        self.dt = dt
        self.Delta = lookahead
        self.yawrate_max = yawrate_max
        self.u_min, self.u_max = u_min, u_max
        self.k_speed = k_speed

    def __call__(self, obs_vec):
        # 你 vector 约定：xb, yb, dist_td, cos(theta), sin(theta), u, v, r, x, y, gx, gy, ...
        xb, yb = float(obs_vec[0]), float(obs_vec[1])
        dist_td = float(obs_vec[2])
        theta = np.arctan2(float(obs_vec[4]), float(obs_vec[3]))

        # LOS：给“航向增量”而不是绝对航向
        dpsi = np.arctan2(yb, max(self.Delta, 1e-6))
        # 航向变化率约束
        dpsi = np.clip(wrap_pi(dpsi), -self.yawrate_max*self.dt, self.yawrate_max*self.dt)
        psi_ref = wrap_pi(theta + dpsi)

        # 速度参考：远快近慢
        u_ref = self.u_min + (self.u_max - self.u_min) * np.tanh(self.k_speed * dist_td)
        return psi_ref, u_ref
