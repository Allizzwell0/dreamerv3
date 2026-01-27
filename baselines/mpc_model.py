import numpy as np

def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

class SimpleAUVModel:
    """
    线性/仿射近似模型（baseline 用，参数靠手调/粗辨识）
    state: [xb, yb, epsi, u, r]
    action: [thrust_norm, delta_norm]  # 均为 [-1,1]
    """
    def __init__(self, dt, cu=0.4, ku=1.0, cr=0.8, kr=1.2, rudder_max=0.6):
        self.dt = dt
        self.cu, self.ku = cu, ku
        self.cr, self.kr = cr, kr
        self.rudder_max = rudder_max

    def step(self, s, a, psi_ref):
        dt = self.dt
        xb, yb, epsi, u, r = s
        thrust, delta = a
        delta = np.clip(delta, -1, 1) * self.rudder_max  # 映射到“物理舵角”量级（近似）

        # surge / yaw-rate
        u2 = u + dt * (-self.cu * u + self.ku * thrust)
        r2 = r + dt * (-self.cr * r + self.kr * delta)

        # 误差运动学（很粗的近似：用 u + epsi 推进误差）
        epsi2 = wrap_pi(epsi + dt * r2)  # epsi = psi - psi_ref，psi_ref 认为外环给定常值/慢变
        xb2 = xb - dt * u2 * np.cos(epsi2)   # 朝目标前向误差减少（符号按你定义可调整）
        yb2 = yb - dt * u2 * np.sin(epsi2)

        # 注意：如果你想更贴合，可直接用 obs 里的 xb,yb 更新方式做线性化
        return np.array([xb2, yb2, epsi2, u2, r2], dtype=float)
