import numpy as np
import cvxpy as cp

def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

class MPCController:
    def __init__(self, model, H=20,
                 w_y=5.0, w_psi=2.0, w_u=1.0, w_du=0.05, w_dd=0.1,
                 thrust_lim=1.0, delta_lim=1.0, dthrust_lim=0.2, ddelta_lim=0.2):
        self.m = model
        self.H = H
        self.w_y, self.w_psi, self.w_u = w_y, w_psi, w_u
        self.w_du, self.w_dd = w_du, w_dd
        self.thrust_lim, self.delta_lim = thrust_lim, delta_lim
        self.dthrust_lim, self.ddelta_lim = dthrust_lim, ddelta_lim

        # cvxpy variables
        self.X = cp.Variable((5, H+1))
        self.U = cp.Variable((2, H))
        self.X0 = cp.Parameter(5)
        self.PSIREF = cp.Parameter()     # scalar
        self.UREF = cp.Parameter()       # scalar
        self.U_PREV = cp.Parameter(2)    # prev action (rate constraints)

        self.prob = self._build_problem()

    def _build_problem(self):
        cost = 0
        cons = [self.X[:,0] == self.X0]
        for k in range(self.H):
            xb, yb, epsi, u, r = self.X[:,k]
            thrust, delta = self.U[:,k]

            # stage cost (focus y error + heading error + speed tracking)
            cost += self.w_y * cp.square(yb)
            cost += self.w_psi * cp.square(epsi)
            cost += self.w_u * cp.square(u - self.UREF)

            # smoothness
            if k == 0:
                du = self.U[:,k] - self.U_PREV
            else:
                du = self.U[:,k] - self.U[:,k-1]
            cost += self.w_du * cp.square(du[0]) + self.w_dd * cp.square(du[1])

            # action bounds
            cons += [
                cp.abs(thrust) <= self.thrust_lim,
                cp.abs(delta)  <= self.delta_lim,
            ]
            # rate bounds
            cons += [
                cp.abs(du[0]) <= self.dthrust_lim,
                cp.abs(du[1]) <= self.ddelta_lim,
            ]

            # dynamics (use python function -> need linear/affine form for strict QP;
            # baseline里你可以直接把 step 展开成仿射近似，或先用小角度线性化。
            # 这里给“框架”，实际你需要把 model.step() 线性化写成 A,B,c。
            raise NotImplementedError("Fill in linearized dynamics: X_{k+1} = A X_k + B U_k + c")

        prob = cp.Problem(cp.Minimize(cost), cons)
        return prob

    def act(self, obs_vec, psi_ref, u_ref, prev_action):
        xb, yb = float(obs_vec[0]), float(obs_vec[1])
        theta = np.arctan2(float(obs_vec[4]), float(obs_vec[3]))
        u = float(obs_vec[5]); r = float(obs_vec[7])

        epsi = wrap_pi(theta - psi_ref)
        x0 = np.array([xb, yb, epsi, u, r], dtype=float)

        # set params
        self.X0.value = x0
        self.PSIREF.value = float(psi_ref)
        self.UREF.value = float(u_ref)
        self.U_PREV.value = np.array(prev_action, dtype=float)

        # solve
        self.prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        u0 = np.array(self.U.value[:,0]).reshape(-1)
        return np.clip(u0, -1.0, 1.0)
