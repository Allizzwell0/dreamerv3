#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QP-MPC controller using cvxpy + OSQP.

Minimize:
  xb, yb, epsi, (u-u_ref), r
  + control magnitude and control rate for smoothness
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

try:
    import cvxpy as cp
except Exception as e:
    cp = None
    _CVXPY_IMPORT_ERR = e

from mpc_model import SimpleAUVErrorModel


@dataclass
class MPCConfig:
    horizon: int = 20

    # weights
    w_x: float = 0.2
    w_y: float = 2.0
    w_psi: float = 1.5
    w_u: float = 0.8
    w_r: float = 0.2

    w_act: float = 1e-2
    w_dact: float = 5e-2

    # constraints
    u_limit: float = 5.0
    r_limit: float = 3.0
    dthrust_limit: float = 0.3
    ddelta_limit: float = 0.3

    solver: str = "OSQP"


class MPCController:
    def __init__(self, model: SimpleAUVErrorModel, cfg: MPCConfig):
        if cp is None:
            raise ImportError(
                "cvxpy is required for MPCController but import failed: "
                f"{_CVXPY_IMPORT_ERR}"
            )
        self.model = model
        self.cfg = cfg

        nx, nu = model.nx, model.nu
        H = int(cfg.horizon)

        # Variables
        self.X = cp.Variable((nx, H + 1))
        self.U = cp.Variable((nu, H))

        # Parameters
        self.X0 = cp.Parameter(nx)
        self.u_ref = cp.Parameter(nonneg=True)
        self.U_prev = cp.Parameter(nu)

        self.A = cp.Parameter((nx, nx))
        self.B = cp.Parameter((nx, nu))
        self.c = cp.Parameter(nx)

        cost = 0
        cons = [self.X[:, 0] == self.X0]

        for k in range(H):
            xb = self.X[0, k]
            yb = self.X[1, k]
            epsi = self.X[2, k]
            u = self.X[3, k]
            r = self.X[4, k]

            uk = self.U[:, k]

            # dynamics
            cons += [self.X[:, k + 1] == self.A @ self.X[:, k] + self.B @ uk + self.c]

            # input bounds
            cons += [uk <= 1.0, uk >= -1.0]

            # state bounds
            cons += [cp.abs(u) <= cfg.u_limit]
            cons += [cp.abs(r) <= cfg.r_limit]

            # rate limits
            du = uk - (self.U_prev if k == 0 else self.U[:, k - 1])
            cons += [cp.abs(du[0]) <= cfg.dthrust_limit]
            cons += [cp.abs(du[1]) <= cfg.ddelta_limit]

            # stage cost
            cost += cfg.w_x * cp.square(xb)
            cost += cfg.w_y * cp.square(yb)
            cost += cfg.w_psi * cp.square(epsi)
            cost += cfg.w_u * cp.square(u - self.u_ref)
            cost += cfg.w_r * cp.square(r)

            cost += cfg.w_act * cp.sum_squares(uk)
            cost += cfg.w_dact * cp.sum_squares(du)

        # terminal cost
        xbT, ybT, epsiT, uT, rT = self.X[:, H]
        cost += 0.5 * cfg.w_x * cp.square(xbT)
        cost += 1.0 * cfg.w_y * cp.square(ybT)
        cost += 1.0 * cfg.w_psi * cp.square(epsiT)
        cost += 0.5 * cfg.w_u * cp.square(uT - self.u_ref)
        cost += 0.2 * cfg.w_r * cp.square(rT)

        self.prob = cp.Problem(cp.Minimize(cost), cons)

    def act(
        self,
        *,
        x0: np.ndarray,
        u_ref: float,
        u_prev: np.ndarray,
        warm_start: bool = True,
        max_iter: int = 10_000,
    ) -> Tuple[np.ndarray, dict]:
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        u_prev = np.asarray(u_prev, dtype=float).reshape(-1)

        A, B, c = self.model.linearize(x0)

        self.X0.value = x0
        self.u_ref.value = float(max(0.0, u_ref))
        self.U_prev.value = u_prev
        self.A.value = A
        self.B.value = B
        self.c.value = c

        info = {"status": None, "obj": None}
        try:
            self.prob.solve(
                solver=getattr(cp, self.cfg.solver),
                warm_start=warm_start,
                max_iter=max_iter,
                verbose=False,
            )
            info["status"] = self.prob.status
            info["obj"] = float(self.prob.value) if self.prob.value is not None else None

            if self.prob.status not in ("optimal", "optimal_inaccurate"):
                return np.clip(u_prev, -1.0, 1.0), info

            u0 = np.asarray(self.U.value[:, 0], dtype=float).reshape(-1)
            return np.clip(u0, -1.0, 1.0), info

        except Exception as e:
            info["status"] = f"error:{e}"
            return np.clip(u_prev, -1.0, 1.0), info
