#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A small linearizable error dynamics model for MPC.

State x = [xb, yb, epsi, u, r]
Control u_cmd = [thrust_norm, rudder_norm] in [-1, 1]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class AUVModelConfig:
    dt: float = 0.05

    # u_dot = -(1/tau_u) * u + k_u * thrust_norm
    tau_u: float = 0.8
    k_u: float = 3.0

    # r_dot = -(1/tau_r) * r + k_r * (rudder_max * rudder_norm)
    tau_r: float = 0.4
    k_r: float = 6.0

    rudder_max: float = 0.6


class SimpleAUVErrorModel:
    def __init__(self, cfg: AUVModelConfig):
        self.cfg = cfg

    @property
    def nx(self) -> int:
        return 5

    @property
    def nu(self) -> int:
        return 2

    def linearize(self, x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Discrete affine model:
          X_{k+1} = A X_k + B U_k + c

        Approx kinematics (small-angle):
          xb_{k+1} = xb_k - dt * u_k
          yb_{k+1} = yb_k + dt * (u * epsi)  (bilinear)

        Linearize u*epsi around (u0, epsi0):
          u*epsi ≈ u0*epsi + epsi0*u - u0*epsi0
        """
        cfg = self.cfg
        dt = float(cfg.dt)

        x0 = np.asarray(x0, dtype=float).reshape(-1)
        assert x0.size == 5, f"x0 must be (5,), got {x0.shape}"
        _, _, epsi0, u0, _ = x0.tolist()

        A = np.eye(5, dtype=float)
        B = np.zeros((5, 2), dtype=float)
        c = np.zeros((5,), dtype=float)

        # xb_{k+1} = xb_k - dt*u
        A[0, 3] = -dt

        # yb_{k+1} = yb_k + dt*(u*epsi)  (linearized)
        A[1, 2] = dt * u0
        A[1, 3] = dt * epsi0
        c[1] = -dt * u0 * epsi0

        # epsi_{k+1} = epsi_k + dt*r
        A[2, 4] = dt

        # u_{k+1} = (1 - dt/tau_u) u + dt*k_u*thrust
        A[3, 3] = 1.0 - dt / max(cfg.tau_u, 1e-6)
        B[3, 0] = dt * cfg.k_u

        # r_{k+1} = (1 - dt/tau_r) r + dt*k_r*(rudder_max*delta)
        A[4, 4] = 1.0 - dt / max(cfg.tau_r, 1e-6)
        B[4, 1] = dt * cfg.k_r * cfg.rudder_max

        return A, B, c
