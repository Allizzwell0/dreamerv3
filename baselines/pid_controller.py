#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PID controllers for LOS-based AUV target tracking.

Design goal:
- Reuse the same LOS reference (psi_ref, u_ref) as the MPC baseline
- Output the same normalized action format as AUVEnv expects:
    action = [thrust_norm, rudder_norm] in [-1, 1]

Convention used in this controller:
- Positive rudder_norm means "turn left" (increase theta) in the *controller*.
  Use `rudder_sign_to_env` in the runner to convert to the environment's sign convention if needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import math
import numpy as np


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class PIDConfig:
    # speed loop (thrust)
    kp_u: float = 0.55
    ki_u: float = 0.08
    kd_u: float = 0.00

    # heading loop (rudder)
    kp_psi: float = 1.60
    ki_psi: float = 0.15
    kd_psi: float = 0.35

    # optional cross-track term directly on yb (body-frame lateral error)
    k_yb: float = 0.00

    # integrator limits (anti-windup)
    i_u_limit: float = 3.0
    i_psi_limit: float = 2.0

    # output limits (normalized)
    thrust_limit: float = 1.0
    rudder_limit: float = 1.0

    # deadband
    epsi_deadband: float = 0.0
    u_deadband: float = 0.0


class PIDController:
    def __init__(self, cfg: PIDConfig):
        self.cfg = cfg
        self._iu = 0.0
        self._ipsi = 0.0
        self._u_prev: Optional[float] = None
        self._epsi_prev: Optional[float] = None

    def reset(self) -> None:
        self._iu = 0.0
        self._ipsi = 0.0
        self._u_prev = None
        self._epsi_prev = None

    def act(
        self,
        *,
        xb: float,
        yb: float,
        theta: float,
        u: float,
        r: float,
        psi_ref: float,
        u_ref: float,
        dt: float,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Args are physical quantities (m, rad, m/s, rad/s).
        Returns:
          action: np.array([thrust_norm, rudder_norm]) in [-1,1]
          info: debug dict
        """
        cfg = self.cfg
        dt = float(max(dt, 1e-6))

        # --- speed error ---
        eu = float(u_ref - u)
        if abs(eu) < cfg.u_deadband:
            eu = 0.0

        if self._u_prev is None:
            du = 0.0
        else:
            du = (u - self._u_prev) / dt
        self._u_prev = float(u)

        # tentative integrator update (anti-windup done after saturation)
        iu_new = float(np.clip(self._iu + eu * dt, -cfg.i_u_limit, cfg.i_u_limit))

        thrust_unsat = cfg.kp_u * eu + cfg.ki_u * iu_new - cfg.kd_u * du
        thrust = float(np.clip(thrust_unsat, -cfg.thrust_limit, cfg.thrust_limit))

        # simple conditional integration: only accept integrator if not saturating against error direction
        if abs(thrust_unsat) <= cfg.thrust_limit or (thrust == cfg.thrust_limit and eu < 0) or (thrust == -cfg.thrust_limit and eu > 0):
            self._iu = iu_new

        # --- heading error ---
        # epsi = theta - psi_ref  (same as baseline); negative means we need to turn left
        epsi = float(wrap_pi(theta - psi_ref))
        if abs(epsi) < cfg.epsi_deadband:
            epsi = 0.0

        # Controller uses rudder_cmd positive = "turn left" => use -epsi
        epsi_ctrl = -epsi

        if self._epsi_prev is None:
            depsi = 0.0
        else:
            depsi = (epsi_ctrl - self._epsi_prev) / dt
        self._epsi_prev = float(epsi_ctrl)

        ipsi_new = float(np.clip(self._ipsi + epsi_ctrl * dt, -cfg.i_psi_limit, cfg.i_psi_limit))

        # add damping with measured yaw rate r (r>0 left turn), so -r damps
        rudder_unsat = (
            cfg.kp_psi * epsi_ctrl +
            cfg.ki_psi * ipsi_new +
            cfg.kd_psi * (-r) +
            cfg.k_yb * float(yb)
        )
        rudder = float(np.clip(rudder_unsat, -cfg.rudder_limit, cfg.rudder_limit))

        if abs(rudder_unsat) <= cfg.rudder_limit or (rudder == cfg.rudder_limit and epsi_ctrl < 0) or (rudder == -cfg.rudder_limit and epsi_ctrl > 0):
            self._ipsi = ipsi_new

        info = {
            "eu": eu,
            "epsi": epsi,
            "thrust_unsat": thrust_unsat,
            "rudder_unsat": rudder_unsat,
            "iu": self._iu,
            "ipsi": self._ipsi,
        }
        return np.array([thrust, rudder], dtype=np.float32), info
