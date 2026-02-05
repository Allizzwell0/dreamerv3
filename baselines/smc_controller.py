#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sliding Mode Control (SMC) for LOS-based AUV target tracking.

Design goal:
- Reuse the same LOS reference (psi_ref, u_ref) as the MPC baseline
- Output action = [thrust_norm, rudder_norm] in [-1, 1]

Convention inside the controller:
- Positive rudder_norm means "turn left" (increase theta).
  Use `rudder_sign_to_env` in the runner to map to the environment's sign if needed.

Notes:
- This is a practical SMC-like controller with a smooth switching term (tanh/sat)
  to reduce chattering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import math
import numpy as np


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def sat(x: float) -> float:
    return float(np.clip(x, -1.0, 1.0))


@dataclass
class SMCConfig:
    # speed loop (thrust) - PI is usually enough
    kp_u: float = 0.55
    ki_u: float = 0.08
    i_u_limit: float = 3.0

    # heading SMC surface: s = r + c1 * epsi
    c1: float = 2.0

    # equivalent / linear term
    k_eq: float = 0.35

    # switching term gain
    k_sw: float = 0.75

    # boundary layer thickness for sat/tanh
    phi: float = 0.25

    # output limits
    thrust_limit: float = 1.0
    rudder_limit: float = 1.0

    # optional additional damping on r
    k_r_damp: float = 0.00


class SMCController:
    def __init__(self, cfg: SMCConfig):
        self.cfg = cfg
        self._iu = 0.0

    def reset(self) -> None:
        self._iu = 0.0

    def act(
        self,
        *,
        theta: float,
        u: float,
        r: float,
        psi_ref: float,
        u_ref: float,
        dt: float,
    ) -> Tuple[np.ndarray, Dict]:
        cfg = self.cfg
        dt = float(max(dt, 1e-6))

        # --- speed PI ---
        eu = float(u_ref - u)
        iu_new = float(np.clip(self._iu + eu * dt, -cfg.i_u_limit, cfg.i_u_limit))
        thrust_unsat = cfg.kp_u * eu + cfg.ki_u * iu_new
        thrust = float(np.clip(thrust_unsat, -cfg.thrust_limit, cfg.thrust_limit))
        # conditional integration
        if abs(thrust_unsat) <= cfg.thrust_limit or (thrust == cfg.thrust_limit and eu < 0) or (thrust == -cfg.thrust_limit and eu > 0):
            self._iu = iu_new

        # --- heading SMC ---
        # epsi = theta - psi_ref; negative => need left turn
        epsi = float(wrap_pi(theta - psi_ref))

        # sliding surface
        s = float(r + cfg.c1 * epsi)

        # smooth switching term: sat(s/phi) with phi>0
        if cfg.phi <= 1e-6:
            sw = sat(s)
        else:
            sw = sat(s / cfg.phi)

        # rudder positive => left, so use u = -k*s - k_sw*sat(s/phi)
        rudder_unsat = (-cfg.k_eq * s) - (cfg.k_sw * sw) + (-cfg.k_r_damp * r)
        rudder = float(np.clip(rudder_unsat, -cfg.rudder_limit, cfg.rudder_limit))

        info = {
            "eu": eu,
            "epsi": epsi,
            "s": s,
            "sw": sw,
            "thrust_unsat": thrust_unsat,
            "rudder_unsat": rudder_unsat,
            "iu": self._iu,
        }
        return np.array([thrust, rudder], dtype=np.float32), info
