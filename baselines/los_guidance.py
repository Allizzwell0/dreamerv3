#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LOS (Line-Of-Sight) guidance for AUV target tracking (body-frame error inputs).

Outputs:
- psi_ref: desired absolute heading angle in world frame (rad)
- u_ref:   desired surge speed (m/s)

Constrained LOS:
  |d(psi_ref)/dt| <= yaw_rate_limit
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple


def wrap_pi(a: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class LOSConfig:
    # LOS geometry
    lookahead: float = 2.0          # Delta (m)
    # Reference rate limits
    yaw_rate_limit: float = 1.0     # rad/s
    # Speed reference shaping
    speed_limit: float = 2.0        # m/s
    speed_min: float = 0.2          # m/s
    k_speed: float = 0.6            # u_ref ≈ k_speed * dist
    slow_heading_gain: float = 0.6  # reduce u_ref when heading error is large (0~1)


class ConstrainedLOS:
    """
    Constrained LOS for a single moving target / reference point.

    Inputs:
      xb, yb: target position in body frame (m)
      theta:  current yaw angle in world frame (rad)
      dist:   distance to target (m) [optional]
      dt:     step time (s)

    Output:
      psi_ref (world frame), u_ref (m/s)
    """

    def __init__(self, cfg: LOSConfig):
        self.cfg = cfg
        self._psi_ref_prev: Optional[float] = None

    def reset(self) -> None:
        self._psi_ref_prev = None

    def compute(
        self,
        *,
        xb: float,
        yb: float,
        theta: float,
        dt: float,
        dist: Optional[float] = None,
    ) -> Tuple[float, float]:
        cfg = self.cfg
        if dist is None:
            dist = math.hypot(xb, yb)

        # 1) LOS aiming (body): atan2(yb, xb + Delta)
        delta = max(1e-6, float(cfg.lookahead))
        los_angle_body = math.atan2(yb, xb + delta)  # yb>0 means target on left (matches your AUVEnv)
        psi_des = wrap_pi(theta + los_angle_body)

        # 2) rate-limit psi_ref
        if self._psi_ref_prev is None:
            psi_ref = psi_des
        else:
            dpsi = wrap_pi(psi_des - self._psi_ref_prev)
            max_step = abs(cfg.yaw_rate_limit) * max(dt, 1e-6)
            dpsi = max(-max_step, min(max_step, dpsi))
            psi_ref = wrap_pi(self._psi_ref_prev + dpsi)

        self._psi_ref_prev = psi_ref

        # 3) speed reference
        u_ref = float(cfg.k_speed) * float(dist)
        u_ref = max(cfg.speed_min, min(cfg.speed_limit, u_ref))

        # slowdown if heading error is large
        heading_err = wrap_pi(psi_ref - theta)
        slowdown = 1.0 - cfg.slow_heading_gain * min(1.0, abs(heading_err) / (math.pi / 2))
        u_ref = max(cfg.speed_min, min(cfg.speed_limit, u_ref * slowdown))

        return psi_ref, u_ref
