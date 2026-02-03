# SAC_base/auv_gym.py
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
sys.path.insert(0, str(ROOT))

# Try the "dreamerv3" style path first, fallback to local file if running standalone.
try:
    from embodied.envs.AUV_Env import AUVEnv  # type: ignore
except Exception:  # pragma: no cover
    from AUV_Env import AUVEnv  # type: ignore


def _tanh_clip(x: float) -> float:
    return float(np.tanh(x))


class AUVGym(gym.Env):
    """Gymnasium wrapper for embodied.Env-based AUVEnv.

    Observation: vector (15,)
    Action: Box([-1, 1], shape=(2,))

    Optional LOS action mixing (wrapper only; does NOT change AUV_Env.py).
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        dt: float = 0.05,
        max_steps: int = 800,
        seed: int = 0,
        **env_kwargs: Any,
    ):
        super().__init__()

        # -------- pop LOS kwargs (so AUVEnv won't see them) --------
        self.los_enable = bool(env_kwargs.pop("los_enable", False))
        # "none" | "replace" | "rudder_only" | "residual" | "blend"
        self.los_mode = str(env_kwargs.pop("los_mode", "rudder_only"))

        # lookahead LOS parameters (use body-frame errors xb/yb)
        self.los_lookahead = float(env_kwargs.pop("los_lookahead", 2.0))
        self.los_k_rudder = float(env_kwargs.pop("los_k_rudder", 2.0))
        # yaw sign convention differs across models; keep this knob
        self.los_rudder_sign = float(env_kwargs.pop("los_rudder_sign", -1.0))
        self.los_rudder_smooth = float(env_kwargs.pop("los_rudder_smooth", 0.3))  # 0..1

        # thrust heuristic (used for replace/residual/blend base)
        self.los_k_thrust = float(env_kwargs.pop("los_k_thrust", 0.15))
        self.los_thrust_min = float(env_kwargs.pop("los_thrust_min", 0.0))   # default no reverse
        self.los_thrust_max = float(env_kwargs.pop("los_thrust_max", 1.0))

        # mixing
        self.los_blend_w = float(env_kwargs.pop("los_blend_w", 0.5))          # for blend
        self.los_residual_scale = float(env_kwargs.pop("los_residual_scale", 0.3))  # for residual
        # ----------------------------------------------------------

        self._env = AUVEnv(dt=dt, max_steps=max_steps, seed=seed, **env_kwargs)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)

        self._zero_act = np.zeros((2,), dtype=np.float32)
        self._last_obs: Optional[np.ndarray] = None
        self._prev_rudder: float = 0.0

    # ---------------------------------------------------------------------
    # Runtime knobs (called by training callback via VecEnv.env_method)
    # ---------------------------------------------------------------------
    def set_los(
        self,
        *,
        enable: Optional[bool] = None,
        mode: Optional[str] = None,
        blend_w: Optional[float] = None,
        residual_scale: Optional[float] = None,
        k_rudder: Optional[float] = None,
        k_thrust: Optional[float] = None,
        rudder_sign: Optional[float] = None,
        lookahead: Optional[float] = None,
        rudder_smooth: Optional[float] = None,
    ) -> None:
        """Update LOS/mixing parameters at runtime. Does NOT touch AUVEnv."""
        if enable is not None:
            self.los_enable = bool(enable)
        if mode is not None:
            self.los_mode = str(mode)
        if blend_w is not None:
            self.los_blend_w = float(np.clip(blend_w, 0.0, 1.0))
        if residual_scale is not None:
            self.los_residual_scale = float(max(0.0, residual_scale))
        if k_rudder is not None:
            self.los_k_rudder = float(k_rudder)
        if k_thrust is not None:
            self.los_k_thrust = float(k_thrust)
        if rudder_sign is not None:
            rs = float(rudder_sign)
            self.los_rudder_sign = float(-1.0 if rs == 0.0 else math.copysign(1.0, rs))
        if lookahead is not None:
            self.los_lookahead = float(max(1e-6, lookahead))
        if rudder_smooth is not None:
            self.los_rudder_smooth = float(np.clip(rudder_smooth, 0.0, 1.0))

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _info_from_traj(self, traj: Dict[str, Any]) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        for k, v in traj.items():
            if isinstance(k, str) and k.startswith("log/"):
                try:
                    info[k] = float(np.asarray(v).item())
                except Exception:
                    info[k] = v
        info["is_first"] = bool(traj.get("is_first", False))
        info["is_last"] = bool(traj.get("is_last", False))
        info["is_terminal"] = bool(traj.get("is_terminal", False))
        return info

    def _compute_los_action(self, obs_vec: np.ndarray) -> np.ndarray:
        """Compute a simple LOS prior action from body-frame errors.

        obs_vec layout (assumed):
          [xb, yb, dist_td, cos(theta), sin(theta), u, v, r, x, y, gx, gy, ...]
        We only need xb, yb.
        """
        xb = float(obs_vec[0])
        yb = float(obs_vec[1])

        # Lookahead LOS: psi = atan2(yb, xb + L)
        psi = math.atan2(yb, xb + self.los_lookahead)

        # Rudder sign knob to match your yaw convention
        rudder_cmd = float(np.clip(self.los_rudder_sign * self.los_k_rudder * psi, -1.0, 1.0))
        rudder = (1.0 - self.los_rudder_smooth) * self._prev_rudder + self.los_rudder_smooth * rudder_cmd
        rudder = float(np.clip(rudder, -1.0, 1.0))
        self._prev_rudder = rudder

        # thrust heuristic: push forward more when target is in front (xb>0)
        thrust = _tanh_clip(self.los_k_thrust * xb)
        thrust = float(np.clip(thrust, self.los_thrust_min, self.los_thrust_max))

        return np.array([thrust, rudder], dtype=np.float32)

    def _mix_action(self, a_rl: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        a_rl = np.asarray(a_rl, dtype=np.float32).reshape(-1)
        if a_rl.size < 2:
            raise ValueError(f"Action must have 2 dims, got {a_rl.shape}")
        a_rl = np.clip(a_rl[:2], -1.0, 1.0)

        debug: Dict[str, float] = {}
        if (not self.los_enable) or (self._last_obs is None) or (self.los_mode.lower() in ("none", "off", "")):
            return a_rl, debug

        a_los = self._compute_los_action(self._last_obs)

        mode = self.los_mode.lower()
        if mode == "replace":
            a = a_los
        elif mode == "rudder_only":
            a = np.array([a_rl[0], a_los[1]], dtype=np.float32)
        elif mode == "blend":
            w = float(np.clip(self.los_blend_w, 0.0, 1.0))
            a = (1.0 - w) * a_los + w * a_rl
        elif mode == "residual":
            s = float(max(0.0, self.los_residual_scale))
            a = a_los + s * a_rl
        else:
            a = np.array([a_rl[0], a_los[1]], dtype=np.float32)

        a = np.clip(a, -1.0, 1.0)

        debug["log/los_a0"] = float(a_los[0])
        debug["log/los_a1"] = float(a_los[1])
        debug["log/rl_a0"] = float(a_rl[0])
        debug["log/rl_a1"] = float(a_rl[1])
        debug["log/mixed_a0"] = float(a[0])
        debug["log/mixed_a1"] = float(a[1])
        return a, debug

    # ---------------------------------------------------------------------
    # Gymnasium API
    # ---------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        _ = options
        super().reset(seed=seed)

        if seed is not None:
            # AUVEnv uses numpy global RNG internally; keep per-env determinism in each subprocess.
            np.random.seed(int(seed))

        self._prev_rudder = 0.0
        traj = self._env.step({"reset": True, "action": self._zero_act})
        if traj is None:
            raise RuntimeError("AUVEnv.step(reset=True) returned None. Check AUV_Env.py reset path.")
        obs = np.asarray(traj["vector"], dtype=np.float32)
        self._last_obs = obs.copy()
        info = self._info_from_traj(traj)
        return obs, info

    def step(self, action: np.ndarray):
        act, dbg = self._mix_action(action)

        traj = self._env.step({"reset": False, "action": act})
        if traj is None:
            raise RuntimeError("AUVEnv.step(reset=False) returned None.")
        obs = np.asarray(traj["vector"], dtype=np.float32)
        self._last_obs = obs.copy()

        reward = float(np.asarray(traj.get("reward", 0.0)).item())
        done = bool(traj.get("is_last", False))
        terminated = bool(traj.get("is_terminal", False))
        truncated = bool(done and not terminated)

        info = self._info_from_traj(traj)
        info.update(dbg)

        return obs, reward, terminated, truncated, info

    def close(self):
        if hasattr(self._env, "close"):
            self._env.close()
