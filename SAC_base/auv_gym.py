# baselines_sac/auv_gym.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# 让 `embodied.envs.AUV_Env` 可 import（与你的 eval_auv.py 类似）
THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
sys.path.insert(0, str(ROOT))

from embodied.envs.AUV_Env import AUVEnv  # noqa


class AUVGym(gym.Env):
    """
    Gymnasium wrapper for your embodied.Env-based AUVEnv.
    Observation: vector (15,)
    Action: Box([-1,1], shape=(2,))
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
        self._env = AUVEnv(dt=dt, max_steps=max_steps, seed=seed, **env_kwargs)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)

        self._zero_act = np.zeros((2,), dtype=np.float32)

    def _info_from_traj(self, traj: Dict[str, Any]) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        for k, v in traj.items():
            if isinstance(k, str) and k.startswith("log/"):
                # 转成 python float，便于 sb3 logger / json
                try:
                    info[k] = float(np.asarray(v).item())
                except Exception:
                    info[k] = v
        # 也可加你关心的非 log 字段
        info["is_first"] = bool(traj.get("is_first", False))
        info["is_last"] = bool(traj.get("is_last", False))
        info["is_terminal"] = bool(traj.get("is_terminal", False))
        return info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        _ = options
        if seed is not None:
            # 你的 env 里有 self.np_random
            if hasattr(self._env, "np_random"):
                self._env.np_random.seed(int(seed))
            np.random.seed(int(seed))

        traj = self._env.step({"reset": True, "action": self._zero_act})
        obs = np.asarray(traj["vector"], dtype=np.float32)
        info = self._info_from_traj(traj)
        return obs, info

    def step(self, action: np.ndarray):
        act = np.asarray(action, dtype=np.float32).reshape(2,)
        traj = self._env.step({"reset": False, "action": act})

        obs = np.asarray(traj["vector"], dtype=np.float32)
        reward = float(np.asarray(traj.get("reward", 0.0)).item())

        done = bool(traj.get("is_last", False))
        terminated = bool(traj.get("is_terminal", False))
        truncated = bool(done and not terminated)

        info = self._info_from_traj(traj)
        return obs, reward, terminated, truncated, info

    def close(self):
        if hasattr(self._env, "close"):
            self._env.close()
