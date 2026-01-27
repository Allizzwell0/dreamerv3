# baselines_sac/train_sac_auv.py
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from auv_gym import AUVGym


def parse_kv_list(kvs: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in kvs:
        if "=" not in item:
            raise ValueError(f"Bad --env item: {item}, expected key=value")
        k, v = item.split("=", 1)
        v = v.strip()
        # 轻量解析：bool/int/float/str
        if v.lower() in ("true", "false"):
            out[k] = (v.lower() == "true")
        else:
            try:
                if "." in v or "e" in v.lower():
                    out[k] = float(v)
                else:
                    out[k] = int(v)
            except Exception:
                out[k] = v
    return out


def make_env_fn(rank: int, seed: int, dt: float, max_steps: int, env_kwargs: Dict[str, Any]):
    def _init():
        env = AUVGym(dt=dt, max_steps=max_steps, seed=seed + rank, **env_kwargs)
        env = Monitor(env)  # 记录 episode reward/len
        return env
    return _init


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", type=str, required=True)
    p.add_argument("--steps", type=int, default=2_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--max_steps", type=int, default=800)

    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--subproc", action="store_true", help="use SubprocVecEnv")
    p.add_argument("--norm_obs", action="store_true", help="use VecNormalize(norm_obs=True)")
    p.add_argument("--norm_reward", action="store_true", help="use VecNormalize(norm_reward=True)")
    p.add_argument("--tb", type=str, default=None, help="tensorboard log dir")

    # SAC 超参（先给一套较稳的默认）
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--buffer_size", type=int, default=1_000_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--train_freq", type=int, default=1)
    p.add_argument("--gradient_steps", type=int, default=1)
    p.add_argument("--learning_starts", type=int, default=10_000)
    p.add_argument("--ent_coef", type=str, default="auto")

    # 允许覆盖 env init 参数：--env moving_goal=True --env max_goal_speed=2.0 ...
    p.add_argument("--env", action="append", default=[], help="AUVEnv kw override: key=value (repeatable)")

    args = p.parse_args()

    logdir = Path(args.logdir).expanduser().resolve()
    logdir.mkdir(parents=True, exist_ok=True)

    env_kwargs = parse_kv_list(args.env)

    # vec env
    env_fns = [make_env_fn(i, args.seed, args.dt, args.max_steps, env_kwargs) for i in range(args.n_envs)]
    if args.subproc and args.n_envs > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    # 可选：观测/回报归一化（推荐至少 norm_obs）
    if args.norm_obs or args.norm_reward:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=args.norm_obs,
            norm_reward=args.norm_reward,
            clip_obs=10.0,
        )

    # checkpoint
    ckpt_dir = logdir / "ckpt"
    ckpt_dir.mkdir(exist_ok=True)
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000 // max(1, args.n_envs),
        save_path=str(ckpt_dir),
        name_prefix="sac",
        save_replay_buffer=False,
        save_vecnormalize=True,  # 如果用了 VecNormalize，会在这里保存统计量
    )

    model = SAC(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        ent_coef=args.ent_coef,
        verbose=1,
        tensorboard_log=args.tb,
        device="cuda",
        seed=args.seed,
    )

    model.learn(total_timesteps=args.steps, callback=[checkpoint_cb])

    # 保存最终模型
    model.save(str(logdir / "sac_final.zip"))
    if isinstance(vec_env, VecNormalize):
        vec_env.save(str(logdir / "vecnormalize.pkl"))


if __name__ == "__main__":
    main()
