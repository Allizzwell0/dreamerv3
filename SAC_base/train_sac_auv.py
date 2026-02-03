# baselines_sac/train_sac_auv.py
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import torch as th

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from auv_gym import AUVGym


def parse_kv_list(kvs: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in kvs:
        if "=" not in item:
            raise ValueError(f"Bad --env item: {item}, expected key=value")
        k, v = item.split("=", 1)
        v = v.strip()
        if v.lower() in ("true", "false"):
            out[k] = (v.lower() == "true")
            continue
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
        env = Monitor(env)  # rollout/ep_rew_mean、ep_len_mean 就靠它
        return env
    return _init


class InfoStatsCallback(BaseCallback):
    """
    把 env 的 info 里 log/* 指标写进 SB3 logger（tensorboard/progress.csv）。
    不改 AUVEnv，只利用 AUVGym 已经透传的 info['log/...']。
    """
    def __init__(self, keys: Optional[List[str]] = None, every: int = 1000, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.keys = keys or [
            "log/dist_td",
            "log/dist_dot_td",
            "log/overshoot",
            "log/r_progress",
            "log/r_dist",
            "log/r_heading",
            "log/energy_cost",
        ]
        self.every = int(every)

    def _on_step(self) -> bool:
        if self.n_calls % self.every != 0:
            return True

        infos = self.locals.get("infos", None)
        if not infos:
            return True

        for k in self.keys:
            vals = []
            for info in infos:
                if k in info:
                    try:
                        vals.append(float(info[k]))
                    except Exception:
                        pass
            if vals:
                # SB3 logger 的命名建议用 env/*
                name = "env/" + k.replace("log/", "").replace("/", "_")
                self.logger.record(name, float(np.mean(vals)))

        return True



class LosAnnealCallback(BaseCallback):
    """Linearly anneal LOS->RL mixing during training (wrapper only, AUVEnv unchanged).

    - mode="blend": los_blend_w: 0 -> 1 (0=pure LOS, 1=pure RL)
    - mode="residual": los_residual_scale: 0 -> 1

    Requires AUVGym.set_los(...) on the env instances.
    """

    def __init__(
        self,
        *,
        mode: str = "blend",
        warmup_steps: int = 0,
        anneal_steps: int = 300_000,
        disable_after: bool = False,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.mode = str(mode).lower()
        self.warmup_steps = int(max(0, warmup_steps))
        self.anneal_steps = int(max(1, anneal_steps))
        self.disable_after = bool(disable_after)

    def _on_step(self) -> bool:
        t = int(self.num_timesteps)
        if t < self.warmup_steps:
            frac = 0.0
        else:
            frac = (t - self.warmup_steps) / float(self.anneal_steps)
            frac = float(np.clip(frac, 0.0, 1.0))

        if self.mode == "blend":
            self.training_env.env_method("set_los", mode="blend", enable=True, blend_w=frac)
        elif self.mode == "residual":
            self.training_env.env_method("set_los", mode="residual", enable=True, residual_scale=frac)
        else:
            return True

        if self.disable_after and frac >= 1.0:
            self.training_env.env_method("set_los", enable=False)

        self.logger.record("env/los_mix_frac", frac)
        return True

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", type=str, required=True)
    p.add_argument("--steps", type=int, default=2_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--max_steps", type=int, default=800)

    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--subproc", action="store_true", help="use SubprocVecEnv (if pickling fails, disable)")
    p.add_argument("--norm_obs", action="store_true", help="VecNormalize(norm_obs=True)")
    p.add_argument("--norm_reward", action="store_true", help="VecNormalize(norm_reward=True)")
    p.add_argument("--tb", type=str, default=None, help="tensorboard log dir")

    # SAC 超参（稳一点的默认）
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--buffer_size", type=int, default=1_000_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--train_freq", type=int, default=1)
    p.add_argument("--gradient_steps", type=int, default=1)
    p.add_argument("--learning_starts", type=int, default=10_000)
    p.add_argument("--ent_coef", type=str, default="auto")
    p.add_argument("--device", type=str, default="cuda")
    # policy / exploration
    p.add_argument("--net_arch", type=str, default="256,256", help="MLP sizes, e.g. 256,256")
    p.add_argument("--use_sde", action="store_true", help="use State Dependent Exploration (often helps)")
    p.add_argument("--sde_sample_freq", type=int, default=-1, help="SDE resample freq, -1 = once per rollout")
    p.add_argument("--log_std_init", type=float, default=-3.0, help="initial log std (if supported)")

    # LOS schedule (optional): start with LOS prior, anneal to pure RL
    p.add_argument("--los_schedule", type=str, default="none", choices=["none", "blend", "residual"])
    p.add_argument("--los_warmup", type=int, default=0)
    p.add_argument("--los_anneal_steps", type=int, default=300_000)
    p.add_argument("--los_disable_after", action="store_true")

    # callback logging
    p.add_argument("--info_every", type=int, default=1000, help="log info stats every N steps")
    p.add_argument("--save_freq", type=int, default=50_000, help="checkpoint save freq (env steps, before /n_envs)")
    p.add_argument("--log_keys", type=str, default="", help="comma-separated info keys, e.g. log/dist_td,log/overshoot")

    # 允许覆盖 env init 参数：--env moving_goal=True --env max_goal_speed=2.0 ...
    p.add_argument("--env", action="append", default=[], help="AUVEnv kw override: key=value (repeatable)")

    args = p.parse_args()

    logdir = Path(args.logdir).expanduser().resolve()
    logdir.mkdir(parents=True, exist_ok=True)

    # 固定随机性（至少让每次对比更一致）
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(args.seed)

    env_kwargs = parse_kv_list(args.env)

    # If user requests a LOS schedule, force wrapper-side LOS options (AUVEnv unchanged).
    if args.los_schedule != "none":
        env_kwargs.setdefault("los_enable", True)
        if args.los_schedule == "blend":
            env_kwargs["los_mode"] = "blend"
            env_kwargs.setdefault("los_blend_w", 0.0)  # start pure LOS
        elif args.los_schedule == "residual":
            env_kwargs["los_mode"] = "residual"
            env_kwargs.setdefault("los_residual_scale", 0.0)  # start pure LOS

    env_fns = [make_env_fn(i, args.seed, args.dt, args.max_steps, env_kwargs) for i in range(args.n_envs)]
    if args.subproc and args.n_envs > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

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
        save_freq=max(1, args.save_freq // max(1, args.n_envs)),
        save_path=str(ckpt_dir),
        name_prefix="sac",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    # info logging
    keys = [s.strip() for s in args.log_keys.split(",") if s.strip()] if args.log_keys else None
    info_cb = InfoStatsCallback(keys=keys, every=args.info_every)

    # policy_kwargs
    net_arch = [int(x) for x in args.net_arch.split(",") if x.strip()]
    policy_kwargs = dict(net_arch=net_arch)
    if args.use_sde:
        # some SB3 versions accept this for SDE
        policy_kwargs["log_std_init"] = float(args.log_std_init)

    callbacks = [checkpoint_cb, info_cb]
    if args.los_schedule != "none":
        callbacks.append(
            LosAnnealCallback(
                mode=args.los_schedule,
                warmup_steps=args.los_warmup,
                anneal_steps=args.los_anneal_steps,
                disable_after=args.los_disable_after,
            )
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
        policy_kwargs=policy_kwargs,
        use_sde=args.use_sde,
        sde_sample_freq=args.sde_sample_freq,
        verbose=1,
        tensorboard_log=args.tb,
        device=args.device,
        seed=args.seed,
    )

    model.learn(total_timesteps=args.steps, callback=callbacks)

    # 保存最终模型
    model.save(str(logdir / "sac_final.zip"))
    if isinstance(vec_env, VecNormalize):
        vec_env.save(str(logdir / "vecnormalize.pkl"))


if __name__ == "__main__":
    main()
