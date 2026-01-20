#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简易 PBT 外挂：对 AUV 环境的 reward / TD 超参做进化搜索（从给定最优参数继续细化），支持中断后继续。

目录结构假定为：
  /home/mayue/WorldModel/Dreamer/
    dreamerv3/
      main.py
      eval_auv.py
    pbt_auv.py   ← 本脚本

运行方式:
  cd /home/mayue/WorldModel/Dreamer
  python pbt_auv.py
"""

from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from eval_auv import evaluate_auv  # 使用你自己写的 eval 函数


# ==================== 配置区 ====================

ROOT = Path(__file__).resolve().parent
MAIN_SCRIPT = ROOT / "dreamerv3" / "main.py"

TASK_NAME = "auv_custom"
CONFIG_NAME = "auv"

BASE_LOGDIR = Path("/home/mayue/logdir/pbt_conf_auv_2")

POP_SIZE = 5
GENERATIONS = 6
STEPS_PER_GEN = int(1e6)

HP_BOUNDS: Dict[str, Tuple[float, float]] = {
    "env.auv.base_k_progress": (1.0, 5.0),
    "env.auv.k_dist": (0.3, 2.0),
    "env.auv.k_ring": (0.0, 1.0),
    "env.auv.bonus_max": (0.5, 3.0),
    "env.auv.hold_bonus": (0.0, 1.5),
    "env.auv.k_speed_near": (0.0, 1.0),
    "env.auv.gamma_far": (0.2, 1.0),
    "env.auv.k_heading_base": (0.1, 0.8),
    "env.auv.td_r": (0.5, 3.0),
    "env.auv.td_N": (3.0, 12.0),
    # ---- agent.* (confidence) ----
    "agent.conf_alpha": (0.2, 6.0),
    "agent.conf_min": (0.0, 0.3),
    "agent.conf_low_eta_min": (0.0, 0.5),
    "agent.conf_low_eta_max": (0.5, 1.0),
    "agent.conf_low_eta_gamma": (0.5, 2.0),
    "agent.conf_low_tau": (0.1, 1.0),
    # "agent.conf_k_thrust": (0.02, 0.6),
    # "agent.conf_k_rudder": (0.2, 4.0),
}

BEST_HP: Dict[str, float] = {
    "env.auv.base_k_progress": 4.995356892068424,
    "env.auv.k_dist": 0.3,
    "env.auv.k_ring": 0.0,
    "env.auv.bonus_max": 2.094027569655085,
    "env.auv.hold_bonus": 0.4880082494158719,
    "env.auv.k_speed_near": 0.41883239676790485,
    "env.auv.gamma_far": 0.943825316852527,
    "env.auv.k_heading_base": 0.1569655750036578,
    "env.auv.td_r": 3.0,
    "env.auv.td_N": 11.050894970380819,
    "agent.conf_alpha": 0.2,
    "agent.conf_min": 0.1,
    "agent.conf_low_eta_min": 0.05,
    "agent.conf_low_eta_max": 1.0,
    "agent.conf_low_eta_gamma": 1.0,
    "agent.conf_low_tau": 0.5,
    # "agent.conf_k_thrust": 0.3,
    # "agent.conf_k_rudder": 2.0,
}

MUTATION_STD_FRAC = 0.2
ELITE_FRAC = 0.5
INIT_MODE = "around_best"  # "around_best" or "random"

# ✅ 断点续跑开关
RESUME = False

# state.json 保存位置
STATE_PATH = BASE_LOGDIR / "state.json"


# ==================== 工具函数 ====================

def random_hparams() -> Dict[str, float]:
    hp: Dict[str, float] = {}
    for name, (lo, hi) in HP_BOUNDS.items():
        hp[name] = random.uniform(lo, hi)
    return hp


def mutate_hparams(parent: Dict[str, float]) -> Dict[str, float]:
    child: Dict[str, float] = {}
    for name, value in parent.items():
        lo, hi = HP_BOUNDS[name]
        std = MUTATION_STD_FRAC * (hi - lo)
        new = value + random.gauss(0.0, std)
        new = max(lo, min(hi, new))
        child[name] = float(new)
    return child


def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def build_argv(hparams: Dict[str, Any], logdir: Path, seed: int) -> List[str]:
    argv: List[str] = [
        "python", str(MAIN_SCRIPT),
        "--task", TASK_NAME,
        "--configs", CONFIG_NAME,
        f"--logdir={str(logdir)}",
        f"--run.steps={STEPS_PER_GEN}",
        f"--seed={seed}",
        "--script=train",
        "--env.auv.debug_trace=False",
    ]

    # 为了可复现，按 key 排序（可选但推荐）
    for key in sorted(hparams.keys()):
        val = hparams[key]
        # bool 统一成 true/false，elements/config 通常能吃
        if isinstance(val, bool):
            val = str(val).lower()
        argv.append(f"--{key}={val}")

    return argv



def run_one(pop_idx: int, gen_idx: int, hparams: Dict[str, Any]) -> Path:
    logdir = BASE_LOGDIR / f"pop{pop_idx:02d}" / f"gen{gen_idx:02d}"
    logdir.mkdir(parents=True, exist_ok=True)

    # 保存当前超参
    save_json(logdir / "hparams.json", hparams)

    # 如果已经训练过（粗判），跳过训练
    ckpt_dir = logdir / "ckpt"
    if ckpt_dir.exists() and any(ckpt_dir.iterdir()):
        print(f"[PBT] Skip training (ckpt exists): pop={pop_idx} gen={gen_idx} -> {logdir}")
        return logdir

    seed = 1000 + pop_idx * 100 + gen_idx
    argv = build_argv(hparams, logdir, seed)

    print(f"[PBT] Run pop={pop_idx} gen={gen_idx}")
    print(" ".join(argv))
    subprocess.run(argv, check=True)
    return logdir


def evaluate_logdir(logdir: Path, eval_seed: int, episodes: int = 10) -> float:
    """
    评估结果落盘到 logdir/eval_pbt/fitness.json。
    若该文件已存在，则直接读取并返回 fitness（支持断点续跑）。
    """
    eval_dir = logdir / "eval_pbt"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_csv = eval_dir / "trajectories.csv"
    fitness_path = eval_dir / "fitness.json"

    if fitness_path.exists():
        data = load_json(fitness_path)
        fitness = float(data["fitness"])
        print(f"[PBT] Skip eval (fitness cached): {logdir} -> fitness={fitness:.3f}")
        return fitness

    try:
        metrics = evaluate_auv(
            str(logdir),
            episodes=episodes,
            dt=0.05,
            max_steps=800,
            success_threshold=1.0,
            track_success_ratio=0.7,
            out_dir=eval_dir,     # ✅ 新版是 out_dir
            seed=eval_seed,
            verbose=False,
            use_env_from_ckpt=True,   # ✅ 推荐：评估环境与训练 config 一致
        )

    except Exception as e:
        print(f"[PBT] WARNING: eval_auv failed for {logdir}: {e}")
        save_json(fitness_path, {"fitness": float("-inf"), "error": str(e)})
        return float("-inf")

    success_rate = float(metrics.get("success_rate", 0.0))
    mean_dist = float(metrics.get("mean_dist_mean", 1e9))
    track_ratio_mean = float(metrics.get("track_ratio_mean", 0.0))
    max_dist_mean = float(metrics.get("max_dist_mean", 1e9))

    fitness = success_rate * 10.0 + track_ratio_mean - mean_dist - 0.5 * max_dist_mean

    save_json(fitness_path, {
        "fitness": fitness,
        "success_rate": success_rate,
        "mean_dist_mean": mean_dist,
        "track_ratio_mean": track_ratio_mean,
        "max_dist_mean": max_dist_mean,
        "metrics": metrics,
    })

    print(
        f"[PBT] Eval {logdir} -> success_rate={success_rate:.3f}, mean_dist={mean_dist:.3f}, "
        f"max_dist_mean={max_dist_mean:.3f}, fitness={fitness:.3f}"
    )
    return fitness


def init_population() -> List[Dict[str, Any]]:
    if INIT_MODE == "random":
        return [random_hparams() for _ in range(POP_SIZE)]
    if INIT_MODE == "around_best":
        pop: List[Dict[str, Any]] = [dict(BEST_HP)]
        for _ in range(1, POP_SIZE):
            pop.append(mutate_hparams(BEST_HP))
        return pop
    raise ValueError(f"Unknown INIT_MODE: {INIT_MODE}")


# ==================== 断点续跑状态管理 ====================

def save_state(state: Dict[str, Any]) -> None:
    save_json(STATE_PATH, state)


def load_state() -> Optional[Dict[str, Any]]:
    if not STATE_PATH.exists():
        return None
    return load_json(STATE_PATH)


def make_initial_state() -> Dict[str, Any]:
    """
    state 结构：
      gen: 当前将要跑的 generation index
      population_hparams: list[dict]
      best_overall_score, best_overall_hp, best_overall_logdir
      py_random_state: random.getstate() 可序列化形式
      np_random_state: np.random.get_state() 可序列化形式
    """
    pop = init_population()
    state: Dict[str, Any] = {
        "gen": 0,
        "population_hparams": pop,
        "best_overall_score": float("-inf"),
        "best_overall_hp": dict(BEST_HP),
        "best_overall_logdir": None,
        # RNG 状态（为了可重复 + 继续跑）
        "py_random_state": repr(random.getstate()),
        "np_random_state": repr(np.random.get_state()),
    }
    return state


def restore_rng(state: Dict[str, Any]) -> None:
    """
    从 state.json 恢复随机数状态。
    注意：这里用 eval(repr(...)) 的方式恢复，是最方便的纯标准库做法。
    如果你不喜欢 eval，可以换成 pickle/base64 序列化。
    """
    try:
        random.setstate(eval(state["py_random_state"]))
        np.random.set_state(eval(state["np_random_state"]))
    except Exception:
        # 恢复失败就算了，不致命
        pass


def update_rng_in_state(state: Dict[str, Any]) -> None:
    state["py_random_state"] = repr(random.getstate())
    state["np_random_state"] = repr(np.random.get_state())


# ==================== 主循环（支持断点续跑） ====================

def main() -> None:
    BASE_LOGDIR.mkdir(parents=True, exist_ok=True)

    if RESUME:
        state = load_state()
        if state is None:
            state = make_initial_state()
            save_state(state)
            print(f"[PBT] No existing state. Start new and save to {STATE_PATH}")
        else:
            print(f"[PBT] Resume from {STATE_PATH}")
            restore_rng(state)
    else:
        # 强制重新开始（会覆盖 state.json，但不会删除旧日志目录）
        state = make_initial_state()
        save_state(state)
        print(f"[PBT] Start fresh (RESUME=False). State saved to {STATE_PATH}")

    # 从 state 取出
    gen = int(state["gen"])
    population_hparams: List[Dict[str, Any]] = state["population_hparams"]
    best_overall_score = float(state["best_overall_score"])
    best_overall_hp: Dict[str, Any] = dict(state["best_overall_hp"])
    best_overall_logdir = state.get("best_overall_logdir", None)

    # 逐代跑，跑完一代就立刻落盘 state（中断安全）
    while gen < GENERATIONS:
        print(f"\n======= Generation {gen} =======")
        scores: List[float] = []
        eval_seed = 2000 + gen * 1000

        # 1) 本代训练 & 评估
        for i in range(POP_SIZE):
            hparams = population_hparams[i]
            logdir = run_one(i, gen, hparams)
            score = evaluate_logdir(logdir, eval_seed=eval_seed, episodes=10)
            scores.append(score)
            print(f"[PBT] pop {i} gen {gen} score = {score:.3f}")

            if score > best_overall_score:
                best_overall_score = score
                best_overall_hp = dict(hparams)
                best_overall_logdir = str(logdir)

                # 立刻保存 best_overall.json
                best_json = BASE_LOGDIR / "best_overall.json"
                save_json(best_json, {
                    "score": best_overall_score,
                    "hparams": best_overall_hp,
                    "logdir": best_overall_logdir,
                })
                print(f"[PBT] New best overall saved: {best_json}")

        # 2) 选精英
        order = np.argsort(scores)[::-1]
        elites = order[: max(1, int(ELITE_FRAC * POP_SIZE))]
        print(f"[PBT] elites indices: {list(map(int, elites))}")

        # 3) 生成下一代
        new_population: List[Dict[str, Any]] = [None] * POP_SIZE  # type: ignore
        for idx in order:
            idx = int(idx)
            if idx in set(map(int, elites)):
                new_population[idx] = dict(population_hparams[idx])
            else:
                parent_idx = int(random.choice(list(map(int, elites))))
                parent_hp = population_hparams[parent_idx]
                new_population[idx] = mutate_hparams(parent_hp)

        population_hparams = new_population

        # 4) 更新 state 并落盘（关键：保证中断后能继续）
        gen += 1
        state["gen"] = gen
        state["population_hparams"] = population_hparams
        state["best_overall_score"] = best_overall_score
        state["best_overall_hp"] = best_overall_hp
        state["best_overall_logdir"] = best_overall_logdir
        update_rng_in_state(state)
        save_state(state)

        # 5) 打印本代最优（用于实时观察）
        best_idx = int(order[0])
        print(f"[PBT] Best in gen {gen-1}: pop {best_idx}, score={scores[best_idx]:.3f}")
        print(f"[PBT] Current best overall score: {best_overall_score:.3f}")
        print(f"[PBT] Current best overall hparams:\n{json.dumps(best_overall_hp, indent=2, ensure_ascii=False)}")

    print("\n[PBT] Finished all generations.")
    print(f"[PBT] Best overall score: {best_overall_score:.3f}")
    print(f"[PBT] Best overall logdir: {best_overall_logdir}")
    print(f"[PBT] Best overall hparams:\n{json.dumps(best_overall_hp, indent=2, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
