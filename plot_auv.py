#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot AUV evaluation results from CSV produced by eval_auv.py

Usage:
  python plot_auv.py --csv eval_trajectories.csv --ep 0

If your CSV includes x,y,goal_x,goal_y (because your obs had them),
this script will also offer an XY trajectory plot.
"""
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os


def load_csv(path):
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def to_float(v, default=np.nan):
    try:
        return float(v)
    except Exception:
        return default


def plot_distance_time(rows, episode=0):
    t, dist = [], []
    for r in rows:
        if int(r["episode"]) == episode:
            t.append(int(r["t"]))
            dist.append(to_float(r["dist"]))
    if not t:
        print(f"No data for episode {episode}")
        return
    plt.figure()
    plt.plot(t, dist)
    plt.xlabel("t (step)")
    plt.ylabel("distance to goal")
    plt.title(f"Episode {episode}: Distance vs Time")
    plt.tight_layout()
    plt.show()


def plot_reward_time(rows, episode=0):
    t, rew = [], []
    for r in rows:
        if int(r["episode"]) == episode:
            t.append(int(r["t"]))
            rew.append(to_float(r["reward"], 0.0))
    if not t:
        print(f"No data for episode {episode}")
        return
    plt.figure()
    plt.plot(t, rew)
    plt.xlabel("t (step)")
    plt.ylabel("reward")
    plt.title(f"Episode {episode}: Reward vs Time")
    plt.tight_layout()
    plt.show()


def plot_speed_time(rows, episode=0):
    t, speed = [], []
    for r in rows:
        if int(r["episode"]) == episode:
            t.append(int(r["t"]))
            u = to_float(r["u"], 0.0)
            v = to_float(r["v"], 0.0)
            speed.append(np.hypot(u, v))
    if not t:
        print(f"No data for episode {episode}")
        return
    plt.figure()
    plt.plot(t, speed)
    plt.xlabel("t (step)")
    plt.ylabel("speed sqrt(u^2+v^2)")
    plt.title(f"Episode {episode}: Speed vs Time")
    plt.tight_layout()
    plt.show()


def plot_returns_hist(rows):
    ep_returns = {}
    ep_partial = {}
    for r in rows:
        ep = int(r["episode"])
        rew = to_float(r["reward"], 0.0)
        ep_partial[ep] = ep_partial.get(ep, 0.0) + rew
        if str(r["is_last"]).lower() == "true":
            ep_returns[ep] = ep_partial[ep]
    if not ep_returns:
        print("No completed episodes found.")
        return
    vals = np.array(list(ep_returns.values()), dtype=float)
    plt.figure()
    plt.hist(vals, bins=20)
    plt.xlabel("episode return")
    plt.ylabel("count")
    plt.title("Returns Histogram")
    plt.tight_layout()
    plt.show()


def plot_xy_trajectory(rows, episode=0):
    xs, ys = [], []
    gx = gy = None
    for r in rows:
        if int(r["episode"]) == episode:
            x = to_float(r["x"])
            y = to_float(r["y"])
            if not np.isnan(x) and not np.isnan(y):
                xs.append(x)
                ys.append(y)
            # goal (if present)
            gx_i = to_float(r["goal_x"])
            gy_i = to_float(r["goal_y"])
            if not np.isnan(gx_i) and not np.isnan(gy_i):
                gx, gy = gx_i, gy_i
    if not xs:
        print("No (x,y) found in CSV (your env may not export them). Skipping XY plot.")
        return
    plt.figure()
    plt.plot(xs, ys, marker=".", linewidth=1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Episode {episode}: XY Trajectory")
    if gx is not None and gy is not None:
        plt.scatter([gx], [gy], marker="x", s=80)
    plt.axis("equal")
    plt.tight_layout()


def main(csv_path, episode_for_curves=0, with_xy=True, save_fig=True):
    rows = load_csv(csv_path)
    save_dir = None
    if save_fig:
        save_dir = os.path.join(os.path.dirname(csv_path), "plots")
        os.makedirs(save_dir, exist_ok=True)
        print(f"[plot_auv] Saving all plots to: {save_dir}")

    def save_or_show(fig, name):
        if save_fig and save_dir:
            fig.savefig(os.path.join(save_dir, name), dpi=300)
        else:
            fig.show()

    # Distance vs Time
    fig = plt.figure()
    plot_distance_time(rows, episode_for_curves)
    save_or_show(fig, f"ep{episode_for_curves}_distance.png")

    # Reward vs Time
    fig = plt.figure()
    plot_reward_time(rows, episode_for_curves)
    save_or_show(fig, f"ep{episode_for_curves}_reward.png")

    # Speed vs Time
    fig = plt.figure()
    plot_speed_time(rows, episode_for_curves)
    save_or_show(fig, f"ep{episode_for_curves}_speed.png")

    # Returns Histogram
    fig = plt.figure()
    plot_returns_hist(rows)
    save_or_show(fig, f"returns_hist.png")

    # XY trajectory
    if with_xy:
        fig = plt.figure()
        plot_xy_trajectory(rows, episode_for_curves)
        save_or_show(fig, f"ep{episode_for_curves}_trajectory.png")



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="eval_trajectories.csv")
    p.add_argument("--ep", type=int, default=0, help="which episode to use for line plots")
    p.add_argument("--no_xy", action="store_true", help="disable XY trajectory plot")
    args = p.parse_args()
    main(args.csv, args.ep, with_xy=(not args.no_xy))
