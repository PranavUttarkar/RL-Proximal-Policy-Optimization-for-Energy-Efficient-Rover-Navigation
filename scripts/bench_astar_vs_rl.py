"""Benchmark physics energy: costmap A* vs PPO on a fixed start/goal set."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "build", ROOT / "build" / "Release"):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dll_windows import ensure_mingw_dll_dirs

ensure_mingw_dll_dirs(ROOT)

import libmars
from backend.rollout import blended_action
from rl.mars_env import MarsRoverEnv
from rl.physics_energy import path_energy_kj
from stable_baselines3 import PPO

BIN = ROOT / "data/costmap/jezero_elevation.bin"
META = json.loads((ROOT / "data/dtm/jezero_meta.json").read_text(encoding="utf-8"))
MPP = float(META["meters_per_pixel"])
GRID = 512
N = int(os.environ.get("MARS_BENCH_N", "50"))
SEED = 42
MAX_STEPS = int(os.environ.get("MARS_RL_WS_MAX_STEPS", "1200"))


def sample_pairs(cost: np.ndarray, n: int, seed: int, margin: int = 15):
    rng = np.random.default_rng(seed)
    pairs = []
    tries = 0
    while len(pairs) < n and tries < n * 400:
        tries += 1
        r0 = int(rng.integers(margin, GRID - margin))
        c0 = int(rng.integers(margin, GRID - margin))
        if cost[r0, c0] >= 0.5:
            continue
        r1 = int(rng.integers(margin, GRID - margin))
        c1 = int(rng.integers(margin, GRID - margin))
        if cost[r1, c1] >= 0.5:
            continue
        if np.hypot(c1 - c0, r1 - r0) <= 50:
            continue
        pairs.append((r0, c0, r1, c1))
    return pairs


def waypoints_energy(elev: np.ndarray, pts_rc: list[tuple[float, float]]) -> dict:
    if len(pts_rc) < 2:
        return {"energy_kj": 0.0, "distance_m": 0.0, "climb_m": 0.0}
    rows = [p[0] for p in pts_rc]
    cols = [p[1] for p in pts_rc]
    return path_energy_kj(elev, rows, cols, MPP)


def main() -> None:
    print("loading terrain + JAX energy...", flush=True)
    grid = libmars.build_full_terrain(str(BIN), GRID, GRID, MPP)
    cost = np.asarray(grid.get_costmap_array())
    elev = np.asarray(grid.get_elevation_array())
    model = PPO.load(str(ROOT / "models" / "mars_ppo_latest.zip"))
    env = MarsRoverEnv(str(BIN), GRID, MPP, random_start_goal=False)
    pairs = sample_pairs(cost, N, SEED)
    print(f"sampled {len(pairs)} pairs  mpp={MPP:.3f}  step_cells={env.step_cells}", flush=True)

    rows = []
    for i, (sr, sc, gr, gc) in enumerate(pairs):
        start = libmars.GridPoint(sr, sc)
        goal = libmars.GridPoint(gr, gc)
        astar = libmars.astar_plan(grid, start, goal)
        straight = libmars.straight_line(grid, start, goal)
        astar_ok = len(astar.waypoints) > 1 and (
            astar.waypoints[-1].row == gr and astar.waypoints[-1].col == gc
        )
        astar_pts = [(float(p.row), float(p.col)) for p in astar.waypoints]
        straight_pts = [(float(p.row), float(p.col)) for p in straight.waypoints]
        aE = waypoints_energy(elev, astar_pts)
        sE = waypoints_energy(elev, straight_pts)

        obs, _ = env.reset(
            options={
                "start_col": sc,
                "start_row": sr,
                "goal_col": gc,
                "goal_row": gr,
            }
        )
        rng = np.random.default_rng(SEED + 1000 + i)
        terminated = False
        truncated = False
        steps = 0
        while not (terminated or truncated) and steps < MAX_STEPS:
            action = blended_action(model, env, obs, rng=rng)
            obs, _, terminated, truncated, _info = env.step(action)
            steps += 1
        rl_pts = [(float(p[1]), float(p[0])) for p in env.trajectory]
        rE = waypoints_energy(elev, rl_pts)
        reached = bool(terminated and np.linalg.norm(env.goal - env.pos) < env.GOAL_RADIUS)

        row = {
            "i": i,
            "start": [sr, sc],
            "goal": [gr, gc],
            "astar_energy": aE["energy_kj"],
            "astar_dist": aE["distance_m"],
            "astar_climb": aE["climb_m"],
            "astar_ok": bool(astar_ok),
            "straight_energy": sE["energy_kj"],
            "straight_dist": sE["distance_m"],
            "rl_energy": rE["energy_kj"],
            "rl_dist": rE["distance_m"],
            "rl_climb": rE["climb_m"],
            "rl_steps": steps,
            "rl_reached": reached,
        }
        rows.append(row)
        tag = "WIN" if reached and rE["energy_kj"] < aE["energy_kj"] else ("lose" if reached else "miss")
        print(
            f"{i:02d} {tag} A*={aE['energy_kj']:.1f}kJ RL={rE['energy_kj']:.1f}kJ "
            f"reach={reached}",
            flush=True,
        )

    ok = [r for r in rows if r["rl_reached"] and r["astar_ok"]]
    sa = sum(r["astar_energy"] for r in ok)
    sr = sum(r["rl_energy"] for r in ok)
    beats = bool(ok) and sr < sa
    pct = (1.0 - sr / sa) * 100.0 if sa > 0 and ok else 0.0
    print(
        json.dumps(
            {
                "n_pairs": len(rows),
                "n_completed": len(ok),
                "sum_astar_kj": sa,
                "sum_rl_kj": sr,
                "rl_pct_less_energy": pct,
                "rl_beats_astar": beats,
            },
            indent=2,
        ),
        flush=True,
    )
    out = ROOT / "results" / "astar_vs_rl_bench.json"
    out.write_text(
        json.dumps(
            {
                "n": len(rows),
                "seed": SEED,
                "mpp": MPP,
                "max_steps": MAX_STEPS,
                "energy": "physics_kj_jax",
                "rows": rows,
                "summary": {
                    "n_completed": len(ok),
                    "sum_astar_kj": sa,
                    "sum_rl_kj": sr,
                    "rl_pct_less_energy": pct,
                    "rl_beats_astar": beats,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}", flush=True)
    if not beats:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
