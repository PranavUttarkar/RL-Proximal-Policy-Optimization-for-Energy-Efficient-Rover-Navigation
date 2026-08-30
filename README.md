# Mars Route Zero

PPO for energy-efficient rover navigation on real Jezero Crater terrain, compared against a classical costmap A\* planner in 3D.

A learned policy steers from local observations. A\* searches the full hazard/slope grid. Both are scored with the same mechanical drive-energy model.

## Why

The range and energy budgets rovers have needs to be computed and optimized throughout their mission time. An  efficient reliable rl policy that internalizes terrain. rolling and gravitational costs can propose a route from a local patch without a global replan and can be applied to new terrains and environments. This same tradeoff shows up on Earth for robots on rough ground.

## Architecture

```
HiRISE DTM ──► libmars (C++17, OpenMP Sobel) ──► costmap + A*
                         │
                         └── pybind11 ──► Gymnasium env ──► PPO (SB3 / PyTorch)
                                                │
                         FastAPI ── /api/plan, WebSocket RL ──► React + Three.js
                         JAX physics energy scores A* and RL the same way
```

1. **Terrain** — HiRISE DTM (AWS Open Data) resampled to a 512×512 float32 DEM. `libmars` builds slope, roughness, and a traversal costmap (OpenMP over rows).
2. **Planners** — 8-connected A\* minimizes costmap edge weights. PPO acts with continuous steering and speed from a 15×15 cost patch plus goal bearing.
3. **Energy** — A\* still *plans* on the costmap. Scoring (and PPO’s energy penalty) uses Perseverance-class mechanics in JAX, not $\int c\,\mathrm{d}s$.
4. **App** — FastAPI serves plans and live rollouts; the frontend draws both paths on the DEM.

```
include/, src/       libmars: I/O, Sobel slope, costmap, A*
python/bindings.cpp  pybind11
rl/                  env, PPO train, Optuna, ONNX export, JAX energy
backend/             FastAPI: /api/plan, WebSocket
frontend/            Vite + React + Three.js
scripts/             DTM pipeline, A* vs RL benchmark
```

## Models

**Slope** (Sobel on the DEM, cell size $\Delta$):

$$
\theta = \arctan\sqrt{g_x^2 + g_y^2}
$$

**Costmap.** With $\hat{s} = \min(1, \theta / 20^\circ)$, roughness $r$, and terrain class $t$:

$$
c = \min\bigl(1,\; 0.6\,\hat{s}^2 + 0.3\,r + 0.1\,t\bigr)
$$

Cells with $\theta > 25^\circ$ (Perseverance limit) are impassable.

**A\*** edge weight (hazard-aware search, not the reported energy):

$$
w(u,v) = \Delta s \bigl(0.01 + \tfrac12(c_u + c_v)\bigr)
$$

**Drive energy** (kJ; $m = 1025\,\mathrm{kg}$, $g = 3.721\,\mathrm{m/s^2}$, $C_{rr} = 0.24$, $k_g = 0.28$). Elevation is bilinear. Climb only — no downhill regen:

$$
E = mg \sum_i \Bigl( C_{rr}\,\Delta s_i + \max(0,\Delta h_i) + k_g\,\Delta s_i \sin^2\theta_i^{\uparrow} \Bigr)
$$

where $\sin\theta_i^{\uparrow} = \max(0,\Delta h_i) / (\Delta s_i + \varepsilon)$. Rolling on extra A\* detours usually dominates.

PPO reward uses the same $E$ per step, plus potential-based goal shaping ($\gamma = 0.99$).

## Benchmark

50 seeded start/goal pairs on Jezero (512×512, $20.89\,\mathrm{m/px}$). A\* on the costmap; PPO blended rollout, 1200-step cap. Sum is over the **22** pairs where RL reached the goal (A\* succeeded on all 50). Replay: `python scripts/bench_astar_vs_rl.py`.

| | $\sum E$ (kJ) | pairs |
|---|---|---|
| Costmap A\* | 132 855 | 22 completed |
| PPO | 98 465 | **26% less**; 20/22 pair wins |

RL finishes 22/50 within the step cap. Incomplete episodes are not in the sum.

## Quick start

CMake 3.18+, C++17, Python 3 with headers, Node.js.

```bash
python scripts/gen_synthetic_terrain.py        # or real HiRISE below
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

`libmars` lands in `build/` — put that on `PYTHONPATH`. Windows MinGW: `cmake --preset windows-mingw`.

```bash
pip install -r requirements.txt
python -m rl.train --smoke
python -m rl.train --timesteps 2000000 --n-envs 8
python -m rl.train --resume models/mars_ppo_latest.zip --timesteps 80000
```

Optuna: `python -m rl.optimize --timesteps-per-trial 12000 --trials 8`. ONNX: `python -m rl.export_onnx`.

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
cd frontend && npm install && npm run dev
```

http://127.0.0.1:5173 — click start/goal, compare A\* and RL.

**HiRISE Jezero:** `./scripts/download_jezero_dtm.sh` (or `.ps1`), then `python scripts/hirise_geotiff_to_elevation_bin.py --input data/dtm/aws_sync/DTEEC_*.tif`. See `scripts/README.md`.

```bash
pytest tests/test_env.py -v
ctest --test-dir build
```

## License

MIT — see `LICENSE`.
