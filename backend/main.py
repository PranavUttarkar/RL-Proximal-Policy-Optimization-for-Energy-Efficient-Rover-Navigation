"""FastAPI service for terrain metadata and path planning (requires built libmars)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from dll_windows import ensure_mingw_dll_dirs

ensure_mingw_dll_dirs(_REPO)

import asyncio
import json
from io import BytesIO

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[misc, assignment]

try:
    from stable_baselines3 import PPO as SB3PPO
except ImportError:
    SB3PPO = None  # type: ignore[misc, assignment]

_ROOT = _REPO
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

for _extra in (
    os.environ.get("MARS_LIBMARS_BUILD", ""),
    str(_ROOT / "build"),
    str(_ROOT / "build" / "Release"),
    str(_ROOT / "build" / "Debug"),
    str(_ROOT / "build" / "RelWithDebInfo"),
):
    if _extra and Path(_extra).is_dir() and _extra not in sys.path:
        sys.path.insert(0, _extra)

import numpy as np

from rl.physics_energy import path_energy_kj

try:
    import libmars
except ImportError as e:
    if sys.platform == "win32" and "DLL load failed" in str(e):
        raise ImportError(
            "libmars could not load its MinGW runtime DLLs. Install MSYS2 UCRT64 and set "
            "MSYS2_UCRT64_BIN to the ucrt64\\\\bin directory, or add that folder to PATH, "
            "then restart. Build output (libmars*.pyd) must be on PYTHONPATH — e.g. "
            "set PYTHONPATH to the CMake build directory."
        ) from e
    raise

_TERRAIN_PATH = Path(
    os.environ.get("MARS_ELEVATION_BIN", _ROOT / "data/costmap/jezero_elevation.bin")
)
_GRID = int(os.environ.get("MARS_GRID_SIZE", "512"))


def _meters_per_pixel() -> float:
    env = os.environ.get("MARS_METERS_PER_PIXEL", "").strip()
    if env:
        return float(env)
    meta_path = _ROOT / "data/dtm/jezero_meta.json"
    if meta_path.is_file():
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            m = data.get("meters_per_pixel")
            if isinstance(m, (int, float)) and m > 0:
                return float(m)
        except (json.JSONDecodeError, OSError):
            pass
    return 1.0


_MPP = _meters_per_pixel()

if not _TERRAIN_PATH.is_file():
    raise FileNotFoundError(
        f"Elevation binary not found: {_TERRAIN_PATH}. "
        "Run: python scripts/gen_synthetic_terrain.py"
    )

TERRAIN = libmars.build_full_terrain(str(_TERRAIN_PATH), _GRID, _GRID, _MPP)
ELEV = np.asarray(TERRAIN.get_elevation_array())
COST = np.asarray(TERRAIN.get_costmap_array())
SLOPE = np.asarray(TERRAIN.get_slope_array())

app = FastAPI(title="Mars Route Zero API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("MARS_CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/terrain")
async def get_terrain_metadata():
    return {
        "width": TERRAIN.width,
        "height": TERRAIN.height,
        "meters_per_pixel": TERRAIN.meters_per_pixel,
        "elevation_min": float(ELEV.min()),
        "elevation_max": float(ELEV.max()),
        "elevation_range_m": float(ELEV.max() - ELEV.min()),
        "mean_slope_deg": float(SLOPE.mean()),
        "traversable_fraction": float((COST < 0.85).mean()),
    }


class PlanRequest(BaseModel):
    start_col: int
    start_row: int
    goal_col: int
    goal_row: int
    planner: str = "astar"


class PlanResponse(BaseModel):
    waypoints: list[list[int]]
    elevations_m: list[float]
    total_cost: float
    total_distance_m: float
    energy_score: float
    planner: str


@app.post("/api/plan", response_model=PlanResponse)
async def plan_route(req: PlanRequest):
    start = libmars.GridPoint(req.start_row, req.start_col)
    goal = libmars.GridPoint(req.goal_row, req.goal_col)
    if req.planner == "astar":
        path = libmars.astar_plan(TERRAIN, start, goal)
    else:
        path = libmars.straight_line(TERRAIN, start, goal)
    elev_m = [float(ELEV[p.row, p.col]) for p in path.waypoints]
    phys = path_energy_kj(
        ELEV,
        [p.row for p in path.waypoints],
        [p.col for p in path.waypoints],
        _MPP,
    )
    return PlanResponse(
        waypoints=[[p.col, p.row] for p in path.waypoints],
        elevations_m=elev_m,
        total_cost=float(path.total_cost),
        total_distance_m=float(phys["distance_m"] or path.total_distance_m),
        energy_score=float(phys["energy_kj"]),
        planner=req.planner,
    )


@app.get("/api/costmap-tiles/{row}/{col}")
async def get_costmap_tile(row: int, col: int, size: int = 64):
    r0 = max(0, row * size)
    c0 = max(0, col * size)
    r1 = min(TERRAIN.height, r0 + size)
    c1 = min(TERRAIN.width, c0 + size)
    tile = COST[r0:r1, c0:c1].flatten().tolist()
    return {"data": tile, "r0": r0, "c0": c0, "rows": r1 - r0, "cols": c1 - c0}


@app.get("/api/rl-status")
async def rl_status():
    """Whether a PPO checkpoint is available for WebSocket / optional ONNX."""
    path = Path(os.environ.get("MARS_PPO_PATH", _ROOT / "models" / "mars_ppo_latest.zip"))
    onnx_path = _ROOT / "frontend" / "public" / "mars_policy.onnx"
    return {
        "ppo_zip_exists": path.is_file(),
        "ppo_zip_path": str(path) if path.is_file() else None,
        "onnx_exists": onnx_path.is_file(),
        "stable_baselines3": SB3PPO is not None,
    }


@app.get("/api/perseverance-waypoints")
async def perseverance_waypoints():
    """Sample / manual waypoints for overlay (replace with real JPL traverse when available)."""
    wp_path = _ROOT / "data" / "perseverance" / "waypoints.json"
    if not wp_path.is_file():
        return {"label": "", "points": []}
    data = json.loads(wp_path.read_text(encoding="utf-8"))
    out = []
    for p in data.get("points", []):
        r, c = int(p["row"]), int(p["col"])
        if 0 <= r < TERRAIN.height and 0 <= c < TERRAIN.width:
            q = {**p, "elevation_m": float(ELEV[r, c])}
            out.append(q)
    return {"label": data.get("label", ""), "points": out}


@app.get("/api/costmap.png")
async def costmap_png():
    """Grayscale PNG of traversal cost [0,1] for terrain overlay."""
    if Image is None:
        raise RuntimeError("Install Pillow: pip install pillow")
    c = np.asarray(COST, dtype=np.float64)
    arr = (np.clip(c, 0.0, 1.0) * 255.0).astype(np.uint8)
    im = Image.fromarray(arr, mode="L")
    buf = BytesIO()
    im.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/api/heightmap.png")
async def heightmap_png():
    """Grayscale PNG (normalized elevation) for Three.js displacement / texture."""
    if Image is None:
        raise RuntimeError("Install Pillow: pip install pillow")
    e = np.asarray(ELEV, dtype=np.float64)
    lo, hi = float(e.min()), float(e.max())
    g = (e - lo) / (hi - lo + 1e-12)
    arr = (np.clip(g, 0.0, 1.0) * 255.0).astype(np.uint8)
    im = Image.fromarray(arr, mode="L")
    buf = BytesIO()
    im.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/api/cell")
async def cell_query(row: int, col: int):
    """Single-cell terrain stats for hover / tooltips."""
    if row < 0 or row >= TERRAIN.height or col < 0 or col >= TERRAIN.width:
        return {"error": "out of bounds"}
    return {
        "row": row,
        "col": col,
        "elevation_m": float(ELEV[row, col]),
        "slope_deg": float(SLOPE[row, col]),
        "cost": float(COST[row, col]),
    }


def _load_ppo():
    path = Path(os.environ.get("MARS_PPO_PATH", _ROOT / "models" / "mars_ppo_latest.zip"))
    if SB3PPO is None or not path.is_file():
        return None
    try:
        return SB3PPO.load(str(path))
    except Exception:
        return None


@app.websocket("/ws/rl-episode")
async def rl_episode_ws(ws: WebSocket):
    """Stream rover steps: PPO if `models/mars_ppo_latest.zip` (or MARS_PPO_PATH) exists, else heuristic."""
    from backend.rollout import blended_action, heuristic_action
    from rl.mars_env import MarsRoverEnv

    await ws.accept()
    max_steps = int(os.environ.get("MARS_RL_WS_MAX_STEPS", "800"))
    delay_s = float(os.environ.get("MARS_RL_WS_DELAY", "0.05"))
    try:
        msg = await ws.receive_json()
        start_col, start_row = int(msg["start"][0]), int(msg["start"][1])
        goal_col, goal_row = int(msg["goal"][0]), int(msg["goal"][1])

        env = MarsRoverEnv(str(_TERRAIN_PATH), _GRID, _MPP, random_start_goal=False)
        obs, _ = env.reset(
            options={
                "start_col": start_col,
                "start_row": start_row,
                "goal_col": goal_col,
                "goal_row": goal_row,
            }
        )
        model = _load_ppo()
        r0, c0 = int(start_row), int(start_col)
        await ws.send_json(
            {
                "step": 0,
                "pos": [float(start_col), float(start_row)],
                "elevation_m": float(ELEV[r0, c0]),
                "cost": float(env.grid.get_cell_cost(start_row, start_col)),
                "dist": float(np.linalg.norm(env.goal - env.pos)),
                "done": False,
                "policy": "ppo" if model is not None else "heuristic",
            }
        )

        rng = np.random.default_rng()
        for step in range(1, max_steps + 1):
            if model is not None:
                action = blended_action(model, env, obs, rng=rng)
            else:
                action = heuristic_action(env)
            obs, _reward, terminated, truncated, info = env.step(action)
            pos = info["pos"]
            ri, ci = int(pos[1]), int(pos[0])
            ri = max(0, min(TERRAIN.height - 1, ri))
            ci = max(0, min(TERRAIN.width - 1, ci))
            await ws.send_json(
                {
                    "step": step,
                    "pos": [float(pos[0]), float(pos[1])],
                    "elevation_m": float(ELEV[ri, ci]),
                    "cost": float(info["cell_cost"]),
                    "dist": float(info["dist_to_goal"]),
                    "done": bool(terminated or truncated),
                }
            )
            await asyncio.sleep(delay_s)
            if terminated or truncated:
                break
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass
