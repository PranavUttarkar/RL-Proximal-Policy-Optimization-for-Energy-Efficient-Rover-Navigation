"""Gymnasium environment wrapping libmars (build C++ extension first)."""
from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

_RL_ROOT = Path(__file__).resolve().parents[1]
if str(_RL_ROOT) not in sys.path:
    sys.path.insert(0, str(_RL_ROOT))

from dll_windows import ensure_mingw_dll_dirs

ensure_mingw_dll_dirs(_RL_ROOT)

import libmars
from rl.physics_energy import segment_energy_kj


class MarsRoverEnv(gym.Env):
    """Mars rover navigation on HiRISE-derived elevation grid (via libmars costmap)."""

    metadata = {"render_modes": ["rgb_array"]}

    PATCH_SIZE = 15
    MAX_STEPS = 2000
    GOAL_RADIUS = 3
    # Fallback when meters_per_pixel is missing; real DTM uses ~1 cell per full-speed step.
    STEP_SIZE_M = 2.0
    HEADING_LIMIT = 30.0
    # Match PPO gamma so potential-based shaping does not distort the objective (Ng et al.).
    GAMMA_SHAPING = 0.99
    # Moves into cells at/above this cost are rejected (stay put, penalty) — discourages rim-loops into craters.
    BLOCK_COST = 0.78
    # Episode ends if the rover is in unsurvivable terrain (after a successful move).
    TERMINATE_COST = 0.92

    # Physics energy (kJ) → reward. ~0.01 × ~20 kJ/step keeps this O(0.2) vs goal +100.
    ENERGY_STEP_PENALTY_SCALE = 0.01

    # "Mountain" / "valley" penalties are based on local elevation contrast.
    # Values are normalized by the elevation range seen in this terrain grid.
    REL_RIDGE_THRESH_NORM = 0.05   # center above neighbor mean by this fraction -> ridge
    REL_VALLEY_THRESH_NORM = 0.05  # center below neighbor mean by this fraction -> valley
    TERRAIN_REL_PENALTY_SCALE = 12.0

    # Extra penalty for steep uphill/downhill transitions during a single move.
    STEEP_SLOPE_THRESH_DEG = 14.0
    STEEP_SLOPE_PENALTY_SCALE = 0.003

    def __init__(
        self,
        terrain_path: str,
        grid_size: int = 512,
        mpp: float = 1.0,
        random_start_goal: bool = True,
    ):
        super().__init__()
        self.grid = libmars.build_full_terrain(terrain_path, grid_size, grid_size, mpp)
        self.W = self.grid.width
        self.H = self.grid.height
        self._max_diag = float(np.hypot(self.W, self.H))
        self.random_start_goal = random_start_goal
        self.meters_per_pixel = float(getattr(self.grid, "meters_per_pixel", mpp) or mpp)
        # Grid kinematics: ~1.2 cells at full throttle (512-scale goals stay reachable).
        self.step_cells = 1.2

        # Cache elevation/cost maps so reward shaping can be computed cheaply.
        # shapes: (H, W)
        self.cost_map = np.asarray(self.grid.get_costmap_array(), dtype=np.float32)
        self.elev_map = np.asarray(self.grid.get_elevation_array(), dtype=np.float32)
        self.elev_min = float(self.elev_map.min())
        self.elev_max = float(self.elev_map.max())
        self.elev_range = max(1e-6, self.elev_max - self.elev_min)

        patch_cells = self.PATCH_SIZE * self.PATCH_SIZE
        obs_dim = patch_cells + 5
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )
        self._reset_state()

    def _reset_state(self):
        self.pos = np.array([self.W // 2, self.H // 2], dtype=float)
        self.goal = np.array([self.W * 3 // 4, self.H * 3 // 4], dtype=float)
        self.heading_deg = 0.0
        self.step_count = 0
        self.trajectory = [self.pos.copy()]
        self._visited: set[tuple[int, int]] = set()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        # Fixed start/goal (e.g. WebSocket demo): col, row order matches planner / libmars
        if "start_col" in options and "goal_col" in options:
            self.pos = np.array(
                [float(options["start_col"]), float(options["start_row"])], dtype=float
            )
            self.goal = np.array(
                [float(options["goal_col"]), float(options["goal_row"])], dtype=float
            )
        elif self.random_start_goal:
            margin = self.PATCH_SIZE
            rng = self.np_random
            while True:
                r = int(rng.integers(margin, self.H - margin))
                c = int(rng.integers(margin, self.W - margin))
                if self.grid.get_cell_cost(r, c) < 0.5:
                    self.pos = np.array([c, r], dtype=float)
                    break
            while True:
                r = int(rng.integers(margin, self.H - margin))
                c = int(rng.integers(margin, self.W - margin))
                dist = np.linalg.norm(np.array([c, r]) - self.pos)
                if self.grid.get_cell_cost(r, c) < 0.5 and dist > 50:
                    self.goal = np.array([c, r], dtype=float)
                    break
        else:
            self._reset_state()

        goal_vec = self.goal - self.pos
        default_heading = float(np.degrees(np.arctan2(goal_vec[1], goal_vec[0])) % 360.0)
        self.heading_deg = float(options.get("heading_deg", default_heading))
        self.step_count = 0
        self.trajectory = [self.pos.copy()]
        self._visited = set()
        self.episode_energy_kj = 0.0
        return self._get_obs(), {}

    def step(self, action):
        steer = float(np.clip(action[0], -1.0, 1.0))
        speed_f = float(np.clip(action[1], 0.0, 1.0))
        delta_h = steer * self.HEADING_LIMIT

        self.heading_deg += delta_h
        self.heading_deg = self.heading_deg % 360.0

        rad = np.deg2rad(self.heading_deg)
        step_cells = self.step_cells * (0.3 + 0.7 * speed_f)
        dx = np.cos(rad) * step_cells
        dy = np.sin(rad) * step_cells

        old_pos = self.pos.copy()
        old_col = int(old_pos[0])
        old_row = int(old_pos[1])
        old_cost = float(self.cost_map[old_row, old_col])
        old_elev = float(self.elev_map[old_row, old_col])

        phi_old = -float(np.linalg.norm(self.goal - self.pos)) / (self._max_diag + 1e-8)

        new_pos = self.pos + np.array([dx, dy], dtype=float)
        pad = self.PATCH_SIZE // 2
        new_pos[0] = np.clip(new_pos[0], pad, self.W - pad - 1)
        new_pos[1] = np.clip(new_pos[1], pad, self.H - pad - 1)

        nrow, ncol = int(new_pos[1]), int(new_pos[0])
        proposed_cost = float(self.cost_map[nrow, ncol])
        blocked = proposed_cost >= self.BLOCK_COST
        if blocked:
            new_pos = self.pos.copy()

        self.pos = new_pos
        self.trajectory.append(self.pos.copy())
        self.step_count += 1

        col, row = int(self.pos[0]), int(self.pos[1])
        cell_cost = float(self.cost_map[row, col])
        cell_elev = float(self.elev_map[row, col])

        dist_to_goal = float(np.linalg.norm(self.goal - self.pos))
        phi_new = -dist_to_goal / (self._max_diag + 1e-8)
        dist_norm = dist_to_goal / (self._max_diag + 1e-8)

        dist_cells = float(np.linalg.norm(self.pos - old_pos))
        dist_m = dist_cells * self.meters_per_pixel
        dist_scale = dist_cells / (self.step_cells + 1e-6)
        step_energy_kj = 0.0
        if not blocked and dist_cells > 1e-8:
            step_energy_kj = segment_energy_kj(
                self.elev_map,
                float(old_pos[1]),
                float(old_pos[0]),
                float(self.pos[1]),
                float(self.pos[0]),
                self.meters_per_pixel,
            )
            self.episode_energy_kj += step_energy_kj

        # Steep per-step transition penalty (mountain crossings + deep descents).
        step_slope_deg = float(
            np.degrees(np.arctan2(abs(cell_elev - old_elev), dist_m + 1e-6))
        )
        steep_slope_pen = (
            self.STEEP_SLOPE_PENALTY_SCALE
            * max(0.0, step_slope_deg - self.STEEP_SLOPE_THRESH_DEG) ** 2
            * dist_scale
        )

        # Local "ridge vs valley" detector: center elevation vs 8-neighbor mean.
        elev_nb = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr = int(np.clip(row + dr, 0, self.H - 1))
                cc = int(np.clip(col + dc, 0, self.W - 1))
                elev_nb.append(float(self.elev_map[rr, cc]))
        nb_mean = float(np.mean(elev_nb)) if elev_nb else cell_elev
        rel_norm = (cell_elev - nb_mean) / self.elev_range
        ridge_delta = max(0.0, rel_norm - self.REL_RIDGE_THRESH_NORM)
        valley_delta = max(0.0, -rel_norm - self.REL_VALLEY_THRESH_NORM)
        terrain_rel_pen = (
            self.TERRAIN_REL_PENALTY_SCALE * (ridge_delta**2 + valley_delta**2) * dist_scale
        )

        goal_vec = self.goal - self.pos
        gdist = float(np.linalg.norm(goal_vec)) + 1e-8
        gdir = goal_vec / gdist
        hdir = np.array([np.cos(rad), np.sin(rad)], dtype=np.float64)
        heading_align = float(np.dot(gdir, hdir))

        reward = 0.0
        # Potential-based shaping toward goal (γ matches PPO; no net gain from a pure spatial loop).
        if not blocked:
            reward += self.GAMMA_SHAPING * phi_new - phi_old
            # Discourage figure-eights: cheap terrain cannot pay for covering the same cells repeatedly.
            cell_key = (row, col)
            if cell_key in self._visited:
                reward -= 0.028
            self._visited.add(cell_key)
        # Only reward "face the goal" when it comes with real progress (potential increased).
        if not blocked and phi_new > phi_old + 1e-5:
            reward += 0.045 * max(0.0, heading_align) * (0.2 + 0.8 * speed_f)
        # Physics energy (rolling + uphill PE). Hazard costmap stays a hard block, not the score.
        reward -= self.ENERGY_STEP_PENALTY_SCALE * step_energy_kj
        reward -= terrain_rel_pen
        reward -= steep_slope_pen
        reward -= 0.035 * abs(steer)
        reward -= 0.018
        if blocked:
            reward -= 0.55

        terminated = False
        truncated = False

        if dist_to_goal < self.GOAL_RADIUS:
            reward += 100.0
            terminated = True
        elif cell_cost >= self.TERMINATE_COST:
            reward -= 25.0
            terminated = True
        elif self.step_count >= self.MAX_STEPS:
            reward -= 10.0 + 12.0 * dist_norm
            truncated = True

        return (
            self._get_obs(),
            reward,
            terminated,
            truncated,
            {
                "cell_cost": cell_cost,
                "dist_to_goal": dist_to_goal,
                "pos": self.pos.copy(),
                "trajectory": self.trajectory.copy(),
                "energy_kj": float(self.episode_energy_kj),
                "step_energy_kj": float(step_energy_kj),
            },
        )

    def _get_obs(self) -> np.ndarray:
        half = self.PATCH_SIZE // 2
        col, row = int(self.pos[0]), int(self.pos[1])

        patch = np.zeros((self.PATCH_SIZE, self.PATCH_SIZE), dtype=np.float32)
        for dr in range(-half, half + 1):
            for dc in range(-half, half + 1):
                r2, c2 = row + dr, col + dc
                if 0 <= r2 < self.H and 0 <= c2 < self.W:
                    patch[dr + half, dc + half] = self.grid.get_cell_cost(r2, c2)
                else:
                    patch[dr + half, dc + half] = 1.0

        goal_vec = self.goal - self.pos
        goal_dist = np.linalg.norm(goal_vec)
        goal_dir = goal_vec / (goal_dist + 1e-8)
        dist_norm = goal_dist / (np.sqrt(self.W**2 + self.H**2))
        heading_rad = np.deg2rad(self.heading_deg)

        context = np.array(
            [
                goal_dir[0],
                goal_dir[1],
                dist_norm,
                np.cos(heading_rad),
                np.sin(heading_rad),
            ],
            dtype=np.float32,
        )

        return np.concatenate([patch.flatten(), context])
