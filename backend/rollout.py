"""Policy rollout helpers for WebSocket (PPO or goal-seeking heuristic)."""
from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np


def heuristic_action(env: Any) -> np.ndarray:
    """Steer toward goal with moderate speed; works without a trained network."""
    goal_vec = env.goal - env.pos
    target_deg = float(np.degrees(np.arctan2(goal_vec[1], goal_vec[0])) % 360.0)
    err = (target_deg - env.heading_deg + 180.0) % 360.0 - 180.0
    delta = float(np.clip(err / env.HEADING_LIMIT, -1.0, 1.0))
    return np.array([delta, 0.88], dtype=np.float32)


def predict_action(model: Any, obs: np.ndarray) -> np.ndarray:
    """SB3 PPO.predict wrapper."""
    action, _ = model.predict(obs, deterministic=True)
    return np.asarray(action, dtype=np.float32).reshape(-1)


def blended_action(
    model: Any,
    env: Any,
    obs: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Mix learned policy with goal-directed heuristic.

    Blend weight rises when closer to the goal (finer docking) and can include a
    stochastic draw toward the heuristic to break limit cycles (env-tunable).
    """
    ppo = predict_action(model, obs)
    heur = heuristic_action(env)
    if rng is None:
        rng = np.random.default_rng()

    base = float(os.environ.get("MARS_RL_GOAL_BLEND", "0.45"))
    d = float(np.linalg.norm(env.goal - env.pos))
    dmax = float(np.hypot(env.W, env.H))
    # Stronger goal bias near the endpoint; keep more policy when far (terrain reasoning).
    near = 1.0 - min(d / max(dmax * 0.32, 1e-6), 1.0)
    alpha = min(1.0, base * (0.2 + 0.8 * near))

    stoch = float(os.environ.get("MARS_RL_GOAL_STOCH", "0.14"))
    if stoch > 0.0 and rng.random() < stoch * (0.35 + 0.65 * near):
        return heur

    out = (1.0 - alpha) * ppo + alpha * heur
    return np.clip(out.astype(np.float32), np.array([-1.0, 0.0]), np.array([1.0, 1.0]))
