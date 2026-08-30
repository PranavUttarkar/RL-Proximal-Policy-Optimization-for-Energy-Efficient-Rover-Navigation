"""JAX rover-drive energy on a DEM (shared by env reward and A*/RL eval).

A* still *plans* on the hazard/slope costmap. Scoring uses a Perseverance-class
mechanical model instead of integrating that costmap:

    E = m g [ C_rr * s  +  Σ max(0, Δh)  +  k_grade * s * sin²(θ_up) ]

- Rolling on Mars regolith dominates long drives (shorter tracks win).
- Gravity is charged only on climbs; no downhill regeneration (solar/RTG).
- Elevation is bilinear, so a continuous RL polyline is not stair-stepped
  the way an 8-connected A* cell-center chain is.

Units: kilojoules. JAX is the source of truth; a NumPy twin matches it for
per-step env calls without paying JIT on 2-point segments.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# Perseverance-class constants (public JPL figures, rounded).
G_MARS = 3.721  # m/s^2
MASS_KG = 1025.0
# Wheel-soil rolling resistance on unconsolidated Mars regolith (literature
# range ~0.15–0.3). Mid-high value is physically real and distance-sensitive.
C_RR = 0.24
# Extra motor work on steep *uphill* only (sin² θ); small vs rolling+climb.
K_GRADE = 0.28

try:
    import jax
    import jax.numpy as jnp

    _HAS_JAX = True
except ImportError:  # pragma: no cover
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    _HAS_JAX = False


def _bilinear_np(elev: np.ndarray, row: np.ndarray, col: np.ndarray) -> np.ndarray:
    h, w = elev.shape
    r0 = np.clip(np.floor(row), 0, h - 2)
    c0 = np.clip(np.floor(col), 0, w - 2)
    dr = np.clip(row - r0, 0.0, 1.0)
    dc = np.clip(col - c0, 0.0, 1.0)
    r0i = r0.astype(np.int32)
    c0i = c0.astype(np.int32)
    r1 = r0i + 1
    c1 = c0i + 1
    v00 = elev[r0i, c0i]
    v01 = elev[r0i, c1]
    v10 = elev[r1, c0i]
    v11 = elev[r1, c1]
    return (1.0 - dr) * ((1.0 - dc) * v00 + dc * v01) + dr * ((1.0 - dc) * v10 + dc * v11)


def _segment_terms_np(
    elev: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    mpp: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-segment (ds_m, climb_m, grade_term)."""
    r0, r1 = rows[:-1], rows[1:]
    c0, c1 = cols[:-1], cols[1:]
    e0 = _bilinear_np(elev, r0, c0)
    e1 = _bilinear_np(elev, r1, c1)
    ds = mpp * np.hypot(r1 - r0, c1 - c0)
    dh = e1 - e0
    climb = np.maximum(dh, 0.0)
    sin_th = climb / (ds + 1e-6)
    grade = ds * (sin_th * sin_th)
    return ds, climb, grade


def path_energy_kj_numpy(elev: np.ndarray, rows: np.ndarray, cols: np.ndarray, mpp: float) -> dict[str, float]:
    rows = np.asarray(rows, dtype=np.float64).reshape(-1)
    cols = np.asarray(cols, dtype=np.float64).reshape(-1)
    if rows.size < 2:
        return {"energy_kj": 0.0, "distance_m": 0.0, "climb_m": 0.0, "rolling_kj": 0.0, "climb_kj": 0.0}
    ds, climb, grade = _segment_terms_np(elev, rows, cols, float(mpp))
    mg = MASS_KG * G_MARS
    rolling_j = mg * C_RR * float(ds.sum())
    climb_j = mg * float(climb.sum())
    grade_j = mg * K_GRADE * float(grade.sum())
    energy_j = rolling_j + climb_j + grade_j
    return {
        "energy_kj": energy_j / 1000.0,
        "distance_m": float(ds.sum()),
        "climb_m": float(climb.sum()),
        "rolling_kj": rolling_j / 1000.0,
        "climb_kj": climb_j / 1000.0,
        "grade_kj": grade_j / 1000.0,
    }


def segment_energy_kj(
    elev: np.ndarray,
    r0: float,
    c0: float,
    r1: float,
    c1: float,
    mpp: float,
) -> float:
    """Physics energy of one move, kJ (NumPy; same formula as the JAX path scorer)."""
    out = path_energy_kj_numpy(elev, np.array([r0, r1]), np.array([c0, c1]), mpp)
    return float(out["energy_kj"])


if _HAS_JAX:

    def _bilinear_jax(elev, row, col):
        h, w = elev.shape
        r0 = jnp.clip(jnp.floor(row), 0, h - 2)
        c0 = jnp.clip(jnp.floor(col), 0, w - 2)
        dr = jnp.clip(row - r0, 0.0, 1.0)
        dc = jnp.clip(col - c0, 0.0, 1.0)
        r0i = r0.astype(jnp.int32)
        c0i = c0.astype(jnp.int32)
        v00 = elev[r0i, c0i]
        v01 = elev[r0i, c0i + 1]
        v10 = elev[r0i + 1, c0i]
        v11 = elev[r0i + 1, c0i + 1]
        return (1.0 - dr) * ((1.0 - dc) * v00 + dc * v01) + dr * ((1.0 - dc) * v10 + dc * v11)

    def _path_energy_jax(elev, rows, cols, mpp):
        r0, r1 = rows[:-1], rows[1:]
        c0, c1 = cols[:-1], cols[1:]
        e0 = _bilinear_jax(elev, r0, c0)
        e1 = _bilinear_jax(elev, r1, c1)
        ds = mpp * jnp.hypot(r1 - r0, c1 - c0)
        climb = jnp.maximum(e1 - e0, 0.0)
        sin_th = climb / (ds + 1e-6)
        grade = ds * (sin_th * sin_th)
        mg = MASS_KG * G_MARS
        energy_j = mg * (C_RR * ds.sum() + climb.sum() + K_GRADE * grade.sum())
        return energy_j / 1000.0, ds.sum(), climb.sum()

    _path_energy_jit = jax.jit(_path_energy_jax)

    def path_energy_kj(elev: Any, rows: Any, cols: Any, mpp: float) -> dict[str, float]:
        rows_np = np.asarray(rows, dtype=np.float32).reshape(-1)
        cols_np = np.asarray(cols, dtype=np.float32).reshape(-1)
        if rows_np.size < 2:
            return {"energy_kj": 0.0, "distance_m": 0.0, "climb_m": 0.0}
        e_kj, dist, climb = _path_energy_jit(
            jnp.asarray(elev, dtype=jnp.float32),
            jnp.asarray(rows_np, dtype=jnp.float32),
            jnp.asarray(cols_np, dtype=jnp.float32),
            jnp.float32(mpp),
        )
        return {
            "energy_kj": float(e_kj),
            "distance_m": float(dist),
            "climb_m": float(climb),
        }

else:

    def path_energy_kj(elev: Any, rows: Any, cols: Any, mpp: float) -> dict[str, float]:
        return path_energy_kj_numpy(elev, rows, cols, mpp)
