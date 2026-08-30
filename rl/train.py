"""Train PPO on MarsRoverEnv (requires built libmars on PYTHONPATH / build/)."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
import json

_ROOT = Path(__file__).resolve().parents[1]


def _default_mpp() -> float:
    meta = _ROOT / "data" / "dtm" / "jezero_meta.json"
    if meta.is_file():
        try:
            m = json.loads(meta.read_text(encoding="utf-8")).get("meters_per_pixel")
            if isinstance(m, (int, float)) and m > 0:
                return float(m)
        except (json.JSONDecodeError, OSError):
            pass
    return 1.0

for _p in (
    _ROOT,
    _ROOT / "build",
    _ROOT / "build" / "Release",
    _ROOT / "build" / "Debug",
):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from rl.mars_env import MarsRoverEnv


@dataclass(frozen=True)
class PPOConfig:
    """Defaults come from the Optuna-tuned run (see rl/optimize.py)."""

    total_timesteps: int = 50_000
    n_envs: int = 1
    learning_rate: float = 2.5e-4
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.92
    clip_range: float = 0.15
    ent_coef: float = 0.0015
    max_grad_norm: float = 0.5
    eval_freq: int = 10_000
    checkpoint_freq: int = 25_000


def train_ppo(
    config: PPOConfig,
    terrain_path: Path,
    grid_size: int,
    meters_per_pixel: float,
    output_dir: Path,
    seed: int,
    tensorboard_dir: Path | None = None,
    resume: Path | None = None,
) -> Path:
    if not terrain_path.is_file():
        raise SystemExit(f"Terrain file not found: {terrain_path}")

    checkpoints_dir = output_dir / "checkpoints"
    best_dir = output_dir / "best"
    eval_logs_dir = output_dir / "eval"
    for directory in (output_dir, checkpoints_dir, best_dir, eval_logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    def make_env():
        return MarsRoverEnv(
            str(terrain_path), grid_size, meters_per_pixel, random_start_goal=True
        )

    train_env = make_vec_env(make_env, n_envs=config.n_envs, seed=seed)
    eval_env = make_vec_env(make_env, n_envs=1, seed=seed + 1)

    if resume is not None and resume.is_file():
        model = PPO.load(
            str(resume),
            env=train_env,
            tensorboard_log=str(tensorboard_dir) if tensorboard_dir else None,
        )
        print(f"Resumed PPO from {resume}", flush=True)
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            max_grad_norm=config.max_grad_norm,
            policy_kwargs=dict(net_arch=[256, 256, 128]),
            tensorboard_log=str(tensorboard_dir) if tensorboard_dir else None,
            seed=seed,
            verbose=1,
        )

    callbacks = [
        EvalCallback(
            eval_env=eval_env,
            best_model_save_path=str(best_dir),
            log_path=str(eval_logs_dir),
            eval_freq=max(config.eval_freq // config.n_envs, 1),
            n_eval_episodes=5,
            deterministic=True,
            render=False,
        ),
        CheckpointCallback(
            save_freq=max(config.checkpoint_freq // config.n_envs, 1),
            save_path=str(checkpoints_dir),
            name_prefix="mars_ppo",
        ),
    ]

    model.learn(total_timesteps=config.total_timesteps, callback=callbacks)

    model_path = output_dir / "mars_ppo_latest"
    model.save(str(model_path))

    train_env.close()
    eval_env.close()
    return model_path.with_suffix(".zip")


def main() -> None:
    ap = argparse.ArgumentParser(description="PPO training for Mars Route Zero")
    ap.add_argument(
        "--terrain",
        type=Path,
        default=_ROOT / "data/costmap/jezero_elevation.bin",
        help="Float32 elevation binary",
    )
    ap.add_argument("--grid", type=int, default=512)
    ap.add_argument("--mpp", type=float, default=None, help="Meters per pixel (default: jezero_meta.json or 1.0)")
    ap.add_argument("--resume", type=Path, default=None, help="Continue from a PPO .zip")
    ap.add_argument("--timesteps", type=int, default=PPOConfig.total_timesteps)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Quick dev run (~8k steps, small batch)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=_ROOT / "models",
        help="Output directory (final zip, checkpoints/, best/, eval/)",
    )
    ap.add_argument("--n-envs", type=int, default=PPOConfig.n_envs)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--learning-rate", type=float, default=PPOConfig.learning_rate)
    ap.add_argument("--n-steps", type=int, default=PPOConfig.n_steps)
    ap.add_argument("--batch-size", type=int, default=PPOConfig.batch_size)
    ap.add_argument("--n-epochs", type=int, default=PPOConfig.n_epochs)
    ap.add_argument("--gamma", type=float, default=PPOConfig.gamma)
    ap.add_argument("--gae-lambda", type=float, default=PPOConfig.gae_lambda)
    ap.add_argument("--clip-range", type=float, default=PPOConfig.clip_range)
    ap.add_argument("--ent-coef", type=float, default=PPOConfig.ent_coef)
    ap.add_argument("--eval-freq", type=int, default=PPOConfig.eval_freq)
    ap.add_argument("--checkpoint-freq", type=int, default=PPOConfig.checkpoint_freq)
    ap.add_argument(
        "--tensorboard",
        type=Path,
        nargs="?",
        const=_ROOT / "logs",
        default=None,
        metavar="DIR",
        help="Log to TensorBoard (requires: pip install tensorboard). Default DIR: ./logs",
    )
    args = ap.parse_args()

    config = PPOConfig(
        total_timesteps=8_192 if args.smoke else args.timesteps,
        n_envs=args.n_envs,
        learning_rate=args.learning_rate,
        n_steps=512 if args.smoke else args.n_steps,
        batch_size=128 if args.smoke else args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        eval_freq=args.eval_freq,
        checkpoint_freq=args.checkpoint_freq,
    )

    mpp = args.mpp if args.mpp is not None else _default_mpp()

    model_zip = train_ppo(
        config=config,
        terrain_path=args.terrain,
        grid_size=args.grid,
        meters_per_pixel=mpp,
        output_dir=args.out_dir,
        seed=args.seed,
        tensorboard_dir=args.tensorboard,
        resume=args.resume,
    )
    print(f"Saved final PPO model: {model_zip}")


if __name__ == "__main__":
    main()
