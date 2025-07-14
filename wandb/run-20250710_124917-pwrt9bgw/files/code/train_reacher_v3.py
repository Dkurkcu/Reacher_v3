# Minimal WandB Logging PPO Script for ReacherV3 with VecNormalize

import os
import wandb
from reacher_v3 import ReacherV3Env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit
from wandb.integration.sb3 import WandbCallback

# === WANDB Setup ===
wandb.init(
    project="reacher-v3",
    name="ppo-minimal-logging",
    config={
        "policy_type": "MlpPolicy",
        "total_timesteps": 200_000,
        "learning_rate": 1e-4,
        "ent_coef": 0.02,
        "clip_range": 0.1,
        "batch_size": 128,
        "normalize_obs": True,
        "normalize_reward": True
    },
    sync_tensorboard=True,
    monitor_gym=False,   #  Skip system resource logging
    save_code=True
)

# === Env Function ===
def make_env():
    env = ReacherV3Env("reacher_v3.xml")
    env = TimeLimit(env, max_episode_steps=200)
    env = Monitor(env)
    return env

# === Create and Normalize Environment ===
venv = DummyVecEnv([make_env])
venv = VecNormalize(venv, norm_obs=True, norm_reward=True)

# === PPO Agent ===
model = PPO(
    "MlpPolicy",
    venv,
    verbose=1,
    tensorboard_log="./ppo_tensorboard/",
    learning_rate=1e-4,
    ent_coef=0.02,
    clip_range=0.1,
    batch_size=128
)

# === Train PPO with Minimal WandB Logging ===
model.learn(
    total_timesteps=200_000,
    callback=WandbCallback(
        gradient_save_freq=0,        #  No gradient tracking
        model_save_freq=0,           #  No model saving during training
        verbose=1,
        log=None                     #  No media (video/image) logging
    )
)

# === Save final model and normalization statistics ===
model.save("ppo_reacher_v3")
venv.save("vec_normalize.pkl")
wandb.finish()
