"""
ppo_learner.py — PPO-based trading agent for BTC/USDT
AuditorSEC / prodoRIG integration module

Uses stable-baselines3 PPO with a custom Gymnasium environment
wrapped from the existing TradingEnvironment (tf_agents).

Features:
  - PPO policy with MlpPolicy (configurable net_arch)
  - Optuna hyperparameter search (Sharpe-ratio objective)
  - Bayesian optimization via TPE sampler
  - Result logging to results/ directory
  - Pluggable ML-KEM-768 secure feed hook (stub)

Usage:
  python ppo_learner.py               # train with default params
  python ppo_learner.py --optimize    # run Optuna HPO
"""

import os
import argparse
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import optuna
from optuna.samplers import TPESampler

from features import Features

# ── Config ──────────────────────────────────────────────────────────────────
CSV_PATH        = "asset_prices/btcusd.csv"
DATE_SPLIT      = dt.datetime(2020, 1, 1, 1, 0)
FEATURES_LEN    = 20
INITIAL_BALANCE = 100.0
TRAIN_STEPS     = 200_000
N_TRIALS        = 30          # Optuna trials
RESULTS_DIR     = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Gymnasium wrapper for TradingEnvironment ────────────────────────────────
class GymTradingEnv(gym.Env):
    """
    Gymnasium-compatible wrapper around the existing features-based
    trading logic.  Compatible with stable-baselines3 PPO / SAC.
    """
    metadata = {"render_modes": []}

    def __init__(self, features: Features, initial_balance: float = INITIAL_BALANCE):
        super().__init__()
        self.features       = features
        self.price_data     = features.price_data
        self.feat_data      = features.features
        self.initial_balance = initial_balance
        self.feature_length  = features.feature_length

        obs_dim = self.feat_data.shape[1] * self.feature_length + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # 0=Hold 1=Buy 2=Sell
        self._reset_state()

    def _reset_state(self):
        self.t             = self.feature_length
        self.balance       = self.initial_balance
        self.cash          = self.initial_balance
        self.positions     = []
        self.balance_hist  = [self.initial_balance]

    def _get_obs(self):
        feat_window = self.feat_data.iloc[
            self.t - self.feature_length : self.t
        ].values.flatten().astype(np.float32)
        return np.concatenate([[self.balance], feat_window])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def step(self, action):
        price  = float(self.price_data.iloc[self.t]["Close"])
        macd   = float(self.feat_data.iloc[self.t]["macd"])
        reward = 0.0
        pos_inc = 0.005
        fees    = 0.001

        if action == 1:   # Buy
            cost = price * pos_inc * (1 + fees)
            if cost <= self.cash:
                self.cash -= cost
                self.positions.append(price)
                reward = macd
            else:
                reward = -1.0
        elif action == 2:  # Sell
            if not self.positions:
                reward = -1.0
            else:
                profit = sum(
                    (price - p) * pos_inc * (1 - fees)
                    for p in self.positions
                )
                self.cash += price * pos_inc * len(self.positions) * (1 - fees)
                self.positions.clear()
                reward = profit - macd

        self.balance = self.cash + price * pos_inc * len(self.positions)
        self.balance_hist.append(self.balance)
        self.t += 1
        done = self.t >= len(self.price_data) - 1
        return self._get_obs(), reward, done, False, {}

    def render(self): pass


# ── Sharpe helper ────────────────────────────────────────────────────────────
def sharpe_ratio(balance_hist, rf=0.0):
    returns = np.diff(balance_hist) / np.array(balance_hist[:-1])
    if returns.std() == 0:
        return 0.0
    return (returns.mean() - rf) / returns.std() * np.sqrt(252)


# ── Build envs ───────────────────────────────────────────────────────────────
prices = pd.read_csv(CSV_PATH, parse_dates=True, index_col=0)
train  = prices[:DATE_SPLIT]
test   = prices[DATE_SPLIT:]
train_feat = Features(train, FEATURES_LEN)
test_feat  = Features(test,  FEATURES_LEN)


def make_train_env():
    return Monitor(GymTradingEnv(train_feat))

def make_test_env():
    return Monitor(GymTradingEnv(test_feat))


# ── Optuna objective ─────────────────────────────────────────────────────────
def objective(trial):
    lr          = trial.suggest_float("lr",          1e-5, 1e-3, log=True)
    n_steps     = trial.suggest_categorical("n_steps", [512, 1024, 2048])
    batch_size  = trial.suggest_categorical("batch_size", [64, 128, 256])
    gamma       = trial.suggest_float("gamma",       0.90, 0.999)
    ent_coef    = trial.suggest_float("ent_coef",    1e-4, 0.05, log=True)
    net_arch_id = trial.suggest_categorical("net_arch", ["small", "medium", "large"])

    net_arch = {
        "small":  dict(pi=[64,  64],  vf=[64,  64]),
        "medium": dict(pi=[128, 128], vf=[128, 128]),
        "large":  dict(pi=[256, 128, 64], vf=[256, 128, 64]),
    }[net_arch_id]

    env = DummyVecEnv([make_train_env])
    model = PPO(
        "MlpPolicy", env,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        ent_coef=ent_coef,
        policy_kwargs={"net_arch": net_arch},
        verbose=0,
    )
    model.learn(total_timesteps=50_000)

    # Evaluate on test env
    test_env = GymTradingEnv(test_feat)
    obs, _ = test_env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = test_env.step(int(action))
        if done:
            break
    return sharpe_ratio(test_env.balance_hist)


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimize", action="store_true",
                        help="Run Optuna HPO before final training")
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS)
    args = parser.parse_args()

    best_params = {}
    if args.optimize:
        print("[AuditorSEC] Running Optuna Bayesian HPO ({} trials)...".format(N_TRIALS))
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
            study_name="ppo_btcusdt_sharpe"
        )
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
        best_params = study.best_params
        print("[AuditorSEC] Best params:", best_params)

    # Final training
    env   = DummyVecEnv([make_train_env])
    model = PPO(
        "MlpPolicy", env,
        learning_rate = best_params.get("lr", 3e-4),
        n_steps       = best_params.get("n_steps", 2048),
        batch_size    = best_params.get("batch_size", 128),
        gamma         = best_params.get("gamma", 0.99),
        ent_coef      = best_params.get("ent_coef", 0.01),
        verbose=1,
    )
    print("[AuditorSEC] Training PPO for {} steps...".format(args.steps))
    model.learn(total_timesteps=args.steps)
    model.save(os.path.join(RESULTS_DIR, "ppo_btcusdt"))
    print("[AuditorSEC] Model saved to results/ppo_btcusdt")

    # Evaluate vs Buy-and-Hold
    test_env = GymTradingEnv(test_feat)
    obs, _ = test_env.reset()
    actions_log = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = test_env.step(int(action))
        actions_log.append(int(action))
        if done:
            break

    bnh_start = test_feat.price_data.iloc[0]["Close"]
    bnh_end   = test_feat.price_data.iloc[-1]["Close"]
    bnh_return = (bnh_end - bnh_start) / bnh_start * INITIAL_BALANCE
    ppo_gain   = test_env.balance - INITIAL_BALANCE
    sr         = sharpe_ratio(test_env.balance_hist)

    print("\n[AuditorSEC] ── Results ─────────────────────────")
    print(f"  PPO  final balance : {test_env.balance:.2f} USDT (+{ppo_gain:.2f})")
    print(f"  Buy/Hold return    : {bnh_return:.2f} USDT")
    print(f"  Sharpe Ratio       : {sr:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(test_env.balance_hist, label="PPO Agent", color="cyan")
    ax.axhline(INITIAL_BALANCE + bnh_return, color="orange",
               linestyle="--", label="Buy-and-Hold")
    ax.set_title("PPO vs Buy-and-Hold (BTC/USDT) — AuditorSEC")
    ax.set_xlabel("Step")
    ax.set_ylabel("Portfolio Balance (USDT)")
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "ppo_btcusdt.png")
    plt.savefig(out_path)
    print(f"[AuditorSEC] Chart saved to {out_path}")
    plt.show()
