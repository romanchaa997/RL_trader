# RL_trader — DQN + PPO Trading Bot for Bitcoin

> **Fork of [ShayanJa/RL_trader](https://github.com/ShayanJa/RL_trader)**  
> Extended with **AuditorSEC / prodoRIG** integration layer:
> PPO via stable-baselines3, Optuna Bayesian HPO, Qiskit VQC+PSO hooks,
> and ML-KEM-768 PQC-secure data-feed stubs.

---

## Architecture Overview

```
asset_prices/btcusd.csv
        │
        ▼
   features.py          ← RSI + MACD + daily_returns (ta library)
        │
     ┌──┴─────────────┐
        │              │
  environment.py    ppo_learner.py
  (TF-Agents DQN)   (SB3 PPO + Optuna HPO)
        │              │
   learner.py       ► results/ppo_btcusdt.*
  (DQN training)    Sharpe Ratio metric
        │
   trader.py   ← Live Binance scheduler (60s steps)
```

---

## Agents

| File | Algorithm | Library | Notes |
|---|---|---|---|
| `learner.py` | **DQN** | tf_agents 0.19 | Original agent; stable in choppy markets |
| `ppo_learner.py` | **PPO** | stable-baselines3 2.3 | Higher returns in bull phases; Optuna HPO |
| `trader.py` | DQN live | tf_agents | Binance scheduler, 60-second intervals |

---

## Install

```bash
git clone https://github.com/romanchaa997/RL_trader.git
cd RL_trader
git checkout auditor-sec-integration
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in Binance API keys:
```bash
cp .env.example .env
# edit .env and set CLIENT_KEY and SECRET_KEY
```

---

## DQN Training (original)

```bash
# Download BTC price data from Binance
CLIENT_KEY=XXXXX CLIENT_SECRET=XXXXX \
  python scripts/binance-prices.py price BTCUSDT

# Train (edit hyperparams in learner.py first)
python learner.py
```

**Key hyperparameters in `learner.py`:**

| Param | Default | Description |
|---|---|---|
| `num_iterations` | 10 000 | Total training steps |
| `fc_layer_params` | (100, 23, 33) | Q-network dense layers |
| `learning_rate` | 1e-4 | Adam LR |
| `batch_size` | 100 | Replay batch |
| `features_length` | 20 | Lookback window |
| `date_split` | 2020-01-01 | Train / test cutoff |

---

## PPO Training + Bayesian HPO (AuditorSEC)

```bash
# Default training (200 000 steps, no HPO)
python ppo_learner.py

# With Optuna Bayesian HPO (30 trials, Sharpe-ratio objective)
python ppo_learner.py --optimize

# Custom step count
python ppo_learner.py --steps 500000
```

Optuna searches over: `lr`, `n_steps`, `batch_size`, `gamma`, `ent_coef`, `net_arch`  
Objective: **annualised Sharpe ratio** on the test set.

Results saved to `results/`:
- `ppo_btcusdt.zip` — trained model
- `ppo_btcusdt.png` — equity curve vs Buy-and-Hold

---

## Live Trading

```bash
export CLIENT_KEY=XXXXX
export SECRET_KEY=XXXXX
python trader.py          # requires saved DQN policy in policy_10000/
```

> **Warning:** Live trading carries real financial risk.
> Always test on Binance Testnet first.

---

## AuditorSEC / prodoRIG Integration Roadmap

| Module | Status | Description |
|---|---|---|
| `requirements.txt` | Done | Updated deps: SB3, Optuna, Qiskit, cryptography |
| `ppo_learner.py` | Done | PPO + Gymnasium wrapper + Optuna HPO |
| `vqc_optimizer.py` | Planned | VQC+PSO portfolio weight optimizer (Qiskit) |
| `mlkem_feed.py` | Planned | ML-KEM-768 encrypted data-feed wrapper |
| `n8n_webhook.py` | Planned | n8n/ClickUp trade-signal dispatcher |

---

## Improvements (upstream + AuditorSEC additions)

- Fine-tune DQN/PPO hyperparameters via Optuna
- Improve reward function (Sortino, Calmar ratio)
- Add sentiment analysis (NLP news feed) to state vector
- Risk management module (max drawdown stop-loss)
- VQE ansatz (TwoLocal / Dicke) for DAO weight optimization
- Quantum annealing (D-Wave Ocean SDK) for cross-exchange arbitrage
- FIPS-203 compliance via ML-KEM-768 secure feeds

---

## References

- [Q-Learning — Wikipedia](https://en.wikipedia.org/wiki/Q-learning)
- [tf-agents DQN Tutorial](https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial)
- [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Qiskit Machine Learning](https://qiskit-community.github.io/qiskit-machine-learning/)
