"""
vqc_optimizer.py — VQC+PSO Quantum Portfolio Optimizer
AuditorSEC / prodoRIG integration module

Uses Qiskit VQC (Variational Quantum Classifier) + Particle Swarm
Optimization (PSO) to compute optimal portfolio weights for multi-asset
trading (e.g., BTC, ETH, SOL).

Features:
  - TwoLocal ansatz (RY/RZ rotations + CZ entanglement)
  - COBYLA classical optimizer
  - Sharpe ratio / Sortino ratio fitness function
  - Fallback to classical PSO if quantum backend unavailable
  - STANAG/LOI compliance hooks (stub)

Usage:
  python vqc_optimizer.py --assets BTC ETH SOL
"""

import argparse
import numpy as np
import pandas as pd
from typing import List, Dict
import warnings
warnings.filterwarnings("ignore")

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import TwoLocal, ZZFeatureMap
    from qiskit.primitives import Sampler
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_machine_learning.algorithms import VQC
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("[VQC] Warning: Qiskit not available, fallback to classical PSO")

# ── PSO Fallback ──────────────────────────────────────────────────────────────
class ParticleSwarmOptimizer:
    """
    Classical PSO for portfolio weight optimization.
    Used when Qiskit is unavailable or as baseline comparison.
    """
    def __init__(self, n_particles=20, n_iterations=50, w=0.7, c1=1.5, c2=1.5):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w  = w   # inertia
        self.c1 = c1  # cognitive
        self.c2 = c2  # social

    def optimize(self, fitness_fn, dim):
        """
        fitness_fn: callable(weights: np.ndarray) -> float
        Returns: best_weights (normalized to sum=1)
        """
        particles = np.random.dirichlet(np.ones(dim), self.n_particles)
        velocities = np.random.randn(self.n_particles, dim) * 0.1
        p_best = particles.copy()
        p_best_scores = np.array([fitness_fn(p) for p in particles])
        g_best_idx = np.argmax(p_best_scores)
        g_best = p_best[g_best_idx].copy()

        for _ in range(self.n_iterations):
            for i in range(self.n_particles):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (
                    self.w * velocities[i]
                    + self.c1 * r1 * (p_best[i] - particles[i])
                    + self.c2 * r2 * (g_best - particles[i])
                )
                particles[i] += velocities[i]
                particles[i] = np.abs(particles[i])
                particles[i] /= particles[i].sum()  # normalize

                score = fitness_fn(particles[i])
                if score > p_best_scores[i]:
                    p_best[i] = particles[i].copy()
                    p_best_scores[i] = score
                    if score > p_best_scores[g_best_idx]:
                        g_best_idx = i
                        g_best = particles[i].copy()
        return g_best


# ── VQC Wrapper ───────────────────────────────────────────────────────────────
class VQCPortfolioOptimizer:
    """
    Quantum VQC-based portfolio optimizer using TwoLocal ansatz.
    """
    def __init__(self, n_qubits=4, reps=2):
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit not installed")
        self.n_qubits = n_qubits
        self.reps = reps

    def build_vqc(self, X_train, y_train):
        feature_map = ZZFeatureMap(feature_dimension=X_train.shape[1], reps=1)
        ansatz = TwoLocal(
            num_qubits=self.n_qubits,
            rotation_blocks=["ry", "rz"],
            entanglement_blocks="cz",
            entanglement="linear",
            reps=self.reps,
        )
        vqc = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=COBYLA(maxiter=100),
            sampler=Sampler(),
        )
        vqc.fit(X_train, y_train)
        return vqc

    def optimize_weights(self, returns: pd.DataFrame) -> np.ndarray:
        """
        returns: DataFrame with columns = assets, rows = time periods
        Returns: optimal weights (summing to 1)
        """
        n_assets = returns.shape[1]
        # Dummy binary classification: positive vs negative returns
        X = returns.values
        y = (X.mean(axis=1) > 0).astype(int)

        # Train VQC
        vqc = self.build_vqc(X, y)

        # Extract learned params as weight proxy (simplified)
        params = vqc._fit_result.x if hasattr(vqc, "_fit_result") else np.random.rand(n_assets)
        weights = np.abs(params[:n_assets])
        weights /= weights.sum()
        return weights


# ── Sharpe / Sortino ──────────────────────────────────────────────────────────
def sharpe_ratio(weights, returns, rf=0.0):
    portfolio_return = (returns @ weights).mean()
    portfolio_std = (returns @ weights).std()
    if portfolio_std == 0:
        return 0.0
    return (portfolio_return - rf) / portfolio_std * np.sqrt(252)

def sortino_ratio(weights, returns, rf=0.0, target=0.0):
    portfolio_returns = returns @ weights
    downside = portfolio_returns[portfolio_returns < target]
    downside_std = downside.std() if len(downside) > 0 else 1e-6
    return (portfolio_returns.mean() - rf) / downside_std * np.sqrt(252)


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", nargs="+", default=["BTC", "ETH", "SOL"],
                        help="Asset symbols")
    parser.add_argument("--method", choices=["vqc", "pso"], default="pso",
                        help="Optimization method")
    args = parser.parse_args()

    # Mock returns data (replace with real data from binance/csv)
    np.random.seed(42)
    n_periods = 100
    returns = pd.DataFrame(
        np.random.randn(n_periods, len(args.assets)) * 0.02,
        columns=args.assets
    )

    print(f"[AuditorSEC] Optimizing {len(args.assets)}-asset portfolio via {args.method.upper()}...")

    if args.method == "vqc" and QISKIT_AVAILABLE:
        opt = VQCPortfolioOptimizer(n_qubits=len(args.assets), reps=2)
        weights = opt.optimize_weights(returns)
    else:
        pso = ParticleSwarmOptimizer(n_particles=30, n_iterations=100)
        fitness = lambda w: sharpe_ratio(w, returns.values)
        weights = pso.optimize(fitness, len(args.assets))

    sharpe = sharpe_ratio(weights, returns.values)
    sortino = sortino_ratio(weights, returns.values)

    print("\n[AuditorSEC] ── Optimal Portfolio Weights ────────────────────")
    for asset, w in zip(args.assets, weights):
        print(f"  {asset:6s} : {w:6.2%}")
    print(f"\n  Sharpe Ratio  : {sharpe:.4f}")
    print(f"  Sortino Ratio : {sortino:.4f}")
    print("\n[AuditorSEC] Optimization complete. Integrate with RL agent via weighted env.")
