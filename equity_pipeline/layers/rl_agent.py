"""
layers/rl_agent.py — Layer 4 RL Agent (STUB)
============================================
Stub for future reinforcement learning execution layer.
Receives portfolio weights from Layer 3 MetaLearner and refines
allocation based on regime signals and learned reward functions.

Interface is intentionally complete so Layer 4 can be plugged in
without touching any Layer 1-3 code.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path


class RLAgent:
    """
    Layer 4: Reinforcement Learning Execution Agent (STUB).

    The agent receives:
        - pred_scores: (N,)   — meta-learner scores for each stock at time t
        - firm_ids:    (N,)   — stock identifiers
        - context:     dict   — optional market regime signals
    and returns:
        - weights:     (N,)   — portfolio weights (sum = 0 for L/S)

    Planned implementation:
        - Algorithm: PPO (Proximal Policy Optimization) or SAC (Soft Actor-Critic)
        - State:   [pred_scores, rolling_vol, drawdown_state, regime_label]
        - Action:  continuous weight vector (N,)
        - Reward:  Sharpe(net_return) - λ * Turnover - μ * MaxDD_penalty
        - Training: offline RL on historical walk-forward episodes

    Reference: PRD Section 4 — RL Execution Layer.
    """

    def __init__(self, cfg, checkpoint_path: Path | None = None):
        self.cfg             = cfg
        self.checkpoint_path = checkpoint_path
        self._is_trained     = False
        self._policy         = None

    def load(self, checkpoint_path: Path) -> None:
        """Load pre-trained RL policy from checkpoint."""
        raise NotImplementedError(
            "[RLAgent] Not implemented yet. "
            "Use MetaLearner ensemble weights for now (Layer 3)."
        )

    def select_weights(
        self,
        pred_scores: np.ndarray,
        firm_ids:    np.ndarray,
        context:     dict | None = None,
    ) -> np.ndarray:
        """
        Generate portfolio weights given meta-learner scores.
        Falls back to equal-weight long/short decile if not trained.
        """
        if not self._is_trained:
            return self._fallback(pred_scores)
        raise NotImplementedError("[RLAgent] Policy not loaded.")

    def _fallback(self, pred_scores: np.ndarray) -> np.ndarray:
        """Equal-weight top/bottom decile."""
        n      = len(pred_scores)
        ranks  = np.argsort(np.argsort(-pred_scores))
        n_long = max(1, int(np.ceil(n * self.cfg.long_pct)))
        n_short= max(1, int(np.ceil(n * self.cfg.short_pct)))
        weights = np.zeros(n, dtype=np.float32)
        weights[ranks < n_long]     =  1.0 / n_long
        weights[ranks >= n - n_short] = -1.0 / n_short
        return weights

    def reward(
        self,
        portfolio_return:  float,
        turnover:          float,
        max_drawdown:      float,
        lambda_turnover:   float = 0.001,
        mu_drawdown:       float = 0.5,
    ) -> float:
        """
        Composite reward function.
        R = portfolio_return - λ * turnover - μ * abs(min(max_drawdown, 0))
        """
        return (portfolio_return
                - lambda_turnover * turnover
                - mu_drawdown * abs(min(max_drawdown, 0)))

    def train(self, episode_data: list[dict]) -> None:
        """
        Train RL policy offline on historical walk-forward episodes.
        episode_data: list of {state, action, reward, next_state} dicts.
        """
        raise NotImplementedError("[RLAgent.train] Not implemented yet.")

    def __repr__(self) -> str:
        return (f"RLAgent(trained={self._is_trained}, "
                f"long_pct={self.cfg.long_pct:.0%}, "
                f"short_pct={self.cfg.short_pct:.0%})")
