"""Q-learning optimization step (the Bellman update).

``QTrainer`` wraps the optimizer and loss so the agent can call a single
``train_step`` for both per-step ("short memory") and batched ("long memory")
learning. It accepts either a single transition or batches of equal length.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class QTrainer:
    """Performs one gradient update toward the Bellman target."""

    def __init__(self, model: nn.Module, learning_rate: float, gamma: float):
        self.gamma = gamma
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done) -> float:
        state = torch.tensor(np.asarray(state), dtype=torch.float)
        next_state = torch.tensor(np.asarray(next_state), dtype=torch.float)
        action = torch.tensor(np.asarray(action), dtype=torch.long)
        reward = torch.tensor(np.asarray(reward), dtype=torch.float)

        # Promote a single transition to a batch of one so the rest of the
        # method can assume a batch dimension.
        if state.dim() == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)

        # 1) Predicted Q-values for the current states.
        pred = self.model(state)

        # 2) Bellman target: Q_new = r + gamma * max_a Q(next) unless terminal.
        target = pred.clone()
        for i in range(len(done)):
            q_new = reward[i]
            if not done[i]:
                q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i]))
            target[i][torch.argmax(action[i]).item()] = q_new

        # 3) One gradient step minimizing (target - prediction)^2.
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
        return loss.item()
