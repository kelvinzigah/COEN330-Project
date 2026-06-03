"""Neural network for Deep Q-Learning.

A small feed-forward network is enough for the 11-feature Snake state. It maps a
state vector to one Q-value per action; the agent picks the action with the
highest predicted value.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearQNet(nn.Module):
    """Feed-forward Q-network: input -> hidden (ReLU) -> output Q-values."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.linear1(x))
        return self.linear2(x)

    def save(self, file_name: str = "model.pth", folder: str = "models") -> str:
        """Persist weights to ``folder/file_name`` and return the full path."""
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, file_name)
        torch.save(self.state_dict(), path)
        return path
