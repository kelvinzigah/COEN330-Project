"""Tests for the Q-network and the agent's state encoding."""
import numpy as np
import torch

from src.snake_ai.game.snake_game import SnakeGameRL
from src.snake_ai.rl.agent import Agent
from src.snake_ai.rl.model import LinearQNet


def test_forward_single_state_shape():
    model = LinearQNet(11, 16, 3)
    out = model(torch.zeros(11))
    assert out.shape == (3,)


def test_forward_batch_shape():
    model = LinearQNet(11, 16, 3)
    out = model(torch.zeros((5, 11)))
    assert out.shape == (5, 3)


def test_get_state_is_eleven_booleans():
    agent = Agent()
    game = SnakeGameRL(width=200, height=200, render=False)
    state = agent.get_state(game)
    assert state.shape == (11,)
    assert set(np.unique(state)).issubset({0, 1})


def test_get_action_returns_one_hot():
    agent = Agent()
    game = SnakeGameRL(width=200, height=200, render=False)
    action = agent.get_action(agent.get_state(game))
    assert sum(action) == 1
    assert len(action) == 3
