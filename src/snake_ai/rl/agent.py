"""DQN agent: senses the game state, chooses actions, and learns from experience.

The agent is the bridge between the environment and the network. It converts the
raw game into an 11-feature state vector, balances exploration vs. exploitation
with an epsilon-greedy policy, stores transitions in a replay buffer, and drives
both short-term (per-step) and long-term (batched) training.
"""
from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch

from .. import config
from ..game.constants import Direction, Point
from ..game.snake_game import SnakeGameRL
from .model import LinearQNet
from .trainer import QTrainer


class Agent:
    def __init__(self, cfg: config.TrainConfig = config.DEFAULT_TRAIN_CONFIG):
        self.cfg = cfg
        self.n_games = 0
        self.epsilon = 0
        self.memory = deque(maxlen=cfg.max_memory)
        self.model = LinearQNet(cfg.state_size, cfg.hidden_size, cfg.action_size)
        self.trainer = QTrainer(self.model, cfg.learning_rate, cfg.gamma)

    def get_state(self, game: SnakeGameRL) -> np.ndarray:
        """Build the 11-value boolean state vector for the current game frame."""
        head = game.snake[0]
        bs = config.BLOCK_SIZE
        point_l = Point(head.x - bs, head.y)
        point_r = Point(head.x + bs, head.y)
        point_u = Point(head.x, head.y - bs)
        point_d = Point(head.x, head.y + bs)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight ahead (in the current heading).
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),
            # Danger on an immediate right turn.
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),
            # Danger on an immediate left turn.
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),
            # Current heading (exactly one is True).
            dir_l, dir_r, dir_u, dir_d,
            # Food location relative to the head.
            game.food.x < head.x,   # food is left
            game.food.x > head.x,   # food is right
            game.food.y < head.y,   # food is up
            game.food.y > head.y,   # food is down
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        """Append one transition to the replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        """Train on a random minibatch sampled from the replay buffer."""
        if len(self.memory) > self.cfg.batch_size:
            sample = random.sample(self.memory, self.cfg.batch_size)
        else:
            sample = list(self.memory)
        if not sample:
            return
        states, actions, rewards, next_states, dones = zip(*sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        """Train on the single transition that just occurred."""
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state) -> list[int]:
        """Epsilon-greedy action: explore early, exploit as games accumulate.

        Returns a one-hot ``[straight, right, left]``.
        """
        self.epsilon = self.cfg.epsilon_start - self.n_games
        action = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)                      # explore
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_tensor)
            move = int(torch.argmax(prediction).item())      # exploit
        action[move] = 1
        return action
