"""Pygame Snake environment exposed as a reinforcement-learning interface.

The environment is deliberately decoupled from the learning code: it knows
nothing about Q-values or neural nets. Each step it accepts a relative action
and returns ``(reward, done, score)``. The agent is responsible for translating
the raw game state into a feature vector.
"""
from __future__ import annotations

import random

import numpy as np
import pygame

from .. import config
from .constants import Color, Direction, Point

# Clockwise ordering of directions. Indexing into this list lets us turn a
# relative action (straight / right / left) into an absolute heading without
# any if/else ladders.
_CLOCKWISE = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]


class SnakeGameRL:
    """Snake game driven by external actions, suitable for RL training."""

    def __init__(self, width=config.DEFAULT_WIDTH, height=config.DEFAULT_HEIGHT,
                 render=True, speed=config.GAME_SPEED):
        if width % config.BLOCK_SIZE or height % config.BLOCK_SIZE:
            raise ValueError(
                f"width/height must be multiples of BLOCK_SIZE={config.BLOCK_SIZE}"
            )
        self.width = width
        self.height = height
        self.render = render
        self.speed = speed

        self._display = None
        self._clock = None
        self._font = None
        if self.render:
            pygame.init()
            self._display = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("COEN 330 — Snake AI")
            self._clock = pygame.time.Clock()
            self._font = pygame.font.SysFont("arial", 25)

        self.reset()

    # -- public RL API --------------------------------------------------------
    def reset(self):
        """Start a fresh episode. State is read back via the public getters."""
        self.direction = Direction.RIGHT
        mid = Point(self.width // 2, self.height // 2)
        self.head = mid
        self.snake = [
            mid,
            Point(mid.x - config.BLOCK_SIZE, mid.y),
            Point(mid.x - 2 * config.BLOCK_SIZE, mid.y),
        ]
        self.score = 0
        self.food = None
        self._frame_iteration = 0
        self._place_food()

    def play_step(self, action):
        """Advance one frame given a relative action one-hot ``[straight, right, left]``.

        Returns ``(reward, game_over, score)``.
        """
        self._frame_iteration += 1
        self._pump_events()

        self._move(action)                 # updates self.direction and self.head
        self.snake.insert(0, self.head)

        reward = config.REWARD_STEP
        game_over = False
        if self._is_collision() or self._timed_out():
            game_over = True
            reward = config.REWARD_DEATH
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = config.REWARD_FOOD
            self._place_food()
        else:
            self.snake.pop()               # move forward: drop the tail

        if self.render:
            self._update_ui()
            self._clock.tick(self.speed)

        return reward, game_over, self.score

    def is_collision(self, point: Point | None = None) -> bool:
        """Public collision test, used by the agent to sense danger ahead."""
        return self._is_collision(point)

    # -- internals ------------------------------------------------------------
    def _timed_out(self) -> bool:
        return self._frame_iteration > config.COLLISION_TIMEOUT_FACTOR * len(self.snake)

    def _pump_events(self):
        """Drain the OS event queue so the window stays responsive."""
        if not self.render:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

    def _place_food(self):
        """Drop food on a random free cell."""
        cells_x = self.width // config.BLOCK_SIZE
        cells_y = self.height // config.BLOCK_SIZE
        while True:
            x = random.randint(0, cells_x - 1) * config.BLOCK_SIZE
            y = random.randint(0, cells_y - 1) * config.BLOCK_SIZE
            candidate = Point(x, y)
            if candidate not in self.snake:
                self.food = candidate
                return

    def _is_collision(self, point: Point | None = None) -> bool:
        pt = point if point is not None else self.head
        # Hits a wall?
        if not (0 <= pt.x < self.width and 0 <= pt.y < self.height):
            return True
        # Hits its own body? When testing the head, skip the head itself.
        body = self.snake[1:] if point is None else self.snake
        return pt in body

    def _move(self, action):
        """Apply a ``[straight, right, left]`` turn relative to current heading."""
        idx = _CLOCKWISE.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = _CLOCKWISE[idx]                 # straight: no change
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = _CLOCKWISE[(idx + 1) % 4]       # right turn (clockwise)
        elif np.array_equal(action, [0, 0, 1]):
            new_dir = _CLOCKWISE[(idx - 1) % 4]       # left turn (counter-cw)
        else:
            raise ValueError(f"action must be a 3-way one-hot, got {action!r}")

        self.direction = new_dir
        x, y = self.head.x, self.head.y
        if new_dir == Direction.RIGHT:
            x += config.BLOCK_SIZE
        elif new_dir == Direction.LEFT:
            x -= config.BLOCK_SIZE
        elif new_dir == Direction.DOWN:
            y += config.BLOCK_SIZE
        elif new_dir == Direction.UP:
            y -= config.BLOCK_SIZE
        self.head = Point(x, y)

    def _update_ui(self):
        self._display.fill(Color.BLACK)
        for segment in self.snake:
            pygame.draw.rect(self._display, Color.BLUE1,
                             pygame.Rect(segment.x, segment.y,
                                         config.BLOCK_SIZE, config.BLOCK_SIZE))
            pygame.draw.rect(self._display, Color.BLUE2,
                             pygame.Rect(segment.x + 4, segment.y + 4, 12, 12))
        pygame.draw.rect(self._display, Color.RED,
                         pygame.Rect(self.food.x, self.food.y,
                                     config.BLOCK_SIZE, config.BLOCK_SIZE))
        text = self._font.render(f"Score: {self.score}", True, Color.WHITE)
        self._display.blit(text, [0, 0])
        pygame.display.flip()
