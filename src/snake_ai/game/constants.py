"""Game-level constants: directions, grid points, and rendering colors."""
from collections import namedtuple
from enum import Enum


class Direction(Enum):
    """The four headings the snake can face."""
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


# A grid coordinate in pixels. x grows to the right, y grows downward
# (standard Pygame screen convention).
Point = namedtuple("Point", ["x", "y"])


class Color:
    """RGB colors used when rendering the game."""
    WHITE = (255, 255, 255)
    RED = (200, 0, 0)
    BLUE1 = (0, 0, 255)
    BLUE2 = (0, 100, 255)
    BLACK = (0, 0, 0)
    GREEN = (0, 200, 0)
