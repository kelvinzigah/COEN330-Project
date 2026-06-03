"""Logic tests for the human-playable Snake (run headless via SDL dummy).

These verify the game *rules* (growth, movement, bounded map, collisions)
without needing a real window or keyboard.
"""
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # no display needed

from src.snake_ai.game.constants import Direction, Point  # noqa: E402
from src.snake_ai.game.play_human import _OPPOSITE, SnakeGameHuman  # noqa: E402


def make_game():
    return SnakeGameHuman(width=200, height=200)


def test_reset_state():
    g = make_game()
    assert len(g.snake) == 3
    assert g.score == 0
    assert g.food not in g.snake
    assert g.direction == Direction.RIGHT


def test_move_forward_keeps_length():
    g = make_game()
    g.food = Point(0, 0)              # keep food out of the path
    before = len(g.snake)
    g._step()
    assert len(g.snake) == before     # tail dropped, head added => same length


def test_eating_grows_and_scores():
    g = make_game()
    g.food = Point(g.head.x + 20, g.head.y)   # one block to the right (ahead)
    before = len(g.snake)
    over = g._step()
    assert over is False
    assert g.score == 1
    assert len(g.snake) == before + 1          # grew by one segment


def test_wall_collision_ends_game():
    g = make_game()
    g.food = Point(0, 0)                        # avoid eating en route
    over = False
    for _ in range(20):                         # drive straight into right wall
        over = g._step()
        if over:
            break
    assert over is True


def test_self_collision_detected():
    g = make_game()
    # Manually place the head onto a body cell and check collision logic.
    g.head = Point(60, 60)
    g.snake = [Point(60, 60), Point(40, 60), Point(40, 80), Point(60, 80)]
    g.snake.insert(0, g.head)                   # head now duplicates a body cell
    assert g._is_collision() is True


def test_reversal_mapping():
    # The UI ignores a key opposite the current heading (no instant suicide).
    assert _OPPOSITE[Direction.RIGHT] == Direction.LEFT
    assert _OPPOSITE[Direction.LEFT] == Direction.RIGHT
    assert _OPPOSITE[Direction.UP] == Direction.DOWN
    assert _OPPOSITE[Direction.DOWN] == Direction.UP
