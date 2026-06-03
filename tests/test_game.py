"""Tests for the Snake environment (run headless, no display needed)."""
from src.snake_ai.game.constants import Direction, Point
from src.snake_ai.game.snake_game import SnakeGameRL


def make_game():
    return SnakeGameRL(width=200, height=200, render=False)


def test_reset_initializes_snake_and_food():
    g = make_game()
    assert len(g.snake) == 3
    assert g.score == 0
    assert g.food is not None
    assert g.food not in g.snake


def test_straight_action_keeps_direction():
    g = make_game()
    g.play_step([1, 0, 0])
    assert g.direction == Direction.RIGHT


def test_right_action_turns_clockwise():
    g = make_game()
    g.play_step([0, 1, 0])
    assert g.direction == Direction.DOWN


def test_left_action_turns_counter_clockwise():
    g = make_game()
    g.play_step([0, 0, 1])
    assert g.direction == Direction.UP


def test_invalid_action_raises():
    g = make_game()
    try:
        g.play_step([1, 1, 0])
    except ValueError:
        return
    raise AssertionError("expected ValueError for a non-one-hot action")


def test_is_collision_detects_walls():
    g = make_game()
    assert g.is_collision(Point(-20, 0)) is True
    assert g.is_collision(Point(g.width, 0)) is True
    assert g.is_collision(Point(0, g.height)) is True


def test_non_multiple_dimensions_rejected():
    try:
        SnakeGameRL(width=205, height=200, render=False)
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-multiple width")
