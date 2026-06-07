#The GUI used to play the game with the keyboard. 
#It shares the same core game rules as the RL environment, but is implemented separately to avoid any risk of AI-specific bugs creeping into the human-playable version.

"""Human-playable Snake — control with the arrow keys.

This version exists to *validate the core game rules* independently of any AI:
bounded map, random food placement, the snake growing on eat, and collision
detection (walls + self). It shares the same constants as the RL environment so
both stay in sync.

Run from the project root:

    python -m src.snake_ai.game.play_human

Controls: arrow keys to steer · R to restart after a game over · Q or window
close to quit.
"""


from __future__ import annotations

import random #Used later to randomly places food on the board at the beginning of the game and after the snake eats the food.

import pygame

from .. import config #Contains constants like BLOCK_SIZE, DEFAULT_WIDTH/HEIGHT, and HUMAN_SPEED.
from .constants import Color, Direction, Point

# Opposite headings — used to reject 180° reversals (a snake can't turn back
# onto its own neck).
_OPPOSITE = {
    Direction.RIGHT: Direction.LEFT,
    Direction.LEFT: Direction.RIGHT,
    Direction.UP: Direction.DOWN,
    Direction.DOWN: Direction.UP,
}

# Arrow keys -> heading.
_KEY_TO_DIR = {
    pygame.K_RIGHT: Direction.RIGHT,
    pygame.K_LEFT: Direction.LEFT,
    pygame.K_UP: Direction.UP,
    pygame.K_DOWN: Direction.DOWN,
}

# Heading -> per-step pixel movement.
_DELTA = {
    Direction.RIGHT: (config.BLOCK_SIZE, 0),
    Direction.LEFT: (-config.BLOCK_SIZE, 0),
    Direction.UP: (0, -config.BLOCK_SIZE),
    Direction.DOWN: (0, config.BLOCK_SIZE),
}


class SnakeGameHuman:
    """A self-contained, keyboard-controlled Snake game."""

    def __init__(self, width=config.DEFAULT_WIDTH, height=config.DEFAULT_HEIGHT,
                 speed=config.HUMAN_SPEED):
        if width % config.BLOCK_SIZE or height % config.BLOCK_SIZE:
            raise ValueError(
                f"width/height must be multiples of BLOCK_SIZE={config.BLOCK_SIZE}"
            )
        self.width = width
        self.height = height
        self.speed = speed

        pygame.init()
        self._display = pygame.display.set_mode((width, height))
        pygame.display.set_caption("COEN 330 — Snake (human)")
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("arial", 25)
        self._big_font = pygame.font.SysFont("arial", 40)

        self.reset()

    def reset(self):
        """Start a new game: snake centered, facing right, fresh food."""
        self.direction = Direction.RIGHT
        self._pending = Direction.RIGHT          # next heading from input
        mid = Point(self.width // 2, self.height // 2)
        self.head = mid
        self.snake = [
            mid,
            Point(mid.x - config.BLOCK_SIZE, mid.y),
            Point(mid.x - 2 * config.BLOCK_SIZE, mid.y),
        ]
        self.score = 0
        self.food = None
        self._place_food()

    # -- core game rules ------------------------------------------------------
    def _place_food(self):
        """Drop food on a random cell that the snake doesn't occupy."""
        cells_x = self.width // config.BLOCK_SIZE
        cells_y = self.height // config.BLOCK_SIZE
        while True:
            x = random.randint(0, cells_x - 1) * config.BLOCK_SIZE
            y = random.randint(0, cells_y - 1) * config.BLOCK_SIZE
            candidate = Point(x, y)
            if candidate not in self.snake:
                self.food = candidate
                return

    def _is_collision(self) -> bool:
        """Wall (bounded map) or self-collision."""
        pt = self.head
        if not (0 <= pt.x < self.width and 0 <= pt.y < self.height):
            return True
        return pt in self.snake[1:]

    def _step(self) -> bool:
        """Advance one tick. Returns True if the game is over."""
        self.direction = self._pending
        dx, dy = _DELTA[self.direction]
        self.head = Point(self.head.x + dx, self.head.y + dy)
        self.snake.insert(0, self.head)

        if self._is_collision():
            return True

        if self.head == self.food:
            self.score += 1            # grow: keep the tail this tick
            self._place_food()
        else:
            self.snake.pop()           # move forward: drop the tail

        return False

    # -- main loop ------------------------------------------------------------
    def run(self):
        """Block until the player quits."""
        game_over = False
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    if game_over:
                        if event.key == pygame.K_r:
                            self.reset()
                            game_over = False
                        elif event.key == pygame.K_q:
                            pygame.quit()
                            return
                    elif event.key in _KEY_TO_DIR:
                        new_dir = _KEY_TO_DIR[event.key]
                        # Compare against the committed direction so a quick
                        # double-tap can't reverse the snake into itself.
                        if new_dir != _OPPOSITE[self.direction]:
                            self._pending = new_dir

            if not game_over:
                game_over = self._step()

            self._draw(game_over)
            self._clock.tick(self.speed)

    # -- rendering ------------------------------------------------------------
    def _draw(self, game_over: bool):
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

        score_text = self._font.render(f"Score: {self.score}", True, Color.WHITE)
        self._display.blit(score_text, [0, 0])

        if game_over:
            self._draw_centered(self._big_font, "Game Over", Color.WHITE, dy=-20)
            self._draw_centered(self._font, "Press R to restart, Q to quit",
                                Color.WHITE, dy=25)

        pygame.display.flip()

    def _draw_centered(self, font, text, color, dy=0):
        surface = font.render(text, True, color)
        rect = surface.get_rect(center=(self.width // 2, self.height // 2 + dy))
        self._display.blit(surface, rect)


def main():
    SnakeGameHuman().run()


if __name__ == "__main__":
    main()
