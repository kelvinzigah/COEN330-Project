"""Entry point: train the DQN agent to play Snake.

Run from the project root:

    python -m src.snake_ai.train

The loop runs forever; stop it with Ctrl+C (or close the game window). The best
model so far is saved to ``models/model.pth`` whenever a new high score is hit.
"""
from __future__ import annotations

import argparse

from .game.snake_game import SnakeGameRL
from .rl.agent import Agent
from .utils.plotting import LivePlot


def train(render: bool = True, plot: bool = True):
    scores: list[int] = []
    mean_scores: list[float] = []
    total_score = 0
    record = 0

    agent = Agent()
    game = SnakeGameRL(render=render)
    live_plot = LivePlot() if plot else None

    try:
        while True:
            # 1) Observe, decide, act.
            state_old = agent.get_state(game)
            action = agent.get_action(state_old)
            reward, done, score = game.play_step(action)
            state_new = agent.get_state(game)

            # 2) Learn from this single step and remember it for later.
            agent.train_short_memory(state_old, action, reward, state_new, done)
            agent.remember(state_old, action, reward, state_new, done)

            if done:
                # 3) End of episode: replay-train and report progress.
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                if score > record:
                    record = score
                    path = agent.model.save()
                    print(f"  ↳ new record {record}, saved to {path}")

                print(f"Game {agent.n_games:>4} | Score {score:>3} | Record {record:>3}")

                scores.append(score)
                total_score += score
                mean_scores.append(total_score / agent.n_games)
                if live_plot:
                    live_plot.update(scores, mean_scores)
    except KeyboardInterrupt:
        print(f"\nStopped after {agent.n_games} games. Best score: {record}.")


def main():
    parser = argparse.ArgumentParser(description="Train the Snake DQN agent.")
    parser.add_argument("--no-render", action="store_true",
                        help="train without the game window (faster).")
    parser.add_argument("--no-plot", action="store_true",
                        help="disable the live matplotlib score plot.")
    args = parser.parse_args()
    train(render=not args.no_render, plot=not args.no_plot)


if __name__ == "__main__":
    main()
