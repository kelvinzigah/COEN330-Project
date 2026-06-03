"""Live training-progress plot using matplotlib.

Shows the per-game score and the running mean so you can watch the agent
improve during training. Plotting is optional and isolated here so headless
runs (e.g. CI or a server) can simply skip it.
"""
from __future__ import annotations

import matplotlib.pyplot as plt


class LivePlot:
    """Incrementally redraws the score and mean-score curves during training."""

    def __init__(self):
        plt.ion()  # interactive mode: update without blocking

    def update(self, scores, mean_scores):
        plt.clf()
        plt.title("Training… (close the window to stop plotting)")
        plt.xlabel("Number of games")
        plt.ylabel("Score")
        plt.plot(scores, label="Score")
        plt.plot(mean_scores, label="Mean score")
        plt.ylim(ymin=0)
        if scores:
            plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
        if mean_scores:
            plt.text(len(mean_scores) - 1, mean_scores[-1], f"{mean_scores[-1]:.1f}")
        plt.legend(loc="upper left")
        plt.pause(0.001)
