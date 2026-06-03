# Architecture

This document explains how the pieces fit together and why the project is
split the way it is. The guiding principle: **the game knows nothing about
learning, and the network knows nothing about the game.** The `Agent` is the
only component that talks to both.

## The reinforcement-learning loop

```
   Agent --- state ---> Q-Net (model)
   Agent <- Q-values -- Q-Net (model)
   Agent --- action --> Game (env)
   Agent <- reward, done, score -- Game (env)
   Agent --- remember + train ---> (replay memory / network)
```

1. `Agent.get_state(game)` reads the game and builds an **11-value** vector.
2. `Agent.get_action(state)` picks an action — random early on (exploration),
   network-driven later (exploitation), controlled by epsilon.
3. `game.play_step(action)` advances one frame and returns `(reward, done, score)`.
4. The agent trains on that single step (**short memory**) and stores it.
5. On death, it replays a random batch from memory (**long memory**) and resets.

## Components

| File | Responsibility |
|---|---|
| `config.py` | Single source of truth for geometry, rewards, and `TrainConfig` (immutable hyperparameters). |
| `game/constants.py` | `Direction` enum, `Point` namedtuple, `Color` palette. |
| `game/snake_game.py` | `SnakeGameRL` — the environment. `reset`, `play_step`, `is_collision`, `_move`. Pure game logic; optional rendering. |
| `rl/model.py` | `LinearQNet` — `11 → 256 → 3` feed-forward net. `forward`, `save`. |
| `rl/trainer.py` | `QTrainer` — the Bellman update (Adam + MSE). Handles single transitions and batches. |
| `rl/agent.py` | `Agent` — state encoding, epsilon-greedy policy, replay buffer, short/long training. |
| `utils/plotting.py` | `LivePlot` — live score/mean curves. |
| `train.py` | Wires it all together in the training loop; CLI flags for headless/no-plot. |

## State vector (11 features, all 0/1)

| Index | Meaning |
|---|---|
| 0 | Danger straight ahead |
| 1 | Danger on a right turn |
| 2 | Danger on a left turn |
| 3–6 | Current heading: left, right, up, down |
| 7–10 | Food is: left, right, up, down (of the head) |

## Action vector (3 features, one-hot)

`[1,0,0]` straight · `[0,1,0]` turn right (clockwise) · `[0,0,1]` turn left.
Relative actions make 180° suicide turns impossible.

## Key hyperparameters (see `config.TrainConfig`)

| Name | Value | Note |
|---|---|---|
| Hidden layer size | 256 | |
| Gamma (discount) | 0.9 | future-reward weight |
| Learning rate | 0.001 | Adam |
| Replay memory | 100,000 | `deque` |
| Batch size | 1,000 | long-memory sample |
| Epsilon | `80 − n_games` | linear exploration decay |

## Staged plan

**Phase 1 — Baseline (done in this scaffold).** Get a complete, demoable agent
that clearly learns. This is the safety net: even if nothing else lands, the
project works end-to-end.

**Phase 2 — Enhancements (the "design" marks).** Candidate improvements, each a
self-contained experiment to measure against the baseline:

- Reward shaping (small reward for moving toward food; penalty for stalling).
- Double DQN / Dueling DQN to reduce Q-value overestimation.
- Target network for training stability.
- Richer state (distance to food, body density, tail direction).
- Hyperparameter sweeps (gamma, lr, hidden size) with plotted comparisons.
- CNN "vision" state (grid pixels) as a stretch goal.

Each experiment should be measured (mean score over N games) and written up so
the report can show *why* a change helped or didn't.
