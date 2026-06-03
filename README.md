# COEN 330 — Snake AI (Reinforcement Learning)

A classic Snake game where an AI agent learns to play — and maximize its score —
using **Deep Q-Learning (DQN)** built with PyTorch and Pygame.

> Course project for **COEN 330: Applied Machine Learning** (Summer 2026).
> Group of 4. Worth 40% of the course grade.

---

## What it does

The agent observes a compact **11-feature state** (danger ahead/right/left,
current heading, and food direction), chooses one of **3 relative actions**
(go straight / turn right / turn left), and is rewarded **+10** for eating food
and **−10** for dying. A small neural network learns, over hundreds of games,
to play better than a hand-coded bot.

## Tech stack

| Purpose | Library |
|---|---|
| Game environment & rendering | Pygame |
| Neural network / DQN | PyTorch |
| State math | NumPy |
| Training plots | Matplotlib |

Runs comfortably on a **CPU** — no GPU required.

## Project structure

```
COEN330-Project/
├── src/snake_ai/
│   ├── config.py            # all constants & hyperparameters (one place)
│   ├── game/
│   │   ├── constants.py     # Direction, Point, Color
│   │   ├── snake_game.py    # SnakeGameRL: the RL environment
│   │   └── play_human.py    # human-playable Snake (arrow keys)
│   ├── rl/
│   │   ├── model.py         # LinearQNet (the network)
│   │   ├── trainer.py       # QTrainer (Bellman update)
│   │   └── agent.py         # Agent (state, action, memory, learning)
│   ├── utils/
│   │   └── plotting.py      # LivePlot (live training curve)
│   └── train.py             # training entry point
├── tests/                   # pytest suite (runs headless)
├── docs/
│   ├── ARCHITECTURE.md      # how the pieces fit + the staged plan
│   └── TASKS.md             # 4-member work split & milestones
├── requirements.txt
└── README.md
```

## Setup (teammates start here)

> **Python version matters.** Use **Python 3.11, 3.12, or 3.13**. As of mid-2026
> PyTorch and Pygame do **not** yet ship wheels for **Python 3.14**, so a 3.14
> environment will fail to install. Check yours with `python --version`. If your
> default `python` is 3.14, install 3.13 and use it explicitly (see notes below).

**1. Clone the repo**

```bash
git clone https://github.com/kelvinzigah/COEN330-Project.git
cd COEN330-Project
```

**2. Create and activate a virtual environment**

<details open>
<summary><b>Windows (PowerShell)</b></summary>

```powershell
py -3.13 -m venv .venv          # or:  python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
If activation is blocked, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
</details>

<details>
<summary><b>macOS</b></summary>

```bash
python3.13 -m venv .venv        # or:  python3 -m venv .venv
source .venv/bin/activate
```
No Python 3.13? Install it with `brew install python@3.13`.
</details>

<details>
<summary><b>Linux</b></summary>

```bash
python3.13 -m venv .venv        # or:  python3 -m venv .venv
source .venv/bin/activate
```
On Debian/Ubuntu you may need: `sudo apt install python3.13-venv`.
</details>

**3. Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

You'll know it worked if `python -c "import torch, pygame"` prints nothing
(no error). When the venv is active your prompt shows `(.venv)`; deactivate
anytime with `deactivate`.

## Play it yourself (human mode)

The quickest way to confirm your setup works — play the game with the arrow
keys (bounded map, random food, the snake grows as you eat):

```bash
python -m src.snake_ai.game.play_human
```

Controls: **arrow keys** to steer · **R** to restart after a game over · **Q**
or closing the window to quit.

> The same command works on Windows, macOS, and Linux once the venv is active.

## Run training

```bash
# watch it learn (game window + live plot)
python -m src.snake_ai.train

# train fast with no window and no plot
python -m src.snake_ai.train --no-render --no-plot
```

The best model so far is saved to `models/model.pth` whenever a new high score
is reached. Stop anytime with **Ctrl+C**.

## Run the tests

```bash
pytest
```

## Roadmap (staged plan)

- **Phase 1 — Baseline (this scaffold):** working DQN that learns to play.
- **Phase 2 — Enhance:** reward shaping, Double/Dueling DQN, hyperparameter
  sweeps, longer-horizon state. These improvements are where the report earns
  its "idea generation / design / validation" marks.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and
[`docs/TASKS.md`](docs/TASKS.md).

## Academic integrity

This implementation was written from scratch for our group. The course uses
code-similarity tools — do not copy external tutorial code verbatim. Cite any
references in the final report.
