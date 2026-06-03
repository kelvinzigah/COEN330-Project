# Task Split & Milestones

Group of **4**. The codebase is intentionally modular so members can own a piece
and work in parallel with minimal merge conflicts. Names are placeholders —
fill them in.

## Suggested ownership

| Member | Owns | Files | Deliverable |
|---|---|---|---|
| **M1 — Environment** | The game / MDP | `game/snake_game.py`, `game/constants.py` | A correct, tested Snake environment with the RL interface. |
| **M2 — Model & Trainer** | The "brain" | `rl/model.py`, `rl/trainer.py` | The network + Bellman update; experiments with architecture. |
| **M3 — Agent & Training** | The policy & loop | `rl/agent.py`, `train.py` | State encoding, epsilon strategy, replay, the training loop. |
| **M4 — Eval, Viz & Report** | Measurement & write-up | `utils/plotting.py`, `tests/`, `docs/`, the report | Metrics, plots, test coverage, and the formal report + demo. |

Everyone contributes to the **report** and the **demo**; M4 coordinates it.

## Working agreement

- Branch per feature; PR into `main`; at least one teammate reviews.
- Keep `main` runnable at all times (`python -m src.snake_ai.train` must work).
- Run `pytest` before every PR.
- All hyperparameters go through `config.py` — no magic numbers in logic.

## Milestones

> Exact dates come from Dr. Bali on **Moodle**. RL content lands ~Week 7.
> Fill in the real deadlines once posted.

| # | Milestone | Target | Owner(s) |
|---|---|---|---|
| 0 | Repo scaffold runs end-to-end, tests pass | ✅ done | all |
| 1 | Baseline agent reaches mean score > 10 | _Week ?_ | M1–M3 |
| 2 | Metrics + plots + baseline write-up | _Week ?_ | M4 |
| 3 | ≥2 enhancements implemented & measured | _Week ?_ | M2, M3 |
| 4 | Final report draft (objectives → design → validation) | _Week ?_ | all |
| 5 | Demo prepared + final submission | _Week ?_ | all |

## Report outline (maps to grading attributes)

1. **Objectives** — what problem, why RL, success criteria.
2. **Idea generation** — alternatives considered (DQN vs. tabular vs. policy
   gradient) and why DQN.
3. **Detailed design** — state/action/reward, network, training loop.
4. **Validation & implementation** — experiments, plots, results, limitations.
5. **Conclusion & future work.**

## Next actions

- [ ] Replace placeholder names + Moodle deadlines above.
- [ ] Confirm Python 3.11/3.12 env on each member's machine.
- [ ] M1–M3: run baseline training to ~300 games, record the curve.
- [ ] M4: set up a results/ folder convention for saved plots.
