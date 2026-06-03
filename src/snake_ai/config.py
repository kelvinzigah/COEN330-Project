"""Central configuration for the Snake AI project.

All tunable constants and hyperparameters live in one place so experiments are
reproducible and the rest of the code stays free of magic numbers. Values match
the Deep Q-Learning approach covered in the COEN 330 course sources.
"""
from dataclasses import dataclass


# --- Game geometry -----------------------------------------------------------
BLOCK_SIZE = 20            # pixel size of one grid cell / snake segment
DEFAULT_WIDTH = 640        # window width  (must be a multiple of BLOCK_SIZE)
DEFAULT_HEIGHT = 480       # window height (must be a multiple of BLOCK_SIZE)
GAME_SPEED = 60            # frames per second while the AI plays/renders
HUMAN_SPEED = 10           # frames per second for the human-playable version

# --- Reward shaping ----------------------------------------------------------
REWARD_FOOD = 10
REWARD_DEATH = -10
REWARD_STEP = 0            # per-step reward (0 = neutral; tune in enhancement phase)

# An episode is force-ended if the snake wanders this many steps (scaled by its
# length) without eating. Prevents infinite loops during training.
COLLISION_TIMEOUT_FACTOR = 100


@dataclass(frozen=True)
class TrainConfig:
    """Immutable bundle of DQN hyperparameters.

    Frozen so a config can be passed around without risk of accidental
    mutation; create a new instance to run a different experiment.
    """
    max_memory: int = 100_000     # replay buffer capacity (transitions)
    batch_size: int = 1000        # long-memory minibatch size
    learning_rate: float = 1e-3   # Adam learning rate
    gamma: float = 0.9            # discount factor for future reward
    hidden_size: int = 256        # hidden layer width of the Q-network
    epsilon_start: int = 80       # exploration: P(random) ~ (epsilon_start - n_games)
    state_size: int = 11          # 11 boolean features (see Agent.get_state)
    action_size: int = 3          # [straight, right, left]


DEFAULT_TRAIN_CONFIG = TrainConfig()
