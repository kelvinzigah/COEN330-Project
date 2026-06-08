"""DQN agent: senses the game state, chooses actions, and learns from experience.

The agent is the bridge between the environment and the network. It converts the
raw game into an 11-feature state vector, balances exploration vs. exploitation
with an epsilon-greedy policy, stores transitions in a replay buffer, and drives
both short-term (per-step) and long-term (batched) training.
"""
from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch

#Connecting the agent to the game environment. The agent uses the SnakeGameRL class to interact with the game, 
#and it uses the LinearQNet and QTrainer classes to learn from its experiences. 
#The agent's main responsibilities include converting the game state into a format suitable for the neural network, choosing actions based on an epsilon-greedy policy, storing transitions in a replay buffer, and training the neural network on both short-term and long-term experiences.
from .. import config
from ..game.constants import Direction, Point
from ..game.snake_game import SnakeGameRL
from .model import LinearQNet #The neural network model that approximates the Q-values for each action given a state. It takes the 11-feature state vector as input and outputs a vector of Q-values for the three possible actions (straight, right, left).
from .trainer import QTrainer


class Agent:
    def __init__(self, cfg: config.TrainConfig = config.DEFAULT_TRAIN_CONFIG):
        self.cfg = cfg #The configuration object that contains hyperparameters for training, such as learning rate, gamma (discount factor), epsilon parameters for exploration, batch size, and memory size for the replay buffer. The agent uses these parameters to control its learning process and behavior during training.
        self.n_games = 0 #How many games the agent has played. This is used to adjust the exploration rate (epsilon) over time, allowing the agent to explore more in the early stages of training and exploit learned strategies as it gains experience.
        self.epsilon = 0
        self.memory = deque(maxlen=cfg.max_memory) #Storage for past experiences (state, action, reward, next_state, done) that the agent can sample from during training. The replay buffer helps break the correlation between consecutive experiences and allows the agent to learn from a more diverse set of past interactions with the environment.
        self.model = LinearQNet(cfg.state_size, cfg.hidden_size, cfg.action_size)
        self.trainer = QTrainer(self.model, cfg.learning_rate, cfg.gamma)

    def get_state(self, game: SnakeGameRL) -> np.ndarray:
        """Build the 11-value boolean state vector for the current game frame."""
        #The 11 values are:
        #1-3: Danger straight, right, left (relative to current heading).
        #4-7: Current heading (one-hot for up/down/left/right).
        #8-11: Food location (relative to head: left, right, up, down).
        #Example, danger straight is represented as: [1, 0, 0] if the snake is currently moving right and there is a collision ahead, otherwise it would be [0, 0, 0].
        head = game.snake[0]
        bs = config.BLOCK_SIZE
        #Four positions around the head, used to detect danger in those directions. The agent checks for potential collisions in the straight, right, and left directions relative to the snake's current heading. For example, if the snake is currently moving right, the agent will check for collisions at point_r (straight), point_d (right), and point_u (left).
        point_l = Point(head.x - bs, head.y)
        point_r = Point(head.x + bs, head.y)
        point_u = Point(head.x, head.y - bs)
        point_d = Point(head.x, head.y + bs)
        #Checking the current direction of the snake to determine how to interpret the danger in the straight, right, and left directions. 
        #For example, if the snake is currently moving right, then danger straight would be a collision at point_r, danger right would be a collision at point_d, and danger left would be a collision at point_u.
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [ #We can think of the states as binary features that describe the current situation of the game from the perspective of the snake. 
        #The agent uses these features to make informed decisions about which action to take next.
            # Danger straight ahead (in the current heading).
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),
            # Danger on an immediate right turn.
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),
            # Danger on an immediate left turn.
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),
            # Current heading (exactly one is True).
            dir_l, dir_r, dir_u, dir_d, #The environment handles this in _move(action), where the action is a one-hot vector representing the relative turn (straight, right, left) the snake should take based on its current heading. The agent uses this information to determine the new direction of the snake after taking an action.
            # Food location relative to the head.
            game.food.x < head.x,   # food is left
            game.food.x > head.x,   # food is right
            game.food.y < head.y,   # food is up
            game.food.y > head.y,   # food is down
            #Example: If the food is located to the left and above the snake's head, the last four values would be [1, 0, 1, 0].
        ]
        return np.array(state, dtype=int) #Converts the boolean state vector into an integer array, where True is represented as 1 and False as 0. This format is suitable for input into the neural network model, which expects numerical data.

    def remember(self, state, action, reward, next_state, done):
        """Append one transition to the replay buffer."""
        self.memory.append((state, action, reward, next_state, done)) #Storing the transition (state, action, reward, next_state, done) in the replay buffer. This allows the agent to learn from past experiences by sampling random batches of transitions during training, which helps break the correlation between consecutive experiences and improves learning stability.

    def train_long_memory(self): #Training at the end of each game, after accumulating a batch of experiences in the replay buffer. The agent samples a random minibatch of transitions from the replay buffer and performs a training step on the neural network uusing these samples.
        """Train on a random minibatch sampled from the replay buffer."""
        if len(self.memory) > self.cfg.batch_size:
            sample = random.sample(self.memory, self.cfg.batch_size)
        else:
            sample = list(self.memory)
        if not sample:
            return
        states, actions, rewards, next_states, dones = zip(*sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done): #Training at the end of each step, immediately after taking an action and observing the resulting reward and new state. This allows the agent to learn from the most recent experience without waiting for a batch of experiences to accumulate in the replay buffer.
        """Train on the single transition that just occurred."""
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state) -> list[int]: 
    #Choosing the agents next move. The agent uses an epsilon-greedy policy to balance exploration and exploitation. Early in training, the agent will explore more by choosing random actions.
    #Later in training, as it gains experience, it will exploit its learned knowledge by choosing actions that maximize the predicted Q-values from the neural network model.
        """Epsilon-greedy action: explore early, exploit as games accumulate.

        Returns a one-hot ``[straight, right, left]``.
        """
        #Agent exploring less and tusting the model more as the number of games increase is called epsilon-greedy policy.
        self.epsilon = self.cfg.epsilon_start - self.n_games #As the number of games played increases, epsilon decreases, reducing the exploration rate and allowing the agent to exploit its learned strategies more often.
        action = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)                      # explore
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_tensor) #Makes a prediction using the neural network model to get the Q-values for each possible action (straight, right, left) based on the current state. 
            #The agent then chooses the action with the highest predicted Q-value, which represents the best expected future reward according to the model.
            move = int(torch.argmax(prediction).item())      # exploit/learn
            #Example: If the model predicts Q-values of [0.5, 0.2, 0.3] for the actions [straight, right, left], 
            #the agent will choose the action "straight" (index 0) because it has the highest Q-value.
        action[move] = 1 #Converting the chosen action index into a one-hot vector format, where the index corresponding to the chosen action is set to 1 and the others are set to 0. For example, if move is 0 (straight), the action will be [1, 0, 0]; if move is 1 (right), the action will be [0, 1, 0]; and if move is 2 (left), the action will be [0, 0, 1].
        return action #The action is used by the environment to update the game state accordingly, and the resulting reward and new state are used by the agent to learn and improve its decision-making over time.
        #The action output is just: [1, 0, 0] for straight, [0, 1, 0] for right, and [0, 0, 1] for left. The environment's (snake_game.py) _move function will interpret this action to update the snake's direction and position in the game.

#At the beginning, the model does not know what moves are good/bad. It learns from experience by taking actions, observing the resulting rewards and new states, and updating the neural network model to better predict the Q-values for future actions. 
#Over time, as the agent plays more games and learns from its experiences, it becomes better at choosing actions that lead to higher rewards, ultimately improving its performance in the Snake game.

