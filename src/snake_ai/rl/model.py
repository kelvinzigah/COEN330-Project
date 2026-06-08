#https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

#https://www.kaggle.com/code/stefancomanita/snake-game-ai-with-reinforcement-learning

#https://docs.pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

import os
import torch
import numpy as np
import torch.nn.functional as F #Common in many examples, helps make the code cleaner.
from torch import nn



 #Defining our neural network model for Deep Q-Learning. We subclass nn.Model, and initialize the neural network layers in the __init__ method.
class DQN(nn.Module):

    def __init__(self, state_size, hidden_size, action_size):
        super().__init__()
        self.first_layer = nn.Linear(state_size, hidden_size) #The input layer is a linear transformation that maps the input features to the hidden layer. 
        self.last_layer = nn.Linear(hidden_size, action_size) #The output layer is a linear transformation that maps the hidden layer to the output Q-values.


    def forward(self, x):
        X = F.relu(self.first_layer(x)) #In the hidden layer, the model can learn combinations of the state features. ReLU (or sigma for an alternate) is used to help the network learn patterns (pattern strengths, from 0 to infinity) in the data by introducing non-linearity. Without it, the network would only be able to learn linear relationships between the input and output, which may not be sufficient for complex tasks like Snake. The values for RelU are from 0 to infinity, which means that it will output 0 for any negative input and will output the input itself for any positive input. This allows the network to learn more complex patterns in the data, as it can capture non-linear relationships between the input features and the output Q-values. The Q values would still be negative, zero, or positive, since its outputting estimates (like a regression problem), and we want to know all good, bad, neutral actions.
        X = self.last_layer(X) #The output of the linear transformation. The output is the 3 Q-values for the 3 possible actions: straight, right, left.

        return self.last_layer(X) 

        #Saving the model weights to a file. The save method creates the specified folder if it doesn't exist.
        path = os.path.join("models", "model.pth")
        torch.save(self.state_dict(), path)




















# """Neural network for Deep Q-Learning.

# A small feed-forward network is enough for the 11-feature Snake state. It maps a
# state vector to one Q-value per action; the agent picks the action with the
# highest predicted value.
# """

# #Here, the output is the 3 Q-values for the 3 possible actions: straight, right, left. The input is the 11 features of the state vector.
# from __future__ import annotations

# import os

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class LinearQNet(nn.Module): #nn.Module is the base class for all neural network modules in PyTorch.
#     """Feed-forward Q-network: input -> hidden (ReLU) -> output Q-values."""

#     def __init__(self, input_size: int, hidden_size: int, output_size: int):
#         super().__init__()
#         self.linear1 = nn.Linear(input_size, hidden_size, bias = True) #The input layer is a linear transformation that maps the input features to the hidden layer. 
#         self.linear2 = nn.Linear(hidden_size, output_size, bias = True)#The output layer is a linear transformation that maps the hidden layer to the output Q-values.
# #In the hidden layer, the model can learn combinations of the state features.
# #ReLU (or sigma for an alternate) is used to help the network learn patterns (pattern strengths, from 0 to infinity) in the data by introducing non-linearity. Without it, the network would only be able to learn linear relationships between the input and output, which may not be sufficient for complex tasks like Snake.
# #The values for RelU are from 0 to infinity, which means that it will output 0 for any negative input and will output the input itself for any positive input. This allows the network to learn more complex patterns in the data, as it can capture non-linear relationships between the input features and the output Q-values.
# #The Q values would still be negative, zero, or positive, since its outputting estimates (like a regression problem), and we want to know all good, bad, neutral actions.


#     def forward(self, x: torch.Tensor) -> torch.Tensor: #The forward method defines how the input data flows through the network. It takes a tensor x as input, applies the first linear transformation, then applies the ReLU activation function, and finally applies the second linear transformation to produce the output Q-values.
#         x = F.relu(self.linear1(x))
#         return self.linear2(x)

#     def save(self, file_name: str = "model.pth", folder: str = "models") -> str:
#         """Persist weights to ``folder/file_name`` and return the full path."""
#         os.makedirs(folder, exist_ok=True)
#         path = os.path.join(folder, file_name)
#         torch.save(self.state_dict(), path)
#         return path