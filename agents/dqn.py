# dqn.py

import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Tuple, List
import pickle

from environment.action import Action, ActionType, ACTION_TYPE_BASE  # Importing from action.py
from environment.common import ResourceCardType, PlayerID  # Importing necessary enums from common.py

VERBOSE_LOGGING = False

# Define constants (adjust these based on your actual game state)
FLATTENED_STATIC_BOARD_STATE_LENGTH = 1817
EXPECTED_NUMBER_OF_PLAYERS = 4
FLATTENED_PLAYER_STATE_LENGTH = 26
FLATTENED_DYNAMIC_BOARD_STATE_LENGTH = 1254
INPUT_STATE_TENSOR_EXPECTED_LENGTH = (
    FLATTENED_DYNAMIC_BOARD_STATE_LENGTH
    + FLATTENED_STATIC_BOARD_STATE_LENGTH
    + EXPECTED_NUMBER_OF_PLAYERS * FLATTENED_PLAYER_STATE_LENGTH
)


class DQN(nn.Module):
    def __init__(self, state_size: int, action_space_size: int, hidden_layer_size=512):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, action_space_size)  # Outputs Q-values for each action
        )

    def forward(self, state):
        return self.network(state)  # [batch_size, action_space_size]


class ReplayBuffer:
    def __init__(self, size: int):
        self.buffer = deque(maxlen=size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),  # Actions as long tensors for indexing
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)


class DQNTrainer:
    def __init__(self,
                 state_size: int,
                 action_space_size: int,
                 gamma=0.99,
                 learning_rate=1e-3,
                 batch_size=64,
                 buffer_size=100000,
                 target_update_freq=1000,
                 num_epochs=1000,
                 max_steps_per_episode=200):

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PyTorch using device: {self.device}")

        # Hyperparameters
        self.state_size = state_size
        self.action_space_size = action_space_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.target_update_freq = target_update_freq
        self.num_epochs = num_epochs
        self.max_steps_per_episode = max_steps_per_episode
        self.model_save_path = f"./model-{state_size}x{action_space_size}-gamma_{gamma}-lr_{learning_rate}-bs_{batch_size}-epochs_{num_epochs}.pth"

        # Initialize Networks and Optimizer
        self.policy_net = DQN(state_size, action_space_size)
        self.target_net = DQN(state_size, action_space_size)
        self.policy_net.to(self.device)
        self.target_net.to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size)

    def train(self, data_iterator):
        steps = 0

        for epoch in range(self.num_epochs):
            total_reward = 0
            for (state, action, reward, next_state, done) in data_iterator:
                if VERBOSE_LOGGING:
                    print("\n")
                    print(f"Input State Tensor Size: {len(state)}")
                    print(f"Input Action Int: {action}")
                    print(f"Reward: {reward}")
                    print(f"Next State Tensor Size: {len(next_state)}")
                    print(f"Game Finished Flag: {done}")

                steps += 1
                total_reward += reward

                self.replay_buffer.add(state, action, reward, next_state, done)

                # Train the policy network if buffer is full
                if len(self.replay_buffer) >= self.batch_size:
                    self.optimize_model()

                # Update target network periodically
                if steps % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

            # Log epoch results
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Total Reward: {total_reward}")

        # Save the trained model
        self.save_model()

    def optimize_model(self):
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough samples to train

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)          # [batch_size, state_size]
        actions = actions.to(self.device)        # [batch_size]
        rewards = rewards.to(self.device)        # [batch_size]
        next_states = next_states.to(self.device) # [batch_size, state_size]
        dones = dones.to(self.device)            # [batch_size]

        # Debugging: Print tensor shapes
        if VERBOSE_LOGGING:
            print(f"States shape: {states.shape}")          # Expected: [batch_size, state_size]
            print(f"Actions shape: {actions.shape}")        # Expected: [batch_size]
            print(f"Rewards shape: {rewards.shape}")        # Expected: [batch_size]
            print(f"Next States shape: {next_states.shape}")# Expected: [batch_size, state_size]
            print(f"Dones shape: {dones.shape}")            # Expected: [batch_size]

        # Q(s, a)
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)  # [batch_size]

        # Compute target Q-values
        with torch.no_grad():
            # Compute max Q(s', a') for next states
            next_q_values = self.target_net(next_states)  # [batch_size, action_space_size]
            max_next_q_values, _ = next_q_values.max(dim=1)  # [batch_size]
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        # Loss
        loss = nn.MSELoss()(q_values, target_q_values)

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self):
        torch.save(self.policy_net.state_dict(), self.model_save_path)
        print(f"Model saved to: {self.model_save_path}")

    def save_action_mapping(self, mapping_path: str):
        with open(mapping_path, 'wb') as f:
            pickle.dump({'ACTION_TYPE_BASE': ACTION_TYPE_BASE}, f)
        print(f"Action mappings saved to: {mapping_path}")

    def load_action_mapping(self, mapping_path: str):
        global ACTION_TYPE_BASE
        with open(mapping_path, 'rb') as f:
            mappings = pickle.load(f)
            ACTION_TYPE_BASE = mappings['ACTION_TYPE_BASE']
        print(f"Action mappings loaded from: {mapping_path}")
