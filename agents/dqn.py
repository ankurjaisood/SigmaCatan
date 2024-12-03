import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_tensor_size=3188, output_action_space_size=832, hidden_layer_size=512):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_tensor_size, hidden_layer_size),  
            nn.ReLU(),                  
            nn.Linear(hidden_layer_size, hidden_layer_size),        
            nn.ReLU(),                  
            nn.Linear(hidden_layer_size, hidden_layer_size//2),        
            nn.ReLU(),                  
            nn.Linear(hidden_layer_size//2, output_action_space_size)
        )

    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = random.sample(range(len(self.buffer)), batch_size)
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.tensor(states, dtype=torch.float32),
                torch.tensor(actions, dtype=torch.long),
                torch.tensor(rewards, dtype=torch.float32),
                torch.tensor(next_states, dtype=torch.float32),
                torch.tensor(dones, dtype=torch.float32))

    def __len__(self):
        return len(self.buffer)
    
class DQNTrainer:
    def __init__(self,
                 input_size,
                 output_size,
                 gamma=0.99,
                 learning_rate=1e-3,
                 batch_size=64,
                 buffer_size=100000,
                 target_update_freq=1000,
                 num_episodes=1000,
                 max_steps_per_episode=200,
                 model_save_path="./dqn_model.pth"):
        
        # Hyperparameters
        self.input_size = input_size
        self.output_size = output_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.target_update_freq = target_update_freq
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.model_save_path = model_save_path

        # Initialize Networks and Optimizer
        self.policy_net = DQN(input_size, output_size)
        self.target_net = DQN(input_size, output_size)
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
            print(f"Epoch {epoch}, Total Reward: {total_reward}")

        # Save the trained model
        self.save_model()

    def optimize_model(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Q(s, a)
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0]
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
