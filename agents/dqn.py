import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

VERBOSE_LOGGING = False

class DQN(nn.Module):
    def __init__(self, input_tensor_size: int, output_action_space_size: int, hidden_layer_size: int = 512):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_tensor_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, output_action_space_size)
        )

    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, size: int):
        self.buffer = deque(maxlen=size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = np.array([self.buffer[idx] for idx in indices], dtype=object)
        
        states, actions, rewards, next_states, dones = zip(*batch)

        # Vectorized conversion to numpy arrays
        states = np.vstack(states).astype(np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.vstack(next_states).astype(np.float32)
        dones = np.array(dones, dtype=np.float32)

        # Convert numpy arrays into PyTorch tensors
        return (torch.from_numpy(states),
                torch.from_numpy(actions),
                torch.from_numpy(rewards),
                torch.from_numpy(next_states),
                torch.from_numpy(dones))

    def __len__(self):
        return len(self.buffer)

class DQNTrainer:
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 gamma=0.99,
                 learning_rate=1e-4,
                 batch_size=512,
                 buffer_size=100000,
                 target_update_freq=2500,
                 num_epochs=1,
                 max_steps_per_episode=200):

        # Device
        torch.cuda.empty_cache()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PyTorch using device: {self.device}")
        self.time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(log_dir=f"./runs/experiment_{self.time}")
        self.hidden_size = (input_size + output_size) // 2

        print(f"Input Size: {input_size}, Output Size: {output_size}, Hidden Size: {self.hidden_size}")

        # Hyperparameters
        self.input_size = input_size
        self.output_size = output_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.target_update_freq = target_update_freq
        self.num_epochs = num_epochs
        self.max_steps_per_episode = max_steps_per_episode
        self.model_save_path = f"./model-{self.time}-{input_size}x{output_size}:{self.hidden_size}-gamma_{gamma}-lr_{learning_rate}-bs_{batch_size}-epochs_{num_epochs}-updatefreq_{target_update_freq}.pth"

        # Initialize Networks and Optimizer
        self.policy_net = DQN(input_size, output_size, self.hidden_size)
        self.target_net = DQN(input_size, output_size, self.hidden_size)
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
                    print(f"Input Action Tensor Size: {len(action)}")
                    print(f"Reward: {reward}")
                    print(f"Next State Tensor Size: {len(next_state)}")
                    print(f"Game Finished Tensor: {done}")

                    #print(f"Input State Tensor:\n {state}")
                    #print(f"Input Action Tensor:\n {action}")
                    #print(f"Next State Tensor:\n {next_state}")


                steps += 1
                total_reward += reward[0]
                # print (f"Total Reward: {total_reward}")

                self.replay_buffer.add(state, action, reward, next_state, done)

                # Train the policy network if buffer is full
                if len(self.replay_buffer) >= self.batch_size:
                    self.optimize_model(steps)

                # Update target network periodically
                if steps % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

            # Log epoch results
            print(f"Epoch {epoch}, Total Reward: {total_reward}")

        # Save the trained model
        self.save_model()

    def optimize_model(self, step_number: int):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = (
            states.to(self.device),
            actions.to(self.device),
            rewards.to(self.device),
            next_states.to(self.device),
            dones.to(self.device)
        )

        # Debugging: Print tensor shapes
        # print(f"States shape: {states.shape}")          # Expected: [batch_size, 3175]
        # print(f"Actions shape: {actions.shape}")        # Expected: [batch_size]
        # print(f"Rewards shape: {rewards.shape}")        # Expected: [batch_size]
        # print(f"Next States shape: {next_states.shape}")# Expected: [batch_size, 3175]
        # print(f"Dones shape: {dones.shape}")            # Expected: [batch_size]

        # Q(s, a)
        q_values = self.policy_net(states).gather(1, actions)
        # print(f"Q-Values shape: {q_values.shape}")    # Expected: [batch_size, 1]

        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards.squeeze() + self.gamma * max_next_q_values * (1 - dones.squeeze())
            target_q_values = target_q_values.unsqueeze(1)
            # print(f"Target Q-Values shape: {target_q_values.shape}")    # Expected: [batch_size, 1]

        # Loss
        loss = nn.MSELoss()(q_values, target_q_values)
        self.writer.add_scalar("Loss", loss, step_number)  # Log the loss

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def save_model(self):
        torch.save(self.policy_net.state_dict(), self.model_save_path)
        print(f"Model saved to: {self.model_save_path}")
