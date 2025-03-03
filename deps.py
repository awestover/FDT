import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from numpy import random as nprand
from math import exp
GRID_SIZE = 8

def generate_maze(buffer, device, nchannels=2, size=16, difficulty=0.5):
    """
    Generate a maze directly as a one-hot encoded tensor with fully vectorized operations.
    
    Args:
        device: The torch device to place the tensor on
        nchannels (int): Number of channels for one-hot encoding
        size (int): Size of the maze (size x size grid)
        difficulty (float): Value between 0.0 and 1.0 controlling maze difficulty
    
    Returns:
        torch.Tensor: One-hot encoded maze of shape [nchannels, size, size]
    """
    one_hot = buffer
    if one_hot is None:
        one_hot = torch.zeros((nchannels, size, size), dtype=torch.float16, device=device)
    wall_cols = torch.arange(1, size, 2, device=device)
    num_walls = len(wall_cols)
    one_hot[0, :, wall_cols] = 1
    difficulty = max(0.0, min(1.0, difficulty))
    hole_percentage = 0.75 - 0.65 * difficulty
    holes_per_wall = max(1, int(hole_percentage * size))
    repeated_wall_cols = wall_cols.repeat_interleave(holes_per_wall)

    all_positions = torch.randint(0, size, (size * num_walls,), device=device)
    hole_rows = all_positions[:num_walls * holes_per_wall]
    one_hot[0, hole_rows, repeated_wall_cols] = 0
    one_hot[0, -1, -1] = 0
    if size % 2 == 0:
        one_hot[0, -1, -2] = 0
    return one_hot

# -----------------------
# ENVIRONMENT
# -----------------------

class GridWorldEnv:
    def __init__(self, device, max_steps):
        """
        CHANNELS
        0 -- wall
        1 -- agent
        """
        self.num_channels = 2
        self.max_steps = max_steps
        self.grid_size = GRID_SIZE
        self.device = device
        self.move_map = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1),   # right
        }
        self.grid = torch.zeros((self.num_channels, GRID_SIZE, GRID_SIZE), dtype=torch.float16, device=self.device)
        self.visit_count = torch.zeros((GRID_SIZE, GRID_SIZE), dtype=torch.int16, device=self.device)
        self.reset()

    def reset(self, maze_difficulty=0.5, dist_to_end=0.2):
        """Reset the environment with the given difficulty."""
        generate_maze(self.grid, device=self.device, nchannels=self.num_channels, size=GRID_SIZE, difficulty=maze_difficulty)

        agentr = int(GRID_SIZE*(1 - nprand.rand()*dist_to_end))
        agentc = int(GRID_SIZE*(1 - nprand.rand()*dist_to_end))
        self.agent_pos = (agentr, agentc)

        self.grid[1, 0, 0] = 1
        self.goal_pos = (GRID_SIZE - 1, GRID_SIZE - 1)
        self.steps = 0
        self.visit_count.zero_()
        return self.grid.clone()

    def step(self, action):
        """
        Take an action in the environment and update the state.

        Args:
            action (int): The action to take (0: up, 1: down, 2: left, 3: right).

        Returns:
            tuple: A tuple containing:
                - grid (torch.Tensor): The new grid state.
                - reward (float): The reward for taking the action.
                - done (bool): Whether the episode has ended.
        """
        self.steps += 1

        # Check if episode has exceeded maximum steps
        if self.steps >= self.max_steps:
            return self.grid.clone(), 0, True

        # Constants for rewards
        INVALID_PENALTY = 1
        SLOW_PENALTY = 1
        REVISIT_PENALTY = 1

        # Map action to movement (row, col)
        dr, dc = self.move_map[action]
        r, c = self.agent_pos
        new_r, new_c = r + dr, c + dc

        # Check for out-of-bounds or hitting a wall (cell==1)
        if (
            new_r < 0
            or new_r >= GRID_SIZE
            or new_c < 0
            or new_c >= GRID_SIZE
            or self.grid[0, new_r, new_c] == 1
        ):
            return self.grid.clone(), -INVALID_PENALTY, False

        # Update agent position in grid
        self.grid[1, r, c] = 0
        self.grid[1, new_r, new_c] = 2
        self.agent_pos = (new_r, new_c)

        done = (new_r, new_c) == self.goal_pos
        
        # Update visit count and calculate reward
        self.visit_count[new_r, new_c] += 1
        reward = -REVISIT_PENALTY * self.visit_count[new_r, new_c].item() - SLOW_PENALTY
        
        return self.grid.clone(), reward, done

# -----------------------
# NEURAL NETWORK
# -----------------------

class DQN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(DQN, self).__init__()
        
        # Calculate the feature size after convolutions and pooling
        feature_size = ((GRID_SIZE // 4) ** 2) * 4 * input_channels

        # CNN layers
        self.cnn_layers = nn.Sequential(
            # First block: 4 → 8 channels, gs → gs/2
            nn.Conv2d(input_channels, 2 * input_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 * input_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Second block: 8 → 16 channels, gs/2 → gs/4
            nn.Conv2d(2 * input_channels, 4 * input_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(4 * input_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # MLP layers
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_actions),
        )

    def forward(self, x):
        return self.mlp(self.cnn_layers(x))

# -----------------------
# REPLAY BUFFER
# -----------------------

class TensorReplayBuffer:
    """
    Replay buffer optimized for storing torch tensors directly.
    """
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """Store a transition in the buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
            
        # Ensure action, reward, and done are tensors
        if not isinstance(action, torch.Tensor):
            action = torch.tensor([action], device=self.device)
        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor([reward], dtype=torch.float, device=self.device)
        if not isinstance(done, torch.Tensor):
            done = torch.tensor([done], dtype=torch.bool, device=self.device)
            
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Sample a batch of transitions from the buffer."""
        batch_indices = nprand.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        batch = [self.buffer[i] for i in batch_indices]

        state, action, reward, next_state, done = zip(*batch)
        
        # Stack the tensors
        state_batch = torch.cat([s.unsqueeze(0) for s in state], dim=0)
        action_batch = torch.cat([a for a in action], dim=0)
        reward_batch = torch.cat([r for r in reward], dim=0)
        next_state_batch = torch.cat([s.unsqueeze(0) for s in next_state], dim=0)
        done_batch = torch.cat([d for d in done], dim=0)
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, device, lr, gamma, buffer_capacity, batch_size, update_target_every):
        self.device = device 
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.steps_done = 0

        # Input channels and actions
        self.input_channels = 2 # wall, agent
        self.num_actions = 4 # up, down, left, right

        # Initialize networks
        self.policy_net = DQN(self.input_channels, self.num_actions).to(self.device)
        self.target_net = DQN(self.input_channels, self.num_actions).to(self.device)
        # Set target network weights to be the same as the policy network to start
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # Set target network to evaluation mode which means that the weights are frozen
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Experience replay buffer
        self.replay_buffer = TensorReplayBuffer(buffer_capacity, self.device)

        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.998

    def select_action(self, state):
        """
        Select an action using epsilon-greedy policy based on current state.
        Expects state to be a tensor already.
        """

        # If we're exploring
        if nprand.rand() < self.epsilon:
            return nprand.randint(self.num_actions)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state.unsqueeze(0))
                return q_values.max(1)[1].item()

    def update(self):
        """
        Update network weights using experiences from the buffer.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(self.batch_size)

        # Current Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

        # Compute target Q values
        with torch.no_grad():
            next_actions = self.policy_net(next_state_batch).max(1)[1].unsqueeze(1)
            next_q_values = self.target_net(next_state_batch).gather(1, next_actions)
            target_q_values = reward_batch.unsqueeze(1) + self.gamma * next_q_values * (~done_batch.unsqueeze(1))

        # loss = |r + gamma * Qtarget(s',a') - Qpolicy(s,a)|
        # ie get better at predicting how good s,a pairs are
        loss = F.smooth_l1_loss(current_q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update target network periodically
        self.steps_done += 1
        if self.steps_done % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def set_epsilon(self, episode, total_episodes):
        self.epsilon = self.epsilon_min + (1.0 - self.epsilon_min) * exp(-3.0 * episode / total_episodes)

