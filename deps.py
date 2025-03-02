import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from mazegen import generate_easy_maze as generate_maze

GRID_SIZE = 8

class GridWorldEnv:
    """
    Environment representing a grid world with visitation tracking.
    
    The grid consists of:
        0: empty cell (agent can move here)
        1: blocked cell (wall)
        2: the agent's current position
        3: the goal position
    
    The agent can take one of four actions:
        0: move up
        1: move down
        2: move left
        3: move right
    
    This enhanced version keeps track of where the agent has been and penalizes
    revisiting states to encourage exploration of new areas.
    """

    def __init__(self, max_steps=200, revisit_penalty_factor=0.2, decay_rate=0.95):
        """
        Initialize the GridWorld environment.

        Args:
            max_steps (int): Maximum number of steps allowed per episode.
            revisit_penalty_factor (float): Factor to multiply the visitation count for penalties.
            decay_rate (float): Rate at which visitation counts decay over time.
        """
        self.max_steps = max_steps
        self.action_space = 4  # 0: up, 1: down, 2: left, 3: right
        self.revisit_penalty_factor = revisit_penalty_factor
        self.decay_rate = decay_rate
        self.grid_size = GRID_SIZE
        self.reset()
    
    def reset(self):
        """
        Reset the environment to the initial state.
        
        Returns:
            np.ndarray: A copy of the initial grid state.
        """
        self.grid = generate_maze(GRID_SIZE)

        # Place the agent (2) at a fixed starting position (top-left)
        self.agent_pos = (0, 0)
        self.grid[self.agent_pos] = 2
        
        # Place the goal (3) at a fixed location (bottom-right)
        self.goal_pos = (GRID_SIZE - 1, GRID_SIZE - 1)
        self.grid[self.goal_pos] = 3
        
        # Initialize visitation map - tracks how many times agent has visited each cell
        self.visit_count = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        
        # Mark the initial position as visited
        self.visit_count[self.agent_pos] = 1.0
        
        self.steps = 0
        return self.get_state()
    
    def get_state(self):
        """
        Get the current state of the grid.
        
        Returns:
            np.ndarray: A copy of the current grid state.
        """
        return self.grid.copy()
    
    def get_visit_map(self):
        """
        Get the current visitation map.
        
        Returns:
            np.ndarray: A normalized copy of the current visitation map.
        """
        # Normalize the visit count for visualization if needed
        if np.max(self.visit_count) > 0:
            normalized_map = self.visit_count / np.max(self.visit_count)
        else:
            normalized_map = self.visit_count
        return normalized_map
    
    def decay_visit_counts(self):
        """
        Decay all visit counts by the decay rate to gradually forget old visits.
        """
        self.visit_count *= self.decay_rate
    
    def step(self, action, gamma=.99):
        """
        Take an action in the environment and update the state.

        Args:
            action (int): The action to take (0: up, 1: down, 2: left, 3: right).
            gamma (float): Discount factor for future rewards.

        Returns:
            tuple: A tuple containing:
                - state (np.ndarray): The new grid state.
                - reward (float): The reward for taking the action.
                - done (bool): Whether the episode has ended.
        """
        self.steps += 1
        
        # Check if episode has exceeded maximum steps
        if self.steps >= self.max_steps:
            return self.get_state(), 0, True

        # Constants for rewards
        INVALID_PENALTY = -2
        WIN_BONUS = 2 * self.max_steps
        
        # Map action to movement (row, col)
        move_map = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1)    # right
        }
        dr, dc = move_map[action]
        r, c = self.agent_pos
        new_r, new_c = r + dr, c + dc
        
        # Check for out-of-bounds or hitting a wall (cell==1)
        if (new_r < 0 or new_r >= GRID_SIZE or 
            new_c < 0 or new_c >= GRID_SIZE or
            self.grid[new_r, new_c] == 1):
            return self.get_state(), INVALID_PENALTY, False

        # Valid move: update the grid
        old_dist = self.dist_to_goal()
        self.agent_pos = (new_r, new_c)
        new_dist = self.dist_to_goal()
        self.grid[r, c] = 0
        self.grid[new_r, new_c] = 2
        
        # Check if goal reached
        done = (new_r, new_c) == self.goal_pos
        
        # Calculate revisit penalty based on the visit count at the new position
        revisit_penalty = self.revisit_penalty_factor * self.visit_count[new_r, new_c]
        
        # Update visitation count for the new position
        self.visit_count[new_r, new_c] += 1.0
        
        # Decay visit counts slightly each step to gradually forget old visits
        self.decay_visit_counts()
        
        if done:
            reward = WIN_BONUS  # Big reward for reaching goal
        else:
            # Base reward from distance change plus a penalty for revisits
            reward = old_dist - gamma * new_dist - revisit_penalty
                
        return self.get_state(), reward, done

    def dist_to_goal(self):
        """
        Calculate Manhattan distance from current position to goal.
        
        Returns:
            float: Manhattan distance to the goal.
        """
        r, c = self.agent_pos
        gr, gc = self.goal_pos
        return abs(gr - r) + abs(gc - c)        

    def render(self):
        """
        Print a text representation of the grid to stdout.
        """
        for row in self.grid:
            print(" ".join(str(int(cell)) for cell in row))
    
    def render_visits(self):
        """
        Print a text representation of the visitation map to stdout.
        """
        visit_map = self.get_visit_map()
        for row in visit_map:
            print(" ".join(f"{cell:.1f}" for cell in row))


class DQN(nn.Module):
    """
    A simplified DQN that removes recurrent components and complex architecture.
    This network takes a grid state and directly predicts Q-values with minimal processing.
    """
    def __init__(self, input_channels, num_actions):
        super(DQN, self).__init__()
        
        #  convolutional layers without batch normalization
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        # Calculate CNN output size
        cnn_output_size = 32 * GRID_SIZE * GRID_SIZE
        
        #  fully connected layers
        self.fc1 = nn.Linear(cnn_output_size, 128)
        self.fc2 = nn.Linear(128, num_actions)
        
    def forward(self, x):
        """
        Forward pass of the simplified DQN.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            q_values: Action values
        """
        # CNN Layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)
        
        return q_values


class ReplayBuffer:
    """
     replay buffer that stores individual transitions rather than sequences.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        """
        Store a transition in the buffer.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer.
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
    
    def __len__(self):
        """
        Return current buffer size.
        """
        return len(self.buffer)


class DQNAgent:
    """
    DQN Agent using a simplified neural network architecture.
    """
    def __init__(self, lr=3e-4, gamma=0.99, buffer_capacity=50000, batch_size=32, 
                 update_target_every=500, device=None):
        """
        Initialize the  DQN agent.
        """
        # Set the device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DQNAgent using device: {self.device}")
        
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.steps_done = 0
        
        # Input channels and actions
        self.input_channels = 4  # one-hot encoded grid values
        self.num_actions = 4  # up, down, left, right
        
        # Initialize networks
        self.policy_net = DQN(self.input_channels, self.num_actions).to(self.device)
        self.target_net = DQN(self.input_channels, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.998
        
        # Reward tracking
        self.reward_window = deque(maxlen=100)
    
    def preprocess(self, state):
        """
        Preprocess a single state into a tensor.
        """
        state_tensor = torch.from_numpy(state).long()
        one_hot = F.one_hot(state_tensor, num_classes=4)
        # Rearrange to (channels, height, width)
        one_hot = one_hot.permute(2, 0, 1).float()
        # Move to the appropriate device
        return one_hot.unsqueeze(0).to(self.device)  # add batch dimension
    
    def select_action(self, state):
        """
        Select an action using epsilon-greedy policy based on current state.
        """
        sample = random.random()
        
        # Current state tensor
        state_tensor = self.preprocess(state)
        
        # If we're exploring
        if sample < self.epsilon:
            #  exploration with some bias toward goal
            r, c = None, None
            for i in range(state.shape[0]):
                for j in range(state.shape[1]):
                    if state[i, j] == 2:  # Agent position
                        r, c = i, j
                        break
                if r is not None:
                    break
            
            if r is not None:
                # Bias exploration toward goal
                goal_r, goal_c = GRID_SIZE - 1, GRID_SIZE - 1
                
                # Calculate direction weights
                weights = np.ones(4) * 0.25  # Equal probability by default
                
                # Up, Down, Left, Right
                if r > 0 and state[r-1, c] != 1:  # Can move up
                    if r > goal_r:  # If agent is below goal, increase up probability
                        weights[0] = 0.4
                if r < GRID_SIZE-1 and state[r+1, c] != 1:  # Can move down
                    if r < goal_r:  # If agent is above goal, increase down probability
                        weights[1] = 0.4
                if c > 0 and state[r, c-1] != 1:  # Can move left
                    if c > goal_c:  # If agent is to the right of goal, increase left probability
                        weights[2] = 0.4
                if c < GRID_SIZE-1 and state[r, c+1] != 1:  # Can move right
                    if c < goal_c:  # If agent is to the left of goal, increase right probability
                        weights[3] = 0.4
                
                weights = weights / weights.sum()  # Normalize
                
                return np.random.choice(self.num_actions, p=weights)
            
            return random.randrange(self.num_actions)
        else:
            # Exploit - use the policy network to make a prediction
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()
    
    def update(self):
        """
        Update network weights using experiences from the buffer.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Preprocess states and next_states
        state_batch = torch.cat([self.preprocess(s) for s in states])
        next_state_batch = torch.cat([self.preprocess(s) for s in next_states])
        
        # Convert other variables to tensors
        action_batch = torch.tensor(actions, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(dones, dtype=torch.float32, device=self.device)
        
        # Current Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute target Q values
        with torch.no_grad():
            # Double DQN
            next_actions = self.policy_net(next_state_batch).max(1)[1].unsqueeze(1)
            next_q_values = self.target_net(next_state_batch).gather(1, next_actions)
            target_q_values = reward_batch.unsqueeze(1) + self.gamma * next_q_values * (1 - done_batch.unsqueeze(1))
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        #  gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.steps_done += 1
        if self.steps_done % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        return loss.item()
    
    def set_epsilon(self, episode, total_episodes):
        """
        Adjust exploration rate based on training progress.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon_min + 
                          (1.0 - self.epsilon_min) * np.exp(-3.0 * episode / total_episodes))
    
    def zero_epsilon(self):
        """
        Set epsilon to zero for evaluation.
        """
        self.epsilon = 0

def init_agent(device):
    agent = DQNAgent(
        lr=1e-4, 
        gamma=0.99, 
        buffer_capacity=50000, 
        batch_size=64,
        update_target_every=200,
        device=device
    )
    return agent
