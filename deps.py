import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from mazegen import generate_maze

GRID_SIZE = 16
# --- Environment Definition ---
class GridWorldEnv:
    """
    Environment representing an 8x8 grid world.
    EDIT: actually the grid_size doesn't need to be 8

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

    An action that would move the agent off the board or into a wall fails,
    leaving the agent in its current position. Every step incurs a reward of -1,
    and the episode ends when the agent reaches the goal or when the maximum steps are reached.
    """

    def __init__(self, max_steps=200):
        """
        Initialize the GridWorld environment.

        Args:
            max_steps (int): Maximum number of steps allowed per episode.
        """
        self.max_steps = max_steps
        self.action_space = 4  # 0: up, 1: down, 2: left, 3: right
        self.reset()
        self.grid_size = GRID_SIZE
    
    def reset(self):
        """
        Reset the environment to the initial state.

        This function creates an 8x8 grid with walls in every odd-numbered column
        (with one randomly removed wall per such column). It places the agent at the
        top-left (position (0,0)) and the goal at the bottom-right.
        
        Returns:
            np.ndarray: A copy of the initial grid state.
        """
        #  self.grid = np.zeros((self.grid_size, self.grid_size))
        #  for col in range(self.grid_size):
        #      if col % 2 == 1:
        #          self.grid[:, col] = 1
        #          hole = random.randint(0, self.grid_size - 1)
        #          self.grid[hole, col] = 0
        self.grid = generate_maze(GRID_SIZE)

        # Place the agent (2) at a fixed starting position (top-left)
        self.agent_pos = (0, 0)
        self.grid[self.agent_pos] = 2
        
        # Place the goal (3) at a fixed location (bottom-right)
        self.goal_pos = (GRID_SIZE - 1, GRID_SIZE - 1)
        self.grid[self.goal_pos] = 3
        
        self.steps = 0
        return self.get_state()
    
    def get_state(self):
        """
        Get the current state of the environment.

        Returns:
            np.ndarray: A copy of the current grid state.
        """
        return self.grid.copy()
    
    def step(self, action):
        """
        Take an action in the environment and update the state.

        Args:
            action (int): The action to take (0: up, 1: down, 2: left, 3: right).

        Returns:
            tuple: A tuple containing:
                - state (np.ndarray): The new grid state.
                - reward (float): The reward for taking the action.
                - done (bool): Whether the episode has ended.
        """
        self.steps += 1
        
        # Get current distance to goal
        old_dist = self.manhattan_distance()
        
        # Terminate if maximum steps exceeded
        if self.steps >= self.max_steps:
            return self.get_state(), 0, True

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
            # Invalid move: small penalty and no state change
            return self.get_state(), -0.2, False
        
        # Valid move: update the grid
        # Clear old agent position
        self.grid[r, c] = 0
        
        # Update agent position
        self.agent_pos = (new_r, new_c)
        self.grid[new_r, new_c] = 2
        
        # Check if new cell is the goal
        done = (new_r, new_c) == self.goal_pos
        
        # Calculate new distance to goal
        new_dist = self.manhattan_distance()
        
        # Reward structure:
        # 1. High reward for reaching goal
        # 2. Small reward for moving closer to goal
        # 3. Small penalty for moving away from goal
        # 4. Small step penalty to encourage efficiency
        if done:
            reward = 10.0  # Big reward for reaching goal
        else:
            # Reward progress toward goal
            if new_dist < old_dist:
                reward = 0.2  # Small reward for getting closer
            elif new_dist > old_dist:
                reward = -0.1  # Small penalty for getting further
            else:
                reward = -0.05  # Tiny penalty for lateral movement
                
        return self.get_state(), reward, done

    def manhattan_distance(self):
        """
        Calculate Manhattan distance from agent to goal.
        
        Returns:
            float: Normalized distance between 0 and 1
        """
        r, c = self.agent_pos
        gr, gc = self.goal_pos
        return (abs(gr - r) + abs(gc - c)) / (2 * GRID_SIZE)
        
    def render(self):
        """
        Render the current state of the grid to the console.
        """
        for row in self.grid:
            print(" ".join(str(int(cell)) for cell in row))

# --- DQN Network Definition ---
class DQN(nn.Module):
    """
    Optimized Deep Q-Network for grid world navigation.
    Uses stride and pooling to reduce spatial dimensions before fully connected layers.
    """
    def __init__(self, input_channels, num_actions):
        super(DQN, self).__init__()
        # Reduce spatial dimensions with stride and pooling
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # Reduce spatial dimensions by half
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # Reduce spatial dimensions by half again
        
        # Calculate input size for fully connected layer (GRID_SIZE/4 due to two pooling layers)
        fc_input_size = 32 * (GRID_SIZE // 4) * (GRID_SIZE // 4)
        
        # Reduce fully connected layer sizes 
        self.fc1 = nn.Linear(fc_input_size, 256)
        self.fc2 = nn.Linear(256, num_actions)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer that stores transitions based on TD error.
    This helps the agent learn more efficiently from important experiences.
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        """
        Initialize the prioritized replay buffer.
        
        Args:
            capacity (int): Maximum size of the buffer
            alpha (float): How much prioritization to use (0 = uniform sampling)
            beta (float): Importance sampling correction factor (0 = no correction)
            beta_increment (float): How much to increase beta per sampling
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = np.ones(capacity)
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        """
        Add a new experience to memory with maximum priority.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode ended
        """
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
            
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """
        Sample a batch of experiences based on their priorities.
        
        Args:
            batch_size (int): Number of experiences to sample
            
        Returns:
            tuple: Batch of experiences and importance sampling weights
        """
        if len(self.buffer) < batch_size:
            indices = np.random.choice(len(self.buffer), batch_size, replace=True)
        else:
            # Calculate sampling probabilities
            priorities = self.priorities[:len(self.buffer)]
            probabilities = priorities ** self.alpha
            probabilities /= probabilities.sum()
            
            # Sample based on priorities
            indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
            
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Get the sampled experiences
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to numpy arrays
        return (
            np.array(states), 
            np.array(actions), 
            np.array(rewards, dtype=np.float32), 
            np.array(next_states), 
            np.array(dones, dtype=np.float32),
            indices,
            np.array(weights, dtype=np.float32)
        )
    
    def update_priorities(self, indices, td_errors):
        """
        Update priorities based on TD errors.
        
        Args:
            indices (list): Indices to update
            td_errors (list): TD errors for each experience
        """
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + 1e-5  # Add small constant to avoid zero priority
    
    def __len__(self):
        """
        Return current buffer size.
        """
        return len(self.buffer)

    
# --- DQN Agent ---
class DQNAgent:
    """
    Deep Q-Network (DQN) agent that interacts with the environment and learns from experiences.
    """

    def __init__(self, lr=5e-4, gamma=0.99, buffer_capacity=10000, batch_size=64, update_target_every=128):
        """
        Initialize the DQN agent.

        Args:
            lr (float): Learning rate for the optimizer.
            gamma (float): Discount factor for future rewards.
            buffer_capacity (int): Capacity of the replay buffer.
            batch_size (int): Number of transitions to sample per training step.
            update_target_every (int): Number of steps between target network updates.
        """
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.steps_done = 0
        
        # Our grid has 4 distinct values (0,1,2,3) -> one-hot encoded to 4 channels.
        self.input_channels = 4
        self.num_actions = 4
        
        self.policy_net = DQN(self.input_channels, self.num_actions)
        self.target_net = DQN(self.input_channels, self.num_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # set target net to evaluation mode
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = PrioritizedReplayBuffer(buffer_capacity)
        self.epsilon = 1.0  # initial epsilon for epsilon-greedy
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99  # decay factor per episode
    
    def preprocess(self, state):
        """
        Preprocess the grid state by converting it into a one-hot encoded tensor.

        Args:
            state (np.ndarray): Grid state of shape (8,8) with values {0,1,2,3}.

        Returns:
            torch.Tensor: One-hot encoded tensor of shape (1, 4, 8, 8).
        """
        state_tensor = torch.from_numpy(state).long()  # shape: (8,8)
        one_hot = F.one_hot(state_tensor, num_classes=4)  # shape: (8,8,4)
        # Rearrange to (channels, height, width)
        one_hot = one_hot.permute(2, 0, 1).float()
        return one_hot.unsqueeze(0)  # add batch dimension -> (1,4,8,8)
    
    def select_action(self, state):
        """
        Select an action using an epsilon-greedy strategy.

        Args:
            state (np.ndarray): Current state of the environment.

        Returns:
            int: The selected action.
        """
        sample = random.random()
        if sample < self.epsilon:
            return random.randrange(self.num_actions)
        else:
            with torch.no_grad():
                state_tensor = self.preprocess(state)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()
    
    def update(self):
        """
        Update the policy network with prioritized experience replay.
        
        Returns:
            float or None: Loss value if update was performed, None otherwise
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample experiences with priorities
        states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(self.batch_size)
        
        # Preprocess states and next_states
        state_batch = torch.cat([self.preprocess(s) for s in states])
        next_state_batch = torch.cat([self.preprocess(s) for s in next_states])
        action_batch = torch.tensor(actions).unsqueeze(1)
        reward_batch = torch.tensor(rewards, dtype=torch.float32)
        done_batch = torch.tensor(dones, dtype=torch.float32)
        weights_batch = torch.tensor(weights, dtype=torch.float32)
        
        # Compute current Q values
        q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute next state values using target network (Double DQN)
        with torch.no_grad():
            # Get actions from policy net
            next_actions = self.policy_net(next_state_batch).max(1)[1].unsqueeze(1)
            # Get Q-values from target net for those actions
            next_q_values = self.target_net(next_state_batch).gather(1, next_actions)
            # Use target Q-values in Bellman equation
            target_q_values = reward_batch.unsqueeze(1) + self.gamma * next_q_values * (1 - done_batch.unsqueeze(1))
        
        # Calculate TD errors for priority updates
        td_errors = torch.abs(q_values - target_q_values).detach().numpy()
        
        # Compute weighted Huber loss (more stable than MSE)
        loss = F.smooth_l1_loss(q_values, target_q_values, reduction='none')
        loss = (loss * weights_batch.unsqueeze(1)).mean()
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update priorities in the replay buffer
        self.replay_buffer.update_priorities(indices, td_errors)
        
        # Periodically update the target network
        self.steps_done += 1
        if self.steps_done % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        return loss.item()
    
    def set_epsilon(self, episode, total_episodes):
        """
        Decay the epsilon value for the epsilon-greedy strategy, ensuring it doesn't go below a minimum threshold.
        """
        self.epsilon = self.epsilon_min + (1.0 - self.epsilon_min) * np.exp(-3.0 * episode / total_episodes)

    def zero_epsilon(self):
        self.epsilon = 0


