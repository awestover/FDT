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

    def dist_to_goal(self):
        return abs(self.goal_pos[1] - self.agent_pos[1])/GRID_SIZE + abs(self.goal_pos[0] - self.agent_pos[0])/GRID_SIZE
    
    def step(self, action):
        """
        Take an action in the environment and update the state.

        Args:
            action (int): The action to take (0: up, 1: down, 2: left, 3: right).

        Returns:
            tuple: A tuple containing:
                - state (np.ndarray): The new grid state.
                - reward (int): The reward for taking the action (-1 per step).
                - done (bool): Whether the episode has ended.
                - info (dict): Additional information (empty in this case).
        """
        self.steps += 1

        # Terminate if maximum steps exceeded.
        if self.steps >= self.max_steps:
            return self.get_state(), 0, True, {}

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
            # Invalid move: agent remains in place.
            reward = -2*self.dist_to_goal()
            done = False
            return self.get_state(), reward, done, {}
        
        # Valid move: update the grid.
        # Clear old agent position.
        self.grid[r, c] = 0
        
        # Check if new cell is the goal.
        self.agent_pos = (new_r, new_c)
        self.grid[new_r, new_c] = 2  
        done = (new_r, new_c) == self.goal_pos
        reward = -self.dist_to_goal() # step penalty (minimizing steps is the objective)
        if done:
            reward = 100
        return self.get_state(), reward, done, {}
    
    def render(self):
        """
        Render the current state of the grid to the console.
        """
        for row in self.grid:
            print(" ".join(str(int(cell)) for cell in row))
        

# --- DQN Network Definition ---
class DQN(nn.Module):
    """
    Deep Q-Network (DQN) using convolutional layers to process the grid state.
    """

    def __init__(self, input_channels, num_actions):
        """
        Initialize the DQN model.

        Args:
            input_channels (int): Number of channels in the input (after one-hot encoding).
            num_actions (int): Number of possible actions.
        """
        super(DQN, self).__init__()
        # For a grid of size grid_size, adjust the architecture accordingly
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # With padding=1 and kernel_size=3, the spatial dimensions remain the same
        # So the feature map will be of size grid_size x grid_size
        self.fc_input_dim = 32 * GRID_SIZE * GRID_SIZE
        self.fc1 = nn.Linear(self.fc_input_dim, 128)
        self.fc2 = nn.Linear(128, num_actions)

        
    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, grid_size, grid_size).

        Returns:
            torch.Tensor: Output Q-values for each action.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.fc_input_dim)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
# --- Replay Buffer ---
class ReplayBuffer:
    """
    Experience replay buffer that stores transitions for training the DQN.
    Uses a deque for efficient FIFO operations.
    """

    def __init__(self, capacity):
        """
        Initialize the replay buffer.

        Args:
            capacity (int): Maximum number of transitions to store.
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.

        Args:
            state (np.ndarray): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (np.ndarray): Next state after the action.
            done (bool): Whether the episode ended.
        """
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        """
        Randomly sample a batch of transitions.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            tuple: A tuple containing batches of states, actions, rewards, next_states, and dones.
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        """
        Return the current size of the buffer.

        Returns:
            int: Number of stored transitions.
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
        self.replay_buffer = ReplayBuffer(buffer_capacity)
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
        Update the policy network using a batch of transitions from the replay buffer.

        This function samples a mini-batch, computes the current Q-values and target Q-values,
        and performs a gradient descent step to minimize the mean-squared error loss.
        Also updates the target network periodically.

        RETURNS loss
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Preprocess states and next_states (each becomes a (batch,4,8,8) tensor)
        state_batch = torch.cat([self.preprocess(s) for s in states])
        next_state_batch = torch.cat([self.preprocess(s) for s in next_states])
        action_batch = torch.tensor(actions).unsqueeze(1)
        reward_batch = torch.tensor(rewards, dtype=torch.float32)
        done_batch = torch.tensor(dones, dtype=torch.float32)
        
        # Compute Q(s,a) for current states using policy_net
        q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute max_a' Q_target(next_state, a') using target_net.
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch)
            max_next_q_values = next_q_values.max(1)[0]
            # For terminal states, the target is just the reward.
            target_q_values = reward_batch + self.gamma * max_next_q_values * (1 - done_batch)
            target_q_values = target_q_values.unsqueeze(1)
        
        # Compute loss
        loss = F.mse_loss(q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        loss.backward()
        self.optimizer.step()
        
        self.steps_done += 1
        # Periodically update the target network.
        if self.steps_done % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss
    
    def set_epsilon(self, episode, total_episodes):
        """
        Decay the epsilon value for the epsilon-greedy strategy, ensuring it doesn't go below a minimum threshold.
        """
        self.epsilon = self.epsilon_min + (1.0 - self.epsilon_min) * np.exp(-3.0 * episode / total_episodes)

    def zero_epsilon(self):
        self.epsilon = 0


